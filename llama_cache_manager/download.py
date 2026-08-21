"""Fetching an artifact from the hub, as a plan that is first shown.

``pull`` follows the shape the deleting commands already have: the reference is
resolved, a plan is built, the plan is printed, and only then does anything
happen. What resolves the reference is the same grouping ``ls`` uses, so
``org/repo:quant`` names the same artifact whether it is already in the cache or
still on the hub.

The hub itself sits behind :class:`Hub`. The logic here never imports
``huggingface_hub``, which keeps the tests free of the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import refs
from .cache import Kind, kind_of
from .refs import Reference


class DownloadError(Exception):
    """The hub cannot be reached, or refused what was asked of it."""


class UnavailableError(DownloadError):
    """The hub answered, and what was asked for is not there or not allowed.

    Kept apart from a hub that did not answer at all. A run over the whole
    cache steps over one repository it cannot have and carries on, but it must
    not step over a network that is down: that would report every repository as
    up to date when nothing was checked.
    """


class SelectionError(Exception):
    """A reference that names no artifact on the hub."""


@dataclass(frozen=True, slots=True)
class FileStatus:
    """One file of a download, as the hub reports it before anything is read."""

    name: str
    size: int
    commit: str
    cached: bool


class Hub(Protocol):
    """Everything ``pull`` needs from the far side of the network.

    Kept to three operations so that a test can write a fake with nothing but
    the standard library.
    """

    def file_names(self, repo_id: str, revision: str) -> tuple[str, ...]:
        """Every file ``repo_id`` holds at ``revision``.

        Raises:
            DownloadError: the hub is unreachable, or holds no such repository.
        """
        ...

    def head(self, repo_id: str, revision: str) -> str:
        """The commit ``revision`` points at on the hub right now.

        Raises:
            DownloadError: the hub is unreachable, or holds no such repository
                or revision.
        """
        ...

    def inspect(self, repo_id: str, name: str, revision: str, cache_dir: Path) -> FileStatus:
        """What fetching one file would cost, without fetching it.

        Raises:
            DownloadError: the hub is unreachable, or holds no such file.
        """
        ...

    def fetch(self, repo_id: str, name: str, revision: str, cache_dir: Path) -> Path:
        """Put one file into ``cache_dir`` and give back where it landed.

        Raises:
            DownloadError: the transfer did not finish.
        """
        ...


@dataclass(frozen=True, slots=True)
class Download:
    """One artifact to fetch, with every file that makes it up."""

    repo_id: str
    revision: str
    quant: str
    kind: Kind
    files: tuple[FileStatus, ...]

    @property
    def reference(self) -> str:
        return f"{self.repo_id}:{self.quant}"

    @property
    def size(self) -> int:
        """Bytes the artifact holds once it is in the cache."""
        return sum(file.size for file in self.files)

    @property
    def transfer(self) -> int:
        """Bytes that still have to come over the network."""
        return sum(file.size for file in self.files if not file.cached)

    @property
    def shards(self) -> int:
        return len(self.files)

    @property
    def cached(self) -> bool:
        return all(file.cached for file in self.files)

    @property
    def commit(self) -> str:
        """The revision the files come from, resolved to a commit hash."""
        return self.files[0].commit


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    """A complete download, with both totals already derived."""

    cache_dir: Path
    downloads: tuple[Download, ...]

    @property
    def size(self) -> int:
        """Bytes the plan puts in the cache, cached files included."""
        return sum(item.size for item in self.downloads)

    @property
    def transfer(self) -> int:
        """Bytes the plan reads over the network."""
        return sum(item.transfer for item in self.downloads)

    @property
    def repo_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for item in self.downloads:
            seen[item.repo_id] = None
        return tuple(seen)


def _artifacts_on_hub(hub: Hub, repo_id: str, revision: str) -> dict[str, tuple[str, ...]]:
    """The artifacts ``repo_id`` offers at ``revision``, keyed by quant.

    Raises:
        DownloadError: the hub is unreachable, or holds no such repository.
        SelectionError: the repository holds no GGUF file at all.
    """
    names = hub.file_names(repo_id, revision)
    grouped = refs.group_by_artifact(repo_id, names)
    if not grouped:
        raise SelectionError(f"{repo_id} holds no GGUF file at revision {revision!r}")
    return grouped


def plan(hub: Hub, cache_dir: Path, references: list[Reference], revision: str) -> DownloadPlan:
    """Work out what ``references`` would fetch into ``cache_dir``.

    Each reference has to name a repository and a quant. The hub cannot be
    searched, so a partial reference is refused with the available quants named
    rather than guessed at.

    Two references that resolve to the same artifact become one download. The
    quant match ignores case, so a repeat need not be spelled the same, and
    counting it twice would tell the reader that a transfer costs double what it
    does.

    Raises:
        DownloadError: the hub is unreachable, or refused a request.
        SelectionError: a reference names no artifact on the hub.
    """
    listings: dict[str, dict[str, tuple[str, ...]]] = {}
    downloads: dict[tuple[str, str], Download] = {}

    for reference in references:
        repo_id = reference.repo_id
        if repo_id is None:
            wanted = f"org/repo:{reference.quant}" if reference.quant else "org/repo:quant"
            raise SelectionError(
                f"{reference.raw!r} names no repository, and the hub cannot be searched "
                f"from here; write {wanted} to fetch it"
            )
        if repo_id not in listings:
            listings[repo_id] = _artifacts_on_hub(hub, repo_id, revision)
        available = listings[repo_id]

        quant = _pick_quant(reference, repo_id, revision, available)
        if (repo_id, quant) in downloads:
            continue

        files = tuple(hub.inspect(repo_id, name, revision, cache_dir) for name in available[quant])
        downloads[repo_id, quant] = Download(
            repo_id=repo_id,
            revision=revision,
            quant=quant,
            kind=kind_of(quant),
            files=files,
        )

    return DownloadPlan(cache_dir=Path(cache_dir), downloads=tuple(downloads.values()))


def _pick_quant(reference: Reference, repo_id: str, revision: str, available: dict) -> str:
    """The one quant a reference names, or an error listing the choices."""
    # Ordered the way ``ls`` orders artifacts, so the offer reads like a listing.
    offered = ", ".join(sorted(available, key=lambda quant: (kind_of(quant).order, quant.lower())))
    if reference.quant is None:
        raise SelectionError(f"{repo_id} needs a quant to fetch; it offers {offered}")
    for quant in available:
        if quant.lower() == reference.quant.lower():
            return quant
    raise SelectionError(
        f"{repo_id} has no quant {reference.quant!r} at revision {revision!r}; it offers {offered}"
    )


def execute(plan: DownloadPlan, hub: Hub) -> None:
    """Fetch everything the plan holds.

    A file already in the cache is still handed to the hub, because the
    snapshot of a new revision needs its link even when the blob behind it is
    there already. That costs no transfer.

    Raises:
        DownloadError: a transfer did not finish.
    """
    for item in plan.downloads:
        for file in item.files:
            hub.fetch(item.repo_id, file.name, item.revision, plan.cache_dir)
