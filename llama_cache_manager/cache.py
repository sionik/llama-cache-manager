"""The cache seen as repositories, revisions and addressable artifacts.

``huggingface_hub.scan_cache_dir`` walks the cache once and reports every file
in every revision together with the blob behind it. This module turns that
report into the units the command line talks about: a repository holds
revisions, a revision holds artifacts, and an artifact is addressed as
``org/repo:quant``.

Two kinds of file live in the cache without belonging to an artifact. Blobs
that no revision points at are strays, and blobs with an ``.incomplete``
suffix are interrupted downloads. Both are found here so that ``prune`` can
offer them, and neither is ever inferred from a file name: the reference set
comes from the same walk that reports the artifacts, so a file type nobody
thought about still counts as a reference.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound

from . import refs
from .refs import Reference

INCOMPLETE_SUFFIX = ".incomplete"


class CacheError(Exception):
    """The cache directory cannot be read."""


class Kind(Enum):
    """What an artifact is for, which decides its place in the listing."""

    MODEL = "model"
    PROJECTOR = "projector"
    EXTRA = "extra"

    @property
    def order(self) -> int:
        return {Kind.MODEL: 0, Kind.PROJECTOR: 1, Kind.EXTRA: 2}[self]


def kind_of(quant: str) -> Kind:
    """What an artifact with this quant label is for.

    Read from the label rather than from the file, because ``pull`` decides the
    same thing for a file it has not downloaded yet.
    """
    lowered = quant.lower()
    if lowered.startswith("mmproj"):
        return Kind.PROJECTOR
    if lowered.startswith("mtp"):
        return Kind.EXTRA
    return Kind.MODEL


@dataclass(frozen=True, slots=True)
class Blob:
    """One file in a repository's ``blobs`` directory."""

    path: Path
    size: int
    modified: float

    @property
    def short_id(self) -> str:
        return self.path.name[:8]


@dataclass(frozen=True, slots=True)
class Artifact:
    """One addressable GGUF file, or one split file with all its shards."""

    quant: str
    kind: Kind
    entries: tuple[Path, ...]
    blobs: tuple[Blob, ...]

    @property
    def size(self) -> int:
        """Bytes held by this artifact's blobs, each counted once."""
        return sum(blob.size for blob in _unique_blobs(self.blobs))

    @property
    def modified(self) -> float:
        return max(blob.modified for blob in self.blobs)

    @property
    def shards(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class Revision:
    """One snapshot of a repository, named by its commit hash."""

    commit: str
    path: Path
    names: tuple[str, ...]
    artifacts: tuple[Artifact, ...]
    others: tuple[Path, ...]
    other_blobs: tuple[Blob, ...]
    # The hub reports a revision with no files at all, and dates it by the
    # directory. Without that fallback such a revision looks like 1970.
    reported_modified: float = 0.0

    @property
    def short_commit(self) -> str:
        return self.commit[:7]

    @property
    def detached(self) -> bool:
        """Whether no name in ``refs`` points at this revision.

        A detached revision cannot be reached by ``org/repo`` any more, so
        nothing but its own hash can select it.
        """
        return not self.names

    @property
    def size(self) -> int:
        return sum(blob.size for blob in _unique_blobs(self.blobs()))

    @property
    def modified(self) -> float:
        return max((blob.modified for blob in self.blobs()), default=self.reported_modified)

    @property
    def entries(self) -> tuple[Path, ...]:
        """Every file in the snapshot, artifacts and other files alike."""
        return tuple(entry for artifact in self.artifacts for entry in artifact.entries) + self.others

    def blobs(self) -> list[Blob]:
        """Every blob the snapshot points at, artifacts and other files alike."""
        found = [blob for artifact in self.artifacts for blob in artifact.blobs]
        found.extend(self.other_blobs)
        return found


@dataclass(frozen=True, slots=True)
class Repo:
    """One cached repository, holding one revision per download."""

    repo_id: str
    path: Path
    revisions: tuple[Revision, ...]
    strays: tuple[Blob, ...]
    incomplete: tuple[Blob, ...]
    accessed: float

    @property
    def org(self) -> str:
        return self.repo_id.partition("/")[0]

    @property
    def name(self) -> str:
        return self.repo_id.partition("/")[2]

    @property
    def size(self) -> int:
        """Bytes on disk, with blobs shared between revisions counted once."""
        blobs = [blob for revision in self.revisions for blob in revision.blobs()]
        blobs.extend(self.strays)
        blobs.extend(self.incomplete)
        return sum(blob.size for blob in _unique_blobs(blobs))

    @property
    def modified(self) -> float:
        return max((revision.modified for revision in self.revisions), default=0.0)

    @property
    def refs_dir(self) -> Path:
        return self.path / "refs"


@dataclass(frozen=True, slots=True)
class Cache:
    """The whole cache, plus the blob reference counts derived from it."""

    path: Path
    repos: tuple[Repo, ...]
    warnings: tuple[str, ...]
    references: Counter[Path] = field(default_factory=Counter)
    entry_blobs: dict[Path, Path] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return sum(repo.size for repo in self.repos)

    def reference_count(self, blob: Path) -> int:
        """How many snapshot entries in the whole cache point at ``blob``."""
        return self.references[blob]

    def blob_behind(self, entry: Path) -> Path | None:
        """The blob a snapshot entry points at, or ``None`` for an unknown entry."""
        return self.entry_blobs.get(entry)

    def is_shared(self, artifact: Artifact) -> bool:
        """Whether every blob of ``artifact`` is also reached from elsewhere.

        Counted per blob. The shards of a split file each have a blob of their
        own, so comparing against the number of entries in the artifact would
        ask a two shard artifact for three references before calling it shared.
        """
        own = Counter(blob.path for blob in artifact.blobs)
        return all(self.reference_count(path) > count for path, count in own.items())


def _unique_blobs(blobs: Iterable[Blob]) -> list[Blob]:
    seen: dict[Path, Blob] = {}
    for blob in blobs:
        seen.setdefault(blob.path, blob)
    return list(seen.values())


def scan(cache_dir: Path) -> Cache:
    """Read ``cache_dir`` and build the model.

    Raises:
        CacheError: the directory is missing, or is not a directory.
    """
    try:
        report = scan_cache_dir(cache_dir)
    except CacheNotFound as error:
        raise CacheError(f"cache directory not found: {cache_dir}") from error
    except ValueError as error:
        raise CacheError(f"cannot read cache directory {cache_dir}: {error}") from error

    references: Counter[Path] = Counter()
    entry_blobs: dict[Path, Path] = {}
    for hub_repo in report.repos:
        for hub_revision in hub_repo.revisions:
            for hub_file in hub_revision.files:
                references[hub_file.blob_path] += 1
                entry_blobs[hub_file.file_path] = hub_file.blob_path

    incomplete_by_repo: dict[Path, list[Blob]] = {}
    for hub_incomplete in report.incomplete_files:
        blobs_dir = hub_incomplete.file_path.parent
        incomplete_by_repo.setdefault(blobs_dir.parent, []).append(
            Blob(
                path=hub_incomplete.file_path,
                size=hub_incomplete.size_on_disk,
                modified=_mtime_of(hub_incomplete.file_path),
            )
        )

    repos = []
    for hub_repo in report.repos:
        revisions = tuple(
            sorted(
                (_revision_of(hub_repo.repo_id, hub_revision) for hub_revision in hub_repo.revisions),
                key=lambda revision: (-revision.modified, revision.commit),
            )
        )
        repos.append(
            Repo(
                repo_id=hub_repo.repo_id,
                path=hub_repo.repo_path,
                revisions=revisions,
                strays=_strays_of(hub_repo.repo_path, references),
                incomplete=tuple(incomplete_by_repo.get(hub_repo.repo_path, ())),
                accessed=hub_repo.last_accessed,
            )
        )

    return Cache(
        path=Path(cache_dir),
        repos=tuple(sorted(repos, key=lambda repo: repo.repo_id.lower())),
        warnings=tuple(str(warning) for warning in report.warnings),
        references=references,
        entry_blobs=entry_blobs,
    )


def _revision_of(repo_id: str, hub_revision) -> Revision:
    gguf: dict[str, tuple[Path, Blob]] = {}
    others: list[Path] = []
    other_blobs: list[Blob] = []

    for hub_file in sorted(hub_revision.files, key=lambda item: str(item.file_path)):
        blob = Blob(
            path=hub_file.blob_path,
            size=hub_file.size_on_disk,
            modified=hub_file.blob_last_modified,
        )
        if not refs.is_gguf(hub_file.file_name):
            others.append(hub_file.file_path)
            other_blobs.append(blob)
            continue
        # The hub reports the base name only, and a snapshot may hold two files
        # of that name in different directories. The path inside the snapshot is
        # what tells them apart, and it is also the name the hub itself uses for
        # the file, so the grouping is the same one ``pull`` applies remotely.
        inside = hub_file.file_path.relative_to(hub_revision.snapshot_path).as_posix()
        gguf[inside] = (hub_file.file_path, blob)

    artifacts = [
        _artifact(
            label,
            tuple(gguf[name][0] for name in names),
            tuple(gguf[name][1] for name in names),
        )
        for label, names in refs.group_by_artifact(repo_id, gguf).items()
    ]

    artifacts.sort(key=lambda artifact: (artifact.kind.order, artifact.quant.lower()))
    return Revision(
        commit=hub_revision.commit_hash,
        path=hub_revision.snapshot_path,
        names=tuple(sorted(hub_revision.refs)),
        artifacts=tuple(artifacts),
        others=tuple(others),
        other_blobs=tuple(other_blobs),
        reported_modified=hub_revision.last_modified,
    )


def _artifact(quant: str, entries: tuple[Path, ...], blobs: tuple[Blob, ...]) -> Artifact:
    return Artifact(quant=quant, kind=kind_of(quant), entries=entries, blobs=blobs)


def _strays_of(repo_path: Path, references: Counter[Path]) -> tuple[Blob, ...]:
    """Blobs in ``repo_path`` that no revision points at.

    Interrupted downloads are left out, because they are reported on their own
    and ``prune`` describes them differently.
    """
    blobs_dir = repo_path / "blobs"
    if not blobs_dir.is_dir():
        return ()

    strays = []
    for candidate in sorted(blobs_dir.iterdir()):
        if not candidate.is_file() or candidate.name.endswith(INCOMPLETE_SUFFIX):
            continue
        # The hub reports every blob as file_path.resolve(), so a symlink
        # anywhere above the blob gives the same file two spellings. Comparing
        # the unresolved one would call a live blob unreferenced.
        resolved = _resolve(candidate)
        if references[resolved] > 0:
            continue
        stats = _stat(resolved)
        if stats is None:
            # Gone since the directory was read, by a download or another prune.
            continue
        strays.append(Blob(path=resolved, size=stats.st_size, modified=stats.st_mtime))
    return tuple(strays)


def _mtime_of(path: Path) -> float:
    stats = _stat(path)
    return stats.st_mtime if stats is not None else 0.0


def _stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


@dataclass(frozen=True, slots=True)
class Match:
    """One artifact together with where it sits."""

    repo: Repo
    revision: Revision
    artifact: Artifact

    @property
    def reference(self) -> str:
        return f"{self.repo.repo_id}:{self.artifact.quant}"


def select(cache: Cache, references: list[Reference]) -> list[Repo]:
    """The listing narrowed to what ``references`` names.

    A repository comes back holding only the artifacts that matched, so the
    sizes printed are the sizes of the rows printed. Unreferenced blobs and
    interrupted downloads are kept only while no reference names a quant,
    because a quant filter is asking about artifacts.

    With no references the model is returned untouched.
    """
    if not references:
        return list(cache.repos)

    selected: list[Repo] = []
    for repo in cache.repos:
        keep_loose_blobs = any(
            reference.quant is None and _selects_repo(reference, repo) for reference in references
        )
        revisions = []
        for revision in repo.revisions:
            artifacts = tuple(
                artifact
                for artifact in revision.artifacts
                if any(_selects(reference, repo, artifact) for reference in references)
            )
            if artifacts:
                revisions.append(replace(revision, artifacts=artifacts, others=(), other_blobs=()))
        if not revisions and not (keep_loose_blobs and (repo.strays or repo.incomplete)):
            continue
        selected.append(
            replace(
                repo,
                revisions=tuple(revisions),
                strays=repo.strays if keep_loose_blobs else (),
                incomplete=repo.incomplete if keep_loose_blobs else (),
            )
        )
    return selected


def repos_matching(cache: Cache, reference: Reference) -> list[Repo]:
    """Every repository that ``reference`` selects."""
    return [repo for repo in cache.repos if _selects_repo(reference, repo)]


def _artifacts_matching(repo: Repo, text: str) -> tuple[Match, ...]:
    return tuple(
        Match(repo=repo, revision=revision, artifact=artifact)
        for revision in repo.revisions
        for artifact in revision.artifacts
        if _contains(artifact.quant, text)
    )


def _selects(reference: Reference, repo: Repo, artifact: Artifact) -> bool:
    if not _selects_repo(reference, repo):
        return False
    if reference.quant is not None:
        return _contains(artifact.quant, reference.quant)
    if reference.text is not None:
        return _contains(repo.repo_id, reference.text) or _contains(artifact.quant, reference.text)
    return True


def _selects_repo(reference: Reference, repo: Repo) -> bool:
    if reference.repo_id is not None:
        return repo.repo_id.lower() == reference.repo_id.lower()
    if reference.text is not None:
        if _contains(repo.repo_id, reference.text):
            return True
        return any(
            _contains(artifact.quant, reference.text)
            for revision in repo.revisions
            for artifact in revision.artifacts
        )
    return True


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


class ResolveError(Exception):
    """A reference that names no artifact, or more than one repository."""


@dataclass(frozen=True, slots=True)
class RepoTarget:
    """A whole repository, as ``rm org/repo`` names it."""

    repo: Repo


@dataclass(frozen=True, slots=True)
class ArtifactTarget:
    """Artifacts inside one repository, in every revision that holds them."""

    repo: Repo
    matches: tuple[Match, ...]


def resolve(cache: Cache, reference: Reference) -> RepoTarget | ArtifactTarget:
    """Work out what a reference names, for a command that deletes.

    Deletion needs one answer, so a reference that fits several repositories is
    refused with the candidates named. Listing accepts the same references and
    keeps every match, which is what :func:`matches` is for.

    Raises:
        ResolveError: the reference names no repository, several repositories,
            no such quant, or a quant without a repository.
    """
    if reference.repo_id is None and reference.quant is not None and reference.text is None:
        raise ResolveError(
            f"{reference.raw!r} names a quant but no repository; "
            f"write org/repo:{reference.quant} to remove it, or "
            f"'ls :{reference.quant}' to see where it is"
        )

    candidates = repos_matching(cache, reference)
    if not candidates:
        raise ResolveError(f"nothing in the cache matches {reference.raw!r}")
    if len(candidates) > 1:
        names = ", ".join(repo.repo_id for repo in candidates)
        raise ResolveError(f"{reference.raw!r} fits {len(candidates)} repositories: {names}")

    repo = candidates[0]
    if reference.quant is None:
        # A bare word may have matched the repository, or only an artifact in
        # it. Listing narrows to the artifact, and a deletion has to mean the
        # same thing, or the same word would delete more than it showed.
        if reference.text is not None and not _contains(repo.repo_id, reference.text):
            return ArtifactTarget(repo=repo, matches=_artifacts_matching(repo, reference.text))
        return RepoTarget(repo=repo)

    found = tuple(
        Match(repo=repo, revision=revision, artifact=artifact)
        for revision in repo.revisions
        for artifact in revision.artifacts
        if artifact.quant.lower() == reference.quant.lower()
    )
    if not found:
        available = ", ".join(
            sorted({artifact.quant for revision in repo.revisions for artifact in revision.artifacts})
        )
        raise ResolveError(f"{repo.repo_id} holds no quant {reference.quant!r}; it holds {available}")
    return ArtifactTarget(repo=repo, matches=found)
