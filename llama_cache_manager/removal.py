"""Deletion as a plan that is first shown and then carried out.

Everything the tool removes is expressed as a plan. The preview prints the
plan, the confirmation asks about the plan, and the deletion walks the same
plan. A dry run is the plan without the last step, so what a dry run reports
and what a real run does cannot drift apart.

The reclaimed size is derived once, in :func:`build`. A blob goes away only
when every snapshot entry that points at it is part of the plan, which is why
deleting one of two revisions that share their blobs correctly reports that
nothing is reclaimed.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .cache import Artifact, Blob, Cache, Repo, Revision


class RemovalError(Exception):
    """A plan asks for something outside the cache directory."""


class Reason(Enum):
    """Why an item is in a plan, which is what the preview groups by."""

    ARTIFACT = "artifact"
    REVISION = "revision"
    REPOSITORY = "repository"
    DETACHED = "detached revision"
    AGED = "revision older than the cutoff"
    SUPERSEDED = "revision replaced by an update"
    STRAY = "unreferenced blob"
    INCOMPLETE = "interrupted download"


@dataclass(frozen=True, slots=True)
class Item:
    """One thing the user asked to remove, before sharing is taken into account."""

    reason: Reason
    repo: Repo
    revision: Revision | None = None
    artifact: Artifact | None = None
    blob: Blob | None = None

    @property
    def label(self) -> str:
        if self.artifact is not None:
            return f"{self.repo.repo_id}:{self.artifact.quant}"
        if self.blob is not None:
            return self.blob.path.name
        if self.revision is not None:
            return self.revision.short_commit
        return self.repo.repo_id

    @property
    def nominal_size(self) -> int:
        """Size of the item on its own, before shared blobs are discounted."""
        if self.artifact is not None:
            return self.artifact.size
        if self.blob is not None:
            return self.blob.size
        if self.revision is not None:
            return self.revision.size
        return self.repo.size

    @property
    def entries(self) -> tuple[Path, ...]:
        if self.artifact is not None:
            return self.artifact.entries
        if self.revision is not None:
            return self.revision.entries
        if self.blob is not None:
            return ()
        return tuple(entry for revision in self.repo.revisions for entry in revision.entries)


@dataclass(frozen=True, slots=True)
class Plan:
    """A complete deletion, with the reclaimed size already derived."""

    cache: Cache
    items: tuple[Item, ...]
    entries: tuple[Path, ...]
    blobs: tuple[Blob, ...]
    revision_dirs: tuple[Path, ...]
    ref_files: tuple[Path, ...]
    repo_dirs: tuple[Path, ...]
    freed: int
    # Paths the scan reported but that lie outside everything the cache owns.
    # Held back rather than removed, and named so the reader knows.
    withheld: tuple[Path, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.items

    def items_for(self, reason: Reason) -> tuple[Item, ...]:
        return tuple(item for item in self.items if item.reason is reason)

    @property
    def repos(self) -> tuple[Repo, ...]:
        seen: dict[str, Repo] = {}
        for item in self.items:
            seen.setdefault(item.repo.repo_id, item.repo)
        return tuple(seen.values())


def build(cache: Cache, items: list[Item]) -> Plan:
    """Turn a list of items into a plan, deriving what disk space comes back.

    A path that lies outside everything the cache owns is held back and named
    in ``Plan.withheld``. The rest of the plan still runs, because one odd
    symlink is no reason to leave the whole cache alone.
    """
    owned = _owned_roots(cache)
    withheld: dict[Path, None] = {}

    def keep(path: Path) -> bool:
        if _inside_any(path, owned):
            return True
        withheld[path] = None
        return False

    entries: dict[Path, None] = {}
    direct_blobs: dict[Path, Blob] = {}

    for item in items:
        for entry in item.entries:
            if keep(entry):
                entries[entry] = None
        if item.blob is not None and keep(item.blob.path):
            direct_blobs[item.blob.path] = item.blob

    blobs_by_path = _blobs_by_path(cache)
    losing_references: Counter[Path] = Counter()
    for entry in entries:
        blob_path = cache.blob_behind(entry)
        if blob_path is not None:
            losing_references[blob_path] += 1

    freed_blobs: dict[Path, Blob] = dict(direct_blobs)
    for blob_path, lost in losing_references.items():
        if cache.reference_count(blob_path) <= lost:
            blob = blobs_by_path.get(blob_path)
            if blob is not None and keep(blob_path):
                freed_blobs[blob_path] = blob

    revision_dirs, ref_files = _emptied_revisions(cache, entries.keys(), items)
    revision_dirs = {path for path in revision_dirs if keep(path)}
    ref_files = {path for path in ref_files if keep(path)}
    repo_dirs = {
        path for path in _emptied_repos(cache, revision_dirs, freed_blobs.keys(), items) if keep(path)
    }

    return Plan(
        cache=cache,
        items=tuple(items),
        entries=tuple(sorted(entries)),
        blobs=tuple(sorted(freed_blobs.values(), key=lambda blob: blob.path)),
        revision_dirs=tuple(sorted(revision_dirs)),
        ref_files=tuple(sorted(ref_files)),
        repo_dirs=tuple(sorted(repo_dirs)),
        freed=_freed_bytes(freed_blobs.values(), ref_files, repo_dirs),
        withheld=tuple(sorted(withheld)),
    )


def execute(plan: Plan) -> None:
    """Carry out ``plan``.

    Raises:
        RemovalError: a path in the plan left the cache directory.
        OSError: the file system refused a deletion.
    """
    _check_inside_cache(plan)

    for repo_dir in plan.repo_dirs:
        if repo_dir.is_symlink():
            # The repository lives elsewhere and is reached through this link.
            # Clear what it holds, then take the link away.
            shutil.rmtree(_resolved(repo_dir), ignore_errors=True)
            repo_dir.unlink(missing_ok=True)
        else:
            shutil.rmtree(repo_dir, ignore_errors=False)

    def gone(path: Path) -> bool:
        return any(path == repo_dir or repo_dir in path.parents for repo_dir in plan.repo_dirs)

    for entry in plan.entries:
        if not gone(entry):
            entry.unlink(missing_ok=True)
    for blob in plan.blobs:
        if not gone(blob.path):
            blob.path.unlink(missing_ok=True)
    for ref_file in plan.ref_files:
        if not gone(ref_file):
            ref_file.unlink(missing_ok=True)
    for revision_dir in plan.revision_dirs:
        if not gone(revision_dir):
            shutil.rmtree(revision_dir, ignore_errors=True)

    for repo in plan.repos:
        if not gone(repo.path):
            _drop_empty_dirs(repo.path, plan.cache.path)


def _freed_bytes(blobs, ref_files: set[Path], repo_dirs: set[Path]) -> int:
    """Bytes the file system gets back, counting every file that goes away.

    A repository directory is measured whole, because that is what is deleted.
    Everything outside such a directory is counted on its own, so a blob or a
    ref file is never counted twice.
    """

    def inside_removed_repo(path: Path) -> bool:
        return any(path == repo_dir or repo_dir in path.parents for repo_dir in repo_dirs)

    total = sum(_bytes_under(repo_dir) for repo_dir in repo_dirs)
    total += sum(blob.size for blob in blobs if not inside_removed_repo(blob.path))
    total += sum(_file_size(path) for path in ref_files if not inside_removed_repo(path))
    return total


def _bytes_under(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += _file_size(path)
    return total


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _blobs_by_path(cache: Cache) -> dict[Path, Blob]:
    found: dict[Path, Blob] = {}
    for repo in cache.repos:
        for revision in repo.revisions:
            for blob in revision.blobs():
                found.setdefault(blob.path, blob)
        for blob in repo.strays:
            found.setdefault(blob.path, blob)
        for blob in repo.incomplete:
            found.setdefault(blob.path, blob)
    return found


def _emptied_revisions(cache: Cache, removed_entries, items: list[Item]) -> tuple[set[Path], set[Path]]:
    removed = set(removed_entries)
    revision_dirs: set[Path] = set()
    ref_files: set[Path] = set()

    # A revision holding no files at all has no entry to remove, so asking
    # whether its entries are gone would never say yes and its directory and
    # ref would stay behind. Being named by the plan is what counts.
    targeted = _targeted_revisions(items)

    wanted_repos = {item.repo.repo_id for item in items}
    for repo in cache.repos:
        if repo.repo_id not in wanted_repos:
            continue
        for revision in repo.revisions:
            emptied = revision.path in targeted or (
                bool(revision.entries) and set(revision.entries) <= removed
            )
            if not emptied:
                continue
            revision_dirs.add(revision.path)
            for name in revision.names:
                ref_files.add(repo.refs_dir / name)
    return revision_dirs, ref_files


def _targeted_revisions(items: list[Item]) -> set[Path]:
    """Revisions the plan removes as a whole rather than file by file."""
    targeted: set[Path] = set()
    for item in items:
        if item.artifact is not None or item.blob is not None:
            continue
        if item.revision is not None:
            targeted.add(item.revision.path)
            continue
        targeted.update(revision.path for revision in item.repo.revisions)
    return targeted


def _emptied_repos(cache: Cache, revision_dirs: set[Path], freed_blobs, items: list[Item]) -> set[Path]:
    freed = set(freed_blobs)
    repo_dirs: set[Path] = set()

    wanted_repos = {item.repo.repo_id for item in items}
    for repo in cache.repos:
        if repo.repo_id not in wanted_repos:
            continue
        if any(revision.path not in revision_dirs for revision in repo.revisions):
            continue
        remaining = {blob.path for revision in repo.revisions for blob in revision.blobs()}
        remaining |= {blob.path for blob in repo.strays}
        remaining |= {blob.path for blob in repo.incomplete}
        if remaining <= freed:
            repo_dirs.add(repo.path)
    return repo_dirs


def _drop_empty_dirs(root: Path, stop: Path) -> None:
    """Remove directories under ``root`` that the deletion left empty."""
    if not root.is_dir():
        return
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        _rmdir_if_empty(path)
    for path in (root / "snapshots", root / "blobs", root / "refs"):
        _rmdir_if_empty(path)
    if root != stop:
        _rmdir_if_empty(root)


def _rmdir_if_empty(path: Path) -> None:
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        # Something appeared in the directory after the check. Leaving it in
        # place is the harmless outcome.
        pass


def _owned_roots(cache: Cache) -> tuple[Path, ...]:
    """The directories the cache owns, and may therefore delete inside.

    The cache root is one. Each repository directory is another, resolved,
    because a repository moved to a second disk by a symlink still belongs to
    the cache and its blobs then sit under that other path.
    """
    roots = {cache.path, _resolved(cache.path)}
    for repo in cache.repos:
        roots.add(repo.path)
        roots.add(_resolved(repo.path))
    return tuple(sorted(roots))


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _inside_any(path: Path, roots: tuple[Path, ...]) -> bool:
    """Whether ``path`` lies under one of ``roots``, as spelled.

    The path is not resolved here. Removing a symlink means removing the link,
    never what it points at, so the spelling is what the deletion acts on.
    """
    parents = set(path.parents)
    return any(root in parents for root in roots)


def _check_inside_cache(plan: Plan) -> None:
    """Last look before deleting, in case a plan was built by hand."""
    owned = _owned_roots(plan.cache)
    candidates = [
        *plan.entries,
        *(blob.path for blob in plan.blobs),
        *plan.revision_dirs,
        *plan.ref_files,
        *plan.repo_dirs,
    ]
    for path in candidates:
        if not _inside_any(path, owned):
            raise RemovalError(f"refusing to remove a path outside the cache: {path}")


def for_artifacts(matches) -> list[Item]:
    """Items for single artifacts, as ``rm org/repo:quant`` asks for."""
    return [
        Item(reason=Reason.ARTIFACT, repo=match.repo, revision=match.revision, artifact=match.artifact)
        for match in matches
    ]


def for_repository(repo: Repo) -> list[Item]:
    """Everything a repository holds, as ``rm org/repo`` asks for.

    Strays and interrupted downloads are named as well, so that the preview
    accounts for every byte the directory takes and nothing is left behind.
    """
    items = [Item(reason=Reason.REPOSITORY, repo=repo)]
    items.extend(Item(reason=Reason.STRAY, repo=repo, blob=blob) for blob in repo.strays)
    items.extend(Item(reason=Reason.INCOMPLETE, repo=repo, blob=blob) for blob in repo.incomplete)
    return items


def garbage(cache: Cache) -> list[Item]:
    """Everything in the cache that nothing can reach any more.

    That is a revision no name points at, a blob no revision points at, and an
    interrupted download. None of these can be selected by ``org/repo``, so
    removing them takes nothing away from the user.

    """
    items: list[Item] = []
    for repo in cache.repos:
        for revision in repo.revisions:
            if revision.detached:
                items.append(Item(reason=Reason.DETACHED, repo=repo, revision=revision))
        for blob in repo.strays:
            items.append(Item(reason=Reason.STRAY, repo=repo, blob=blob))
        for blob in repo.incomplete:
            items.append(Item(reason=Reason.INCOMPLETE, repo=repo, blob=blob))
    return items


def older_than_in(repo: Repo, cutoff: float) -> list[Item]:
    """Revisions of one repository that are older than ``cutoff``."""
    return [
        Item(reason=Reason.AGED, repo=repo, revision=revision)
        for revision in repo.revisions
        if revision.modified < cutoff
    ]


def older_than(cache: Cache, cutoff: float, exclude: set[Path] | None = None) -> list[Item]:
    """Revisions whose newest file was downloaded before ``cutoff``.

    Unlike :func:`garbage` this reaches revisions that a name still points at,
    so it removes models the user can still use. The caller shows the plan and
    asks before anything happens.
    """
    skip = exclude or set()
    items: list[Item] = []
    for repo in cache.repos:
        for revision in repo.revisions:
            if revision.path in skip or revision.modified >= cutoff:
                continue
            items.append(Item(reason=Reason.AGED, repo=repo, revision=revision))
    return items
