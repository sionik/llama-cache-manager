"""Keeping what the cache holds level with what the hub holds.

An update is a download followed by a removal, and neither half is new. What
this module adds is the comparison in between: for every name the cache follows,
ask the hub what that name points at now, and where the two differ, fetch the
quants the cache already holds at the newer commit.

Fetching a name writes the new commit into ``refs/<name>``, so the revision
that name pointed at before is usually left with no name at all. Such a
revision is what an update removes, and nothing else: a revision another name
still reaches, a revision that was already detached, a stray blob or an
interrupted download were not made unreachable by this run, and ``prune`` is
the command for those.

The reclaimed size can only be worked out once the new files exist, because a
blob the new revision also points at is not freed by dropping the old one. So
the removal plan is built from a fresh scan taken after the download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import download, refs
from .cache import Cache, Repo, Revision
from .download import DownloadPlan, Hub, SelectionError, UnavailableError
from .removal import Item, Reason


@dataclass(frozen=True, slots=True)
class Tracked:
    """One name in the cache, and the revision it points at."""

    repo: Repo
    revision: Revision
    name: str

    @property
    def repo_id(self) -> str:
        return self.repo.repo_id

    @property
    def quants(self) -> tuple[str, ...]:
        """The artifacts the cache holds at this revision."""
        return tuple(artifact.quant for artifact in self.revision.artifacts)


@dataclass(frozen=True, slots=True)
class Outdated:
    """A tracked name whose commit on the hub is not the one in the cache."""

    tracked: Tracked
    remote_commit: str

    @property
    def repo_id(self) -> str:
        return self.tracked.repo_id

    @property
    def name(self) -> str:
        return self.tracked.name

    @property
    def local_commit(self) -> str:
        return self.tracked.revision.commit

    @property
    def quants(self) -> tuple[str, ...]:
        return self.tracked.quants


@dataclass(frozen=True, slots=True)
class Skipped:
    """A name an update left alone, and why."""

    repo_id: str
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class Check:
    """What one pass over the names found.

    A name the hub would not serve is reported rather than raised, so that one
    repository the user cannot have does not stop the rest of the cache.
    """

    outdated: tuple[Outdated, ...]
    skipped: tuple[Skipped, ...]


@dataclass(frozen=True, slots=True)
class UpdatePlan:
    """What an update would fetch, and what it could not."""

    downloads: DownloadPlan
    replaced: tuple[Outdated, ...]
    skipped: tuple[Skipped, ...]

    @property
    def empty(self) -> bool:
        return not self.downloads.downloads

    @property
    def repo_ids(self) -> tuple[str, ...]:
        """The repositories the update touches, each named once."""
        seen: dict[str, None] = {}
        for item in self.replaced:
            seen[item.repo_id] = None
        return tuple(seen)

    @property
    def replaced_revisions(self) -> int:
        """How many revisions the update leaves behind, counted once each.

        ``replaced`` holds one entry per name. Two names of one repository can
        point at the same commit, and moving both leaves one revision behind,
        not two.
        """
        return len({(item.repo_id, item.local_commit) for item in self.replaced})


def tracked(cache: Cache, repos: list[Repo]) -> list[Tracked]:
    """Every name in ``repos`` that can be held against the hub.

    A detached revision is left out. Nothing records which name it once
    followed, so there is nothing to compare it against.
    """
    found = []
    for repo in repos:
        for revision in repo.revisions:
            if not revision.artifacts:
                # A revision holding no GGUF file has nothing to fetch at a
                # newer commit, so there is nothing an update could replace.
                # A cache shared with other huggingface_hub users holds such
                # revisions, and asking the hub about them would move a name
                # this tool never downloaded.
                continue
            found.extend(Tracked(repo=repo, revision=revision, name=name) for name in revision.names)
    return found


def check(hub: Hub, names: list[Tracked]) -> Check:
    """Hold every name against the hub.

    One request per name, and no file lists: the commit hash is enough to say
    whether there is anything to do.

    A name the hub answers about but will not serve, because the repository was
    taken down or went private or wants a licence, is reported as skipped. A
    hub that does not answer at all is raised instead: every name would be
    skipped, and a cache called up to date when nothing was checked is worse
    than a run that fails.

    Raises:
        DownloadError: the hub is unreachable, or refused a request.
    """
    stale = []
    skipped = []
    for item in names:
        try:
            remote = hub.head(item.repo_id, item.name)
        except UnavailableError as error:
            skipped.append(Skipped(repo_id=item.repo_id, name=item.name, reason=str(error)))
            continue
        if remote != item.revision.commit:
            stale.append(Outdated(tracked=item, remote_commit=remote))
    return Check(outdated=tuple(stale), skipped=tuple(skipped))


def plan(hub: Hub, cache_dir: Path, found: Check) -> UpdatePlan:
    """Work out what fetching the newer revisions would cost.

    One repository is planned at a time, so a repository that no longer offers
    a quant the cache holds is skipped with a reason rather than taking the
    whole run down with it. What :func:`check` already skipped is carried
    through, so the caller has one list to report.

    Raises:
        DownloadError: the hub is unreachable, or refused a request.
    """
    downloads = []
    replaced = []
    skipped = list(found.skipped)

    for item in found.outdated:
        references = [
            refs.Reference(raw=f"{item.repo_id}:{quant}", repo_id=item.repo_id, quant=quant)
            for quant in item.quants
        ]
        try:
            one = download.plan(hub, cache_dir, references, item.name)
        except (SelectionError, UnavailableError) as error:
            skipped.append(Skipped(repo_id=item.repo_id, name=item.name, reason=str(error)))
            continue
        downloads.extend(one.downloads)
        replaced.append(item)

    return UpdatePlan(
        downloads=DownloadPlan(cache_dir=Path(cache_dir), downloads=tuple(downloads)),
        replaced=tuple(replaced),
        skipped=tuple(skipped),
    )


def superseded(cache: Cache, replaced: list[Outdated]) -> list[Item]:
    """Removal items for the revisions an update has replaced.

    ``cache`` should be a scan taken after the download, so that the plan built
    from these items reports what the file system really gives back: a blob the
    new revision points at as well is no longer counted.

    A revision that a name still points at is left alone. The download moves
    the name it followed, but a second name, such as a tag that was fetched at
    the same commit, keeps the revision reachable, and reachable is what
    ``prune`` and ``rm`` mean by a revision the user still has.
    """
    wanted = {(item.repo_id, item.local_commit) for item in replaced}
    items = []
    for repo in cache.repos:
        for revision in repo.revisions:
            if revision.detached and (repo.repo_id, revision.commit) in wanted:
                items.append(Item(reason=Reason.SUPERSEDED, repo=repo, revision=revision))
    return items
