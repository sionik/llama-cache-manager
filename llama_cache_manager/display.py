"""Terminal output.

One renderer draws the listing and the deletion preview, so a plan is shown in
the same shape the user already read in ``ls``. Column widths are worked out
across the whole output rather than per repository, so the size column lines up
everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TextIO

from . import age
from .cache import Artifact, Blob, Cache, Kind, Repo, Revision
from .download import Download, DownloadPlan
from .removal import Item, Plan, Reason

SCHEMA_VERSION = 1

_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")


def human_size(size: int) -> str:
    """A size with one decimal and a binary unit, as ``ls`` prints it."""
    value = float(size)
    for unit in _UNITS:
        if value < 1024.0 or unit == _UNITS[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable: the unit list ends with a return")


def count(number: int, singular: str, plural: str | None = None) -> str:
    """``1 blob`` or ``3 blobs``, with the number in front."""
    word = singular if number == 1 else (plural or f"{singular}s")
    return f"{number} {word}"


@dataclass(frozen=True, slots=True)
class Style:
    """The colours in use, empty when the output is not a terminal."""

    reset: str = ""
    repo: str = ""
    total: str = ""
    revision: str = ""
    detached: str = ""
    model: str = ""
    projector: str = ""
    extra: str = ""
    stray: str = ""
    dim: str = ""

    @classmethod
    def plain(cls) -> Style:
        return cls()

    @classmethod
    def coloured(cls) -> Style:
        return cls(
            reset="\033[0m",
            repo="\033[1m",
            total="\033[36m",
            revision="\033[1m",
            detached="\033[31m",
            model="\033[34m",
            projector="\033[33m",
            extra="\033[35m",
            stray="\033[31m",
            dim="\033[90m",
        )

    def paint(self, colour: str, text: str) -> str:
        if not colour:
            return text
        return f"{colour}{text}{self.reset}"

    def for_kind(self, kind: Kind) -> str:
        return {Kind.MODEL: self.model, Kind.PROJECTOR: self.projector, Kind.EXTRA: self.extra}[kind]


def pick_style(when: str, stream: TextIO) -> Style:
    """Decide on colour from ``--color``, ``NO_COLOR`` and the stream."""
    if when == "always":
        return Style.coloured()
    if when == "never":
        return Style.plain()
    if os.environ.get("NO_COLOR"):
        return Style.plain()
    if not stream.isatty():
        return Style.plain()
    return Style.coloured()


@dataclass(frozen=True, slots=True)
class Layout:
    """Column widths shared by every line of one listing.

    Repository names and artifact labels get a column each, because a long
    repository name would otherwise push every artifact size far to the right.
    """

    repo: int
    artifact: int
    size: int

    @classmethod
    def measure(cls, repo_names, artifact_labels, sizes) -> Layout:
        return cls(
            repo=max((len(name) for name in repo_names), default=0),
            artifact=max((len(label) for label in artifact_labels), default=0),
            size=max((len(human_size(size)) for size in sizes), default=0),
        )


def _artifact_label(artifact: Artifact) -> str:
    return f":{artifact.quant}"


def render_brief(repos: list[Repo], style: Style, out: TextIO) -> None:
    """One line per repository, for a look at what holds the space."""
    layout = Layout.measure([repo.repo_id for repo in repos], [], [repo.size for repo in repos])
    for repo in repos:
        print(
            f"{style.paint(style.repo, repo.repo_id.ljust(layout.repo))}"
            f"  {style.paint(style.total, human_size(repo.size).rjust(layout.size))}",
            file=out,
        )


def render_cache(
    cache: Cache,
    repos: list[Repo],
    now: float,
    style: Style,
    garbage: Plan | None,
    show_warnings: bool,
    out: TextIO,
) -> None:
    """Print the listing: repository, revisions, then addressable artifacts."""
    repo_names = [repo.repo_id for repo in repos]
    labels = [
        "    " + _artifact_label(artifact)
        for repo in repos
        for revision in repo.revisions
        for artifact in revision.artifacts
    ]
    sizes = [repo.size for repo in repos]
    sizes.extend(
        artifact.size for repo in repos for revision in repo.revisions for artifact in revision.artifacts
    )
    layout = Layout.measure(repo_names, labels, sizes)

    for index, repo in enumerate(repos):
        if index:
            print(file=out)
        _print_repo(repo, cache, now, style, layout, out)

    if repos:
        print(file=out)
    print(
        style.paint(
            style.dim,
            f"{human_size(sum(repo.size for repo in repos))} in "
            f"{count(len(repos), 'repository', 'repositories')}, "
            f"{count(sum(len(repo.revisions) for repo in repos), 'revision')}",
        ),
        file=out,
    )
    if garbage is not None and not garbage.empty:
        print(style.paint(style.dim, _garbage_hint(garbage)), file=out)
    for line in warning_lines(cache, show_warnings):
        print(style.paint(style.dim if not show_warnings else style.stray, line), file=out)


def _print_repo(repo: Repo, cache: Cache, now: float, style: Style, layout: Layout, out: TextIO) -> None:
    print(
        f"{style.paint(style.repo, repo.repo_id.ljust(layout.repo))}"
        f"  {style.paint(style.total, human_size(repo.size).rjust(layout.size))}"
        f"  {style.paint(style.dim, count(len(repo.revisions), 'revision'))}",
        file=out,
    )
    for revision in repo.revisions:
        _print_revision(repo, revision, cache, now, style, layout, out)

    loose = [(blob, "unreferenced blob") for blob in repo.strays]
    loose.extend((blob, "interrupted download") for blob in repo.incomplete)
    if loose:
        # Set apart from the revisions above, which these files do not belong to.
        print(
            f"  {style.paint(style.stray, 'not referenced')}"
            f"  {style.paint(style.dim, human_size(sum(blob.size for blob, _ in loose)))}",
            file=out,
        )
        for blob, what in loose:
            _print_blob(blob, what, style, layout, out)


def _print_revision(
    repo: Repo, revision: Revision, cache: Cache, now: float, style: Style, layout: Layout, out: TextIO
) -> None:
    if revision.detached:
        where = style.paint(style.detached, "detached")
    else:
        where = style.paint(style.dim, " ".join(revision.names))
    line = f"  {style.paint(style.revision, revision.short_commit)}  {where}"
    line += f"  {style.paint(style.dim, age.describe(now - revision.modified))}"
    # Only worth saying on a detached revision, where the reader is deciding
    # whether pruning it would win anything.
    if (
        revision.detached
        and revision.artifacts
        and all(cache.is_shared(artifact) for artifact in revision.artifacts)
    ):
        line += f"  {style.paint(style.dim, 'shares all blobs')}"
    print(line, file=out)

    for artifact in revision.artifacts:
        label = ("    " + _artifact_label(artifact)).ljust(layout.artifact)
        size = human_size(artifact.size).rjust(layout.size)
        line = f"{style.paint(style.for_kind(artifact.kind), label)}  {style.paint(style.dim, size)}"
        if artifact.shards > 1:
            line += f"  {style.paint(style.dim, count(artifact.shards, 'shard'))}"
        print(line, file=out)


def _print_blob(blob: Blob, what: str, style: Style, layout: Layout, out: TextIO) -> None:
    label = f"    {blob.short_id}".ljust(layout.artifact)
    print(
        f"{style.paint(style.stray, label)}"
        f"  {style.paint(style.dim, human_size(blob.size).rjust(layout.size))}"
        f"  {style.paint(style.dim, what)}",
        file=out,
    )


def warning_lines(cache: Cache, show_warnings: bool) -> list[str]:
    """What the scan skipped, in full or as a count."""
    if not cache.warnings:
        return []
    if show_warnings:
        return [f"warning: {warning}" for warning in cache.warnings]
    return [
        f"{count(len(cache.warnings), 'directory', 'directories')} skipped, "
        "not a cache repository (--warnings to see them)"
    ]


def _garbage_hint(garbage: Plan) -> str:
    parts = []
    for reason, singular in (
        (Reason.DETACHED, "detached revision"),
        (Reason.STRAY, "unreferenced blob"),
        (Reason.INCOMPLETE, "interrupted download"),
    ):
        found = garbage.items_for(reason)
        if found:
            parts.append(count(len(found), singular))
    reclaims = human_size(garbage.freed) if garbage.freed else "nothing"
    return f"prune would remove {', '.join(parts)}, reclaiming {reclaims}"


def render_plan(
    plan: Plan, now: float, style: Style, dry_run: bool, out: TextIO, heading: str | None = None
) -> None:
    """Print what a plan would remove, grouped the way ``ls`` groups.

    ``heading`` replaces the default line for a caller whose removal was
    already agreed to, such as the one an update makes after its download.
    """
    if heading is None:
        heading = "Would remove" if dry_run else "About to remove"
    print(style.paint(style.repo, heading), file=out)
    print(file=out)

    repo_names = [item.repo.repo_id for item in plan.items]
    labels: list[str] = []
    sizes = [item.nominal_size for item in plan.items]
    for item in plan.items:
        if item.artifact is not None:
            labels.append("  " + _artifact_label(item.artifact))
            sizes.append(item.artifact.size)
        if item.revision is not None:
            for artifact in item.revision.artifacts:
                labels.append("    " + _artifact_label(artifact))
                sizes.append(artifact.size)
        if item.blob is not None:
            labels.append(f"    {item.blob.short_id}")
    layout = Layout.measure(repo_names, labels, sizes)

    for repo in plan.repos:
        print(style.paint(style.repo, repo.repo_id), file=out)
        for item in plan.items:
            if item.repo.repo_id != repo.repo_id:
                continue
            _print_item(item, plan, now, style, layout, out)
        print(file=out)

    print(style.paint(style.total, _plan_summary(plan)), file=out)
    detail = _plan_detail(plan)
    if detail:
        print(style.paint(style.dim, detail), file=out)
    for path in plan.withheld:
        print(
            style.paint(style.stray, f"left alone, it points out of the cache: {path}"),
            file=out,
        )


def _print_item(item: Item, plan: Plan, now: float, style: Style, layout: Layout, out: TextIO) -> None:
    if item.blob is not None:
        _print_blob(item.blob, item.reason.value, style, layout, out)
        return

    if item.artifact is not None:
        label = ("  " + _artifact_label(item.artifact)).ljust(layout.artifact)
        line = f"{style.paint(style.for_kind(item.artifact.kind), label)}"
        line += f"  {style.paint(style.dim, human_size(item.artifact.size).rjust(layout.size))}"
        if item.revision is not None:
            line += f"  {style.paint(style.dim, item.revision.short_commit)}"
        print(line, file=out)
        return

    if item.revision is not None:
        marker = style.paint(style.detached, item.reason.value)
        print(
            f"  {style.paint(style.revision, item.revision.short_commit)}  {marker}"
            f"  {style.paint(style.dim, age.describe(now - item.revision.modified))}",
            file=out,
        )
        for artifact in item.revision.artifacts:
            label = ("    " + _artifact_label(artifact)).ljust(layout.artifact)
            print(
                f"{style.paint(style.for_kind(artifact.kind), label)}"
                f"  {style.paint(style.dim, human_size(artifact.size).rjust(layout.size))}",
                file=out,
            )
        return

    print(
        f"  {style.paint(style.detached, 'the whole repository')}"
        f"  {style.paint(style.dim, human_size(item.repo.size).rjust(layout.size))}",
        file=out,
    )


def _plan_summary(plan: Plan) -> str:
    if plan.freed == 0:
        return "Reclaims nothing: every blob is also reached from something that stays"
    return f"Reclaims {human_size(plan.freed)}"


def _plan_detail(plan: Plan) -> str:
    parts = []
    if plan.entries:
        parts.append(count(len(plan.entries), "file"))
    if plan.blobs:
        parts.append(count(len(plan.blobs), "blob"))
    if plan.revision_dirs:
        parts.append(count(len(plan.revision_dirs), "revision"))
    if plan.repo_dirs:
        parts.append(count(len(plan.repo_dirs), "repository directory", "repository directories"))
    return ", ".join(parts)


def cache_as_json(cache: Cache, repos: list[Repo], now: float) -> dict:
    """The listing as data, with byte counts and stamps left unformatted."""
    return {
        "schema": SCHEMA_VERSION,
        "cache_dir": str(cache.path),
        "size": sum(repo.size for repo in repos),
        "repositories": [
            {
                "repo_id": repo.repo_id,
                "path": str(repo.path),
                "size": repo.size,
                "revisions": [
                    {
                        "commit": revision.commit,
                        "refs": list(revision.names),
                        "detached": revision.detached,
                        "size": revision.size,
                        "modified": revision.modified,
                        "artifacts": [
                            {
                                "reference": f"{repo.repo_id}:{artifact.quant}",
                                "quant": artifact.quant,
                                "kind": artifact.kind.value,
                                "size": artifact.size,
                                "modified": artifact.modified,
                                "shards": artifact.shards,
                                "shared": cache.is_shared(artifact),
                                "files": [str(entry) for entry in artifact.entries],
                            }
                            for artifact in revision.artifacts
                        ],
                    }
                    for revision in repo.revisions
                ],
                "unreferenced_blobs": [{"path": str(blob.path), "size": blob.size} for blob in repo.strays],
                "interrupted_downloads": [
                    {"path": str(blob.path), "size": blob.size} for blob in repo.incomplete
                ],
            }
            for repo in repos
        ],
        "warnings": list(cache.warnings),
    }


def plan_as_json(plan: Plan, dry_run: bool) -> dict:
    """A plan as data, so a script can check it before agreeing to it."""
    return {
        "schema": SCHEMA_VERSION,
        "dry_run": dry_run,
        "reclaims": plan.freed,
        "items": [
            {
                "reason": item.reason.value,
                "reference": item.label,
                "repo_id": item.repo.repo_id,
                "commit": item.revision.commit if item.revision else None,
                "quant": item.artifact.quant if item.artifact else None,
                "nominal_size": item.nominal_size,
            }
            for item in plan.items
        ],
        "files": [str(entry) for entry in plan.entries],
        "blobs": [str(blob.path) for blob in plan.blobs],
        "revisions": [str(path) for path in plan.revision_dirs],
        "repositories": [str(path) for path in plan.repo_dirs],
        "withheld": [str(path) for path in plan.withheld],
    }


def render_download_plan(plan: DownloadPlan, style: Style, dry_run: bool, out: TextIO) -> None:
    """Print what a download would fetch, grouped the way ``ls`` groups."""
    heading = "Would download into" if dry_run else "About to download into"
    print(f"{style.paint(style.repo, heading)} {style.paint(style.dim, str(plan.cache_dir))}", file=out)
    print(file=out)

    labels = ["  :" + item.quant for item in plan.downloads]
    sizes = [item.size for item in plan.downloads]
    layout = Layout.measure(list(plan.repo_ids), labels, sizes)

    for repo_id in plan.repo_ids:
        for index, item in enumerate(item for item in plan.downloads if item.repo_id == repo_id):
            if index == 0:
                print(
                    f"{style.paint(style.repo, repo_id)}"
                    f"  {style.paint(style.dim, item.revision)}"
                    f"  {style.paint(style.revision, item.commit[:7])}",
                    file=out,
                )
            _print_download(item, style, layout, out)
        print(file=out)

    print(style.paint(style.total, _download_summary(plan)), file=out)
    detail = _download_detail(plan)
    if detail:
        print(style.paint(style.dim, detail), file=out)


def _print_download(item: Download, style: Style, layout: Layout, out: TextIO) -> None:
    label = ("  :" + item.quant).ljust(layout.artifact)
    line = f"{style.paint(style.for_kind(item.kind), label)}"
    line += f"  {style.paint(style.dim, human_size(item.size).rjust(layout.size))}"
    if item.shards > 1:
        line += f"  {style.paint(style.dim, count(item.shards, 'shard'))}"
    if item.cached:
        line += f"  {style.paint(style.dim, 'already in the cache')}"
    elif item.transfer < item.size:
        line += f"  {style.paint(style.dim, human_size(item.transfer) + ' to fetch')}"
    print(line, file=out)


def _download_summary(plan: DownloadPlan) -> str:
    if plan.transfer == 0:
        return "Transfers nothing, every file is already in the cache"
    if plan.transfer == plan.size:
        return f"Transfers {human_size(plan.transfer)}"
    return f"Transfers {human_size(plan.transfer)} of {human_size(plan.size)}"


def _download_detail(plan: DownloadPlan) -> str:
    """What the list above does not already say.

    A single artifact needs no total: its row already carries the size and the
    shard count.
    """
    if len(plan.downloads) < 2:
        return ""
    parts = [count(len(plan.downloads), "artifact")]
    files = sum(item.shards for item in plan.downloads)
    if files != len(plan.downloads):
        parts.append(count(files, "file"))
    return ", ".join(parts)


def download_plan_as_json(plan: DownloadPlan, dry_run: bool) -> dict:
    """A download as data, so a script can check the size before agreeing."""
    return {
        "schema": SCHEMA_VERSION,
        "dry_run": dry_run,
        "cache_dir": str(plan.cache_dir),
        "size": plan.size,
        "transfer": plan.transfer,
        "downloads": [
            {
                "reference": item.reference,
                "repo_id": item.repo_id,
                "revision": item.revision,
                "commit": item.commit,
                "quant": item.quant,
                "kind": item.kind.value,
                "size": item.size,
                "transfer": item.transfer,
                "shards": item.shards,
                "cached": item.cached,
                "files": [
                    {"name": file.name, "size": file.size, "cached": file.cached} for file in item.files
                ],
            }
            for item in plan.downloads
        ],
    }


def render_outdated(items, style: Style, out: TextIO) -> None:
    """One line per name the hub has moved, old commit to new."""
    layout = Layout.measure([item.repo_id for item in items], [], [])
    width = max((len(item.name) for item in items), default=0)
    for item in items:
        print(
            f"{style.paint(style.repo, item.repo_id.ljust(layout.repo))}"
            f"  {style.paint(style.dim, item.name.ljust(width))}"
            f"  {style.paint(style.revision, item.local_commit[:7])}"
            f" {style.paint(style.dim, '->')} "
            f"{style.paint(style.revision, item.remote_commit[:7])}",
            file=out,
        )


def skipped_lines(skipped) -> list[str]:
    """Why a repository was left out of an update."""
    return [f"left alone, {item.reason}" for item in skipped]


def update_plan_as_json(plan, dry_run: bool, keep: bool) -> dict:
    """An update as data: what moved, what it fetches, what it leaves alone."""
    document = download_plan_as_json(plan.downloads, dry_run)
    document["keep"] = keep
    document["updates"] = [
        {
            "repo_id": item.repo_id,
            "ref": item.name,
            "local_commit": item.local_commit,
            "remote_commit": item.remote_commit,
            "quants": list(item.quants),
        }
        for item in plan.replaced
    ]
    document["skipped"] = [
        {"repo_id": item.repo_id, "ref": item.name, "reason": item.reason} for item in plan.skipped
    ]
    return document
