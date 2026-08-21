"""The command line.

``ls`` is the default command, so ``llama-cache-manager`` and
``llama-cache-manager unsloth`` both list. Every command that deletes or
downloads builds a plan, prints it, and asks before touching the disk.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import click

from . import age, display, download, refs, removal
from . import cache as cache_model
from . import update as update_model
from .age import CutoffError
from .cache import ArtifactTarget, CacheError, RepoTarget, ResolveError
from .download import DownloadError, SelectionError
from .refs import ReferenceError
from .removal import RemovalError

VERSION = "0.2.0"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NO_MATCH = 3
EXIT_CANCELLED = 4

COLOR_CHOICES = ("auto", "always", "never")
SORT_CHOICES = ("name", "size", "age")


def _use_default_pipe_behaviour() -> None:
    """End quietly when a reader such as ``head`` closes the pipe.

    Python turns a closed pipe into an exception and prints it while shutting
    down. A command line tool should just stop, the way every other one does.
    """
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def default_cache_dir() -> Path:
    """Where the cache is looked for when ``--cache-dir`` is not given.

    ``LLAMA_CACHE`` wins, because that is what llama.cpp itself reads. The
    llama.cpp default location comes next, and the Hugging Face hub cache last.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    from_env = os.environ.get("LLAMA_CACHE")
    if from_env:
        return Path(from_env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        under_hf_home = Path(hf_home) / "llama.cpp"
        if under_hf_home.is_dir():
            return under_hf_home
    llama_default = Path.home() / ".cache" / "llama.cpp"
    if llama_default.is_dir():
        return llama_default
    return Path(HF_HUB_CACHE)


def default_download_dir() -> Path:
    """Where a download goes when ``--cache-dir`` is not given.

    A writer has to agree with the reader, or ``pull`` would fetch into a
    directory ``ls`` does not read and fetch the same file again on the next
    run. So a cache that exists wins, exactly as it does for a listing.

    The choice only matters on its own when there is no cache at all, and then
    it falls to the llama.cpp location. That is the one ``llama-server -hf``
    reads, and creating the hub cache instead would hide the model from the
    program the download was for.
    """
    from_env = os.environ.get("LLAMA_CACHE")
    if from_env:
        return Path(from_env)
    existing = default_cache_dir()
    if existing.is_dir():
        return existing
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "llama.cpp"
    return Path.home() / ".cache" / "llama.cpp"


def fail(message: str, code: int = EXIT_ERROR) -> click.ClickException:
    error = click.ClickException(message)
    error.exit_code = code
    return error


class DefaultCommandGroup(click.Group):
    """A group that reads an unknown first word as a filter for ``ls``."""

    default_command = "ls"

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if args and args[0].startswith("-"):
                raise
            command = self.get_command(ctx, self.default_command)
            assert command is not None
            return self.default_command, command, args


def complete_reference(ctx, param, incomplete):
    """Offer ``org/repo`` and ``org/repo:quant`` while the user types."""
    try:
        cache = cache_model.scan(_cache_dir_from(ctx))
    except (CacheError, OSError):
        return []
    candidates = []
    for repo in cache.repos:
        candidates.append(repo.repo_id)
        quants = {artifact.quant for revision in repo.revisions for artifact in revision.artifacts}
        candidates.extend(f"{repo.repo_id}:{quant}" for quant in sorted(quants))
    return [item for item in candidates if item.lower().startswith(incomplete.lower())]


def complete_repository(ctx, param, incomplete):
    """Offer ``org/repo`` only, for a command that refuses a quant."""
    try:
        cache = cache_model.scan(_cache_dir_from(ctx))
    except (CacheError, OSError):
        return []
    return [repo.repo_id for repo in cache.repos if repo.repo_id.lower().startswith(incomplete.lower())]


def _cache_dir_from(ctx) -> Path:
    root = ctx.find_root()
    given = (root.params or {}).get("cache_dir")
    return Path(given) if given else default_cache_dir()


@click.group(
    cls=DefaultCommandGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-c",
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Cache root. Default: $LLAMA_CACHE, $HF_HOME/llama.cpp, ~/.cache/llama.cpp, or the hub cache.",
)
@click.option(
    "--color",
    type=click.Choice(COLOR_CHOICES),
    default="auto",
    show_default=True,
    help="When to colour the output.",
)
@click.version_option(VERSION, "-V", "--version", prog_name="llama-cache-manager")
@click.pass_context
def main(ctx: click.Context, cache_dir: Path | None, color: str) -> None:
    """Fetch, inspect and delete GGUF models in a llama.cpp or Hugging Face cache.

    Models are named the way huggingface.co and llama.cpp's -hf option name
    them, as org/repo:quant, for example
    unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL.

    Exit codes: 0 done, 1 error, 2 wrong usage, 3 nothing matched,
    4 cancelled at the prompt.
    """
    _use_default_pipe_behaviour()
    ctx.ensure_object(dict)
    ctx.obj["cache_dir"] = cache_dir
    ctx.obj["color"] = color
    if ctx.invoked_subcommand is None:
        ctx.invoke(ls)


def _open_cache(ctx: click.Context):
    given = ctx.obj.get("cache_dir")
    path = Path(given) if given else default_cache_dir()
    if not path.exists():
        raise fail(f"cache directory not found: {path}", EXIT_NO_MATCH)
    if not path.is_dir():
        raise fail(f"not a directory: {path}", EXIT_ERROR)
    try:
        return cache_model.scan(path.resolve())
    except CacheError as error:
        raise fail(str(error), EXIT_ERROR) from error


def _style(ctx: click.Context) -> display.Style:
    return display.pick_style(ctx.obj.get("color", "auto"), sys.stdout)


def _parse_references(values) -> list[refs.Reference]:
    parsed = []
    for value in values:
        try:
            parsed.append(refs.parse(value))
        except ReferenceError as error:
            raise fail(str(error), EXIT_USAGE) from error
    return parsed


@main.command("ls")
@click.argument("filters", nargs=-1, shell_complete=complete_reference)
@click.option(
    "--sort",
    "sort_key",
    type=click.Choice(SORT_CHOICES),
    default="name",
    show_default=True,
    help="Order the repositories.",
)
@click.option("-r", "--reverse", is_flag=True, help="Turn the order around.")
@click.option("--brief", is_flag=True, help="One line per repository.")
@click.option("--json", "as_json", is_flag=True, help="Print the listing as JSON.")
@click.option("--warnings", "show_warnings", is_flag=True, help="Show what the scan skipped.")
@click.pass_context
def ls(
    ctx: click.Context,
    filters,
    sort_key: str,
    reverse: bool,
    brief: bool,
    as_json: bool,
    show_warnings: bool,
) -> None:
    """List cached models, revisions and artifacts.

    A filter is a reference or part of one: an org, a repository, :QUANT,
    or a word to look for. Several filters widen the listing.
    """
    cache = _open_cache(ctx)
    references = _parse_references(filters)
    repos = cache_model.select(cache, references)

    _sort_repos(repos, sort_key, reverse)

    now = time.time()
    if as_json:
        json.dump(display.cache_as_json(cache, repos, now), sys.stdout, indent=2)
        print()
        if not repos:
            # The report is complete and on stdout. The code says it is empty.
            ctx.exit(EXIT_NO_MATCH)
        return

    style = _style(ctx)
    if not repos:
        # Said before the error, because a warning is often the reason there is
        # nothing to list.
        for line in display.warning_lines(cache, show_warnings):
            click.echo(style.paint(style.stray, line), err=True)
        which = " ".join(reference.raw for reference in references)
        message = f"nothing matches {which}" if references else f"the cache is empty: {cache.path}"
        raise fail(message, EXIT_NO_MATCH)

    if brief:
        display.render_brief(repos, style, sys.stdout)
        return
    # The hint belongs to a survey of the whole cache. Beside a filtered total
    # it would count revisions that are not on screen, so it stays away.
    hint = None if references else _garbage_hint(cache)
    display.render_cache(cache, repos, now, style, hint, show_warnings, sys.stdout)


# Size and age read best with the largest and the newest first, names with the
# first letter first. --reverse turns whichever of those applies around.
_SORT_KEYS = {
    "name": (lambda repo: repo.repo_id.lower(), False),
    "size": (lambda repo: repo.size, True),
    "age": (lambda repo: repo.modified, True),
}


def _sort_repos(repos, sort_key: str, reverse: bool) -> None:
    key, descending = _SORT_KEYS[sort_key]
    repos.sort(key=key, reverse=descending != reverse)


def _garbage_hint(cache):
    """The plan behind the prune hint, or nothing when it cannot be built.

    The hint is an aside. A cache that no plan can describe still has a listing
    worth printing.
    """
    try:
        return removal.build(cache, removal.garbage(cache))
    except RemovalError:
        return None


@main.command("rm")
@click.argument("references", nargs=-1, required=True, shell_complete=complete_reference)
@click.option(
    "--until",
    "cutoff_spec",
    metavar="SPEC",
    help="Keep to the revisions older than SPEC, such as '30d' or '2026-07-01'.",
)
@click.option("-n", "--dry-run", is_flag=True, help="Show the plan and stop.")
@click.option("-y", "--yes", is_flag=True, help="Do not ask.")
@click.option("--json", "as_json", is_flag=True, help="Print the plan as JSON.")
@click.pass_context
def remove(
    ctx: click.Context,
    references,
    cutoff_spec: str | None,
    dry_run: bool,
    yes: bool,
    as_json: bool,
) -> None:
    """Remove a repository or one quant from it.

    REFERENCES are org/repo or org/repo:quant. A shorter word is accepted while
    it fits one repository only. With --until only the revisions older than the
    cutoff go, which is prune --until held to the repositories named here.
    """
    cache = _open_cache(ctx)
    cutoff = _cutoff(cutoff_spec)
    items: list[removal.Item] = []
    for reference in _parse_references(references):
        try:
            target = cache_model.resolve(cache, reference)
        except ResolveError as error:
            raise fail(str(error), EXIT_NO_MATCH) from error
        if isinstance(target, RepoTarget):
            if cutoff is None:
                items.extend(removal.for_repository(target.repo))
            else:
                items.extend(removal.older_than_in(target.repo, cutoff))
        elif isinstance(target, ArtifactTarget):
            matches = target.matches
            if cutoff is not None:
                matches = tuple(match for match in matches if match.artifact.modified < cutoff)
            items.extend(removal.for_artifacts(matches))
    nothing = "Nothing to remove." if cutoff is None else "Nothing older than the cutoff."
    _carry_out(ctx, cache, items, dry_run, yes, as_json, nothing)


@main.command("prune")
@click.option(
    "--until",
    "cutoff_spec",
    metavar="SPEC",
    help="Also remove revisions older than SPEC, such as '30d', '2 weeks' or '2026-07-01'.",
)
@click.option("-n", "--dry-run", is_flag=True, help="Show the plan and stop.")
@click.option("-y", "--yes", is_flag=True, help="Do not ask.")
@click.option("--json", "as_json", is_flag=True, help="Print the plan as JSON.")
@click.pass_context
def prune(ctx: click.Context, cutoff_spec: str | None, dry_run: bool, yes: bool, as_json: bool) -> None:
    """Remove what nothing can reach any more.

    That is a revision no name points at, a blob no revision points at, and an
    interrupted download. With --until it also removes revisions that are older
    than the cutoff, and those may still be in use.
    """
    cache = _open_cache(ctx)
    items = removal.garbage(cache)

    cutoff = _cutoff(cutoff_spec)
    if cutoff is not None:
        covered = {item.revision.path for item in items if item.revision is not None}
        items.extend(removal.older_than(cache, cutoff, covered))

    _carry_out(ctx, cache, items, dry_run, yes, as_json, "Nothing to prune.")


def _cutoff(spec: str | None) -> float | None:
    """Read an --until value, or nothing when it was not given."""
    if spec is None:
        return None
    try:
        return age.cutoff(spec, datetime.now())
    except CutoffError as error:
        raise fail(str(error), EXIT_USAGE) from error


def _carry_out(ctx, cache, items, dry_run: bool, yes: bool, as_json: bool, nothing_message: str) -> None:
    try:
        plan = removal.build(cache, items)
    except RemovalError as error:
        raise fail(str(error), EXIT_ERROR) from error

    if as_json:
        # JSON output means nobody is watching, so there is nobody to ask.
        # Removing has to be stated on the command line, and the answer comes
        # before the document, which would otherwise claim a removal that the
        # run then refuses to make.
        if not dry_run and not plan.empty and not yes:
            raise fail("--json removes nothing on its own: add -n to preview or -y to remove", EXIT_USAGE)
        json.dump(display.plan_as_json(plan, dry_run), sys.stdout, indent=2)
        print()
        if not dry_run and not plan.empty:
            _execute(plan)
        return

    style = _style(ctx)
    if plan.empty:
        print(style.paint(style.dim, nothing_message))
        return

    display.render_plan(plan, time.time(), style, dry_run, sys.stdout)
    if dry_run:
        return

    print()
    if not yes and not click.confirm("Remove these?", default=False):
        raise fail("cancelled", EXIT_CANCELLED)

    _execute(plan)
    print(style.paint(style.total, f"Removed {display.human_size(plan.freed)}."))


def _execute(plan) -> None:
    try:
        removal.execute(plan)
    except RemovalError as error:
        raise fail(str(error), EXIT_ERROR) from error
    except OSError as error:
        raise fail(f"could not finish the removal: {error}", EXIT_ERROR) from error


@main.command("pull")
@click.argument("references", nargs=-1, required=True)
@click.option(
    "--revision",
    default="main",
    show_default=True,
    metavar="REV",
    help="Branch, tag or commit to fetch.",
)
@click.option("-n", "--dry-run", is_flag=True, help="Show the plan and stop.")
@click.option("-y", "--yes", is_flag=True, help="Do not ask.")
@click.option("--json", "as_json", is_flag=True, help="Print the plan as JSON.")
@click.pass_context
def pull(ctx: click.Context, references, revision: str, dry_run: bool, yes: bool, as_json: bool) -> None:
    """Fetch one quant of a repository from the hub into the cache.

    REFERENCES are org/repo:quant. The quant has to be named, because the hub
    cannot be searched from here and the quants of one repository differ by tens
    of gigabytes. A reference without one comes back with the choices named.

    A file the cache already holds is not fetched again, so pulling a second
    quant of a model transfers only what is new.
    """
    target = _target_dir(ctx)
    hub = _open_hub(ctx)
    try:
        plan = download.plan(hub, target, _parse_references(references), revision)
    except SelectionError as error:
        raise fail(str(error), EXIT_NO_MATCH) from error
    except DownloadError as error:
        raise fail(str(error), EXIT_ERROR) from error

    if as_json:
        # The same rule the removal commands follow: a document nobody watches
        # cannot be answered, so the decision belongs on the command line.
        if not dry_run and not yes:
            raise fail("--json downloads nothing on its own: add -n to preview or -y to fetch", EXIT_USAGE)
        json.dump(display.download_plan_as_json(plan, dry_run), sys.stdout, indent=2)
        print()
        if not dry_run:
            _fetch(plan, hub)
        return

    style = _style(ctx)
    display.render_download_plan(plan, style, dry_run, sys.stdout)
    if dry_run:
        return

    # Nothing to transfer leaves nothing to weigh up. What such a run still
    # does is add the links a snapshot is missing, which costs no bandwidth.
    if plan.transfer and not yes:
        print()
        if not click.confirm("Download these?", default=False):
            raise fail("cancelled", EXIT_CANCELLED)

    _fetch(plan, hub)
    # A run that transferred nothing needs no closing line: the summary above
    # already said that everything was there.
    if plan.transfer:
        print()
        print(style.paint(style.total, f"Fetched {display.human_size(plan.transfer)}."))


@main.command("update")
@click.argument("references", nargs=-1, shell_complete=complete_repository)
@click.option("--keep", is_flag=True, help="Keep the revisions an update replaces.")
@click.option("-n", "--dry-run", is_flag=True, help="Show the plan and stop.")
@click.option("-y", "--yes", is_flag=True, help="Do not ask.")
@click.option("--json", "as_json", is_flag=True, help="Print the plan as JSON.")
@click.pass_context
def update(ctx: click.Context, references, keep: bool, dry_run: bool, yes: bool, as_json: bool) -> None:
    """Fetch a newer revision of what the cache already holds.

    With no REFERENCES every repository the cache follows is checked, otherwise
    only the ones named. One request per name says whether there is anything to
    do, so a check across the whole cache costs little.

    What the cache holds decides what is fetched: the same quants, at the newer
    commit. The revision each name pointed at before is then removed, because
    nothing can reach it any more. With --keep it stays, and prune offers it
    later.
    """
    cache = _open_cache(ctx)
    repos = _repos_to_update(cache, _parse_references(references))
    hub = _open_hub(ctx)

    try:
        found = update_model.check(hub, update_model.tracked(cache, repos))
        plan = update_model.plan(hub, cache.path, found)
    except DownloadError as error:
        raise fail(str(error), EXIT_ERROR) from error

    style = _style(ctx)
    if as_json:
        if not dry_run and not yes and not plan.empty:
            raise fail("--json updates nothing on its own: add -n to preview or -y to update", EXIT_USAGE)
        json.dump(display.update_plan_as_json(plan, dry_run, keep), sys.stdout, indent=2)
        print()
        if not dry_run and not plan.empty:
            _carry_out_update(ctx, plan, hub, keep, style, quiet=True)
        return

    for line in display.skipped_lines(plan.skipped):
        print(style.paint(style.stray, line))
    checked = display.count(len(repos), "repository", "repositories")
    if plan.empty:
        nothing = "Nothing can be updated." if plan.skipped else "Everything is up to date."
        print(style.paint(style.dim, f"{nothing} {checked} checked."))
        return

    # Said before the list, so that a single line of output is not mistaken for
    # a check that never ran. Counted in repositories, the same unit as the
    # number it sits beside.
    print(style.paint(style.dim, f"{checked} checked, {len(plan.repo_ids)} to update."))
    print()
    display.render_outdated(plan.replaced, style, sys.stdout)
    print()
    display.render_download_plan(plan.downloads, style, dry_run, sys.stdout)
    if not keep:
        print(
            style.paint(
                style.dim,
                f"Then removes {display.count(plan.replaced_revisions, 'superseded revision')}.",
            )
        )
    if dry_run:
        return

    # Asked even when nothing has to be transferred, because an update removes
    # as well, and a file dropped upstream leaves the old revision holding a
    # blob that nothing else does.
    if not yes:
        print()
        if not click.confirm("Update these?", default=False):
            raise fail("cancelled", EXIT_CANCELLED)

    _carry_out_update(ctx, plan, hub, keep, style, quiet=False)


def _repos_to_update(cache, references):
    """The repositories an update should look at.

    A quant is refused rather than obeyed. Fetching one quant of a repository
    moves the name to the new revision, which leaves every other quant behind
    in a revision the update would then remove.
    """
    named = [reference for reference in references if reference.quant is not None]
    if named:
        which = ", ".join(reference.raw for reference in named)
        raise fail(
            f"update works on whole repositories, so it cannot take a quant: {which}. "
            "Write org/repo to update everything the cache holds of it.",
            EXIT_USAGE,
        )
    if not references:
        return list(cache.repos)

    chosen: dict[str, object] = {}
    for reference in references:
        found = cache_model.repos_matching(cache, reference)
        if not found:
            raise fail(f"nothing in the cache matches {reference.raw!r}", EXIT_NO_MATCH)
        for repo in found:
            chosen[repo.repo_id] = repo
    return list(chosen.values())


def _carry_out_update(ctx, plan, hub, keep: bool, style, quiet: bool) -> None:
    """Fetch the newer revisions, then drop the ones they replaced."""
    _fetch(plan.downloads, hub)
    if not quiet and plan.downloads.transfer:
        print()
        print(style.paint(style.total, f"Fetched {display.human_size(plan.downloads.transfer)}."))
    if keep:
        if not quiet:
            print(style.paint(style.dim, "The replaced revisions are kept, and prune will offer them."))
        return

    # Scanned again on purpose. What the removal gives back depends on the files
    # that have just arrived: a blob the new revision points at as well stays.
    fresh = _open_cache(ctx)
    try:
        removal_plan = removal.build(fresh, update_model.superseded(fresh, list(plan.replaced)))
    except RemovalError as error:
        raise fail(str(error), EXIT_ERROR) from error
    if removal_plan.empty:
        return

    if not quiet:
        print()
        display.render_plan(
            removal_plan,
            time.time(),
            style,
            False,
            sys.stdout,
            heading="Removing what the update replaced",
        )
    _execute(removal_plan)
    if not quiet and removal_plan.freed:
        print(style.paint(style.total, f"Removed {display.human_size(removal_plan.freed)}."))


def _target_dir(ctx: click.Context) -> Path:
    """Where a download goes, which need not exist yet."""
    given = ctx.obj.get("cache_dir")
    path = Path(given) if given else default_download_dir()
    if path.exists() and not path.is_dir():
        raise fail(f"not a directory: {path}", EXIT_ERROR)
    return path


def _open_hub(ctx: click.Context):
    """The hub to read, with the one a caller put in the context winning.

    The tests put a fake there, which is what keeps them off the network.
    """
    existing = ctx.obj.get("hub")
    if existing is not None:
        return existing
    from .hub import HubApi

    return HubApi()


def _fetch(plan, hub) -> None:
    try:
        download.execute(plan, hub)
    except DownloadError as error:
        raise fail(str(error), EXIT_ERROR) from error


@main.command("help")
@click.argument("topic", required=False)
@click.pass_context
def help_command(ctx: click.Context, topic: str | None) -> None:
    """Show the help of the program, or of one command."""
    group = ctx.parent.command if ctx.parent else main
    if topic is None:
        click.echo(ctx.parent.get_help() if ctx.parent else main.get_help(ctx))
        return
    command = group.get_command(ctx, topic)
    if command is None:
        raise fail(f"no command named {topic!r}", EXIT_USAGE)
    with click.Context(command, info_name=topic, parent=ctx.parent) as sub:
        click.echo(command.get_help(sub))


@main.command("completions")
@click.argument("shell", type=click.Choice(("bash", "zsh", "fish")))
def completions(shell: str) -> None:
    """Print the completion script for SHELL on stdout.

    The script is generated here, so it does not depend on any file being
    installed next to the program.
    """
    from click.shell_completion import get_completion_class

    completion_class = get_completion_class(shell)
    if completion_class is None:
        raise fail(f"no completion support for {shell}", EXIT_USAGE)
    program = "llama-cache-manager"
    variable = "_LLAMA_CACHE_MANAGER_COMPLETE"
    print(completion_class(main, {}, program, variable).source())


# The names the shell version used. Without them the fallback below would read
# "remove" as a filter and list instead of removing.
main.add_command(ls, "list")
main.add_command(remove, "remove")


if __name__ == "__main__":
    main()
