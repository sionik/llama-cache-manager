"""The command line, driven the way a user drives it."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from conftest import COMMIT_A, COMMIT_B

from llama_cache_manager.cli import (
    EXIT_CANCELLED,
    EXIT_NO_MATCH,
    EXIT_OK,
    EXIT_USAGE,
    main,
)

REPO = "unsloth/Foo-30B-GGUF"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def filled_cache(fake_cache):
    fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000, "mmproj-BF16.gguf": 700})
    fake_cache.revision("other/Bar-GGUF", COMMIT_B, {"Bar-Q8_0.gguf": 900}, ref=None)
    fake_cache.blob(REPO, "stray", 400)
    return fake_cache


def run(runner, cache, *args):
    return runner.invoke(main, ["-c", str(cache.root), "--color", "never", *args])


class TestListing:
    def test_lists_without_a_command(self, runner, filled_cache):
        result = run(runner, filled_cache)
        assert result.exit_code == EXIT_OK
        assert f"{REPO}" in result.output
        assert ":Q4_K_M" in result.output

    def test_reads_a_bare_word_as_a_filter(self, runner, filled_cache):
        result = run(runner, filled_cache, "unsloth")
        assert result.exit_code == EXIT_OK
        assert REPO in result.output
        assert "other/Bar-GGUF" not in result.output

    def test_filters_by_quant(self, runner, filled_cache):
        result = run(runner, filled_cache, "ls", ":Q8_0")
        assert result.exit_code == EXIT_OK
        assert "other/Bar-GGUF" in result.output
        assert REPO not in result.output

    def test_reports_that_nothing_matched(self, runner, filled_cache):
        result = run(runner, filled_cache, "ls", "nothing-like-this")
        assert result.exit_code == EXIT_NO_MATCH

    def test_names_a_missing_cache_directory(self, runner, tmp_path):
        result = runner.invoke(main, ["-c", str(tmp_path / "nowhere"), "ls"])
        assert result.exit_code == EXIT_NO_MATCH
        assert "cache directory not found" in result.output

    def test_offers_the_reference_of_every_artifact_as_json(self, runner, filled_cache):
        result = run(runner, filled_cache, "ls", "--json")
        assert result.exit_code == EXIT_OK
        report = json.loads(result.output)
        references = [
            artifact["reference"]
            for repo in report["repositories"]
            for revision in repo["revisions"]
            for artifact in revision["artifacts"]
        ]
        assert f"{REPO}:Q4_K_M" in references
        assert report["repositories"][0]["size"] > 0

    def test_hints_at_prune_when_the_cache_holds_garbage(self, runner, filled_cache):
        result = run(runner, filled_cache, "ls")
        assert "prune would remove" in result.output


class TestRemove:
    def test_shows_a_plan_and_keeps_the_files_on_a_dry_run(self, runner, filled_cache):
        result = run(runner, filled_cache, "rm", "-n", f"{REPO}:Q4_K_M")
        assert result.exit_code == EXIT_OK
        assert "Would remove" in result.output
        assert "Reclaims 4.9 KiB" in result.output
        assert (filled_cache.repo_path(REPO) / "snapshots" / COMMIT_A / "Foo-30B-Q4_K_M.gguf").exists()

    def test_keeps_the_files_when_the_prompt_is_declined(self, runner, filled_cache):
        result = runner.invoke(
            main,
            ["-c", str(filled_cache.root), "--color", "never", "rm", f"{REPO}:Q4_K_M"],
            input="n\n",
        )
        assert result.exit_code == EXIT_CANCELLED
        assert (filled_cache.repo_path(REPO) / "snapshots" / COMMIT_A / "Foo-30B-Q4_K_M.gguf").exists()

    def test_removes_one_quant_and_leaves_the_rest(self, runner, filled_cache):
        result = run(runner, filled_cache, "rm", "-y", f"{REPO}:Q4_K_M")
        assert result.exit_code == EXIT_OK
        snapshot = filled_cache.repo_path(REPO) / "snapshots" / COMMIT_A
        assert not (snapshot / "Foo-30B-Q4_K_M.gguf").exists()
        assert (snapshot / "mmproj-BF16.gguf").exists()

    def test_refuses_a_quant_without_a_repository(self, runner, filled_cache):
        result = run(runner, filled_cache, "rm", "-y", ":Q4_K_M")
        assert result.exit_code == EXIT_NO_MATCH
        assert "names a quant but no repository" in result.output

    def test_names_the_candidates_when_a_word_fits_several_repositories(self, runner, fake_cache):
        fake_cache.revision("unsloth/Foo-A-GGUF", COMMIT_A, {"Foo-A-Q4_K_M.gguf": 10})
        fake_cache.revision("unsloth/Foo-B-GGUF", COMMIT_B, {"Foo-B-Q4_K_M.gguf": 10})
        result = run(runner, fake_cache, "rm", "-y", "Foo")
        assert result.exit_code == EXIT_NO_MATCH
        assert "unsloth/Foo-A-GGUF" in result.output
        assert "unsloth/Foo-B-GGUF" in result.output

    def test_lists_the_quants_it_does_have(self, runner, filled_cache):
        result = run(runner, filled_cache, "rm", "-y", f"{REPO}:Q2_K")
        assert result.exit_code == EXIT_NO_MATCH
        assert "Q4_K_M" in result.output

    def test_rejects_a_reference_the_grammar_cannot_read(self, runner, filled_cache):
        result = run(runner, filled_cache, "rm", "-y", "a:b:c")
        assert result.exit_code == EXIT_USAGE

    def test_treats_a_missing_reference_as_wrong_usage(self, runner, filled_cache):
        result = run(runner, filled_cache, "rm")
        assert result.exit_code == EXIT_USAGE


class TestPrune:
    def test_removes_only_what_nothing_can_reach(self, runner, filled_cache):
        result = run(runner, filled_cache, "prune", "-y")
        assert result.exit_code == EXIT_OK
        assert not (filled_cache.repo_path(REPO) / "blobs" / "stray").exists()
        assert not filled_cache.repo_path("other/Bar-GGUF").exists()
        assert (filled_cache.repo_path(REPO) / "snapshots" / COMMIT_A).is_dir()

    def test_says_so_when_there_is_no_garbage(self, runner, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 10})
        result = run(runner, fake_cache, "prune", "-y")
        assert result.exit_code == EXIT_OK
        assert "Nothing to prune" in result.output

    def test_rejects_a_cutoff_it_cannot_read(self, runner, filled_cache):
        result = run(runner, filled_cache, "prune", "-y", "--until", "soon")
        assert result.exit_code == EXIT_USAGE
        assert "cannot read cutoff" in result.output

    def test_reaches_an_old_revision_that_a_name_still_points_at(self, runner, filled_cache):
        import os

        blob = filled_cache.repo_path(REPO) / "blobs" / f"{COMMIT_A[:6]}-Foo-30B-Q4_K_M.gguf"
        for path in (filled_cache.repo_path(REPO) / "blobs").iterdir():
            os.utime(path, (1000, 1000))
        assert blob.exists()

        result = run(runner, filled_cache, "prune", "-y", "--until", "1d")

        assert result.exit_code == EXIT_OK
        assert not filled_cache.repo_path(REPO).exists()

    def test_reports_the_plan_as_json_without_touching_anything(self, runner, filled_cache):
        result = run(runner, filled_cache, "prune", "-n", "--json")
        assert result.exit_code == EXIT_OK
        report = json.loads(result.output)
        assert report["dry_run"] is True
        assert report["reclaims"] > 0
        assert {item["reason"] for item in report["items"]} == {"detached revision", "unreferenced blob"}
        assert (filled_cache.repo_path(REPO) / "blobs" / "stray").exists()


class TestCompletions:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_prints_a_script_for_each_shell(self, runner, shell):
        result = runner.invoke(main, ["completions", shell])
        assert result.exit_code == EXIT_OK
        assert "_LLAMA_CACHE_MANAGER_COMPLETE" in result.output

    def test_rejects_a_shell_it_does_not_know(self, runner):
        result = runner.invoke(main, ["completions", "csh"])
        assert result.exit_code == EXIT_USAGE


class TestVersion:
    def test_prints_the_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == EXIT_OK
        assert "llama-cache-manager" in result.output


class TestPackaging:
    def test_the_version_in_the_code_matches_the_project_metadata(self):
        import re
        from pathlib import Path

        from llama_cache_manager.cli import VERSION

        pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
        assert declared is not None
        assert declared.group(1) == VERSION


class TestJsonIsNotInteractive:
    def test_refuses_to_remove_from_a_json_run_without_yes(self, runner, filled_cache):
        result = run(runner, filled_cache, "prune", "--json")
        assert result.exit_code == EXIT_USAGE
        assert "-y to remove" in result.output
        assert (filled_cache.repo_path(REPO) / "blobs" / "stray").exists()

    def test_removes_from_a_json_run_when_yes_is_given(self, runner, filled_cache):
        result = run(runner, filled_cache, "prune", "--json", "-y")
        assert result.exit_code == EXIT_OK
        assert not (filled_cache.repo_path(REPO) / "blobs" / "stray").exists()


class TestRetiredNames:
    def test_remove_still_removes(self, runner, filled_cache):
        result = run(runner, filled_cache, "remove", "-y", f"{REPO}:Q4_K_M")
        assert result.exit_code == EXIT_OK
        assert not (filled_cache.repo_path(REPO) / "snapshots" / COMMIT_A / "Foo-30B-Q4_K_M.gguf").exists()

    def test_list_still_lists(self, runner, filled_cache):
        result = run(runner, filled_cache, "list")
        assert result.exit_code == EXIT_OK
        assert REPO in result.output

    def test_help_names_a_command(self, runner, filled_cache):
        result = run(runner, filled_cache, "help", "rm")
        assert result.exit_code == EXIT_OK
        assert "Remove a repository or one quant" in result.output

    def test_help_on_its_own_shows_the_commands(self, runner, filled_cache):
        result = run(runner, filled_cache, "help")
        assert result.exit_code == EXIT_OK
        assert "prune" in result.output

    def test_help_rejects_a_command_it_does_not_have(self, runner, filled_cache):
        result = run(runner, filled_cache, "help", "frobnicate")
        assert result.exit_code == EXIT_USAGE


class TestNoMatchIsAlwaysReported:
    def test_reports_no_match_in_the_exit_code_of_a_json_run(self, runner, filled_cache):
        result = run(runner, filled_cache, "ls", "--json", "nothing-like-this")
        assert result.exit_code == EXIT_NO_MATCH
        assert json.loads(result.output)["repositories"] == []


class TestListingSurvivesAnUnusablePlan:
    def test_lists_when_a_stray_points_out_of_the_cache(self, runner, filled_cache, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "blob").write_bytes(b"\0" * 10)
        (filled_cache.repo_path(REPO) / "blobs" / "wandering").symlink_to(outside / "blob")

        result = run(runner, filled_cache, "ls")

        assert result.exit_code == EXIT_OK
        assert REPO in result.output
        assert "Traceback" not in result.output


class TestDefaultCacheDir:
    def test_prefers_llama_cache(self, monkeypatch, tmp_path):
        from llama_cache_manager.cli import default_cache_dir

        monkeypatch.setenv("LLAMA_CACHE", str(tmp_path / "from-llama-cache"))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        assert default_cache_dir() == tmp_path / "from-llama-cache"

    def test_looks_under_hf_home_as_the_shell_version_did(self, monkeypatch, tmp_path):
        from llama_cache_manager.cli import default_cache_dir

        under_hf_home = tmp_path / "hf" / "llama.cpp"
        under_hf_home.mkdir(parents=True)
        monkeypatch.delenv("LLAMA_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
        assert default_cache_dir() == under_hf_home


class TestOrder:
    @pytest.fixture
    def three_repos(self, fake_cache):
        fake_cache.revision("org/Big-GGUF", COMMIT_A, {"Big-Q4_K_M.gguf": 3000})
        fake_cache.revision("org/Mid-GGUF", COMMIT_B, {"Mid-Q4_K_M.gguf": 2000})
        fake_cache.revision("org/Small-GGUF", "c" * 40, {"Small-Q4_K_M.gguf": 1000})
        return fake_cache

    def _names(self, output):
        return [line.split()[0] for line in output.splitlines() if line.startswith("org/")]

    def test_sorts_by_name_first_letter_first(self, runner, three_repos):
        result = run(runner, three_repos, "ls", "--brief")
        assert self._names(result.output) == ["org/Big-GGUF", "org/Mid-GGUF", "org/Small-GGUF"]

    def test_sorts_by_size_largest_first(self, runner, three_repos):
        result = run(runner, three_repos, "ls", "--brief", "--sort", "size")
        assert self._names(result.output) == ["org/Big-GGUF", "org/Mid-GGUF", "org/Small-GGUF"]

    def test_turns_the_size_order_around(self, runner, three_repos):
        result = run(runner, three_repos, "ls", "--brief", "--sort", "size", "-r")
        assert self._names(result.output) == ["org/Small-GGUF", "org/Mid-GGUF", "org/Big-GGUF"]

    def test_turns_the_name_order_around(self, runner, three_repos):
        result = run(runner, three_repos, "ls", "--brief", "-r")
        assert self._names(result.output) == ["org/Small-GGUF", "org/Mid-GGUF", "org/Big-GGUF"]

    def test_brief_leaves_out_revisions_and_artifacts(self, runner, three_repos):
        result = run(runner, three_repos, "ls", "--brief")
        assert ":Q4_K_M" not in result.output
        assert COMMIT_A[:7] not in result.output


class TestRemoveUntil:
    @pytest.fixture
    def old_and_new(self, fake_cache):
        import os

        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        fake_cache.revision(REPO, COMMIT_B, {"Foo-30B-Q6_K.gguf": 200}, ref="v2")
        old = fake_cache.repo_path(REPO) / "blobs" / f"{COMMIT_A[:6]}-Foo-30B-Q4_K_M.gguf"
        os.utime(old, (1000, 1000))
        return fake_cache

    def test_keeps_to_the_revisions_older_than_the_cutoff(self, runner, old_and_new):
        result = run(runner, old_and_new, "rm", "-y", "--until", "1d", REPO)

        assert result.exit_code == EXIT_OK
        assert not (old_and_new.repo_path(REPO) / "snapshots" / COMMIT_A).exists()
        assert (old_and_new.repo_path(REPO) / "snapshots" / COMMIT_B).is_dir()

    def test_says_so_when_nothing_is_that_old(self, runner, old_and_new):
        result = run(runner, old_and_new, "rm", "-y", "--until", "100y", REPO)

        assert result.exit_code == EXIT_OK
        assert "Nothing older than the cutoff" in result.output
        assert (old_and_new.repo_path(REPO) / "snapshots" / COMMIT_A).exists()

    def test_holds_a_quant_to_the_cutoff_as_well(self, runner, old_and_new):
        result = run(runner, old_and_new, "rm", "-y", "--until", "1d", f"{REPO}:Q6_K")

        assert result.exit_code == EXIT_OK
        assert "Nothing older than the cutoff" in result.output

    def test_rejects_a_cutoff_it_cannot_read(self, runner, old_and_new):
        result = run(runner, old_and_new, "rm", "-y", "--until", "soon", REPO)
        assert result.exit_code == EXIT_USAGE


class TestManPage:
    """The man page is the documentation a package installs, so it has to keep up."""

    @pytest.fixture
    def page(self):
        from pathlib import Path

        return (Path(__file__).parent.parent / "man" / "llama-cache-manager.1").read_text()

    def test_names_every_command(self, page):
        from llama_cache_manager.cli import main

        missing = [name for name in main.commands if name not in page]
        assert missing == []

    def test_names_every_long_option(self, page):
        import click

        from llama_cache_manager.cli import main

        options = set()
        for command in [main, *main.commands.values()]:
            for parameter in command.params:
                if not isinstance(parameter, click.Option):
                    continue
                options.update(name for name in parameter.opts if name.startswith("--") and name != "--help")
        # roff needs the hyphens escaped, so compare against the escaped form.
        missing = [name for name in sorted(options) if name.replace("-", "\\-") not in page]
        assert missing == []

    def test_carries_the_current_version(self, page):
        from llama_cache_manager.cli import VERSION

        assert VERSION in page


class TestCutoffErrorsAreUsageErrors:
    @pytest.mark.parametrize("spec", ["100000y", "999999999999d", "0001-01-01"])
    def test_reports_a_cutoff_beyond_the_calendar_as_wrong_usage(self, runner, filled_cache, spec):
        result = run(runner, filled_cache, "prune", "-n", "--until", spec)

        assert result.exit_code == EXIT_USAGE
        assert "beyond the calendar" in result.output
        assert "Traceback" not in result.output

    def test_the_same_for_rm(self, runner, filled_cache):
        result = run(runner, filled_cache, "rm", "-n", "--until", "100000y", REPO)

        assert result.exit_code == EXIT_USAGE
        assert "Traceback" not in result.output


class TestAWordMeansTheSameThingTwice:
    @pytest.fixture
    def two_quants(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000, "Foo-30B-Q6_K.gguf": 6000})
        return fake_cache

    def test_removes_only_what_the_same_word_listed(self, runner, two_quants):
        result = run(runner, two_quants, "rm", "-y", "Q4_K_M")

        assert result.exit_code == EXIT_OK
        snapshot = two_quants.repo_path(REPO) / "snapshots" / COMMIT_A
        assert not (snapshot / "Foo-30B-Q4_K_M.gguf").exists()
        assert (snapshot / "Foo-30B-Q6_K.gguf").exists()


class TestWarningsSurviveAnEmptyListing:
    def test_says_what_it_skipped_before_saying_the_cache_is_empty(self, runner, fake_cache):
        broken = fake_cache.repo_path("org/Broken-GGUF") / "refs"
        broken.mkdir(parents=True)
        (broken / "main").write_text("a" * 40)

        result = run(runner, fake_cache, "ls")

        assert result.exit_code == EXIT_NO_MATCH
        assert "1 directory skipped" in result.output
        assert "the cache is empty" in result.output

    def test_names_the_reason_with_warnings(self, runner, fake_cache):
        broken = fake_cache.repo_path("org/Broken-GGUF") / "refs"
        broken.mkdir(parents=True)
        (broken / "main").write_text("a" * 40)

        result = run(runner, fake_cache, "ls", "--warnings")

        assert "Snapshots dir doesn't exist" in result.output


class TestJsonSaysNothingItWillNotDo:
    def test_refuses_before_printing_a_plan_it_will_not_carry_out(self, runner, filled_cache):
        result = run(runner, filled_cache, "prune", "--json")

        assert result.exit_code == EXIT_USAGE
        assert "dry_run" not in result.output
        assert (filled_cache.repo_path(REPO) / "blobs" / "stray").exists()


class TestUntilOnAQuant:
    def test_judges_the_artifact_by_its_own_age(self, runner, fake_cache):
        import os

        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000, "Foo-30B-Q6_K.gguf": 6000})
        old = fake_cache.repo_path(REPO) / "blobs" / f"{COMMIT_A[:6]}-Foo-30B-Q4_K_M.gguf"
        os.utime(old, (1000, 1000))

        # The revision is fresh, because the other artifact in it is fresh.
        aged = run(runner, fake_cache, "rm", "-y", "--until", "1d", f"{REPO}:Q4_K_M")
        assert aged.exit_code == EXIT_OK
        assert not (fake_cache.repo_path(REPO) / "snapshots" / COMMIT_A / "Foo-30B-Q4_K_M.gguf").exists()

        fresh = run(runner, fake_cache, "rm", "-y", "--until", "1d", f"{REPO}:Q6_K")
        assert "Nothing older than the cutoff" in fresh.output


class TestHintFollowsTheListing:
    def test_leaves_the_hint_out_of_a_filtered_listing(self, runner, filled_cache):
        unfiltered = run(runner, filled_cache, "ls")
        filtered = run(runner, filled_cache, "ls", REPO)

        assert "prune would remove" in unfiltered.output
        assert "prune would remove" not in filtered.output


class TestModuleRoute:
    """Running the package as a module must present the tool's own name."""

    def test_names_itself_as_the_command_not_as_a_module(self):
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "llama_cache_manager", "--help"],
            capture_output=True,
            text=True,
            cwd=repo,
            check=True,
        )

        assert result.stdout.startswith("Usage: llama-cache-manager")
