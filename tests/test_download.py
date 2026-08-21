"""Fetching an artifact from the hub, with the hub itself faked out."""

from __future__ import annotations

import pytest
from conftest import COMMIT_A, COMMIT_B, FakeHub

from llama_cache_manager import download, refs
from llama_cache_manager.cache import Kind
from llama_cache_manager.download import DownloadError, SelectionError

REPO = "unsloth/Foo-30B-GGUF"

SPLIT = {
    "Foo-30B-UD-Q4_K_XL-00001-of-00002.gguf": 300,
    "Foo-30B-UD-Q4_K_XL-00002-of-00002.gguf": 200,
    "mmproj-Foo-30B-BF16.gguf": 70,
    "README.md": 5,
}


@pytest.fixture
def hub():
    return FakeHub({REPO: {"commit": COMMIT_A, "files": dict(SPLIT)}})


def plan_for(hub, cache_dir, *raw, revision="main"):
    return download.plan(hub, cache_dir, [refs.parse(text) for text in raw], revision)


class TestPlanning:
    def test_puts_every_shard_of_one_artifact_in_one_download(self, hub, tmp_path):
        plan = plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL")

        assert len(plan.downloads) == 1
        assert plan.downloads[0].reference == f"{REPO}:UD-Q4_K_XL"
        assert plan.downloads[0].shards == 2
        assert plan.downloads[0].size == 500

    def test_reads_the_kind_of_the_artifact(self, hub, tmp_path):
        plan = plan_for(hub, tmp_path, f"{REPO}:mmproj-BF16")

        assert plan.downloads[0].kind is Kind.PROJECTOR

    def test_counts_only_the_files_that_are_not_cached_yet_as_transfer(self, tmp_path):
        hub = FakeHub(
            {REPO: {"commit": COMMIT_A, "files": dict(SPLIT)}},
            cached=["Foo-30B-UD-Q4_K_XL-00001-of-00002.gguf"],
        )

        plan = plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL")

        assert plan.size == 500
        assert plan.transfer == 200

    def test_calls_an_artifact_cached_once_every_shard_is_there(self, tmp_path):
        hub = FakeHub({REPO: {"commit": COMMIT_A, "files": dict(SPLIT)}}, cached=list(SPLIT))

        plan = plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL")

        assert plan.downloads[0].cached
        assert plan.transfer == 0

    def test_reports_the_commit_the_files_come_from(self, hub, tmp_path):
        plan = plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL")

        assert plan.downloads[0].commit == COMMIT_A

    def test_matches_a_quant_whatever_its_case(self, hub, tmp_path):
        plan = plan_for(hub, tmp_path, f"{REPO}:ud-q4_k_xl")

        assert plan.downloads[0].quant == "UD-Q4_K_XL"

    def test_takes_several_references_at_once(self, tmp_path):
        hub = FakeHub(
            {
                REPO: {"commit": COMMIT_A, "files": dict(SPLIT)},
                "other/Bar-GGUF": {"commit": COMMIT_B, "files": {"Bar-Q8_0.gguf": 90}},
            }
        )

        plan = plan_for(hub, tmp_path, f"{REPO}:mmproj-BF16", "other/Bar-GGUF:Q8_0")

        assert [item.reference for item in plan.downloads] == [
            f"{REPO}:mmproj-BF16",
            "other/Bar-GGUF:Q8_0",
        ]

    def test_names_one_artifact_once_however_often_it_is_asked_for(self, hub, tmp_path):
        # The quant match ignores case, so a repeat need not be spelled the same.
        # Counting it twice would double both totals of a plan the user reads
        # before agreeing to the transfer.
        plan = plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL", f"{REPO}:ud-q4_k_xl")

        assert len(plan.downloads) == 1
        assert plan.size == 500
        assert plan.transfer == 500

    def test_fetches_a_repeated_reference_once(self, hub, tmp_path):
        plan = plan_for(hub, tmp_path, f"{REPO}:mmproj-BF16", f"{REPO}:mmproj-BF16")

        download.execute(plan, hub)

        assert [name for _, name in hub.fetched] == ["mmproj-Foo-30B-BF16.gguf"]

    def test_reads_the_file_list_of_one_repository_once(self, hub, tmp_path):
        # Two references into the same repository, so one listing must do.
        plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL", f"{REPO}:mmproj-BF16")

        assert hub.listed == [REPO]


class TestSelection:
    def test_names_the_available_quants_when_the_reference_carries_none(self, hub, tmp_path):
        with pytest.raises(SelectionError) as caught:
            plan_for(hub, tmp_path, REPO)

        message = str(caught.value)
        assert "UD-Q4_K_XL" in message
        assert "mmproj-BF16" in message

    def test_names_the_available_quants_when_the_quant_is_unknown(self, hub, tmp_path):
        with pytest.raises(SelectionError) as caught:
            plan_for(hub, tmp_path, f"{REPO}:Q2_K")

        assert "Q2_K" in str(caught.value)
        assert "UD-Q4_K_XL" in str(caught.value)

    def test_refuses_a_reference_without_a_repository(self, hub, tmp_path):
        with pytest.raises(SelectionError) as caught:
            plan_for(hub, tmp_path, ":UD-Q4_K_XL")

        assert "org/repo" in str(caught.value)

    def test_refuses_a_bare_word_because_the_hub_cannot_be_searched(self, hub, tmp_path):
        with pytest.raises(SelectionError) as caught:
            plan_for(hub, tmp_path, "foo")

        assert "org/repo" in str(caught.value)

    def test_says_so_when_the_repository_holds_no_gguf_file(self, tmp_path):
        hub = FakeHub({REPO: {"commit": COMMIT_A, "files": {"README.md": 5}}})

        with pytest.raises(SelectionError) as caught:
            plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL")

        assert "no GGUF file" in str(caught.value)


class TestExecution:
    def test_fetches_every_file_of_every_download(self, hub, tmp_path):
        plan = plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL")

        download.execute(plan, hub)

        assert [name for _, name in hub.fetched] == [
            "Foo-30B-UD-Q4_K_XL-00001-of-00002.gguf",
            "Foo-30B-UD-Q4_K_XL-00002-of-00002.gguf",
        ]

    def test_fetches_nothing_outside_the_plan(self, hub, tmp_path):
        plan = plan_for(hub, tmp_path, f"{REPO}:mmproj-BF16")

        download.execute(plan, hub)

        assert [name for _, name in hub.fetched] == ["mmproj-Foo-30B-BF16.gguf"]


class TestUnreachableHub:
    def test_reports_a_hub_it_cannot_reach(self, hub, tmp_path):
        hub.reachable = False

        with pytest.raises(DownloadError):
            plan_for(hub, tmp_path, f"{REPO}:UD-Q4_K_XL")

    def test_reports_a_repository_that_is_not_there(self, hub, tmp_path):
        with pytest.raises(DownloadError):
            plan_for(hub, tmp_path, "nobody/Missing-GGUF:Q4_K_M")
