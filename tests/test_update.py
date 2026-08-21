"""Keeping the cache current, with the hub faked out."""

from __future__ import annotations

import pytest
from conftest import COMMIT_A, COMMIT_B, FakeHub

from llama_cache_manager import update
from llama_cache_manager.cache import scan
from llama_cache_manager.download import DownloadError, UnavailableError
from llama_cache_manager.removal import Reason

REPO = "unsloth/Foo-30B-GGUF"
OTHER = "other/Bar-GGUF"
COMMIT_C = "c" * 40

LOCAL = {"Foo-30B-Q4_K_M.gguf": 500, "mmproj-Foo-30B-BF16.gguf": 70}
REMOTE = {"Foo-30B-Q4_K_M.gguf": 600, "mmproj-Foo-30B-BF16.gguf": 70}


@pytest.fixture
def cache(fake_cache):
    """One repository at commit A, tracking main."""
    fake_cache.revision(REPO, COMMIT_A, dict(LOCAL))
    return fake_cache


def hub_at(commit, files=None, cache=None, repo_id=REPO):
    return FakeHub({repo_id: {"commit": commit, "files": dict(files or REMOTE)}}, cache=cache)


def tracked_of(fake_cache):
    model = scan(fake_cache.root)
    return model, update.tracked(model, list(model.repos))


def stale_of(fake_cache, hub):
    _, names = tracked_of(fake_cache)
    return list(update.check(hub, names).outdated)


def after_download(fake_cache, stale):
    """The cache as it looks once the update has fetched, with the names moved."""
    for item in stale:
        fake_cache.revision(item.repo_id, item.remote_commit, dict(REMOTE), ref=item.name)
    return scan(fake_cache.root)


class TestTracking:
    def test_tracks_a_revision_a_name_points_at(self, cache):
        _, names = tracked_of(cache)

        assert [(item.repo.repo_id, item.name, item.revision.commit) for item in names] == [
            (REPO, "main", COMMIT_A)
        ]

    def test_leaves_out_a_detached_revision(self, fake_cache):
        # Nothing says which name it once followed, so there is nothing to
        # compare it against.
        fake_cache.revision(REPO, COMMIT_A, dict(LOCAL), ref=None)

        _, names = tracked_of(fake_cache)

        assert names == []

    def test_leaves_out_a_revision_holding_no_gguf_file(self, fake_cache):
        # A cache shared with other huggingface_hub users holds repositories
        # this tool never downloaded. Nothing in them would be fetched at a
        # newer commit, so nothing in them may be replaced either.
        fake_cache.revision("meta/Text-Model", COMMIT_A, {"config.json": 10})

        _, names = tracked_of(fake_cache)

        assert names == []

    def test_tracks_every_name_of_a_repository(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, dict(LOCAL))
        fake_cache.ref(REPO, COMMIT_A, "v1.0")

        _, names = tracked_of(fake_cache)

        assert sorted(item.name for item in names) == ["main", "v1.0"]


class Refusing(FakeHub):
    """A hub that will not serve one repository, however it is asked."""

    def __init__(self, repos, refuse, error, **kwargs):
        super().__init__(repos, **kwargs)
        self.refuse = refuse
        self.error = error

    def head(self, repo_id, revision):
        self.heads.append((repo_id, revision))
        if repo_id == self.refuse:
            raise self.error
        return self.repos[repo_id]["commit"]


class TestChecking:
    def test_reports_a_name_the_hub_has_moved(self, cache):
        _, names = tracked_of(cache)

        found = update.check(hub_at(COMMIT_B), names)

        assert [(item.repo_id, item.local_commit, item.remote_commit) for item in found.outdated] == [
            (REPO, COMMIT_A, COMMIT_B)
        ]

    def test_reports_nothing_when_the_commits_agree(self, cache):
        _, names = tracked_of(cache)

        found = update.check(hub_at(COMMIT_A), names)

        assert found.outdated == ()
        assert found.skipped == ()

    def test_costs_one_request_per_name_and_reads_no_file_lists(self, cache):
        # A cache-wide check has to be cheap: a commit hash says whether there
        # is anything to do, so nothing else is asked for.
        _, names = tracked_of(cache)
        hub = hub_at(COMMIT_B)

        update.check(hub, names)

        assert hub.heads == [(REPO, "main")]
        assert hub.listed == []

    def test_names_the_quants_the_cache_holds(self, cache):
        _, names = tracked_of(cache)

        found = update.check(hub_at(COMMIT_B), names)

        assert sorted(found.outdated[0].quants) == ["Q4_K_M", "mmproj-BF16"]

    def test_steps_over_a_repository_the_hub_will_not_serve(self, fake_cache):
        # A cache holds repositories that were taken down, or went private, or
        # need a licence nobody accepted. One of those must not stop the rest.
        fake_cache.revision(REPO, COMMIT_A, dict(LOCAL))
        fake_cache.revision(OTHER, COMMIT_A, {"Bar-Q8_0.gguf": 90})
        hub = Refusing(
            {
                REPO: {"commit": COMMIT_B, "files": dict(REMOTE)},
                OTHER: {"commit": COMMIT_B, "files": {"Bar-Q8_0.gguf": 95}},
            },
            refuse=REPO,
            error=UnavailableError("the hub has no unsloth/Foo-30B-GGUF"),
        )
        _, names = tracked_of(fake_cache)

        found = update.check(hub, names)

        assert [item.repo_id for item in found.outdated] == [OTHER]
        assert [(item.repo_id, item.name) for item in found.skipped] == [(REPO, "main")]
        assert "no unsloth/Foo-30B-GGUF" in found.skipped[0].reason

    def test_stops_when_the_hub_cannot_be_reached(self, cache):
        # Every name would be skipped, and a cache reported as up to date when
        # nothing was checked is worse than a run that fails.
        _, names = tracked_of(cache)
        hub = hub_at(COMMIT_B)
        hub.reachable = False

        with pytest.raises(DownloadError):
            update.check(hub, names)


class TestPlanning:
    def test_plans_the_quants_the_cache_holds_at_the_new_revision(self, cache, tmp_path):
        _, names = tracked_of(cache)
        hub = hub_at(COMMIT_B)
        found = update.check(hub, names)

        plan = update.plan(hub, tmp_path, found)

        assert sorted(item.reference for item in plan.downloads.downloads) == [
            f"{REPO}:Q4_K_M",
            f"{REPO}:mmproj-BF16",
        ]
        assert plan.downloads.downloads[0].revision == "main"

    def test_prices_the_download_at_the_new_revision(self, cache, tmp_path):
        _, names = tracked_of(cache)
        hub = hub_at(COMMIT_B)
        found = update.check(hub, names)

        plan = update.plan(hub, tmp_path, found)

        # The remote Q4_K_M is 600 where the cached one was 500.
        assert plan.downloads.size == 670

    def test_skips_a_repository_whose_quant_is_gone_and_keeps_the_rest(self, fake_cache, tmp_path):
        fake_cache.revision(REPO, COMMIT_A, dict(LOCAL))
        fake_cache.revision(OTHER, COMMIT_A, {"Bar-Q8_0.gguf": 90})
        hub = FakeHub(
            {
                # The projector is gone from the hub, so this repository cannot
                # be updated as it stands.
                REPO: {"commit": COMMIT_B, "files": {"Foo-30B-Q4_K_M.gguf": 600}},
                OTHER: {"commit": COMMIT_B, "files": {"Bar-Q8_0.gguf": 95}},
            }
        )
        model = scan(fake_cache.root)
        found = update.check(hub, update.tracked(model, list(model.repos)))

        plan = update.plan(hub, tmp_path, found)

        assert [item.repo_id for item in plan.replaced] == [OTHER]
        assert [item.repo_id for item in plan.skipped] == [REPO]
        assert "mmproj-BF16" in plan.skipped[0].reason

    def test_holds_nothing_when_every_repository_is_skipped(self, cache, tmp_path):
        _, names = tracked_of(cache)
        hub = hub_at(COMMIT_B, files={"Foo-30B-Q4_K_M.gguf": 600})
        found = update.check(hub, names)

        plan = update.plan(hub, tmp_path, found)

        assert plan.empty
        assert plan.skipped


class TestSuperseded:
    def test_names_the_revision_the_update_replaced(self, cache):
        stale = stale_of(cache, hub_at(COMMIT_B))

        items = update.superseded(after_download(cache, stale), stale)

        assert [(item.reason, item.revision.commit) for item in items] == [(Reason.SUPERSEDED, COMMIT_A)]

    def test_leaves_a_revision_no_update_touched_alone(self, fake_cache):
        # A detached revision that was already there is prune's business, not
        # an update's.
        fake_cache.revision(REPO, COMMIT_A, dict(LOCAL))
        fake_cache.revision(REPO, COMMIT_C, {"Foo-30B-Q4_K_M.gguf": 400}, ref=None)
        stale = stale_of(fake_cache, hub_at(COMMIT_B))

        items = update.superseded(after_download(fake_cache, stale), stale)

        assert [item.revision.commit for item in items] == [COMMIT_A]

    def test_leaves_a_revision_a_second_name_still_points_at_alone(self, fake_cache):
        # The tag was fetched at the same commit as main. Moving main leaves
        # the revision reachable by the tag, so removing it would take a model
        # away that the user can still name.
        fake_cache.revision(REPO, COMMIT_A, dict(LOCAL))
        fake_cache.ref(REPO, COMMIT_A, "v1.0")
        stale = [item for item in stale_of(fake_cache, hub_at(COMMIT_B)) if item.name == "main"]

        items = update.superseded(after_download(fake_cache, stale), stale)

        assert items == []
