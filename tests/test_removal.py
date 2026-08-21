"""What a plan promises has to be what the disk gives back."""

from __future__ import annotations

import pytest
from conftest import COMMIT_A, COMMIT_B

from llama_cache_manager import cache as cache_model
from llama_cache_manager import refs, removal
from llama_cache_manager.cache import scan
from llama_cache_manager.removal import Reason, RemovalError

REPO = "unsloth/Foo-30B-GGUF"


def plan_for_reference(cache, text):
    target = cache_model.resolve(cache, refs.parse(text))
    if isinstance(target, cache_model.RepoTarget):
        return removal.build(cache, removal.for_repository(target.repo))
    return removal.build(cache, removal.for_artifacts(target.matches))


class TestSharedBlobs:
    def test_reclaims_a_shared_blob_once_when_every_reference_goes(self, fake_cache):
        # One quant, the same blob, reachable from two revisions. A second
        # quant keeps both revisions alive, so only the blob is at stake.
        fake_cache.blob(REPO, "shared", 3000)
        fake_cache.link(REPO, COMMIT_A, "Foo-30B-Q4_K_M.gguf", "shared")
        fake_cache.link(REPO, COMMIT_B, "Foo-30B-Q4_K_M.gguf", "shared")
        fake_cache.blob(REPO, "kept-a", 10)
        fake_cache.blob(REPO, "kept-b", 10)
        fake_cache.link(REPO, COMMIT_A, "Foo-30B-Q6_K.gguf", "kept-a")
        fake_cache.link(REPO, COMMIT_B, "Foo-30B-Q6_K.gguf", "kept-b")
        fake_cache.ref(REPO, COMMIT_A, "main")

        plan = plan_for_reference(scan(fake_cache.root), f"{REPO}:Q4_K_M")

        # Two entries go, and the one blob behind both is counted once.
        assert len(plan.entries) == 2
        assert len(plan.blobs) == 1
        assert plan.freed == 3000

    def test_reclaims_nothing_when_a_revision_that_stays_shares_every_blob(self, fake_cache):
        fake_cache.blob(REPO, "shared", 3000)
        fake_cache.link(REPO, COMMIT_A, "Foo-30B-Q4_K_M.gguf", "shared")
        fake_cache.link(REPO, COMMIT_B, "Foo-30B-Q4_K_M.gguf", "shared")
        fake_cache.ref(REPO, COMMIT_A, "main")

        cache = scan(fake_cache.root)
        detached = [rev for rev in cache.repos[0].revisions if rev.detached]
        item = removal.Item(reason=Reason.DETACHED, repo=cache.repos[0], revision=detached[0])
        plan = removal.build(cache, [item])

        assert plan.freed == 0
        assert plan.blobs == ()


class TestPlanMatchesDisk:
    def test_frees_exactly_what_the_plan_promised(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000, "mmproj-BF16.gguf": 700})
        fake_cache.revision("unsloth/Bar-GGUF", COMMIT_B, {"Bar-Q8_0.gguf": 900})
        fake_cache.blob(REPO, "stray", 400)

        cache = scan(fake_cache.root)
        plan = plan_for_reference(cache, f"{REPO}:Q4_K_M")
        before = fake_cache.bytes_on_disk()

        removal.execute(plan)

        assert before - fake_cache.bytes_on_disk() == plan.freed == 5000

    def test_frees_exactly_what_the_plan_promised_for_a_whole_repository(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000, "config.json": 40})
        fake_cache.blob(REPO, "stray", 400)
        fake_cache.blob(REPO, "half.incomplete", 70)

        cache = scan(fake_cache.root)
        plan = plan_for_reference(cache, REPO)
        before = fake_cache.bytes_on_disk()

        removal.execute(plan)

        # The whole directory goes: both blobs, the stray, the interrupted
        # download and the ref file that named the revision.
        assert before - fake_cache.bytes_on_disk() == plan.freed == before
        assert not fake_cache.repo_path(REPO).exists()

    def test_frees_exactly_what_the_plan_promised_for_garbage(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000})
        fake_cache.revision(REPO, COMMIT_B, {"Foo-30B-Q6_K.gguf": 6000}, ref=None)
        fake_cache.blob(REPO, "stray", 400)
        fake_cache.blob(REPO, "half.incomplete", 70)

        cache = scan(fake_cache.root)
        plan = removal.build(cache, removal.garbage(cache))
        before = fake_cache.bytes_on_disk()

        removal.execute(plan)

        assert before - fake_cache.bytes_on_disk() == plan.freed == 6000 + 400 + 70
        assert (fake_cache.repo_path(REPO) / "snapshots" / COMMIT_A).is_dir()


class TestOtherFilesAreReferences:
    def test_does_not_call_a_blob_stray_when_only_a_json_file_points_at_it(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000, "config.json": 40})

        cache = scan(fake_cache.root)

        assert cache.repos[0].strays == ()
        plan = removal.build(cache, removal.garbage(cache))
        assert plan.empty

    def test_keeps_a_blob_that_a_json_file_still_needs(self, fake_cache):
        # A blob shared between a GGUF entry and a config entry is contrived,
        # but it must survive the removal of the GGUF entry alone.
        fake_cache.blob(REPO, "shared", 800)
        fake_cache.link(REPO, COMMIT_A, "Foo-30B-Q4_K_M.gguf", "shared")
        fake_cache.link(REPO, COMMIT_A, "config.json", "shared")
        fake_cache.ref(REPO, COMMIT_A)

        cache = scan(fake_cache.root)
        plan = plan_for_reference(cache, f"{REPO}:Q4_K_M")
        removal.execute(plan)

        assert plan.freed == 0
        assert (fake_cache.repo_path(REPO) / "blobs" / "shared").exists()


class TestRefsAndDirectories:
    def test_removes_the_ref_file_of_a_removed_revision(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        fake_cache.revision(REPO, COMMIT_B, {"Foo-30B-Q6_K.gguf": 100}, ref="v2")

        cache = scan(fake_cache.root)
        target = cache_model.resolve(cache, refs.parse(f"{REPO}:Q4_K_M"))
        plan = removal.build(cache, removal.for_artifacts(target.matches))
        removal.execute(plan)

        assert not (fake_cache.repo_path(REPO) / "refs" / "main").exists()
        assert (fake_cache.repo_path(REPO) / "refs" / "v2").exists()

    def test_keeps_the_repository_when_another_revision_stays(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        fake_cache.revision(REPO, COMMIT_B, {"Foo-30B-Q6_K.gguf": 100}, ref="v2")

        cache = scan(fake_cache.root)
        plan = plan_for_reference(cache, f"{REPO}:Q4_K_M")
        removal.execute(plan)

        assert fake_cache.repo_path(REPO).is_dir()
        assert scan(fake_cache.root).repos[0].revisions[0].commit == COMMIT_B


class TestGuard:
    def test_refuses_a_path_outside_the_cache(self, fake_cache, tmp_path):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        cache = scan(fake_cache.root)
        outside = tmp_path / "elsewhere" / "victim"
        outside.parent.mkdir(parents=True)
        outside.write_text("keep me")

        broken = removal.Plan(
            cache=cache,
            items=(),
            entries=(outside,),
            blobs=(),
            revision_dirs=(),
            ref_files=(),
            repo_dirs=(),
            freed=0,
        )

        with pytest.raises(RemovalError, match="outside the cache"):
            removal.execute(broken)
        assert outside.exists()


class TestOlderThan:
    def test_selects_only_revisions_older_than_the_cutoff(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        fake_cache.revision(REPO, COMMIT_B, {"Foo-30B-Q6_K.gguf": 100}, ref="v2")
        old_blob = fake_cache.repo_path(REPO) / "blobs" / f"{COMMIT_A[:6]}-Foo-30B-Q4_K_M.gguf"
        import os

        os.utime(old_blob, (1000, 1000))

        cache = scan(fake_cache.root)
        items = removal.older_than(cache, cutoff=2000)

        assert [item.revision.commit for item in items] == [COMMIT_A]


class TestSymlinkedRepository:
    """A repository moved to another disk is reached through a symlink."""

    def _build(self, fake_cache, tmp_path):
        store = tmp_path / "store" / "models--unsloth--Foo-30B-GGUF"
        (store / "blobs").mkdir(parents=True)
        (store / "refs").mkdir(parents=True)
        (store / "snapshots" / COMMIT_A).mkdir(parents=True)
        (store / "refs" / "main").write_text(COMMIT_A)
        (store / "blobs" / "b1").write_bytes(b"\0" * 500)
        (store / "blobs" / "stray").write_bytes(b"\0" * 90)
        (store / "snapshots" / COMMIT_A / "Foo-30B-Q4_K_M.gguf").symlink_to("../../blobs/b1")
        fake_cache.root.mkdir(parents=True, exist_ok=True)
        (fake_cache.root / "models--unsloth--Foo-30B-GGUF").symlink_to(store)
        return store

    def test_plans_a_removal_through_the_symlink(self, fake_cache, tmp_path):
        self._build(fake_cache, tmp_path)

        plan = plan_for_reference(scan(fake_cache.root), f"{REPO}:Q4_K_M")

        assert plan.withheld == ()
        # The blob, plus the ref file of the revision that ends up empty.
        assert plan.freed == 500 + len(COMMIT_A)

    def test_removes_the_whole_repository_through_the_symlink(self, fake_cache, tmp_path):
        store = self._build(fake_cache, tmp_path)

        plan = plan_for_reference(scan(fake_cache.root), REPO)
        removal.execute(plan)

        assert not store.exists()
        assert not (fake_cache.root / "models--unsloth--Foo-30B-GGUF").exists()


class TestPathsOutsideTheCache:
    def test_leaves_a_blob_that_points_out_of_the_cache_and_removes_the_rest(self, fake_cache, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "victim"
        victim.write_bytes(b"\0" * 111)
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 700})
        (fake_cache.repo_path(REPO) / "blobs" / "wandering").symlink_to(victim)
        fake_cache.blob(REPO, "realstray", 300)

        cache = scan(fake_cache.root)
        plan = removal.build(cache, removal.garbage(cache))
        removal.execute(plan)

        assert plan.withheld == (victim,)
        assert plan.freed == 300
        assert victim.exists()
        assert not (fake_cache.repo_path(REPO) / "blobs" / "realstray").exists()


class TestRevisionWithoutFiles:
    def test_removes_a_repository_that_holds_an_empty_revision_with_a_ref(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        (fake_cache.repo_path(REPO) / "snapshots" / COMMIT_B).mkdir(parents=True)
        fake_cache.ref(REPO, COMMIT_B, "v2")

        cache = scan(fake_cache.root)
        plan = plan_for_reference(cache, REPO)
        removal.execute(plan)

        assert not fake_cache.repo_path(REPO).exists()

    def test_removes_an_empty_revision_named_on_its_own(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        empty = fake_cache.repo_path(REPO) / "snapshots" / COMMIT_B
        empty.mkdir(parents=True)
        fake_cache.ref(REPO, COMMIT_B, "v2")

        cache = scan(fake_cache.root)
        item = next(
            removal.Item(reason=Reason.REVISION, repo=cache.repos[0], revision=revision)
            for revision in cache.repos[0].revisions
            if revision.commit == COMMIT_B
        )
        removal.execute(removal.build(cache, [item]))

        assert not empty.exists()
        assert not (fake_cache.repo_path(REPO) / "refs" / "v2").exists()
        assert (fake_cache.repo_path(REPO) / "snapshots" / COMMIT_A).is_dir()
