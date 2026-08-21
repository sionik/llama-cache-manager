"""What the scan makes of a cache directory."""

from __future__ import annotations

import pytest
from conftest import COMMIT_A, COMMIT_B

from llama_cache_manager import refs
from llama_cache_manager.cache import CacheError, Kind, repos_matching, scan, select

REPO = "unsloth/Foo-30B-GGUF"


def only_repo(fake_cache):
    return scan(fake_cache.root).repos[0]


def _references(repos):
    return [
        f"{repo.repo_id}:{artifact.quant}"
        for repo in repos
        for revision in repo.revisions
        for artifact in revision.artifacts
    ]


class TestArtifacts:
    def test_makes_one_artifact_out_of_every_shard_of_a_split_file(self, fake_cache):
        fake_cache.revision(
            REPO,
            COMMIT_A,
            {
                "Foo-30B-UD-Q4_K_XL-00001-of-00002.gguf": 100,
                "Foo-30B-UD-Q4_K_XL-00002-of-00002.gguf": 200,
            },
        )

        artifacts = only_repo(fake_cache).revisions[0].artifacts

        assert [artifact.quant for artifact in artifacts] == ["UD-Q4_K_XL"]
        assert artifacts[0].shards == 2
        assert artifacts[0].size == 300

    def test_reads_a_quant_that_sits_in_its_own_directory(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"UD-Q4_K_XL/Foo-30B-UD-Q4_K_XL-00001-of-00001.gguf": 100})

        artifacts = only_repo(fake_cache).revisions[0].artifacts

        assert [artifact.quant for artifact in artifacts] == ["UD-Q4_K_XL"]

    def test_keeps_files_apart_when_two_names_would_share_a_label(self, fake_cache):
        # Both stems reduce to "extra", so neither may take the shared label.
        fake_cache.revision(REPO, COMMIT_A, {"extra-Foo-30B.gguf": 10, "Foo-30B-extra.gguf": 20})

        quants = sorted(artifact.quant for artifact in only_repo(fake_cache).revisions[0].artifacts)

        assert quants == ["Foo-30B-extra", "extra-Foo-30B"]

    def test_sorts_models_before_projectors_and_extras(self, fake_cache):
        fake_cache.revision(
            REPO,
            COMMIT_A,
            {"mtp-Foo-30B.gguf": 10, "mmproj-BF16.gguf": 20, "Foo-30B-Q4_K_M.gguf": 30},
        )

        kinds = [artifact.kind for artifact in only_repo(fake_cache).revisions[0].artifacts]

        assert kinds == [Kind.MODEL, Kind.PROJECTOR, Kind.EXTRA]

    def test_leaves_files_that_are_not_gguf_out_of_the_artifacts(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 30, "config.json": 5})

        revision = only_repo(fake_cache).revisions[0]

        assert [artifact.quant for artifact in revision.artifacts] == ["Q4_K_M"]
        assert len(revision.others) == 1
        assert revision.size == 35


class TestSizes:
    def test_counts_a_blob_shared_between_revisions_once(self, fake_cache):
        fake_cache.blob(REPO, "shared", 1000)
        fake_cache.link(REPO, COMMIT_A, "Foo-30B-Q4_K_M.gguf", "shared")
        fake_cache.link(REPO, COMMIT_B, "Foo-30B-Q4_K_M.gguf", "shared")
        fake_cache.ref(REPO, COMMIT_A)

        repo = only_repo(fake_cache)

        assert repo.size == 1000
        assert sum(revision.size for revision in repo.revisions) == 2000

    def test_counts_strays_and_interrupted_downloads_in_the_repository_size(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        fake_cache.blob(REPO, "stray", 40)
        fake_cache.blob(REPO, "half.incomplete", 7)

        assert only_repo(fake_cache).size == 147


class TestReachability:
    def test_calls_a_revision_detached_when_no_name_points_at_it(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 10})
        fake_cache.revision(REPO, COMMIT_B, {"Foo-30B-Q6_K.gguf": 10}, ref=None)

        by_commit = {revision.commit: revision for revision in only_repo(fake_cache).revisions}

        assert by_commit[COMMIT_A].detached is False
        assert by_commit[COMMIT_B].detached is True

    def test_finds_a_blob_that_nothing_points_at(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 10})
        fake_cache.blob(REPO, "orphan", 40)

        strays = only_repo(fake_cache).strays

        assert [blob.path.name for blob in strays] == ["orphan"]
        assert strays[0].size == 40

    def test_reports_an_interrupted_download_apart_from_the_strays(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 10})
        fake_cache.blob(REPO, "half.incomplete", 7)

        repo = only_repo(fake_cache)

        assert repo.strays == ()
        assert [blob.path.name for blob in repo.incomplete] == ["half.incomplete"]


class TestSelection:
    @pytest.fixture
    def two_repos(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 10, "mmproj-BF16.gguf": 5})
        fake_cache.revision("other/Bar-GGUF", COMMIT_B, {"Bar-Q8_0.gguf": 10})
        return scan(fake_cache.root)

    def test_selects_by_organisation(self, two_repos):
        found = repos_matching(two_repos, refs.parse("unsloth"))
        assert [repo.repo_id for repo in found] == [REPO]

    def test_selects_by_quant_across_repositories(self, two_repos):
        found = select(two_repos, [refs.parse(":Q8_0")])
        assert _references(found) == ["other/Bar-GGUF:Q8_0"]

    def test_selects_one_artifact_by_its_full_reference(self, two_repos):
        found = select(two_repos, [refs.parse(f"{REPO}:mmproj-BF16")])
        assert _references(found) == [f"{REPO}:mmproj-BF16"]

    def test_ignores_case_in_a_reference(self, two_repos):
        found = select(two_repos, [refs.parse(f"{REPO.lower()}:q4_k_m")])
        assert _references(found) == [f"{REPO}:Q4_K_M"]

    def test_leaves_loose_blobs_out_when_a_quant_is_named(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 10})
        fake_cache.blob(REPO, "orphan", 40)
        cache = scan(fake_cache.root)

        assert select(cache, [refs.parse(":Q4_K_M")])[0].strays == ()
        assert select(cache, [refs.parse("unsloth")])[0].strays != ()

    def test_reports_only_the_size_of_the_rows_it_kept(self, two_repos):
        found = select(two_repos, [refs.parse(f"{REPO}:Q4_K_M")])
        assert found[0].size == 10


class TestMissingCache:
    def test_says_which_directory_it_could_not_read(self, tmp_path):
        with pytest.raises(CacheError, match="not found"):
            scan(tmp_path / "nowhere")


class TestPathSpellings:
    """The hub reports blobs resolved, so a symlink gives one file two names."""

    def test_does_not_call_a_live_blob_stray_when_the_blobs_directory_is_a_symlink(self, fake_cache):
        elsewhere = fake_cache.root / "elsewhere"
        elsewhere.mkdir(parents=True)
        (elsewhere / "live").write_bytes(b"\0" * 1000)
        repo = fake_cache.repo_path(REPO)
        (repo / "snapshots" / COMMIT_A).mkdir(parents=True)
        (repo / "blobs").symlink_to("../elsewhere")
        fake_cache.link(REPO, COMMIT_A, "Foo-30B-Q4_K_M.gguf", "live")
        fake_cache.ref(REPO, COMMIT_A)

        found = only_repo(fake_cache)

        assert found.strays == ()
        assert found.size == 1000

    def test_keeps_a_stray_whose_own_path_is_a_symlink_out_of_the_cache(self, fake_cache, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "blob").write_bytes(b"\0" * 10)
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 100})
        (fake_cache.repo_path(REPO) / "blobs" / "wandering").symlink_to(outside / "blob")

        strays = only_repo(fake_cache).strays

        # Reported, and the removal guard is what refuses to follow it out.
        assert [blob.path for blob in strays] == [outside / "blob"]


class TestSameNameInTwoDirectories:
    def test_keeps_two_files_of_one_name_in_different_directories_apart(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Q4_K_M/model.gguf": 500, "Q8_0/model.gguf": 900})

        artifacts = only_repo(fake_cache).revisions[0].artifacts

        assert sorted(artifact.quant for artifact in artifacts) == ["Q4_K_M/model", "Q8_0/model"]
        assert all(artifact.shards == 1 for artifact in artifacts)

    def test_still_groups_the_shards_that_share_a_directory(self, fake_cache):
        fake_cache.revision(
            REPO,
            COMMIT_A,
            {
                "UD-Q4_K_XL/Foo-30B-UD-Q4_K_XL-00001-of-00002.gguf": 100,
                "UD-Q4_K_XL/Foo-30B-UD-Q4_K_XL-00002-of-00002.gguf": 200,
            },
        )

        artifacts = only_repo(fake_cache).revisions[0].artifacts

        assert [(artifact.quant, artifact.shards) for artifact in artifacts] == [("UD-Q4_K_XL", 2)]


class TestSharing:
    def test_calls_a_split_file_shared_when_another_revision_holds_it_too(self, fake_cache):
        for name, size in (("s1", 100), ("s2", 200)):
            fake_cache.blob(REPO, name, size)
        for commit in (COMMIT_A, COMMIT_B):
            fake_cache.link(REPO, commit, "Foo-30B-Q6_K-00001-of-00002.gguf", "s1")
            fake_cache.link(REPO, commit, "Foo-30B-Q6_K-00002-of-00002.gguf", "s2")
        fake_cache.ref(REPO, COMMIT_A)

        cache = scan(fake_cache.root)
        artifact = cache.repos[0].revisions[0].artifacts[0]

        assert artifact.shards == 2
        assert cache.is_shared(artifact) is True

    def test_does_not_call_a_split_file_shared_when_it_stands_alone(self, fake_cache):
        fake_cache.revision(
            REPO,
            COMMIT_A,
            {"Foo-30B-Q6_K-00001-of-00002.gguf": 100, "Foo-30B-Q6_K-00002-of-00002.gguf": 200},
        )

        cache = scan(fake_cache.root)

        assert cache.is_shared(cache.repos[0].revisions[0].artifacts[0]) is False


class TestEmptyRevision:
    def test_dates_a_revision_without_files_by_its_directory(self, fake_cache):
        import os

        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 10})
        empty = fake_cache.repo_path(REPO) / "snapshots" / COMMIT_B
        empty.mkdir(parents=True)
        os.utime(empty, (1_700_000_000, 1_700_000_000))

        by_commit = {revision.commit: revision for revision in only_repo(fake_cache).revisions}

        assert by_commit[COMMIT_B].modified == 1_700_000_000


class TestBareWordMeansTheSameThingTwice:
    """A word that narrowed a listing must not widen a removal."""

    @pytest.fixture
    def two_quants(self, fake_cache):
        fake_cache.revision(REPO, COMMIT_A, {"Foo-30B-Q4_K_M.gguf": 5000, "Foo-30B-Q6_K.gguf": 6000})
        return scan(fake_cache.root)

    def test_a_word_matching_only_a_quant_resolves_to_that_artifact(self, two_quants):
        from llama_cache_manager.cache import ArtifactTarget, resolve

        target = resolve(two_quants, refs.parse("Q4_K_M"))

        assert isinstance(target, ArtifactTarget)
        assert [match.artifact.quant for match in target.matches] == ["Q4_K_M"]

    def test_a_word_matching_the_repository_resolves_to_the_repository(self, two_quants):
        from llama_cache_manager.cache import RepoTarget, resolve

        assert isinstance(resolve(two_quants, refs.parse("Foo-30B")), RepoTarget)

    def test_selects_the_same_artifacts_as_a_listing(self, two_quants):
        from llama_cache_manager.cache import resolve

        listed = _references(select(two_quants, [refs.parse("Q4_K_M")]))
        target = resolve(two_quants, refs.parse("Q4_K_M"))

        assert listed == [match.reference for match in target.matches]
