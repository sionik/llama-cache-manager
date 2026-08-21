import pytest

from llama_cache_manager import refs
from llama_cache_manager.refs import Reference, ReferenceError


class TestQuantOfFile:
    @pytest.mark.parametrize(
        ("repo_id", "file_name", "expected"),
        [
            # Names taken from real unsloth GGUF repositories.
            ("unsloth/gemma-4-12B-it-GGUF", "gemma-4-12b-it-UD-Q6_K_XL.gguf", "UD-Q6_K_XL"),
            ("unsloth/gemma-4-12B-it-GGUF", "mmproj-BF16.gguf", "mmproj-BF16"),
            ("unsloth/Muse-Glimmer-30B-GGUF", "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
            ("unsloth/Muse-Glimmer-30B-GGUF", "mmproj-Muse-Glimmer-30B-Q8_0.gguf", "mmproj-Q8_0"),
            ("unsloth/gemma-4-26B-A4B-it-qat-GGUF", "mtp-gemma-4-26B-A4B-it.gguf", "mtp"),
            ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ],
    )
    def test_reads_the_quant_shown_on_the_hub(self, repo_id, file_name, expected):
        assert refs.quant_of_file(repo_id, file_name) == expected

    def test_ignores_case_differences_between_repository_and_file(self):
        # The repository says 12B-it, the file says 12b-it.
        assert refs.quant_of_file("unsloth/gemma-4-12B-it-GGUF", "gemma-4-12b-it-Q4_K_M.gguf") == "Q4_K_M"

    def test_gives_every_shard_of_a_split_file_the_same_label(self):
        first = refs.quant_of_file("unsloth/Big-30B-GGUF", "Big-30B-UD-Q4_K_XL-00001-of-00002.gguf")
        second = refs.quant_of_file("unsloth/Big-30B-GGUF", "Big-30B-UD-Q4_K_XL-00002-of-00002.gguf")
        assert first == second == "UD-Q4_K_XL"

    def test_keeps_the_stem_when_the_file_is_only_the_repository_name(self):
        assert refs.quant_of_file("unsloth/Big-30B-GGUF", "Big-30B.gguf") == "Big-30B"

    def test_strips_a_short_repository_name_in_full(self):
        assert refs.quant_of_file("unsloth/Qwen3-GGUF", "Qwen3-Q4_K_M.gguf") == "Q4_K_M"
        assert refs.quant_of_file("google/gemma-GGUF", "mmproj-gemma-F16.gguf") == "mmproj-F16"

    def test_does_not_strip_a_fragment_of_a_longer_repository_name(self):
        # "gemma" on its own is a fragment of the name and must stay put.
        assert (
            refs.quant_of_file("unsloth/gemma-4-26B-it-GGUF", "mmproj-gemma-tuned-F16.gguf")
            == "mmproj-gemma-tuned-F16"
        )


class TestBaseName:
    def test_drops_the_gguf_marker(self):
        assert refs.base_name("unsloth/gemma-4-12B-it-GGUF") == "gemma-4-12B-it"

    def test_keeps_a_name_without_the_marker(self):
        assert refs.base_name("unsloth/Muse-Glimmer-30B") == "Muse-Glimmer-30B"


class TestParse:
    def test_reads_a_repository(self):
        assert refs.parse("unsloth/Qwen3.8-27B-GGUF") == Reference(
            raw="unsloth/Qwen3.8-27B-GGUF", repo_id="unsloth/Qwen3.8-27B-GGUF"
        )

    def test_reads_a_repository_and_quant(self):
        parsed = refs.parse("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL")
        assert parsed.repo_id == "unsloth/Qwen3.8-27B-GGUF"
        assert parsed.quant == "UD-Q4_K_XL"

    def test_reads_the_on_disk_spelling(self):
        assert refs.parse("models--unsloth--Qwen3.8-27B-GGUF").repo_id == "unsloth/Qwen3.8-27B-GGUF"

    def test_reads_a_quant_on_its_own(self):
        parsed = refs.parse(":UD-Q4_K_XL")
        assert parsed.quant == "UD-Q4_K_XL"
        assert parsed.repo_id is None

    def test_reads_a_bare_word_as_free_text(self):
        assert refs.parse("qwen").text == "qwen"

    def test_keeps_a_dash_containing_org(self):
        assert refs.parse("models--mlx-community--Foo").repo_id == "mlx-community/Foo"

    @pytest.mark.parametrize("raw", ["", "   ", "a:b:c", "unsloth/", "/repo", "repo:", "word:Q4"])
    def test_rejects_text_that_is_not_a_reference(self, raw):
        with pytest.raises(ReferenceError):
            refs.parse(raw)


class TestFolderNames:
    def test_round_trips_a_repository_id(self):
        folder = refs.folder_name_of_repo_id("unsloth/Qwen3.8-27B-GGUF")
        assert folder == "models--unsloth--Qwen3.8-27B-GGUF"
        assert refs.repo_id_of_folder(folder) == "unsloth/Qwen3.8-27B-GGUF"

    def test_rejects_a_folder_without_an_org(self):
        with pytest.raises(ReferenceError):
            refs.repo_id_of_folder("models--onlyname")


class TestGroupByArtifact:
    def test_puts_every_shard_of_a_split_file_under_one_label(self):
        groups = refs.group_by_artifact(
            "unsloth/Big-30B-GGUF",
            [
                "Big-30B-UD-Q4_K_XL-00002-of-00002.gguf",
                "Big-30B-UD-Q4_K_XL-00001-of-00002.gguf",
            ],
        )

        assert groups == {
            "UD-Q4_K_XL": (
                "Big-30B-UD-Q4_K_XL-00001-of-00002.gguf",
                "Big-30B-UD-Q4_K_XL-00002-of-00002.gguf",
            )
        }

    def test_leaves_out_what_is_not_a_gguf_file(self):
        groups = refs.group_by_artifact(
            "unsloth/Big-30B-GGUF", ["README.md", "config.json", "Big-30B-Q4_K_M.gguf"]
        )

        assert list(groups) == ["Q4_K_M"]

    def test_keeps_the_path_when_two_files_would_share_a_label(self):
        # Both file names reduce to the same quant, so neither may own it.
        groups = refs.group_by_artifact("unsloth/Big-30B-GGUF", ["Q4_K_M/Big-30B.gguf", "spare/Big-30B.gguf"])

        assert sorted(groups) == ["Q4_K_M/Big-30B", "spare/Big-30B"]

    def test_reads_a_quant_that_sits_in_its_own_directory(self):
        groups = refs.group_by_artifact("unsloth/Big-30B-GGUF", ["UD-Q4_K_XL/Big-30B-UD-Q4_K_XL.gguf"])

        assert list(groups) == ["UD-Q4_K_XL"]
