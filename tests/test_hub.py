"""What the hub adapter makes of the failures the library raises.

Nothing here goes near the network. Each library error is raised inside the
translation on its own, because the order of the clauses decides which message
a user reads, and the library's exception tree is not a flat one.
"""

from __future__ import annotations

import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from llama_cache_manager.download import DownloadError, UnavailableError
from llama_cache_manager.hub import _translated

WHAT = "unsloth/Foo-30B-GGUF/Foo-30B-Q4_K_M.gguf"


class Response:
    """The bare minimum an ``HfHubHTTPError`` wants of a response."""

    status_code = 404
    headers: dict[str, str] = {}
    request = None
    text = ""

    def json(self) -> dict:
        return {}


def raising(error: Exception) -> str:
    with pytest.raises(DownloadError) as caught, _translated(WHAT, "main"):
        raise error
    return str(caught.value)


def test_reports_an_unreachable_hub_as_unreachable():
    # LocalEntryNotFoundError is a subclass of EntryNotFoundError, so a clause
    # for the parent placed first would report this as a missing file.
    message = raising(LocalEntryNotFoundError("Cannot reach the hub\nand more detail"))

    assert "cannot reach the hub" in message.lower()
    assert "no file" not in message


def test_keeps_only_the_first_line_of_a_long_library_message():
    message = raising(LocalEntryNotFoundError("Cannot reach the hub\nand more detail"))

    assert "and more detail" not in message


def test_reports_a_missing_file_as_missing():
    assert "no file" in raising(EntryNotFoundError("gone"))


def test_names_the_revision_that_is_not_there():
    message = raising(RevisionNotFoundError("gone", response=Response()))

    assert "'main'" in message


def test_says_what_to_do_about_a_gated_repository():
    message = raising(GatedRepoError("gated", response=Response()))

    assert "hf auth login" in message


def test_offers_the_token_as_a_reason_for_a_missing_repository():
    message = raising(RepositoryNotFoundError("gone", response=Response()))

    assert "token" in message


def test_reports_a_refusal_from_the_hub():
    assert "refused" in raising(HfHubHTTPError("no", response=Response()))


def test_reports_a_reference_the_library_will_not_accept():
    assert WHAT in raising(HFValidationError("bad repo id"))


def test_reports_a_cache_that_cannot_be_written():
    assert "into the cache" in raising(PermissionError("read-only file system"))


def test_names_what_failed_in_every_message():
    errors = [
        EntryNotFoundError("gone"),
        LocalEntryNotFoundError("offline"),
        RevisionNotFoundError("gone", response=Response()),
        GatedRepoError("gated", response=Response()),
        RepositoryNotFoundError("gone", response=Response()),
        HfHubHTTPError("no", response=Response()),
        HFValidationError("bad"),
        PermissionError("read-only file system"),
    ]

    assert all(WHAT in raising(error) for error in errors)


def raised_by(error: Exception) -> type:
    with pytest.raises(DownloadError) as caught, _translated(WHAT, "main"):
        raise error
    return type(caught.value)


class TestWhoIsToBlame:
    """A run over the whole cache steps over one repository it cannot have.

    It must not step over a network that is down, because that would report
    every repository as fine. So the two cases are separate classes.
    """

    @pytest.mark.parametrize(
        "error",
        [
            RepositoryNotFoundError("gone", response=Response()),
            RevisionNotFoundError("gone", response=Response()),
            EntryNotFoundError("gone"),
            GatedRepoError("gated", response=Response()),
            HFValidationError("bad repo id"),
        ],
        ids=["repository", "revision", "file", "gated", "invalid"],
    )
    def test_calls_an_answer_the_user_cannot_use_unavailable(self, error):
        assert raised_by(error) is UnavailableError

    @pytest.mark.parametrize(
        "error",
        [
            LocalEntryNotFoundError("offline"),
            HfHubHTTPError("too many requests", response=Response()),
            PermissionError("read-only file system"),
        ],
        ids=["offline", "refused", "unwritable"],
    )
    def test_leaves_a_missing_answer_as_a_plain_failure(self, error):
        assert raised_by(error) is DownloadError


class TestWhatFailed:
    """A network failure must not be reported as a write failure.

    Both arrive as an OSError, because a connection error is one, so the
    clause order decides which message the user reads.
    """

    @pytest.mark.parametrize(
        "error",
        [
            OfflineModeIsEnabled("offline mode is enabled"),
            ConnectionError("connection reset by peer"),
            TimeoutError("timed out"),
        ],
        ids=["offline", "reset", "timeout"],
    )
    def test_calls_a_connection_failure_unreachable(self, error):
        message = raising(error)

        assert "cannot reach the hub" in message
        assert "write" not in message

    def test_still_reports_a_cache_it_cannot_write(self):
        assert "into the cache" in raising(PermissionError("read-only file system"))
