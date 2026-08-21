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
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from llama_cache_manager.download import DownloadError
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
