"""The hub as ``huggingface_hub`` reaches it.

This is the only place that talks to the network. It implements the
:class:`~llama_cache_manager.download.Hub` port and turns every failure the
library raises into a :class:`~llama_cache_manager.download.DownloadError` that
says what to do about it, so nothing above this module has to know the library's
exception tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .download import DownloadError, FileStatus


class HubApi:
    """Reads huggingface.co, with the token taken from the usual places.

    ``huggingface_hub`` finds the token in ``HF_TOKEN`` or in the file that
    ``hf auth login`` writes, so a gated repository needs no option here.
    """

    def __init__(self) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi()

    def file_names(self, repo_id: str, revision: str) -> tuple[str, ...]:
        with _translated(repo_id, revision):
            return tuple(self._api.list_repo_files(repo_id, revision=revision))

    def inspect(self, repo_id: str, name: str, revision: str, cache_dir: Path) -> FileStatus:
        from huggingface_hub import hf_hub_download

        with _translated(f"{repo_id}/{name}", revision):
            report = hf_hub_download(
                repo_id,
                name,
                revision=revision,
                cache_dir=cache_dir,
                dry_run=True,
            )
        return FileStatus(
            name=name,
            size=report.file_size,
            commit=report.commit_hash,
            # What matters to the reader is the transfer, not whether a link
            # still has to be made. A blob that is already there downloads
            # nothing even when this revision has never been checked out.
            cached=not report.will_download,
        )

    def fetch(self, repo_id: str, name: str, revision: str, cache_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        with _translated(f"{repo_id}/{name}", revision):
            return Path(hf_hub_download(repo_id, name, revision=revision, cache_dir=cache_dir))


@contextmanager
def _translated(what: str, revision: str) -> Iterator[None]:
    """Turn a library failure into a ``DownloadError`` naming ``what``."""
    from huggingface_hub.errors import (
        DisabledRepoError,
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        yield
    except GatedRepoError as error:
        raise DownloadError(
            f"{what} is gated: accept its licence on huggingface.co and log in with 'hf auth login'"
        ) from error
    except DisabledRepoError as error:
        raise DownloadError(f"{what} has been disabled on the hub") from error
    except RepositoryNotFoundError as error:
        raise DownloadError(
            f"the hub has no {what}, or it is private and the token does not reach it"
        ) from error
    except RevisionNotFoundError as error:
        raise DownloadError(f"the hub has no revision {revision!r} of {what}") from error
    except LocalEntryNotFoundError as error:
        # Offline, or the hub did not answer. This clause has to come before the
        # one below: the library derives this error from EntryNotFoundError, so a
        # parent clause placed first would report an unreachable hub as a file
        # that is not there. The message the library gives is long, so only its
        # first line is kept.
        raise DownloadError(f"cannot reach the hub for {what}: {_first_line(error)}") from error
    except EntryNotFoundError as error:
        raise DownloadError(f"the hub has no file {what}") from error
    except HFValidationError as error:
        raise DownloadError(f"the hub cannot be asked for {what}: {error}") from error
    except HfHubHTTPError as error:
        raise DownloadError(f"the hub refused {what}: {_first_line(error)}") from error
    except OSError as error:
        raise DownloadError(f"could not write {what} into the cache: {error}") from error


def _first_line(error: Exception) -> str:
    return str(error).strip().splitlines()[0] if str(error).strip() else error.__class__.__name__
