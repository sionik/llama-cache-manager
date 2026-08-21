"""A cache on disk, built one file at a time, for the tests to work on."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llama_cache_manager import refs


class FakeCache:
    """Builds a Hugging Face style cache directory.

    Blobs are created by name so that a test can point two snapshot entries at
    the same blob, which is the case the size accounting has to get right.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def repo_path(self, repo_id: str) -> Path:
        return self.root / refs.folder_name_of_repo_id(repo_id)

    def blob(self, repo_id: str, name: str, size: int) -> Path:
        path = self.repo_path(repo_id) / "blobs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * size)
        return path

    def link(self, repo_id: str, commit: str, entry: str, blob_name: str) -> Path:
        path = self.repo_path(repo_id) / "snapshots" / commit / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        target = os.path.relpath(self.repo_path(repo_id) / "blobs" / blob_name, path.parent)
        path.symlink_to(target)
        return path

    def ref(self, repo_id: str, commit: str, name: str = "main") -> Path:
        path = self.repo_path(repo_id) / "refs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commit)
        return path

    def revision(self, repo_id: str, commit: str, files: dict[str, int], ref: str | None = "main") -> None:
        """Add a revision whose entries each get a blob of their own."""
        for entry, size in files.items():
            blob_name = f"{commit[:6]}-{Path(entry).name}"
            self.blob(repo_id, blob_name, size)
            self.link(repo_id, commit, entry, blob_name)
        if ref is not None:
            self.ref(repo_id, commit, ref)

    def bytes_on_disk(self) -> int:
        """Bytes held by every regular file in the cache."""
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        return total


@pytest.fixture
def fake_cache(tmp_path: Path) -> FakeCache:
    return FakeCache(tmp_path / "cache")


COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
