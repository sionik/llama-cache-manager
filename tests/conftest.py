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


class FakeHub:
    """A hub holding a few repositories, standing in for the network.

    ``repos`` maps ``org/repo`` to ``{"commit": hash, "files": {name: size}}``.
    Every call is recorded, so a test can say what was asked for as well as
    what came back. With a ``cache`` given, :meth:`fetch` writes the file into
    that cache the way a real download would, which lets a test pull and then
    list.
    """

    def __init__(self, repos: dict, cache: FakeCache | None = None, cached=()) -> None:
        self.repos = repos
        self.cache = cache
        self.cached = set(cached)
        self.fetched: list[tuple[str, str]] = []
        self.listed: list[str] = []
        self.reachable = True

    def _repo(self, repo_id: str, revision: str) -> dict:
        from llama_cache_manager.download import DownloadError

        if not self.reachable:
            raise DownloadError(f"cannot reach the hub for {repo_id}: no route to host")
        if repo_id not in self.repos:
            raise DownloadError(f"no repository {repo_id} on the hub")
        return self.repos[repo_id]

    def file_names(self, repo_id: str, revision: str) -> tuple[str, ...]:
        files = self._repo(repo_id, revision)["files"]
        self.listed.append(repo_id)
        return tuple(files)

    def inspect(self, repo_id: str, name: str, revision: str, cache_dir: Path):
        from llama_cache_manager.download import FileStatus

        repo = self._repo(repo_id, revision)
        return FileStatus(
            name=name,
            size=repo["files"][name],
            commit=repo["commit"],
            cached=name in self.cached or (repo_id, name) in self.fetched,
        )

    def fetch(self, repo_id: str, name: str, revision: str, cache_dir: Path) -> Path:
        repo = self._repo(repo_id, revision)
        self.fetched.append((repo_id, name))
        if self.cache is None:
            return cache_dir / name
        blob_name = f"{repo['commit'][:6]}-{Path(name).name}"
        self.cache.blob(repo_id, blob_name, repo["files"][name])
        path = self.cache.link(repo_id, repo["commit"], name, blob_name)
        self.cache.ref(repo_id, repo["commit"])
        return path
