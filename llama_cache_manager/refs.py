"""The reference grammar of the tool: ``org/repo:quant``.

This is the form shown on huggingface.co and accepted by llama.cpp's ``-hf``
option, so a reference can be copied between the three without translation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CACHE_PREFIX = "models--"


class ReferenceError(ValueError):
    """A reference the grammar cannot read."""


@dataclass(frozen=True, slots=True)
class Reference:
    """One reference, as typed by the user.

    ``repo_id`` is set when the text names a repository (``org/repo`` or the
    on-disk ``models--org--repo``). ``quant`` is set when the text carries a
    ``:quant`` part. ``text`` holds a bare word, which names no part in
    particular and matches against all of them.
    """

    raw: str
    repo_id: str | None = None
    quant: str | None = None
    text: str | None = None

    def __str__(self) -> str:
        return self.raw


def parse(raw: str) -> Reference:
    """Read one reference.

    Accepted forms::

        unsloth/Qwen3.8-27B-GGUF                 a repository
        unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL      one artifact in a repository
        models--unsloth--Qwen3.8-27B-GGUF        the on-disk spelling
        :UD-Q4_K_XL                              a quant, in any repository
        qwen                                     a bare word, matches anything

    Raises:
        ReferenceError: the text is empty, carries more than one ``:``, or
            names an empty part.
    """
    if not raw or not raw.strip():
        raise ReferenceError("empty reference")

    head, sep, quant = raw.partition(":")
    if sep and ":" in quant:
        raise ReferenceError(f"more than one ':' in reference: {raw!r}")
    if sep and not quant:
        raise ReferenceError(f"reference ends in ':' with no quant: {raw!r}")

    quant_part = quant or None

    if not head:
        return Reference(raw=raw, quant=quant_part)

    if head.startswith(CACHE_PREFIX):
        return Reference(raw=raw, repo_id=repo_id_of_folder(head), quant=quant_part)

    if "/" in head:
        org, _, name = head.partition("/")
        if not org or not name or "/" in name:
            raise ReferenceError(f"expected org/repo, got {head!r} in reference {raw!r}")
        return Reference(raw=raw, repo_id=head, quant=quant_part)

    if quant_part:
        # A quant was named, so the head can only be the repository.
        raise ReferenceError(f"expected org/repo before ':', got {head!r} in reference {raw!r}")
    return Reference(raw=raw, text=head)


def repo_id_of_folder(folder_name: str) -> str:
    """Turn ``models--org--repo`` into ``org/repo``.

    Raises:
        ReferenceError: the name does not carry both an org and a repo part.
    """
    rest = folder_name[len(CACHE_PREFIX) :] if folder_name.startswith(CACHE_PREFIX) else folder_name
    org, sep, name = rest.partition("--")
    if not sep or not org or not name:
        raise ReferenceError(f"not a cache folder name: {folder_name!r}")
    return f"{org}/{name}"


def folder_name_of_repo_id(repo_id: str) -> str:
    """Turn ``org/repo`` into the on-disk ``models--org--repo``."""
    return CACHE_PREFIX + repo_id.replace("/", "--")


def base_name(repo_id: str) -> str:
    """The part of a repository name that file names repeat.

    ``unsloth/gemma-4-12B-it-GGUF`` gives ``gemma-4-12B-it``. The trailing
    ``-GGUF`` marker carries no information about a single file.
    """
    name = repo_id.rpartition("/")[2]
    if name.lower().endswith("-gguf"):
        name = name[: -len("-gguf")]
    return name


_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$")
_SEPARATOR_RUN = re.compile(r"[-_.]{2,}")
_GGUF_SUFFIX = ".gguf"


def is_gguf(file_name: str) -> bool:
    """Whether a file name names a GGUF file."""
    return file_name.lower().endswith(_GGUF_SUFFIX)


def quant_of_file(repo_id: str, file_name: str) -> str:
    """The quant label that addresses ``file_name`` inside ``repo_id``.

    The label is what remains of the file name once the repository name and the
    shard counter are taken out, which is the same string huggingface.co shows
    as the quantisation of that file::

        gemma-4-12b-it-UD-Q6_K_XL.gguf              -> UD-Q6_K_XL
        mmproj-Muse-Glimmer-30B-Q8_0.gguf           -> mmproj-Q8_0
        mmproj-BF16.gguf                            -> mmproj-BF16
        mtp-gemma-4-26B-A4B-it.gguf                 -> mtp
        Big-30B-UD-Q4_K_XL-00001-of-00002.gguf      -> UD-Q4_K_XL

    All shards of one split file give the same label, so a split file is one
    artifact rather than one artifact per shard.
    """
    stem = file_name[: -len(_GGUF_SUFFIX)] if is_gguf(file_name) else file_name
    stem = _SHARD_SUFFIX.sub("", stem)
    stripped = _strip_base_name(stem, base_name(repo_id))
    return stripped or stem


def _strip_base_name(stem: str, base: str) -> str:
    """Take the repository name out of a file name stem.

    The full name is tried first, then shorter prefixes of it, because some
    files carry only part of it. ``mtp-gemma-4-26B-A4B-it`` sits in the
    repository ``gemma-4-26B-A4B-it-qat-GGUF`` and drops the ``-qat`` marker.

    A shortened prefix must keep at least two segments. A single segment out of
    a longer name is a fragment such as ``gemma``, which appears inside words
    that have nothing to do with the repository. The complete name is always
    tried, however short it is, so ``Qwen3-GGUF`` still labels its files.
    """
    for candidate in _base_name_prefixes(base):
        position = stem.lower().find(candidate.lower())
        if position < 0:
            continue
        rest = stem[:position] + stem[position + len(candidate) :]
        return _SEPARATOR_RUN.sub("-", rest).strip("-_.")
    return stem


def _base_name_prefixes(base: str) -> list[str]:
    segments = [segment for segment in base.split("-") if segment]
    if not segments:
        return []
    shortest = 1 if len(segments) == 1 else 2
    return ["-".join(segments[:count]) for count in range(len(segments), shortest - 1, -1)]
