# Contributing

## Scope

- Keep the tool small and model-centric.
- One runtime dependency, `huggingface_hub`, which walks the cache. Anything
  else needs a reason.
- Keep the output useful in plain text first. Colour is optional, and
  `NO_COLOR` and a pipe both turn it off.
- Every deletion is a plan that is printed before it runs. Nothing removes a
  file without a preview and a question.

## Layout

| `llama_cache_manager/` | Holds |
| ------ | ----- |
| `refs.py` | the `org/repo:quant` grammar and the quant label of a file |
| `cache.py` | the scan, and the repository, revision and artifact model |
| `removal.py` | plans, what they reclaim, and carrying them out |
| `age.py` | reading `--until` and writing the age column |
| `display.py` | the listing, the preview and the JSON output |
| `cli.py` | the commands |

## Running from the working tree

Inside the repository, `python -m llama_cache_manager` works as it stands. From
anywhere else, put the repository on the import path rather than changing the
working directory, so that a relative `--cache-dir` still means what the caller
meant:

```text
PYTHONPATH=/path/to/llama-cache-manager python -m llama_cache_manager ls
```

The module route names itself `llama-cache-manager`, so the help text and the
completion scripts match the installed console script.

## Changes

- Write the test first and watch it fail.
- Update `README.md` when the behaviour changes.
- `ruff format` and `ruff check` before committing.

## Testing

```text
python -m pytest
ruff check llama_cache_manager tests
ruff format --check llama_cache_manager tests
```

The tests build a cache directory on disk, so they cover the real layout,
symlinks and shared blobs included. Anything that deletes gets a test that
compares the reclaimed size the plan promised against the bytes the file system
actually gave back.

Worth checking by hand against a real cache before a release:

```text
llama-cache-manager ls
llama-cache-manager ls :SOME-QUANT
llama-cache-manager ls --json
llama-cache-manager rm -n org/repo
llama-cache-manager rm -n org/repo:quant
llama-cache-manager prune -n
llama-cache-manager prune -n --until 30d
source <(llama-cache-manager completions bash)
```
