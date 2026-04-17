# llama-cache-manager

Small Bash CLI for inspecting and deleting GGUF models from a `llama.cpp` / Hugging Face cache-style model store.
It uses the same model reference format as llama.cpp's `-hf` option:
`org/repo[:quant]`

- `unsloth/gpt-oss-20b-GGUF`
- `unsloth/Qwen3.5-9B-GGUF:Q4_K_XL`

## Usage

```text
Usage:
  llama-cache-manager [OPTIONS] <command>

Options:
  -c DIR, --cache-dir DIR  Cache root. Default: LLAMA_CACHE or ~/.cache/llama.cpp
  -h, --help               Show this help
  -V, --version            Show version

Commands:
  completions             Print shell completion script to stdout
  help                    Show global or command-specific help
  ls, list                List cached models, artifacts, and unreferenced blobs
  prune                   Remove old snapshots and unreferenced blobs
  rm, remove              Remove MODEL or MODEL:QUANT from the cache
```

Running `llama-cache-manager` without a command is equivalent to `llama-cache-manager ls`.
Artifact rows include the local cache age derived from the blob mtime, for example `cached 7 days ago (on 2026-04-07 15:23)`. Unreferenced files left behind in `blobs/` are listed separately under each model.

### Shell Completions

Print completion scripts to stdout:

Examples:

```text
source <(llama-cache-manager completions bash)
llama-cache-manager completions zsh > ~/.zfunc/_llama-cache-manager
llama-cache-manager completions fish > ~/.config/fish/completions/llama-cache-manager.fish
```

The repository also contains the raw completion files under
[`completions/`](./completions).

### Examples

```text
llama-cache-manager ls
llama-cache-manager --version
llama-cache-manager ls unsloth
llama-cache-manager ls :UD-Q4_K_XL
llama-cache-manager completions bash
llama-cache-manager prune
llama-cache-manager prune --until "7 days"
llama-cache-manager prune --dry-run --until 3days
llama-cache-manager rm -f unsloth/gpt-oss-20b-GGUF
llama-cache-manager remove --dry-run unsloth/Qwen3.5-9B-GGUF:Q4_K_XL
llama-cache-manager -c /srv/llama-models list
llama-cache-manager help rm
```

### Color

Color is enabled only on TTYs and can be disabled with `NO_COLOR=1`.

### Remove Semantics

Supported remove targets:

- `rm MODEL`: Removes the whole model with all quants and leftover blob files
- `rm MODEL:QUANT`: Removes a single quant and also cleans unreferenced blobs in that model

### Prune Semantics

`prune` without `--until` keeps only the newest snapshot revision per model and
removes older revisions. It also removes unreferenced files from `blobs/`.

`prune --until SPEC` removes snapshot revisions whose newest referenced blob is
older than the cutoff. Unreferenced files from `blobs/` are still included as prune candidates.

Accepted cutoff formats include relative ages such as:

- `7 days`
- `3days`
- `12h`

Absolute timestamps such as `2026-04-01` or `2026-04-01 12:30` are also accepted.

### List Filters

`ls` accepts optional filters.

- plain text matches against `org/repo`
- `:QUANT` matches against quant labels
- `org/repo:QUANT` matches the full reference

Multiple filters are combined with OR.

## Technical Notes
### Cache Layout

The tool expects a cache layout like this:

```text
<cache-root>/
  models--<org>--<repo>/
    blobs/
    refs/
    snapshots/<revision>/*.gguf
```

The `.gguf` files under `snapshots/` are symlinks into `blobs/`.
Regular files in `blobs/` without any snapshot symlink are treated as unreferenced leftovers and surfaced explicitly.

That matters for deletion:

- the tool removes snapshot symlinks first
- then removes blobs only if they are no longer referenced
- and it can clean unreferenced blob files that were never linked from a snapshot

This avoids naive file deletion that would leave the cache in an inconsistent state.

### Defaults

Cache root resolution:

1. `LLAMA_CACHE`, if set
2. otherwise `${HF_HOME:-$HOME/.cache}/llama.cpp`

### Notes For Future Changes

- This tool intentionally works at two user-facing levels only:
  model and quant within a model.
- It does not expose direct removal of arbitrary artifact names.
- `mmproj` is listed because it is relevant for multimodal models, but it is not currently an addressable `rm` target.
- Quant detection is filename-based and currently assumes names
  like `...-Q4_K_XL.gguf` or `...-BF16.gguf`.
- If naming conventions change, the parsing logic in
  [`llama-cache-manager`](./llama-cache-manager) will
  need to be updated.
- If multiple snapshots of the same model are present,
  blob deletion remains reference-aware within that model
  directory. The tool does not rewrite `refs`.
