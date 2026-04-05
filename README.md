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

Commands:
  completions             Print shell completion script to stdout
  help                    Show global or command-specific help
  ls, list                List cached models and artifacts
  rm, remove              Remove MODEL or MODEL:QUANT from the cache
```

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
llama-cache-manager ls unsloth
llama-cache-manager ls :UD-Q4_K_XL
llama-cache-manager completions bash
llama-cache-manager rm -f unsloth/gpt-oss-20b-GGUF
llama-cache-manager remove --dry-run unsloth/Qwen3.5-9B-GGUF:Q4_K_XL
llama-cache-manager -c /srv/llama-models list
llama-cache-manager help rm
```

### Color

Color is enabled only on TTYs and can be disabled with `NO_COLOR=1`.

### Remove Semantics

Supported remove targets:

- `rm MODEL`: Removes the whole model with all quants
- `rm MODEL:QUANT`: Removes a single quant

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

That matters for deletion:

- the tool removes snapshot symlinks first
- then removes blobs only if they are no longer referenced

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
