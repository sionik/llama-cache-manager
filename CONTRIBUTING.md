# Contributing

## Scope

- Keep the tool small, model-centric, and shell-friendly.
- The project is a Bash CLI. Keep dependencies minimal and/or optional.
- Prefer portability over cleverness.
- Keep output useful in plain text first. Color is optional.
- Respect non-interactive usage and `NO_COLOR`.

## Changes

- update `README.md` when necessary
- format the script using `shfmt`
- test your changes!

## Testing

At minimum, check:

- `bash -n ./llama-cache-manager.sh`
- `./llama-cache-manager.sh --help`
- `./llama-cache-manager.sh list --help`
- `./llama-cache-manager.sh rm --help`
- `./llama-cache-manager.sh -n rm ...` is not valid
- `./llama-cache-manager.sh rm -n ...` works
- `./llama-cache-manager.sh list` against a real cache
- `./llama-cache-manager.sh rm -n MODEL`
- `./llama-cache-manager.sh rm -n MODEL:QUANT`

