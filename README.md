# llama-cache-manager

Fetch, inspect and delete GGUF models in a `llama.cpp` or Hugging Face model
cache.

Models are named the way huggingface.co and llama.cpp's `-hf` option name them,
as `org/repo:quant`, so a reference can be copied between the three without
translation:

```text
unsloth/Qwen3.8-27B-GGUF
unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

The quant is the part of the file name that is not the repository name, which is
the same string the hub shows for that file. Every shard of a split file shares
one reference, so a model in three parts is one artifact.

## Listing

`ls` is the default command, so the bare program lists everything:

```text
$ llama-cache-manager
unsloth/gemma-4-12B-it-GGUF           10.1 GiB  1 revision
  fc034cf  main  27 days ago
    :UD-Q6_K_XL    10.0 GiB
    :mmproj-BF16  167.0 MiB

unsloth/Muse-Glimmer-30B-GGUF         16.7 GiB  2 revisions
  1afeb8e  detached  10 days ago  shares all blobs
    :UD-Q4_K_XL    14.8 GiB
    :mmproj-Q8_0    1.9 GiB
  faa5b02  main  10 days ago
    :UD-Q4_K_XL    14.8 GiB
    :mmproj-Q8_0    1.9 GiB

67.8 GiB in 5 repositories, 6 revisions
prune would remove 1 detached revision, reclaiming nothing
```

Read a reference down the tree: the repository on the first line, the `:quant`
on the artifact line. The size next to a repository counts each file once, even
when two revisions share it, which is why removing one of the two revisions
above reclaims nothing.

`detached` marks a revision that no name points at any more. Nothing can select
it, so `prune` offers it first.

Any part of a reference works as a filter, and the sizes then describe the rows
shown:

```text
llama-cache-manager unsloth              every repository of one org
llama-cache-manager ls :UD-Q4_K_XL       one quant, wherever it is
llama-cache-manager ls unsloth/gemma-4-12B-it-GGUF
llama-cache-manager ls --sort size       biggest first
llama-cache-manager ls --sort age -r     oldest first
llama-cache-manager ls --brief           one line per repository
llama-cache-manager ls --json            byte counts and timestamps, unformatted
```

`--sort` takes `name`, `size` or `age`. `name` runs first letter first, the
other two start with the largest and the newest, and `-r`/`--reverse` turns
that around. `--brief` drops to one line per repository.

## Fetching

`pull` fetches one quant from the hub into the same cache the other commands
read:

```text
$ llama-cache-manager pull unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
About to download into /home/simon/.cache/llama.cpp

unsloth/Qwen3.8-27B-GGUF  main  1a2b3c4
  :UD-Q4_K_XL  14.8 GiB  2 shards

Transfers 14.8 GiB

Download these? [y/N]:
```

The quant has to be named. The hub cannot be searched from here, and the quants
of one repository differ by tens of gigabytes, so a reference without one comes
back with the choices named:

```text
$ llama-cache-manager pull unsloth/Qwen3.8-27B-GGUF
Error: unsloth/Qwen3.8-27B-GGUF needs a quant to fetch; it offers UD-Q4_K_XL, mmproj-BF16
```

What the plan calls a transfer is what has to come over the network. A file the
cache already holds is not fetched again, so pulling the projector next to a
model it shares files with reports only what is new:

```text
llama-cache-manager pull -n unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL   what it would cost
llama-cache-manager pull --revision v1.2 unsloth/Qwen3.8-27B-GGUF:Q4_K_M
llama-cache-manager pull -y unsloth/Qwen3.8-27B-GGUF:mmproj-BF16  do not ask
```

A gated or private repository needs a token, which `huggingface_hub` reads from
`HF_TOKEN` or from the file `hf auth login` writes.

## Removing

Every command that deletes prints the plan first, then asks. `--dry-run` prints
the plan and stops. `--yes` skips the question.

```text
llama-cache-manager rm unsloth/Qwen3.8-27B-GGUF                 the repository
llama-cache-manager rm unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL      one quant
llama-cache-manager rm -n unsloth/Qwen3.8-27B-GGUF:mmproj-BF16  plan only
llama-cache-manager rm --until 30d unsloth/Qwen3.8-27B-GGUF     old revisions only
```

A shorter word is accepted while it fits one repository, and named candidates
come back when it fits several. `remove` and `list` work as the older names of
`rm` and `ls`, and `help COMMAND` prints the help of one command.

`--until` holds the removal to the revisions older than the cutoff, which is
`prune --until` restricted to the repositories named.

`--json` prints the plan as data. Since nothing is watching, it removes only
with `-y`, and `-n` gives the plan alone. The fields are written down in
[`docs/json-schema.md`](docs/json-schema.md).

What the plan says it reclaims is what the file system gives back. A blob is
counted only when every entry that points at it is going away, so deleting a
quant that another revision still uses reports that it reclaims nothing.

## Pruning

`prune` removes what nothing can reach any more:

- revisions no name points at
- blobs no revision points at
- interrupted downloads

None of these can be selected by `org/repo`, so removing them takes nothing
away.

`--until` goes further and reaches revisions that are still in use:

```text
llama-cache-manager prune                     only unreachable files
llama-cache-manager prune -n                  what that would be
llama-cache-manager prune --until 30d         also anything older than 30 days
llama-cache-manager prune --until "2 weeks"
llama-cache-manager prune --until 2026-07-01
```

Ages come from the modification time of the file, which is when it was
downloaded. Access times are not used, because mount options such as
`relatime` make them unreliable.

## Where the cache is

In order: `--cache-dir`, `$LLAMA_CACHE`, `$HF_HOME/llama.cpp` when it exists,
`~/.cache/llama.cpp` when it exists, then the Hugging Face hub cache, which
follows `$HF_HUB_CACHE` and `$HF_HOME`.

`pull` writes into the cache the other commands read, so a pull and the listing
that follows it cannot land in different places. The one case where it differs
is a machine with no cache at all: there `pull` creates `~/.cache/llama.cpp`
rather than the hub cache, because that is the one `llama-server -hf` reads.

## Install

```text
pipx install llama-cache-manager
```

or from a clone:

```text
pipx install .
```

The only dependency is `huggingface_hub`, which does the cache walk and the
transfers, and the `click` it brings along.

## Shell completions

The script is generated by the program, so nothing has to be installed
alongside it:

```text
source <(llama-cache-manager completions bash)
llama-cache-manager completions zsh  > ~/.zfunc/_llama-cache-manager
llama-cache-manager completions fish > ~/.config/fish/completions/llama-cache-manager.fish
```

Completion offers references, `org/repo` and `org/repo:quant` alike, read from
the cache as you type.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | done |
| 1 | error |
| 2 | wrong usage |
| 3 | nothing matched, or no cache directory (also with `--json`) |
| 4 | cancelled at the prompt |

## Documentation

`man llama-cache-manager` once installed, or `man -l man/llama-cache-manager.1`
from a clone. The JSON fields are in [`docs/json-schema.md`](docs/json-schema.md).

## Tests

```text
python -m pytest
ruff check llama_cache_manager tests
ruff format --check llama_cache_manager tests
```
