# JSON output

`--json` prints data rather than a table. Sizes are byte counts and times are
Unix timestamps, both unformatted, so nothing has to be parsed back out of a
human string.

Every document carries `"schema"`. Version 1 is described here. A new field may
appear within a version, so read by name and ignore what you do not know. A
field that is removed or changes meaning gets a new version number.

## `ls --json`

```json
{
  "schema": 1,
  "cache_dir": "/srv/llama-models",
  "size": 72777888000,
  "repositories": [ ... ],
  "warnings": ["Repo path is not a valid HuggingFace cache directory: ..."]
}
```

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `schema` | integer | version of this document |
| `cache_dir` | string | the cache that was read, resolved |
| `size` | integer | bytes held by the repositories listed, each blob counted once |
| `repositories` | array | one entry per repository, in the order `--sort` asked for |
| `warnings` | array of string | what the scan skipped, the same text `--warnings` prints |

### repository

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `repo_id` | string | `org/repo` |
| `path` | string | the repository directory |
| `size` | integer | bytes on disk, a blob shared between revisions counted once |
| `revisions` | array | one entry per revision, newest first |
| `unreferenced_blobs` | array | blobs no revision points at, each `{path, size}` |
| `interrupted_downloads` | array | leftover `*.incomplete` files, each `{path, size}` |

Under a filter, `size` covers only the artifacts that matched, and the two blob
lists are empty unless the filter named a repository rather than a quant.

### revision

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `commit` | string | the full commit hash |
| `refs` | array of string | names pointing here, such as `["main"]` |
| `detached` | boolean | no name points here, so `org/repo` cannot reach it |
| `size` | integer | bytes of this revision, blobs shared with another revision included |
| `modified` | number | newest file in the revision, as a Unix timestamp |
| `artifacts` | array | the addressable GGUF files |

### artifact

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `reference` | string | what to pass to `rm`, as `org/repo:quant` |
| `quant` | string | the quant part of the reference |
| `kind` | string | `model`, `projector` or `extra` |
| `size` | integer | bytes, each blob counted once |
| `modified` | number | newest blob of the artifact, as a Unix timestamp |
| `shards` | integer | files making up this artifact, above 1 for a split file |
| `shared` | boolean | every blob is also reached from somewhere else, so removing this alone reclaims nothing |
| `files` | array of string | the snapshot entries |

## `rm --json` and `prune --json`

The plan, before anything is removed. With `-n` it is only printed. Without
`-n` it needs `-y` as well, because JSON output means nobody is there to answer
a prompt.

```json
{
  "schema": 1,
  "dry_run": true,
  "reclaims": 709,
  "items": [
    {
      "reason": "detached revision",
      "reference": "bbbbbbb",
      "repo_id": "unsloth/Big-30B-GGUF",
      "commit": "bbbbbbbb...",
      "quant": null,
      "nominal_size": 1000
    }
  ],
  "files": ["..."],
  "blobs": ["..."],
  "revisions": ["..."],
  "repositories": [],
  "withheld": []
}
```

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `schema` | integer | version of this document |
| `dry_run` | boolean | whether `-n` was given |
| `reclaims` | integer | bytes the file system gets back, shared blobs discounted |
| `items` | array | what was asked for, in the order it was asked |
| `files` | array of string | snapshot entries to unlink |
| `blobs` | array of string | blobs losing their last reference |
| `revisions` | array of string | snapshot directories that end up empty |
| `repositories` | array of string | repository directories that go entirely |
| `withheld` | array of string | paths the scan reported that lie outside everything the cache owns, left alone |

`reclaims` is the number to act on. An item's `nominal_size` is what the item
holds on its own, which is larger whenever a blob stays behind for somebody
else.

### item

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `reason` | string | `artifact`, `revision`, `repository`, `detached revision`, `revision older than the cutoff`, `unreferenced blob` or `interrupted download` |
| `reference` | string | the artifact reference, the short commit, or the blob name |
| `repo_id` | string | `org/repo` |
| `commit` | string or null | the revision, where the item names one |
| `quant` | string or null | the quant, where the item names one |
| `nominal_size` | integer | bytes of the item on its own |

## `pull --json`

The download, before anything is fetched. With `-n` it is only printed. Without
`-n` it needs `-y` as well, for the same reason the removal commands do.

```json
{
  "schema": 1,
  "dry_run": true,
  "cache_dir": "/srv/llama-models",
  "size": 15869711974,
  "transfer": 7301444403,
  "downloads": [
    {
      "reference": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
      "repo_id": "unsloth/Qwen3.8-27B-GGUF",
      "revision": "main",
      "commit": "1a2b3c4d...",
      "quant": "UD-Q4_K_XL",
      "kind": "model",
      "size": 15869711974,
      "transfer": 7301444403,
      "shards": 2,
      "cached": false,
      "files": [
        {
          "name": "Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf",
          "size": 8568267571,
          "cached": true
        }
      ]
    }
  ]
}
```

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `schema` | integer | version of this document |
| `dry_run` | boolean | whether `-n` was given |
| `cache_dir` | string | where the files would land, which may not exist yet |
| `size` | integer | bytes the download holds once it is in the cache |
| `transfer` | integer | bytes that have to come over the network |
| `downloads` | array | one entry per artifact, in the order the references were given |

`transfer` is the number to act on. `size` is larger whenever the cache already
holds part of the artifact.

### download

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `reference` | string | `org/repo:quant`, the same string `ls` prints |
| `repo_id` | string | `org/repo` |
| `revision` | string | what `--revision` asked for, such as `main` |
| `commit` | string | the commit that revision resolved to |
| `quant` | string | the quant part of the reference |
| `kind` | string | `model`, `projector` or `extra` |
| `size` | integer | bytes of the artifact |
| `transfer` | integer | bytes of it that are not cached yet |
| `shards` | integer | files making up this artifact, above 1 for a split file |
| `cached` | boolean | every file is there already, so this fetches nothing |
| `files` | array | one entry per file, each `{name, size, cached}` |

`name` is the path of the file inside the repository, which is what the hub
calls it.

## Exit codes

`--json` writes its document first and then exits with the usual code, so
`ls --json` that matched nothing prints `"repositories": []` and exits 3.
