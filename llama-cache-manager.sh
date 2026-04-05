#!/usr/bin/env bash

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DEFAULT_CACHE_DIR="${LLAMA_CACHE:-${HF_HOME:-$HOME/.cache}/llama.cpp}"
DEFAULT_CACHE_FALLBACK="${HF_HOME:-$HOME/.cache}/llama.cpp"
DEFAULT_CACHE_HELP_FALLBACK="${DEFAULT_CACHE_FALLBACK/#$HOME/\~}"
DEFAULT_CACHE_HELP="LLAMA_CACHE or $DEFAULT_CACHE_HELP_FALLBACK"

handle_sigpipe() {
	exit 0
}

trap handle_sigpipe PIPE

safe_printf() {
	/usr/bin/printf "$@" 2>/dev/null || exit 0
}

string_length() {
	printf '%s' "$1" | wc -m | tr -d ' '
}

init_colors() {
	COLOR_RESET=""
	COLOR_MODEL=""
	COLOR_TOTAL_SIZE=""
	COLOR_QUANT=""
	COLOR_MMPROJ=""
	COLOR_SIZE=""

	if [ -n "${NO_COLOR:-}" ]; then
		return 0
	fi

	if [ ! -t 1 ]; then
		return 0
	fi

	COLOR_RESET=$'\033[0m'
	COLOR_MODEL=$'\033[1m'
	COLOR_TOTAL_SIZE=$'\033[36m'
	COLOR_QUANT=$'\033[34m'
	COLOR_MMPROJ=$'\033[33m'
	COLOR_SIZE=$'\033[90m'
}

paint() {
	local color="$1"
	local text="$2"

	if [ -z "$color" ]; then
		printf '%s' "$text"
	else
		printf '%s%s%s' "$color" "$text" "$COLOR_RESET"
	fi
}

usage() {
	cat <<EOF
Usage:
  $SCRIPT_NAME [OPTIONS] <command>

Options:
  -c DIR, --cache-dir DIR  Cache root. Default: $DEFAULT_CACHE_HELP
  -h, --help               Show this help

Commands:
  help                    Show global or command-specific help
  ls, list                List cached models and artifacts
  rm, remove              Remove MODEL or MODEL:QUANT from the cache
EOF
}

usage_list() {
	cat <<EOF
Usage:
  $SCRIPT_NAME {ls|list}

List cached models grouped by model, with artifact sizes and total size.
EOF
}

usage_rm() {
	cat <<EOF
Usage:
  $SCRIPT_NAME {rm|remove} [-n|--dry-run] [-f|--force] MODEL[:QUANT]

Options:
  -n, --dry-run            Print actions without deleting
  -f, --force              Delete without confirmation
EOF
}

die() {
	echo "error: $*" >&2
	exit 1
}

have_cmd() {
	command -v "$1" >/dev/null 2>&1
}

resolve_path() {
	if have_cmd realpath; then
		realpath "$1"
	else
		readlink -f "$1"
	fi
}

file_size_bytes() {
	if stat -c '%s' "$1" >/dev/null 2>&1; then
		stat -c '%s' "$1"
	else
		stat -f '%z' "$1"
	fi
}

human_size() {
	local bytes="$1"
	local units=(B KiB MiB GiB TiB)
	local unit=0
	local whole="$bytes"
	local frac=0

	while [ "$whole" -ge 1024 ] && [ "$unit" -lt 4 ]; do
		frac=$(((whole % 1024) * 10 / 1024))
		whole=$((whole / 1024))
		unit=$((unit + 1))
	done

	printf '%s.%s%s' "$whole" "$frac" "${units[$unit]}"
}

confirm() {
	local prompt="$1"

	if [ "$FORCE" -eq 1 ]; then
		return 0
	fi

	printf '%s [y/N] ' "$prompt" >&2
	read -r answer
	case "$answer" in
	y | Y | yes | YES)
		return 0
		;;
	*)
		return 1
		;;
	esac
}

delete_path() {
	local path="$1"

	if [ "$DRY_RUN" -eq 1 ]; then
		printf 'DRY-RUN rm %s\n' "$path"
	else
		rm -rf -- "$path"
		printf 'deleted %s\n' "$path"
	fi
}

delete_empty_dir() {
	local path="$1"

	if [ "$DRY_RUN" -eq 1 ]; then
		printf 'DRY-RUN rmdir %s\n' "$path"
	else
		rmdir -- "$path"
		printf 'deleted %s\n' "$path"
	fi
}

model_dir_path() {
	printf '%s/%s' "$CACHE_DIR" "$1"
}

display_model_ref() {
	local model="$1"

	if printf '%s\n' "$model" | grep -Eq '^models--[^/]+--.+$'; then
		printf '%s\n' "$model" | sed -E 's/^models--([^/]+)--(.+)$/\1\/\2/'
	else
		printf '%s\n' "$model"
	fi
}

normalize_model_ref() {
	local ref="$1"

	if printf '%s\n' "$ref" | grep -Eq '^models--[^/]+--.+$'; then
		printf '%s\n' "$ref"
		return 0
	fi

	if printf '%s\n' "$ref" | grep -Eq '^[^/]+/.+$'; then
		printf 'models--%s\n' "$(printf '%s\n' "$ref" | sed 's/\//--/')"
		return 0
	fi

	die "invalid model reference: $ref"
}

split_model_quant_ref() {
	local ref="$1"
	local model_part quant_part

	if printf '%s\n' "$ref" | grep -q ':'; then
		model_part="${ref%%:*}"
		quant_part="${ref#*:}"
	else
		model_part="$ref"
		quant_part=""
	fi

	printf '%s\n%s\n' "$model_part" "$quant_part"
}

artifact_label() {
	local file_name="$1"
	local quant

	quant="$(printf '%s\n' "$file_name" | sed -E 's/.*-((IQ|Q|BF)[A-Z0-9_]+)\.gguf$/\1/; t; s/^.*$/-/')"

	case "$file_name" in
	mmproj-*.gguf)
		printf 'mmproj %s\n' "$quant"
		;;
	*)
		if [ "$quant" = "-" ]; then
			printf '%s\n' "$file_name"
		else
			printf '%s\n' "$quant"
		fi
		;;
	esac
}

artifact_color() {
	local label="$1"

	case "$label" in
	mmproj*)
		printf '%s' "$COLOR_MMPROJ"
		;;
	*)
		printf '%s' "$COLOR_QUANT"
		;;
	esac
}

artifact_sort_key() {
	local label="$1"

	case "$label" in
	mmproj*)
		printf '1|%s\n' "$label"
		;;
	*)
		printf '0|%s\n' "$label"
		;;
	esac
}

list_models() {
	local model_dir snapshot_dir model_ref
	local entry_name entry_path blob_path size_bytes size_h label
	local model_total_bytes total_h
	local first_model=1
	local max_width label_width
	local artifact_prefix artifact_text padded_text
	local total_size_text line_size_text
	local -a row_kind row_label row_size
	local -a model_artifact_label model_artifact_size model_artifact_sort
	local row_count=0
	local model_artifact_count

	max_width=0

	while read -r model_dir; do
		snapshot_dir="$model_dir/snapshots"
		[ -d "$snapshot_dir" ] || continue
		model_ref="$(display_model_ref "$(basename "$model_dir")")"
		model_total_bytes=0
		model_artifact_count=0

		label_width="$(string_length "$model_ref")"
		if [ "$label_width" -gt "$max_width" ]; then
			max_width="$label_width"
		fi

		while read -r entry_path; do
			entry_name="$(basename "$entry_path")"
			blob_path="$(resolve_path "$entry_path")"
			size_bytes="$(file_size_bytes "$blob_path")"
			size_h="$(human_size "$size_bytes")"
			model_total_bytes=$((model_total_bytes + size_bytes))
			label="$(artifact_label "$entry_name")"
			artifact_text="  - $label"
			label_width="$(string_length "$artifact_text")"
			if [ "$label_width" -gt "$max_width" ]; then
				max_width="$label_width"
			fi
			model_artifact_label[$model_artifact_count]="$label"
			model_artifact_size[$model_artifact_count]="$size_h"
			model_artifact_sort[$model_artifact_count]="$(artifact_sort_key "$label")"
			model_artifact_count=$((model_artifact_count + 1))
		done < <(find "$snapshot_dir" -mindepth 2 -maxdepth 2 -type l -name '*.gguf' | sort)

		total_h="$(human_size "$model_total_bytes")"
		row_kind[$row_count]="M"
		row_label[$row_count]="$model_ref"
		row_size[$row_count]="$total_h"
		row_count=$((row_count + 1))

		local j sorted_indexes
		sorted_indexes="$(
			for ((j = 0; j < model_artifact_count; j++)); do
				printf '%s\t%s\n' "${model_artifact_sort[$j]}" "$j"
			done | sort | cut -f2
		)"

		while read -r j; do
			[ -n "$j" ] || continue
			row_kind[$row_count]="A"
			row_label[$row_count]="${model_artifact_label[$j]}"
			row_size[$row_count]="${model_artifact_size[$j]}"
			row_count=$((row_count + 1))
		done <<<"$sorted_indexes"

		row_kind[$row_count]="B"
		row_label[$row_count]=""
		row_size[$row_count]=""
		row_count=$((row_count + 1))
	done < <(find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -type d -name 'models--*' | sort)

	local i kind field1 field2
	for ((i = 0; i < ${#row_kind[@]}; i++)); do
		kind="${row_kind[$i]}"
		field1="${row_label[$i]}"
		field2="${row_size[$i]}"

		case "$kind" in
		A)
			artifact_prefix="  - $field1"
			line_size_text="[$(printf '%8s' "$field2")]"
			padded_text="$(printf "%-*s" "$max_width" "$artifact_prefix")"
			safe_printf "%s %s\n" \
				"$(paint "$(artifact_color "$field1")" "$padded_text")" \
				"$(paint "$COLOR_SIZE" "$line_size_text")"
			;;
		M)
			if [ "$first_model" -eq 0 ]; then
				safe_printf '\n'
			fi
			first_model=0
			total_size_text="[$(printf '%8s' "$field2")]"
			padded_text="$(printf "%-*s" "$max_width" "$field1")"
			safe_printf "%s %s\n" \
				"$(paint "$COLOR_MODEL" "$padded_text")" \
				"$(paint "$COLOR_TOTAL_SIZE" "$total_size_text")"
			;;
		B) ;;
		esac
	done
}

ensure_model_exists() {
	local model="$1"
	[ -d "$(model_dir_path "$model")" ] || die "model not found: $(display_model_ref "$model")"
}

count_blob_refs() {
	local model="$1"
	local blob_path="$2"
	local skip_path="${3:-}"

	find "$(model_dir_path "$model")/snapshots" -type l -exec sh -c '
    target="$1"
    link="$2"
    skip="$3"
    if [ -n "$skip" ] && [ "$link" = "$skip" ]; then
      exit 0
    fi
    if [ "$(readlink -f "$link")" = "$target" ]; then
      printf x
    fi
  ' sh "$blob_path" {} "$skip_path" \; | wc -c | tr -d ' '
}

delete_blob_if_unreferenced() {
	local model="$1"
	local blob_path="$2"
	local skip_path="${3:-}"
	local refs

	refs="$(count_blob_refs "$model" "$blob_path" "$skip_path")"

	if [ "$refs" -eq 0 ] && [ -e "$blob_path" ]; then
		delete_path "$blob_path"
	fi
}

prune_empty_dirs() {
	local model="$1"
	local model_path
	local snapshot_dir

	model_path="$(model_dir_path "$model")"

	while read -r snapshot_dir; do
		[ -n "$snapshot_dir" ] || continue
		delete_empty_dir "$snapshot_dir"
	done < <(find "$model_path/snapshots" -mindepth 1 -type d -empty 2>/dev/null | sort -r)

	if ! find "$model_path/snapshots" -type l | grep -q .; then
		if confirm "Delete empty model directory $model_path?"; then
			delete_path "$model_path"
		fi
	fi
}

rm_quant_ref() {
	local ref="$1"
	local model="$1"
	local quant=""
	local matches=0
	local link_path blob_path entry_name
	local model_quant_ref model_ref

	model_quant_ref="$(split_model_quant_ref "$ref")"
	model_ref="$(printf '%s\n' "$model_quant_ref" | sed -n '1p')"
	quant="$(printf '%s\n' "$model_quant_ref" | sed -n '2p')"
	[ -n "$quant" ] || die "rm requires MODEL or MODEL:QUANT"
	model="$model_ref"

	model="$(normalize_model_ref "$model")"
	ensure_model_exists "$model"

	find "$(model_dir_path "$model")/snapshots" -type l -name '*.gguf' | sort | while read -r link_path; do
		entry_name="$(basename "$link_path")"
		if printf '%s\n' "$entry_name" | grep -Eq -- "-${quant}\.gguf$"; then
			printf '%s\n' "$link_path"
		fi
	done >/tmp/llama-cache-manager.matches.$$

	matches="$(wc -l </tmp/llama-cache-manager.matches.$$ | tr -d ' ')"
	[ "$matches" -gt 0 ] || {
		rm -f /tmp/llama-cache-manager.matches.$$
		die "quant not found in model: $quant"
	}

	if confirm "Delete $matches file(s) for quant $quant from $(display_model_ref "$model")?"; then
		while read -r link_path; do
			blob_path="$(resolve_path "$link_path")"
			delete_path "$link_path"
			delete_blob_if_unreferenced "$model" "$blob_path" "$link_path"
		done </tmp/llama-cache-manager.matches.$$
		prune_empty_dirs "$model"
	fi

	rm -f /tmp/llama-cache-manager.matches.$$
}

rm_model_ref() {
	local ref="$1"
	local model
	local model_path

	model="$(normalize_model_ref "$ref")"
	ensure_model_exists "$model"
	model_path="$(model_dir_path "$model")"

	if confirm "Delete full model cache $(display_model_ref "$model")?"; then
		delete_path "$model_path"
	fi
}

rm_ref() {
	local ref="$1"
	local parts model quant

	parts="$(split_model_quant_ref "$ref")"
	model="$(printf '%s\n' "$parts" | sed -n '1p')"
	quant="$(printf '%s\n' "$parts" | sed -n '2p')"

	if [ -n "$quant" ]; then
		rm_quant_ref "$ref"
	else
		rm_model_ref "$model"
	fi
}

CACHE_DIR="$DEFAULT_CACHE_DIR"
FORCE=0
DRY_RUN=0
init_colors

while [ "$#" -gt 0 ]; do
	case "$1" in
	-c | --cache-dir)
		shift
		[ "$#" -gt 0 ] || die "$1 requires a value"
		CACHE_DIR="$1"
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		break
		;;
	esac
	shift
done

[ -d "$CACHE_DIR" ] || die "cache directory not found: $CACHE_DIR"
[ "$#" -gt 0 ] || {
	usage
	exit 1
}

case "$1" in
help)
	case "${2:-}" in
	"" | -h | --help)
		usage
		;;
	ls | list)
		usage_list
		;;
	rm | remove)
		usage_rm
		;;
	*)
		die "unknown command: $2"
		;;
	esac
	;;
ls | list)
	case "${2:-}" in
	"")
		list_models
		;;
	-h | --help)
		[ "$#" -eq 2 ] || die "list --help takes no additional arguments"
		usage_list
		;;
	*)
		die "list takes no additional arguments"
		;;
	esac
	;;
rm | remove)
	shift
	while [ "$#" -gt 0 ]; do
		case "$1" in
		-n | --dry-run)
			DRY_RUN=1
			;;
		-f | --force)
			FORCE=1
			;;
		-h | --help)
			usage_rm
			exit 0
			;;
		*)
			break
			;;
		esac
		shift
	done

	[ "$#" -eq 1 ] || die "rm requires MODEL or MODEL:QUANT"
	rm_ref "$1"
	;;
-h | --help)
	usage
	;;
*)
	die "unknown command: $1"
	;;
esac
