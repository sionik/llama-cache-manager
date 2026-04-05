_llama_cache_manager() {
	local cur prev cmd i skip_next
	local -a suggestions

	cur="${COMP_WORDS[COMP_CWORD]}"
	prev="${COMP_WORDS[COMP_CWORD-1]}"
	cmd=""
	skip_next=0

	for ((i = 1; i < COMP_CWORD; i++)); do
		if [ "$skip_next" -eq 1 ]; then
			skip_next=0
			continue
		fi

		case "${COMP_WORDS[i]}" in
		-c|--cache-dir)
			skip_next=1
			;;
		help|ls|list|rm|remove)
			cmd="${COMP_WORDS[i]}"
			break
			;;
		esac
	done

	if [ "$prev" = "-c" ] || [ "$prev" = "--cache-dir" ]; then
		COMPREPLY=($(compgen -d -- "$cur"))
		return 0
	fi

	if [ -z "$cmd" ]; then
		suggestions=(-c --cache-dir -h --help help ls list rm remove)
		COMPREPLY=($(compgen -W "${suggestions[*]}" -- "$cur"))
		return 0
	fi

	case "$cmd" in
	help)
		COMPREPLY=($(compgen -W "help ls list rm remove" -- "$cur"))
		;;
	ls|list)
		if [[ "$cur" == -* ]]; then
			COMPREPLY=($(compgen -W "-h --help" -- "$cur"))
		else
			COMPREPLY=($(compgen -W "$("${COMP_WORDS[0]}" __complete filters 2>/dev/null)" -- "$cur"))
		fi
		;;
	rm|remove)
		if [[ "$cur" == -* ]]; then
			COMPREPLY=($(compgen -W "-n --dry-run -f --force -h --help" -- "$cur"))
		else
			COMPREPLY=($(compgen -W "$("${COMP_WORDS[0]}" __complete rm 2>/dev/null)" -- "$cur"))
		fi
		;;
	esac
}

complete -F _llama_cache_manager llama-cache-manager
