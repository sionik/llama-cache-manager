function __lcm_command
    set -l tokens (commandline -opc)
    for token in $tokens
        switch $token
            case help ls list prune rm remove
                echo $token
                return 0
        end
    end
    return 1
end

function __lcm_no_command
    not __lcm_command >/dev/null
end

function __lcm_using_command
    set -l current (__lcm_command)
    test "$current" = "$argv[1]"
end

for cmd in llama-cache-manager
    complete -c $cmd -n '__lcm_no_command' -s c -l cache-dir -r -d 'Cache root'
    complete -c $cmd -n '__lcm_no_command' -s h -l help -d 'Show help'
    complete -c $cmd -n '__lcm_no_command' -s V -l version -d 'Show version'
    complete -c $cmd -n '__lcm_no_command' -a 'help ls list prune rm remove'

    complete -c $cmd -n '__lcm_using_command help' -a 'help ls list prune rm remove'

    complete -c $cmd -n '__lcm_using_command ls' -s h -l help -d 'Show help'
    complete -c $cmd -n '__lcm_using_command list' -s h -l help -d 'Show help'
    complete -c $cmd -n '__lcm_using_command ls; and not string match -qr "^-.*" -- (commandline -ct)' -a "($cmd __complete filters 2>/dev/null)"
    complete -c $cmd -n '__lcm_using_command list; and not string match -qr "^-.*" -- (commandline -ct)' -a "($cmd __complete filters 2>/dev/null)"

    complete -c $cmd -n '__lcm_using_command prune' -s n -l dry-run -d 'Print actions without deleting'
    complete -c $cmd -n '__lcm_using_command prune' -s f -l force -d 'Delete without confirmation'
    complete -c $cmd -n '__lcm_using_command prune' -l until -r -d 'Delete snapshots older than cutoff; default keeps newest per model'
    complete -c $cmd -n '__lcm_using_command prune' -s h -l help -d 'Show help'

    complete -c $cmd -n '__lcm_using_command rm' -s n -l dry-run -d 'Print actions without deleting'
    complete -c $cmd -n '__lcm_using_command rm' -s f -l force -d 'Delete without confirmation'
    complete -c $cmd -n '__lcm_using_command rm' -s h -l help -d 'Show help'
    complete -c $cmd -n '__lcm_using_command remove' -s n -l dry-run -d 'Print actions without deleting'
    complete -c $cmd -n '__lcm_using_command remove' -s f -l force -d 'Delete without confirmation'
    complete -c $cmd -n '__lcm_using_command remove' -s h -l help -d 'Show help'
    complete -c $cmd -n '__lcm_using_command rm; and not string match -qr "^-.*" -- (commandline -ct)' -a "($cmd __complete rm 2>/dev/null)"
    complete -c $cmd -n '__lcm_using_command remove; and not string match -qr "^-.*" -- (commandline -ct)' -a "($cmd __complete rm 2>/dev/null)"
end
