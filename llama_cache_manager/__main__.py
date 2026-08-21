from .cli import main

if __name__ == "__main__":
    # Name the program as the user typed it, so that the help text and the
    # completion variable match the installed console script rather than
    # reading "python -m llama_cache_manager".
    main(prog_name="llama-cache-manager")
