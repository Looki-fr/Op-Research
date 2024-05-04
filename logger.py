from __future__ import print_function
import pathlib
import builtins as __builtin__


class Settings:
    """A class to store global variables."""
    endfiles: str = []
    outfile: str = ""
    path = pathlib.Path(__file__).parent
    verbose: bool = False
    debug: bool = False

    # Colors
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    MAGENTA = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def clear_color():
    """Set colors to empty string."""
    Settings.RED = Settings.GREEN = Settings.YELLOW = Settings.BLUE = Settings.MAGENTA = Settings.CYAN = Settings.WHITE = Settings.BOLD = Settings.UNDERLINE = Settings.RESET = ""


def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    if Settings.debug:
        if Settings.outfile in Settings.endfiles:
            with open(Settings.path / Settings.outfile, "a") as f:
                sep = kwargs.get("sep", " ")
                end = kwargs.get("end", "\n")
                f.write(sep.join(map(str, args)) + end)
        else:
            with open(Settings.path / Settings.outfile, "w+") as f:
                sep = kwargs.get("sep", " ")
                end = kwargs.get("end", "\n")
                f.write(sep.join(map(str, args)) + end)
            Settings.endfiles.append(Settings.outfile)
        return __builtin__.print(*args, **kwargs)
    else:
        return __builtin__.print(*args, **kwargs)


def vprint(*args, **kwargs):
    """My custom verbose print() function."""
    if Settings.verbose:
        # add color to verbose print
        print(Settings.BLUE, end="")
        _ = print(*args, **kwargs)
        print(Settings.RESET, end="")
        return _
    else:
        return None
