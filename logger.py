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
