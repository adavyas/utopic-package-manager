import sys

from . import _native, installer


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "setup":
        raise SystemExit(installer.setup(args[1:]))
    if args and args[0] == "run":
        args = args[1:]
    _native.main("utopic", args)
