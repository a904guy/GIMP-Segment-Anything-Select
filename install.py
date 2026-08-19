#!/usr/bin/env python3
"""Install (or remove) the Segment Anything plug-in for GIMP 3.

    python3 install.py            # install into every GIMP 3.x profile found
    python3 install.py --list     # show what was found, change nothing
    python3 install.py --uninstall
    python3 install.py --dir /path/to/plug-ins

This only copies the plug-in itself.  The Python environment, PyTorch and
the model weights are downloaded later, from the plug-in's own Setup
window, into a separate folder that this script never touches.
"""

import argparse
import os
import platform
import shutil
import stat
import sys

PLUGIN_NAME = "segment-anything"
SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plug-ins", PLUGIN_NAME)

# GIMP 3 releases that share the same plug-in API.
KNOWN_VERSIONS = ["3.0", "3.2", "3.4", "3.6"]


def candidate_roots():
    """Every directory GIMP 3 might read user plug-ins from."""
    home = os.path.expanduser("~")
    system = platform.system()
    roots = []
    if system == "Windows":
        appdata = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
        roots.append(os.path.join(appdata, "GIMP"))
    elif system == "Darwin":
        roots.append(os.path.join(home, "Library", "Application Support", "GIMP"))
    else:
        config = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
        roots.append(os.path.join(config, "GIMP"))
        # Flatpak keeps its own configuration tree.
        roots.append(os.path.join(home, ".var", "app", "org.gimp.GIMP", "config", "GIMP"))
        # Snap does too.
        roots.append(os.path.join(home, "snap", "gimp", "current", ".config", "GIMP"))
    return roots


def find_plugin_dirs(create_missing=False):
    """Locate ``plug-ins`` folders for installed GIMP 3 profiles."""
    found = []
    for root in candidate_roots():
        if not os.path.isdir(root):
            continue
        versions = sorted(
            name for name in os.listdir(root)
            if name.startswith("3.") and os.path.isdir(os.path.join(root, name))
        )
        for version in versions or (KNOWN_VERSIONS if create_missing else []):
            target = os.path.join(root, version, "plug-ins")
            if os.path.isdir(target) or create_missing:
                found.append(target)
    return found


def install_to(target_dir, quiet=False):
    destination = os.path.join(target_dir, PLUGIN_NAME)
    os.makedirs(target_dir, exist_ok=True)

    if os.path.islink(destination):
        os.unlink(destination)
    elif os.path.isdir(destination):
        shutil.rmtree(destination)

    shutil.copytree(
        SOURCE, destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    # GIMP only runs plug-in entry points that are executable on Unix.
    entry = os.path.join(destination, PLUGIN_NAME + ".py")
    mode = os.stat(entry).st_mode
    os.chmod(entry, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if not quiet:
        print("Installed -> %s" % destination)
    return destination


def uninstall_from(target_dir, quiet=False):
    destination = os.path.join(target_dir, PLUGIN_NAME)
    if os.path.islink(destination):
        os.unlink(destination)
    elif os.path.isdir(destination):
        shutil.rmtree(destination)
    else:
        return False
    if not quiet:
        print("Removed  -> %s" % destination)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uninstall", action="store_true",
                        help="remove the plug-in instead of installing it")
    parser.add_argument("--list", action="store_true",
                        help="show the GIMP profiles that were found and exit")
    parser.add_argument("--dir", action="append", default=None, metavar="PATH",
                        help="install into this plug-ins folder (repeatable)")
    args = parser.parse_args()

    if not os.path.isdir(SOURCE):
        print("Cannot find the plug-in source at %s" % SOURCE, file=sys.stderr)
        return 1

    targets = args.dir or find_plugin_dirs()

    if args.list:
        print("Plug-in source: %s" % SOURCE)
        if targets:
            print("GIMP plug-in folders found:")
            for target in targets:
                marker = "installed" if os.path.exists(os.path.join(target, PLUGIN_NAME)) else "-"
                print("  %-70s %s" % (target, marker))
        else:
            print("No GIMP 3 profile found. Start GIMP once, then run this again,")
            print("or pass --dir with the path shown in GIMP under")
            print("Edit > Preferences > Folders > Plug-ins.")
        return 0

    if not targets:
        print("No GIMP 3 profile found.", file=sys.stderr)
        print("Start GIMP once so it creates its settings folder, then run this "
              "again - or pass --dir with the path from\n"
              "  Edit > Preferences > Folders > Plug-ins", file=sys.stderr)
        return 1

    if args.uninstall:
        removed = sum(1 for target in targets if uninstall_from(target))
        if not removed:
            print("Nothing to remove.")
        else:
            print("\nRestart GIMP to finish removing it.")
            print("The downloaded models and Python environment are separate; "
                  "remove them from the plug-in's Setup window before "
                  "uninstalling, or delete the folder it names.")
        return 0

    for target in targets:
        install_to(target)
    print("\nRestart GIMP, then open an image and choose:")
    print("  Select > Segment Anything...")
    print("\nThe first run offers to download the model runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
