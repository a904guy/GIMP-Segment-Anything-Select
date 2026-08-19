#!/usr/bin/env python3
"""Run the test suite.

    python3 tests/run.py            # everything that does not need a GPU
    python3 tests/run.py --gimp     # also the tests that run inside GIMP
    python3 tests/run.py --render out.png

The default set uses a stub model, so it needs neither the downloaded
weights nor a graphics card.
"""

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PURE_TESTS = ["test_masks", "test_rle_client", "test_platforms",
              "test_package", "test_protocol", "test_widgets"]


def run_pure(render=None):
    failures = []
    for name in PURE_TESTS:
        print("\n== %s ==" % name)
        module = __import__(name)
        try:
            if name == "test_widgets" and render:
                module.run(save_render=render)
            else:
                module.run()
        except Exception as error:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures.append("%s: %s" % (name, error))
    return failures


GIMP_TESTS = [("test_gimp_io", ".gimp_io_result"),
              ("test_gimp_dialog", ".gimp_dialog_result")]


def find_gimp_console():
    """Locate gimp-console on Linux, macOS or Windows."""
    names = ["gimp-console", "gimp-console-3.0", "gimp-console-3.2",
             "gimp-console-3", "gimp-console-2.99"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found

    system = platform.system()
    patterns = []
    if system == "Darwin":
        patterns = ["/Applications/GIMP*.app/Contents/MacOS/gimp-console*",
                    os.path.expanduser("~/Applications/GIMP*.app/Contents/MacOS/gimp-console*")]
    elif system == "Windows":
        for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            patterns.append(os.path.join(root, "GIMP *", "bin", "gimp-console-*.exe"))
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def run_gimp():
    binary = find_gimp_console()
    if not binary:
        print("\n== inside GIMP ==\n  skipped: gimp-console not found on PATH")
        return []
    failures = []
    for name, marker_name in GIMP_TESTS:
        failures += _run_one_gimp_test(binary, name, marker_name)
    return failures


def _run_one_gimp_test(binary, name, marker_name):
    print("\n== %s (inside GIMP) ==" % name)
    script = os.path.join(HERE, name + ".py")
    marker = os.path.join(HERE, marker_name)
    if os.path.exists(marker):
        os.remove(marker)
    result = subprocess.run(
        [binary, "-idf", "-s", "--quit", "--batch-interpreter", "python-fu-eval",
         "-b", 'exec(open(%r).read())' % script],
        capture_output=True, text=True, timeout=600,
        env=dict(os.environ, SAM_GIMP_TEST_DIR=HERE),
    )
    if not os.path.exists(marker):
        print("  FAILED: the test produced no result. GIMP said:")
        print("\n".join((result.stdout or "").splitlines()[-15:]))
        print("\n".join((result.stderr or "").splitlines()[-15:]))
        return ["%s produced no result" % name]
    with open(marker) as handle:
        content = handle.read()
    os.remove(marker)
    print(content.split("\n", 1)[1] if "\n" in content else content)
    return [] if content.startswith("PASSED") else [name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gimp", action="store_true",
                        help="also run the tests that need gimp-console")
    parser.add_argument("--render", metavar="PNG",
                        help="write a picture of the two panes for inspection")
    args = parser.parse_args()

    failures = run_pure(args.render)
    if args.gimp:
        failures += run_gimp()

    print("\n" + "=" * 60)
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
