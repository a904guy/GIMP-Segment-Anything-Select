"""The source tree must stay directly installable.

Both install routes copy `plug-ins/segment-anything` as-is: GitHub's
"Download ZIP" serves it straight from git, and the release workflow zips
the same folder. So the checks here are on the tree itself rather than on
any build output.
"""

import os
import stat
import subprocess

import conftest_paths as helpers
from conftest_paths import check

PLUGIN = helpers.PLUGIN
ENTRY = os.path.join(PLUGIN, "segment-anything.py")


def test_entry_point_is_executable():
    mode = os.stat(ENTRY).st_mode
    check(bool(mode & stat.S_IXUSR),
          "the entry point is executable on disk, which GIMP requires on Unix")

    # GitHub builds its "Download ZIP" with git archive, which copies the mode
    # recorded in the index.  If that is 100644 the download installs a
    # plug-in GIMP silently ignores.
    result = subprocess.run(
        ["git", "ls-files", "-s", "--", os.path.relpath(ENTRY, helpers.REPO)],
        capture_output=True, text=True, cwd=helpers.REPO)
    if result.returncode == 0 and result.stdout.strip():
        recorded = result.stdout.split()[0]
        check(recorded == "100755",
              "git records the entry point as 100755 so Download ZIP stays installable")
    else:
        print("  ..  skipped the git mode check (file not staged yet)")


def test_required_files_present():
    required = [
        "segment-anything.py",
        os.path.join("samgimp", "__init__.py"),
        os.path.join("samgimp", "dialog_main.py"),
        os.path.join("samgimp", "dialog_setup.py"),
        os.path.join("samgimp", "canvas.py"),
        os.path.join("samgimp", "seglist.py"),
        os.path.join("samgimp", "client.py"),
        os.path.join("samgimp", "setup_env.py"),
        os.path.join("samgimp", "gimpio.py"),
        os.path.join("backend", "sam_server.py"),
        os.path.join("backend", "maskutil.py"),
        os.path.join("backend", "selfcheck.py"),
        os.path.join("backend", "fetch_model.py"),
        os.path.join("backend", "adapters", "sam2.py"),
        os.path.join("backend", "adapters", "sam3.py"),
    ]
    missing = [r for r in required if not os.path.isfile(os.path.join(PLUGIN, r))]
    check(not missing, "every file the plug-in needs is present (%d checked)" % len(required))


def test_no_build_leftovers_are_tracked():
    # Running the tests creates __pycache__ in the working tree, so what
    # matters is that none of it is tracked: git decides what "Download ZIP"
    # serves, and the release workflow strips the rest.
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                            cwd=helpers.REPO)
    if result.returncode != 0:
        print("  ..  skipped (not a git repository)")
        return
    tracked = result.stdout.splitlines()
    strays = [f for f in tracked
              if "__pycache__" in f or f.endswith((".pyc", ".pyo"))]
    check(not strays, "git tracks no compiled leftovers")
    if tracked:
        check(not any(f.startswith("dist/") for f in tracked),
              "no build output is tracked")


def test_folder_name_matches_entry_point():
    # GIMP looks for <folder>/<folder>.py when scanning the plug-ins directory.
    folder = os.path.basename(PLUGIN)
    check(os.path.isfile(os.path.join(PLUGIN, folder + ".py")),
          "the folder and its entry point share a name, as GIMP expects")


def test_line_endings_are_pinned():
    attributes = os.path.join(helpers.REPO, ".gitattributes")
    check(os.path.isfile(attributes), ".gitattributes exists")
    text = open(attributes).read()
    check("eol=lf" in text,
          "line endings are pinned to LF so a Windows clone keeps the shebang intact")


def run():
    test_entry_point_is_executable()
    test_required_files_present()
    test_no_build_leftovers_are_tracked()
    test_folder_name_matches_entry_point()
    test_line_endings_are_pinned()


if __name__ == "__main__":
    run()
