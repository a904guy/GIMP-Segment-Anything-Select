"""Exercise the Linux, macOS and Windows branches on whichever host runs.

The platform-specific code is mostly path and process handling that cannot
be reached on a single machine, so ``platform.system`` and friends are
stubbed and the resulting decisions are checked.
"""

import os
import platform
import sys

import conftest_paths as helpers  # noqa: F401
from conftest_paths import check


class FakePlatform(object):
    """Temporarily make the process look like another operating system."""

    def __init__(self, system, machine="x86_64", environ=None, libc=("glibc", "2.39")):
        self.system = system
        self.machine = machine
        self.environ = environ or {}
        self.libc = libc
        self._saved = {}

    def __enter__(self):
        self._real = (platform.system, platform.machine, platform.libc_ver, os.name)
        platform.system = lambda: self.system
        platform.machine = lambda: self.machine
        platform.libc_ver = lambda *a, **k: self.libc
        for key, value in self.environ.items():
            self._saved[key] = os.environ.get(key)
            os.environ[key] = value
        # data_dir() honours this override, which would mask the real logic
        self._saved["SAM_GIMP_HOME"] = os.environ.pop("SAM_GIMP_HOME", None)
        return self

    def __exit__(self, *exc):
        platform.system, platform.machine, platform.libc_ver, _ = self._real
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def test_data_locations():
    from samgimp import env

    with FakePlatform("Linux", environ={"XDG_DATA_HOME": "/home/u/.local/share"}):
        check(env.data_dir() == "/home/u/.local/share/sam-gimp",
              "Linux keeps its data under XDG_DATA_HOME")
        check(env.venv_python().endswith(os.path.join("runtime", "bin", "python")),
              "Linux looks for the interpreter in runtime/bin")

    with FakePlatform("Darwin", machine="arm64"):
        expected = os.path.join(os.path.expanduser("~"), "Library",
                                "Application Support", "sam-gimp")
        check(env.data_dir() == expected,
              "macOS uses ~/Library/Application Support")

    with FakePlatform("Windows", environ={"LOCALAPPDATA": r"C:\Users\u\AppData\Local"}):
        # os.path.join uses the host separator, so compare the joined form
        # rather than a hard-coded backslash path.
        check(env.data_dir() == os.path.join(r"C:\Users\u\AppData\Local", "SamGimp"),
              "Windows uses LOCALAPPDATA")
        check(env.venv_python().endswith(os.path.join("Scripts", "python.exe")),
              "Windows looks for Scripts\\python.exe")


def test_uv_assets():
    from samgimp import setup_env

    cases = [
        ("Linux", "x86_64", ("glibc", "2.39"), "uv-x86_64-unknown-linux-gnu.tar.gz"),
        ("Linux", "aarch64", ("glibc", "2.39"), "uv-aarch64-unknown-linux-gnu.tar.gz"),
        ("Linux", "x86_64", ("", ""), "uv-x86_64-unknown-linux-musl.tar.gz"),
        ("Darwin", "arm64", ("", ""), "uv-aarch64-apple-darwin.tar.gz"),
        ("Darwin", "x86_64", ("", ""), "uv-x86_64-apple-darwin.tar.gz"),
        ("Windows", "AMD64", ("", ""), "uv-x86_64-pc-windows-msvc.zip"),
        ("Windows", "arm64", ("", ""), "uv-aarch64-pc-windows-msvc.zip"),
    ]
    for system, machine, libc, expected in cases:
        with FakePlatform(system, machine, libc=libc):
            if system == "Linux" and libc == ("", ""):
                # musl detection also looks for the loader on disk
                real = setup_env.is_musl
                setup_env.is_musl = lambda: True
                try:
                    got = setup_env.uv_asset_name()
                finally:
                    setup_env.is_musl = real
            else:
                got = setup_env.uv_asset_name()
            check(got == expected,
                  "%s/%s downloads %s" % (system, machine, expected))

    with FakePlatform("Haiku"):
        try:
            setup_env.uv_asset_name()
            raise AssertionError("an unknown system should be refused")
        except setup_env.InstallError as error:
            check("Unsupported operating system" in str(error),
                  "an unsupported system gives a clear message")


def test_compute_choice():
    from samgimp import setup_env

    real_driver = setup_env.nvidia_driver_version
    real_rosetta = setup_env.is_rosetta
    try:
        setup_env.nvidia_driver_version = lambda: None
        setup_env.is_rosetta = lambda: False

        with FakePlatform("Darwin", machine="arm64"):
            check(setup_env.detect_compute() == "mps",
                  "Apple Silicon selects the Metal build")
        with FakePlatform("Darwin", machine="x86_64"):
            check(setup_env.detect_compute() == "cpu",
                  "Intel Macs get the processor build")

        setup_env.is_rosetta = lambda: True
        with FakePlatform("Darwin", machine="x86_64"):
            check(setup_env.detect_compute() == "mps",
                  "an x86 build under Rosetta still targets Apple Silicon")
        setup_env.is_rosetta = lambda: False

        for system in ("Linux", "Windows"):
            with FakePlatform(system):
                check(setup_env.detect_compute() == "cpu",
                      "%s without a driver falls back to the processor" % system)
            setup_env.nvidia_driver_version = lambda: 560.0
            with FakePlatform(system):
                check(setup_env.detect_compute() == "cuda",
                      "%s with a recent driver selects CUDA" % system)
            setup_env.nvidia_driver_version = lambda: 470.0
            with FakePlatform(system):
                check(setup_env.detect_compute() == "cpu",
                      "%s with an old driver falls back to the processor" % system)
            setup_env.nvidia_driver_version = lambda: None
    finally:
        setup_env.nvidia_driver_version = real_driver
        setup_env.is_rosetta = real_rosetta


def test_plugin_install_targets():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sam_installer", os.path.join(helpers.REPO, "install.py"))
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)

    with FakePlatform("Windows", environ={"APPDATA": r"C:\Users\u\AppData\Roaming"}):
        roots = installer.candidate_roots()
        check(roots == [os.path.join(r"C:\Users\u\AppData\Roaming", "GIMP")],
              "Windows looks in APPDATA for GIMP profiles")

    with FakePlatform("Darwin"):
        roots = installer.candidate_roots()
        check(roots[0].endswith(os.path.join("Library", "Application Support", "GIMP")),
              "macOS looks in ~/Library/Application Support")

    with FakePlatform("Linux", environ={"XDG_CONFIG_HOME": "/home/u/.config"}):
        roots = installer.candidate_roots()
        check("/home/u/.config/GIMP" in roots, "Linux looks in XDG_CONFIG_HOME")
        check(any("org.gimp.GIMP" in r for r in roots), "Linux also checks Flatpak")
        check(any("snap" in r for r in roots), "Linux also checks Snap")


def test_child_environment():
    from samgimp import env

    os.environ["LD_LIBRARY_PATH"] = "/opt/gimp/lib"
    os.environ["PYTHONHOME"] = "/opt/gimp"
    os.environ["DYLD_LIBRARY_PATH"] = "/Applications/GIMP.app/Contents/lib"
    try:
        child = env.child_environment({"HF_HOME": "/models"})
        for name in ("LD_LIBRARY_PATH", "PYTHONHOME", "DYLD_LIBRARY_PATH"):
            check(name not in child,
                  "%s from the GIMP wrapper is not passed to the backend" % name)
        check(child["PYTHONNOUSERSITE"] == "1",
              "user site-packages cannot shadow the private install")
        check(child["HF_HOME"] == "/models", "explicit settings still get through")
    finally:
        for name in ("LD_LIBRARY_PATH", "PYTHONHOME", "DYLD_LIBRARY_PATH"):
            os.environ.pop(name, None)


def test_modifier_naming():
    from samgimp import env
    with FakePlatform("Darwin", machine="arm64"):
        check(env.primary_modifier_name() == "Command",
              "macOS text says Command, not Ctrl")
    for system in ("Linux", "Windows"):
        with FakePlatform(system):
            check(env.primary_modifier_name() == "Ctrl",
                  "%s text says Ctrl" % system)


def test_windows_long_paths_probe():
    from samgimp import setup_env
    with FakePlatform("Linux"):
        check(setup_env.windows_long_paths_enabled() is None,
              "the Windows path-length probe is skipped elsewhere")


def run():
    test_data_locations()
    test_uv_assets()
    test_compute_choice()
    test_plugin_install_targets()
    test_child_environment()
    test_modifier_naming()
    test_windows_long_paths_probe()


if __name__ == "__main__":
    run()
