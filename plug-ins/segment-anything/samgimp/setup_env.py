"""Builds the private Python environment the backend runs in.

This module runs inside GIMP's interpreter, so it is limited to the
standard library.  GIMP's Python version is fixed by the build and may have
no PyTorch wheels available, which is the reason the heavy dependencies are
installed into a separate environment on a pinned Python version instead.

Everything installed lives under a single directory (``env.data_dir()``),
so uninstalling means removing one folder.
"""

import glob
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

from . import env

# Python version for the private environment.  SAM 3 requires >= 3.12.
TARGET_PYTHON = "3.12"

UV_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download/%s"

# Minimum NVIDIA driver that can run CUDA 12.x builds (12.x minor version
# compatibility): 525 on Linux, 528 on Windows.
MIN_CUDA_DRIVER = {"Linux": 525.0, "Windows": 528.0}

CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

RUNTIME_PACKAGES = [
    "transformers>=5.0",
    "huggingface_hub>=0.34",
    "safetensors>=0.4",
    "numpy>=1.26",
    "pillow>=10.0",
    "accelerate>=1.0",
    "scipy>=1.11",
]


class InstallError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Platform probing
# --------------------------------------------------------------------------


def is_musl():
    """True on a musl-based Linux such as Alpine, which needs other builds."""
    if platform.system() != "Linux":
        return False
    try:
        if platform.libc_ver()[0] == "glibc":
            return False
    except (OSError, ValueError):
        pass
    return bool(glob.glob("/lib/ld-musl-*.so.1"))


def is_rosetta():
    """True when an x86_64 interpreter is being translated on Apple Silicon.

    GIMP running under Rosetta reports x86_64, which would otherwise get an
    Intel toolchain and lose Metal acceleration on an Apple Silicon machine.
    """
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "1"


def target_machine():
    """The architecture to build for, seeing through Rosetta translation."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if is_rosetta():
        return "arm64"
    return machine


def uv_asset_name():
    system = platform.system()
    arm = target_machine() == "arm64"
    if system == "Windows":
        return "uv-aarch64-pc-windows-msvc.zip" if arm else "uv-x86_64-pc-windows-msvc.zip"
    if system == "Darwin":
        return "uv-aarch64-apple-darwin.tar.gz" if arm else "uv-x86_64-apple-darwin.tar.gz"
    if system == "Linux":
        libc = "musl" if is_musl() else "gnu"
        arch = "aarch64" if arm else "x86_64"
        return "uv-%s-unknown-linux-%s.tar.gz" % (arch, libc)
    raise InstallError(
        "Unsupported operating system: %s. This plug-in supports Linux, "
        "macOS and Windows." % system)


def nvidia_driver_version():
    """Driver version as a float, or None when no NVIDIA GPU is usable."""
    exe = shutil.which("nvidia-smi")
    if not exe and platform.system() == "Windows":
        candidate = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"
        )
        exe = candidate if os.path.isfile(candidate) else None
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    match = re.search(r"(\d+)\.(\d+)", out.stdout)
    if not match:
        return None
    return float("%s.%s" % (match.group(1), match.group(2)))


def detect_compute():
    """Pick which PyTorch build to install: 'cuda', 'mps' or 'cpu'."""
    system = platform.system()
    if system == "Darwin":
        # Apple Silicon gets Metal acceleration from the stock wheels.
        return "mps" if target_machine() == "arm64" else "cpu"
    driver = nvidia_driver_version()
    if driver is not None and driver >= MIN_CUDA_DRIVER.get(system, 525.0):
        return "cuda"
    return "cpu"


def windows_long_paths_enabled():
    """Whether Windows accepts paths longer than 260 characters.

    PyTorch's bundled CUDA packages nest deeply enough to exceed the old
    limit, so an install can fail late with a confusing error when long path
    support is off.  Returns None on other systems or when unreadable.
    """
    if platform.system() != "Windows":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")
            return bool(value)
    except (ImportError, OSError):
        return None


def describe_compute(compute):
    if compute == "cuda":
        return "NVIDIA GPU (CUDA 12.8 build, about 3 GB to download)"
    if compute == "mps":
        return "Apple Silicon GPU (Metal, about 1 GB to download)"
    return "CPU only (about 300 MB to download; segmentation will be slow)"


# --------------------------------------------------------------------------
# Running subprocesses with streamed output
# --------------------------------------------------------------------------


def _no_window_kwargs():
    if platform.system() == "Windows":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def _run(cmd, report, cwd=None, extra_env=None, should_cancel=None):
    """Run a command, streaming each output line to ``report``."""
    report("$ " + " ".join(env.display_path(part) for part in cmd))
    child_env = env.child_environment(extra_env)

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
        **_no_window_kwargs()
    )
    tail = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-40]
                report(line)
            if should_cancel and should_cancel():
                proc.terminate()
                raise Cancelled("Installation cancelled.")
    finally:
        proc.stdout.close()
    code = proc.wait()
    if code != 0:
        raise InstallError(
            "Command failed (exit %d): %s\n%s" % (code, " ".join(cmd), "\n".join(tail[-15:]))
        )


# --------------------------------------------------------------------------
# uv bootstrap
# --------------------------------------------------------------------------


def _uv_local_path():
    name = "uv.exe" if platform.system() == "Windows" else "uv"
    return os.path.join(env.tools_dir(), name)


def find_uv():
    """An existing uv we can use, or None."""
    local = _uv_local_path()
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    found = shutil.which("uv")
    return found or None


def download_uv(report, should_cancel=None):
    """Fetch a private copy of uv.  Returns its path."""
    env.ensure_dirs()
    asset = uv_asset_name()
    url = UV_RELEASE % asset
    report("Downloading uv from %s" % url)

    # Not TemporaryDirectory: on Windows a virus scanner can still hold the
    # freshly extracted binary when the context manager tries to remove the
    # folder, which would abort an otherwise successful install.
    work = tempfile.mkdtemp(dir=env.temp_dir())
    try:
        archive = os.path.join(work, asset)
        _download(url, archive, report, should_cancel=should_cancel)
        extract_to = os.path.join(work, "x")
        os.makedirs(extract_to, exist_ok=True)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_to)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                _safe_extract(tf, extract_to)

        binary_name = "uv.exe" if platform.system() == "Windows" else "uv"
        source = None
        for root, _dirs, files in os.walk(extract_to):
            if binary_name in files:
                source = os.path.join(root, binary_name)
                break
        if not source:
            raise InstallError("uv binary not found inside %s" % asset)

        target = _uv_local_path()
        shutil.copy2(source, target)
        os.chmod(target, os.stat(target).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        report("Installed uv at %s" % env.display_path(target))
        return target
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _safe_extract(tar, path):
    """tarfile.extractall with a guard against paths escaping the target."""
    base = os.path.realpath(path)
    for member in tar.getmembers():
        dest = os.path.realpath(os.path.join(path, member.name))
        if not (dest == base or dest.startswith(base + os.sep)):
            raise InstallError("Refusing to extract unsafe path: %s" % member.name)
    tar.extractall(path)


def _download(url, dest, report, should_cancel=None):
    request = urllib.request.Request(url, headers={"User-Agent": "sam-gimp-installer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        with open(dest, "wb") as handle:
            while True:
                if should_cancel and should_cancel():
                    raise Cancelled("Installation cancelled.")
                chunk = response.read(262144)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last and pct % 5 == 0:
                        report("  %d%% (%.1f MB)" % (pct, done / 1048576.0))
                        last = pct


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------


def environment_is_healthy(report=None):
    """True when the private environment already exists and imports cleanly."""
    python = env.venv_python()
    if not os.path.isfile(python):
        return False
    check = os.path.join(env.backend_dir(), "selfcheck.py")
    if not os.path.isfile(check):
        return False
    try:
        result = subprocess.run(
            [python, check],
            capture_output=True, text=True, timeout=180,
            env=env.child_environment({"HF_HOME": env.models_dir()}),
            **_no_window_kwargs()
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if report and result.stdout:
        for line in result.stdout.splitlines():
            report(line)
    return result.returncode == 0 and "SELFCHECK OK" in (result.stdout or "")


def install(settings, report, should_cancel=None, progress=None, force=False):
    """Build the private environment and fetch the selected model weights.

    ``report(str)`` receives log lines; ``progress(fraction, label)`` drives a
    progress bar.  When a healthy environment already exists it is reused and
    only the model weights are fetched, so switching models is quick.
    Raises InstallError / Cancelled on failure.
    """
    def step(fraction, label):
        report("")
        report("=== %s ===" % label)
        if progress:
            progress(fraction, label)

    env.ensure_dirs()
    compute = settings.get("compute", "auto")
    if compute == "auto":
        compute = detect_compute()
    report("Platform: %s %s" % (platform.system(), platform.machine()))
    if is_rosetta():
        report("Note: running under Rosetta; building for Apple Silicon instead.")
    if is_musl():
        report("Detected a musl-based Linux; using musl builds.")
    if windows_long_paths_enabled() is False:
        report("Warning: Windows long path support is disabled. PyTorch's CUDA "
               "files nest deeply and the install may fail with a path-too-long "
               "error. See the README for how to enable it.")
    report("Compute:  %s" % describe_compute(compute))
    report("Target:   %s / %s" % (settings.get("backend"), settings.get("model")))
    report("Install location: %s" % env.display_path(env.data_dir()))

    if not force:
        step(0.03, "Checking for an existing environment")
        if environment_is_healthy(report):
            report("")
            report("The Python environment is already in place - reusing it.")
            return _install_weights_only(settings, report, should_cancel, step)
        report("No usable environment yet; building one.")

    # 1. uv -----------------------------------------------------------------
    step(0.05, "Locating uv (Python environment manager)")
    uv = find_uv()
    if uv:
        report("Using existing uv: %s" % uv)
    else:
        uv = download_uv(report, should_cancel)

    # 2. interpreter + venv --------------------------------------------------
    step(0.12, "Creating an isolated Python %s environment" % TARGET_PYTHON)
    uv_env = {
        "UV_PYTHON_INSTALL_DIR": os.path.join(env.tools_dir(), "python"),
        "UV_CACHE_DIR": os.path.join(env.cache_dir(), "uv"),
        "UV_NO_CONFIG": "1",
    }
    _run([uv, "python", "install", TARGET_PYTHON], report,
         extra_env=uv_env, should_cancel=should_cancel)
    if os.path.isdir(env.venv_dir()):
        report("Removing previous environment at %s" % env.venv_dir())
        shutil.rmtree(env.venv_dir(), ignore_errors=True)
    _run([uv, "venv", "--python", TARGET_PYTHON, env.venv_dir()], report,
         extra_env=uv_env, should_cancel=should_cancel)

    python = env.venv_python()
    if not os.path.isfile(python):
        raise InstallError("Environment creation did not produce %s" % python)

    pip_base = [uv, "pip", "install", "--python", python]

    # 3. torch ---------------------------------------------------------------
    step(0.20, "Installing PyTorch (%s)" % compute)
    torch_cmd = list(pip_base) + ["torch", "torchvision"]
    if compute == "cuda":
        torch_cmd += ["--index-url", CUDA_INDEX]
    elif compute == "cpu":
        torch_cmd += ["--index-url", CPU_INDEX]
    # mps/macOS uses the default PyPI wheels, which are Metal-enabled.
    _run(torch_cmd, report, extra_env=uv_env, should_cancel=should_cancel)

    # 4. the rest ------------------------------------------------------------
    step(0.55, "Installing transformers and supporting libraries")
    _run(list(pip_base) + RUNTIME_PACKAGES, report,
         extra_env=uv_env, should_cancel=should_cancel)

    # 5. verify --------------------------------------------------------------
    step(0.70, "Verifying the environment")
    check = os.path.join(env.backend_dir(), "selfcheck.py")
    _run([python, check], report,
         extra_env={"HF_HOME": env.models_dir()}, should_cancel=should_cancel)

    # 6. weights -------------------------------------------------------------
    step(0.78, "Downloading model weights: %s" % settings.get("model"))
    fetch = os.path.join(env.backend_dir(), "fetch_model.py")
    fetch_env = {
        "HF_HOME": env.models_dir(),
        "HUGGINGFACE_HUB_CACHE": os.path.join(env.models_dir(), "hub"),
    }
    if settings.get("hf_token"):
        fetch_env["HF_TOKEN"] = settings["hf_token"]
    _run([python, fetch, "--model", settings["model"], "--backend", settings["backend"]],
         report, extra_env=fetch_env, should_cancel=should_cancel)

    # 7. record --------------------------------------------------------------
    step(1.0, "Finishing up")
    settings = dict(settings)
    settings["installed"] = True
    settings["compute"] = compute
    env.save_settings(settings)
    report("")
    report("Installation complete.")
    return settings


def _install_weights_only(settings, report, should_cancel, step):
    """Fetch weights into an environment that is already built."""
    step(0.3, "Downloading model weights: %s" % settings.get("model"))
    fetch = os.path.join(env.backend_dir(), "fetch_model.py")
    fetch_env = {
        "HF_HOME": env.models_dir(),
        "HUGGINGFACE_HUB_CACHE": os.path.join(env.models_dir(), "hub"),
    }
    if settings.get("hf_token"):
        fetch_env["HF_TOKEN"] = settings["hf_token"]
    _run([env.venv_python(), fetch, "--model", settings["model"],
          "--backend", settings["backend"]],
         report, extra_env=fetch_env, should_cancel=should_cancel)

    step(1.0, "Finishing up")
    settings = dict(settings)
    settings["installed"] = True
    if settings.get("compute", "auto") == "auto":
        settings["compute"] = detect_compute()
    env.save_settings(settings)
    report("")
    report("Ready.")
    return settings


def uninstall(report=None, keep_models=False):
    """Remove the private environment (and optionally keep the weights)."""
    def say(message):
        if report:
            report(message)
    targets = [env.venv_dir(), env.tools_dir(), os.path.join(env.cache_dir(), "uv")]
    if not keep_models:
        targets.append(env.models_dir())
    for path in targets:
        if os.path.exists(path):
            say("Removing %s" % path)
            shutil.rmtree(path, ignore_errors=True)
    settings = env.load_settings()
    settings["installed"] = False
    env.save_settings(settings)
    say("Done.")


def installed_summary():
    """Short human-readable description of what is installed, or None."""
    if not env.is_installed():
        return None
    settings = env.load_settings()
    return "%s / %s (%s)" % (
        "SAM 3" if settings["backend"] == "sam3" else "SAM 2.1",
        settings["model"].split("/")[-1],
        settings.get("compute", "?"),
    )
