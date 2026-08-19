"""Platform paths, settings and small helpers.

This module runs inside GIMP's own Python interpreter, so it must stay
pure standard library + no heavy imports.  Everything torch-related lives
in the backend, which runs in its own isolated interpreter.
"""

import json
import os
import platform

APP_NAME = "sam-gimp"
SETTINGS_VERSION = 1

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------


def _windows_dir(var, fallback):
    base = os.environ.get(var)
    if base:
        return os.path.join(base, "SamGimp")
    return os.path.join(os.path.expanduser("~"), fallback, "SamGimp")


def data_dir():
    """Where we keep the private interpreter, model weights and settings."""
    override = os.environ.get("SAM_GIMP_HOME")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    system = platform.system()
    if system == "Windows":
        return _windows_dir("LOCALAPPDATA", os.path.join("AppData", "Local"))
    if system == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", APP_NAME)
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_NAME)


def cache_dir():
    system = platform.system()
    if system == "Windows":
        return os.path.join(data_dir(), "cache")
    if system == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches", APP_NAME)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, APP_NAME)


def venv_dir():
    return os.path.join(data_dir(), "runtime")


def venv_python():
    """Path to the interpreter inside our private environment."""
    if platform.system() == "Windows":
        return os.path.join(venv_dir(), "Scripts", "python.exe")
    return os.path.join(venv_dir(), "bin", "python")


def models_dir():
    return os.path.join(data_dir(), "models")


def tools_dir():
    return os.path.join(data_dir(), "tools")


def logs_dir():
    return os.path.join(data_dir(), "logs")


def temp_dir():
    return os.path.join(cache_dir(), "tmp")


def settings_path():
    return os.path.join(data_dir(), "settings.json")


def daemon_path():
    return os.path.join(data_dir(), "daemon.json")


def plugin_dir():
    """Root of the installed plug-in (the folder holding backend/ and samgimp/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def backend_dir():
    return os.path.join(plugin_dir(), "backend")


# Variables a GIMP build may inject to point at its own bundled libraries.
# An AppImage, Flatpak or macOS .app wrapper sets these so GIMP finds its own
# Python and shared objects; inheriting them in the backend would make our
# self-contained interpreter load the wrong ones.
_INHERITED_BLOCKLIST = (
    "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
    "LD_LIBRARY_PATH", "LD_PRELOAD",
    "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    "GI_TYPELIB_PATH", "VIRTUAL_ENV",
)


def child_environment(extra=None):
    """Environment for a helper process, cleaned of GIMP's own settings."""
    environment = dict(os.environ)
    for name in _INHERITED_BLOCKLIST:
        environment.pop(name, None)

    # Put our own interpreter first so it resolves its own libraries.
    binaries = os.path.join(venv_dir(), "Scripts" if platform.system() == "Windows" else "bin")
    if os.path.isdir(binaries):
        environment["PATH"] = binaries + os.pathsep + environment.get("PATH", "")

    environment["PYTHONUNBUFFERED"] = "1"
    # Never let a user site-packages directory shadow the private install.
    environment["PYTHONNOUSERSITE"] = "1"
    if extra:
        environment.update(extra)
    return environment


def primary_modifier_name():
    """What to call the multi-select modifier in user-facing text."""
    return "Command" if platform.system() == "Darwin" else "Ctrl"


def display_path(path):
    """Path with the home directory shortened to ``~`` for display."""
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def ensure_dirs():
    for path in (data_dir(), models_dir(), tools_dir(), logs_dir(), temp_dir()):
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

DEFAULTS = {
    "version": SETTINGS_VERSION,
    "installed": False,
    "backend": "sam2",              # "sam2" | "sam3"
    "model": "facebook/sam2.1-hiera-large",
    "device": "auto",               # auto | cuda | mps | cpu
    "compute": "auto",              # auto | cuda | cpu  (which torch build was installed)
    "points_per_side": 32,
    "pred_iou_thresh": 0.75,
    "stability_score_thresh": 0.88,
    "min_area_percent": 0.05,       # drop specks smaller than this % of the image
    "max_segments": 254,
    "inference_max_side": 1536,
    "preview_max_side": 1400,
    "idle_timeout_sec": 900,
    "hf_token": "",
    "last_text_prompt": "",
    "selection_op": "replace",
    "feather": 0.0,
}


def load_settings():
    data = dict(DEFAULTS)
    try:
        with open(settings_path(), "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if isinstance(stored, dict):
            data.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return data


def save_settings(settings):
    ensure_dirs()
    payload = {k: v for k, v in settings.items() if k in DEFAULTS}
    payload["version"] = SETTINGS_VERSION
    tmp = settings_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, settings_path())


def is_installed():
    """True when the private environment exists and looks usable."""
    return os.path.isfile(venv_python()) and load_settings().get("installed", False)


# --------------------------------------------------------------------------
# Model catalogue
# --------------------------------------------------------------------------

# Each entry: id -> (label, approximate download size, notes)
SAM2_MODELS = [
    ("facebook/sam2.1-hiera-tiny", "SAM 2.1 Tiny - fastest, lowest VRAM", "~150 MB"),
    ("facebook/sam2.1-hiera-small", "SAM 2.1 Small", "~185 MB"),
    ("facebook/sam2.1-hiera-base-plus", "SAM 2.1 Base+", "~325 MB"),
    ("facebook/sam2.1-hiera-large", "SAM 2.1 Large - best quality (recommended)", "~900 MB"),
]

SAM3_MODELS = [
    ("facebook/sam3", "SAM 3 - concept + geometry prompts", "~3.5 GB"),
]


def models_for(backend):
    return SAM3_MODELS if backend == "sam3" else SAM2_MODELS


def default_model_for(backend):
    return models_for(backend)[-1][0] if backend == "sam2" else SAM3_MODELS[0][0]


def backend_needs_token(backend):
    """SAM 3 weights are gated on Hugging Face and need an access token."""
    return backend == "sam3"
