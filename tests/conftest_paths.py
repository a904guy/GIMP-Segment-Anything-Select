"""Shared helpers for the test suite (no external test runner required)."""

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(REPO, "plug-ins", "segment-anything")
BACKEND = os.path.join(PLUGIN, "backend")

if PLUGIN not in sys.path:
    sys.path.insert(0, PLUGIN)


STUB_ADAPTER = '''"""Stub adapters used by the tests: synthetic masks, no torch required."""
import numpy as np


class StubAdapter(object):
    backend_id = "sam2"
    display_name = "Stub SAM"

    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = "cpu"

    def load(self, report=None):
        if report:
            report("stub model loaded")

    def capabilities(self):
        return {"modes": ["everything", "text"], "supports_text": True,
                "text_hint": "stub"}

    def describe(self):
        return {"backend": "sam2", "display_name": "Stub SAM",
                "model": self.model_id, "device": "cpu"}

    def _shapes(self, w, h):
        shapes = []
        big = np.zeros((h, w), bool)
        big[int(h * .1):int(h * .9), int(w * .1):int(w * .9)] = True
        shapes.append((big, 0.99))
        for (y0, y1, x0, x1, score) in ((.2, .4, .2, .4, .95), (.6, .8, .6, .8, .9)):
            m = np.zeros((h, w), bool)
            m[int(h * y0):int(h * y1), int(w * x0):int(w * x1)] = True
            shapes.append((m, score))
        speck = np.zeros((h, w), bool)      # below the minimum-area filter
        speck[0:2, 0:2] = True
        shapes.append((speck, 0.5))
        shapes.append((big.copy(), 0.8))    # duplicate, must be removed
        return shapes

    def segment_everything(self, image, params, report=None):
        w, h = image.size
        return [{"mask": m, "score": s, "label": None} for m, s in self._shapes(w, h)]

    def segment_text(self, image, text, params, report=None):
        w, h = image.size
        return [{"mask": m, "score": s, "label": text}
                for m, s in self._shapes(w, h)[:3]]


def create(backend, model_id, device="auto"):
    return StubAdapter(model_id, device)


class AdapterError(RuntimeError):
    pass


def resolve_device(requested="auto"):
    return "cpu"
'''


def make_stub_backend():
    """A copy of the backend whose adapters are replaced by the stub."""
    root = tempfile.mkdtemp(prefix="sam-gimp-testbed-")
    target = os.path.join(root, "backend")
    shutil.copytree(BACKEND, target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(os.path.join(target, "adapters", "__init__.py"), "w") as handle:
        handle.write(STUB_ADAPTER)
    return root, target


class Sandbox(object):
    """Redirects the plug-in's data directory and backend at a stub."""

    def __init__(self, use_stub=True):
        self.home = tempfile.mkdtemp(prefix="sam-gimp-home-")
        self.root = None
        self.use_stub = use_stub

    def __enter__(self):
        os.environ["SAM_GIMP_HOME"] = self.home
        from samgimp import client, env
        self._env = env
        if self.use_stub:
            self.root, backend = make_stub_backend()
            env.backend_dir = lambda: backend
            env.venv_python = lambda: sys.executable
            client.env.backend_dir = env.backend_dir
            client.env.venv_python = env.venv_python
        return self

    def __exit__(self, *exc):
        try:
            from samgimp import client
            client.BackendClient().shutdown()
        except Exception:
            pass
        shutil.rmtree(self.home, ignore_errors=True)
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)
        os.environ.pop("SAM_GIMP_HOME", None)
        return False


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok  %s" % message)
