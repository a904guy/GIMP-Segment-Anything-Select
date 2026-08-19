"""Model adapters.  Each one wraps a SAM family model behind one interface."""

from .base import AdapterError


def create(backend, model_id, device="auto"):
    if backend == "sam3":
        from .sam3 import Sam3Adapter
        return Sam3Adapter(model_id, device)
    if backend == "sam2":
        from .sam2 import Sam2Adapter
        return Sam2Adapter(model_id, device)
    raise AdapterError("Unknown backend: %r (expected 'sam2' or 'sam3')" % backend)
