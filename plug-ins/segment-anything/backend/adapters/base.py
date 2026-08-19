"""Shared adapter plumbing."""

import inspect

import numpy as np


class AdapterError(RuntimeError):
    pass


def resolve_device(requested="auto"):
    """Return a torch device string honouring the user's preference."""
    import torch

    if requested and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise AdapterError("CUDA was requested but no CUDA device is available.")
        if requested == "mps" and not getattr(torch.backends, "mps", None):
            raise AdapterError("MPS was requested but this build has no Metal support.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def autocast_dtype(device):
    import torch

    if device == "cuda":
        # bfloat16 needs Ampere or newer; fall back to fp16 on older cards.
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def call_tolerantly(func, *args, **kwargs):
    """Call ``func`` dropping keyword arguments it does not accept.

    The transformers pipeline signatures drift between releases.  Rather
    than pinning a narrow version range, unsupported knobs are dropped and
    reported, so a newer transformers cannot break segmentation outright.
    """
    dropped = []
    try:
        signature = inspect.signature(func)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_kwargs:
            allowed = set(signature.parameters)
            for key in list(kwargs):
                if key not in allowed:
                    dropped.append(key)
                    kwargs.pop(key)
    except (TypeError, ValueError):
        pass

    while True:
        try:
            return func(*args, **kwargs), dropped
        except TypeError as error:
            message = str(error)
            culprit = None
            for key in list(kwargs):
                if key in message and "unexpected keyword" in message:
                    culprit = key
                    break
            if culprit is None:
                raise
            kwargs.pop(culprit)
            dropped.append(culprit)


class BaseAdapter(object):
    """Common behaviour for the SAM 2 and SAM 3 adapters."""

    backend_id = "base"
    display_name = "SAM"

    def __init__(self, model_id, device="auto"):
        self.model_id = model_id
        self.requested_device = device
        self.device = None
        self._pipeline = None

    # -- lifecycle ---------------------------------------------------------

    def load(self, report=None):
        raise NotImplementedError

    def capabilities(self):
        raise NotImplementedError

    def describe(self):
        return {
            "backend": self.backend_id,
            "display_name": self.display_name,
            "model": self.model_id,
            "device": self.device,
        }

    # -- "segment everything" ---------------------------------------------

    def _mask_generation_pipeline(self, report=None):
        """Lazily build the transformers automatic mask-generation pipeline."""
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from transformers import pipeline

        if report:
            report("Preparing automatic mask generation (%s)" % self.model_id)
        # Full precision is required here.  The pipeline passes its boxes to
        # torchvision NMS, which rejects half-precision scores alongside
        # float32 boxes ("dets should have the same type as scores").  The
        # vision encoder dominates the runtime, so little is lost.
        self._pipeline = pipeline("mask-generation", model=self.model_id,
                                  device=self.device)
        return self._pipeline

    def release_unused(self, keeping, report=None):
        """Drop model heads other than ``keeping`` to free memory."""
        return False

    def segment_everything(self, image, params, report=None):
        """Grid-prompt the model and return every distinct region it finds."""
        self.release_unused("everything", report)
        generator = self._mask_generation_pipeline(report)
        points_per_side = int(params.get("points_per_side", 32))
        call_kwargs = {
            "points_per_batch": int(params.get("points_per_batch", 64)),
            "points_per_crop": points_per_side,
            "pred_iou_thresh": float(params.get("pred_iou_thresh", 0.75)),
            "stability_score_thresh": float(params.get("stability_score_thresh", 0.88)),
            "crops_n_layers": int(params.get("crops_n_layers", 0)),
        }
        if report:
            report("Running %dx%d point grid" % (points_per_side, points_per_side))

        (outputs, dropped) = call_tolerantly(generator, image, **call_kwargs)
        if dropped and report:
            report("Note: this transformers version ignores %s" % ", ".join(sorted(set(dropped))))

        masks = outputs["masks"] if isinstance(outputs, dict) else outputs
        scores = outputs.get("scores") if isinstance(outputs, dict) else None
        results = []
        for index, mask in enumerate(masks):
            array = _to_bool_array(mask)
            if array is None or not array.any():
                continue
            score = 1.0
            if scores is not None:
                try:
                    score = float(scores[index])
                except (IndexError, TypeError, ValueError):
                    score = 1.0
            results.append({"mask": array, "score": score, "label": None})
        return results


def _to_bool_array(mask):
    """Normalise whatever the pipeline returned into a 2-D boolean array."""
    try:
        import torch
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[0] if array.shape[0] == 1 else array.any(axis=0)
    if array.ndim != 2:
        return None
    if array.dtype != bool:
        array = array > 0.5 if array.dtype.kind == "f" else array.astype(bool)
    return array
