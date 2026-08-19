"""SAM 2.1 adapter (public weights, no login required)."""

from .base import AdapterError, BaseAdapter, resolve_device


class Sam2Adapter(BaseAdapter):
    backend_id = "sam2"
    display_name = "SAM 2.1"

    def load(self, report=None):
        self.device = resolve_device(self.requested_device)
        if report:
            report("Loading %s on %s" % (self.model_id, self.device))
        try:
            from transformers import Sam2Model  # noqa: F401  (import check only)
        except ImportError as error:
            raise AdapterError(
                "This transformers build has no SAM 2 support (%s). "
                "Re-run Setup to update the environment." % error
            )
        # Building the pipeline is what actually pulls weights into memory.
        self._mask_generation_pipeline(report)
        if report:
            report("Model ready.")

    def capabilities(self):
        return {
            "modes": ["everything"],
            "supports_text": False,
            "text_hint": "",
        }
