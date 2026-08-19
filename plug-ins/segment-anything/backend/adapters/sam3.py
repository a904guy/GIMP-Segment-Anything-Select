"""SAM 3 adapter.

SAM 3 does everything SAM 2 does (grid-prompted "segment everything" via the
tracker head) and adds Promptable Concept Segmentation: give it a short noun
phrase such as "front wheel" and it returns every matching instance as its
own segment, already named.  Both modes are exposed to the plug-in.

The two modes are served by two separate multi-gigabyte heads, so only the
one in use is kept resident; switching modes releases the other.  Loading
both at once roughly doubles the memory needed and is what makes SAM 3 fail
on cards that could otherwise run it.

The weights are gated on Hugging Face, so the user has to accept Meta's
licence at https://huggingface.co/facebook/sam3 and supply an access token
during setup.
"""

from .base import AdapterError, BaseAdapter, autocast_dtype, resolve_device


def _free_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Sam3Adapter(BaseAdapter):
    backend_id = "sam3"
    display_name = "SAM 3"

    def __init__(self, model_id, device="auto"):
        BaseAdapter.__init__(self, model_id, device)
        self._concept_model = None
        self._concept_processor = None

    def load(self, report=None):
        self.device = resolve_device(self.requested_device)
        if report:
            report("Loading %s on %s" % (self.model_id, self.device))
        try:
            from transformers import Sam3Model, Sam3Processor  # noqa: F401
        except ImportError as error:
            raise AdapterError(
                "This transformers build has no SAM 3 support (%s). "
                "Re-run Setup to update the environment." % error
            )
        # Neither head is built here: which one is needed depends on the
        # mode, and they are large enough that loading both is wasteful.
        if report:
            report("Model ready.")

    def capabilities(self):
        return {
            "modes": ["everything", "text"],
            "supports_text": True,
            "text_hint": "A short noun phrase, e.g. \"red car\" or \"person\"",
        }

    # -- concept (text) prompting -----------------------------------------

    def release_unused(self, keeping, report=None):
        """Keep only the head the requested mode needs."""
        freed = False
        if keeping == "everything" and self._concept_model is not None:
            if report:
                report("Releasing the concept head to free memory")
            self._concept_model = None
            self._concept_processor = None
            freed = True
        if keeping == "text" and self._pipeline is not None:
            if report:
                report("Releasing the grid-prompt head to free memory")
            self._pipeline = None
            freed = True
        if freed:
            _free_memory()
        return freed

    def _concept(self, report=None):
        if self._concept_model is not None:
            return self._concept_model, self._concept_processor
        import torch
        from transformers import Sam3Model, Sam3Processor

        if report:
            report("Loading the concept-segmentation head")
        kwargs = {}
        dtype = autocast_dtype(self.device)
        if dtype is not None:
            kwargs["dtype"] = dtype
        model = Sam3Model.from_pretrained(self.model_id, **kwargs)
        model = model.to(self.device)
        model.eval()
        self._concept_model = model
        self._concept_processor = Sam3Processor.from_pretrained(self.model_id)
        return self._concept_model, self._concept_processor

    def segment_text(self, image, text, params, report=None):
        """Return one segment per instance matching the noun phrase."""
        import torch

        text = (text or "").strip()
        if not text:
            raise AdapterError("Enter a word or short phrase to search for.")

        self.release_unused("text", report)
        model, processor = self._concept(report)
        if report:
            report("Searching for %r" % text)

        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        original_sizes = inputs.get("original_sizes")
        if hasattr(original_sizes, "tolist"):
            original_sizes = original_sizes.tolist()
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(params.get("text_threshold", 0.5)),
            mask_threshold=float(params.get("mask_threshold", 0.5)),
            target_sizes=original_sizes,
        )[0]

        from .base import _to_bool_array

        masks = results.get("masks", [])
        scores = results.get("scores", [])
        segments = []
        for index, mask in enumerate(masks):
            array = _to_bool_array(mask)
            if array is None or not array.any():
                continue
            try:
                score = float(scores[index])
            except (IndexError, TypeError, ValueError):
                score = 1.0
            segments.append({"mask": array, "score": score, "label": text})
        if report:
            report("Found %d instance(s) of %r" % (len(segments), text))
        return segments
