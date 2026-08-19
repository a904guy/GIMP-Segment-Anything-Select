#!/usr/bin/env python3
"""Segment Anything helper daemon for the GIMP plug-in.

Runs inside the private environment created by the setup wizard, owns the
loaded model, and answers newline-delimited JSON requests over a loopback
socket.  It stays resident between plug-in invocations (subject to an idle
timeout) so that the expensive model load happens once per session rather
than once per use.

Protocol
--------
Request   {"id": 1, "token": "...", "op": "segment", ...}
Progress  {"id": 1, "progress": {"message": "...", "fraction": 0.4}}
Success   {"id": 1, "ok": true, "result": {...}}
Failure   {"id": 1, "ok": false, "error": "..."}
"""

import argparse
import base64
import datetime
import io
import json
import os
import secrets
import socketserver
import sys
import threading
import time
import traceback
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import maskutil  # noqa: E402

PROTOCOL_VERSION = 1
SERVER_VERSION = "1.0.0"

_log_file = None
_log_lock = threading.Lock()


def log(message):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (stamp, message)
    with _log_lock:
        if _log_file:
            try:
                _log_file.write(line + "\n")
                _log_file.flush()
            except (OSError, ValueError):
                pass
        else:
            print(line, file=sys.stderr)


# --------------------------------------------------------------------------
# Model session
# --------------------------------------------------------------------------


# How many past segmentations stay addressable.  Sessions hold only run
# lengths, so several are cheap, and more than one is needed: two open images
# mean two dialogs, and the second must not invalidate the first.
MAX_SESSIONS = 6


class Engine(object):
    """Owns the adapter and recent segmentation results."""

    def __init__(self):
        self.lock = threading.RLock()
        self.adapter = None
        self.backend = None
        self.model_id = None
        self.sessions = {}          # session id -> {segments, infer_size, image_size}
        self.session_order = []     # oldest first

    # -- model -------------------------------------------------------------

    def ensure_model(self, backend, model_id, device, report=None):
        if (self.adapter is not None
                and self.backend == backend
                and self.model_id == model_id):
            return self.adapter
        import adapters

        if self.adapter is not None:
            log("switching model: %s -> %s" % (self.model_id, model_id))
            self.adapter = None
            _free_memory()

        adapter = adapters.create(backend, model_id, device)
        adapter.load(report)
        self.adapter = adapter
        self.backend = backend
        self.model_id = model_id
        return adapter

    def describe(self):
        if self.adapter is None:
            return {"loaded": False}
        info = self.adapter.describe()
        info["loaded"] = True
        info.update(self.adapter.capabilities())
        return info

    # -- segmentation ------------------------------------------------------

    def segment(self, params, report):
        from PIL import Image

        image_path = params["image_path"]
        image = Image.open(image_path).convert("RGB")
        full_w, full_h = image.size
        image_size = (full_w, full_h)

        max_side = int(params.get("inference_max_side", 1536))
        infer_image = image
        if max_side and max(full_w, full_h) > max_side:
            scale = max_side / float(max(full_w, full_h))
            infer_size = (max(1, int(round(full_w * scale))),
                          max(1, int(round(full_h * scale))))
            infer_image = image.resize(infer_size, Image.LANCZOS)
            report("Working at %dx%d for speed (image is %dx%d)"
                   % (infer_size[0], infer_size[1], full_w, full_h), 0.05)

        adapter = self.ensure_model(
            params["backend"], params["model"], params.get("device", "auto"),
            lambda message: report(message, 0.1),
        )

        mode = params.get("mode", "everything")
        started = time.time()
        if mode == "text":
            if not hasattr(adapter, "segment_text"):
                raise RuntimeError("%s cannot search by text." % adapter.display_name)
            report("Searching the image...", 0.2)
            found = adapter.segment_text(
                infer_image, params.get("text", ""), params,
                lambda message: report(message, 0.3),
            )
        else:
            report("Segmenting the whole image...", 0.2)
            found = adapter.segment_everything(
                infer_image, params, lambda message: report(message, 0.3),
            )
        log("model produced %d masks in %.1fs" % (len(found), time.time() - started))

        return self._package(found, params, report, infer_image.size, image_size)

    def _package(self, found, params, report, infer_size, image_size):
        """Filter, sort, nest and encode what the model returned."""
        import numpy as np

        report("Sorting %d regions..." % len(found), 0.75)
        infer_w, infer_h = infer_size
        total_pixels = float(infer_w * infer_h)
        min_frac = float(params.get("min_area_percent", 0.05)) / 100.0
        # 255 is the label map's "no region here" marker, so ids must stay
        # below it however the setting was edited.
        max_segments = max(1, min(254, int(params.get("max_segments", 254))))

        kept = []
        for item in found:
            mask = item["mask"]
            if mask.shape != (infer_h, infer_w):
                mask = maskutil.downscale_mask(mask, (infer_w, infer_h))
            area = int(mask.sum())
            if area <= 0 or area / total_pixels < min_frac:
                continue
            kept.append({"mask": mask, "score": item.get("score", 1.0),
                         "label": item.get("label"), "area": area})

        # Drop near-duplicates: the point grid often finds the same object
        # from several seed points.
        kept = _deduplicate(kept, float(params.get("dedup_iou", 0.85)))

        kept.sort(key=lambda item: -item["area"])
        truncated = 0
        if len(kept) > max_segments:
            truncated = len(kept) - max_segments
            kept = kept[:max_segments]

        masks = [item["mask"] for item in kept]
        report("Working out which regions sit inside which...", 0.85)
        parents = maskutil.build_hierarchy(masks) if masks else []

        preview_max = int(params.get("preview_max_side", 1400))
        full_w, full_h = image_size
        scale = min(1.0, preview_max / float(max(full_w, full_h) or 1))
        preview_size = (max(1, int(round(full_w * scale))),
                        max(1, int(round(full_h * scale))))

        report("Preparing the preview...", 0.92)
        label_map = (maskutil.build_label_map(masks, preview_size)
                     if masks else np.full(preview_size[::-1], 255, dtype=np.uint8))

        scale_x = full_w / float(infer_w)
        scale_y = full_h / float(infer_h)

        stored = []
        payload = []
        counters = {}
        for index, item in enumerate(kept):
            mask = item["mask"]
            label = item.get("label")
            if label:
                counters[label] = counters.get(label, 0) + 1
                name = "%s %d" % (label, counters[label])
            else:
                name = "Segment %d" % (index + 1)

            bbox = maskutil.bbox_of(mask)
            full_bbox = [int(round(bbox[0] * scale_x)), int(round(bbox[1] * scale_y)),
                         int(round(bbox[2] * scale_x)), int(round(bbox[3] * scale_y))]
            preview_mask = maskutil.downscale_mask(mask, preview_size)

            stored.append(maskutil.encode_rle(mask))
            payload.append({
                "id": index,
                "name": name,
                "label": label,
                "score": round(float(item.get("score", 1.0)), 4),
                "area": int(round(item["area"] * scale_x * scale_y)),
                "area_frac": round(item["area"] / (infer_w * infer_h), 6),
                "bbox": full_bbox,
                "parent": parents[index] if index < len(parents) else None,
                "rle_preview": maskutil.encode_rle(preview_mask),
            })

        session_id = uuid.uuid4().hex
        self.sessions[session_id] = {
            "segments": stored,
            "infer_size": infer_size,
            "image_size": image_size,
        }
        self.session_order.append(session_id)
        while len(self.session_order) > MAX_SESSIONS:
            self.sessions.pop(self.session_order.pop(0), None)

        return {
            "session": session_id,
            "image_size": [full_w, full_h],
            "preview_size": list(preview_size),
            "infer_size": [infer_w, infer_h],
            "label_map": _encode_label_png(label_map),
            "segments": payload,
            "truncated": truncated,
        }

    # -- mask retrieval ----------------------------------------------------

    def combine(self, params):
        """Union the requested segments and return one full-resolution mask."""
        import numpy as np

        session = self.sessions.get(params.get("session"))
        if session is None:
            raise RuntimeError(
                "This segmentation is no longer available. "
                "Press Find Regions to run it again."
            )
        segments = session["segments"]
        ids = [int(i) for i in params.get("ids", [])]
        infer_w, infer_h = session["infer_size"]
        full_w, full_h = session["image_size"]

        union = np.zeros((infer_h, infer_w), dtype=bool)
        for index in ids:
            if index < 0 or index >= len(segments):
                continue
            union |= maskutil.decode_rle(segments[index], infer_w, infer_h)

        if (infer_w, infer_h) != (full_w, full_h):
            union = maskutil.upscale_mask(union, (full_w, full_h))
        return {
            "width": full_w,
            "height": full_h,
            "rle": maskutil.encode_rle(union),
            "empty": not bool(union.any()),
        }


def _deduplicate(items, iou_threshold):
    """Drop masks that are essentially the same region as a bigger one."""
    import numpy as np

    if len(items) < 2 or iou_threshold >= 1.0:
        return items
    grid = 192
    height, width = items[0]["mask"].shape
    scale = max(height, width) / float(grid)
    small_size = (max(1, int(round(width / scale))), max(1, int(round(height / scale))))
    stack = np.stack([
        maskutil.downscale_mask(item["mask"], small_size).ravel() for item in items
    ]).astype(np.float32)
    areas = stack.sum(axis=1)
    intersection = stack @ stack.T

    order = sorted(range(len(items)), key=lambda i: -areas[i])
    keep = []
    for index in order:
        duplicate = False
        for chosen in keep:
            union = areas[index] + areas[chosen] - intersection[index, chosen]
            if union > 0 and intersection[index, chosen] / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(index)
    keep_set = set(keep)
    return [item for i, item in enumerate(items) if i in keep_set]


def _encode_label_png(label_map):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(label_map, mode="L").save(buffer, format="PNG", optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _flush_logs():
    with _log_lock:
        if _log_file:
            try:
                _log_file.flush()
                os.fsync(_log_file.fileno())
            except (OSError, ValueError):
                pass


def _free_memory():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


class Handler(socketserver.StreamRequestHandler):
    timeout = 1800

    def handle(self):
        server = self.server
        try:
            line = self.rfile.readline()
        except OSError:
            return
        if not line:
            return
        request_id = None
        try:
            request = json.loads(line.decode("utf-8"))
            request_id = request.get("id")
            if not secrets.compare_digest(str(request.get("token", "")), server.token):
                self._reply({"id": request_id, "ok": False, "error": "Bad token"})
                return
            server.touch()
            result = self._dispatch(request)
            self._reply({"id": request_id, "ok": True, "result": result})
        except Exception as error:  # noqa: BLE001 - every failure is reported back
            log("request failed: %s" % traceback.format_exc())
            self._reply({"id": request_id, "ok": False, "error": _friendly(error)})
        finally:
            server.touch()

    # -- helpers -----------------------------------------------------------

    def _reply(self, payload):
        try:
            self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.wfile.flush()
        except OSError:
            pass

    def _progress(self, request_id):
        def report(message, fraction=None):
            log("  %s" % message)
            self._reply({"id": request_id,
                         "progress": {"message": message, "fraction": fraction}})
        return report

    def _dispatch(self, request):
        server = self.server
        engine = server.engine
        op = request.get("op")
        report = self._progress(request.get("id"))

        if op == "ping":
            info = {"version": SERVER_VERSION, "protocol": PROTOCOL_VERSION,
                    "pid": os.getpid()}
            info.update(engine.describe())
            return info

        if op == "shutdown":
            server.request_shutdown()
            return {"stopping": True}

        if op == "load":
            with engine.lock:
                engine.ensure_model(request["backend"], request["model"],
                                    request.get("device", "auto"),
                                    lambda message: report(message))
                return engine.describe()

        if op == "segment":
            with engine.lock:
                return engine.segment(request, report)

        if op == "combine":
            with engine.lock:
                return engine.combine(request)

        if op == "unload":
            with engine.lock:
                engine.adapter = None
                engine.sessions = {}
                engine.session_order = []
                _free_memory()
                return {"loaded": False}

        raise RuntimeError("Unknown operation: %r" % op)


def _friendly(error):
    """Turn common failures into something a GIMP user can act on."""
    text = str(error)
    lowered = text.lower()
    gated_markers = ("gatedrepo", "gated repo", "awaiting a review",
                     "401 client error", "403 client error",
                     "is not authorized", "access to model")
    if "out of memory" in lowered:
        return (text + "\n\nThe graphics card ran out of memory. Try a smaller "
                "model, a coarser Detail setting, or a smaller working size in "
                "Setup. Closing other GPU applications also helps.")
    if any(marker in lowered for marker in gated_markers):
        return (text + "\n\nSAM 3's weights are gated: accept the licence at "
                "https://huggingface.co/facebook/sam3 and paste an access token "
                "into the plug-in's Setup dialog.")
    return text


class Server(socketserver.ThreadingTCPServer):
    # On Windows SO_REUSEADDR lets an unrelated process bind a port already
    # in use, so it is enabled only on POSIX, where it merely skips TIME_WAIT.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def __init__(self, state_file, idle_timeout):
        socketserver.ThreadingTCPServer.__init__(self, ("127.0.0.1", 0), Handler)
        self.token = secrets.token_hex(24)
        self.engine = Engine()
        self.state_file = state_file
        self.idle_timeout = idle_timeout
        self._last_used = time.time()
        self._stop = threading.Event()

    def touch(self):
        self._last_used = time.time()

    def request_shutdown(self):
        self._stop.set()
        threading.Thread(target=self.shutdown, daemon=True).start()

    def write_state(self):
        state = {
            "port": self.server_address[1],
            "token": self.token,
            "pid": os.getpid(),
            "version": SERVER_VERSION,
            "protocol": PROTOCOL_VERSION,
            "started": time.time(),
        }
        directory = os.path.dirname(self.state_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, self.state_file)

    def clear_state(self):
        try:
            os.remove(self.state_file)
        except OSError:
            pass

    def watch_idle(self):
        while not self._stop.is_set():
            if self._stop.wait(15.0):
                return
            if self.idle_timeout <= 0:
                continue
            idle = time.time() - self._last_used
            if idle > self.idle_timeout:
                log("idle for %.0fs, shutting down" % idle)
                self.request_shutdown()
                return


def main(argv=None):
    global _log_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--idle-timeout", type=int, default=900)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args(argv)

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        path = os.path.join(args.log_dir, "backend.log")
        try:
            if os.path.exists(path) and os.path.getsize(path) > 1024 * 1024:
                os.replace(path, path + ".1")
        except OSError:
            pass
        _log_file = open(path, "a", encoding="utf-8")

    log("=" * 60)
    log("backend %s starting (python %s)" % (SERVER_VERSION, sys.version.split()[0]))

    server = Server(args.state_file, args.idle_timeout)
    server.write_state()
    log("listening on 127.0.0.1:%d" % server.server_address[1])

    threading.Thread(target=server.watch_idle, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.clear_state()
        log("backend stopped")

    # PyTorch and CUDA leave non-daemon threads behind, so returning normally
    # can leave the process resident and still holding GPU memory.  Logs and
    # state have already been flushed at this point.
    _flush_logs()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
