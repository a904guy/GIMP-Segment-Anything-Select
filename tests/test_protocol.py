"""End-to-end protocol test against a stub model: no GPU or weights needed."""

import base64
import io
import json
import os
import socket
import sys

import conftest_paths as helpers
from conftest_paths import Sandbox, check

from PIL import Image


def run():
    with Sandbox() as sandbox:
        from samgimp import client, env
        sys.path.insert(0, helpers.BACKEND)
        import maskutil

        image_path = os.path.join(sandbox.home, "scene.png")
        Image.new("RGB", (800, 600), (30, 90, 160)).save(image_path)

        backend = client.BackendClient()
        info = backend.ping()
        check(info["protocol"] == client.PROTOCOL_VERSION,
              "the daemon starts and speaks the expected protocol")
        check(info["loaded"] is False, "no model is loaded until one is asked for")

        messages = []
        request = {
            "image_path": image_path, "backend": "sam2", "model": "stub/model",
            "device": "cpu", "mode": "everything", "points_per_side": 16,
            "inference_max_side": 512, "preview_max_side": 400,
            "min_area_percent": 0.05,
        }
        result = backend.call("segment", request,
                              on_progress=lambda p: messages.append(p["message"]))

        check(len(messages) >= 3, "progress is reported while working")
        check(result["image_size"] == [800, 600], "the full image size is reported")
        check(result["preview_size"] == [400, 300], "the preview is scaled to the cap")
        segments = result["segments"]
        check(len(segments) == 3,
              "specks are dropped and duplicates removed (5 masks -> 3 regions)")
        check([s["parent"] for s in segments] == [None, 0, 0],
              "the two inner regions are nested under the outer one")
        check(segments[0]["area"] > segments[1]["area"],
              "regions are ordered largest first")

        label = Image.open(io.BytesIO(base64.b64decode(result["label_map"])))
        check(label.size == (400, 300) and label.mode == "L",
              "a single-channel label map accompanies the regions")

        combined = backend.call("combine",
                                {"session": result["session"], "ids": [1, 2]})
        check((combined["width"], combined["height"]) == (800, 600),
              "combining returns a full-resolution mask")
        mask = maskutil.decode_rle(combined["rle"], combined["width"], combined["height"])
        check(30000 < int(mask.sum()) < 45000,
              "the combined mask covers both chosen regions")

        text_result = backend.call("segment", dict(request, mode="text", text="cat"))
        check([s["name"] for s in text_result["segments"]] == ["cat 1", "cat 2", "cat 3"],
              "text mode names regions after the phrase")

        # Both sessions must still work: a user can have two images open.
        for name, res in (("first", result), ("second", text_result)):
            again = backend.call("combine", {"session": res["session"], "ids": [0]})
            check(not again["empty"], "the %s session is still usable" % name)

        try:
            backend.call("combine", {"session": "nope", "ids": [0]})
            raise AssertionError("an unknown session should be refused")
        except client.BackendError as error:
            check("no longer available" in str(error),
                  "an unknown session gives an actionable message")

        state = json.load(open(env.daemon_path()))
        sock = socket.create_connection(("127.0.0.1", state["port"]), 5)
        sock.sendall(json.dumps({"id": 1, "token": "wrong", "op": "ping"}).encode() + b"\n")
        reply = json.loads(sock.recv(4096).decode())
        sock.close()
        check(reply["ok"] is False and reply["error"] == "Bad token",
              "requests without the right token are rejected")

        backend.shutdown()
        check(not client.is_running(), "the daemon stops when asked")


if __name__ == "__main__":
    run()
