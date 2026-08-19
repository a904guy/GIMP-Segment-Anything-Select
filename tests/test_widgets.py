"""The two selection panes, driven offscreen with a real backend payload."""

import base64
import os

import conftest_paths  # noqa: F401
from conftest_paths import Sandbox, check

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402


def _pump(milliseconds=300):
    done = [False]
    GLib.timeout_add(milliseconds, lambda: (done.__setitem__(0, True), False)[1])
    while not done[0]:
        Gtk.main_iteration_do(True)


def _pixbuf_from_base64(data):
    loader = GdkPixbuf.PixbufLoader.new_with_type("png")
    loader.write(base64.b64decode(data))
    loader.close()
    return loader.get_pixbuf()


def run(save_render=None):
    with Sandbox() as sandbox:
        from samgimp import canvas as canvas_module, client, seglist

        width, height = 640, 480
        image = Image.new("RGB", (width, height), (240, 240, 235))
        draw = ImageDraw.Draw(image)
        draw.rectangle([64, 48, 576, 432], fill=(70, 110, 180))
        draw.rectangle([128, 96, 256, 192], fill=(220, 90, 70))
        draw.rectangle([384, 288, 512, 384], fill=(90, 190, 120))
        path = os.path.join(sandbox.home, "scene.png")
        image.save(path)

        backend = client.BackendClient()
        result = backend.call("segment", {
            "image_path": path, "backend": "sam2", "model": "stub", "device": "cpu",
            "mode": "everything", "inference_max_side": 512,
            "preview_max_side": 400, "min_area_percent": 0.05,
        })
        segments = result["segments"]

        picks = []
        canvas = canvas_module.SegmentCanvas(
            on_pick=lambda seg_id, additive: picks.append((seg_id, additive)))
        listing = seglist.SegmentList()

        preview = GdkPixbuf.Pixbuf.new_from_file_at_scale(
            path, result["preview_size"][0], result["preview_size"][1], False)
        canvas.set_image(preview)
        canvas.set_segments(segments, _pixbuf_from_base64(result["label_map"]),
                            result["image_size"])
        listing.set_segments(segments)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.pack1(listing, False, False)
        paned.pack2(canvas, True, False)
        paned.set_position(280)
        window = Gtk.OffscreenWindow()
        window.set_default_size(1000, 560)
        window.add(paned)
        window.show_all()
        _pump()

        scale_x = result["preview_size"][0] / float(width)
        scale_y = result["preview_size"][1] / float(height)

        def hit(x, y):
            """Map full-image coordinates to a click inside the drawing area."""
            origin_x, origin_y = canvas._origin()
            return canvas._segment_at(origin_x + x * scale_x * canvas.zoom,
                                      origin_y + y * scale_y * canvas.zoom)

        check(hit(320, 240) == 0, "clicking the background region picks it")
        check(hit(190, 140) == 1, "clicking a nested region picks the nested one")
        check(hit(450, 330) == 2, "clicking the other nested region picks it")
        check(hit(10, 10) is None, "clicking outside every region picks nothing")

        # The image is centred when it does not fill the pane; clicking the
        # empty margin beside it must not pick anything.
        origin_x, origin_y = canvas._origin()
        check(origin_x >= 0 and origin_y >= 0, "the image origin is never negative")
        if origin_x > 2:
            check(canvas._segment_at(origin_x / 2.0, origin_y + 10) is None,
                  "clicking the margin beside the image picks nothing")

        listing.set_selected_ids([1, 2])
        check(sorted(listing.get_selected_ids()) == [1, 2],
              "the list reflects a programmatic selection")

        listing.include_children = True
        check(listing.expand_with_children([0]) == [0, 1, 2],
              "selecting a parent can pull in its nested pieces")
        listing.include_children = False
        check(listing.expand_with_children([0]) == [0],
              "with nesting off, only the clicked region is used")
        check(sorted(listing.descendants_of(0)) == [1, 2],
              "descendants are found through the tree")

        canvas.set_selection([1, 2])
        check(canvas.selected == {1, 2}, "the canvas takes the same selection")

        canvas.fit_to_window()
        _pump(150)
        check(canvas.zoom > 0, "fit-to-window computes a usable zoom")
        before = canvas.zoom
        canvas.zoom_by(2.0)
        check(abs(canvas.zoom - before * 2.0) < 1e-6, "zooming in doubles the scale")

        window.queue_draw()
        canvas.area.queue_draw()
        _pump()
        rendered = window.get_pixbuf()
        check(rendered is not None and rendered.get_width() > 0,
              "the panes render without error")
        if save_render:
            rendered.savev(save_render, "png", [], [])
            print("  ..  render written to %s" % save_render)

        backend.shutdown()


if __name__ == "__main__":
    import sys
    run(save_render=sys.argv[1] if len(sys.argv) > 1 else None)
