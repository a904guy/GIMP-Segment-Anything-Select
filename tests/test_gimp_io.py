"""GIMP-side image and mask handling.

Must run inside GIMP:

    gimp-console -idf -s --quit --batch-interpreter python-fu-eval \
        -b 'exec(open("tests/test_gimp_io.py").read())'

``tests/run.py`` does this for you.
"""

import os
import sys
import traceback

# GIMP runs this through exec(), so there is no __file__ to work from.
HERE = os.environ.get("SAM_GIMP_TEST_DIR") or os.path.join(os.getcwd(), "tests")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plug-ins", "segment-anything"))

_results = []


def check(condition, message):
    _results.append(("ok  " if condition else "FAIL", message))


def run():
    import gi
    gi.require_version("Gimp", "3.0")
    gi.require_version("Gegl", "0.4")
    from gi.repository import Gegl, Gimp

    from samgimp import gimpio

    width, height = 200, 120
    image = Gimp.Image.new(width, height, Gimp.ImageBaseType.RGB)
    base = Gimp.Layer.new(image, "base", width, height,
                          Gimp.ImageType.RGB_IMAGE, 100.0, Gimp.LayerMode.NORMAL)
    image.insert_layer(base, None, 0)
    base.fill(Gimp.FillType.WHITE)

    # A layer smaller than the canvas, to exercise the offset handling.
    small = Gimp.Layer.new(image, "small", 80, 60,
                           Gimp.ImageType.RGBA_IMAGE, 100.0, Gimp.LayerMode.NORMAL)
    image.insert_layer(small, None, 0)
    small.fill(Gimp.FillType.TRANSPARENT)
    small.set_offsets(50, 30)

    pixbuf = gimpio.composite_pixbuf(image)
    check(pixbuf.get_width() == width and pixbuf.get_height() == height,
          "the flattened composite comes back at full size")
    png = gimpio.write_png(pixbuf)
    check(os.path.getsize(png) > 0, "the composite exports to a PNG")
    check(gimpio.scaled_pixbuf(pixbuf, 100).get_width() == 100,
          "the preview scales to the requested cap")

    mask = bytearray(width * height)
    for row in range(10, 40):
        mask[row * width + 20:row * width + 70] = b"\xff" * 50

    gimpio.apply_selection(image, mask, width, height, "replace")
    bounds = Gimp.Selection.bounds(image)
    check((bounds.x1, bounds.y1, bounds.x2, bounds.y2) == (20, 10, 70, 40),
          "a mask becomes a selection with exactly the right bounds")
    check(len(image.get_channels()) == 0,
          "the temporary channel is cleaned up afterwards")

    gimpio.apply_selection(image, mask, width, height, "replace", feather=3.0)
    feathered = Gimp.Selection.bounds(image)
    check(feathered.x1 < 20 and feathered.x2 > 70,
          "feathering widens the selection")

    Gimp.Selection.none(image)
    layer_mask = gimpio.add_layer_mask(image, small, mask, width, height)
    data = layer_mask.get_buffer().get(Gegl.Rectangle.new(0, 0, 80, 60), 1.0,
                                       "Y' u8", Gegl.AbyssPolicy.CLAMP)
    lit = sum(1 for value in data if value > 127)
    check(lit == 200,
          "a layer mask on an offset layer gets exactly the overlapping pixels")

    new_layer = gimpio.copy_to_new_layer(image, base, mask, width, height, "cut")
    pixels = new_layer.get_buffer().get(Gegl.Rectangle.new(0, 0, width, height), 1.0,
                                        "R'G'B'A u8", Gegl.AbyssPolicy.CLAMP)
    opaque = sum(1 for i in range(3, len(pixels), 4) if pixels[i] > 127)
    check(opaque == 1500, "a new layer keeps only the masked pixels")
    check(new_layer.has_alpha(), "the new layer has transparency")

    channel = gimpio.save_as_channel(image, mask, width, height, "kept")
    check(channel.get_name() == "kept" and not channel.get_visible(),
          "a saved channel is created, hidden by default")

    image.delete()
    os.remove(png)


try:
    run()
except Exception:
    _results.append(("FAIL", "exception: " + traceback.format_exc()))

_report = "\n".join("  %s%s" % (status, message) for status, message in _results)
_failed = any(status.startswith("FAIL") for status, _ in _results)
with open(os.path.join(HERE, ".gimp_io_result"), "w") as handle:
    handle.write(("FAILED\n" if _failed else "PASSED\n") + _report)
print(_report)
