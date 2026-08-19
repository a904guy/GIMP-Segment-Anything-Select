"""Dialog wiring that needs GIMP: the mode dropdown reflects the model.

Run through ``tests/run.py --gimp``.
"""

import os
import sys
import traceback

HERE = os.environ.get("SAM_GIMP_TEST_DIR") or os.path.join(os.getcwd(), "tests")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plug-ins", "segment-anything"))

_results = []


def check(condition, message):
    _results.append(("ok  " if condition else "FAIL", message))


def run():
    import gi
    gi.require_version("Gimp", "3.0")
    gi.require_version("GimpUi", "3.0")
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gimp, GimpUi

    GimpUi.init("sam-dialog-test")
    from samgimp import dialog_main, env

    image = Gimp.Image.new(64, 48, Gimp.ImageBaseType.RGB)
    layer = Gimp.Layer.new(image, "base", 64, 48, Gimp.ImageType.RGB_IMAGE,
                           100.0, Gimp.LayerMode.NORMAL)
    image.insert_layer(layer, None, 0)
    layer.fill(Gimp.FillType.WHITE)

    settings = env.load_settings()
    # Build the window but never let its idle handler run, so no segmentation
    # starts and the test needs no model weights.
    dialog = dialog_main.SegmentAnythingDialog(image, [layer], settings)
    try:
        rows = [(row[0], row[2]) for row in dialog.mode_store]
        check([row[0] for row in rows] == [dialog_main.MODE_EVERYTHING,
                                           dialog_main.MODE_TEXT],
              "the Find dropdown offers both modes")

        for backend, expected in (("sam2", False), ("sam3", True)):
            dialog._capabilities = {"supports_text": backend == "sam3"}
            dialog._model_name = "SAM 3" if backend == "sam3" else "SAM 2.1"
            dialog._sync_mode_widgets()
            selectable = dialog.mode_store[1][2]
            check(selectable is expected,
                  "with %s installed, 'Find by description' is %s"
                  % (backend.upper(), "selectable" if expected else "greyed out"))
            check(dialog.mode_hint.get_visible() is (not expected),
                  "the explanation is %s for %s"
                  % ("shown" if not expected else "hidden", backend.upper()))

        # With SAM 2 the mode must fall back rather than stick on text.
        dialog._capabilities = {"supports_text": False}
        dialog.mode_combo.set_active_id(dialog_main.MODE_TEXT)
        dialog._sync_mode_widgets()
        check(dialog.mode_combo.get_active_id() == dialog_main.MODE_EVERYTHING,
              "text mode cannot be forced on a model that cannot read text")
        check(not dialog.text_entry.get_visible(),
              "the description box stays hidden when it cannot be used")

        # With SAM 3 the text box appears.
        dialog._capabilities = {"supports_text": True}
        dialog.mode_combo.set_active_id(dialog_main.MODE_TEXT)
        dialog._sync_mode_widgets()
        check(dialog.mode_combo.get_active_id() == dialog_main.MODE_TEXT,
              "text mode sticks when the model supports it")
        check(dialog.text_entry.get_visible(),
              "the description box appears in text mode")
        check(not dialog.detail_combo.get_visible(),
              "the Detail control is hidden in text mode")
    finally:
        dialog.destroy()
        image.delete()


try:
    run()
except Exception:
    _results.append(("FAIL", "exception: " + traceback.format_exc()))

_report = "\n".join("  %s%s" % (status, message) for status, message in _results)
_failed = any(status.startswith("FAIL") for status, _ in _results)
with open(os.path.join(HERE, ".gimp_dialog_result"), "w") as handle:
    handle.write(("FAILED\n" if _failed else "PASSED\n") + _report)
print(_report)
