"""The main Segment Anything window.

The left pane lists the regions the model found, nested by containment.
The right pane draws those regions on the image for direct picking.  Both
panes edit one shared set of selected region ids and are kept in step.

The controls along the bottom choose what that set becomes when applied: a
selection, a layer mask, a new layer or a stored channel.
"""

import threading

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, Gimp, GimpUi, GLib, Gtk  # noqa: E402

import base64  # noqa: E402
import os  # noqa: E402

from . import canvas as canvas_module  # noqa: E402
from . import client, dialog_setup, env, gimpio, log, rle, seglist  # noqa: E402

RESPONSE_APPLY = 1
RESPONSE_CLOSE = 2

MODE_EVERYTHING = "everything"
MODE_TEXT = "text"

OUTPUTS = [
    ("selection", "Selection"),
    ("layer_mask", "Layer mask on the active layer"),
    ("new_layer", "New layer from the selected regions"),
    ("channel", "Saved channel"),
]

SELECTION_OPS = [
    ("replace", "Replace selection"),
    ("add", "Add to selection"),
    ("subtract", "Subtract from selection"),
    ("intersect", "Intersect with selection"),
]

DETAIL_LEVELS = [
    (16, "Coarse - fewer, bigger regions"),
    (24, "Medium"),
    (32, "Fine (default)"),
    (48, "Very fine - many small regions, slower"),
    (64, "Maximum - slowest"),
]


class SegmentAnythingDialog(GimpUi.Dialog):
    def __init__(self, image, drawables, settings):
        GimpUi.Dialog.__init__(
            self, title="Segment Anything", role="sam-gimp-main",
            use_header_bar=True,
        )
        self.image = image
        self.drawables = drawables or []
        self.settings = dict(settings)
        self.client = client.BackendClient()

        self.result = None            # segmentation payload from the backend
        self.selected_ids = []
        self._worker = None
        self._busy = False
        self._syncing = False
        self._image_path = None
        self._capabilities = {}
        self._model_name = "SAM 2.1"

        self.set_default_size(1180, 780)
        self.close_button = self.add_button("_Close", RESPONSE_CLOSE)
        self.apply_button = self.add_button("_Apply", RESPONSE_APPLY)
        self.set_default_response(RESPONSE_APPLY)
        self.connect("response", self._on_response)

        content = self.get_content_area()
        content.set_spacing(8)
        content.set_border_width(10)
        content.pack_start(self._build_toolbar(), False, False, 0)
        content.pack_start(self._build_panes(), True, True, 0)
        content.pack_start(self._build_output_row(), False, False, 0)
        content.pack_start(self._build_status(), False, False, 0)

        self.show_all()
        self._sync_mode_widgets()
        self._update_apply_sensitivity()
        GLib.idle_add(self._start_up)

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _build_toolbar(self):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # ComboBoxText cannot mark a row unavailable, so the model carries a
        # "sensitive" column.  Without it the row would accept a click and
        # then revert, with no indication of why.
        self.mode_store = Gtk.ListStore(str, str, bool)
        self.mode_store.append([MODE_EVERYTHING, "Everything in the image", True])
        self.mode_store.append([MODE_TEXT, "Find by description", False])
        self.mode_combo = Gtk.ComboBox.new_with_model(self.mode_store)
        mode_cell = Gtk.CellRendererText()
        self.mode_combo.pack_start(mode_cell, True)
        self.mode_combo.add_attribute(mode_cell, "text", 1)
        self.mode_combo.add_attribute(mode_cell, "sensitive", 2)
        self.mode_combo.set_id_column(0)
        self.mode_combo.set_active_id(MODE_EVERYTHING)
        self.mode_combo.connect("changed", lambda _c: self._sync_mode_widgets())
        box.pack_start(Gtk.Label(label="Find:"), False, False, 0)
        box.pack_start(self.mode_combo, False, False, 0)

        self.text_entry = Gtk.Entry()
        self.text_entry.set_placeholder_text("e.g. red car")
        self.text_entry.set_text(self.settings.get("last_text_prompt", ""))
        self.text_entry.set_width_chars(18)
        self.text_entry.set_no_show_all(True)
        self.text_entry.connect("activate", lambda _e: self._start_segmentation())
        box.pack_start(self.text_entry, False, False, 0)

        # Explains the greyed-out row without needing the popup to be open.
        self.mode_hint = Gtk.Label()
        self.mode_hint.set_no_show_all(True)
        self.mode_hint.get_style_context().add_class("dim-label")
        self.mode_hint.set_markup(
            "<small>Searching by description needs SAM 3 (see Setup)</small>")
        self.mode_hint.set_tooltip_text(
            "SAM 2.1 finds regions but cannot read words. Install SAM 3 from "
            "the Setup window to search for things by name.")
        box.pack_start(self.mode_hint, False, False, 0)

        self.detail_combo = Gtk.ComboBoxText()
        for value, label in DETAIL_LEVELS:
            self.detail_combo.append(str(value), label)
        self.detail_combo.set_active_id(str(self.settings.get("points_per_side", 32)))
        self.detail_label = Gtk.Label(label="Detail:")
        box.pack_start(self.detail_label, False, False, 0)
        box.pack_start(self.detail_combo, False, False, 0)

        self.run_button = Gtk.Button(label="Find Regions")
        self.run_button.get_style_context().add_class("suggested-action")
        self.run_button.connect("clicked", lambda _b: self._start_segmentation())
        box.pack_start(self.run_button, False, False, 0)

        box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)

        self.show_all_toggle = Gtk.CheckButton(label="Tint all regions")
        self.show_all_toggle.set_active(True)
        self.show_all_toggle.connect(
            "toggled", lambda b: self.canvas.set_show_all_regions(b.get_active()))
        box.pack_start(self.show_all_toggle, False, False, 0)

        for label, tip, callback in (
            ("-", "Zoom out", lambda: self.canvas.zoom_by(1 / 1.25)),
            ("Fit", "Fit the image in the window", self.canvas_fit),
            ("+", "Zoom in", lambda: self.canvas.zoom_by(1.25)),
        ):
            button = Gtk.Button(label=label)
            button.set_tooltip_text(tip)
            button.connect("clicked", lambda _b, cb=callback: cb())
            box.pack_start(button, False, False, 0)

        spacer = Gtk.Box()
        box.pack_start(spacer, True, True, 0)

        self.model_label = Gtk.Label()
        self.model_label.get_style_context().add_class("dim-label")
        box.pack_start(self.model_label, False, False, 0)

        setup_button = Gtk.Button(label="Setup...")
        setup_button.set_tooltip_text("Change the model, or reinstall the runtime")
        setup_button.connect("clicked", lambda _b: self._open_setup())
        box.pack_start(setup_button, False, False, 0)
        return box

    def _build_panes(self):
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.segment_list = seglist.SegmentList(
            on_selection_changed=self._on_list_selection,
            on_hover=self._on_list_hover,
        )
        self.canvas = canvas_module.SegmentCanvas(
            on_pick=self._on_canvas_pick,
            on_hover=self._on_canvas_hover,
        )
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        heading = Gtk.Label()
        heading.set_markup("<b>Regions</b>")
        heading.set_xalign(0.0)
        left.pack_start(heading, False, False, 0)
        left.pack_start(self.segment_list, True, True, 0)
        paned.pack1(left, False, False)
        paned.pack2(self.canvas, True, False)
        paned.set_position(300)
        return paned

    def _build_output_row(self):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.pack_start(Gtk.Label(label="Apply as:"), False, False, 0)

        self.output_combo = Gtk.ComboBoxText()
        for value, label in OUTPUTS:
            self.output_combo.append(value, label)
        self.output_combo.set_active_id("selection")
        self.output_combo.connect("changed", lambda _c: self._sync_output_widgets())
        box.pack_start(self.output_combo, False, False, 0)

        self.op_combo = Gtk.ComboBoxText()
        for value, label in SELECTION_OPS:
            self.op_combo.append(value, label)
        self.op_combo.set_active_id(self.settings.get("selection_op", "replace"))
        box.pack_start(self.op_combo, False, False, 0)

        self.feather_label = Gtk.Label(label="Feather:")
        box.pack_start(self.feather_label, False, False, 0)
        adjustment = Gtk.Adjustment(
            value=float(self.settings.get("feather", 0.0)),
            lower=0.0, upper=100.0, step_increment=0.5, page_increment=5.0,
        )
        self.feather_spin = Gtk.SpinButton(adjustment=adjustment, digits=1)
        self.feather_spin.set_tooltip_text("Soften the mask edge, in pixels")
        box.pack_start(self.feather_spin, False, False, 0)

        box.pack_start(Gtk.Box(), True, True, 0)

        self.selection_label = Gtk.Label()
        self.selection_label.get_style_context().add_class("dim-label")
        box.pack_start(self.selection_label, False, False, 0)
        return box

    def _build_status(self):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(True)
        self.progress.set_text("")
        self.progress.set_valign(Gtk.Align.CENTER)
        box.pack_start(self.progress, True, True, 0)
        return box

    def canvas_fit(self):
        self.canvas.fit_to_window()

    # ------------------------------------------------------------------
    # Start-up
    # ------------------------------------------------------------------

    def _start_up(self):
        try:
            pixbuf = gimpio.composite_pixbuf(self.image)
        except Exception as error:  # noqa: BLE001
            log.exception("could not read the image")
            self._error("Could not read the image: %s" % error)
            return False

        self.full_pixbuf = pixbuf
        preview = gimpio.scaled_pixbuf(pixbuf, int(self.settings.get("preview_max_side", 1400)))
        self.canvas.set_image(preview)
        self._set_status("Preparing...")

        if not env.is_installed():
            self._set_status("The Segment Anything runtime is not installed yet.")
            self._prompt_install()
            return False

        self._describe_model()
        GLib.idle_add(self._start_segmentation)
        return False

    def _describe_model(self):
        settings = env.load_settings()
        name = "SAM 3" if settings.get("backend") == "sam3" else "SAM 2.1"
        self._model_name = name
        self.model_label.set_text("%s - %s" % (name, settings.get("model", "").split("/")[-1]))
        self._capabilities = {
            "supports_text": settings.get("backend") == "sam3",
        }
        self._sync_mode_widgets()

    def _prompt_install(self):
        dialog = Gtk.MessageDialog(
            transient_for=self, modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text="Set up Segment Anything?",
        )
        dialog.format_secondary_text(
            "The first run downloads a private Python environment and the "
            "model weights (1-4 GB depending on the model). It only happens "
            "once."
        )
        answer = dialog.run()
        dialog.destroy()
        if answer == Gtk.ResponseType.OK:
            self._open_setup()

    def _open_setup(self):
        dialog_setup.run_setup(parent=self, settings=env.load_settings())
        self.settings = env.load_settings()
        if env.is_installed():
            self._describe_model()
            self._set_status("Ready. Press Find Regions.")

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _sync_mode_widgets(self):
        supports_text = bool(self._capabilities.get("supports_text"))

        # Update row availability before reading the mode, so the dropdown
        # reflects the current model.
        self.mode_store[1][2] = supports_text

        mode = self.mode_combo.get_active_id() or MODE_EVERYTHING
        if mode == MODE_TEXT and not supports_text:
            self.mode_combo.set_active_id(MODE_EVERYTHING)
            mode = MODE_EVERYTHING

        text_mode = mode == MODE_TEXT
        self.text_entry.set_visible(text_mode)
        self.text_entry.set_sensitive(text_mode)
        self.detail_combo.set_visible(not text_mode)
        self.detail_label.set_visible(not text_mode)
        self.mode_hint.set_visible(not supports_text)
        self.mode_combo.set_tooltip_text(
            "The installed model is %s, which finds regions but cannot read "
            "words. Install SAM 3 from Setup to search by description."
            % self._model_name
            if not supports_text else
            "Segment everything in the image, or search for something by name"
        )

    def _sync_output_widgets(self):
        is_selection = (self.output_combo.get_active_id() == "selection")
        self.op_combo.set_sensitive(is_selection)

    def _start_segmentation(self):
        if self._busy:
            return
        if not env.is_installed():
            self._prompt_install()
            return

        settings = env.load_settings()
        mode = self.mode_combo.get_active_id() or MODE_EVERYTHING
        text = self.text_entry.get_text().strip()
        if mode == MODE_TEXT and not text:
            self._error("Type what you are looking for, for example \"person\".")
            return

        if self._image_path is None:
            try:
                self._image_path = gimpio.write_png(self.full_pixbuf)
            except Exception as error:  # noqa: BLE001
                log.exception("could not export the image")
                self._error("Could not export the image: %s" % error)
                return

        request = {
            "image_path": self._image_path,
            "backend": settings.get("backend", "sam2"),
            "model": settings.get("model"),
            "device": settings.get("device", "auto"),
            "mode": mode,
            "text": text,
            "points_per_side": int(self.detail_combo.get_active_id() or 32),
            "pred_iou_thresh": settings.get("pred_iou_thresh", 0.75),
            "stability_score_thresh": settings.get("stability_score_thresh", 0.88),
            "min_area_percent": settings.get("min_area_percent", 0.05),
            "max_segments": settings.get("max_segments", 254),
            "inference_max_side": settings.get("inference_max_side", 1536),
            "preview_max_side": settings.get("preview_max_side", 1400),
        }

        self._set_busy(True)
        self._set_status("Starting the model (this can take a while the first time)...")
        self.progress.set_fraction(0.02)

        def work():
            try:
                result = self.client.call(
                    "segment", request, timeout=1800.0,
                    on_progress=lambda p: GLib.idle_add(self._on_progress, p),
                )
                GLib.idle_add(self._segmentation_done, result, None)
            except Exception as error:  # noqa: BLE001 - reported in the dialog
                log.exception("segmentation failed")
                GLib.idle_add(self._segmentation_done, None, str(error))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _on_progress(self, payload):
        message = payload.get("message") or ""
        fraction = payload.get("fraction")
        if fraction is not None:
            self.progress.set_fraction(max(0.0, min(1.0, float(fraction))))
        else:
            self.progress.pulse()
        if message:
            self.progress.set_text(message)
        return False

    def _segmentation_done(self, result, error):
        self._set_busy(False)
        if error:
            self._error(error)
            return False

        self.result = result
        segments = result.get("segments", [])
        label_pixbuf = _pixbuf_from_base64(result.get("label_map"))
        self.canvas.set_segments(segments, label_pixbuf, result.get("image_size"))
        self.segment_list.set_segments(segments)
        self.selected_ids = []
        self._update_apply_sensitivity()

        message = "Found %d region%s." % (len(segments), "" if len(segments) == 1 else "s")
        if result.get("truncated"):
            message += (" %d smaller ones were left out (the list is capped at %d)."
                        % (result["truncated"], self.settings.get("max_segments", 254)))
        if not segments:
            message = ("No regions found. Try a finer Detail setting, or a "
                       "different description.")
        self._set_status(message)
        self.progress.set_fraction(1.0 if segments else 0.0)
        return False

    # ------------------------------------------------------------------
    # Selection sync
    # ------------------------------------------------------------------

    def _on_list_selection(self, ids):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.selected_ids = self.segment_list.expand_with_children(ids)
            self.canvas.set_selection(self.selected_ids)
        finally:
            self._syncing = False
        self._update_apply_sensitivity()

    def _on_canvas_pick(self, segment_id, additive):
        if self._syncing:
            return
        current = set(self.selected_ids)
        if segment_id is None:
            if not additive:
                current = set()
        elif additive:
            if segment_id in current:
                current.discard(segment_id)
                for child in self.segment_list.descendants_of(segment_id):
                    current.discard(child)
            else:
                current.add(segment_id)
        else:
            current = {segment_id}

        self._syncing = True
        try:
            expanded = self.segment_list.expand_with_children(sorted(current))
            self.selected_ids = expanded
            self.canvas.set_selection(expanded)
            self.segment_list.set_selected_ids(expanded)
        finally:
            self._syncing = False
        self._update_apply_sensitivity()

    def _on_list_hover(self, segment_id):
        if self.canvas.hovered != segment_id:
            self.canvas.hovered = segment_id
            self.canvas.area.queue_draw()

    def _on_canvas_hover(self, segment_id):
        pass

    def _update_apply_sensitivity(self):
        count = len(self.selected_ids)
        self.apply_button.set_sensitive(count > 0 and not self._busy)
        if count:
            self.selection_label.set_text(
                "%d region%s selected" % (count, "" if count == 1 else "s"))
        else:
            self.selection_label.set_text(
                "Click a region in the picture, or pick one from the list "
                "(%s+click for several)" % env.primary_modifier_name())

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def _on_response(self, _dialog, response):
        if response == RESPONSE_APPLY:
            self.stop_emission_by_name("response")
            self._apply()
        elif response in (RESPONSE_CLOSE, Gtk.ResponseType.DELETE_EVENT):
            self._remember_settings()
            self._discard_export()
            self.destroy()

    def _discard_export(self):
        """Delete the PNG handed to the backend for this image."""
        if not self._image_path:
            return
        try:
            os.remove(self._image_path)
        except OSError:
            pass
        self._image_path = None

    def _remember_settings(self):
        settings = env.load_settings()
        settings["points_per_side"] = int(self.detail_combo.get_active_id() or 32)
        settings["selection_op"] = self.op_combo.get_active_id() or "replace"
        settings["feather"] = float(self.feather_spin.get_value())
        settings["last_text_prompt"] = self.text_entry.get_text().strip()
        env.save_settings(settings)

    def _apply(self):
        if not self.selected_ids or self.result is None or self._busy:
            return
        self._set_busy(True)
        self._set_status("Building the mask at full resolution...")

        request = {"session": self.result["session"], "ids": list(self.selected_ids)}

        def work():
            try:
                mask = self.client.call("combine", request, timeout=300.0)
                GLib.idle_add(self._apply_mask, mask, None)
            except Exception as error:  # noqa: BLE001
                log.exception("combine failed")
                GLib.idle_add(self._apply_mask, None, str(error))

        threading.Thread(target=work, daemon=True).start()

    def _apply_mask(self, payload, error):
        self._set_busy(False)
        if error:
            self._error(error)
            return False
        if payload.get("empty"):
            self._error("The selected regions produced an empty mask.")
            return False

        width = payload["width"]
        height = payload["height"]
        try:
            mask = rle.decode(payload["rle"], width, height)
        except Exception as inner:  # noqa: BLE001
            log.exception("mask decode failed")
            self._error("Could not decode the mask: %s" % inner)
            return False

        output = self.output_combo.get_active_id() or "selection"
        feather = float(self.feather_spin.get_value())
        operation = self.op_combo.get_active_id() or "replace"
        name = self._suggest_name()

        self.image.undo_group_start()
        try:
            if output == "selection":
                gimpio.apply_selection(self.image, mask, width, height, operation, feather)
                done = "Selection updated."
            elif output == "channel":
                gimpio.save_as_channel(self.image, mask, width, height, name)
                done = "Saved as channel \"%s\"." % name
            else:
                layer = self._target_layer()
                if layer is None:
                    raise RuntimeError(
                        "Select a layer in the Layers panel first."
                    )
                if output == "layer_mask":
                    gimpio.add_layer_mask(self.image, layer, mask, width, height)
                    done = "Layer mask added to \"%s\"." % layer.get_name()
                else:
                    new_layer = gimpio.copy_to_new_layer(
                        self.image, layer, mask, width, height, name)
                    done = "Created layer \"%s\"." % new_layer.get_name()
        except Exception as inner:  # noqa: BLE001
            self.image.undo_group_end()
            log.exception("apply failed")
            self._error(str(inner))
            return False
        self.image.undo_group_end()
        Gimp.displays_flush()

        self._remember_settings()
        self._discard_export()
        self._set_status(done)
        self.destroy()
        return False

    def _target_layer(self):
        layers = self.image.get_selected_layers()
        if layers:
            return layers[0]
        for drawable in self.drawables:
            if drawable.is_layer():
                return drawable
        layers = self.image.get_layers()
        return layers[0] if layers else None

    def _suggest_name(self):
        if not self.result:
            return "Segment"
        by_id = {segment["id"]: segment for segment in self.result.get("segments", [])}
        names = [by_id[i]["name"] for i in self.selected_ids if i in by_id]
        if not names:
            return "Segment"
        if len(names) == 1:
            return names[0]
        return "%s +%d" % (names[0], len(names) - 1)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _set_busy(self, busy):
        self._busy = busy
        self.run_button.set_sensitive(not busy)
        self.mode_combo.set_sensitive(not busy)
        self.detail_combo.set_sensitive(not busy)
        self.text_entry.set_sensitive(not busy and self.mode_combo.get_active_id() == MODE_TEXT)
        self._update_apply_sensitivity()
        window = self.get_window()
        if window is not None:
            window.set_cursor(
                Gdk.Cursor.new_from_name(window.get_display(), "wait") if busy else None)

    def _set_status(self, message):
        self.progress.set_text(message)

    def _error(self, message):
        self._set_status("Error")
        self.progress.set_fraction(0.0)
        dialog = Gtk.MessageDialog(
            transient_for=self, modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.CLOSE,
            text="Segment Anything",
        )
        dialog.format_secondary_text(str(message))
        dialog.run()
        dialog.destroy()


def _pixbuf_from_base64(data):
    if not data:
        return None
    raw = base64.b64decode(data)
    loader = GdkPixbuf.PixbufLoader.new_with_type("png")
    loader.write(raw)
    loader.close()
    return loader.get_pixbuf()


def run(image, drawables, settings):
    """Show the dialog and block until it closes."""
    dialog = SegmentAnythingDialog(image, drawables, settings)
    loop = GLib.MainLoop()
    dialog.connect("destroy", lambda *_a: loop.quit())
    loop.run()
    return True
