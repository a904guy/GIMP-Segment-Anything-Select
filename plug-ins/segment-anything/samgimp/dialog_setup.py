"""Setup window: choose a model and install what it needs.

Installation runs on a worker thread so the dialog stays responsive during
multi-gigabyte downloads, and the output of each step is shown in a log
pane alongside the progress bar.
"""

import threading

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GimpUi", "3.0")
from gi.repository import GimpUi, GLib, Gtk  # noqa: E402

from . import client, env, log, setup_env  # noqa: E402

RESPONSE_INSTALL = 1
RESPONSE_CLOSE = 2
RESPONSE_REMOVE = 3


class SetupDialog(GimpUi.Dialog):
    def __init__(self, parent=None, settings=None):
        GimpUi.Dialog.__init__(
            self, title="Segment Anything - Setup", role="sam-gimp-setup",
            use_header_bar=True,
        )
        self.settings = dict(settings or env.load_settings())
        self.installed_ok = env.is_installed()
        self._worker = None
        self._cancel = threading.Event()

        self.set_default_size(720, 620)
        if parent is not None:
            self.set_transient_for(parent)

        self.close_button = self.add_button("_Close", RESPONSE_CLOSE)
        self.remove_button = self.add_button("_Remove Installation", RESPONSE_REMOVE)
        self.install_button = self.add_button("_Install", RESPONSE_INSTALL)
        self.set_default_response(RESPONSE_INSTALL)
        self.connect("response", self._on_response)

        content = self.get_content_area()
        content.set_spacing(10)
        content.set_border_width(12)
        content.pack_start(self._build_form(), False, False, 0)
        content.pack_start(self._build_progress(), False, False, 0)
        content.pack_start(self._build_log(), True, True, 0)

        self._update_state()
        self.show_all()
        # After show_all, so the token field's visibility is not overridden.
        self._sync_model_choices()
        self._refresh_status()

    # -- widgets -----------------------------------------------------------

    def _build_form(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=8)
        row = 0

        self.status_label = Gtk.Label()
        self.status_label.set_xalign(0.0)
        self.status_label.set_line_wrap(True)
        grid.attach(self.status_label, 0, row, 2, 1)
        row += 1

        grid.attach(_label("Model family:"), 0, row, 1, 1)
        self.backend_combo = Gtk.ComboBoxText()
        self.backend_combo.append("sam2", "SAM 2.1 - open weights, no account needed")
        self.backend_combo.append("sam3", "SAM 3 - newer, adds text prompts (needs a Hugging Face account)")
        self.backend_combo.set_active_id(self.settings.get("backend", "sam2"))
        self.backend_combo.connect("changed", self._on_backend_changed)
        grid.attach(self.backend_combo, 1, row, 1, 1)
        row += 1

        grid.attach(_label("Model:"), 0, row, 1, 1)
        self.model_combo = Gtk.ComboBoxText()
        grid.attach(self.model_combo, 1, row, 1, 1)
        row += 1

        grid.attach(_label("Run on:"), 0, row, 1, 1)
        self.device_combo = Gtk.ComboBoxText()
        self.device_combo.append("auto", "Automatic (recommended)")
        self.device_combo.append("cuda", "NVIDIA GPU (CUDA)")
        self.device_combo.append("mps", "Apple Silicon GPU (Metal)")
        self.device_combo.append("cpu", "CPU only")
        self.device_combo.set_active_id(self.settings.get("device", "auto"))
        grid.attach(self.device_combo, 1, row, 1, 1)
        row += 1

        self.token_label = _label("Hugging Face token:")
        grid.attach(self.token_label, 0, row, 1, 1)
        self.token_entry = Gtk.Entry()
        self.token_label.set_no_show_all(True)
        self.token_entry.set_no_show_all(True)
        self.token_entry.set_visibility(False)
        self.token_entry.set_placeholder_text("hf_...")
        self.token_entry.set_text(self.settings.get("hf_token", ""))
        self.token_entry.set_tooltip_text(
            "SAM 3's weights are gated. Accept the licence at "
            "huggingface.co/facebook/sam3, then paste a token from "
            "huggingface.co/settings/tokens"
        )
        grid.attach(self.token_entry, 1, row, 1, 1)
        row += 1

        self.detected_label = Gtk.Label()
        self.detected_label.set_xalign(0.0)
        self.detected_label.get_style_context().add_class("dim-label")
        self.detected_label.set_line_wrap(True)
        grid.attach(self.detected_label, 0, row, 2, 1)
        return grid

    def _build_progress(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(True)
        self.progress.set_text("Ready")
        box.pack_start(self.progress, False, False, 0)
        return box

    def _build_log(self):
        frame = Gtk.Frame(label="Details")
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.log_view = Gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_cursor_visible(False)
        self.log_view.set_monospace(True)
        self.log_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.log_buffer = self.log_view.get_buffer()
        scroller.add(self.log_view)
        frame.add(scroller)
        return frame

    # -- state -------------------------------------------------------------

    def _sync_model_choices(self):
        backend = self.backend_combo.get_active_id() or "sam2"
        self.model_combo.remove_all()
        for model_id, label, size in env.models_for(backend):
            self.model_combo.append(model_id, "%s  (%s)" % (label, size))
        wanted = self.settings.get("model")
        if not self.model_combo.set_active_id(wanted):
            self.model_combo.set_active_id(env.default_model_for(backend))

        needs_token = env.backend_needs_token(backend)
        self.token_label.set_visible(needs_token)
        self.token_entry.set_visible(needs_token)

    def _on_backend_changed(self, _combo):
        self._sync_model_choices()

    def _refresh_status(self):
        compute = setup_env.detect_compute()
        self.detected_label.set_text(
            "Detected hardware: %s\nInstall location: %s"
            % (setup_env.describe_compute(compute), env.display_path(env.data_dir()))
        )
        summary = setup_env.installed_summary()
        if summary:
            self.status_label.set_markup(
                "<b>Installed:</b> %s\nChoose different options below and press "
                "Install to change it." % _escape(summary)
            )
        else:
            self.status_label.set_markup(
                "<b>Not installed yet.</b>\nThis downloads a private copy of "
                "Python, PyTorch and the model weights into a single folder. "
                "Nothing outside that folder is touched, and removing it "
                "undoes everything."
            )

    def _update_state(self, busy=False):
        self.install_button.set_sensitive(not busy)
        self.remove_button.set_sensitive(not busy and env.is_installed())
        self.backend_combo.set_sensitive(not busy)
        self.model_combo.set_sensitive(not busy)
        self.device_combo.set_sensitive(not busy)
        self.token_entry.set_sensitive(not busy)
        self.close_button.set_label("_Cancel" if busy else "_Close")

    # -- logging -----------------------------------------------------------

    def _append(self, text):
        end = self.log_buffer.get_end_iter()
        self.log_buffer.insert(end, text + "\n")
        mark = self.log_buffer.create_mark(None, self.log_buffer.get_end_iter(), False)
        self.log_view.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)
        self.log_buffer.delete_mark(mark)
        return False

    def report(self, text):
        GLib.idle_add(self._append, text)

    def set_progress(self, fraction, label):
        def update():
            self.progress.set_fraction(max(0.0, min(1.0, fraction)))
            self.progress.set_text(label)
            return False
        GLib.idle_add(update)

    # -- actions -----------------------------------------------------------

    def _collect(self):
        settings = dict(self.settings)
        settings["backend"] = self.backend_combo.get_active_id() or "sam2"
        settings["model"] = self.model_combo.get_active_id() or env.default_model_for(settings["backend"])
        settings["device"] = self.device_combo.get_active_id() or "auto"
        settings["hf_token"] = self.token_entry.get_text().strip()
        settings["compute"] = "auto"
        return settings

    def _on_response(self, _dialog, response):
        if response == RESPONSE_INSTALL:
            self._start_install()
        elif response == RESPONSE_REMOVE:
            self._remove()
        elif response in (RESPONSE_CLOSE, Gtk.ResponseType.DELETE_EVENT):
            if self._worker and self._worker.is_alive():
                self._cancel.set()
                self.report("Cancelling...")
                self.stop_emission_by_name("response")
                return
            self.destroy()

    def _start_install(self):
        if self._worker and self._worker.is_alive():
            return
        settings = self._collect()
        if env.backend_needs_token(settings["backend"]) and not settings["hf_token"]:
            self._append(
                "SAM 3 needs a Hugging Face access token.\n"
                "  1. Sign in and accept the licence at https://huggingface.co/facebook/sam3\n"
                "  2. Create a token at https://huggingface.co/settings/tokens\n"
                "  3. Paste it above and press Install again.\n"
            )
            return

        self.settings = settings
        self._cancel.clear()
        self._update_state(busy=True)
        self.log_buffer.set_text("")
        self.set_progress(0.0, "Starting")

        # A running backend would hold the old environment open on Windows.
        try:
            client.BackendClient().shutdown()
        except Exception:
            pass

        def work():
            try:
                result = setup_env.install(
                    settings, self.report,
                    should_cancel=self._cancel.is_set,
                    progress=self.set_progress,
                )
                GLib.idle_add(self._finished, result, None)
            except setup_env.Cancelled:
                GLib.idle_add(self._finished, None, "Cancelled.")
            except Exception as error:  # noqa: BLE001 - surfaced in the log
                log.exception("install failed")
                GLib.idle_add(self._finished, None, str(error))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _finished(self, settings, error):
        self._update_state(busy=False)
        if error:
            self.set_progress(0.0, "Failed")
            self._append("\n*** %s ***" % error)
        else:
            self.settings = settings
            self.installed_ok = True
            self.set_progress(1.0, "Installed")
            self._append("\nYou can close this window and run "
                         "Select > Segment Anything.")
        self._refresh_status()
        return False

    def _remove(self):
        confirm = Gtk.MessageDialog(
            transient_for=self, modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text="Remove the Segment Anything runtime?",
        )
        confirm.format_secondary_text(
            "This deletes the private Python environment and the downloaded "
            "model weights from:\n%s\n\nThe plug-in itself stays installed."
            % env.display_path(env.data_dir())
        )
        answer = confirm.run()
        confirm.destroy()
        if answer != Gtk.ResponseType.OK:
            return
        try:
            client.BackendClient().shutdown()
        except Exception:
            pass
        self.log_buffer.set_text("")
        setup_env.uninstall(self.report)
        self.installed_ok = False
        self._update_state()
        self._refresh_status()


def _label(text):
    widget = Gtk.Label(label=text)
    widget.set_xalign(0.0)
    return widget


def _escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def run_setup(parent=None, settings=None):
    """Show the setup dialog modally.  Returns True if a runtime is present."""
    dialog = SetupDialog(parent=parent, settings=settings)
    dialog.set_modal(True)
    loop = GLib.MainLoop()
    dialog.connect("destroy", lambda *_a: loop.quit())
    loop.run()
    return env.is_installed()
