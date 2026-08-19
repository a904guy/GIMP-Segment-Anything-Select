"""Tree view of the detected regions.

Regions contained by another region are shown as its children, which keeps
an object and its parts together.  The view is multi-select, so the
platform's usual selection modifiers apply.
"""

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, Gtk  # noqa: E402

from . import env, palette  # noqa: E402

COL_ID = 0
COL_SWATCH = 1
COL_NAME = 2
COL_DETAIL = 3


class SegmentList(Gtk.Box):
    def __init__(self, on_selection_changed=None, on_hover=None):
        Gtk.Box.__init__(self, orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.on_selection_changed = on_selection_changed
        self.on_hover = on_hover
        self._updating = False
        self._rows = {}          # segment id -> Gtk.TreeIter
        self._children = {}      # segment id -> [child ids]
        self.include_children = False

        self.store = Gtk.TreeStore(int, GdkPixbuf.Pixbuf, str, str)
        self.view = Gtk.TreeView(model=self.store)
        self.view.set_headers_visible(True)
        self.view.set_enable_search(True)
        self.view.set_search_column(COL_NAME)
        self.view.set_tooltip_column(COL_NAME)
        self.view.set_has_tooltip(True)

        selection = self.view.get_selection()
        selection.set_mode(Gtk.SelectionMode.MULTIPLE)
        selection.connect("changed", self._on_selection_changed)

        # Several GTK themes override a cell background colour, so the swatch
        # is drawn as a pixbuf instead.
        swatch = Gtk.CellRendererPixbuf()
        column = Gtk.TreeViewColumn("", swatch, pixbuf=COL_SWATCH)
        column.set_sizing(Gtk.TreeViewColumnSizing.FIXED)
        column.set_fixed_width(22)
        self.view.append_column(column)

        name_cell = Gtk.CellRendererText()
        name_cell.set_property("ellipsize", 3)  # PANGO_ELLIPSIZE_END
        name_column = Gtk.TreeViewColumn("Region", name_cell, text=COL_NAME)
        name_column.set_expand(True)
        name_column.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        self.view.append_column(name_column)
        # Otherwise the expander is drawn in the swatch column and hides the
        # colour chip.
        self.view.set_expander_column(name_column)

        detail_cell = Gtk.CellRendererText()
        detail_cell.set_property("xalign", 1.0)
        detail_column = Gtk.TreeViewColumn("Size", detail_cell, text=COL_DETAIL)
        self.view.append_column(detail_column)

        self.view.add_events(Gdk.EventMask.POINTER_MOTION_MASK
                             | Gdk.EventMask.LEAVE_NOTIFY_MASK)
        self.view.connect("motion-notify-event", self._on_motion)
        self.view.connect("leave-notify-event", self._on_leave)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_shadow_type(Gtk.ShadowType.IN)
        scroller.add(self.view)
        scroller.set_size_request(240, 180)
        self.pack_start(scroller, True, True, 0)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        for label, callback, tip in (
            ("All", self.select_all, "Select every region"),
            ("None", self.select_none, "Clear the selection"),
            ("Invert", self.invert_selection, "Select everything not currently selected"),
        ):
            button = Gtk.Button(label=label)
            button.set_tooltip_text(tip)
            button.connect("clicked", lambda _b, cb=callback: cb())
            controls.pack_start(button, True, True, 0)
        self.pack_start(controls, False, False, 0)

        self.nested_toggle = Gtk.CheckButton(label="Include nested pieces")
        self.nested_toggle.set_tooltip_text(
            "When a region is selected, also select the regions found inside it"
        )
        self.nested_toggle.connect("toggled", self._on_nested_toggled)
        self.pack_start(self.nested_toggle, False, False, 0)

        self.summary = Gtk.Label(label="")
        self.summary.set_xalign(0.0)
        self.summary.set_line_wrap(True)
        self.summary.get_style_context().add_class("dim-label")
        self.pack_start(self.summary, False, False, 0)

    # -- population --------------------------------------------------------

    def set_segments(self, segments):
        self._updating = True
        self.store.clear()
        self._rows = {}
        self._children = {}

        by_id = {segment["id"]: segment for segment in segments}
        for segment in segments:
            parent = segment.get("parent")
            if parent is not None and parent in by_id:
                self._children.setdefault(parent, []).append(segment["id"])

        def add(segment, parent_iter):
            row = self.store.append(parent_iter, [
                segment["id"],
                _swatch(segment["id"]),
                segment["name"],
                _format_size(segment),
            ])
            self._rows[segment["id"]] = row
            for child_id in self._children.get(segment["id"], []):
                add(by_id[child_id], row)

        for segment in segments:
            if segment.get("parent") is None or segment["parent"] not in by_id:
                add(segment, None)

        self.view.expand_all()
        self._updating = False
        self._update_summary(segments)

    def _update_summary(self, segments):
        nested = sum(len(kids) for kids in self._children.values())
        if not segments:
            self.summary.set_text("No regions found.")
        else:
            self.summary.set_text(
                "%d regions, %d of them nested inside another."
                % (len(segments), nested)
            )

    # -- selection ---------------------------------------------------------

    def get_selected_ids(self):
        selection = self.view.get_selection()
        model, paths = selection.get_selected_rows()
        return [model[path][COL_ID] for path in paths]

    def set_selected_ids(self, ids):
        """Update the highlighted rows without re-emitting a change."""
        wanted = set(ids)
        selection = self.view.get_selection()
        self._updating = True
        try:
            selection.unselect_all()
            for segment_id in wanted:
                row = self._rows.get(segment_id)
                if row is not None:
                    selection.select_iter(row)
            first = next(iter(sorted(wanted)), None)
            if first is not None and first in self._rows:
                path = self.store.get_path(self._rows[first])
                self.view.scroll_to_cell(path, None, False, 0.0, 0.0)
        finally:
            self._updating = False

    def descendants_of(self, segment_id):
        found = []
        stack = list(self._children.get(segment_id, []))
        while stack:
            current = stack.pop()
            found.append(current)
            stack.extend(self._children.get(current, []))
        return found

    def expand_with_children(self, ids):
        """Add nested pieces when the user asked for them."""
        if not self.include_children:
            return list(ids)
        expanded = set(ids)
        for segment_id in list(ids):
            expanded.update(self.descendants_of(segment_id))
        return sorted(expanded)

    def select_all(self):
        self.view.get_selection().select_all()

    def select_none(self):
        self.view.get_selection().unselect_all()

    def invert_selection(self):
        current = set(self.get_selected_ids())
        everything = set(self._rows)
        self.set_selected_ids(everything - current)
        self._emit()

    # -- events ------------------------------------------------------------

    def _on_nested_toggled(self, button):
        self.include_children = button.get_active()
        self._emit()

    def _on_selection_changed(self, _selection):
        if self._updating:
            return
        self._emit()

    def _emit(self):
        if self.on_selection_changed:
            self.on_selection_changed(self.get_selected_ids())

    def _on_motion(self, _widget, event):
        if not self.on_hover:
            return False
        hit = self.view.get_path_at_pos(int(event.x), int(event.y))
        segment_id = None
        if hit and hit[0] is not None:
            segment_id = self.store[hit[0]][COL_ID]
        self.on_hover(segment_id)
        return False

    def _on_leave(self, _widget, _event):
        if self.on_hover:
            self.on_hover(None)
        return False


_SWATCH_CACHE = {}


def _swatch(segment_id, size=12):
    """A small solid-colour chip identifying the region on the canvas."""
    pixbuf = _SWATCH_CACHE.get(segment_id)
    if pixbuf is None:
        red, green, blue = palette.color_for(segment_id)
        pixbuf = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, size, size)
        pixbuf.fill((int(red * 255) << 24) | (int(green * 255) << 16)
                    | (int(blue * 255) << 8) | 0xFF)
        _SWATCH_CACHE[segment_id] = pixbuf
    return pixbuf


def _format_size(segment):
    fraction = segment.get("area_frac", 0.0) * 100.0
    if fraction >= 10:
        return "%.0f%%" % fraction
    if fraction >= 1:
        return "%.1f%%" % fraction
    return "%.2f%%" % fraction
