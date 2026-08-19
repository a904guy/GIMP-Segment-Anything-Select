"""Scrollable image view with per-region overlays.

A plain click selects one region, and the platform's selection modifier
(Control, or Command on macOS) extends the selection.  The middle button or
Alt-drag pans, and the primary modifier plus scroll zooms.

Clicks are resolved against the label map supplied by the backend, which
holds one byte per preview pixel giving the smallest region covering it.
That makes hit-testing a single array lookup regardless of region count.
"""

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

import cairo  # noqa: E402

from . import palette, rle  # noqa: E402

NO_SEGMENT = 255

ALL_ALPHA = 0.25        # every region, so the user can see what is available
HOVER_ALPHA = 0.45
SELECTED_ALPHA = 0.55
MIN_ZOOM = 0.05
MAX_ZOOM = 16.0


class SegmentCanvas(Gtk.ScrolledWindow):
    """Scrollable, zoomable image view with per-region overlays."""

    def __init__(self, on_pick=None, on_hover=None):
        Gtk.ScrolledWindow.__init__(self)
        self.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.set_shadow_type(Gtk.ShadowType.IN)

        self.on_pick = on_pick
        self.on_hover = on_hover

        self.pixbuf = None
        self.preview_size = (0, 0)
        self.bbox_scale = 1.0   # full-resolution bbox coords -> preview coords
        self.segments = []
        self.selected = set()
        self.hovered = None
        self.show_all_regions = True
        self.zoom = 1.0
        self._fit = True

        self._label_pixels = None
        self._label_rowstride = 0
        self._label_channels = 3
        self._mask_cache = {}
        self._all_overlay = None
        self._pan_origin = None

        self.area = Gtk.DrawingArea()
        self.area.add_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.LEAVE_NOTIFY_MASK
            | Gdk.EventMask.SCROLL_MASK
            | Gdk.EventMask.SMOOTH_SCROLL_MASK
        )
        self.area.connect("draw", self._on_draw)
        self.area.connect("button-press-event", self._on_button_press)
        self.area.connect("button-release-event", self._on_button_release)
        self.area.connect("motion-notify-event", self._on_motion)
        self.area.connect("leave-notify-event", self._on_leave)
        self.area.connect("scroll-event", self._on_scroll)
        self.connect("size-allocate", self._on_size_allocate)

        self.add(self.area)

    # -- content -----------------------------------------------------------

    def set_image(self, pixbuf):
        self.pixbuf = pixbuf
        self.preview_size = (pixbuf.get_width(), pixbuf.get_height())
        self.clear_segments()
        self._fit = True
        self._apply_zoom()

    def clear_segments(self):
        self.segments = []
        self.selected = set()
        self.hovered = None
        self._mask_cache = {}
        self._all_overlay = None
        self._label_pixels = None
        self.area.queue_draw()

    def set_segments(self, segments, label_map_pixbuf, image_size=None):
        """``segments`` are the dicts the backend returned."""
        self.segments = segments
        if image_size and image_size[0]:
            self.bbox_scale = self.preview_size[0] / float(image_size[0])
        else:
            self.bbox_scale = 1.0
        self.selected = set()
        self.hovered = None
        self._mask_cache = {}
        self._all_overlay = None
        if label_map_pixbuf is not None:
            self._label_pixels = label_map_pixbuf.get_pixels()
            self._label_rowstride = label_map_pixbuf.get_rowstride()
            self._label_channels = label_map_pixbuf.get_n_channels()
        else:
            self._label_pixels = None
        self.area.queue_draw()

    def set_selection(self, ids):
        new = set(ids)
        if new != self.selected:
            self.selected = new
            self.area.queue_draw()

    def set_show_all_regions(self, enabled):
        self.show_all_regions = bool(enabled)
        self.area.queue_draw()

    # -- zoom --------------------------------------------------------------

    def fit_to_window(self):
        self._fit = True
        self._apply_zoom()

    def set_zoom(self, zoom, keep_centre=True):
        self._fit = False
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        self._apply_zoom(keep_centre=keep_centre)

    def zoom_by(self, factor):
        self.set_zoom(self.zoom * factor)

    def _fit_zoom(self):
        width, height = self.preview_size
        if not width or not height:
            return 1.0
        allocation = self.get_allocation()
        available_w = max(32, allocation.width - 4)
        available_h = max(32, allocation.height - 4)
        return min(available_w / float(width), available_h / float(height))

    def _apply_zoom(self, keep_centre=False):
        width, height = self.preview_size
        if not width or not height:
            return
        if self._fit:
            self.zoom = self._fit_zoom()
        self.area.set_size_request(
            max(1, int(round(width * self.zoom))),
            max(1, int(round(height * self.zoom))),
        )
        self.area.queue_draw()

    def _origin(self):
        """Top-left of the image inside the drawing area, centred when it fits."""
        allocation = self.area.get_allocation()
        width = self.preview_size[0] * self.zoom
        height = self.preview_size[1] * self.zoom
        return (max(0.0, (allocation.width - width) / 2.0),
                max(0.0, (allocation.height - height) / 2.0))

    def _on_size_allocate(self, _widget, _allocation):
        if self._fit:
            GLib.idle_add(self._apply_zoom, priority=GLib.PRIORITY_LOW)

    # -- overlay construction ---------------------------------------------

    def _mask_surface(self, segment_id):
        surface = self._mask_cache.get(segment_id)
        if surface is not None:
            return surface
        segment = self._segment_by_id(segment_id)
        if segment is None:
            return None
        width, height = self.preview_size
        surface = rle.a8_surface(segment["rle_preview"], width, height)
        self._mask_cache[segment_id] = surface
        return surface

    def _segment_by_id(self, segment_id):
        for segment in self.segments:
            if segment["id"] == segment_id:
                return segment
        return None

    def _build_all_overlay(self):
        """One pre-rendered surface tinting every region, drawn once."""
        width, height = self.preview_size
        if not width or not height:
            return None
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        context = cairo.Context(surface)
        # Largest first, so nested regions are drawn over their container.
        ordered = sorted(self.segments, key=lambda s: -s.get("area", 0))
        for segment in ordered:
            mask = self._mask_surface(segment["id"])
            if mask is None:
                continue
            red, green, blue = palette.color_for(segment["id"])
            context.set_source_rgb(red, green, blue)
            context.mask_surface(mask, 0, 0)
        surface.flush()
        return surface

    # -- drawing -----------------------------------------------------------

    def _on_draw(self, _widget, context):
        if self.pixbuf is None:
            return False
        context.set_source_rgb(0.12, 0.12, 0.13)
        context.paint()

        origin_x, origin_y = self._origin()
        context.save()
        context.translate(origin_x, origin_y)
        context.scale(self.zoom, self.zoom)

        Gdk.cairo_set_source_pixbuf(context, self.pixbuf, 0, 0)
        pattern = context.get_source()
        pattern.set_filter(cairo.FILTER_GOOD if self.zoom < 1.0 else cairo.FILTER_NEAREST)
        context.paint()

        if self.segments:
            if self.show_all_regions:
                if self._all_overlay is None:
                    self._all_overlay = self._build_all_overlay()
                if self._all_overlay is not None:
                    context.set_source_surface(self._all_overlay, 0, 0)
                    context.paint_with_alpha(ALL_ALPHA)

            if self.hovered is not None and self.hovered not in self.selected:
                self._paint_segment(context, self.hovered, HOVER_ALPHA)

            for segment_id in self.selected:
                self._paint_segment(context, segment_id, SELECTED_ALPHA)

        context.restore()

        # Outline every selected region so the boundary is unmistakable even
        # where the fill sits over a busy photo.
        if self.selected:
            self._paint_outlines(context, origin_x, origin_y)
        return False

    def _paint_segment(self, context, segment_id, alpha):
        mask = self._mask_surface(segment_id)
        if mask is None:
            return
        red, green, blue = palette.color_for(segment_id)
        context.set_source_rgba(red, green, blue, alpha)
        context.mask_surface(mask, 0, 0)

    def _paint_outlines(self, context, origin_x=0.0, origin_y=0.0):
        """Draw a bounding box for each selected region, in device space."""
        width, height = self.preview_size
        if not width:
            return
        context.save()
        context.translate(origin_x, origin_y)
        context.set_line_width(1.0)
        context.set_dash([4.0, 3.0])
        for segment_id in self.selected:
            segment = self._segment_by_id(segment_id)
            if segment is None:
                continue
            preview_scale = self.bbox_scale
            x, y, w, h = segment["bbox"]
            rect = (x * preview_scale * self.zoom, y * preview_scale * self.zoom,
                    w * preview_scale * self.zoom, h * preview_scale * self.zoom)
            context.set_source_rgba(0, 0, 0, 0.8)
            context.rectangle(*rect)
            context.stroke()
            red, green, blue = palette.color_for(segment_id)
            context.set_dash([4.0, 3.0], 4.0)
            context.set_source_rgba(red, green, blue, 1.0)
            context.rectangle(*rect)
            context.stroke()
        context.restore()

    # -- platform modifiers ------------------------------------------------

    def _modifier(self, intent):
        """The modifier GDK uses for ``intent`` on this platform.

        Asking GDK rather than hard-coding Control matters on macOS, where
        the selection modifier is Command and Control+click is the system
        secondary-click gesture.
        """
        try:
            return self.area.get_modifier_mask(intent)
        except (AttributeError, TypeError):
            return Gdk.ModifierType.CONTROL_MASK

    def _is_additive(self, state):
        modify = self._modifier(Gdk.ModifierIntent.MODIFY_SELECTION)
        extend = self._modifier(Gdk.ModifierIntent.EXTEND_SELECTION)
        return bool(state & (modify | extend))

    # -- hit testing -------------------------------------------------------

    def _segment_at(self, widget_x, widget_y):
        if self._label_pixels is None or self.zoom <= 0:
            return None
        origin_x, origin_y = self._origin()
        x = int((widget_x - origin_x) / self.zoom)
        y = int((widget_y - origin_y) / self.zoom)
        width, height = self.preview_size
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        offset = y * self._label_rowstride + x * self._label_channels
        try:
            value = self._label_pixels[offset]
        except IndexError:
            return None
        if isinstance(value, bytes):
            value = value[0]
        if value == NO_SEGMENT:
            return None
        return int(value)

    # -- events ------------------------------------------------------------

    def _on_button_press(self, _widget, event):
        if event.button == 2 or (event.button == 1 and event.state & Gdk.ModifierType.MOD1_MASK):
            self._pan_origin = (event.x_root, event.y_root,
                                self.get_hadjustment().get_value(),
                                self.get_vadjustment().get_value())
            return True
        if event.button != 1:
            return False
        if event.type != Gdk.EventType.BUTTON_PRESS:
            return False

        additive = self._is_additive(event.state)
        segment_id = self._segment_at(event.x, event.y)
        if self.on_pick:
            self.on_pick(segment_id, additive)
        return True

    def _on_button_release(self, _widget, event):
        if event.button == 2 or self._pan_origin:
            self._pan_origin = None
            return True
        return False

    def _on_motion(self, _widget, event):
        if self._pan_origin:
            start_x, start_y, h_value, v_value = self._pan_origin
            self.get_hadjustment().set_value(h_value - (event.x_root - start_x))
            self.get_vadjustment().set_value(v_value - (event.y_root - start_y))
            return True
        segment_id = self._segment_at(event.x, event.y)
        if segment_id != self.hovered:
            self.hovered = segment_id
            self.area.queue_draw()
            if self.on_hover:
                self.on_hover(segment_id)
        return False

    def _on_leave(self, _widget, _event):
        if self.hovered is not None:
            self.hovered = None
            self.area.queue_draw()
            if self.on_hover:
                self.on_hover(None)
        return False

    def _on_scroll(self, _widget, event):
        # Control on Linux and Windows, Command on macOS.
        zoom_modifier = self._modifier(Gdk.ModifierIntent.PRIMARY_ACCELERATOR)
        if not (event.state & zoom_modifier):
            return False
        direction = event.direction
        if direction == Gdk.ScrollDirection.SMOOTH:
            _ok, _dx, dy = event.get_scroll_deltas()
            factor = 1.0 - dy * 0.15
        elif direction == Gdk.ScrollDirection.UP:
            factor = 1.15
        elif direction == Gdk.ScrollDirection.DOWN:
            factor = 1.0 / 1.15
        else:
            return False
        self.set_zoom(self.zoom * factor)
        return True
