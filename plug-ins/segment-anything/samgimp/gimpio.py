"""Pixel transfer between GIMP and the plug-in.

Outbound: flatten the visible image into a PNG for the backend, plus a
GdkPixbuf used for the preview.

Inbound: apply a mask (one byte per pixel, 0 or 255) as a selection, a
layer mask, a new layer or a stored channel.
"""

import os
import time

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, Gegl, Gimp, GLib  # noqa: E402

from . import env  # noqa: E402

SELECTION_OPS = {
    "replace": Gimp.ChannelOps.REPLACE,
    "add": Gimp.ChannelOps.ADD,
    "subtract": Gimp.ChannelOps.SUBTRACT,
    "intersect": Gimp.ChannelOps.INTERSECT,
}


# --------------------------------------------------------------------------
# GIMP -> plug-in
# --------------------------------------------------------------------------


def composite_pixbuf(image):
    """The flattened, visible image as a GdkPixbuf."""
    width, height = image.get_width(), image.get_height()
    duplicate = image.duplicate()
    try:
        duplicate.flatten()
        layers = duplicate.get_layers()
        if not layers:
            raise RuntimeError("The image has no visible layers to segment.")
        buffer = layers[0].get_buffer()
        rect = Gegl.Rectangle.new(0, 0, width, height)
        data = buffer.get(rect, 1.0, "R'G'B'A u8", Gegl.AbyssPolicy.CLAMP)
    finally:
        duplicate.delete()

    return GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(data), GdkPixbuf.Colorspace.RGB, True, 8,
        width, height, width * 4,
    )


def write_png(pixbuf, prefix="sam-input"):
    """Save a pixbuf to the plug-in's temp folder and return the path."""
    env.ensure_dirs()
    path = os.path.join(env.temp_dir(), "%s-%d.png" % (prefix, int(time.time() * 1000)))
    pixbuf.savev(path, "png", [], [])
    return path


def scaled_pixbuf(pixbuf, max_side):
    """Downscale for on-screen preview, preserving aspect ratio."""
    width, height = pixbuf.get_width(), pixbuf.get_height()
    longest = max(width, height)
    if not max_side or longest <= max_side:
        return pixbuf
    scale = max_side / float(longest)
    return pixbuf.scale_simple(
        max(1, int(round(width * scale))), max(1, int(round(height * scale))),
        GdkPixbuf.InterpType.BILINEAR,
    )


def cleanup_temp(max_age_seconds=86400):
    """Remove stale exported PNGs so the cache cannot grow without bound."""
    directory = env.temp_dir()
    if not os.path.isdir(directory):
        return
    cutoff = time.time() - max_age_seconds
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# plug-in -> GIMP
# --------------------------------------------------------------------------


def _fill_drawable_from_mask(drawable, mask, mask_width, mask_height,
                             offset_x=0, offset_y=0):
    """Write mask bytes into a single-channel drawable.

    ``offset_x/y`` locate the drawable inside the mask, so layers that are
    smaller than the canvas get the right slice.
    """
    width, height = drawable.get_width(), drawable.get_height()
    if (width, height) == (mask_width, mask_height) and not (offset_x or offset_y):
        payload = bytes(mask)
    else:
        rows = []
        blank = b"\x00" * width
        for row in range(height):
            source_row = row + offset_y
            if source_row < 0 or source_row >= mask_height:
                rows.append(blank)
                continue
            start = source_row * mask_width + offset_x
            chunk = bytes(mask[start:start + width])
            if len(chunk) < width:
                chunk += b"\x00" * (width - len(chunk))
            rows.append(chunk)
        payload = b"".join(rows)

    buffer = drawable.get_buffer()
    buffer.set(Gegl.Rectangle.new(0, 0, width, height), "Y' u8", payload)
    buffer.flush()
    drawable.update(0, 0, width, height)


def mask_to_channel(image, mask, width, height, name="Segment Anything"):
    """Create (and insert) a channel holding the mask."""
    channel = Gimp.Channel.new(image, name, width, height, 100.0, Gegl.Color.new("black"))
    image.insert_channel(channel, None, 0)
    _fill_drawable_from_mask(channel, mask, width, height)
    return channel


def apply_selection(image, mask, width, height, operation="replace", feather=0.0):
    """Load the mask into the image's selection."""
    channel = mask_to_channel(image, mask, width, height, "sam-temp-selection")
    try:
        image.select_item(SELECTION_OPS.get(operation, Gimp.ChannelOps.REPLACE), channel)
    finally:
        image.remove_channel(channel)
    if feather and feather > 0:
        Gimp.Selection.feather(image, float(feather))
    return True


def save_as_channel(image, mask, width, height, name="Segment Anything"):
    """Keep the mask as a named channel the user can reuse later."""
    channel = mask_to_channel(image, mask, width, height, name)
    channel.set_visible(False)
    return channel


def add_layer_mask(image, layer, mask, width, height, replace_existing=True):
    """Attach the mask to ``layer`` as a layer mask."""
    existing = layer.get_mask()
    if existing is not None:
        if not replace_existing:
            raise RuntimeError(
                "%s already has a layer mask. Remove it first, or choose a "
                "different output." % layer.get_name()
            )
        layer.remove_mask(Gimp.MaskApplyMode.DISCARD)

    layer_mask = layer.create_mask(Gimp.AddMaskType.WHITE)
    layer.add_mask(layer_mask)
    offsets = layer.get_offsets()
    _fill_drawable_from_mask(layer_mask, mask, width, height,
                             offset_x=offsets[1], offset_y=offsets[2])
    return layer_mask


def copy_to_new_layer(image, layer, mask, width, height, name="Segment"):
    """Duplicate the layer and knock out everything outside the mask."""
    copy = layer.copy()
    copy.set_name(name)
    image.insert_layer(copy, layer.get_parent(), image.get_item_position(layer))
    if not copy.has_alpha():
        copy.add_alpha()

    layer_mask = copy.create_mask(Gimp.AddMaskType.WHITE)
    copy.add_mask(layer_mask)
    offsets = copy.get_offsets()
    _fill_drawable_from_mask(layer_mask, mask, width, height,
                             offset_x=offsets[1], offset_y=offsets[2])
    copy.remove_mask(Gimp.MaskApplyMode.APPLY)
    return copy
