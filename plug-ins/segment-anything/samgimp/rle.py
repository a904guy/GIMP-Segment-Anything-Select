"""Run-length decoding for the plug-in side.

numpy cannot be assumed present in GIMP's bundled Python, so masks are
expanded into a ``bytearray`` using slice assignment.  One slice write per
run keeps the cost low even for multi-megapixel masks.
"""

import cairo

_ON = b"\xff"


def decode(counts, width, height, stride=None):
    """Expand run lengths into a bytearray of 0/255 bytes.

    When ``stride`` is given the rows are padded to that width, which is
    what cairo wants for an A8 surface.
    """
    size = width * height
    flat = bytearray(size)
    position = 0
    value = False
    for count in counts:
        if value and count:
            end = position + count
            if end > size:
                end = size
            flat[position:end] = _ON * (end - position)
        position += count
        if position >= size and value:
            break
        value = not value

    if stride is None or stride == width:
        return flat
    padded = bytearray(stride * height)
    for row in range(height):
        source = row * width
        target = row * stride
        padded[target:target + width] = flat[source:source + width]
    return padded


def union_into(target, counts, width, height, stride):
    """OR a run-length mask into an existing strided buffer."""
    position = 0
    value = False
    size = width * height
    for count in counts:
        if value and count:
            start, end = position, min(position + count, size)
            row = start // width
            offset = start % width
            remaining = end - start
            while remaining > 0:
                take = min(remaining, width - offset)
                base = row * stride + offset
                target[base:base + take] = _ON * take
                remaining -= take
                row += 1
                offset = 0
        position += count
        if position >= size and value:
            break
        value = not value
    return target


def a8_surface(counts, width, height):
    """Build a cairo alpha-only surface from run lengths."""
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, width)
    data = decode(counts, width, height, stride)
    # pycairo keeps its own reference to ``data`` for the surface's lifetime.
    return cairo.ImageSurface.create_for_data(data, cairo.FORMAT_A8, width, height, stride)


def empty_a8(width, height):
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, width)
    data = bytearray(stride * height)
    surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_A8, width, height, stride)
    return surface, data, stride
