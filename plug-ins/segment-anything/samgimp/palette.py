"""Overlay colours for regions.

Hues advance by the golden-ratio increment so consecutive regions are well
separated, with saturation and value alternating slightly to keep nearby
hues apart for colour-blind viewers.
"""

_GOLDEN = 0.6180339887498949


def _hsv_to_rgb(hue, saturation, value):
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - f * saturation)
    t = value * (1.0 - (1.0 - f) * saturation)
    i %= 6
    return [
        (value, t, p), (q, value, p), (p, value, t),
        (p, q, value), (t, p, value), (value, p, q),
    ][i]


_CACHE = {}


def color_for(index):
    """Stable (r, g, b) floats in 0..1 for segment ``index``."""
    if index in _CACHE:
        return _CACHE[index]
    hue = (0.13 + index * _GOLDEN) % 1.0
    saturation = 0.72 if index % 2 == 0 else 0.92
    value = 1.0 if index % 3 else 0.86
    rgb = _hsv_to_rgb(hue, saturation, value)
    _CACHE[index] = rgb
    return rgb


def hex_for(index):
    r, g, b = color_for(index)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
