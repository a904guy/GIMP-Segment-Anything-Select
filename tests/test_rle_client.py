"""Plug-in side run-length decoding (pure Python, no numpy)."""

import conftest_paths  # noqa: F401  (puts the plug-in on sys.path)
from conftest_paths import check

import cairo

from samgimp import palette, rle


def test_decode():
    check(list(rle.decode([3, 5, 2], 5, 2)) == [0, 0, 0, 255, 255, 255, 255, 255, 0, 0],
          "runs expand to the right bytes")
    check(list(rle.decode([0, 4], 2, 2)) == [255] * 4,
          "a leading zero-run means the mask starts filled")
    check(list(rle.decode([], 2, 2)) == [0] * 4, "no runs means an empty mask")
    check(len(rle.decode([2, 2], 2, 2)) == 4, "the buffer is exactly width*height")
    # A run longer than the buffer must be clipped, not raise.
    check(len(rle.decode([0, 999], 2, 2)) == 4, "over-long runs are clipped safely")


def test_padded_and_surface():
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, 5)
    padded = rle.decode([3, 5, 2], 5, 2, stride=stride)
    check(len(padded) == stride * 2, "padded buffers match the cairo stride")
    check(list(padded[0:5]) == [0, 0, 0, 255, 255], "row 0 content survives padding")

    surface = rle.a8_surface([3, 5, 2], 5, 2)
    check(surface.get_format() == cairo.FORMAT_A8 and surface.get_width() == 5,
          "a cairo alpha surface is produced")


def test_union():
    surface, data, stride = rle.empty_a8(10, 4)
    rle.union_into(data, [5, 10, 5], 10, 4, stride)
    counts = [list(data[y * stride:y * stride + 10]).count(255) for y in range(4)]
    check(counts == [5, 5, 0, 0], "a run spanning rows lands on the right pixels")


def test_palette():
    colors = [palette.hex_for(i) for i in range(64)]
    check(len(set(colors)) == 64, "the first 64 region colours are all distinct")
    check(palette.hex_for(3) == palette.hex_for(3), "colours are stable per region")


def run():
    test_decode()
    test_padded_and_surface()
    test_union()
    test_palette()


if __name__ == "__main__":
    run()
