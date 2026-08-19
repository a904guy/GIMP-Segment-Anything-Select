"""Mask encoding, hierarchy and label maps (backend side, needs numpy)."""

import sys

import conftest_paths as helpers
from conftest_paths import check

sys.path.insert(0, helpers.BACKEND)

import numpy as np  # noqa: E402
import maskutil  # noqa: E402


def test_rle_round_trip():
    rng = np.random.default_rng(1234)
    survived = all(
        (maskutil.decode_rle(maskutil.encode_rle(mask), 23, 17) == mask).all()
        for mask in (rng.random((17, 23)) > 0.6 for _ in range(200))
    )
    check(survived, "200 random masks survive an encode/decode round trip")

    for filled in (np.ones((4, 4), bool), np.zeros((4, 4), bool)):
        counts = maskutil.encode_rle(filled)
        check((maskutil.decode_rle(counts, 4, 4) == filled).all(),
              "all-%s mask round trips" % ("on" if filled.any() else "off"))
        check(counts[0] == 0 or not filled[0][0],
              "encoding always starts with a background run")


def test_hierarchy_and_labels():
    big = np.zeros((100, 100), bool); big[10:90, 10:90] = True
    inner_a = np.zeros((100, 100), bool); inner_a[20:40, 20:40] = True
    inner_b = np.zeros((100, 100), bool); inner_b[60:80, 60:80] = True
    outside = np.zeros((100, 100), bool); outside[95:100, 0:5] = True

    parents = maskutil.build_hierarchy([big, inner_a, inner_b, outside])
    check(parents == [None, 0, 0, None],
          "contained masks are nested under their container")

    labels = maskutil.build_label_map([big, inner_a, inner_b, outside], (50, 50))
    check(labels[15, 15] == 1, "the smallest covering mask wins a pixel")
    check(labels[25, 25] == 0, "a pixel only inside the big mask maps to it")
    check(labels[0, 49] == 255, "uncovered pixels are marked empty")


def test_bbox_and_scaling():
    mask = np.zeros((60, 80), bool); mask[10:30, 20:50] = True
    check(maskutil.bbox_of(mask) == [20, 10, 30, 20], "bounding box is exact")
    check(maskutil.bbox_of(np.zeros((5, 5), bool)) == [0, 0, 0, 0],
          "an empty mask has an empty bounding box")
    small = maskutil.downscale_mask(mask, (40, 30))
    check(small.shape == (30, 40), "downscaling gives the requested shape")
    check(small.any(), "downscaling keeps the region")


def run():
    test_rle_round_trip()
    test_hierarchy_and_labels()
    test_bbox_and_scaling()


if __name__ == "__main__":
    run()
