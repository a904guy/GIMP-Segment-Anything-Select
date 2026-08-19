"""Mask post-processing: run-length encoding, nesting and label maps.

This runs in the private environment, so numpy is available here.  Masks
are sent to the plug-in as run-length data rather than raw bitmaps, which
takes a 4000x3000 mask from 12 MB down to a few kilobytes and can be
expanded again without numpy on the receiving side.
"""

import numpy as np


def encode_rle(mask):
    """Row-major run lengths, always starting with a run of background.

    ``[3, 5, 2]`` means 3 background pixels, then 5 foreground, then 2
    background, reading the mask left-to-right, top-to-bottom.
    """
    flat = np.ascontiguousarray(mask, dtype=bool).ravel()
    if flat.size == 0:
        return []
    changes = np.flatnonzero(flat[1:] != flat[:-1])
    edges = np.concatenate(([-1], changes, [flat.size - 1]))
    counts = np.diff(edges).astype(np.int64).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return counts


def decode_rle(counts, width, height):
    """Inverse of :func:`encode_rle` (used by the backend's own tests)."""
    flat = np.zeros(width * height, dtype=bool)
    pos = 0
    value = False
    for count in counts:
        if value and count:
            flat[pos:pos + count] = True
        pos += count
        value = not value
    return flat.reshape(height, width)


def bbox_of(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y0, y1 = np.flatnonzero(rows)[[0, -1]]
    x0, x1 = np.flatnonzero(cols)[[0, -1]]
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]


def downscale_mask(mask, size):
    """Nearest-neighbour resize of a boolean mask to (width, height)."""
    width, height = size
    src_h, src_w = mask.shape
    if (src_w, src_h) == (width, height):
        return mask
    ys = (np.arange(height) * (src_h / float(height))).astype(np.int64).clip(0, src_h - 1)
    xs = (np.arange(width) * (src_w / float(width))).astype(np.int64).clip(0, src_w - 1)
    return mask[ys[:, None], xs[None, :]]


def upscale_mask(mask, size):
    """Nearest-neighbour resize back up to full resolution."""
    return downscale_mask(mask, size)


def build_hierarchy(masks, containment=0.85, min_ratio=1.25, grid=192):
    """Nest each mask under the smallest mask that visually contains it.

    Returns a list of parent indices (``None`` for top-level masks).  The
    test is done on heavily downscaled copies and vectorised as a single
    matrix product, so a hundred masks cost a fraction of a second.
    """
    count = len(masks)
    parents = [None] * count
    if count < 2:
        return parents

    height, width = masks[0].shape
    scale = max(height, width) / float(grid)
    small_size = (max(1, int(round(width / scale))), max(1, int(round(height / scale))))

    stack = np.stack([
        downscale_mask(mask, small_size).ravel() for mask in masks
    ]).astype(np.float32)
    areas = stack.sum(axis=1)
    # intersection[i, j] = pixels shared by mask i and mask j
    intersection = stack @ stack.T

    order = np.argsort(-areas)  # largest first
    for small_idx in range(count):
        small_area = areas[small_idx]
        if small_area <= 0:
            continue
        best = None
        best_area = None
        for big_idx in order:
            if big_idx == small_idx:
                continue
            big_area = areas[big_idx]
            if big_area < small_area * min_ratio:
                continue
            if intersection[small_idx, big_idx] / small_area < containment:
                continue
            # ``order`` is largest-first, so keep scanning for a tighter fit
            if best_area is None or big_area < best_area:
                best, best_area = int(big_idx), big_area
        parents[small_idx] = best
    return parents


def build_label_map(masks, size):
    """A single-byte-per-pixel map for instant click hit-testing.

    Pixel value ``i`` means "segment ``i`` is the smallest segment covering
    this pixel"; 255 means no segment.  Painting from largest to smallest
    means the smallest one wins, which is what a user expects when they
    click inside a nested object.
    """
    width, height = size
    label = np.full((height, width), 255, dtype=np.uint8)
    areas = [int(mask.sum()) for mask in masks]
    for index in sorted(range(len(masks)), key=lambda i: -areas[i])[:255]:
        small = downscale_mask(masks[index], size)
        label[small] = index
    return label
