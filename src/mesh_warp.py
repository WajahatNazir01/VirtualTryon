"""
mesh_warp.py
Piecewise-affine (triangulated) warping between a garment's 6 landmark
points and 6 corresponding live body points. Unlike a single 4-point
perspective warp, this lets the sleeves warp independently from the
torso/hem -- which is what a shirt's actual shape requires.
"""

import numpy as np
import cv2

# Hexagon traversal order: top-left, top-right, then clockwise down the
# right side, across the bottom, and back up the left side. A synthetic
# "center" point (added at warp time) turns this into 6 fan triangles.
HEXAGON_ORDER = [
    "left_shoulder", "right_shoulder", "right_sleeve",
    "right_hem", "left_hem", "left_sleeve",
]


def _center_of(points: dict):
    keys = ["left_shoulder", "right_shoulder", "left_hem", "right_hem"]
    pts = np.array([points[k] for k in keys], dtype=float)
    return tuple(pts.mean(axis=0))


def build_triangles(points: dict):
    """Returns a list of (name_a, name_b, 'center') triangles covering the
    hexagon, and injects 'center' into a copy of the points dict."""
    pts = dict(points)
    pts["center"] = _center_of(points)
    n = len(HEXAGON_ORDER)
    triangles = []
    for i in range(n):
        a = HEXAGON_ORDER[i]
        b = HEXAGON_ORDER[(i + 1) % n]
        triangles.append((a, b, "center"))
    return triangles, pts


def _warp_triangle(src_img, dst_canvas, t_src, t_dst):
    """Warps one triangular region from src_img into dst_canvas (RGBA)."""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t_src_rect = np.float32([(p[0] - r1[0], p[1] - r1[1]) for p in t_src])
    t_dst_rect = np.float32([(p[0] - r2[0], p[1] - r2[1]) for p in t_dst])

    sx0, sy0 = max(r1[0], 0), max(r1[1], 0)
    sx1, sy1 = min(r1[0] + r1[2], src_img.shape[1]), min(r1[1] + r1[3], src_img.shape[0])
    if sx1 <= sx0 or sy1 <= sy0:
        return
    src_rect = src_img[sy0:sy1, sx0:sx1]
    if src_rect.shape[0] != r1[3] or src_rect.shape[1] != r1[2]:
        # Triangle bounding box partly outside the source image -- pad.
        padded = np.zeros((r1[3], r1[2], src_img.shape[2]), dtype=src_img.dtype)
        padded[0:src_rect.shape[0], 0:src_rect.shape[1]] = src_rect
        src_rect = padded

    dx0, dy0 = max(r2[0], 0), max(r2[1], 0)
    dx1, dy1 = min(r2[0] + r2[2], dst_canvas.shape[1]), min(r2[1] + r2[3], dst_canvas.shape[0])
    if dx1 <= dx0 or dy1 <= dy0:
        return

    warp_mat = cv2.getAffineTransform(t_src_rect, t_dst_rect)
    size = (r2[2], r2[3])
    warped = cv2.warpAffine(src_rect, warp_mat, size, None,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rect), 255)

    # Clip to the visible part of the destination canvas
    clip_x0, clip_y0 = dx0 - r2[0], dy0 - r2[1]
    clip_x1, clip_y1 = dx1 - r2[0], dy1 - r2[1]
    warped = warped[clip_y0:clip_y1, clip_x0:clip_x1]
    mask = mask[clip_y0:clip_y1, clip_x0:clip_x1]

    idx = mask > 0
    dst_patch = dst_canvas[dy0:dy1, dx0:dx1]
    dst_patch[idx] = warped[idx]
    dst_canvas[dy0:dy1, dx0:dx1] = dst_patch


def warp_garment_mesh(garment_rgba, src_points: dict, dst_points: dict, frame_shape):
    """
    Warps the garment onto a transparent canvas the size of the frame,
    using 6-point triangulation instead of a single 4-point quad.
    Returns an RGBA canvas ready for warp.composite().
    """
    h, w = frame_shape[:2]
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    triangles, src_pts = build_triangles(src_points)
    _, dst_pts = build_triangles(dst_points)

    for a, b, c in triangles:
        t_src = [src_pts[a], src_pts[b], src_pts[c]]
        t_dst = [dst_pts[a], dst_pts[b], dst_pts[c]]
        _warp_triangle(garment_rgba, canvas, t_src, t_dst)

    return canvas