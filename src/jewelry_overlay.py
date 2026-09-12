"""
jewelry_overlay.py
Composites a jewelry PNG (RGBA) onto a frame at a given position, scaled
to the detected body/hand size and rotated to match swing angle for
dangling items. Rigid items (rings, studs) skip rotation/physics.
"""

import cv2
import numpy as np


def _rotate_image(img_rgba, angle_degrees):
    """Rotates an RGBA image around its center, expanding the canvas so
    nothing gets clipped."""
    h, w = img_rgba.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        img_rgba, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )


def composite_at(frame_bgr, item_rgba, anchor_top_px, scale, angle_degrees=0.0):
    """
    Places item_rgba so its TOP-CENTER sits at anchor_top_px (the point it
    hangs/attaches from -- ear lobe, neck base, finger joint), scaled by
    `scale` (target width in pixels / item's native width), and rotated by
    angle_degrees (0 for rigid items like rings/studs).
    """
    if scale <= 0:
        return frame_bgr

    h0, w0 = item_rgba.shape[:2]
    new_w = max(1, int(w0 * scale))
    new_h = max(1, int(h0 * scale))
    resized = cv2.resize(item_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if angle_degrees != 0.0:
        resized = _rotate_image(resized, angle_degrees)

    rh, rw = resized.shape[:2]
    x0 = int(anchor_top_px[0] - rw / 2)
    y0 = int(anchor_top_px[1])

    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1 = x0 + rw, y0 + rh

    # Clip to frame bounds
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(frame_w, x1), min(frame_h, y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return frame_bgr

    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    patch = resized[src_y0:src_y1, src_x0:src_x1]
    alpha = patch[:, :, 3:4].astype(float) / 255.0
    rgb = patch[:, :, :3].astype(float)

    region = frame_bgr[dst_y0:dst_y1, dst_x0:dst_x1].astype(float)
    blended = rgb * alpha + region * (1 - alpha)
    frame_bgr[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)
    return frame_bgr