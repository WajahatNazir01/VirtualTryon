"""
chain_render.py
Draws the physics-simulated chain curve using colors/thickness sampled
from the ACTUAL chain photo (not AI-regenerated), so the product stays
visually accurate while still moving realistically.
"""

import cv2
import numpy as np


def sample_chain_appearance(chain_rgba):
    """
    Samples the dominant color and a rough visual 'thickness' from the
    actual chain cutout, so the rendered rope matches the real product's
    color/shine instead of a generic placeholder.
    """
    alpha = chain_rgba[:, :, 3]
    mask = alpha > 20
    if not np.any(mask):
        return (200, 200, 200), 4  # fallback: light gray, thin

    rgb_pixels = chain_rgba[:, :, :3][mask]
    avg_color_rgb = rgb_pixels.mean(axis=0)
    avg_color_bgr = (int(avg_color_rgb[2]), int(avg_color_rgb[1]), int(avg_color_rgb[0]))

    # Rough thickness estimate: how "thick" the visible chain pixels are
    # relative to the image's smaller dimension -- a thin chain photo has
    # a low mask-coverage ratio; a thick/bold chain has more.
    coverage_ratio = mask.sum() / mask.size
    thickness = max(2, int(coverage_ratio * min(chain_rgba.shape[:2]) * 0.15))

    return avg_color_bgr, thickness


def draw_chain(frame_bgr, points, color_bgr, thickness, highlight=True):
    """
    Fallback stylized renderer (plain colored rope) -- kept for cases with
    no source image or as a fast debug view. Prefer draw_textured_chain
    for actual product rendering.
    """
    pts = np.array(points, dtype=np.int32)

    cv2.polylines(frame_bgr, [pts], isClosed=False, color=color_bgr,
                  thickness=thickness, lineType=cv2.LINE_AA)

    if highlight:
        highlight_color = tuple(min(255, c + 60) for c in color_bgr)
        cv2.polylines(frame_bgr, [pts], isClosed=False, color=highlight_color,
                      thickness=max(1, thickness // 3), lineType=cv2.LINE_AA)

    for p in points:
        cv2.circle(frame_bgr, (int(p[0]), int(p[1])), max(1, thickness // 2),
                    color_bgr, -1, lineType=cv2.LINE_AA)

    return frame_bgr


def draw_textured_chain(frame_bgr, points, chain_rgba, thickness_px=None):
    """
    Maps the ACTUAL chain photo onto the simulated curve: slices the
    source image into one strip per curve segment (left-to-right across
    the image = along the chain's length), and perspective-warps each
    strip onto its corresponding segment of the curve. This is what
    actually shows your real product, not a generic colored line.
    """
    ch, cw = chain_rgba.shape[:2]
    if thickness_px is None:
        thickness_px = max(8, ch // 3)

    num_segments = len(points) - 1
    if num_segments < 1:
        return frame_bgr

    frame_h, frame_w = frame_bgr.shape[:2]

    for i in range(num_segments):
        p0 = np.array(points[i], dtype=float)
        p1 = np.array(points[i + 1], dtype=float)
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-3:
            continue
        unit_dir = seg_vec / seg_len
        normal = np.array([-unit_dir[1], unit_dir[0]])
        half_t = thickness_px / 2

        dst_quad = np.float32([
            p0 - normal * half_t,
            p1 - normal * half_t,
            p1 + normal * half_t,
            p0 + normal * half_t,
        ])

        x0 = int(cw * i / num_segments)
        x1 = int(cw * (i + 1) / num_segments)
        x1 = max(x1, x0 + 1)
        src_slice = chain_rgba[:, x0:x1]
        sh, sw = src_slice.shape[:2]
        src_quad = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])

        r = cv2.boundingRect(dst_quad)
        rx, ry, rw, rh = r
        if rw <= 0 or rh <= 0:
            continue
        dst_quad_local = dst_quad - np.float32([rx, ry])

        M = cv2.getPerspectiveTransform(src_quad, dst_quad_local)
        warped = cv2.warpPerspective(
            src_slice, M, (rw, rh), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )

        fx0, fy0 = max(0, rx), max(0, ry)
        fx1, fy1 = min(frame_w, rx + rw), min(frame_h, ry + rh)
        if fx1 <= fx0 or fy1 <= fy0:
            continue
        wx0, wy0 = fx0 - rx, fy0 - ry
        wx1, wy1 = wx0 + (fx1 - fx0), wy0 + (fy1 - fy0)

        patch = warped[wy0:wy1, wx0:wx1]
        if patch.shape[2] != 4:
            continue
        alpha = patch[:, :, 3:4].astype(float) / 255.0
        rgb = patch[:, :, :3].astype(float)
        region = frame_bgr[fy0:fy1, fx0:fx1].astype(float)
        blended = rgb * alpha + region * (1 - alpha)
        frame_bgr[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)

    return frame_bgr