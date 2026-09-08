"""
warp.py
Category-aware anchor detection + warping. Instead of one hardcoded
anchor strategy for every garment, each category (upper_body_sleeved,
upper_body_sleeveless, lower_body, headwear) has its own garment-side
anchor detector and body-side target function. category_classifier.py
picks which category to use automatically.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Generic pixel-scanning primitives (shared building blocks)
# ---------------------------------------------------------------------------

def _row_extent(alpha, y):
    nonzero = np.where(alpha[y] > 10)[0]
    if len(nonzero) == 0:
        return None
    return int(nonzero[0]), int(nonzero[-1])


def _max_width_in_band(alpha, y0, y1):
    max_width = -1
    for y in range(y0, y1):
        ext = _row_extent(alpha, y)
        if ext:
            max_width = max(max_width, ext[1] - ext[0])
    return max_width


def topmost_near_full_width_row(alpha, y_start_frac, y_end_frac, threshold_ratio=0.9):
    """First row (top-down) whose width is already close to the band's max.
    Correct for garments that flare outward immediately (sleeves)."""
    h = alpha.shape[0]
    y0, y1 = int(h * y_start_frac), int(h * y_end_frac)
    max_width = _max_width_in_band(alpha, y0, y1)
    if max_width <= 0:
        return None, None, None
    threshold = max_width * threshold_ratio
    for y in range(y0, y1):
        ext = _row_extent(alpha, y)
        if ext and (ext[1] - ext[0]) >= threshold:
            return y, ext[0], ext[1]
    return None, None, None


def bottommost_near_full_width_row(alpha, y_start_frac, y_end_frac, threshold_ratio=0.9):
    """Mirror of topmost_near_full_width_row, scanning from the bottom up."""
    h = alpha.shape[0]
    y0, y1 = int(h * y_start_frac), int(h * y_end_frac)
    max_width = _max_width_in_band(alpha, y0, y1)
    if max_width <= 0:
        return None, None, None
    threshold = max_width * threshold_ratio
    for y in range(y1 - 1, y0 - 1, -1):
        ext = _row_extent(alpha, y)
        if ext and (ext[1] - ext[0]) >= threshold:
            return y, ext[0], ext[1]
    return None, None, None


def topmost_row_with_content(alpha, y_max_frac):
    """Very first row (from image top) with any visible pixel at all.
    Correct for narrow straps that never reach 'near full width' early."""
    h = alpha.shape[0]
    y1 = int(h * y_max_frac)
    for y in range(0, y1):
        ext = _row_extent(alpha, y)
        if ext:
            return y, ext[0], ext[1]
    return None, None, None


def _true_widest_row_in_band(alpha, y_start_frac, y_end_frac):
    """Returns the single widest row's (y, left, right) in the band --
    used to find actual sleeve tips, unlike topmost_near_full_width_row
    which deliberately looks for the top-most near-max-width row instead."""
    h = alpha.shape[0]
    y0, y1 = int(h * y_start_frac), int(h * y_end_frac)
    best = (-1, None, None, None)
    for y in range(y0, y1):
        ext = _row_extent(alpha, y)
        if ext:
            width = ext[1] - ext[0]
            if width > best[0]:
                best = (width, y, ext[0], ext[1])
    return best[1], best[2], best[3]


def _fallback(h, w, y_frac, l_frac=0.1, r_frac=0.9):
    return int(h * y_frac), int(w * l_frac), int(w * r_frac)


# ---------------------------------------------------------------------------
# Category-specific garment-side anchor detectors
# All return 4 points: [top-left, top-right, bottom-right, bottom-left]
# ---------------------------------------------------------------------------

def _anchors_upper_body_sleeved(garment_rgba, shoulder_band=(0.10, 0.40), hem_band=(0.85, 1.0)):
    alpha = garment_rgba[:, :, 3]
    h, w = alpha.shape
    sy, lsx, rsx = topmost_near_full_width_row(alpha, *shoulder_band)
    hy, lhx, rhx = bottommost_near_full_width_row(alpha, *hem_band)
    if sy is None:
        sy, lsx, rsx = _fallback(h, w, 0.2)
    if hy is None:
        hy, lhx, rhx = _fallback(h, w, 0.9)
    return np.float32([[lsx, sy], [rsx, sy], [rhx, hy], [lhx, hy]])


def _anchors_upper_body_sleeveless(garment_rgba, shoulder_band=(0.10, 0.40), hem_band=(0.85, 1.0)):
    alpha = garment_rgba[:, :, 3]
    h, w = alpha.shape
    sy, lsx, rsx = topmost_row_with_content(alpha, shoulder_band[1])
    hy, lhx, rhx = bottommost_near_full_width_row(alpha, *hem_band)
    if sy is None:
        sy, lsx, rsx = _fallback(h, w, 0.05)
    if hy is None:
        hy, lhx, rhx = _fallback(h, w, 0.9)
    return np.float32([[lsx, sy], [rsx, sy], [rhx, hy], [lhx, hy]])


def _anchors_lower_body(garment_rgba, waist_band=(0.0, 0.20), hem_band=(0.85, 1.0)):
    """Waistband = top (near-full-width, trousers are widest at the waist),
    combined leg-opening width = bottom."""
    alpha = garment_rgba[:, :, 3]
    h, w = alpha.shape
    wy, lwx, rwx = topmost_near_full_width_row(alpha, *waist_band)
    hy, lhx, rhx = bottommost_near_full_width_row(alpha, *hem_band)
    if wy is None:
        wy, lwx, rwx = _fallback(h, w, 0.05)
    if hy is None:
        hy, lhx, rhx = _fallback(h, w, 0.95)
    return np.float32([[lwx, wy], [rwx, wy], [rhx, hy], [lhx, hy]])


def _anchors_headwear(garment_rgba, brim_band=(0.30, 0.60), crown_frac=0.05):
    """Brim (widest part) = bottom edge of the quad; crown top = top edge,
    using the same left/right extent as the brim for a simple box fit."""
    alpha = garment_rgba[:, :, 3]
    h, w = alpha.shape
    by, lbx, rbx = topmost_near_full_width_row(alpha, *brim_band, threshold_ratio=0.85)
    if by is None:
        by, lbx, rbx = _fallback(h, w, 0.5)
    ty = int(h * crown_frac)
    return np.float32([[lbx, ty], [rbx, ty], [rbx, by], [lbx, by]])


ANCHOR_DETECTORS = {
    "upper_body_sleeved": _anchors_upper_body_sleeved,
    "upper_body_sleeveless": _anchors_upper_body_sleeveless,
    "lower_body": _anchors_lower_body,
    "headwear": _anchors_headwear,
}


def garment_anchor_points(garment_rgba, category="upper_body_sleeved"):
    """Dispatches to the right category-specific anchor detector."""
    fn = ANCHOR_DETECTORS.get(category, _anchors_upper_body_sleeved)
    return fn(garment_rgba)


# ---------------------------------------------------------------------------
# 6-point mesh landmarks for upper-body garments (shoulders, sleeve tips, hem)
# A shirt is not a flat rectangle -- sleeves stick outward independently of
# the torso/hem, so a single 4-point quad warp always distorts it. These 6
# points let mesh_warp.py triangulate and warp each region independently.
# ---------------------------------------------------------------------------

def garment_landmarks_upper_body(garment_rgba, sleeveless=False,
                                  shoulder_band=(0.10, 0.40),
                                  sleeve_band=(0.05, 0.55),
                                  hem_band=(0.85, 1.0)):
    alpha = garment_rgba[:, :, 3]
    h, w = alpha.shape

    if sleeveless:
        sy, lsx, rsx = topmost_row_with_content(alpha, shoulder_band[1])
    else:
        sy, lsx, rsx = topmost_near_full_width_row(alpha, *shoulder_band)
    if sy is None:
        sy, lsx, rsx = _fallback(h, w, 0.15)

    sleeve_y, sl_lx, sl_rx = _true_widest_row_in_band(alpha, *sleeve_band)
    if sleeve_y is None:
        sleeve_y, sl_lx, sl_rx = sy, lsx, rsx

    hy, lhx, rhx = bottommost_near_full_width_row(alpha, *hem_band)
    if hy is None:
        hy, lhx, rhx = _fallback(h, w, 0.9)

    return {
        "left_shoulder": (lsx, sy),
        "right_shoulder": (rsx, sy),
        "left_sleeve": (sl_lx, sleeve_y),
        "right_sleeve": (sl_rx, sleeve_y),
        "left_hem": (lhx, hy),
        "right_hem": (rhx, hy),
    }


def body_landmarks_upper_body(pose_points, hem_ratio=1.1, width_multiplier=1.25):
    ls = np.array(pose_points["left_shoulder"], dtype=float)
    rs = np.array(pose_points["right_shoulder"], dtype=float)
    le = np.array(pose_points["left_elbow"], dtype=float)
    re = np.array(pose_points["right_elbow"], dtype=float)

    ls_w, rs_w = _widen(ls, rs, width_multiplier)
    shoulder_width = np.linalg.norm(ls_w - rs_w)
    drop = shoulder_width * hem_ratio
    l_hem, r_hem = ls_w + [0, drop], rs_w + [0, drop]

    return {
        "left_shoulder": tuple(ls_w),
        "right_shoulder": tuple(rs_w),
        "left_sleeve": tuple(le),
        "right_sleeve": tuple(re),
        "left_hem": tuple(l_hem),
        "right_hem": tuple(r_hem),
    }


# ---------------------------------------------------------------------------
# Category-specific body-side target quads
# ---------------------------------------------------------------------------

def _widen(a, b, multiplier):
    center = (a + b) / 2
    return center + (a - center) * multiplier, center + (b - center) * multiplier


def _target_upper_body(pose_points, hem_ratio=1.1, width_multiplier=1.25):
    ls = np.array(pose_points["left_shoulder"], dtype=float)
    rs = np.array(pose_points["right_shoulder"], dtype=float)
    ls, rs = _widen(ls, rs, width_multiplier)
    shoulder_width = np.linalg.norm(ls - rs)
    drop = shoulder_width * hem_ratio
    l_hem, r_hem = ls + [0, drop], rs + [0, drop]
    return np.float32([rs, ls, l_hem, r_hem])


def _target_lower_body(pose_points, width_multiplier=1.15):
    lh = np.array(pose_points["left_hip"], dtype=float)
    rh = np.array(pose_points["right_hip"], dtype=float)
    la = np.array(pose_points["left_ankle"], dtype=float)
    ra = np.array(pose_points["right_ankle"], dtype=float)
    lh, rh = _widen(lh, rh, width_multiplier)
    la, ra = _widen(la, ra, width_multiplier)
    return np.float32([rh, lh, la, ra])


def _target_headwear(pose_points, width_multiplier=1.3, top_ratio=0.9):
    le = np.array(pose_points["left_ear"], dtype=float)
    re = np.array(pose_points["right_ear"], dtype=float)
    nose = np.array(pose_points["nose"], dtype=float)
    le, re = _widen(le, re, width_multiplier)
    ear_width = np.linalg.norm(le - re)
    rise = ear_width * top_ratio
    top_l, top_r = le + [0, -rise], re + [0, -rise]
    return np.float32([top_r, top_l, le, re])


TARGET_FUNCS = {
    "upper_body_sleeved": _target_upper_body,
    "upper_body_sleeveless": _target_upper_body,
    "lower_body": _target_lower_body,
    "headwear": _target_headwear,
}

REQUIRED_LANDMARKS = {
    "upper_body_sleeved": ["left_shoulder", "right_shoulder"],
    "upper_body_sleeveless": ["left_shoulder", "right_shoulder"],
    "lower_body": ["left_hip", "right_hip", "left_ankle", "right_ankle"],
    "headwear": ["left_ear", "right_ear", "nose"],
}


def body_target_points(pose_points, category="upper_body_sleeved"):
    """Dispatches to the right category-specific body target function."""
    fn = TARGET_FUNCS.get(category, _target_upper_body)
    return fn(pose_points)


# ---------------------------------------------------------------------------
# Warp + composite (unchanged, category-agnostic)
# ---------------------------------------------------------------------------

def warp_garment(garment_rgba, src_pts, dst_pts, frame_shape):
    h, w = frame_shape[:2]
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(
        garment_rgba, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def composite(frame_bgr, warped_rgba):
    alpha = warped_rgba[:, :, 3:4].astype(float) / 255.0
    garment_rgb = warped_rgba[:, :, :3].astype(float)
    frame = frame_bgr.astype(float)
    blended = garment_rgb * alpha + frame * (1 - alpha)
    return blended.astype(np.uint8)