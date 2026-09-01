"""
warp.py
Maps a garment's reference corners onto detected body keypoints,
warps the garment image to fit, and alpha-composites it onto the video frame.
"""

import cv2
import numpy as np


def garment_anchor_points(garment_rgba, garment_type="sleeved", shoulder_band=(0.10, 0.40), hem_band=(0.85, 1.0)):
    """
    Automatically detects the 4 reference corners on the garment cutout by
    scanning its alpha mask, instead of guessing fixed percentages. This
    handles non-rectangular silhouettes (flared/tapered sleeves, etc.)
    much better than hardcoded coordinates.

    - Shoulder points: within the upper band of the image, finds the row
      where the garment is widest (its sleeve tips), and uses that row's
      left/right edges.
    - Hem points: within the lower band, finds the bottommost row that
      still has visible pixels, and uses that row's left/right edges.

    Order: top-left (left shoulder), top-right (right shoulder),
           bottom-right (right hem), bottom-left (left hem)
    """
    alpha = garment_rgba[:, :, 3]
    h, w = alpha.shape

    def widest_row_in_band(y_start_frac, y_end_frac):
        """Returns the max garment width found anywhere in the band."""
        y0, y1 = int(h * y_start_frac), int(h * y_end_frac)
        max_width = -1
        for y in range(y0, y1):
            nonzero = np.where(alpha[y] > 10)[0]
            if len(nonzero) == 0:
                continue
            width = int(nonzero[-1]) - int(nonzero[0])
            if width > max_width:
                max_width = width
        return max_width

    def topmost_near_full_width_row(y_start_frac, y_end_frac, threshold_ratio=0.9):
        """
        Scans top-to-bottom and returns the first (topmost) row whose width
        is already close to the band's max width. This correctly finds
        strap tops on tank tops (narrow-then-suddenly-wide) as well as
        sleeve tips on t-shirts (already wide near the top), instead of
        drifting down to wherever the single widest row happens to be.
        """
        y0, y1 = int(h * y_start_frac), int(h * y_end_frac)
        max_width = widest_row_in_band(y_start_frac, y_end_frac)
        if max_width <= 0:
            return None, None, None
        threshold = max_width * threshold_ratio
        for y in range(y0, y1):
            nonzero = np.where(alpha[y] > 10)[0]
            if len(nonzero) == 0:
                continue
            left, right = int(nonzero[0]), int(nonzero[-1])
            if (right - left) >= threshold:
                return y, left, right
        return None, None, None

    def bottommost_near_full_width_row(y_start_frac, y_end_frac, threshold_ratio=0.9):
        """Mirror of topmost_near_full_width_row, scanning from the bottom up."""
        y0, y1 = int(h * y_start_frac), int(h * y_end_frac)
        max_width = widest_row_in_band(y_start_frac, y_end_frac)
        if max_width <= 0:
            return None, None, None
        threshold = max_width * threshold_ratio
        for y in range(y1 - 1, y0 - 1, -1):
            nonzero = np.where(alpha[y] > 10)[0]
            if len(nonzero) == 0:
                continue
            left, right = int(nonzero[0]), int(nonzero[-1])
            if (right - left) >= threshold:
                return y, left, right
        return None, None, None

    def topmost_row_with_content(y_max_frac):
        """
        Returns the very topmost row (from the top of the image) that has
        any non-transparent pixels, with its left/right extent. Used for
        sleeveless garments, where the straps are narrower than the torso,
        so waiting for "near full width" skips past them entirely.
        """
        y1 = int(h * y_max_frac)
        for y in range(0, y1):
            nonzero = np.where(alpha[y] > 10)[0]
            if len(nonzero) > 0:
                return y, int(nonzero[0]), int(nonzero[-1])
        return None, None, None

    if garment_type == "sleeveless":
        shoulder_row, l_shoulder_x, r_shoulder_x = topmost_row_with_content(shoulder_band[1])
    else:
        shoulder_row, l_shoulder_x, r_shoulder_x = topmost_near_full_width_row(*shoulder_band)
    hem_row, l_hem_x, r_hem_x = bottommost_near_full_width_row(*hem_band)

    # Fallback to simple percentages if detection fails (e.g. blank band)
    if shoulder_row is None:
        shoulder_row, l_shoulder_x, r_shoulder_x = int(h * 0.2), int(w * 0.1), int(w * 0.9)
    if hem_row is None:
        hem_row, l_hem_x, r_hem_x = int(h * 0.9), int(w * 0.1), int(w * 0.9)

    pts = np.float32([
        [l_shoulder_x, shoulder_row],
        [r_shoulder_x, shoulder_row],
        [r_hem_x, hem_row],
        [l_hem_x, hem_row],
    ])
    return pts


def body_target_points(pose_points, hem_ratio=1.1, width_multiplier=1.25):
    """
    Builds the destination quadrilateral on the person's body for a shirt.

    Top corners use the detected shoulders, widened by width_multiplier --
    MediaPipe's shoulder landmarks sit at the shoulder joint, which is
    inside the body's actual outer edge, so the raw landmark distance
    under-measures real shoulder width. Bottom corners are estimated as a
    multiple of (widened) shoulder width below the shoulders, rather than
    using detected hips directly -- hip landmarks are unreliable when
    seated or occluded (e.g. behind a desk/chair/arms).

    width_multiplier scales the shoulder-to-shoulder distance outward from
    its center point. ~1.25 is a reasonable starting point; increase if
    shoulders still stick out past the shirt, decrease if it's too wide.

    hem_ratio controls how far down the hem sits, in units of (widened)
    shoulder width. ~1.1 approximates a normal t-shirt length; increase for
    longer shirts, decrease for cropped ones.
    """
    left_shoulder = np.array(pose_points["left_shoulder"], dtype=float)
    right_shoulder = np.array(pose_points["right_shoulder"], dtype=float)

    center = (left_shoulder + right_shoulder) / 2
    left_shoulder = center + (left_shoulder - center) * width_multiplier
    right_shoulder = center + (right_shoulder - center) * width_multiplier

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hem_drop = shoulder_width * hem_ratio

    left_hem = left_shoulder + np.array([0, hem_drop])
    right_hem = right_shoulder + np.array([0, hem_drop])

    dst = np.float32([
        right_shoulder,  # top-left in image = person's right shoulder (mirrored view)
        left_shoulder,   # top-right = person's left shoulder
        left_hem,        # bottom-right = estimated hem below left shoulder
        right_hem,       # bottom-left = estimated hem below right shoulder
    ])
    return dst


def warp_garment(garment_rgba, src_pts, dst_pts, frame_shape):
    """
    Warps the garment (with alpha channel) using a perspective transform
    so its src_pts land on dst_pts, onto a canvas the size of the frame.
    """
    h, w = frame_shape[:2]
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        garment_rgba, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def composite(frame_bgr, warped_rgba):
    """
    Alpha-blends the warped garment (BGRA) onto the frame (BGR).
    """
    alpha = warped_rgba[:, :, 3:4].astype(float) / 255.0
    garment_rgb = warped_rgba[:, :, :3].astype(float)
    frame = frame_bgr.astype(float)

    blended = garment_rgb * alpha + frame * (1 - alpha)
    return blended.astype(np.uint8)