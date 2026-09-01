"""
warp.py
Maps a garment's reference corners onto detected body keypoints,
warps the garment image to fit, and alpha-composites it onto the video frame.
"""

import cv2
import numpy as np


def garment_anchor_points(garment_rgba):
    """
    Defines the 4 reference corners on the garment cutout image itself,
    as percentages of its width/height. Assumes the garment is a shirt
    cropped fairly tightly (little padding around it).

    Order: top-left (left shoulder), top-right (right shoulder),
           bottom-right (right hem), bottom-left (left hem)
    """
    h, w = garment_rgba.shape[:2]
    pts = np.float32([
        [w * 0.12, h * 0.05],   # top-left shoulder
        [w * 0.88, h * 0.05],   # top-right shoulder
        [w * 0.88, h * 0.95],   # bottom-right hem
        [w * 0.12, h * 0.95],   # bottom-left hem
    ])
    return pts


def body_target_points(pose_points, hem_ratio=1.6):
    """
    Builds the destination quadrilateral on the person's body for a shirt.

    Top corners use the detected shoulders directly. Bottom corners are
    estimated as a multiple of shoulder width below the shoulders, rather
    than using detected hips directly -- hip landmarks are unreliable when
    seated or occluded (e.g. behind a desk/chair/arms), which was causing
    the shirt to render shrunken and misplaced.

    hem_ratio controls how far down the hem sits, in units of shoulder
    width. ~1.6 approximates a normal t-shirt length; increase for longer
    shirts, decrease for cropped ones.
    """
    left_shoulder = np.array(pose_points["left_shoulder"], dtype=float)
    right_shoulder = np.array(pose_points["right_shoulder"], dtype=float)

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