"""
main.py
End-to-end pipeline: takes a raw garment photo, segments it, classifies
its category (CLIP), then runs the live webcam overlay using the
landmark set and anchor logic that matches that category automatically.

Usage:
    python main.py ../data/garments/shirt.jpg
    python main.py ../data/garments/shirt.jpg --category upper_body_sleeved   (skip classifier)
"""

import sys
import os
import json
import argparse
import cv2
import mediapipe as mp

from garment_segment import remove_background
from category_classifier import classify_garment, CATEGORIES
from pose_detect import get_pose_landmarks
from warp import (
    garment_anchor_points, body_target_points, warp_garment, composite,
    garment_landmarks_upper_body, body_landmarks_upper_body, REQUIRED_LANDMARKS,
)
from mesh_warp import warp_garment_mesh

mp_pose = mp.solutions.pose

MESH_CATEGORIES = {"upper_body_sleeved", "upper_body_sleeveless"}


def prepare_garment(photo_path: str):
    """Segments the background and returns the path to the cutout PNG."""
    base, _ = os.path.splitext(photo_path)
    cutout_path = base + "_cutout.png"
    remove_background(photo_path, cutout_path)
    return cutout_path


def main(photo_path: str, category_override: str = None):
    cutout_path = prepare_garment(photo_path)
    print(f"Garment cutout saved to: {cutout_path}")

    if category_override:
        category = category_override
        print(f"Using specified category: {category}")
    else:
        print("Classifying garment category...")
        category = classify_garment(photo_path)
        print(f"Detected category: {category}")

    garment_rgba = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
    if garment_rgba is None or garment_rgba.shape[2] != 4:
        print("Failed to load a valid RGBA garment cutout.")
        sys.exit(1)

    use_mesh = category in MESH_CATEGORIES
    sleeveless = category == "upper_body_sleeveless"

    if use_mesh:
        landmarks_path = os.path.splitext(cutout_path)[0] + "_landmarks.json"
        if os.path.exists(landmarks_path):
            with open(landmarks_path) as f:
                raw = json.load(f)
            garment_landmarks = {k: tuple(v) for k, v in raw.items()}
            print(f"Using manually-marked landmarks from: {landmarks_path}")
        else:
            print(f"No manual landmarks found at {landmarks_path} -- falling back to auto-detection.")
            print(f"For more reliable results, run: python mark_garment_points.py {cutout_path}")
            garment_landmarks = garment_landmarks_upper_body(garment_rgba, sleeveless=sleeveless)
        print(f"Garment landmarks: {garment_landmarks}")
        required = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow"}
    else:
        src_pts = garment_anchor_points(garment_rgba, category=category)
        required = set(REQUIRED_LANDMARKS.get(category, []))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        sys.exit(1)

    printed_body_landmarks = False
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            points = get_pose_landmarks(frame, pose)

            if points and required.issubset(points.keys()):
                if use_mesh:
                    body_landmarks = body_landmarks_upper_body(points)
                    if not printed_body_landmarks:
                        print(f"Body landmarks: {body_landmarks}")
                        printed_body_landmarks = True
                    warped = warp_garment_mesh(garment_rgba, garment_landmarks, body_landmarks, frame.shape)
                else:
                    dst_pts = body_target_points(points, category=category)
                    warped = warp_garment(garment_rgba, src_pts, dst_pts, frame.shape)
                frame = composite(frame, warped)

            cv2.imshow(f"Virtual Try-On MVP [{category}]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the original garment photo (not the cutout)")
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES.keys()),
        default=None,
        help="Skip the classifier and force a category",
    )
    args = parser.parse_args()
    main(args.photo, args.category)