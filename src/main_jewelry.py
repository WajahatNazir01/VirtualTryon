"""
main_jewelry.py
Demo: live earring try-on (dangling, physics-driven swing) and/or ring
try-on (rigid, no physics needed) using the webcam.

Usage:
    python main_jewelry.py --earring ../data/jewelry/earring_cutout.png
    python main_jewelry.py --ring ../data/jewelry/ring_cutout.png
    python main_jewelry.py --earring ../data/jewelry/earring_cutout.png --ring ../data/jewelry/ring_cutout.png
"""

import sys
import time
import argparse
import cv2
import mediapipe as mp

from jewelry_landmarks import get_ear_and_neck_anchors, get_neck_span_anchors, HandLandmarkDetector
from physics import DanglingPhysics
from chain_physics import ChainPhysics
from chain_render import sample_chain_appearance, draw_chain, draw_textured_chain
from jewelry_overlay import composite_at

mp_pose = mp.solutions.pose


def load_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        print(f"Could not load a valid RGBA image at {path} (run garment_segment.py on it first).")
        sys.exit(1)
    return img


def main(earring_path=None, ring_path=None, necklace_path=None, bracelet_path=None):
    earring_img = load_rgba(earring_path) if earring_path else None
    ring_img = load_rgba(ring_path) if ring_path else None
    necklace_img = load_rgba(necklace_path) if necklace_path else None
    bracelet_img = load_rgba(bracelet_path) if bracelet_path else None

    hand_detector = HandLandmarkDetector(max_hands=2) if (ring_img is not None or bracelet_img is not None) else None

    # One pendulum per ear / one for the necklace pendant -- initialized
    # lazily once we see the first anchor.
    left_pendulum = None
    right_pendulum = None
    chain_color, chain_thickness = sample_chain_appearance(necklace_img) if necklace_img is not None else (None, None)
    chain_physics = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        sys.exit(1)

    last_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - last_time
            last_time = now

            if earring_img is not None or necklace_img is not None:
                anchors = get_ear_and_neck_anchors(frame, pose)
            else:
                anchors = None

            if earring_img is not None and anchors:
                shoulder_width = anchors["shoulder_width"]
                # Earring drop length + jewelry width both scale with
                # shoulder width, so it stays proportional at any distance.
                drop_length = shoulder_width * 0.12
                item_scale = (shoulder_width * 0.05) / earring_img.shape[1]

                for side, anchor in (("left", anchors["left_ear"]), ("right", anchors["right_ear"])):
                    pendulum = left_pendulum if side == "left" else right_pendulum
                    if pendulum is None:
                        pendulum = DanglingPhysics(anchor, drop_length)
                        if side == "left":
                            left_pendulum = pendulum
                        else:
                            right_pendulum = pendulum

                    pendulum.set_length(drop_length)
                    bob_pos, angle_rad = pendulum.update(anchor, dt)
                    angle_deg = -angle_rad * (180 / 3.14159)  # sign matches cv2 rotation convention
                    frame = composite_at(frame, earring_img, anchor, item_scale, angle_deg)

            if necklace_img is not None and anchors:
                left_anchor, right_anchor = get_neck_span_anchors(anchors)
                if chain_physics is None:
                    chain_physics = ChainPhysics(left_anchor, right_anchor)
                curve_points = chain_physics.update(left_anchor, right_anchor, dt)
                frame = draw_textured_chain(frame, curve_points, necklace_img)

            if (ring_img is not None or bracelet_img is not None) and hand_detector is not None:
                for hand in hand_detector.get_anchors(frame):
                    if ring_img is not None:
                        ring_scale = (hand["hand_scale"] * 0.5) / ring_img.shape[1]
                        frame = composite_at(frame, ring_img, hand["ring_pip"], ring_scale, angle_degrees=0.0)
                    if bracelet_img is not None:
                        bracelet_scale = (hand["hand_scale"] * 1.6) / bracelet_img.shape[1]
                        frame = composite_at(frame, bracelet_img, hand["wrist"], bracelet_scale, angle_degrees=0.0)

            cv2.imshow("Jewelry Try-On Demo (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if hand_detector:
        hand_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--earring", help="Path to earring cutout PNG (RGBA)")
    parser.add_argument("--ring", help="Path to ring cutout PNG (RGBA)")
    parser.add_argument("--necklace", help="Path to necklace/chain cutout PNG (RGBA)")
    parser.add_argument("--bracelet", help="Path to bracelet cutout PNG (RGBA)")
    args = parser.parse_args()

    if not any([args.earring, args.ring, args.necklace, args.bracelet]):
        print("Provide at least one of --earring, --ring, --necklace, --bracelet")
        sys.exit(1)
        

    main(args.earring, args.ring, args.necklace, args.bracelet)