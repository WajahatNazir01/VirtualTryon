"""
main.py
Ties everything together: loads a pre-segmented garment PNG (RGBA),
opens the webcam, detects pose each frame, warps the garment onto
the detected torso, and displays the live composited result.

Run garment_segment.py once beforehand to produce the transparent
garment PNG from your pasted clothing image.
"""

import sys
import cv2
import mediapipe as mp

from pose_detect import get_pose_landmarks
from warp import garment_anchor_points, body_target_points, warp_garment, composite

mp_pose = mp.solutions.pose


def main(garment_path: str, garment_type: str = "sleeved"):
    garment_rgba = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    if garment_rgba is None:
        print(f"Could not load garment image at {garment_path}")
        sys.exit(1)
    if garment_rgba.shape[2] != 4:
        print("Garment image has no alpha channel — run garment_segment.py on it first.")
        sys.exit(1)

    src_pts = garment_anchor_points(garment_rgba, garment_type=garment_type)

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            points = get_pose_landmarks(frame, pose)

            if points:
                dst_pts = body_target_points(points)
                warped = warp_garment(garment_rgba, src_pts, dst_pts, frame.shape)
                frame = composite(frame, warped)

            cv2.imshow("Virtual Try-On MVP", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Usage: python main.py <garment_rgba_png> [sleeved|sleeveless]
    if len(sys.argv) not in (2, 3):
        print("Usage: python main.py <garment_rgba_png> [sleeved|sleeveless]")
        sys.exit(1)

    garment_type = sys.argv[2] if len(sys.argv) == 3 else "sleeved"
    main(sys.argv[1], garment_type)