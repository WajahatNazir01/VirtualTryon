"""
debug_body_mesh.py
Visualizes, live on your webcam feed, the body landmarks MediaPipe detects
and the target quad (mesh) that the garment will be warped onto -- computed
by warp.body_target_points(). Use this to verify body-side alignment before
worrying about the garment cutout at all.
"""

import cv2
import mediapipe as mp
import numpy as np

from pose_detect import get_pose_landmarks
from warp import body_target_points

mp_pose = mp.solutions.pose


def draw_landmarks(frame, points):
    for name, (x, y) in points.items():
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(frame, name, (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_target_mesh(frame, dst_pts, rows=4, cols=4):
    """
    dst_pts order: TL (right shoulder), TR (left shoulder),
                   BR (left hem), BL (right hem)  -- see body_target_points
    Draws the quad outline plus an interpolated grid inside it, and labels
    each corner in red.
    """
    tl, tr, br, bl = dst_pts

    # Quad outline
    pts = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)

    # Interpolated grid lines (bilinear interpolation across the quad)
    for i in range(1, rows):
        t = i / rows
        left = tl + (bl - tl) * t
        right = tr + (br - tr) * t
        cv2.line(frame, tuple(left.astype(int)), tuple(right.astype(int)), (0, 200, 0), 1)
    for j in range(1, cols):
        t = j / cols
        top = tl + (tr - tl) * t
        bottom = bl + (br - bl) * t
        cv2.line(frame, tuple(top.astype(int)), tuple(bottom.astype(int)), (0, 200, 0), 1)

    labels = ["TL", "TR", "BR", "BL"]
    for (x, y), label in zip([tl, tr, br, bl], labels):
        pt = (int(x), int(y))
        cv2.circle(frame, pt, 8, (0, 0, 255), -1)
        cv2.putText(frame, label, (pt[0] + 10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            points = get_pose_landmarks(frame, pose)

            if points:
                draw_landmarks(frame, points)
                dst_pts = body_target_points(points)
                draw_target_mesh(frame, dst_pts)

            cv2.imshow("Body Mesh Debug (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()