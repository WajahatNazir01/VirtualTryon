"""
pose_detect.py
Wraps MediaPipe Pose to return pixel-space coordinates of the body
keypoints we care about for clothing overlay (shoulders, hips, ears, nose).
"""

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# Landmark indices we care about (MediaPipe's 33-point pose model)
LANDMARK_IDS = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
}


def get_pose_landmarks(frame, pose):
    """
    Runs pose detection on a single BGR frame.
    Returns a dict {name: (x_px, y_px)} for the keypoints in LANDMARK_IDS,
    or None if no person is detected.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    points = {}
    for name, idx in LANDMARK_IDS.items():
        lm = landmarks[idx]
        points[name] = (int(lm.x * w), int(lm.y * h))

    return points


if __name__ == "__main__":
    # Quick standalone test: draws detected keypoints live on your webcam feed.
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            points = get_pose_landmarks(frame, pose)

            if points:
                for name, (x, y) in points.items():
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, name, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Pose Detection Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()