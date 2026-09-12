"""
jewelry_landmarks.py
Anchor point detection for jewelry categories:
  - earrings  -> ear position (from Pose Landmarker)
  - necklace  -> neck/collarbone midpoint (from Pose Landmarker)
  - bracelet  -> wrist (from Hand Landmarker)
  - ring      -> finger joint (from Hand Landmarker)

Only load the detector a given jewelry type actually needs -- don't run
face/hand/pose detection all at once if you only need one of them.
"""

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# --- Pose-based anchors (ears, neck) ---------------------------------------

POSE_LANDMARK_IDS = {
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
}


def get_ear_and_neck_anchors(frame, pose):
    """Returns {'left_ear':(x,y), 'right_ear':(x,y), 'neck':(x,y)} in pixel
    coords, or None if no person detected. 'neck' is estimated as the
    shoulder midpoint, offset upward toward the chin."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None

    lm = result.pose_landmarks.landmark
    pts = {name: (lm[idx].x * w, lm[idx].y * h) for name, idx in POSE_LANDMARK_IDS.items()}

    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    shoulder_width = ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5
    # Neck sits above the shoulder midpoint -- offset scaled by shoulder
    # width so it works at any distance from the camera.
    neck = (shoulder_mid[0], shoulder_mid[1] - shoulder_width * 0.15)

    return {
        "left_ear": pts["left_ear"],
        "right_ear": pts["right_ear"],
        "neck": neck,
        "shoulder_width": shoulder_width,  # useful for scaling jewelry size
    }


def get_neck_span_anchors(anchors, span_fraction=0.35):
    """
    Two points near the base of the neck (approximating the collarbones)
    for a necklace/chain to span between, instead of hanging from one
    single point -- this is what lets it actually drape in a curve.
    """
    neck = anchors["neck"]
    shoulder_width = anchors["shoulder_width"]
    half_span = shoulder_width * span_fraction / 2
    left = (neck[0] - half_span, neck[1])
    right = (neck[0] + half_span, neck[1])
    return left, right


# --- Hand-based anchors (wrist, finger joints) ------------------------------

HAND_LANDMARK_IDS = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_mcp": 5,
    "ring_mcp": 13,
    "ring_pip": 14,
    "pinky_mcp": 17,
}


class HandLandmarkDetector:
    """Wraps MediaPipe's classic Hands solution (simpler to set up than the
    newer Tasks API and sufficient for anchor points, not full 3D mesh)."""

    def __init__(self, max_hands=1, detection_confidence=0.6, tracking_confidence=0.6):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def get_anchors(self, frame):
        """Returns a list of per-hand anchor dicts (pixel coords), or []
        if no hand detected. Each dict has wrist, ring_mcp, ring_pip, etc.,
        plus 'hand_scale' (wrist-to-middle-finger-mcp distance) for sizing."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        hands_out = []
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            pts = {name: (lm[idx].x * w, lm[idx].y * h) for name, idx in HAND_LANDMARK_IDS.items()}
            scale = (
                (pts["wrist"][0] - pts["index_mcp"][0]) ** 2
                + (pts["wrist"][1] - pts["index_mcp"][1]) ** 2
            ) ** 0.5
            pts["hand_scale"] = scale
            hands_out.append(pts)
        return hands_out

    def close(self):
        self._hands.close()