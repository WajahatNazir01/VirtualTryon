"""
mark_garment_points.py
Interactive tool: click the 6 correspondence points on a garment cutout,
in order, and save them to a JSON file next to the image. This replaces
fragile automatic point detection (widest-row scanning etc.) with
reliable one-time manual marking -- click once, reuse forever for that
garment image.

Usage:
    python mark_garment_points.py ../data/garments/shirt_cutout.png
"""

import sys
import os
import json
import cv2
import numpy as np

POINT_ORDER = [
    "left_shoulder", "right_shoulder", "right_sleeve",
    "right_hem", "left_hem", "left_sleeve",
]


def _composite_on_checkerboard(img):
    """If the image has an alpha channel, composite it onto a checkerboard
    so transparent areas are visible while clicking."""
    if img.shape[2] != 4:
        return img.copy()

    h, w = img.shape[:2]
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    block = 20
    for y in range(0, h, block):
        for x in range(0, w, block):
            color = (230, 230, 230) if ((x // block) + (y // block)) % 2 == 0 else (200, 200, 200)
            checker[y:y + block, x:x + block] = color

    alpha = img[:, :, 3:4].astype(float) / 255.0
    rgb = img[:, :, :3].astype(float)
    vis = (rgb * alpha + checker.astype(float) * (1 - alpha)).astype(np.uint8)
    return vis


def mark_points(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    base = _composite_on_checkerboard(img)
    points = {}

    def redraw():
        disp = base.copy()
        for name, (x, y) in points.items():
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(disp, name, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        idx = len(points)
        if idx < len(POINT_ORDER):
            cv2.putText(disp, f"Click: {POINT_ORDER[idx]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        else:
            cv2.putText(disp, "All 6 marked. Press 's' to save, 'r' to reset, 'q' to quit.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
        cv2.imshow("Mark garment points", disp)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(POINT_ORDER):
            points[POINT_ORDER[len(points)]] = (x, y)
            redraw()

    cv2.namedWindow("Mark garment points")
    cv2.setMouseCallback("Mark garment points", on_click)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
            redraw()
        elif key == ord("s") and len(points) == len(POINT_ORDER):
            out_path = os.path.splitext(image_path)[0] + "_landmarks.json"
            with open(out_path, "w") as f:
                json.dump(points, f, indent=2)
            print(f"Saved landmarks to: {out_path}")
            break
        elif key == ord("q"):
            print("Cancelled -- nothing saved.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mark_garment_points.py <garment_cutout.png>")
        sys.exit(1)

    print("Click these 6 points on the shirt IN ORDER (shown in top-left corner):")
    for i, name in enumerate(POINT_ORDER, 1):
        print(f"  {i}. {name}")
    print("Press 's' to save once all 6 are placed, 'r' to redo, 'q' to cancel.")

    mark_points(sys.argv[1])