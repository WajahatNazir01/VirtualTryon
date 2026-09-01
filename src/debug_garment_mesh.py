"""
debug_garment_mesh.py
Visualizes the garment cutout: draws a grid mesh over it and marks the
4 anchor points used for warping (from warp.garment_anchor_points).

Use this after running garment_segment.py to sanity-check that the
anchor points actually land on the shirt's real shoulder/hem corners
before running the live camera overlay.
"""

import sys
import cv2
import numpy as np

from warp import garment_anchor_points


def draw_mesh(garment_rgba, rows=6, cols=6, garment_type="sleeved"):
    """
    Returns a BGR copy of the garment (composited onto a checkerboard so
    transparency is visible) with a grid mesh and anchor points drawn on top.
    """
    h, w = garment_rgba.shape[:2]

    # Composite onto a light gray checkerboard so transparent areas are visible
    checker = np.zeros((h, w, 3), dtype=np.uint8)
    block = 20
    for y in range(0, h, block):
        for x in range(0, w, block):
            color = (230, 230, 230) if ((x // block) + (y // block)) % 2 == 0 else (200, 200, 200)
            checker[y:y + block, x:x + block] = color

    alpha = garment_rgba[:, :, 3:4].astype(float) / 255.0
    rgb = garment_rgba[:, :, :3].astype(float)
    vis = (rgb * alpha + checker.astype(float) * (1 - alpha)).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR) if garment_rgba.shape[2] == 4 else vis

    # Draw grid lines
    for i in range(1, cols):
        x = int(w * i / cols)
        cv2.line(vis, (x, 0), (x, h), (0, 200, 0), 1)
    for i in range(1, rows):
        y = int(h * i / rows)
        cv2.line(vis, (0, y), (w, y), (0, 200, 0), 1)

    # Draw the bounding box (full image edge, since garment_segment.py
    # should have already cropped tightly to the garment)
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 165, 255), 2)

    # Draw the 4 anchor points used for warping, labeled
    anchors = garment_anchor_points(garment_rgba, garment_type=garment_type)
    labels = ["TL (l.shoulder)", "TR (r.shoulder)", "BR (r.hem)", "BL (l.hem)"]
    for (x, y), label in zip(anchors, labels):
        pt = (int(x), int(y))
        cv2.circle(vis, pt, 8, (0, 0, 255), -1)
        cv2.putText(vis, label, (pt[0] + 10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return vis


if __name__ == "__main__":
    # Usage: python debug_garment_mesh.py <garment_rgba_png> [sleeved|sleeveless]
    if len(sys.argv) not in (2, 3):
        print("Usage: python debug_garment_mesh.py <garment_rgba_png> [sleeved|sleeveless]")
        sys.exit(1)

    garment_type = sys.argv[2] if len(sys.argv) == 3 else "sleeved"

    garment = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    if garment is None or garment.shape[2] != 4:
        print("Could not load a valid RGBA garment image (run garment_segment.py first).")
        sys.exit(1)

    vis = draw_mesh(garment, garment_type=garment_type)
    cv2.imshow("Garment Mesh Debug (press any key to close)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()