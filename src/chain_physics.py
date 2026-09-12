"""
chain_physics.py
Verlet-integration rope simulation for a necklace/chain: instead of one
rigid image, the chain is simulated as a series of connected points that
sag naturally under gravity between two fixed anchors (left/right
collarbone), and resettle when the anchors move (you turning, walking).

This is the standard "rope/cloth simulation" technique used in games --
much more physically honest than warping a single static image.
"""

import numpy as np


class ChainPhysics:
    def __init__(self, anchor_a, anchor_b, num_segments=12, sag_factor=1.15,
                 gravity=900.0, iterations=6):
        """
        anchor_a, anchor_b: the two fixed points the chain hangs between
            (e.g. left/right collarbone).
        num_segments: how many straight sub-segments approximate the curve
            -- more segments = smoother curve, more computation.
        sag_factor: total chain length as a multiple of the straight-line
            distance between anchors (>1 means it hangs loose rather than
            pulled taut).
        """
        self.num_segments = num_segments
        self.gravity = gravity
        self.iterations = iterations

        anchor_a = np.array(anchor_a, dtype=float)
        anchor_b = np.array(anchor_b, dtype=float)
        straight_dist = np.linalg.norm(anchor_b - anchor_a)
        self.total_length = straight_dist * sag_factor
        self.segment_length = self.total_length / num_segments

        # Initialize points along a straight line -- physics will relax
        # this into a natural sag within the first few frames.
        self.points = [anchor_a + (anchor_b - anchor_a) * (i / num_segments)
                       for i in range(num_segments + 1)]
        self.points = [np.array(p, dtype=float) for p in self.points]
        self.prev_points = [p.copy() for p in self.points]

    def update(self, anchor_a, anchor_b, dt):
        """Advance the simulation by dt seconds. Returns the list of
        (x, y) points forming the current chain curve."""
        anchor_a = np.array(anchor_a, dtype=float)
        anchor_b = np.array(anchor_b, dtype=float)

        # Re-derive segment length from current anchor distance so the
        # chain rescales smoothly as you move closer/further from camera.
        straight_dist = np.linalg.norm(anchor_b - anchor_a)
        self.total_length = max(straight_dist * 1.15, straight_dist + 1e-3)
        self.segment_length = self.total_length / self.num_segments

        self.points[0] = anchor_a
        self.points[-1] = anchor_b

        gravity_step = np.array([0.0, self.gravity]) * dt * dt

        # Verlet integration for free (non-anchor) points.
        for i in range(1, self.num_segments):
            velocity = self.points[i] - self.prev_points[i]
            self.prev_points[i] = self.points[i].copy()
            self.points[i] = self.points[i] + velocity + gravity_step

        # Distance-constraint relaxation, several passes for stability.
        for _ in range(self.iterations):
            self.points[0] = anchor_a
            self.points[-1] = anchor_b
            for i in range(self.num_segments):
                p1, p2 = self.points[i], self.points[i + 1]
                delta = p2 - p1
                dist = np.linalg.norm(delta)
                if dist < 1e-6:
                    continue
                diff = (dist - self.segment_length) / dist
                pinned1 = (i == 0)
                pinned2 = (i + 1 == self.num_segments)
                if pinned1 and pinned2:
                    continue
                elif pinned1:
                    self.points[i + 1] = p2 - diff * delta
                elif pinned2:
                    self.points[i] = p1 + diff * delta
                else:
                    self.points[i] = p1 + 0.5 * diff * delta
                    self.points[i + 1] = p2 - 0.5 * diff * delta

        return [tuple(p) for p in self.points]