"""
physics.py
Lightweight damped-pendulum simulation for jewelry that hangs and swings
(earrings, necklace pendants) rather than staying rigidly fixed to its
anchor point. Rigid jewelry (rings, studs) doesn't need this -- just
attach directly to the anchor with no physics object.
"""

import math
import numpy as np


class DanglingPhysics:
    """
    A pendulum hanging from a moving anchor point. As the anchor (ear,
    neck) moves -- especially suddenly, e.g. turning your head -- the
    bob swings and settles instead of teleporting instantly, which is
    what makes it read as a real hanging object instead of a sticker.

    length: rest length of the chain/drop, in pixels (scale this by the
            detected body/hand size so it looks consistent at any distance
            from the camera).
    gravity: pulls the bob back toward hanging straight down.
    stiffness: how strongly sideways anchor movement pushes the bob.
    damping: how quickly the swinging settles (higher = less bouncy).
    """

    def __init__(self, anchor, length, gravity=1400.0, stiffness=25.0, damping=8.0):
        self.anchor = np.array(anchor, dtype=float)
        self.length = length
        self.angle = 0.0             # radians from straight down
        self.angular_velocity = 0.0
        self.gravity = gravity
        self.stiffness = stiffness
        self.damping = damping
        self._prev_anchor = np.array(anchor, dtype=float)

    def update(self, new_anchor, dt):
        """Advance the simulation by dt seconds given the anchor's new
        position. Returns (bob_pixel_pos, angle_radians)."""
        new_anchor = np.array(new_anchor, dtype=float)
        anchor_velocity = (new_anchor - self._prev_anchor) / max(dt, 1e-4)
        self._prev_anchor = new_anchor
        self.anchor = new_anchor

        # Pendulum restoring force toward hanging straight down (angle=0),
        # damping to settle, and a perturbation from horizontal anchor
        # motion (e.g. turning your head swings the earring sideways).
        angular_accel = -(self.gravity / self.length) * math.sin(self.angle)
        angular_accel -= self.damping * self.angular_velocity
        angular_accel -= (anchor_velocity[0] / self.length) * (self.stiffness / 100.0)

        self.angular_velocity += angular_accel * dt
        self.angle += self.angular_velocity * dt

        # Clamp so it can't swing into nonsense angles from a bad frame/jitter
        self.angle = max(-1.2, min(1.2, self.angle))

        bob = self.anchor + self.length * np.array([math.sin(self.angle), math.cos(self.angle)])
        return bob, self.angle

    def set_length(self, length):
        """Call this each frame with a body-scale-derived length so the
        jewelry stays proportional as the person moves closer/farther."""
        self.length = length