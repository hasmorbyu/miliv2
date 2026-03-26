
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from simulation.world import WorldState

DRONE_SPEED = 5.0
INTERCEPT_SPEED = 8.0
OBSTACLE_BUFFER = 20
REPULSION_STRENGTH = 3.0


def compute_velocity(
    world: WorldState,
    target_pos: Optional[Tuple[float, float]],
    mode_is_intercept: bool = False,
) -> np.ndarray:
    drone = world.drone_pos


    if target_pos is not None:
        to_target = np.array(target_pos, dtype=float) - drone
        dist = np.linalg.norm(to_target)
        if dist > 1.0:
            attract = (to_target / dist)
        else:
            attract = np.zeros(2)
    else:

        angle = np.arctan2(world.drone_vel[1], world.drone_vel[0]) + 0.04
        attract = np.array([np.cos(angle), np.sin(angle)])


    repulse = np.zeros(2)
    for obs in world.obstacles:
        to_obs = obs.position - drone
        dist_obs = np.linalg.norm(to_obs)
        threshold = obs.radius + OBSTACLE_BUFFER + 18
        if 0 < dist_obs < threshold:

            strength = REPULSION_STRENGTH * (1.0 - dist_obs / threshold)
            repulse -= (to_obs / dist_obs) * strength

    combined = attract + repulse
    norm = np.linalg.norm(combined)
    if norm > 1e-6:
        combined /= norm

    speed = INTERCEPT_SPEED if mode_is_intercept else DRONE_SPEED
    return combined * speed
