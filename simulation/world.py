from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

RNG = np.random.default_rng(7)


@dataclass
class Obstacle:
    obs_id: int
    position: np.ndarray
    velocity: np.ndarray          
    radius: int = 28


@dataclass
class WorldState:
    width: int
    height: int
    drone_pos: np.ndarray
    drone_vel: np.ndarray
    obstacles: List[Obstacle] = field(default_factory=list)
    neutralized_count: int = 0


    @classmethod
    def create(cls, width: int, height: int, num_obstacles: int = 6) -> "WorldState":
        drone_pos = np.array([width // 2, height // 2], dtype=float)
        drone_vel = np.array([0.0, 0.0])
        obstacles = []
        for i in range(num_obstacles):

            while True:
                pos = RNG.uniform([40, 40], [width - 40, height - 40])
                if np.linalg.norm(pos - drone_pos) > 160:
                    break
            vel = RNG.uniform(-0.6, 0.6, 2)
            obstacles.append(Obstacle(obs_id=i, position=pos, velocity=vel))
        return cls(width=width, height=height, drone_pos=drone_pos,
                   drone_vel=drone_vel, obstacles=obstacles)


    def update(self, dt: float = 1.0):
        self.drone_pos += self.drone_vel * dt


        self.drone_pos[0] = np.clip(self.drone_pos[0], 0, self.width - 1)
        self.drone_pos[1] = np.clip(self.drone_pos[1], 0, self.height - 1)


        for obs in self.obstacles:
            obs.position += obs.velocity
            for dim, limit in enumerate([self.width, self.height]):
                if obs.position[dim] < obs.radius:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] > limit - obs.radius:
                    obs.position[dim] = limit - obs.radius
                    obs.velocity[dim] *= -1
