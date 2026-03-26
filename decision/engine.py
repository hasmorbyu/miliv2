from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import math
import os

from tracking.tracker import Track

INTERCEPT_RADIUS = 45   
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "events.log")


class Mode(str, Enum):
    PATROL = "PATROL"
    DETECT = "DETECT"
    TRACK = "TRACK"
    INTERCEPT = "INTERCEPT"


@dataclass
class Decision:
    mode: Mode
    target_id: Optional[int]
    target_confidence: float
    target_predicted_pos: Optional[Tuple[float, float]]
    status: str


class DecisionEngine:
    def __init__(self):
        self.mode = Mode.PATROL
        self.active_target_id: Optional[int] = None
        self.neutralized: set = set()
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


    def update(self, tracks: List[Track], drone_pos) -> Decision:

        live_tracks = [t for t in tracks if t.track_id not in self.neutralized]

        if not live_tracks:
            self.mode = Mode.PATROL
            self.active_target_id = None
            return Decision(Mode.PATROL, None, 0.0, None, "Scanning area")

        self.mode = Mode.DETECT


        target = self._select_target(live_tracks, drone_pos)
        self.active_target_id = target.track_id
        predicted = target.predict_next()

        dist = math.hypot(predicted[0] - drone_pos[0], predicted[1] - drone_pos[1])

        if dist < INTERCEPT_RADIUS:
            self.mode = Mode.INTERCEPT
            self.neutralized.add(target.track_id)
            self.active_target_id = None
            self._log_event(f"NEUTRALIZED track_id={target.track_id} class={target.class_label}")
            return Decision(Mode.INTERCEPT, target.track_id, target.confidence,
                            predicted, f"Target {target.track_id} neutralized!")

        self.mode = Mode.TRACK
        return Decision(Mode.TRACK, target.track_id, target.confidence,
                        predicted, f"Tracking ID {target.track_id}")


    def _select_target(self, tracks: List[Track], drone_pos) -> Track:

        if self.active_target_id is not None:
            for t in tracks:
                if t.track_id == self.active_target_id:
                    return t


        best, best_score = None, -1.0
        for t in tracks:
            dx = t.centroid[0] - drone_pos[0]
            dy = t.centroid[1] - drone_pos[1]
            dist = math.hypot(dx, dy) + 1e-6
            score = 0.7 * (1.0 / dist) * 500 + 0.3 * t.confidence
            if score > best_score:
                best_score = score
                best = t
        return best


    def _log_event(self, msg: str):
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")
