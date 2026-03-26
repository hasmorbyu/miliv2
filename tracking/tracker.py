from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from detection.yolo_detector import Detection

MAX_DISTANCE = 120      
MAX_LOST_FRAMES = 15
HISTORY_LEN = 60


@dataclass
class Track:
    track_id: int
    class_label: str
    confidence: float
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    velocity: Tuple[float, float] = (0.0, 0.0)
    lost_frames: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    def update(self, detection: Detection):
        prev = self.centroid
        self.centroid = detection.centroid
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_label = detection.class_label
        self.velocity = (
            detection.centroid[0] - prev[0],
            detection.centroid[1] - prev[1],
        )
        self.history.append(detection.centroid)
        self.lost_frames = 0

    def predict_next(self) -> Tuple[float, float]:
        return (
            self.centroid[0] + self.velocity[0],
            self.centroid[1] + self.velocity[1],
        )


class Tracker:
    def __init__(self):
        self._next_id = 0
        self.tracks: Dict[int, Track] = {}


    def update(self, detections: List[Detection]) -> List[Track]:
        """Match detections to existing tracks; return active track list."""
        if not self.tracks:
            for d in detections:
                self._spawn(d)
        else:
            self._match_and_update(detections)


        dead = [tid for tid, t in self.tracks.items() if t.lost_frames > MAX_LOST_FRAMES]
        for tid in dead:
            del self.tracks[tid]

        return list(self.tracks.values())


    def _spawn(self, d: Detection) -> Track:
        t = Track(
            track_id=self._next_id,
            class_label=d.class_label,
            confidence=d.confidence,
            centroid=d.centroid,
            bbox=d.bbox,
        )
        t.history.append(d.centroid)
        self.tracks[self._next_id] = t
        self._next_id += 1
        return t

    def _match_and_update(self, detections: List[Detection]):
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids], dtype=float)
        det_centroids = np.array([d.centroid for d in detections], dtype=float) if detections else np.empty((0, 2))

        matched_tracks = set()
        matched_dets = set()

        if len(detections) > 0:

            diff = track_centroids[:, None, :] - det_centroids[None, :, :]
            dists = np.linalg.norm(diff, axis=2)

            while True:
                if dists.size == 0:
                    break
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                ti, di = idx
                if dists[ti, di] > MAX_DISTANCE:
                    break
                tid = track_ids[ti]
                if ti not in matched_tracks and di not in matched_dets:
                    self.tracks[tid].update(detections[di])
                    matched_tracks.add(ti)
                    matched_dets.add(di)
                dists[ti, :] = np.inf
                dists[:, di] = np.inf

        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid].lost_frames += 1

        for di, d in enumerate(detections):
            if di not in matched_dets:
                self._spawn(d)
