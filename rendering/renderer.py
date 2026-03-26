from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Tuple
import math
import cv2
import numpy as np

from tracking.tracker import Track
from decision.engine import Decision, Mode
from assets.sprites import overlay_sprite


HUD_BG = (20, 20, 20)
GREEN = (50, 230, 50)
YELLOW = (0, 220, 240)
RED = (30, 30, 255)
WHITE = (240, 240, 240)
CYAN = (230, 220, 40)


TRACK_PALETTE = [
    (255, 100, 50), (50, 255, 180), (180, 50, 255),
    (50, 200, 255), (255, 200, 50), (200, 50, 200),
    (50, 255, 50),  (255, 50, 100),
]


class Renderer:
    WINDOW = "Drone Simulation – Live"

    def __init__(self, width: int, height: int,
                 drone_sprite, obstacle_sprite,
                 show_preview: bool = True):
        self.w = width
        self.h = height
        self.drone_sprite = drone_sprite
        self.obstacle_sprite = obstacle_sprite
        self.show_preview = show_preview
        self._intercept_flash = 0

        if show_preview:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW, min(width, 1280), min(height, 720))


    def render(self, frame: np.ndarray,
               tracks: List[Track],
               decision: Decision,
               drone_pos,
               obstacles,
               neutralized_count: int) -> Tuple[np.ndarray, bool]:
        """
        Draw all overlays onto `frame` (modified in place).
        Returns (frame, should_stop).
        """
        out = frame.copy()


        if decision.mode == Mode.INTERCEPT:
            self._intercept_flash = 8
        if self._intercept_flash > 0:
            self._draw_vignette(out)
            self._intercept_flash -= 1


        for obs in obstacles:
            overlay_sprite(out, self.obstacle_sprite,
                           int(obs.position[0]), int(obs.position[1]))
            cv2.circle(out, (int(obs.position[0]), int(obs.position[1])),
                       obs.radius, (0, 80, 200), 1, cv2.LINE_AA)


        self._draw_tracks(out, tracks, decision.target_id)


        dx, dy = int(drone_pos[0]), int(drone_pos[1])
        overlay_sprite(out, self.drone_sprite, dx, dy)

        cv2.circle(out, (dx, dy), 4, WHITE, -1, cv2.LINE_AA)


        if decision.target_predicted_pos is not None:
            tx, ty = int(decision.target_predicted_pos[0]), int(decision.target_predicted_pos[1])
            self._draw_crosshair(out, tx, ty)


            cv2.line(out, (dx, dy), (tx, ty), (0, 180, 255, 100), 1, cv2.LINE_AA)


        self._draw_hud(out, decision, neutralized_count)


        should_stop = False
        if self.show_preview:
            cv2.imshow(self.WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                should_stop = True

        return out, should_stop


    def close(self):
        if self.show_preview:
            cv2.destroyAllWindows()


    def _draw_tracks(self, frame, tracks, target_id):
        for track in tracks:
            color = TRACK_PALETTE[track.track_id % len(TRACK_PALETTE)]
            is_target = (track.track_id == target_id)


            hist = list(track.history)
            for i in range(1, len(hist)):
                alpha = i / len(hist)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, hist[i - 1], hist[i], c, 2 if is_target else 1, cv2.LINE_AA)


            x1, y1, x2, y2 = track.bbox
            box_color = YELLOW if is_target else GREEN
            thickness = 2 if is_target else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)


            label = f"ID:{track.track_id} {track.class_label} {track.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


    def _draw_crosshair(self, frame, cx, cy, size=20):
        col = RED
        cv2.line(frame, (cx - size, cy), (cx + size, cy), col, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), col, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), size, col, 1, cv2.LINE_AA)


    def _draw_hud(self, frame, decision: Decision, neutralized_count: int):
        lines = [
            f"MODE:   {decision.mode.value}",
            f"TARGET: {decision.target_id if decision.target_id is not None else '---'}",
            f"CONF:   {decision.target_confidence:.2f}",
            f"STATUS: {decision.status}",
            f"NEUTRALIZED: {neutralized_count}",
        ]
        pad = 8
        lh = 22
        panel_h = len(lines) * lh + pad * 2
        panel_w = 260
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), HUD_BG, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), CYAN, 1, cv2.LINE_AA)

        mode_colors = {Mode.PATROL: GREEN, Mode.DETECT: YELLOW,
                       Mode.TRACK: YELLOW, Mode.INTERCEPT: RED}
        for i, line in enumerate(lines):
            color = mode_colors.get(decision.mode, WHITE) if i == 0 else WHITE
            if i == 4:
                color = RED if neutralized_count > 0 else WHITE
            cv2.putText(frame, line, (14, 8 + pad + (i + 1) * lh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


    def _draw_vignette(self, frame):
        """Red vignette flash on interception."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360,
                    (0, 0, 200), int(min(w, h) * 0.25))
        cv2.addWeighted(frame, 0.75, mask, 0.45, 0, frame)
