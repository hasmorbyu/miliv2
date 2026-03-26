"""
assets/sprites.py
Generates drone and obstacle sprites as BGRA NumPy arrays.
"""
import numpy as np
import cv2


def make_drone_sprite(size: int = 48) -> np.ndarray:
    """Blue diamond-shaped drone sprite with glow."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    pts = np.array([
        [cx, 2],
        [size - 2, cy],
        [cx, size - 2],
        [2, cy],
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], (255, 180, 80, 80))
    pts_inner = (pts * 0.75 + np.array([cx * 0.125, cy * 0.125])).astype(np.int32)

    pts_inner = ((pts - np.array([cx, cy])) * 0.72 + np.array([cx, cy])).astype(np.int32)
    cv2.fillPoly(img, [pts_inner], (255, 200, 50, 220))

    pts_core = ((pts - np.array([cx, cy])) * 0.45 + np.array([cx, cy])).astype(np.int32)
    cv2.fillPoly(img, [pts_core], (255, 255, 255, 255))
    return img


def make_obstacle_sprite(size: int = 36) -> np.ndarray:
    """Orange hexagon obstacle sprite."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    angles = [np.pi / 2 + i * np.pi / 3 for i in range(6)]
    r = size // 2 - 2
    pts = np.array([[int(cx + r * np.cos(a)), int(cy + r * np.sin(a))] for a in angles], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 120, 255, 200))
    pts_inner = ((pts - np.array([cx, cy])) * 0.6 + np.array([cx, cy])).astype(np.int32)
    cv2.fillPoly(img, [pts_inner], (0, 80, 200, 240))
    return img


def overlay_sprite(frame: np.ndarray, sprite_bgra: np.ndarray, cx: int, cy: int) -> None:
    """Alpha-composite sprite centred at (cx, cy) onto BGR frame in-place."""
    h, w = sprite_bgra.shape[:2]
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    fx0 = max(x0, 0); fy0 = max(y0, 0)
    fx1 = min(x1, frame.shape[1]); fy1 = min(y1, frame.shape[0])
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sx0, sy0 = fx0 - x0, fy0 - y0
    sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)
    roi = frame[fy0:fy1, fx0:fx1]
    sprite_crop = sprite_bgra[sy0:sy1, sx0:sx1]
    alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0
    bgr = sprite_crop[:, :, :3].astype(np.float32)
    roi[:] = (bgr * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
