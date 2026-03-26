"""
assets/generate_background.py
Generates a synthetic 30-second 720p background video if no real video exists.
Called automatically by main.py when assets/background.mp4 is absent.
"""
import os
import cv2
import numpy as np

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "background.mp4")
WIDTH, HEIGHT = 1280, 720
FPS = 25
DURATION_SEC = 40


def generate():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))
    rng = np.random.default_rng(42)


    num_blobs = 6
    blob_pos = rng.uniform([60, 60], [WIDTH - 60, HEIGHT - 60], (num_blobs, 2))
    blob_vel = rng.uniform(-1.5, 1.5, (num_blobs, 2))
    blob_colors = [tuple(int(c) for c in rng.integers(60, 200, 3)) for _ in range(num_blobs)]
    blob_sizes = rng.integers(25, 60, num_blobs)

    total_frames = FPS * DURATION_SEC
    for f in range(total_frames):

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for y in range(HEIGHT):
            v = int(15 + 20 * y / HEIGHT)
            frame[y, :] = (v, v + 5, v + 10)


        offset = (f * 2) % 80
        for gx in range(-offset, WIDTH, 80):
            cv2.line(frame, (gx, 0), (gx, HEIGHT), (30, 40, 50), 1)
        for gy in range(-offset, HEIGHT, 80):
            cv2.line(frame, (0, gy), (WIDTH, gy), (30, 40, 50), 1)


        blob_pos += blob_vel
        for i in range(num_blobs):
            for dim, limit in enumerate([WIDTH, HEIGHT]):
                if blob_pos[i, dim] < 30 or blob_pos[i, dim] > limit - 30:
                    blob_vel[i, dim] *= -1
            cx, cy = int(blob_pos[i, 0]), int(blob_pos[i, 1])
            sz = int(blob_sizes[i])
            col = blob_colors[i]
            if i % 2 == 0:
                cv2.rectangle(frame, (cx - sz, cy - sz), (cx + sz, cy + sz), col, -1)
            else:
                cv2.circle(frame, (cx, cy), sz, col, -1)

        writer.write(frame)

    writer.release()
    print(f"[assets] Synthetic background video written to {OUTPUT_PATH}  ({total_frames} frames)")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate()
