from __future__ import annotations
import argparse
import os
import sys
import time

import cv2
import numpy as np

from assets.sprites import make_drone_sprite, make_obstacle_sprite
import assets.generate_background as bg_gen
from detection.yolo_detector import detect as yolo_detect
from tracking.tracker import Tracker
from simulation.world import WorldState
from decision.engine import DecisionEngine, Mode
from planning.path_planner import compute_velocity
from rendering.renderer import Renderer
from comms.ground_station import GroundStation, DroneState

DEFAULT_BG_PATH = os.path.join(os.path.dirname(__file__), "assets", "background.mp4")
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "output.mp4")
TARGET_FPS      = 25
NUM_OBSTACLES   = 5


def parse_args():
    p = argparse.ArgumentParser(description="Autonomous Drone Simulation")
    p.add_argument("--input",      type=str,   default=None,
                   help="Path to background video (default: auto-generated)")
    p.add_argument("--model",      type=str,   default=None,
                   help="Custom YOLO model path (default: auto-download drone model)")
    p.add_argument("--frames",     type=int,   default=0,
                   help="Max frames to process (0 = all)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live preview window (headless mode)")
    p.add_argument("--verbose",    action="store_true",
                   help="Print telemetry to console each frame")
    p.add_argument("--conf",       type=float, default=0.25,
                   help="YOLO confidence threshold (default 0.25 for drone detection)")
    p.add_argument("--obstacles",  type=int,   default=NUM_OBSTACLES,
                   help="Number of moving obstacles")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = args.input or DEFAULT_BG_PATH
    if not os.path.isfile(video_path):
        print("[main] No background video found – generating synthetic one…")
        video_path = bg_gen.generate()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[main] Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[main] Source: {video_path}  ({width}×{height}  {src_fps:.1f}fps  {total_src_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, TARGET_FPS, (width, height))

    drone_sprite    = make_drone_sprite(56)
    obstacle_sprite = make_obstacle_sprite(42)
    world           = WorldState.create(width, height, num_obstacles=args.obstacles)
    tracker         = Tracker()
    engine          = DecisionEngine()
    renderer        = Renderer(width, height, drone_sprite, obstacle_sprite,
                               show_preview=not args.no_display)
    gs              = GroundStation(use_udp=True, verbose=args.verbose)

    frame_num   = 0
    max_frames  = args.frames if args.frames > 0 else 10 ** 9
    t_start     = time.time()

    print("[main] Starting simulation – press 'q' in preview window to stop early.")
    print(f"[main] Output will be saved to: {OUTPUT_PATH}")

    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        detections = yolo_detect(frame, model_name=args.model, conf_threshold=args.conf)

        tracks = tracker.update(detections)

        decision = engine.update(tracks, world.drone_pos)

        is_intercept = (decision.mode == Mode.INTERCEPT)
        world.drone_vel = compute_velocity(
            world,
            target_pos=decision.target_predicted_pos,
            mode_is_intercept=is_intercept,
        )

        world.update(dt=1.0)

        rendered, should_stop = renderer.render(
            frame, tracks, decision, world.drone_pos,
            world.obstacles, engine.neutralized_count if hasattr(engine, 'neutralized_count') else len(engine.neutralized)
        )

        
        state = DroneState(
            frame=frame_num,
            position=(float(world.drone_pos[0]), float(world.drone_pos[1])),
            velocity=(float(world.drone_vel[0]), float(world.drone_vel[1])),
            mode=decision.mode.value,
            target_id=decision.target_id,
            target_confidence=decision.target_confidence,
            status=decision.status,
            neutralized_count=len(engine.neutralized),
            timestamp=time.time(),
        )
        gs.send(state)

        
        writer.write(rendered)

        frame_num += 1


        if frame_num % 50 == 0:
            elapsed = time.time() - t_start
            fps_now = frame_num / elapsed if elapsed > 0 else 0
            print(f"[main] Frame {frame_num:5d}  "
                  f"MODE={decision.mode.value:<9s}  "
                  f"NEUTRALIZED={len(engine.neutralized)}  "
                  f"SIM_FPS={fps_now:.1f}")

        if should_stop:
            print("[main] User pressed 'q' – stopping early.")
            break

    elapsed = time.time() - t_start
    cap.release()
    writer.release()
    renderer.close()
    gs.close()

    print(f"\n[main] Done.  {frame_num} frames processed in {elapsed:.1f}s"
          f"  ({frame_num / elapsed:.1f} fps)")
    print(f"[main] Output video : {OUTPUT_PATH}")
    print(f"[main] Telemetry log: {os.path.join(os.path.dirname(__file__), 'logs', 'telemetry.jsonl')}")
    print(f"[main] Events log   : {os.path.join(os.path.dirname(__file__), 'logs', 'events.log')}")


if __name__ == "__main__":
    main()
