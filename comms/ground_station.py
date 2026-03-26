
from __future__ import annotations
import json
import os
import socket
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "telemetry.jsonl")
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005


@dataclass
class DroneState:
    frame: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    mode: str
    target_id: Optional[int]
    target_confidence: float
    status: str
    neutralized_count: int
    timestamp: float


class GroundStation:
    def __init__(self, use_udp: bool = True, verbose: bool = False):
        self.use_udp = use_udp
        self.verbose = verbose
        self._sock: Optional[socket.socket] = None
        os.makedirs(LOG_DIR, exist_ok=True)

        open(LOG_FILE, "w").close()
        if use_udp:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    def send(self, state: DroneState):
        record = asdict(state)

        record["position"] = list(state.position)
        record["velocity"] = list(state.velocity)


        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")


        if self.verbose:
            print(f"[GS] Frame {state.frame:04d}  MODE={state.mode:<9s}  "
                  f"TARGET={state.target_id}  POS=({state.position[0]:.1f},{state.position[1]:.1f})  "
                  f"STATUS={state.status}")


        if self.use_udp and self._sock:
            try:
                payload = json.dumps(record).encode()
                self._sock.sendto(payload, (UDP_HOST, UDP_PORT))
            except OSError:
                pass

    def close(self):
        if self._sock:
            self._sock.close()
