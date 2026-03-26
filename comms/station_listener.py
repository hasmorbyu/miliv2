
import json
import socket

HOST = "127.0.0.1"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
print(f"[Listener] Waiting for telemetry on udp://{HOST}:{PORT}  (Ctrl+C to stop)")

try:
    while True:
        data, _ = sock.recvfrom(4096)
        record = json.loads(data.decode())
        print(
            f"[Frame {record.get('frame', '?'):04d}] "
            f"MODE={record.get('mode'):<9s}  "
            f"TGT={record.get('target_id')}  "
            f"CONF={record.get('target_confidence', 0):.2f}  "
            f"POS={record.get('position')}  "
            f"STATUS={record.get('status')}"
        )
except KeyboardInterrupt:
    print("\n[Listener] Stopped.")
finally:
    sock.close()
