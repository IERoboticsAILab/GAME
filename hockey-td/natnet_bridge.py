"""
NatNet -> UDP bridge for the hockey game. Runs outside TouchDesigner.

Reads the players' rigid bodies straight from Motive (natnet_client.py, pure Python,
next to this file) and sends each one to the game's UDP port in the same format
hydra's forwarders used: "[x, y, z, heading]", position in cm, heading in radians.

    python natnet_bridge.py                 # game on this PC
    python natnet_bridge.py 10.205.3.218    # game on another PC

Start it, then open the .toe. Stop with Ctrl+C.
"""
import math
import socket
import sys
import time

from natnet_client import NatNetClient

MOTIVE_IP = "10.205.3.3"  # Motive PC (Gaia), streaming NatNet in unicast
# Rigid body name in Motive -> game UDP port (hockey_game.PLAYERS: 9879 left, 9878 right).
BODIES = {
    "GAME_Player2": 9878,
    "GAME_Player3": 9879,
}
STATUS_EVERY = 5.0  # seconds between status lines


def heading(q):
    """Rotation about the vertical (Z) axis in radians, as hydra's forwarders sent it."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    game_ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = {name: 0 for name in BODIES}
    lost = {name: 0 for name in BODIES}
    frames = [0]  # rigid body samples received for any body

    def on_rigid_body(rb):
        frames[0] += 1
        port = BODIES.get(rb.name)
        if port is None:
            return
        if not rb.tracking_valid:  # Motive lost it: send nothing, the mallet stays put
            lost[rb.name] += 1
            return
        x, y, z = (v * 100 for v in rb.position)
        sock.sendto(f"[{x},{y},{z},{heading(rb.rotation)}]".encode(), (game_ip, port))
        sent[rb.name] += 1

    client = NatNetClient(server_ip=MOTIVE_IP, use_multicast=False)
    client.on_rigid_body = on_rigid_body
    client.start()
    print(f"Motive {MOTIVE_IP} -> game {game_ip}: "
          + ", ".join(f"{name} on port {port}" for name, port in BODIES.items()))
    try:
        while True:
            time.sleep(STATUS_EVERY)
            if client.id_to_name and not frames[0]:
                # Motive answered (we have the body names) but no frames arrive: Windows
                # Firewall lets replies from port 1510 in, but drops the frames from 1511.
                print(time.strftime("%H:%M:%S"), f"Motive answers but no frames arrive. Windows Firewall is"
                      f" probably blocking UDP from {MOTIVE_IP} port 1511 (see README).", flush=True)
                continue
            frames[0] = 0
            parts = []
            for name in BODIES:
                if sent[name]:
                    parts.append(f"{name} {sent[name] / STATUS_EVERY:.0f}/s")
                elif lost[name]:
                    parts.append(f"{name} NOT TRACKED")
                elif name in client.id_to_name.values():
                    parts.append(f"{name} no data")
                else:
                    parts.append(f"{name} NOT IN MOTIVE")
                sent[name] = lost[name] = 0
            print(time.strftime("%H:%M:%S"), " | ".join(parts), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
