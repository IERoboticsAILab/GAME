"""
Air hockey for two players in two rooms, one projector each. Port of hockey.gaml,
with rigid-body puck physics.

Pure Python (numpy + cv2, both bundled with TouchDesigner), so it runs inside a
Script TOP or standalone: `python3 hockey_game.py` runs the self-check.

Renders the whole table as one image; TouchDesigner sends the same image to
both projectors over NDI (see README.md).

Physics: the puck is a spinning disc on an air cushion (small linear drag). Mallets are
kinematic (moved by people, so effectively infinite mass) and their velocity is estimated
from the tracker, so hits transfer momentum. Every contact uses restitution plus Coulomb
rim friction, which is what makes glancing hits spin the puck. Each frame is split into
substeps small enough that a fast puck or a jumpy tracker sample can't tunnel through.
"""
import math
import random
import re
import socket
from collections import deque

import cv2
import numpy as np

# TABLE (same units as the GAMA model; 1 unit is roughly 2.5 cm on the lab floor)
W, H = 137.5, 68.5
PX = 16  # pixels per table unit -> 2200 x 1096 image
RAIL = 1.2  # rail thickness
GOAL_HALF = 11.0  # half the goal mouth (mouth is 22 wide, as in the GAMA model)
POSTS = [(RAIL, H / 2 - GOAL_HALF), (RAIL, H / 2 + GOAL_HALF),
         (W - RAIL, H / 2 - GOAL_HALF), (W - RAIL, H / 2 + GOAL_HALF)]

# TABLES. One per room, all in the same Motive space. correction is where the table sits
# in Motive (see README); mirror_x/mirror_y flip an axis that runs the wrong way.
# A tracker plays on whichever table it is over, so players can share a table or split up.
TABLES = {
    "room1": dict(correction=(134.1, -316.0), mirror_x=True, mirror_y=False),
    "room2": dict(correction=(557.8, -297.2), mirror_x=True, mirror_y=False),
}

# PLAYERS. Ports are 9876 + the lab's UMH Base ID (2). Side matches the original:
# the first port (9878) got odd id 3 -> right half, the second (9879) id 4 -> left half.
PLAYERS = [
    dict(side="left", port=9879),
    dict(side="right", port=9878),
]

# GAME
WIN_SCORE = 10
SPIN_TO_START = 90.0  # accumulated rotation needed on a Start square
START_SPOT = {"left": (W / 4, H / 2), "right": (3 * W / 4, H / 2)}
GOAL_PAUSE = 1.5  # seconds of "GOAL!" before the puck comes back
SERVE_SPEED = 25.0  # opening face-off push; after a goal the puck is placed, not served

# PHYSICS (table units and seconds). These are the knobs to tune on site.
PUCK_R = 2.5
MALLET_R = 4.0
AIR_DRAG = 0.12  # 1/s: speed lost to the air cushion, ~11 % per second
SPIN_DRAG = 0.8  # 1/s
E_WALL = 0.85  # restitution, 1 = perfectly bouncy
E_MALLET = 0.75
E_POST = 0.6
FRICTION = 0.2  # rim friction; turns glancing hits into spin
MAX_PUCK_SPEED = 160.0
MAX_MALLET_SPEED = 250.0  # faster tracker jumps are glitches, clamped
MALLET_SMOOTH = 0.5  # 0..1, how quickly mallet velocity follows the tracker (lower = smoother)
MALLET_TIMEOUT = 0.25  # s without tracker data -> mallet treated as still
GLITCH = 20.0  # tracker samples further than this outside the table are ignored
PHYS_HZ = 240  # minimum physics substeps per second

# Mallets stay inside the rails and fully on their own half.
X_LIMITS = {"left": (RAIL + MALLET_R, W / 2 - MALLET_R), "right": (W / 2 + MALLET_R, W - RAIL - MALLET_R)}
Y_LIMITS = (RAIL + MALLET_R, H - RAIL - MALLET_R)

# DISPLAY
# Text rotation per half, degrees counter-clockwise, so it reads upright for a player
# standing at their goal end facing the centre line. Set to 0 for no rotation.
FACING = {"left": -90, "right": 90}
TRAIL = 8  # frames of puck trail
RING_TIME = 0.35  # s an impact flash lasts
RING_MIN = 20.0  # impulse needed for an impact flash

# SOUND (files in sounds/, played by TouchDesigner; see README)
SOUND_FULL = {"hit": 120.0, "wall": 150.0, "post": 100.0}  # impulse that plays at full volume
SOUND_MIN = 0.15  # quieter cues are skipped, so a puck resting on a mallet stays silent
SOUND_GAP = 0.06  # s, minimum time between two of the same sound

# COLOURS (RGB)
SURFACE = (232, 242, 248)
HOLE = (203, 216, 228)
RAIL_C = (92, 100, 118)
SLOT = (12, 12, 16)
LINE_RED = (205, 40, 50)
PUCK_C = (24, 24, 30)
IMPACT = (255, 170, 30)
DARK, WHITE, BLACK, GREEN = (40, 45, 55), (255, 255, 255), (0, 0, 0), (0, 150, 0)
PLAYER_C = {"left": (215, 45, 45), "right": (35, 95, 215)}

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def px(x, y):
    return int(round(x * PX)), int(round(y * PX))


def rgba(c):
    return tuple(int(v) for v in c[:3]) + (255,)


def to_table(tx, ty, t):
    """Raw Motive position -> table units, using one table's calibration."""
    cx, cy = t["correction"]
    x, y = tx - cx, -(ty + cy)
    return (W - x if t["mirror_x"] else x), (H - y if t["mirror_y"] else y)


def mix(c, bg, a):
    """c blended over bg with opacity a (fake alpha: the table behind is ~uniform)."""
    return rgba([bg[i] + (c[i] - bg[i]) * a for i in range(3)])


class Player:
    def __init__(self, side, port, tables=TABLES, bind=True):
        self.side = side
        self.tables = tables
        self.table = None  # name of the table the tracker was last seen over
        self.pos = self.prev = START_SPOT[side]  # prev = position at the last physics step
        self.vel = (0.0, 0.0)
        self.t_sample, self.p_sample = None, self.pos
        self.rot = 0.0
        self.spin = 0.0
        self.color = PLAYER_C[side]
        self.sock = None
        if bind:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.sock.bind(("0.0.0.0", port))

    def poll(self, now):
        got = False
        while self.sock:
            try:
                data = self.sock.recv(4096)
            except BlockingIOError:
                break
            except ConnectionResetError:  # Windows reports ICMP errors on UDP recv
                continue
            got = self.handle(data.decode(errors="ignore")) or got
        self.track(now, got)

    def handle(self, text):
        """Apply one tracker message. Returns True if it moved the mallet."""
        nums = NUM_RE.findall(text)
        if len(nums) < 4:
            return False
        rot = float(nums[3]) * -100
        self.spin += abs(rot - self.rot)
        self.rot = rot
        for name, t in self.tables.items():  # first table the point is over
            x, y = to_table(float(nums[0]), float(nums[1]), t)
            if -GLITCH <= x <= W + GLITCH and -GLITCH <= y <= H + GLITCH:
                self.table = name
                break
        else:  # between tables, or a glitch
            return False
        lo, hi = X_LIMITS[self.side]
        self.pos = (min(max(x, lo), hi), min(max(y, Y_LIMITS[0]), Y_LIMITS[1]))
        return True

    def track(self, now, got):
        """Smoothed mallet velocity from tracker samples; trackers are noisy and bursty."""
        if got:
            if self.t_sample is not None and now > self.t_sample:
                dt = now - self.t_sample
                vx = (self.pos[0] - self.p_sample[0]) / dt
                vy = (self.pos[1] - self.p_sample[1]) / dt
                s = math.hypot(vx, vy)
                if s > MAX_MALLET_SPEED:
                    vx, vy = vx * MAX_MALLET_SPEED / s, vy * MAX_MALLET_SPEED / s
                a = MALLET_SMOOTH
                self.vel = (self.vel[0] + a * (vx - self.vel[0]), self.vel[1] + a * (vy - self.vel[1]))
            self.t_sample, self.p_sample = now, self.pos
        elif self.t_sample is None or now - self.t_sample > MALLET_TIMEOUT:
            self.vel = (0.0, 0.0)

    def close(self):
        if self.sock:
            self.sock.close()


class Game:
    def __init__(self, players=PLAYERS, bind=True):
        self.players = [Player(**p, bind=bind) for p in players]
        self.scores = {"left": 0, "right": 0}
        self.started = False
        self.first = True
        self.t = None  # last frame's clock (from TouchDesigner)
        self.clock = 0.0  # game time, advances with step()
        self.flash = None  # side that just scored, during the goal pause
        self.pause_until = 0.0
        self.serve_side = None
        self.trail = deque(maxlen=TRAIL)
        self.rings = []  # impact flashes: (x, y, t0, strength 0..1)
        self.events = []  # this frame's ("hit"/"wall"/"post", impulse) and ("goal", side)
        self.sounds = []  # this frame's (sound name, volume 0..1), read by TouchDesigner
        self._last_sound = {}
        self.reset_puck()
        self._bg_key = None

    def reset_game(self):
        """Start a new game immediately (skips the Start squares)."""
        self.scores = {"left": 0, "right": 0}
        self.started = True
        self.first = False
        self.flash = None
        self.serve()

    def stop_game(self):
        """Back to the start screen with the Start squares."""
        self.scores = {"left": 0, "right": 0}
        self.started = False
        self.first = True
        self.flash = None
        self.reset_puck()

    def reset_puck(self):
        """Puck still, at centre ice."""
        self.puck, self.v = (W / 2, H / 2), (0.0, 0.0)
        self.w = self.angle = 0.0  # spin (rad/s) and orientation
        self.live = True
        self.trail.clear()

    def serve(self, side=None):
        """side=None: face-off at centre with a gentle random push towards one player.
        Otherwise the puck is placed, still, on that (conceding) player's side."""
        self.reset_puck()
        if side is None:
            a = random.choice((0.0, math.pi)) + random.uniform(-0.6, 0.6)
            self.v = (SERVE_SPEED * math.cos(a), SERVE_SPEED * math.sin(a))
        else:
            self.puck = (W / 2 - 15 if side == "left" else W / 2 + 15, H / 2)

    def update(self, now):
        """Call once per frame with a clock in seconds."""
        dt = 0.0 if self.t is None else min(max(now - self.t, 0.0), 0.1)
        self.t = now
        self.events = []
        for p in self.players:
            p.poll(now)
        self.step(dt)
        self.sounds = self._sound_cues()

    def _sound_cues(self):
        """Turn this frame's events into at most one cue per sound, loudness from the impulse."""
        loud = {}
        for kind, val in self.events:
            vol = 1.0 if kind == "goal" else math.sqrt(min(1.0, val / SOUND_FULL[kind]))  # sqrt ~ perceived loudness
            loud[kind] = max(loud.get(kind, 0.0), vol)
        cues = []
        for kind, vol in loud.items():
            if vol >= SOUND_MIN and self.clock - self._last_sound.get(kind, -1.0) >= SOUND_GAP:
                self._last_sound[kind] = self.clock
                cues.append((kind, round(vol, 3)))
        return cues

    def step(self, dt):
        self.clock += dt
        if not self.started:
            for p in self.players:
                if math.dist(p.pos, START_SPOT[p.side]) <= 3.0:
                    if p.spin >= SPIN_TO_START:
                        p.spin = 0.0
                        self.reset_game()
                        break
                else:
                    p.spin = 0.0
        elif not self.live:
            if self.clock >= self.pause_until:
                self.flash = None
                self.serve(self.serve_side)
        else:
            self._physics(dt)
            if self.live:
                self.trail.append(self.puck)
        for p in self.players:
            p.prev = p.pos
        self.rings = [r for r in self.rings if self.clock - r[2] < RING_TIME]

    # ---------- physics ----------

    def _physics(self, dt):
        if dt <= 0:
            return
        # substeps: at least PHYS_HZ, and small enough that nothing moves > half a puck radius
        sweep = max(math.dist(p.prev, p.pos) for p in self.players)
        travel = math.hypot(*self.v) * dt + sweep
        n = min(200, max(1, math.ceil(dt * PHYS_HZ), math.ceil(travel / (0.5 * PUCK_R))))
        h = dt / n
        for i in range(1, n + 1):
            f = i / n  # mallets glide from last frame's position to this one's
            mallets = [((p.prev[0] + (p.pos[0] - p.prev[0]) * f, p.prev[1] + (p.pos[1] - p.prev[1]) * f), p.vel)
                       for p in self.players]
            self._substep(h, mallets)
            if self._check_goal():
                return

    def _substep(self, h, mallets):
        k = math.exp(-AIR_DRAG * h)
        self.v = (self.v[0] * k, self.v[1] * k)
        self.w *= math.exp(-SPIN_DRAG * h)
        self.puck = (self.puck[0] + self.v[0] * h, self.puck[1] + self.v[1] * h)
        self.angle += self.w * h
        for c, cv in mallets:
            self._hit_circle(c, MALLET_R, cv, E_MALLET, "hit")
        self._hit_walls()
        for post in POSTS:
            self._hit_circle(post, 0.0, (0.0, 0.0), E_POST, "post")
        s = math.hypot(*self.v)
        if s > MAX_PUCK_SPEED:
            self.v = (self.v[0] * MAX_PUCK_SPEED / s, self.v[1] * MAX_PUCK_SPEED / s)

    def _hit_walls(self):
        lo, hi = RAIL + PUCK_R, H - RAIL - PUCK_R
        if self.puck[1] < lo:
            self._contact((0.0, 1.0), lo - self.puck[1], (0.0, 0.0), E_WALL, "wall")
        elif self.puck[1] > hi:
            self._contact((0.0, -1.0), self.puck[1] - hi, (0.0, 0.0), E_WALL, "wall")
        if abs(self.puck[1] - H / 2) < GOAL_HALF:
            return  # in front of a goal mouth: no end rail, only the posts
        lo, hi = RAIL + PUCK_R, W - RAIL - PUCK_R
        if self.puck[0] < lo:
            self._contact((1.0, 0.0), lo - self.puck[0], (0.0, 0.0), E_WALL, "wall")
        elif self.puck[0] > hi:
            self._contact((-1.0, 0.0), self.puck[0] - hi, (0.0, 0.0), E_WALL, "wall")

    def _hit_circle(self, c, r, cv, e, kind):
        dx, dy = self.puck[0] - c[0], self.puck[1] - c[1]
        d = math.hypot(dx, dy)
        reach = PUCK_R + r
        if d >= reach:
            return
        n = (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)
        self._contact(n, reach - d, cv, e, kind)

    def _contact(self, n, pen, vo, e, kind):
        """Resolve a contact: n points from the other body to the puck, vo is its velocity."""
        nx, ny = n
        self.puck = (self.puck[0] + nx * pen, self.puck[1] + ny * pen)
        # velocity of the puck's rim at the contact point (r = -n R), relative to the other body
        rx, ry = -nx * PUCK_R, -ny * PUCK_R
        vrx = self.v[0] - self.w * ry - vo[0]
        vry = self.v[1] + self.w * rx - vo[1]
        vn = vrx * nx + vry * ny
        if vn >= 0:
            return  # already separating
        jn = -(1 + e) * vn  # normal impulse per unit puck mass
        tx, ty = -ny, nx
        vt = vrx * tx + vry * ty
        # tangential impulse: 3 = 1/m + R^2/I for a solid disc; capped by Coulomb friction
        jt = max(-FRICTION * jn, min(FRICTION * jn, -vt / 3))
        self.v = (self.v[0] + jn * nx + jt * tx, self.v[1] + jn * ny + jt * ty)
        self.w -= 2 * jt / PUCK_R  # r x t = -R, I = R^2 / 2
        self.events.append((kind, jn))
        if jn > RING_MIN:
            self.rings.append((self.puck[0] + rx, self.puck[1] + ry, self.clock, min(jn / 120, 1.0)))

    def _check_goal(self):
        x, y = self.puck
        if abs(y - H / 2) >= GOAL_HALF:
            return False
        if x < RAIL:
            scorer = "right"
        elif x > W - RAIL:
            scorer = "left"
        else:
            return False
        self.scores[scorer] += 1
        self.events.append(("goal", scorer))
        if self.scores[scorer] >= WIN_SCORE:
            self.started = False
            self.reset_puck()
        else:
            self.live = False
            self.flash = scorer
            self.serve_side = "left" if scorer == "right" else "right"
            self.pause_until = self.clock + GOAL_PAUSE
        return True

    # ---------- rendering ----------

    def _draw_background(self):
        """Table + per-half text. Only redrawn when score/state changes."""
        img = np.empty((int(H * PX), int(W * PX), 4), np.uint8)
        img[:] = rgba(SURFACE)
        for x in np.arange(2.5, W, 2.5):  # air holes
            for y in np.arange(2.5, H, 2.5):
                cv2.circle(img, px(x, y), max(1, PX // 8), rgba(HOLE), -1, cv2.LINE_AA)
        line = int(0.5 * PX)
        cv2.line(img, px(W / 2, 0), px(W / 2, H), rgba(LINE_RED), line)
        cv2.circle(img, px(W / 2, H / 2), 9 * PX, rgba(LINE_RED), int(0.35 * PX), cv2.LINE_AA)
        cv2.circle(img, px(W / 2, H / 2), PX, rgba(LINE_RED), -1, cv2.LINE_AA)
        for gx, start in ((0, -90), (W, 90)):  # goal creases
            cv2.ellipse(img, px(gx, H / 2), (16 * PX, 16 * PX), 0, start, start + 180,
                        rgba(LINE_RED), int(0.35 * PX), cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), rgba(RAIL_C), int(2 * RAIL * PX))
        for gx in (0, W - RAIL):  # goal slots in the end rails
            cv2.rectangle(img, px(gx, H / 2 - GOAL_HALF), px(gx + RAIL, H / 2 + GOAL_HALF), rgba(SLOT), -1)
        for post in POSTS:
            cv2.circle(img, px(*post), int(0.6 * PX), rgba((150, 155, 165)), -1, cv2.LINE_AA)

        hw = img.shape[1] // 2
        for side, x0 in (("left", 0), ("right", hw)):
            # draw upright on a transparent layer in half-local coords, rotate, composite
            layer = np.zeros((img.shape[0], hw, 4), np.uint8)
            cx, cy = W / 4, H / 2
            text(layer, f"{self.scores['left']} : {self.scores['right']}", (cx, cy - 19), 5, rgba(DARK))
            if self.first:
                text(layer, "Stand on Start and spin to play!", (cx, cy - 12), 3, rgba(GREEN))
            elif self.flash:
                text(layer, "GOAL!", (cx, cy - 12), 7, rgba(PLAYER_C[self.flash]))
            elif not self.started:
                if self.scores["left"] >= WIN_SCORE:
                    text(layer, "Player 1 Wins!", (cx, cy - 12), 5, rgba(PLAYER_C["left"]))
                elif self.scores["right"] >= WIN_SCORE:
                    text(layer, "Player 2 Wins!", (cx, cy - 12), 5, rgba(PLAYER_C["right"]))
            if not self.started:
                cv2.rectangle(layer, px(cx - 5, cy - 5), px(cx + 5, cy + 5), rgba(BLACK), -1)
                text(layer, "Start", (cx, cy), 2.5, rgba(WHITE), max_w=9)
            if FACING[side]:
                m = cv2.getRotationMatrix2D((hw / 2, img.shape[0] / 2), FACING[side], 1)
                layer = cv2.warpAffine(layer, m, (hw, img.shape[0]))
            # layer is premultiplied (AA edges fade toward 0), so: out = layer + bg * (1 - alpha)
            region = img[:, x0:x0 + hw]
            inv = 255 - layer[..., 3:].astype(np.uint16)
            region[:] = layer + (region * inv // 255).astype(np.uint8)
        return img

    def render(self):
        """Full table, top row = top of table."""
        key = (self.scores["left"], self.scores["right"], self.started, self.first, self.flash)
        if key != self._bg_key:
            self._bg_key, self._bg = key, self._draw_background()
        img = self._bg.copy()
        for x, y, t0, s in self.rings:
            a = (self.clock - t0) / RING_TIME
            r = PUCK_R * 0.5 + (2 + 6 * s) * a
            cv2.circle(img, px(x, y), int(r * PX), mix(IMPACT, SURFACE, (1 - a) * (0.4 + 0.6 * s)),
                       max(1, int(0.3 * PX)), cv2.LINE_AA)
        speed = math.hypot(*self.v)
        if self.live and speed > 30:
            pts = list(self.trail)[:-1]
            for i, (x, y) in enumerate(pts):
                f = (i + 1) / (len(pts) + 1)
                cv2.circle(img, px(x, y), int(PUCK_R * PX * (0.4 + 0.5 * f)),
                           mix(PUCK_C, SURFACE, 0.3 * f * min(1.0, speed / 100)), -1, cv2.LINE_AA)
        for p in self.players:
            draw_mallet(img, p.pos, p.color)
        if self.live:
            draw_puck(img, self.puck, self.angle)
        return img

    def frame(self):
        """render() flipped for TouchDesigner, whose numpy origin is bottom-left."""
        return np.ascontiguousarray(self.render()[::-1])

    def close(self):
        for p in self.players:
            p.close()


def shadow(img, pos, r):
    cv2.circle(img, px(pos[0] + 0.5, pos[1] + 0.7), int(r * PX), mix(BLACK, SURFACE, 0.22), -1, cv2.LINE_AA)


def draw_mallet(img, pos, color):
    shadow(img, pos, MALLET_R)
    c, r = px(*pos), MALLET_R * PX
    cv2.circle(img, c, int(r), mix(BLACK, color, 0.4), -1, cv2.LINE_AA)  # rim
    cv2.circle(img, c, int(r * 0.85), rgba(color), -1, cv2.LINE_AA)  # body
    cv2.circle(img, c, int(r * 0.5), mix(BLACK, color, 0.25), -1, cv2.LINE_AA)  # handle base
    cv2.circle(img, c, int(r * 0.4), mix(WHITE, color, 0.35), -1, cv2.LINE_AA)  # handle
    cv2.circle(img, px(pos[0] - 0.15 * MALLET_R, pos[1] - 0.15 * MALLET_R), int(r * 0.12),
               mix(WHITE, color, 0.8), -1, cv2.LINE_AA)  # highlight


def draw_puck(img, pos, angle):
    shadow(img, pos, PUCK_R)
    c, r = px(*pos), PUCK_R * PX
    cv2.circle(img, c, int(r), rgba(PUCK_C), -1, cv2.LINE_AA)
    cv2.circle(img, c, int(r * 0.72), rgba((75, 75, 88)), max(1, int(r * 0.06)), cv2.LINE_AA)
    ca, sa = math.cos(angle), math.sin(angle)  # notch shows the spin
    cv2.line(img, px(pos[0] + ca * PUCK_R * 0.3, pos[1] + sa * PUCK_R * 0.3),
             px(pos[0] + ca * PUCK_R * 0.85, pos[1] + sa * PUCK_R * 0.85),
             rgba((225, 225, 232)), max(1, int(r * 0.16)), cv2.LINE_AA)


def text(img, s, centre, height, color, max_w=W / 2 * 0.9):
    """Draw s centred at centre, `height` table units tall, shrunk to fit max_w."""
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(height * PX / 12))
    (tw, th), _ = cv2.getTextSize(s, font, 1.0, thick)
    scale = min(height * PX / th, max_w * PX / tw)
    (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
    x, y = px(*centre)
    cv2.putText(img, s, (x - tw // 2, y + th // 2), font, scale, color, thick, cv2.LINE_AA)


# One game per TouchDesigner session; the module stays cached in sys.modules.
_game = None


def get_game():
    global _game
    if _game is None:
        _game = Game()
    return _game


def close():
    global _game
    if _game:
        _game.close()
    _game = None


if __name__ == "__main__":
    import time

    random.seed(1)
    g = Game(bind=False)
    left, right = g.players
    def msg(p, x, y, rot=0.0, table="room1"):  # tracker message for point (x, y) on a table
        t = p.tables[table]
        cx, cy = t["correction"]
        x, y = (W - x if t["mirror_x"] else x), (H - y if t["mirror_y"] else y)
        return p.handle(f"[{x + cx}, {-y - cy}, 0, {rot}]")

    def near(a, b):  # mirrored tables round-trip with float error
        return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9

    def place(puck, v, w=0.0):
        g.puck, g.v, g.w, g.live = puck, v, w, True

    def still(p, pos):
        p.pos = p.prev = pos
        p.vel = (0.0, 0.0)

    # tracker: parsing, clamping to own half, glitches ignored
    assert msg(left, 20, 30) and near(left.pos, (20, 30))
    msg(left, 100, 30)
    assert left.pos == (W / 2 - MALLET_R, 30)
    assert not msg(left, 500, 30) and not left.handle("garbage")
    # tables: a tracker plays on whichever table it is over; between tables it is ignored
    assert msg(left, 30, 40, table="room2") and near(left.pos, (30, 40)) and left.table == "room2"
    assert msg(right, 100, 20, table="room2") and right.table == "room2"
    assert msg(left, 20, 30) and near(left.pos, (20, 30)) and left.table == "room1"
    c1, c2 = TABLES["room1"]["correction"], TABLES["room2"]["correction"]
    assert not left.handle(f"{(c1[0] + W + c2[0]) / 2} {-(c1[1] + c2[1]) / 2 - H / 2} 0 0")
    m = Player("left", 0, {"m": dict(correction=(0, 0), mirror_x=True, mirror_y=False)}, bind=False)
    assert m.handle("10 -20 0 0") and m.pos == (W / 2 - MALLET_R, 20)  # x 10 -> W - 10, clamped to own half
    assert msg(m, 20, 30, table="m") and near(m.pos, (20, 30))
    # mallet velocity from samples: smoothed, and zeroed when the tracker goes quiet
    msg(left, 20, 30); left.track(0.0, True)
    msg(left, 25, 30); left.track(0.1, True)
    assert abs(left.vel[0] - 50 * MALLET_SMOOTH) < 1e-9, left.vel
    left.track(1.0, False)
    assert left.vel == (0.0, 0.0)

    # spin on the Start square starts the game
    msg(left, *START_SPOT["left"])
    left.spin = 0
    msg(left, *START_SPOT["left"], 1.0)  # rot -100 -> spin 100
    g.step(0.016)
    assert g.started and not g.first
    still(left, (10, 60)); still(right, (130, 60))

    # wall: bounces with restitution, drag barely slows it
    place((40, 20), (0, -50))
    g.step(0.5)
    assert 35 < g.v[1] < 43, g.v
    # a moving mallet transfers momentum: still puck leaves faster than the mallet
    place((40, 30), (0, 0))
    left.prev, left.pos, left.vel = (33, 30), (34, 30), (60, 0)
    g.step(1 / 60)
    assert g.v[0] > 90, g.v
    # no tunnelling: 16 units in one frame, still bounces off a mallet in the way
    still(left, (40, 30))
    place((60, 30), (-160, 0))
    g.step(0.1)
    assert g.v[0] > 0 and g.puck[0] > 40, (g.puck, g.v)
    # glancing hit spins the puck
    still(left, (30, 34.5))
    place((40, 30), (-80, 0))
    g.step(0.1)
    assert abs(g.w) > 1, g.w
    # goal: scores, pauses, then the puck is placed still on the conceding side
    still(left, (10, 60))
    place((10, H / 2), (-80, 0))
    g.step(0.2)
    assert g.scores["right"] == 1 and not g.live and g.flash == "right"
    g.step(GOAL_PAUSE + 0.01)
    assert g.live and g.flash is None and g.puck == (W / 2 - 15, H / 2) and g.v == (0, 0)
    # posts: a puck aimed at a post doesn't score
    place((10, H / 2 + GOAL_HALF), (-80, 0))
    g.step(0.3)
    assert g.scores["right"] == 1 and g.v[0] > 0, (g.scores, g.v)
    # winning ends the game
    g.scores["right"] = WIN_SCORE - 1
    place((10, H / 2), (-80, 0))
    g.step(0.2)
    assert not g.started and g.scores["right"] == WIN_SCORE
    # sound cues: one per kind per frame, loudest wins, soft touches silent, no machine-gun repeats
    g.events = [("hit", 30.0), ("hit", 120.0), ("wall", 1.0), ("goal", "left")]
    assert sorted(g._sound_cues()) == [("goal", 1.0), ("hit", 1.0)], g._sound_cues()
    g.events = [("hit", 120.0)]
    assert g._sound_cues() == []  # same clock -> within SOUND_GAP
    g.clock += SOUND_GAP
    g.events = [("hit", 30.0)]
    assert g._sound_cues() == [("hit", 0.5)]
    # stop_game -> start screen
    g.stop_game()
    assert not g.started and g.first and g.scores == {"left": 0, "right": 0}

    # UDP path
    u = Player("left", 0, {"t": dict(correction=(0, 0), mirror_x=False, mirror_y=False)})
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"10 -20 0 0", ("127.0.0.1", u.sock.getsockname()[1]))
    time.sleep(0.05)
    u.poll(0.0)
    assert u.pos == (10, 20), u.pos
    u.close()

    # preview: a fast shot mid-game
    g.reset_game()
    still(left, (30, 25)); still(right, (105, 45))
    place((75, 40), (-120, -45), 6.0)
    for _ in range(12):
        g.step(1 / 60)
    assert g.frame().shape == (int(H * PX), int(W * PX), 4)
    t0 = time.perf_counter()
    for _ in range(60):
        g.frame()
    print(f"frame: {(time.perf_counter() - t0) / 60 * 1000:.1f} ms")
    cv2.imwrite("preview.png", cv2.cvtColor(g.render(), cv2.COLOR_RGBA2BGRA))
    print("ok, wrote preview.png")
