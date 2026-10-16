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
import bisect
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
GOAL_PAUSE = 1.5  # minimum seconds after a goal before the puck comes back
SERVE_SPEED = 25.0  # opening face-off push; after a goal the puck is placed, not served

# MODES. The start screen shows one square per mode in each half; spin on one to play it.
# Offsets run across the player's view (their left to right), in table units from the half's centre.
MODE_SQUARES = {"powerups": -20.0, "classic": 0.0, "computer": 20.0}
MODE_LABEL = {"powerups": "Power-ups", "classic": "Classic", "computer": "vs Computer"}
MODE_RADIUS = 4.0  # how close to a square's centre the mallet must be
ATTRACT_AFTER = 30.0  # s with no tracker moving on the start screen -> the computer plays itself
ABANDON_AFTER = 60.0  # s with no tracker moving mid-game -> back to the start screen
ACTIVE_MOVE = 1.0  # a tracker counts as moving once it is this far from where it last was

# COMPUTER PLAYER
AI_SPEED = 70.0  # top mallet speed (people manage ~250)
AI_REACTION = 0.15  # s, how far its aim lags behind the puck
AI_ERROR = 7.0  # how far off its guarding can be; picked again each time the puck comes over
AI_INCOMING = 20.0  # a puck coming at its goal faster than this gets goalkeeping, not a strike

# POWER-UPS (power-ups mode)
POWER_EVERY = 8.0  # s from one power-up being collected to the next appearing
POWER_TIME = 8.0  # s an effect lasts
FREEZE_TIME = 2.0  # s the opponent's mallet is frozen
POWER_R = 3.0  # radius of a power-up on the table
BIG = 1.6  # mallet size with "big"
SMALL_GOAL = 0.5  # opponent's goal mouth with "small_goal"
POWER_HIT = 1.5  # puck speed off your mallet with "power_hit"
POWERS = ("big", "small_goal", "power_hit", "freeze")
POWER_LABEL = {"big": "BIG MALLET", "small_goal": "SMALL GOAL", "power_hit": "POWER HIT", "frozen": "FROZEN!"}

# GOAL REPLAY (all modes)
REPLAY_TIME = 2.5  # s of play before a goal that the replay shows
REPLAY_SPEED = 0.5  # 0.5 = half speed
GOAL_FLASH = 1.0  # s of "GOAL!" before the replay starts

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
STALL_TIME = 3.0  # s a puck may sit still before the "air" drifts it back towards the centre
STALL_SPEED = 3.0  # slower than this counts as still
STALL_PUSH = 12.0  # speed of that drift; a puck in a deep corner can't be reached by any mallet
MALLET_SMOOTH = 0.5  # 0..1, how quickly mallet velocity follows the tracker (lower = smoother)
MALLET_TIMEOUT = 0.25  # s without tracker data -> mallet treated as still
GLITCH = 20.0  # tracker samples further than this outside the table are ignored
PHYS_HZ = 240  # minimum physics substeps per second


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
POWER_C = {"big": (240, 150, 20), "small_goal": (150, 60, 200), "power_hit": (230, 40, 120),
           "freeze": (40, 170, 230), "frozen": (40, 170, 230)}

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


def limits(side, r):
    """Where a mallet of radius r may be: inside the rails and fully on its own half."""
    x = (RAIL + r, W / 2 - r) if side == "left" else (W / 2 + r, W - RAIL - r)
    return x, (RAIL + r, H - RAIL - r)


def clamp(pos, side, r):
    (x0, x1), (y0, y1) = limits(side, r)
    return min(max(pos[0], x0), x1), min(max(pos[1], y0), y1)


def other(side):
    return "right" if side == "left" else "left"


def square_spot(side, offset):
    """Table position of a mode square: `offset` across the player's view, turned the same
    way as that half's text (cv2.getRotationMatrix2D, counter-clockwise on screen)."""
    a = math.radians(FACING[side])
    cx = W / 4 if side == "left" else 3 * W / 4
    return cx + math.cos(a) * offset, H / 2 - math.sin(a) * offset


def mix(c, bg, a):
    """c blended over bg with opacity a (fake alpha: the table behind is ~uniform)."""
    return rgba([bg[i] + (c[i] - bg[i]) * a for i in range(3)])


class Player:
    def __init__(self, side, port, tables=TABLES, bind=True):
        self.side = side
        self.tables = tables
        self.table = None  # name of the table the tracker was last seen over
        self.pos = self.prev = START_SPOT[side]  # prev = position at the last physics step
        self.raw = self.moved_from = self.pos  # tracker position (pos can be the computer's or frozen)
        self.r = MALLET_R  # current radius ("big" power-up changes it)
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
        self.raw = self.pos = clamp((x, y), self.side, self.r)
        return True

    def moved(self):
        """True once the tracker has moved ACTIVE_MOVE since the last time this returned True.
        The NatNet bridge streams every Motive frame, so arriving data alone doesn't mean anyone is there."""
        if math.dist(self.raw, self.moved_from) < ACTIVE_MOVE:
            return False
        self.moved_from = self.raw
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
        self.mode = "classic"  # "classic", "powerups", "computer" or "attract"
        self.ai = set()  # sides the computer plays
        self.ai_aim = {}  # side -> where the computer is heading (lags the puck by AI_REACTION)
        self.ai_err = {}  # side -> (puck in its half?, guarding error for that visit)
        self.seen = 0.0  # clock when a tracker last moved
        self.mode_since = 0.0  # clock when the current mode started
        self.effects = {"left": {}, "right": {}}  # side -> {power-up kind: clock it wears off}
        self.item = None  # (kind, (x, y)): power-up waiting on the table
        self.next_item = 0.0
        self.history = deque()  # (clock, puck, angle, speed, mallets, item) for the goal replay
        self.replay, self.replay_t, self.replay_start = [], [], 0.0
        self.reset_puck()
        self._bg_key = None

    def start(self, mode="classic", side="left"):
        """Start a game in `mode`. For "computer", the computer plays the half opposite `side`."""
        self.mode = mode
        self.ai = {"computer": {other(side)}, "attract": {"left", "right"}}.get(mode, set())
        self.ai_aim, self.ai_err = {}, {}
        self.mode_since = self.clock
        self.reset_game()

    def reset_game(self):
        """Start a new game immediately in the current mode (skips the start screen)."""
        self.scores = {"left": 0, "right": 0}
        self.started = True
        self.first = False
        self.flash = None
        if self.mode != "attract":  # someone started this (spin, Textport, key): they're here
            self.seen = self.clock
        self.effects = {"left": {}, "right": {}}
        self.item, self.next_item = None, self.clock + POWER_EVERY
        self.replay = []
        self.serve()

    def stop_game(self):
        """Back to the start screen with the mode squares."""
        self.scores = {"left": 0, "right": 0}
        self.started = False
        self.first = True
        self.flash = None
        self.mode, self.ai = "classic", set()
        self.effects = {"left": {}, "right": {}}
        self.item, self.replay = None, []
        self.reset_puck()

    def reset_puck(self):
        """Puck still, at centre ice."""
        self.puck, self.v = (W / 2, H / 2), (0.0, 0.0)
        self.w = self.angle = 0.0  # spin (rad/s) and orientation
        self.live = True
        self.trail.clear()
        self.history.clear()
        self.still_since = self.clock

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
            if p.moved():
                self.seen = self.clock
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
        self._idle()
        if self.started:
            self._powers()
            for p in self.players:
                if self.has(p.side, "frozen"):
                    p.pos, p.vel = p.prev, (0.0, 0.0)
                elif p.side in self.ai:
                    self._ai_move(p, dt)
        if not self.started:
            self._start_screen()
        elif not self.live:
            if self.clock >= self.pause_until:  # goal pause and replay are over
                self.flash, self.replay = None, []
                if max(self.scores.values()) >= WIN_SCORE:
                    self.started = False
                    self.reset_puck()
                else:
                    self.serve(self.serve_side)
        else:
            self._physics(dt)
            if self.live:
                self._unstall()
                self._collect()
                self._record()
                self.trail.append(self.puck)
        for p in self.players:
            p.prev = p.pos
        self.rings = [r for r in self.rings if self.clock - r[2] < RING_TIME]

    def _idle(self):
        """Attract mode when nobody is around; out of it as soon as a tracker moves."""
        if self.mode == "attract":
            if self.seen > self.mode_since or not self.started:  # someone's here, or that game ended
                self.stop_game()
        elif not self.started and self.clock - self.seen > ATTRACT_AFTER:
            self.start("attract")
        elif self.started and self.clock - self.seen > ABANDON_AFTER:
            self.stop_game()

    def _start_screen(self):
        """Stand on a mode square and spin to play that mode."""
        for p in self.players:
            on = next((m for m, off in MODE_SQUARES.items()
                       if math.dist(p.pos, square_spot(p.side, off)) <= MODE_RADIUS), None)
            if on is None:
                p.spin = 0.0
            elif p.spin >= SPIN_TO_START:
                p.spin = 0.0
                self.start(on, p.side)
                break

    def _unstall(self):
        """A puck left still for STALL_TIME drifts back towards the centre, so a game can't
        get stuck with the puck in a corner where no mallet can push it out."""
        if math.hypot(*self.v) >= STALL_SPEED:
            self.still_since = self.clock
        elif self.clock - self.still_since > STALL_TIME:
            dx, dy = W / 2 - self.puck[0], H / 2 - self.puck[1]
            n = math.hypot(dx, dy) or 1.0
            self.v = (dx / n * STALL_PUSH, dy / n * STALL_PUSH)
            self.still_since = self.clock

    # ---------- computer player ----------

    def _ai_move(self, p, dt):
        """Guard the goal while the puck is away; get behind it and strike when it's in our half."""
        me, toward = p.prev, (1.0 if p.side == "left" else -1.0)
        bx, by = self.puck
        ours = self.live and (bx - W / 2) * toward < 0  # puck in our half
        if ours != self.ai_err.get(p.side, (None,))[0]:  # it came over (or left): new error
            self.ai_err[p.side] = (ours, random.uniform(-AI_ERROR, AI_ERROR))
        if ours and self.v[0] * toward > -AI_INCOMING:  # in our half and not a shot at us: play it
            gx = W if p.side == "left" else 0.0
            ax, ay = gx - bx, H / 2 - by
            n = math.hypot(ax, ay) or 1.0
            ax, ay = ax / n, ay / n  # from the puck towards their goal
            back = PUCK_R + p.r + 1.0
            raw = (bx - ax * back, by - ay * back)
            behind = clamp(raw, p.side, p.r)  # by the rails, as close behind as we can get
            ux, uy = bx - me[0], by - me[1]
            n = math.hypot(ux, uy) or 1.0
            ux, uy = ux / n, uy / n  # from us towards the puck
            their_way = ux * ax + uy * ay > 0.3  # hitting it from here sends it towards their goal
            if abs(behind[0] - raw[0]) > 1.0:
                # too close to our end rail to get behind: any hit from the wrong angle only goes into
                # the boards or our own goal. Hit it if the angle is good, else keep clear on the
                # centre side and wait for it to drift out (see _unstall)
                if their_way and n < back + 3:
                    target = (bx + ux * 4, by + uy * 4)
                else:  # wait beside the line it will drift along, so it drifts past us
                    dx, dy = W / 2 - bx, H / 2 - by
                    c = math.hypot(dx, dy) or 1.0
                    off = back + 4
                    spots = [(bx - dy / c * off * k, by + dx / c * off * k) for k in (1.0, -1.0)]
                    target = min(spots, key=lambda q: math.dist(q, clamp(q, p.side, p.r)))
            elif (bx - me[0]) * ax + (by - me[1]) * ay < -back * 0.5:
                # on the wrong side: go round it on the side we're on, not through it
                side = 1.0 if (me[0] - bx) * -ay + (me[1] - by) * ax > 0 else -1.0
                target = (behind[0] - ay * side * back * 1.5, behind[1] + ax * side * back * 1.5)
            elif math.dist(me, behind) < 3.0 and their_way:
                target = (bx + ux * 4, by + uy * 4)  # in position: strike through it
            else:
                target = behind
        else:
            gx = 0.0 if p.side == "left" else W
            gh = self.goal_half(p.side)
            gy = by + self.ai_err[p.side][1]
            target = (gx + toward * (RAIL + p.r + 6), H / 2 + min(max(gy - H / 2, -gh), gh))
        aim = self.ai_aim.get(p.side, target)
        k = min(1.0, dt / AI_REACTION) if AI_REACTION > 0 else 1.0
        aim = (aim[0] + (target[0] - aim[0]) * k, aim[1] + (target[1] - aim[1]) * k)
        self.ai_aim[p.side] = aim
        d = math.dist(me, aim)
        f = min(1.0, AI_SPEED * dt / d) if d > 1e-9 else 0.0
        p.pos = clamp((me[0] + (aim[0] - me[0]) * f, me[1] + (aim[1] - me[1]) * f), p.side, p.r)
        p.vel = ((p.pos[0] - me[0]) / dt, (p.pos[1] - me[1]) / dt) if dt > 0 else (0.0, 0.0)

    # ---------- power-ups ----------

    def has(self, side, kind):
        return self.effects[side].get(kind, -1.0) > self.clock

    def active(self, side):
        return tuple(k for k in POWER_LABEL if self.has(side, k))

    def goal_half(self, side):
        """Half the mouth of `side`'s goal; the opponent's "small_goal" shrinks it."""
        return GOAL_HALF * (SMALL_GOAL if self.has(other(side), "small_goal") else 1.0)

    def posts(self):
        gl, gr = self.goal_half("left"), self.goal_half("right")
        return [(RAIL, H / 2 - gl), (RAIL, H / 2 + gl), (W - RAIL, H / 2 - gr), (W - RAIL, H / 2 + gr)]

    def _powers(self):
        """Mallet sizes from "big", and a new power-up on the table when it's time."""
        for p in self.players:
            p.r = MALLET_R * (BIG if self.has(p.side, "big") else 1.0)
            p.pos = clamp(p.pos, p.side, p.r)
        if self.mode == "powerups" and self.item is None and self.live and self.clock >= self.next_item:
            for _ in range(20):  # somewhere in either half, not on top of the puck
                x = random.uniform(12, W / 2 - 8) + random.choice((0.0, W / 2 - 4))
                spot = (x, random.uniform(8, H - 8))
                if math.dist(spot, self.puck) > 10:
                    break
            self.item = (random.choice(POWERS), spot)

    def _collect(self):
        if self.item is None:
            return
        kind, spot = self.item
        for p in self.players:
            if math.dist(p.pos, spot) < p.r + POWER_R:
                if kind == "freeze":
                    self.effects[other(p.side)]["frozen"] = self.clock + FREEZE_TIME
                else:
                    self.effects[p.side][kind] = self.clock + POWER_TIME
                self.item, self.next_item = None, self.clock + POWER_EVERY
                return

    # ---------- goal replay ----------

    def _record(self):
        """Keep the last REPLAY_TIME seconds of play."""
        mallets = [(p.pos, p.r, p.color, self.active(p.side)) for p in self.players]
        self.history.append((self.clock, self.puck, self.angle, math.hypot(*self.v), mallets, self.item))
        while self.history[0][0] < self.clock - REPLAY_TIME:
            self.history.popleft()

    def replaying(self):
        return bool(self.replay) and self.clock >= self.replay_start

    def _replay_frame(self):
        """(frame, trail points) for the moment of the replay we're at now."""
        t = self.replay_t[0] + (self.clock - self.replay_start) * REPLAY_SPEED
        i = max(0, min(len(self.replay) - 1, bisect.bisect_right(self.replay_t, t) - 1))
        return self.replay[i], [f[1] for f in self.replay[max(0, i - TRAIL + 1):i]]

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
            mallets = [((p.prev[0] + (p.pos[0] - p.prev[0]) * f, p.prev[1] + (p.pos[1] - p.prev[1]) * f),
                        p.vel, p.r, self.has(p.side, "power_hit")) for p in self.players]
            self._substep(h, mallets)
            if self._check_goal():
                return

    def _substep(self, h, mallets):
        k = math.exp(-AIR_DRAG * h)
        self.v = (self.v[0] * k, self.v[1] * k)
        self.w *= math.exp(-SPIN_DRAG * h)
        self.puck = (self.puck[0] + self.v[0] * h, self.puck[1] + self.v[1] * h)
        self.angle += self.w * h
        for c, cv, r, power in mallets:
            if self._hit_circle(c, r, cv, E_MALLET, "hit") and power:
                self.v = (self.v[0] * POWER_HIT, self.v[1] * POWER_HIT)
        self._hit_walls()
        for post in self.posts():
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
        # end rails, except in front of a goal mouth (there only the posts)
        lo, hi = RAIL + PUCK_R, W - RAIL - PUCK_R
        mouth = abs(self.puck[1] - H / 2)
        if self.puck[0] < lo and mouth >= self.goal_half("left"):
            self._contact((1.0, 0.0), lo - self.puck[0], (0.0, 0.0), E_WALL, "wall")
        elif self.puck[0] > hi and mouth >= self.goal_half("right"):
            self._contact((-1.0, 0.0), self.puck[0] - hi, (0.0, 0.0), E_WALL, "wall")

    def _hit_circle(self, c, r, cv, e, kind):
        """True if the puck bounced off the circle."""
        dx, dy = self.puck[0] - c[0], self.puck[1] - c[1]
        d = math.hypot(dx, dy)
        reach = PUCK_R + r
        if d >= reach:
            return False
        n = (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)
        return self._contact(n, reach - d, cv, e, kind)

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
            return False  # already separating
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
        return True

    def _check_goal(self):
        x, y = self.puck
        mouth = abs(y - H / 2)
        if x < RAIL and mouth < self.goal_half("left"):
            scorer = "right"
        elif x > W - RAIL and mouth < self.goal_half("right"):
            scorer = "left"
        else:
            return False
        self.scores[scorer] += 1
        self.events.append(("goal", scorer))
        # "GOAL!", then the replay, then the serve (or, for the winning goal, the result)
        self._record()
        self.live = False
        self.flash = scorer
        self.serve_side = other(scorer)
        self.replay = list(self.history)
        self.replay_t = [f[0] for f in self.replay]
        self.replay_start = self.clock + GOAL_FLASH
        span = self.replay_t[-1] - self.replay_t[0]
        self.pause_until = max(self.clock + GOAL_PAUSE, self.replay_start + span / REPLAY_SPEED)
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
        for gx, side in ((0, "left"), (W - RAIL, "right")):  # goal slots in the end rails
            gh = self.goal_half(side)
            cv2.rectangle(img, px(gx, H / 2 - gh), px(gx + RAIL, H / 2 + gh), rgba(SLOT), -1)
        for post in self.posts():
            cv2.circle(img, px(*post), int(0.6 * PX), rgba((150, 155, 165)), -1, cv2.LINE_AA)

        hw = img.shape[1] // 2
        for side, x0 in (("left", 0), ("right", hw)):
            # draw upright on a transparent layer in half-local coords, rotate, composite
            layer = np.zeros((img.shape[0], hw, 4), np.uint8)
            cx, cy = W / 4, H / 2
            text(layer, f"{self.scores['left']} : {self.scores['right']}", (cx, cy - 19), 5, rgba(DARK))
            if self.first:
                text(layer, "Pick a mode: stand on it and spin!", (cx, cy - 12), 3, rgba(GREEN))
            elif self.flash:
                word = "REPLAY" if self.replaying() else "GOAL!"
                text(layer, word, (cx, cy - 12), 7, rgba(PLAYER_C[self.flash]))
            elif not self.started:
                for s in ("left", "right"):
                    if self.scores[s] >= WIN_SCORE:
                        who = "Computer" if s in self.ai else ("Player 1" if s == "left" else "Player 2")
                        text(layer, f"{who} Wins!", (cx, cy - 12), 5, rgba(PLAYER_C[s]))
            elif self.mode == "attract":
                text(layer, "Grab a mallet to play!", (cx, cy - 12), 4, rgba(GREEN))
            elif side in self.ai:
                text(layer, "Computer", (cx, cy - 12), 3, rgba(DARK))
            for i, k in enumerate(self.active(side)):
                text(layer, POWER_LABEL[k], (cx, cy + 12 + 4 * i), 3, rgba(POWER_C[k]))
            if not self.started:  # one square per mode, across the player's view
                for mode, off in MODE_SQUARES.items():
                    cv2.rectangle(layer, px(cx + off - 6, cy - 6), px(cx + off + 6, cy + 6), rgba(BLACK), -1)
                    text(layer, MODE_LABEL[mode], (cx + off, cy), 2, rgba(WHITE), max_w=11)
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
        replaying = self.replaying()
        key = (self.scores["left"], self.scores["right"], self.started, self.first, self.flash, replaying,
               self.mode, tuple(sorted(self.ai)), self.goal_half("left"), self.goal_half("right"),
               self.active("left"), self.active("right"))
        if key != self._bg_key:
            self._bg_key, self._bg = key, self._draw_background()
        img = self._bg.copy()
        if replaying:  # the recorded play, at REPLAY_SPEED
            (_, puck, angle, speed, mallets, item), trail = self._replay_frame()
        else:
            for x, y, t0, s in self.rings:
                a = (self.clock - t0) / RING_TIME
                r = PUCK_R * 0.5 + (2 + 6 * s) * a
                cv2.circle(img, px(x, y), int(r * PX), mix(IMPACT, SURFACE, (1 - a) * (0.4 + 0.6 * s)),
                           max(1, int(0.3 * PX)), cv2.LINE_AA)
            puck = self.puck if self.live else None
            angle, speed, item = self.angle, math.hypot(*self.v), self.item
            trail = list(self.trail)[:-1] if self.live else []
            mallets = [(p.pos, p.r, p.color, self.active(p.side)) for p in self.players]
        if item:
            draw_item(img, *item)
        if speed > 30:
            for i, (x, y) in enumerate(trail):
                f = (i + 1) / (len(trail) + 1)
                cv2.circle(img, px(x, y), int(PUCK_R * PX * (0.4 + 0.5 * f)),
                           mix(PUCK_C, SURFACE, 0.3 * f * min(1.0, speed / 100)), -1, cv2.LINE_AA)
        for pos, r, color, effects in mallets:
            draw_mallet(img, pos, color, r)
            for i, k in enumerate(effects):  # one coloured ring per active power-up
                cv2.circle(img, px(*pos), int((r + 0.8 + 0.9 * i) * PX), mix(POWER_C[k], SURFACE, 0.85),
                           max(1, int(0.4 * PX)), cv2.LINE_AA)
        if puck:
            draw_puck(img, puck, angle)
        return img

    def frame(self):
        """render() flipped for TouchDesigner, whose numpy origin is bottom-left."""
        return np.ascontiguousarray(self.render()[::-1])

    def close(self):
        for p in self.players:
            p.close()


def shadow(img, pos, r):
    cv2.circle(img, px(pos[0] + 0.5, pos[1] + 0.7), int(r * PX), mix(BLACK, SURFACE, 0.22), -1, cv2.LINE_AA)


def draw_mallet(img, pos, color, radius=MALLET_R):
    shadow(img, pos, radius)
    c, r = px(*pos), radius * PX
    cv2.circle(img, c, int(r), mix(BLACK, color, 0.4), -1, cv2.LINE_AA)  # rim
    cv2.circle(img, c, int(r * 0.85), rgba(color), -1, cv2.LINE_AA)  # body
    cv2.circle(img, c, int(r * 0.5), mix(BLACK, color, 0.25), -1, cv2.LINE_AA)  # handle base
    cv2.circle(img, c, int(r * 0.4), mix(WHITE, color, 0.35), -1, cv2.LINE_AA)  # handle
    cv2.circle(img, px(pos[0] - 0.15 * radius, pos[1] - 0.15 * radius), int(r * 0.12),
               mix(WHITE, color, 0.8), -1, cv2.LINE_AA)  # highlight


def draw_item(img, kind, pos):
    """A power-up: coloured disc with a white symbol that reads the same from either end."""
    shadow(img, pos, POWER_R)
    x, y = pos
    cv2.circle(img, px(x, y), int(POWER_R * PX), rgba(POWER_C[kind]), -1, cv2.LINE_AA)
    w, s = max(1, int(0.35 * PX)), POWER_R * 0.55
    if kind == "big":  # ring
        cv2.circle(img, px(x, y), int(s * PX), rgba(WHITE), w, cv2.LINE_AA)
    elif kind == "small_goal":  # dot
        cv2.circle(img, px(x, y), int(s * 0.45 * PX), rgba(WHITE), -1, cv2.LINE_AA)
    elif kind == "power_hit":  # plus
        cv2.line(img, px(x - s, y), px(x + s, y), rgba(WHITE), w, cv2.LINE_AA)
        cv2.line(img, px(x, y - s), px(x, y + s), rgba(WHITE), w, cv2.LINE_AA)
    else:  # freeze: snowflake
        for a in (0, math.pi / 3, 2 * math.pi / 3):
            dx, dy = math.cos(a) * s, math.sin(a) * s
            cv2.line(img, px(x - dx, y - dy), px(x + dx, y + dy), rgba(WHITE), w, cv2.LINE_AA)


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

    def run(game, secs, until=lambda: False, dt=1 / 60):
        """Step for up to secs; stop early once until() is true."""
        for _ in range(round(secs / dt)):
            game.step(dt)
            if until():
                return True
        return False

    g.seen = math.inf  # these checks are about play: keep attract mode out of the way

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
    # goal: scores, "GOAL!", replay of the shot at half speed, then the puck is placed still
    # on the conceding side
    still(left, (10, 60))
    g.reset_puck()
    place((40, H / 2), (-60, 0))
    run(g, 1.0, lambda: not g.live)
    assert g.scores["right"] == 1 and not g.live and g.flash == "right"
    span = g.replay_t[-1] - g.replay_t[0]
    assert 0.3 < span <= REPLAY_TIME and g.replay[0][1][0] > g.replay[-1][1][0]  # starts upfield, ends in the goal
    assert not g.replaying()
    g.step(GOAL_FLASH + 0.01)
    assert g.replaying() and near(g._replay_frame()[0][1], g.replay[0][1])  # replay starts at the beginning
    g.render()
    g.step(span / REPLAY_SPEED * 0.5)
    mid = g._replay_frame()[0][0] - g.replay_t[0]
    assert abs(mid - span / 2) < 0.05, mid  # half way through the replay = half way through the play
    g.step(g.pause_until - g.clock + 0.01)
    assert g.live and g.flash is None and not g.replay and g.puck == (W / 2 - 15, H / 2) and g.v == (0, 0)
    # posts: a puck aimed at a post doesn't score
    place((10, H / 2 + GOAL_HALF), (-80, 0))
    g.step(0.3)
    assert g.scores["right"] == 1 and g.v[0] > 0, (g.scores, g.v)
    # winning goal: replay first, then the result
    g.scores["right"] = WIN_SCORE - 1
    place((10, H / 2), (-80, 0))
    g.step(0.2)
    assert g.started and g.scores["right"] == WIN_SCORE and g.replay
    g.step(g.pause_until - g.clock + 0.01)
    assert not g.started and not g.first
    g.render()
    # sound cues: one per kind per frame, loudest wins, soft touches silent, no machine-gun repeats
    g.events = [("hit", 30.0), ("hit", 120.0), ("wall", 1.0), ("goal", "left")]
    assert sorted(g._sound_cues()) == [("goal", 1.0), ("hit", 1.0)], g._sound_cues()
    g.events = [("hit", 120.0)]
    assert g._sound_cues() == []  # same clock -> within SOUND_GAP
    g.clock += SOUND_GAP + 1e-9  # float: clock + gap - clock can land just under gap
    g.events = [("hit", 30.0)]
    assert g._sound_cues() == [("hit", 0.5)]
    # stop_game -> start screen
    g.stop_game()
    assert not g.started and g.first and g.scores == {"left": 0, "right": 0}

    # start screen: each mode square starts its mode, and is drawn where the code looks for it
    for p in g.players:
        still(p, START_SPOT[p.side] if p.side == "right" else (8, 8))
    img = g.render()
    for side in ("left", "right"):
        for off in MODE_SQUARES.values():
            sx, sy = square_spot(side, off)
            for cx, cy in ((sx + 5, sy + 5), (sx - 5, sy + 5), (sx + 5, sy - 5)):
                assert tuple(img[int(cy * PX), int(cx * PX)][:3]) == BLACK, (side, off)
            assert tuple(img[int(sy * PX), int((sx + 8) * PX)][:3]) != BLACK
    for mode, off in MODE_SQUARES.items():
        g.stop_game()
        left.rot = left.spin = 0.0
        spot = square_spot("left", off)
        msg(left, *spot); msg(left, *spot, 1.0)  # stand on it and spin
        g.step(0.016)
        assert g.started and g.mode == mode, (mode, g.mode)
        assert g.ai == ({"right"} if mode == "computer" else set())
    g.stop_game()

    # computer player (AI_ERROR off, so it plays its best): blocks a shot at its goal ...
    AI_ERROR, ai_error = 0.0, AI_ERROR
    c = Game(bind=False)
    cl, cr = c.players
    c.seen = math.inf
    c.start("computer", "left")
    assert c.ai == {"right"}
    still(cl, (10, 60))
    c.puck, c.v, c.w = (80, H / 2), (70, 0), 0.0
    run(c, 2.0, lambda: c.v[0] < 0)
    assert c.v[0] < 0 and c.scores["left"] == 0, (c.puck, c.v, c.scores)
    # ... and clears a puck sitting in its half towards the other goal, without scoring on itself
    for spot in ((105, 20), (100, 50), (120, 34)):
        c.reset_puck()
        c.puck = spot
        run(c, 3.0, lambda: c.puck[0] < W / 2)
        assert c.puck[0] < W / 2 and c.scores["left"] == 0, (spot, c.puck, c.scores)
    AI_ERROR = ai_error

    # attract: nobody moves for ATTRACT_AFTER -> the computer plays both sides
    a = Game(bind=False)
    al = a.players[0]
    t = 0.0
    while t < ATTRACT_AFTER + 1:
        t += 0.1
        a.update(t)
    assert a.started and a.mode == "attract" and a.ai == {"left", "right"}
    a.render()
    puck0 = a.puck
    for _ in range(300):
        t += 1 / 60
        a.update(t)
    assert math.dist(a.puck, puck0) > 5  # the game is being played
    # ... until a tracker moves: back to the start screen
    msg(al, 20, 20)
    a.update(t + 0.016)
    assert not a.started and a.first and a.mode == "classic" and not a.ai
    # starting a game by hand (Textport) while attract is running sticks, even with no tracker moving
    for _ in range(int((ABANDON_AFTER + 5) / 0.1)):
        t += 0.1
        a.update(t)
    assert a.mode == "attract"  # nobody has moved for over ABANDON_AFTER
    a.start("classic")
    for _ in range(30):
        t += 1 / 60
        a.update(t)
    assert a.mode == "classic" and a.started, a.mode
    # an attract game that reaches WIN_SCORE starts over instead of sitting on the result
    a.start("attract")
    a.scores["left"] = WIN_SCORE - 1
    a.puck, a.v = (W - 8, H / 2), (120, 0)
    a.players[1].pos = a.players[1].prev = (W - 10, 8)
    while a.live:
        a.step(1 / 60)
    assert a.scores["left"] == WIN_SCORE
    a.step(a.pause_until - a.clock + 0.01)
    a.step(1 / 60)
    assert a.mode == "classic" and not a.started  # back to the start screen ...
    assert run(a, ATTRACT_AFTER + 1, lambda: a.mode == "attract")  # ... and attract again
    assert a.started and a.scores == {"left": 0, "right": 0}
    # a game nobody is playing any more ends after ABANDON_AFTER, and attract takes over
    a.start("classic")
    t0 = t
    while t < t0 + ABANDON_AFTER + 1:
        t += 0.1
        a.update(t)
    assert a.mode == "attract", a.mode

    # power-ups
    pg = Game(bind=False)
    pl, pr = pg.players
    pg.seen = math.inf
    pg.start("powerups")
    pg.reset_puck()
    pg.next_item = math.inf  # place them by hand
    still(pl, (30, 30)); still(pr, (110, 30))
    # big: collected by touching it, mallet grows, wears off after POWER_TIME
    pg.item = ("big", (31, 31))
    pg.step(1 / 60); pg.step(1 / 60)
    assert pg.item is None and pl.r == MALLET_R * BIG and pr.r == MALLET_R
    pg.render()
    pg.clock += POWER_TIME
    pg.step(1 / 60)
    assert pl.r == MALLET_R and pg.active("left") == ()
    # small_goal: the opponent's goal shrinks, so a shot that used to score bounces
    pg.item = ("small_goal", (110, 30))
    pg.step(1 / 60)
    assert pg.goal_half("left") == GOAL_HALF * SMALL_GOAL and pg.goal_half("right") == GOAL_HALF
    still(pl, (40, 60))
    pg.puck, pg.v = (10, H / 2 + 8), (-80, 0)
    pg.step(0.3)
    assert pg.scores["right"] == 0 and pg.v[0] > 0, (pg.scores, pg.v)
    pg.clock += POWER_TIME
    # power_hit: the same hit sends the puck POWER_HIT times faster
    def launch():
        pg.reset_puck()
        pg.puck = (40, 30)
        pl.prev, pl.pos, pl.vel = (33, 30), (34, 30), (60, 0)
        pg.step(1 / 60)
        return pg.v[0]
    plain = launch()
    pg.effects["left"]["power_hit"] = pg.clock + POWER_TIME
    assert abs(launch() - min(plain * POWER_HIT, MAX_PUCK_SPEED)) < 2, (plain, pg.v)
    pg.clock += POWER_TIME
    # freeze: the opponent's mallet stops following its tracker for FREEZE_TIME
    pg.reset_puck()
    still(pl, (30, 30)); still(pr, (100, 30))
    pg.item = ("freeze", (30, 30))
    pg.step(1 / 60)
    assert pg.has("right", "frozen")
    msg(pr, 110, 40)
    pg.step(1 / 60)
    assert pr.pos == (100, 30)
    pg.clock += FREEZE_TIME
    msg(pr, 110, 40)
    pg.step(1 / 60)
    assert near(pr.pos, (110, 40))
    # a new one appears on the table when it's time
    pg.next_item = pg.clock
    pg.step(1 / 60)
    kind, (ix, iy) = pg.item
    assert kind in POWERS and 0 < ix < W and 0 < iy < H
    pg.render()

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
