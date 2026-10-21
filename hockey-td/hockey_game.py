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
SPIN_TO_START = 90.0  # rotation needed on a mode square, in heading radians x 100 (90 ~ 50 degrees)
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

# COMPUTER PLAYER, per difficulty (picked after spinning on "vs Computer"):
#   speed     top mallet speed (people manage ~250)
#   reaction  s, how far its aim lags behind where it wants to be
#   error     how far off its guarding can be; picked again each time the puck comes over
#   incoming  a puck coming at its goal faster than this gets goalkeeping, not a strike
#   predict   guard where the puck WILL cross (bounces included), not where it is now
#   banks     also consider bank shots off the side walls, away from the opponent's mallet
#   precision how close to the spot behind the puck it must be before it strikes (smaller = more
#             accurate shots, but fewer of them; 1.4 scored most in simulated games)
AI_LEVELS = {
    "easy": dict(speed=45.0, reaction=0.25, error=10.0, incoming=20.0, predict=False, banks=False, precision=3.0),
    "medium": dict(speed=70.0, reaction=0.15, error=7.0, incoming=20.0, predict=False, banks=False, precision=3.0),
    "hard": dict(speed=220.0, reaction=0.02, error=0.5, incoming=20.0, predict=True, banks=True, precision=1.4),
}
CAREFUL_SPEED = 70.0  # top speed going round the puck or near its own end rail, so it doesn't clip it in
LEVEL_SQUARES = {"easy": -20.0, "medium": 0.0, "hard": 20.0}  # replace the mode squares while choosing
CHOOSE_TIMEOUT = 15.0  # s to pick a difficulty before the mode squares come back
CHOOSE_GRACE = 1.5  # s after "vs Computer" during which spinning doesn't pick (you're still spinning)

# SHOTS AND RALLIES
KMH = 0.025 * 3.6  # table units/s -> km/h (1 unit is about 2.5 cm)
RALLY_SHOW = 5  # show the rally counter from this many hits
HIT_GAP = 0.25  # s: the same mallet touching the puck again within this is still the same hit
HIT_MIN = 5.0  # impulse below this is a nudge, not a hit

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

# PAUSES (during a game)
EXIT_SPIN = 360.0  # degrees of turning, within SPIN_WINDOW, that opens "Exit to menu?"
SPIN_WINDOW = 3.0  # s
EXIT_CONFIRM = 5.0  # s to spin again and exit; otherwise the game carries on
LOST_AFTER = 0.5  # s with no data from a tracked mallet -> paused until it's back
RESUME_DELAY = 2.0  # s of "Get ready!" once it's back

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


def bounce(v, w, n, vo, e):
    """Puck velocity v and spin w after touching a body moving at vo, n pointing from that body to
    the puck: restitution e plus Coulomb rim friction (what makes glancing hits spin the puck).
    Returns (v, w, normal impulse); the impulse is 0 if they're already separating."""
    nx, ny = n
    # velocity of the puck's rim at the contact point (r = -n R), relative to the other body
    rx, ry = -nx * PUCK_R, -ny * PUCK_R
    vrx = v[0] - w * ry - vo[0]
    vry = v[1] + w * rx - vo[1]
    vn = vrx * nx + vry * ny
    if vn >= 0:
        return v, w, 0.0
    jn = -(1 + e) * vn  # normal impulse per unit puck mass
    tx, ty = -ny, nx
    vt = vrx * tx + vry * ty
    # tangential impulse: 3 = 1/m + R^2/I for a solid disc; capped by Coulomb friction
    jt = max(-FRICTION * jn, min(FRICTION * jn, -vt / 3))
    return (v[0] + jn * nx + jt * tx, v[1] + jn * ny + jt * ty), w - 2 * jt / PUCK_R, jn  # I = R^2 / 2


def seg_dist(p, a, b):
    """Distance from point p to the segment a-b."""
    ax, ay = b[0] - a[0], b[1] - a[1]
    L = ax * ax + ay * ay
    t = 0.0 if L == 0 else min(1.0, max(0.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L))
    return math.dist(p, (a[0] + ax * t, a[1] + ay * t))


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
        self.heading = None  # last heading from the tracker (radians)
        self.turn = 0.0  # total turning so far (radians, +- by direction, no wrap-around)
        self.spin = 0.0  # turning on a mode square (either direction, heading radians x 100)
        self.fresh = False  # a tracker message moved the mallet since the game last looked
        self.last_msg = None  # game wall time of the last such message
        self.present = False  # tracked during this game, so losing it pauses the game
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
        h = float(nums[3])
        if self.heading is not None:
            d = (h - self.heading + math.pi) % (2 * math.pi) - math.pi  # +-180 degree wrap is no turn
            self.turn += d
            self.spin += abs(d) * 100
        self.heading = h
        for name, t in self.tables.items():  # first table the point is over
            x, y = to_table(float(nums[0]), float(nums[1]), t)
            if -GLITCH <= x <= W + GLITCH and -GLITCH <= y <= H + GLITCH:
                self.table = name
                break
        else:  # between tables, or a glitch
            return False
        self.raw = self.pos = clamp((x, y), self.side, self.r)
        self.fresh = True
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
        self.clock = 0.0  # game time, advances with step() but stands still while paused
        self.wall = 0.0  # advances with step() always
        self.pause = None  # None, "exit" (Exit to menu?), "lost" (waiting for a mallet) or "ready"
        self.pause_side, self.pause_end = None, 0.0
        self.turns = {}  # side -> deque of (wall, turn) over the last SPIN_WINDOW
        self.started_wall = 0.0
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
        self.ai_shot = {}  # side -> shot the computer is lining up (index into its candidates)
        self.level = "medium"  # computer difficulty
        self.choosing = None  # (side, wall time it started) while that player picks a difficulty
        self.rally, self.last_hit = 0, (None, -1.0)  # hits since the serve; (side, clock) of the last
        self.shot = None  # (side, km/h) of the last hit
        self.goal_kmh, self.new_record = None, False  # shown with "GOAL!"
        self.best_shot, self.best_rally = 0.0, 0  # this session, people only (not the computer)
        self.seen = 0.0  # clock when a tracker last moved
        self.mode_since = 0.0  # clock when the current mode started
        self.effects = {"left": {}, "right": {}}  # side -> {power-up kind: clock it wears off}
        self.item = None  # (kind, (x, y)): power-up waiting on the table
        self.next_item = 0.0
        self.history = deque()  # (clock, puck, angle, speed, mallets, item) for the goal replay
        self.replay, self.replay_t, self.replay_start = [], [], 0.0
        self.reset_puck()
        self._bg_key = None

    def start(self, mode="classic", side="left", level="medium"):
        """Start a game in `mode`. For "computer", the computer plays the half opposite `side`
        at `level` ("easy", "medium" or "hard")."""
        self.mode, self.level, self.choosing = mode, level, None
        self.ai_shot = {}
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
        self.pause, self.turns, self.started_wall = None, {}, self.wall
        for p in self.players:  # only mallets being tracked now can pause the game by going missing
            p.present = p.last_msg is not None and self.wall - p.last_msg <= LOST_AFTER
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
        self.pause, self.choosing = None, None
        self.reset_puck()

    def reset_puck(self):
        """Puck still, at centre ice."""
        self.puck, self.v = (W / 2, H / 2), (0.0, 0.0)
        self.w = self.angle = 0.0  # spin (rad/s) and orientation
        self.live = True
        self.trail.clear()
        self.history.clear()
        self.still_since = self.clock
        self.rally, self.last_hit, self.shot = 0, (None, -1.0), None

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
        self.wall += dt
        for p in self.players:
            if p.fresh:
                p.fresh, p.last_msg = False, self.wall
                p.present = p.present or self.started
        if self._paused():  # everything stands still: puck, timers, replay
            for p in self.players:
                p.prev = p.pos  # no sweep through the puck when play resumes
            return
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

    def _paused(self):
        """Run the pauses; True while the game is paused.
        A tracked mallet that goes missing pauses until it's back (then "Get ready!").
        A full spin of a mallet asks "Exit to menu?"; another spin exits, else play carries on."""
        if not self.started or self.mode == "attract":
            self.pause = None
            return False
        humans = [p for p in self.players if p.side not in self.ai]
        lost = [p for p in humans if p.present and self.wall - p.last_msg > LOST_AFTER]
        spun = next((p for p in humans if self._full_spin(p)), None)
        if lost:
            self.pause, self.pause_side = "lost", lost[0].side
        elif self.pause == "lost":
            self.pause, self.pause_end = "ready", self.wall + RESUME_DELAY
        elif self.pause == "exit":
            if spun:
                self.stop_game()
                return False
            if self.wall >= self.pause_end:
                self.pause = None
        elif self.pause == "ready":
            if self.wall >= self.pause_end:
                self.pause = None
        elif spun:
            self.pause, self.pause_side, self.pause_end = "exit", spun.side, self.wall + EXIT_CONFIRM
        return self.pause is not None

    def _full_spin(self, p):
        """True when p has turned EXIT_SPIN within SPIN_WINDOW (then it starts counting afresh).
        Not right after the game starts, so the spin that picked the mode doesn't count."""
        if self.wall - self.started_wall < SPIN_WINDOW:
            self.turns.pop(p.side, None)
            return False
        log = self.turns.setdefault(p.side, deque())
        log.append((self.wall, p.turn))
        while log[0][0] < self.wall - SPIN_WINDOW:
            log.popleft()
        if max(abs(p.turn - t) for _, t in log) >= math.radians(EXIT_SPIN):
            log.clear()
            return True
        return False

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
        """Stand on a mode square and spin to play that mode. "vs Computer" first turns that
        player's squares into Easy / Medium / Hard."""
        if self.choosing and self.wall - self.choosing[1] > CHOOSE_TIMEOUT:
            self.choosing = None
        for p in self.players:
            choosing = self.choosing and self.choosing[0] == p.side
            squares = LEVEL_SQUARES if choosing else MODE_SQUARES
            on = next((m for m, off in squares.items()
                       if math.dist(p.pos, square_spot(p.side, off)) <= MODE_RADIUS), None)
            if on is None or (choosing and self.wall - self.choosing[1] < CHOOSE_GRACE):
                p.spin = 0.0
            elif p.spin >= SPIN_TO_START:
                p.spin = 0.0
                if choosing:
                    self.start("computer", p.side, on)
                elif on == "computer":
                    self.choosing = (p.side, self.wall)
                else:
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
        lv = AI_LEVELS[self.level]
        speed = lv["speed"]
        me, toward = p.prev, (1.0 if p.side == "left" else -1.0)
        bx, by = self.puck
        ours = self.live and (bx - W / 2) * toward < 0  # puck in our half
        if ours != self.ai_err.get(p.side, (None,))[0]:  # it came over (or left): new error
            self.ai_err[p.side] = (ours, random.uniform(-lv["error"], lv["error"]))
        if ours and self.v[0] * toward > -lv["incoming"]:  # in our half and not a shot at us: play it
            ax, ay = self._shot_aim(p, lv["banks"])  # from the puck towards where we'll send it
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
                speed = min(speed, CAREFUL_SPEED)  # close to our own goal: no wild moves
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
                speed = min(speed, CAREFUL_SPEED)  # don't clip it on the way round
            elif math.dist(me, behind) < lv["precision"] and their_way:
                target = (bx + ux * 4, by + uy * 4)  # in position: strike through it
            else:
                target = behind
        else:
            gx = 0.0 if p.side == "left" else W
            gh = self.goal_half(p.side)
            line = gx + toward * (RAIL + p.r + 6)  # where we guard, in front of the goal
            gy = by
            if lv["predict"] and self.live and self.v[0] * toward < -1.0:  # coming our way
                gy = self._crossing(line)
            gy += self.ai_err[p.side][1]
            target = (line, H / 2 + min(max(gy - H / 2, -gh), gh))
        aim = self.ai_aim.get(p.side, target)
        k = min(1.0, dt / lv["reaction"]) if lv["reaction"] > 0 else 1.0
        aim = (aim[0] + (target[0] - aim[0]) * k, aim[1] + (target[1] - aim[1]) * k)
        self.ai_aim[p.side] = aim
        d = math.dist(me, aim)
        f = min(1.0, speed * dt / d) if d > 1e-9 else 0.0
        p.pos = clamp((me[0] + (aim[0] - me[0]) * f, me[1] + (aim[1] - me[1]) * f), p.side, p.r)
        p.vel = ((p.pos[0] - me[0]) / dt, (p.pos[1] - me[1]) / dt) if dt > 0 else (0.0, 0.0)

    def _crossing(self, x_line, horizon=2.0, h=1 / 120):
        """y where the puck will cross x = x_line, following its drag and the side-wall bounces
        (mallets and posts ignored). The current y if it won't get there within `horizon` s."""
        (x, y), v, w = self.puck, self.v, self.w
        lo, hi = RAIL + PUCK_R, H - RAIL - PUCK_R
        k, kw = math.exp(-AIR_DRAG * h), math.exp(-SPIN_DRAG * h)
        for _ in range(int(horizon / h)):  # the same steps as _substep, walls only
            v, w = (v[0] * k, v[1] * k), w * kw
            nx, ny = x + v[0] * h, y + v[1] * h
            if (x - x_line) * (nx - x_line) <= 0:
                return y + (ny - y) * ((x_line - x) / (nx - x) if nx != x else 0.0)
            x, y = nx, ny
            if y < lo:
                y = lo
                v, w, _ = bounce(v, w, (0.0, 1.0), (0.0, 0.0), E_WALL)
            elif y > hi:
                y = hi
                v, w, _ = bounce(v, w, (0.0, -1.0), (0.0, 0.0), E_WALL)
        return self.puck[1]

    def _shot_aim(self, p, banks):
        """Unit vector from the puck along the shot we're lining up: straight at their goal, or
        (with banks) off the top or bottom wall, whichever path keeps furthest from their mallet."""
        bx, by = self.puck
        gx = W if p.side == "left" else 0.0
        goal = (gx, H / 2)
        paths = [[(bx, by), goal]]
        if banks:
            for wall in (RAIL + PUCK_R, H - RAIL - PUCK_R):
                # the goal mirrored in that wall, aim there to bank into it; the bounce keeps only
                # E_WALL of the sideways speed, so the image sits that much further out
                my = wall - (H / 2 - wall) / E_WALL
                if (wall - by) * (my - by) > 0:  # the wall is between us and the mirror image
                    t = (wall - by) / (my - by)
                    paths.append([(bx, by), (bx + (gx - bx) * t, wall), goal])
        opp = next(q.pos for q in self.players if q.side != p.side)
        clear = [min(seg_dist(opp, a, b) for a, b in zip(path, path[1:])) for path in paths]
        best = max(range(len(paths)), key=clear.__getitem__)
        keep = self.ai_shot.get(p.side)
        if keep is not None and keep < len(paths) and clear[keep] > clear[best] - 3.0:
            best = keep  # don't flip between shots over small differences
        self.ai_shot[p.side] = best
        tx, ty = paths[best][1]
        n = math.hypot(tx - bx, ty - by) or 1.0
        return (tx - bx) / n, (ty - by) / n

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
                        p.vel, p.r, self.has(p.side, "power_hit"), p.side) for p in self.players]
            self._substep(h, mallets)
            if self._check_goal():
                return

    def _substep(self, h, mallets):
        k = math.exp(-AIR_DRAG * h)
        self.v = (self.v[0] * k, self.v[1] * k)
        self.w *= math.exp(-SPIN_DRAG * h)
        self.puck = (self.puck[0] + self.v[0] * h, self.puck[1] + self.v[1] * h)
        self.angle += self.w * h
        for c, cv, r, power, side in mallets:
            jn = self._hit_circle(c, r, cv, E_MALLET, "hit")
            if jn:
                if power:
                    self.v = (self.v[0] * POWER_HIT, self.v[1] * POWER_HIT)
                self._hit(side, jn)
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
        """The impulse if the puck bounced off the circle, else 0 (so: true if it bounced)."""
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
        self.v, self.w, jn = bounce(self.v, self.w, n, vo, e)
        if not jn:
            return 0.0  # already separating
        self.events.append((kind, jn))
        if jn > RING_MIN:
            self.rings.append((self.puck[0] - nx * PUCK_R, self.puck[1] - ny * PUCK_R, self.clock,
                               min(jn / 120, 1.0)))
        return jn

    def _hit(self, side, jn):
        """A mallet hit the puck: count the rally (once per hit, not per contact) and the shot speed."""
        if jn < HIT_MIN:
            return
        who, t = self.last_hit
        if who != side or self.clock - t > HIT_GAP:
            self.rally += 1
        self.last_hit = (side, self.clock)
        self.shot = (side, min(math.hypot(*self.v), MAX_PUCK_SPEED) * KMH)

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
        # shot speed for the goal screen, and the session records (people only)
        self.goal_kmh = self.shot[1] if self.shot else math.hypot(*self.v) * KMH
        new = []
        if self.mode != "attract":
            if self.shot and self.shot[0] == scorer and scorer not in self.ai and self.goal_kmh > self.best_shot:
                self.best_shot = self.goal_kmh
                new.append("Fastest shot")
            if self.rally > self.best_rally:
                self.best_rally = self.rally
                new.append("Longest rally")
        self.new_record = " + ".join(new) if new else False
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
            choosing = not self.started and self.choosing and self.choosing[0] == side
            text(layer, f"{self.scores['left']} : {self.scores['right']}", (cx, cy - 19), 5, rgba(DARK))
            if self.pause:
                who = "Player 1" if self.pause_side == "left" else "Player 2"
                big, small = {"exit": ("Exit to menu?", "Spin again to exit"),
                              "lost": (f"Waiting for {who}", "Bring the mallet back to the table"),
                              "ready": ("Get ready!", "")}[self.pause]
                text(layer, big, (cx, cy - 12), 4, rgba(DARK))
                if small:
                    text(layer, small, (cx, cy - 6), 2.5, rgba(DARK))
            elif choosing:
                text(layer, "Pick a difficulty: stand on it and spin!", (cx, cy - 12), 3, rgba(GREEN))
            elif self.first:
                text(layer, "Pick a mode: stand on it and spin!", (cx, cy - 12), 3, rgba(GREEN))
            elif self.flash:
                word = "REPLAY" if self.replaying() else "GOAL!"
                text(layer, word, (cx, cy - 12), 7, rgba(PLAYER_C[self.flash]))
                if self.goal_kmh is not None:
                    text(layer, f"{self.goal_kmh:.1f} km/h", (cx, cy - 5), 2.5, rgba(DARK))
                if self.new_record:
                    text(layer, f"NEW RECORD! {self.new_record}", (cx, cy - 1), 2.5, rgba(GREEN))
            elif not self.started:
                for s in ("left", "right"):
                    if self.scores[s] >= WIN_SCORE:
                        who = "Computer" if s in self.ai else ("Player 1" if s == "left" else "Player 2")
                        text(layer, f"{who} Wins!", (cx, cy - 12), 5, rgba(PLAYER_C[s]))
            elif self.mode == "attract":
                text(layer, "Grab a mallet to play!", (cx, cy - 12), 4, rgba(GREEN))
            elif side in self.ai:
                text(layer, f"Computer ({self.level.capitalize()})", (cx, cy - 12), 3, rgba(DARK))
            for i, k in enumerate(self.active(side)):
                text(layer, POWER_LABEL[k], (cx, cy + 12 + 4 * i), 3, rgba(POWER_C[k]))
            if not self.started:  # one square per mode (or difficulty), across the player's view
                squares = LEVEL_SQUARES if choosing else MODE_SQUARES
                for name, off in squares.items():
                    fill = LINE_RED if name == "hard" else BLACK
                    cv2.rectangle(layer, px(cx + off - 6, cy - 6), px(cx + off + 6, cy + 6), rgba(fill), -1)
                    label = name.capitalize() if choosing else MODE_LABEL[name]
                    text(layer, label, (cx + off, cy), 2, rgba(WHITE), max_w=11)
                if self.best_shot or self.best_rally:
                    text(layer, f"Fastest shot {self.best_shot:.1f} km/h   Longest rally {self.best_rally}",
                         (cx, cy + 11), 2, rgba(DARK))
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
        key = (self.pause, self.pause_side,
               self.scores["left"], self.scores["right"], self.started, self.first, self.flash, replaying,
               self.mode, tuple(sorted(self.ai)), self.goal_half("left"), self.goal_half("right"),
               self.active("left"), self.active("right"), self.choosing and self.choosing[0], self.level,
               self.goal_kmh, self.new_record, self.best_shot, self.best_rally)
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
        if self.started and self.live and not self.pause and self.rally >= RALLY_SHOW:
            for side in ("left", "right"):
                stamp(img, f"Rally {self.rally}", side, (0, -6), 3, rgba(DARK))
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


def stamp(img, s, side, offset, height, color):
    """Text that changes every frame, drawn straight onto the frame, upright for `side`'s player.
    offset is (across their view, towards them) from the centre of their half, as in the background
    layer. Redrawing the whole background for it would take ~100x longer."""
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(height * PX / 12))
    (tw, th), _ = cv2.getTextSize(s, font, 1.0, thick)
    scale = height * PX / th
    (tw, th), base = cv2.getTextSize(s, font, scale, thick)
    n = max(tw, th + base) + 4
    patch = np.zeros((n, n, 4), np.uint8)
    cv2.putText(patch, s, (n // 2 - tw // 2, n // 2 + th // 2), font, scale, color, thick, cv2.LINE_AA)
    if FACING[side]:
        m = cv2.getRotationMatrix2D((n / 2, n / 2), FACING[side], 1)
        patch = cv2.warpAffine(patch, m, (n, n))
    a = math.radians(FACING[side])
    dx, dy = offset
    x = (W / 4 if side == "left" else 3 * W / 4) + math.cos(a) * dx + math.sin(a) * dy
    y = H / 2 - math.sin(a) * dx + math.cos(a) * dy
    x0, y0 = int(x * PX) - n // 2, int(y * PX) - n // 2
    sx, sy = max(0, -x0), max(0, -y0)  # clip to the image
    ex, ey = min(n, img.shape[1] - x0), min(n, img.shape[0] - y0)
    if ex <= sx or ey <= sy:
        return
    patch = patch[sy:ey, sx:ex]
    region = img[y0 + sy:y0 + ey, x0 + sx:x0 + ex]
    inv = 255 - patch[..., 3:].astype(np.uint16)  # premultiplied, as in _draw_background
    region[:] = patch + (region * inv // 255).astype(np.uint8)


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
    msg(left, *START_SPOT["left"], 1.0)  # 1 radian of turning -> spin 100
    g.step(0.016)
    assert g.started and not g.first
    left.present = right.present = False  # from here on the checks move the mallets by hand, not by tracker
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
        left.heading, left.spin = None, 0.0
        spot = square_spot("left", off)
        msg(left, *spot); msg(left, *spot, 1.0)  # stand on it and spin
        g.step(0.016)
        if mode == "computer":  # first the squares turn into Easy / Medium / Hard
            assert not g.started and g.choosing[0] == "left"
            img = g.render()
            hx, hy = square_spot("left", LEVEL_SQUARES["hard"])
            assert tuple(img[int((hy + 5) * PX), int((hx + 5) * PX)][:3]) == LINE_RED  # Hard is red
            msg(left, *spot, 2.0)  # still spinning from picking "vs Computer": doesn't pick yet
            g.step(0.016)
            assert not g.started
            g.step(CHOOSE_GRACE)
            spot = square_spot("left", LEVEL_SQUARES["hard"])
            msg(left, *spot, 2.0); msg(left, *spot, 3.0)
            g.step(0.016)
            assert g.level == "hard"
        assert g.started and g.mode == mode, (mode, g.mode)
        assert g.ai == ({"right"} if mode == "computer" else set())
    # nobody picks a difficulty: the mode squares come back
    g.stop_game()
    left.heading, left.spin = None, 0.0
    spot = square_spot("left", MODE_SQUARES["computer"])
    msg(left, *spot); msg(left, *spot, 1.0)
    g.step(0.016)
    assert g.choosing
    still(left, (8, 8))
    g.step(CHOOSE_TIMEOUT + 0.1); g.step(0.016)
    assert g.choosing is None and not g.started
    g.stop_game()

    # computer player (medium, aiming error off): blocks a shot at its goal ...
    ai_error, AI_LEVELS["medium"]["error"] = AI_LEVELS["medium"]["error"], 0.0
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
    AI_LEVELS["medium"]["error"] = ai_error

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

    # heading wrap-around (+180 -> -180 degrees) is a tiny turn, not a spin
    wp = Player("left", 0, bind=False)
    msg(wp, 20, 20, 3.1); msg(wp, 20, 20, -3.1)
    assert wp.spin < 10 and abs(wp.turn) < 0.1, (wp.spin, wp.turn)

    # pauses: both mallets tracked (the bridge streams every frame)
    q = Game(bind=False)
    ql, qr = q.players
    q.seen = math.inf
    still(ql, (30, 30)); still(qr, (110, 30))

    def feed(p, turn=0.0):  # one tracker message for p where it is, turned a bit more
        msg(p, *p.pos, (p.heading or 0.0) + turn)

    def play(secs, spin=0.0, left_tracked=True):  # trackers streaming; left turns `spin` radians
        n = round(secs * 60)
        for _ in range(n):
            if left_tracked:
                feed(ql, spin / n)
            feed(qr)
            q.step(1 / 60)

    play(0.1)
    q.start("classic")
    full = math.radians(EXIT_SPIN) + 0.2
    # the spin that started the game doesn't count, even if it carries on past a full turn
    play(1.0, full)
    play(SPIN_WINDOW)
    assert q.pause is None
    # small turns while playing don't count either
    play(2.0, full * 0.8)
    assert q.pause is None
    # a full spin: "Exit to menu?", and the game stands still
    play(1.0, full)
    assert q.pause == "exit" and q.pause_side == "left"
    c0, puck0 = q.clock, q.puck
    q.render()
    play(1.0)
    assert q.clock == c0 and q.puck == puck0
    # nothing more within EXIT_CONFIRM: play carries on
    play(EXIT_CONFIRM)
    assert q.pause is None and q.started and q.clock > c0
    # spin, then spin again: back to the start screen
    play(1.0, full)
    assert q.pause == "exit"
    play(1.0, full)
    assert not q.started and q.first and q.pause is None
    # a tracked mallet going missing pauses until it's back, then "Get ready!"
    q.start("classic")
    play(0.5)
    c0 = q.clock
    play(1.0, left_tracked=False)
    assert q.pause == "lost" and q.pause_side == "left" and q.clock - c0 < LOST_AFTER + 0.05
    q.render()
    play(0.1)
    assert q.pause == "ready"
    play(RESUME_DELAY + 0.05)
    assert q.pause is None and q.started
    # a mallet that wasn't tracked when the game started doesn't hold the game up
    q.start("classic")
    for _ in range(120):
        feed(qr)
        q.step(1 / 60)
    q.start("classic")  # left last seen 2 s ago: not part of this game
    for _ in range(60):
        feed(qr)
        q.step(1 / 60)
    assert q.pause is None
    # nor does the computer's half (whatever its tracker does), but the human's does
    play(0.1)
    q.start("computer", "left")
    for _ in range(60):
        feed(ql)
        q.step(1 / 60)
    assert q.pause is None
    play(1.0, left_tracked=False)
    assert q.pause == "lost" and q.pause_side == "left"

    # shots, rallies and records
    sg = Game(bind=False)
    sl, sr = sg.players
    sg.seen = math.inf
    sg.start("classic")
    sg.reset_puck()
    still(sl, (40, 30)); still(sr, (100, 30))
    sg.puck, sg.v = (50, 30), (-60, 0)  # into the left mallet ...
    run(sg, 0.5, lambda: sg.rally == 1)
    assert sg.rally == 1 and sg.shot[0] == "left" and sg.shot[1] > 0
    run(sg, 2.0, lambda: sg.rally == 2)  # ... back across into the right one
    assert sg.rally == 2 and sg.shot[0] == "right"
    # pushing the puck along for a while is one hit, not one per frame
    sg.reset_puck()
    sg.puck = (45, 30)
    for i in range(30):
        sl.prev, sl.pos, sl.vel = (40 + i * 0.5, 30), (40.5 + i * 0.5, 30), (30, 0)
        sg.step(1 / 60)
    assert sg.rally == 1, sg.rally
    # a goal shows the shot's speed; a person's shot sets the record, the computer's doesn't
    sg.reset_puck()
    still(sr, (100, 60))
    sg.puck = (60, H / 2)
    sl.prev, sl.pos, sl.vel = (50, H / 2), (56, H / 2), (120, 0)  # moving into it this frame
    run(sg, 2.0, lambda: not sg.live)
    assert sg.scores["left"] == 1 and sg.goal_kmh > 10 and sg.best_shot == sg.goal_kmh
    assert sg.new_record and "Fastest shot" in sg.new_record
    assert abs(sg.goal_kmh - sg.shot[1]) < 1e-9 and sg.goal_kmh <= MAX_PUCK_SPEED * KMH
    sg.render()
    best = sg.best_shot
    sg.start("computer", "right", "hard")  # computer on the left
    sg.reset_puck()
    still(sr, (100, 60))
    sg.puck, sg.v = (W / 2 - 3, H / 2), (0, 0)
    run(sg, 20.0, lambda: not sg.live)
    assert sg.scores["left"] == 1 and sg.best_shot == best  # the computer scored: no record
    # rally counter drawn live (no background redraw) from RALLY_SHOW hits
    sg.start("classic")
    sg.rally = RALLY_SHOW - 1
    quiet = sg.render()
    sg.rally = RALLY_SHOW
    assert (sg.render() != quiet).any() and sg._bg is not None
    # records on the start screen
    sg.stop_game()
    sg.render()

    # hard computer: predicts where the puck crosses, bounces included ...
    hg = Game(bind=False)
    hg.seen = math.inf
    hg.start("classic")
    for p in hg.players:
        still(p, (8, 8) if p.side == "left" else (W - 8, 8))
    hg.puck, hg.v = (40, 15), (90, 70)  # will bounce off the bottom wall on the way
    guess = hg._crossing(120.0)
    while hg.puck[0] < 120:
        hg.step(1 / 120)
    assert abs(hg.puck[1] - guess) < 1.0, (hg.puck, guess)
    # ... and saves fast straight shots and bank shots
    for i, (y0, ty) in enumerate(((10, 30), (55, 40), (20, -28), (50, 96))):  # last two: off a wall
        hg = Game(bind=False)
        hg.seen = math.inf
        hg.start("computer", "left", "hard")
        still(hg.players[0], (10, 60))
        hg.reset_puck()
        n = math.hypot(W - 60, ty - y0)
        hg.puck, hg.v = (60, y0), ((W - 60) / n * 160, (ty - y0) / n * 160)
        run(hg, 3.0, lambda: not hg.live)
        assert hg.scores["left"] == 0, i

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
    pg.next_item = math.inf  # collecting one scheduled the next
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
    pg.next_item = math.inf  # collecting one scheduled the next
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
    pg.next_item = math.inf  # collecting one scheduled the next
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
    pg.next_item = math.inf
    msg(pr, 110, 40)
    pg.step(1 / 60)
    assert near(pr.pos, (110, 40))
    # a new one appears on the table when it's time (mallets parked in corners, out of its reach)
    still(pl, (RAIL + MALLET_R, RAIL + MALLET_R)); still(pr, (W - RAIL - MALLET_R, H - RAIL - MALLET_R))
    pr.present = False
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
