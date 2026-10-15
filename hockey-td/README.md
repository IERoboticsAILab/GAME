# Air hockey, two rooms, two projectors (TouchDesigner)

Port of `../hockey.gaml`. One TouchDesigner host runs the game and renders the full
table. The same full image goes to both projectors over NDI, one projector per room,
so both players see the whole table with both paddles. Each player can only move on
their own half.

## Files
- `hockey_game.py`: game logic, UDP input, drawing (numpy + cv2). Run
  `python3 hockey_game.py` to self-check (it writes `preview.png`).
- `script_top_callbacks.py`: paste into the Script TOP's callbacks DAT.
- `natnet_bridge.py` + `natnet_client.py`: read the players from Motive and send them to the game (run outside TouchDesigner, see below).

## TouchDesigner network
Save your `.toe` in this folder, next to `hockey_game.py`.

```
script1 (Script TOP) ─┬─ [projection mapping, room 1] ─ ndiout_p1 (NDI Out TOP)
                      └─ [projection mapping, room 2] ─ ndiout_p2 (NDI Out TOP)
```
1. Add a Script TOP and paste `script_top_callbacks.py` into its callbacks DAT.
   It outputs 2200×1096 (`PX = 16` pixels per table unit in `hockey_game.py`).
   A Script TOP only cooks when something changes, so add an **Execute DAT** with
   **Frame Start** turned on and this in it, which makes the game update every frame:
   ```python
   def onFrameStart(frame):
       op('script1').cook(force=True)
       return
   ```
2. Feed `script1` into one projection mapping per room (Stoner, Corner Pin or Kantan Mapper).
   Each room gets its own mapping because each floor and projector is different.
3. Add two NDI Out TOPs and set their **Name** parameters to different values, e.g. `hockey_P1` and `hockey_P2`.
   At each site, the NDI receiver feeding the projector picks its stream.

### `r` to reset (optional)
Add a Keyboard In DAT and put this in its callbacks. Keep the default `onKey`
signature your TD build generates, and add only the body:
```python
import hockey_game
def onKey(dat, key, character, alt, lAlt, rAlt, ctrl, lCtrl, rCtrl, shift, lShift, rShift, state, time, cmd, lCmd, rCmd):
    if key == 'r' and state:
        hockey_game.get_game().reset_game()
```

## Tracking input
Start the bridge, then open the `.toe`. On the TD PC, in this folder:
```
python natnet_bridge.py
```
It reads the two rigid bodies from Motive (`10.205.3.3`, NatNet unicast) and sends them to the game on
this PC. It needs any Python 3, nothing to install. Every 5 s it prints each body's rate, or
`NOT TRACKED` / `NOT IN MOTIVE`. Stop it with Ctrl+C. If the game runs on another PC, pass that PC's IP:
`python natnet_bridge.py 10.205.3.218`. Motive must stream NatNet (Enable on, Unicast, Rigid Bodies on).

`BODIES` at the top of `natnet_bridge.py` maps each rigid body to a port: `GAME_Player2` to 9878 and
`GAME_Player3` to 9879. If the mallets appear on the wrong halves, swap the names. A body Motive can't
see is not sent, so its mallet stays put.

The message format and ports are the same as the GAMA version. Trackers send UDP to the TD host:

| Player | Half | Port | Scores into |
|---|---|---|---|
| Player 1 | left  | 9879 | right goal |
| Player 2 | right | 9878 | left goal  |

(Ports are 9876 + the lab's UMH Base ID of 2. Port-to-side matches the old model, where the first port, 9878, got odd id 3, which meant the right half.)

Each message needs at least 4 numbers: `x, y, _, rotation`.

### Tables
Each room's table is an entry in `TABLES`, all in the same Motive space. A tracker plays on
whichever table it is over, so both players can share one table, or each take their UMH to a
different room and play like that. No switch needed: walk over and play. Between tables a
tracker is ignored and its mallet stays put.

Table position is `(x - cx, -(y + cy))` using that table's `correction`, so `cx` is the Motive x
of the table's left edge and `cy` is minus the Motive y of its top edge. Calibrate each
table so a point on that room's floor maps to the same point on its projection (x from 0 to
137.5, y from 0 to 68.5). If an axis runs the wrong way, set `mirror_x` or `mirror_y` for that
table; that flips it in place (`137.5 - x`), the table stays where it is.
The tables must be more than 20 units apart in Motive (`GLITCH`), or a tracker can match the wrong one.

Each player can only move on their own half, set by port: player 1 on x 0 to 68.75, player 2 on x 68.75 to 137.5.

You can test without trackers. With the room1 correction, this puts player 1 at room1 table (20, 30):
```
python3 -c "import socket;socket.socket(2,2).sendto(b'251.6 286.0 0 0',('127.0.0.1',9879))"
```

## Tuning (top of `hockey_game.py`)
Game: `WIN_SCORE`, `SPIN_TO_START`, `GOAL_PAUSE`, `SERVE_SPEED`, `PLAYERS`.
Physics: `AIR_DRAG` (how fast the puck slows), `E_WALL`, `E_MALLET` and `E_POST` (bounciness, 0 to 1),
`FRICTION` (how much glancing hits spin the puck), `MAX_PUCK_SPEED`, `PUCK_R`, `MALLET_R`,
`MALLET_SMOOTH` (lower means steadier mallet velocity from a jittery tracker).
Display: `FACING` (text rotation per half, which assumes each player stands at their goal end; set it to 0 to turn rotation off), `TRAIL`, `PX`, colours.

## Sound
`sounds/` holds `hit.wav`, `wall.wav`, `post.wav` and `goal.wav`, made by `make_sounds.py`.
Copy the `sounds` folder next to the `.toe`. Each frame the game lists which sounds to play and how loud,
based on how hard the hit was, in `get_game().sounds`.

In TouchDesigner:
1. Add four **Audio File In CHOPs** named `snd_hit`, `snd_wall`, `snd_post` and `snd_goal`. On each, set
   **File** to `sounds/hit.wav` (and so on), **Play Mode** to *Sequential*, **Repeat** to *Off*,
   **Volume** to `0` and **Play** to on.
2. Wire all four into a **Math CHOP** and set **Combine CHOPs** to *Add*.
3. Wire the Math CHOP into an **Audio Device Out CHOP** and pick the speakers as its device.
4. Make the Execute DAT's `onFrameStart` look like this:
   ```python
   import hockey_game

   def onFrameStart(frame):
       op('script1').cook(force=True)
       for name, vol in hockey_game.get_game().sounds:
           snd = op('snd_' + name)
           if snd:
               snd.par.volume = vol
               snd.par.cuepulse.pulse()
       return
   ```
Loudness knobs are `SOUND_FULL`, `SOUND_MIN` and `SOUND_GAP` in `hockey_game.py`.

## Textport commands
```python
import hockey_game; hockey_game.get_game().reset_game()   # start playing now
import hockey_game; hockey_game.get_game().stop_game()    # back to the Start squares
```
`get_game().events` lists each frame's hits, wall and post bounces, and goals, which you could use to trigger sounds.

## Getting NDI to the remote sites
NDI discovery only works on one LAN or subnet by default. If the other room is on a different
network, set up NDI Bridge (free, part of NDI Tools) between the sites, or use NDI Discovery Server.
At each site you need something that receives NDI and feeds the projector: NDI Studio Monitor
full-screen on a PC, an NDI decoder box, or a projector with NDI built in.

## Reloading after editing `hockey_game.py`
The UDP ports stay bound while the module is loaded. In the Textport, run:
```python
import hockey_game, importlib; hockey_game.close(); importlib.reload(hockey_game)
```
Restarting TouchDesigner also works.

## Differences from the GAMA version
- Keystone and `calculate_offset` were removed. Projection mapping in TD replaces them.
- Physics is rigid-body: the puck keeps its momentum on an air cushion. Mallet speed is estimated from
  the tracker, so it carries into each hit, and bounces lose energy according to restitution. Glancing
  hits spin the puck through friction. The goal mouths have real posts, and fast shots can't pass through mallets.
  It no longer depends on frame rate.
- After a goal there's a "GOAL!" pause, then the puck is placed on the side that was scored on.
- Mallets that step past the centre line or off the table stop at the edge instead of freezing.
- Each half has its own score and messages, and debug `write` spam is gone.
