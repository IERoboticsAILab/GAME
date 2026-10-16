# GAME: air hockey

## Run it
1. **Projectors**: in Home Assistant, switch on the projectors **North East** and **South East**.
   Then switch the projectors on with the remote control.
2. **Sound**: in Home Assistant, switch on the sound system and click **USB**.
3. **OptiTrack**: in Home Assistant, switch on OptiTrack. Then in **Motive** (on the Motive PC), make sure
   the rigid bodies **GAME_Player2** and **GAME_Player3** are selected and being tracked.
4. **Bridge** (on the TouchDesigner PC): open a terminal in the `hockey-td` folder and run:
   ```
   python natnet_bridge.py
   ```
   Leave it open.
5. **TouchDesigner**: open the `.toe` file in the `hockey-td` folder.
6. **Play**: stand on a square (Power-ups, Classic, vs Computer) and spin the mallet.

## Textport commands (in TouchDesigner: Alt+T)
```python
import hockey_game; hockey_game.get_game().start("classic")    # or "powerups", "computer", "attract"
import hockey_game; hockey_game.get_game().stop_game()         # back to the start screen
import hockey_game, importlib; hockey_game.close(); importlib.reload(hockey_game)   # after editing hockey_game.py
```

Full details: `hockey-td/README.md`.
