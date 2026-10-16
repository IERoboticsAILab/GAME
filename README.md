# GAME: air hockey

## Run it
1. **Projectors**: switch them on.
2. **Sound**: switch on the sound system and connect it to the TouchDesigner PC with the USB cable.
3. **Motive** (on the Motive PC): open it and check the pucks are tracked.
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
