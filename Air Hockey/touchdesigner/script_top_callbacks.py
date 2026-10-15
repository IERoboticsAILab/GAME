# Script TOP callbacks. Paste into the Script TOP's callbacks DAT (e.g. script1_callbacks).
# hockey_game.py must sit next to the .toe file.
import sys

if project.folder not in sys.path:
    sys.path.insert(0, project.folder)
import hockey_game


def onSetupParameters(scriptOp):
    return


def onPulse(par):
    return


def onCook(scriptOp):
    game = hockey_game.get_game()
    game.update(absTime.seconds)  # cooked every frame by an Execute DAT (see README)
    scriptOp.copyNumpyArray(game.frame())
