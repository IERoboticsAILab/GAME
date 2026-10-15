"""
Synthesizes the game's sound effects into sounds/*.wav (48 kHz, 16-bit stereo).
Run `python3 make_sounds.py` to regenerate after tweaking; needs numpy.

hit   mallet on puck: hard plastic clack
wall  puck on rail: short thud with a little rattle
post  puck on a goal post: metallic ping
goal  puck dropping into the slot, then a two-tone score buzzer
"""
import os
import wave

import numpy as np

SR = 48000
rng = np.random.default_rng(7)


def tt(d):
    return np.arange(int(SR * d)) / SR


def mode(f, decay, d, amp=1.0):
    """One damped resonance: how struck objects ring."""
    t = tt(d)
    return amp * np.sin(2 * np.pi * f * t) * np.exp(-t / decay)


def burst(decay, d, amp=1.0, bright=True):
    """Noise transient; bright = high-passed (click), else low-passed (thump)."""
    t = tt(d)
    n = rng.standard_normal(len(t))
    if bright:
        n = np.diff(n, prepend=0.0)
    else:
        for i in range(1, len(n)):  # one-pole low-pass
            n[i] = n[i - 1] + 0.08 * (n[i] - n[i - 1])
        n /= np.abs(n).max()
    return amp * n * np.exp(-t / decay)


def at(sig, start, total):
    out = np.zeros(int(SR * total))
    i = int(SR * start)
    out[i:i + len(sig)] += sig[:len(out) - i]
    return out


def tone(f, d, amp=1.0):
    """Soft square wave with click-free attack/release, like an arcade buzzer."""
    t = tt(d)
    env = np.minimum(1, t / 0.005) * np.minimum(1, (d - t) / 0.03)
    return amp * np.tanh(3 * np.sin(2 * np.pi * f * t)) * env


def hit():
    d = 0.15
    return (mode(1850, 0.012, d, 0.9) + mode(3150, 0.008, d, 0.6)
            + mode(420, 0.02, d, 0.5) + burst(0.002, d, 0.5))


def wall():
    d = 0.3
    return (mode(170, 0.07, d) + mode(510, 0.04, d, 0.7) + mode(1150, 0.025, d, 0.3)
            + burst(0.004, d, 0.6, bright=False))


def post():
    d = 0.7
    return (mode(2400, 0.22, d) + mode(3710, 0.15, d, 0.6) + mode(5480, 0.09, d, 0.35)
            + burst(0.0015, d, 0.3))


def goal():
    total = 1.0
    s = at(mode(110, 0.09, 0.4) + mode(240, 0.05, 0.4, 0.6) + burst(0.01, 0.4, 0.5, bright=False), 0, total)
    for i, when in enumerate(np.sort(rng.uniform(0.05, 0.28, 6))):  # puck rattling down the chute
        s += at(mode(rng.uniform(900, 1400), 0.005, 0.05, 0.35 * 0.8 ** i), when, total)
    s += at(tone(660, 0.16, 0.45), 0.35, total)
    s += at(tone(880, 0.4, 0.45), 0.53, total)
    return s


def save(name, sig, peak):
    sig = sig / np.abs(sig).max() * peak
    sig[-int(SR * 0.01):] *= np.linspace(1, 0, int(SR * 0.01))  # no click at the end
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(os.path.join("sounds", name + ".wav"), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(np.repeat(pcm, 2).tobytes())  # same signal left and right


if __name__ == "__main__":
    os.makedirs("sounds", exist_ok=True)
    for name, fn, peak in (("hit", hit, 0.9), ("wall", wall, 0.8), ("post", post, 0.6), ("goal", goal, 0.9)):
        save(name, fn(), peak)
        print("wrote sounds/" + name + ".wav")
