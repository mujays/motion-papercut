"""Procedural background music + paper sound effects (numpy only, no internet, no sample files).

music.bed(duration, mood) -> float32 mono 48 kHz, peak ~0.9. Moods: playful (default), calm, curious, upbeat.
Instruments: soft plucks (additive, string-like), music-box glockenspiel melody, round bass, felt pad,
light shaker + soft kick. Always ends on the home chord with a gentle fade.
"""
import numpy as np

import sfx as BASE

SR = 48000
_rng = np.random.default_rng(5)


def _t(d):
    return np.arange(int(d * SR)) / SR


def _norm(x, peak=1.0):
    return (x / (np.max(np.abs(x)) + 1e-9) * peak).astype(np.float32)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# ----------------------------------------------------------------------------- instruments
_cache = {}


def pluck(f, d=0.9):
    k = ("p", round(f, 2), d)
    if k not in _cache:
        t = _t(d)
        y = np.zeros_like(t)
        for h in range(1, 9):
            y += np.sin(2 * np.pi * f * h * t + h) * (0.9 / h ** 1.1) * np.exp(-t * (3.2 + h * 2.2))
        y += _rng.normal(0, 1, len(t)) * np.exp(-t * 180) * 0.08
        y[:96] *= np.linspace(0, 1, 96)
        _cache[k] = y.astype(np.float32)
    return _cache[k]


def glock(f, d=1.4):
    k = ("g", round(f, 2), d)
    if k not in _cache:
        t = _t(d)
        y = np.sin(2 * np.pi * f * t) * np.exp(-t * 3.0) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 9)
        y += 0.12 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 16)
        y[:60] *= np.linspace(0, 1, 60)
        _cache[k] = y.astype(np.float32)
    return _cache[k]


def bass(f, d):
    t = _t(d)
    env = np.minimum(1, t / 0.02) * np.exp(-t * 1.6)
    y = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)) * env
    rel = min(len(t), int(0.06 * SR))
    y[-rel:] *= np.linspace(1, 0, rel)
    return y.astype(np.float32)


def pad(freqs, d):
    t = _t(d)
    y = np.zeros_like(t)
    for f in freqs:
        for det in (-0.6, 0.6):
            ph = 2 * np.pi * (f + det) * t
            y += (2 / np.pi) * np.arcsin(np.sin(ph)) * 0.5
    y = BASE._fft_filter(y, None, 1400)
    a = min(len(t), int(0.5 * SR))
    env = np.ones_like(t)
    env[:a] = np.linspace(0, 1, a)
    env[-a:] *= np.linspace(1, 0, a)
    return (y * env).astype(np.float32)


def shaker():
    if "sh" not in _cache:
        t = _t(0.07)
        y = BASE._fft_filter(_rng.normal(0, 1, len(t)), 5000, 12000) * np.exp(-t * 60)
        _cache["sh"] = _norm(y, 1.0)
    return _cache["sh"]


def kick():
    if "k" not in _cache:
        t = _t(0.3)
        f = 50 + 70 * np.exp(-t * 30)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12)
        _cache["k"] = _norm(y, 1.0)
    return _cache["k"]


# ----------------------------------------------------------------------------- arrangement
MOODS = {
    #           bpm  root  progression (scale degrees as semitone offsets of chord roots + quality)
    "playful": (104, 60, [(0, "M"), (7, "M"), (9, "m"), (5, "M")], True),
    "upbeat": (118, 62, [(0, "M"), (5, "M"), (9, "m"), (7, "M")], True),
    "calm": (80, 57, [(0, "M"), (9, "m"), (5, "M"), (7, "M")], False),
    "curious": (92, 57, [(0, "m"), (8, "M"), (3, "M"), (10, "M")], False),
}
PENTA = {"M": [0, 2, 4, 7, 9], "m": [0, 3, 5, 7, 10]}


def _add(buf, x, at, g):
    a = int(at * SR)
    if a >= len(buf):
        return
    m = min(len(x), len(buf) - a)
    buf[a:a + m] += x[:m] * g


def bed(duration, mood="playful", seed=3):
    bpm, root, prog, drums = MOODS.get(str(mood).lower(), MOODS["playful"])
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * 4
    n = int((duration + 2.0) * SR)
    buf = np.zeros(n, np.float32)
    nbars = int(np.ceil(duration / bar)) + 1
    key_scale = PENTA["M" if prog[0][1] == "M" else "m"]
    # melody motif (2 bars, 8th notes, sparse), repeated with small variations
    motif = []
    deg = 2
    for i in range(16):
        if rng.random() < 0.45:
            deg = int(np.clip(deg + rng.choice([-2, -1, 1, 1, 2]), 0, 7))
            motif.append((i, deg))
    for b in range(nbars):
        t0 = b * bar
        if t0 > duration + 0.5:
            break
        off, q = prog[b % len(prog)]
        r = root + off
        third = 4 if q == "M" else 3
        chord = [r, r + third, r + 7]
        last = t0 + bar > duration - 0.2
        # pad
        _add(buf, pad([midi(c) for c in chord], bar + 0.3), t0, 0.05)
        # bass on 1 and 3
        _add(buf, bass(midi(r - 24), beat * 1.9), t0, 0.32)
        _add(buf, bass(midi(r - 24 + (7 if b % 2 else 0)), beat * 1.9), t0 + 2 * beat, 0.26)
        # plucked arpeggio (8ths)
        arp = [chord[0], chord[1], chord[2], chord[1] + 12, chord[2], chord[1], chord[0] + 12, chord[1]]
        for i, nn in enumerate(arp):
            if last and i > 0:
                break
            _add(buf, pluck(midi(nn)), t0 + i * beat / 2, 0.16 if i % 2 == 0 else 0.11)
        # melody (music box) — from bar 2 on, every other phrase
        if b >= 1 and (b // 2) % 2 == 1 and not last:
            for i, d in motif:
                if (b % 2) * 8 <= i < (b % 2) * 8 + 8:
                    s = key_scale[d % 5] + 12 * (d // 5)
                    _add(buf, glock(midi(root + 12 + s)), t0 + (i % 8) * beat / 2, 0.10)
        if drums and not last:
            for i in range(8):
                _add(buf, shaker(), t0 + i * beat / 2 + 0.004, 0.035 if i % 2 else 0.05)
            _add(buf, kick(), t0, 0.16)
            _add(buf, kick(), t0 + 2 * beat, 0.12)
        if last:
            _add(buf, pluck(midi(root), 2.5), t0, 0.2)
            _add(buf, glock(midi(root + 24), 2.5), t0 + 0.02, 0.12)
            break
    buf = buf[:int(duration * SR)]
    f = min(len(buf), int(1.6 * SR))
    buf[-f:] *= np.linspace(1, 0, f) ** 1.5
    a = min(len(buf), int(0.4 * SR))
    buf[:a] *= np.linspace(0, 1, a)
    return _norm(buf, 0.9)


# ----------------------------------------------------------------------------- paper SFX
def paper():
    """short paper slide / rustle."""
    t = _t(0.26)
    x = _rng.normal(0, 1, len(t))
    y = BASE._fft_filter(x, 1400, 7000)
    grain = (np.abs(_rng.normal(0, 1, len(t) // 240 + 1)) ** 2).repeat(240)[:len(t)]
    env = np.sin(np.pi * np.clip(t / 0.26, 0, 1)) ** 1.5
    return _norm(y * env * (0.5 + grain), 0.9)


def tap():
    """soft card-on-table tap (stickers landing)."""
    t = _t(0.08)
    y = np.sin(2 * np.pi * 620 * t) * np.exp(-t * 70) + 0.6 * BASE._fft_filter(_rng.normal(0, 1, len(t)), 800, 5000) * np.exp(-t * 120)
    return _norm(y, 0.9)


def stamp():
    """paper thump for badges / big stickers."""
    t = _t(0.35)
    f = 70 + 120 * np.exp(-t * 25)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)
    y += 0.5 * BASE._fft_filter(_rng.normal(0, 1, len(t)), 600, 4000) * np.exp(-t * 40)
    return _norm(y, 0.95)


def snip():
    """scissor snip (two quick clicks)."""
    out = np.zeros(int(0.16 * SR), np.float32)
    for k, at in enumerate((0.0, 0.07)):
        t = _t(0.05)
        c = BASE._fft_filter(_rng.normal(0, 1, len(t)), 2500, 9000) * np.exp(-t * 160)
        a = int(at * SR)
        out[a:a + len(c)] += c * (1 - 0.3 * k)
    return _norm(out, 0.8)


def softpop():
    t = _t(0.12)
    f = 700 * np.exp(-t * 26) + 260
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 34)
    return _norm(y, 0.9)


def correct():
    a = BASE.ding(1046.5, 0.5)
    b = BASE.ding(1568.0, 0.8)
    out = np.zeros(len(b) + int(0.09 * SR), np.float32)
    out[:len(a)] += a
    out[int(0.09 * SR):int(0.09 * SR) + len(b)] += b
    return _norm(out, 0.9)


def wrong():
    t = _t(0.38)
    f = 220 - 60 * t / 0.38
    y = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.4 + np.sin(2 * np.pi * np.cumsum(f) / SR)
    y = BASE._fft_filter(y, None, 1800) * np.exp(-t * 5)
    return _norm(y, 0.8)


LIB = dict(BASE.LIB)
LIB.update({"paper": paper, "tap": tap, "stamp": stamp, "snip": snip, "softpop": softpop,
            "correct": correct, "wrong": wrong})
GAIN_DB = dict(BASE.GAIN_DB)
GAIN_DB.update({"paper": -20, "tap": -18, "stamp": -14, "snip": -21, "softpop": -19, "correct": -18,
                "wrong": -20, "pop": -19, "tick": -24})
LOOPS = BASE.LOOPS
_sfx = {}


def get(name):
    if name not in _sfx:
        _sfx[name] = LIB[name]().astype(np.float32)
    return _sfx[name]
