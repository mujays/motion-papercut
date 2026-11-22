"""Procedural sound-effect library (numpy only, no internet, no sample files needed).
All functions return float32 mono arrays at SR (48 kHz), peak ~1.0.
"""
import numpy as np

SR = 48000
_rng = np.random.default_rng(7)


def _t(d):
    return np.arange(int(d * SR)) / SR


def _norm(x, peak=1.0):
    m = np.max(np.abs(x)) + 1e-9
    return (x / m * peak).astype(np.float32)


def _fft_filter(x, lo=None, hi=None):
    """Brick-ish band filter with soft edges via FFT."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4)
    if hi:
        g *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(X * g, n)


def _sweep_filter(x, lo0, lo1, hi0, hi1, chunks=24):
    """Time-varying band filter (overlap-add of filtered chunks)."""
    n = len(x)
    out = np.zeros(n)
    size = n // chunks + 1
    win_len = size * 2
    w = np.hanning(win_len)
    for i in range(-1, chunks + 1):
        a = i * size - size // 2
        seg = np.zeros(win_len)
        s0, s1 = max(a, 0), min(a + win_len, n)
        if s1 <= s0:
            continue
        seg[s0 - a:s1 - a] = x[s0:s1]
        p = min(max((i + 0.5) / chunks, 0), 1)
        lo = lo0 + (lo1 - lo0) * p
        hi = hi0 + (hi1 - hi0) * p
        y = _fft_filter(seg * w, lo, hi)
        out[s0:s1] += y[s0 - a:s1 - a]
    return out


def _env(n, attack, release, curve=4.0):
    a = int(attack * SR)
    e = np.ones(n)
    if a > 0:
        e[:a] = np.linspace(0, 1, a) ** 1.5
    r = n - a
    if r > 0:
        e[a:] = np.exp(-curve * np.linspace(0, 1, r) * (len(e[a:]) / SR) / max(release, 1e-3))
    return e


def pop():
    t = _t(0.16)
    f = 950 * np.exp(-t * 22) + 330
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 32)
    y[:120] += _rng.normal(0, 0.4, 120) * np.linspace(1, 0, 120)
    return _norm(y)


def tick():
    t = _t(0.05)
    y = np.sin(2 * np.pi * 3200 * t) * np.exp(-t * 140)
    y += _fft_filter(_rng.normal(0, 1, len(t)), 3000, None) * np.exp(-t * 300) * 0.5
    return _norm(y, 0.8)


def typing(n=5, gap=0.045):
    out = np.zeros(int((n * gap + 0.06) * SR))
    for i in range(n):
        s = tick() * (0.6 + 0.4 * _rng.random())
        a = int(i * gap * SR)
        out[a:a + len(s)] += s
    return _norm(out, 0.8)


def whoosh(d=0.55, up=True):
    n = int(d * SR)
    x = _rng.normal(0, 1, n)
    if up:
        y = _sweep_filter(x, 200, 900, 900, 5000)
    else:
        y = _sweep_filter(x, 900, 200, 5000, 900)
    t = np.linspace(0, 1, n)
    e = np.sin(np.pi * t ** 0.8) ** 2
    return _norm(y * e)


def swish():
    return whoosh(0.28, True)


def ding(f0=1318.5, d=1.1):
    t = _t(d)
    y = (np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(2 * np.pi * 2 * f0 * t) * np.exp(-t * 6)
         + 0.12 * np.sin(2 * np.pi * 3.01 * f0 * t) * np.exp(-t * 9))
    y *= np.exp(-t * 4.5)
    y[:200] *= np.linspace(0, 1, 200)
    return _norm(y, 0.9)


def chime():
    a = ding(1318.5, 1.2)
    b = ding(1975.5, 1.2)
    out = np.zeros(len(a) + int(0.11 * SR))
    out[:len(a)] += a
    out[int(0.11 * SR):int(0.11 * SR) + len(b)] += b * 0.8
    return _norm(out, 0.9)


def sparkle():
    out = np.zeros(int(0.6 * SR))
    for i, f in enumerate([2093, 2637, 3136, 4186]):
        t = _t(0.35)
        y = np.sin(2 * np.pi * f * t) * np.exp(-t * 14)
        a = int(i * 0.05 * SR)
        out[a:a + len(y)] += y * (1 - i * 0.12)
    return _norm(out, 0.7)


def bubble():
    out = np.zeros(int(0.35 * SR))
    for i in range(3):
        t = _t(0.09)
        f = 280 + 900 * (t / t[-1]) ** 1.5 + i * 80
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1])
        a = int(i * 0.08 * SR)
        out[a:a + len(y)] += y
    return _norm(out, 0.8)


def thunder(d=3.2):
    n = int(d * SR)
    white = _rng.normal(0, 1, n)
    brown = np.cumsum(white)
    brown -= np.convolve(brown, np.ones(4000) / 4000, mode="same")
    rumble = _fft_filter(brown, 25, 260)
    rumble = rumble / (np.max(np.abs(rumble)) + 1e-9)
    t = np.linspace(0, d, n)
    mod = np.interp(t, np.linspace(0, d, 40), 0.5 + _rng.random(40))
    env = (1 - np.exp(-t * 18)) * np.exp(-t * 1.3)
    crack = _fft_filter(white, 1500, 9000) * np.exp(-t * 28) * 0.6
    return _norm(rumble * env * mod + crack)


def boom(d=1.4):
    t = _t(d)
    f = 85 * np.exp(-t * 2.5) + 35
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    nz = _fft_filter(_rng.normal(0, 1, len(t)), 30, 400) * np.exp(-t * 6)
    nz /= np.max(np.abs(nz)) + 1e-9
    return _norm(y + 0.5 * nz)


def riser(d=1.0):
    n = int(d * SR)
    t = np.linspace(0, 1, n)
    nz = _sweep_filter(_rng.normal(0, 1, n), 300, 2500, 1500, 9000)
    nz /= np.max(np.abs(nz)) + 1e-9
    f = 220 + 900 * t ** 2
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.35
    env = t ** 2 * (1 - np.clip((t - 0.95) / 0.05, 0, 1))
    return _norm((nz + tone) * env)


def blip():
    t = _t(0.09)
    y = np.sin(2 * np.pi * (700 + 600 * t / t[-1]) * t) * np.exp(-t * 30)
    return _norm(y, 0.8)


def rise_tone(d=0.7):
    t = _t(d)
    f = 400 + 500 * t / d
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 0.5 * 0.7
    return _norm(y, 0.6)


def rain_loop(d):
    n = int(d * SR)
    x = _rng.normal(0, 1, n)
    y = _fft_filter(x, 900, 7000)
    drops = np.zeros(n)
    idx = _rng.integers(0, n - 400, int(d * 25))
    for i in idx:
        drops[i:i + 400] += np.sin(np.linspace(0, 40, 400)) * np.exp(-np.linspace(0, 8, 400)) * _rng.random()
    y = y / (np.max(np.abs(y)) + 1e-9) + 0.3 * drops
    fade = int(0.8 * SR)
    if n > 2 * fade:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return _norm(y, 0.5)


def wind_loop(d):
    n = int(d * SR)
    y = _sweep_filter(_rng.normal(0, 1, n), 150, 250, 600, 900, chunks=8)
    t = np.linspace(0, d, n)
    y *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * t)
    fade = int(0.8 * SR)
    if n > 2 * fade:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return _norm(y, 0.5)


LIB = {
    "pop": pop, "tick": tick, "typing": typing, "whoosh": whoosh, "swish": swish,
    "ding": ding, "chime": chime, "sparkle": sparkle, "bubble": bubble,
    "thunder": thunder, "boom": boom, "riser": riser, "blip": blip, "rise": rise_tone,
}
LOOPS = {"rain": rain_loop, "wind": wind_loop}

# default gain (dB) of each effect relative to a normalized voice track
GAIN_DB = {
    "pop": -17, "tick": -22, "typing": -21, "whoosh": -19, "swish": -20,
    "ding": -19, "chime": -18, "sparkle": -21, "bubble": -19,
    "thunder": -9, "boom": -11, "riser": -18, "blip": -20, "rise": -21,
    "rain": -27, "wind": -28,
}

_cache = {}


def get(name):
    if name not in _cache:
        _cache[name] = LIB[name]().astype(np.float32)
    return _cache[name]
