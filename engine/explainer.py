#!/usr/bin/env python3
"""
EXPLAINER ENGINE — vertical 9:16 flat-vector science explainer videos
(night-navy world, one continuous camera, punchy captions with a yellow keyword,
callout labels, comparison cards, procedural sound effects).

Usage:
  python explainer.py spec.json -o out.mp4                  # full render
  python explainer.py spec.json -o out.mp4 --preview 1,5,9  # PNG stills only (fast check)
  python explainer.py spec.json -o out.mp4 --part 1/4       # render a quarter (video only)
  python explainer.py spec.json -o out.mp4 --finalize 4     # join 4 parts + mix audio
  python explainer.py spec.json -o out.mp4 --timing         # only print word timings

Dependencies: numpy, Pillow, opencv-python (cv2), ffmpeg (system or imageio_ffmpeg).
"""
import argparse, json, math, os, re, shutil, subprocess, sys, wave, hashlib

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sfx as SFX  # noqa: E402
try:  # svg/prop objects only; older installs may not have this file
    import svgkit  # noqa: E402
except ImportError:  # pragma: no cover
    svgkit = None

FONT_DIR = os.path.join(HERE, "fonts")
F_CAPTION = os.path.join(FONT_DIR, "Fredoka-Bold.ttf")
F_TITLE = os.path.join(FONT_DIR, "Montserrat-Black.ttf")
F_TITLE2 = os.path.join(FONT_DIR, "Montserrat-ExtraBold.ttf")
F_LABEL = os.path.join(FONT_DIR, "Rubik-Bold.ttf")

DESIGN_W, DESIGN_H = 720, 1280

# ----------------------------------------------------------------------------- palette
PAL = {
    "yellow": (255, 210, 40), "gold": (255, 200, 80), "white": (245, 247, 255),
    "blue": (52, 140, 225), "lightblue": (130, 195, 255), "sky": (110, 180, 255),
    "navy": (20, 26, 54), "red": (232, 92, 104), "pink": (240, 150, 170),
    "green": (100, 215, 135), "purple": (185, 150, 235), "orange": (240, 130, 70),
    "grey": (150, 155, 180), "dark": (28, 28, 46), "brown": (150, 95, 60),
    "rock": (150, 140, 178), "mountain": (98, 84, 124), "grass": (46, 86, 60),
    "dirt": (46, 41, 68), "sand": (205, 175, 120), "cyan": (90, 220, 230),
}
# Backgrounds: (top, bottom, pattern). Light/dark mode is derived from luminance and drives the UI tokens.
BACKGROUNDS = {
    # dark
    "night": ((22, 32, 66), (46, 66, 126), None),
    "storm": ((30, 36, 68), (52, 64, 112), None),
    "dusk": ((40, 34, 80), (150, 90, 120), None),
    "space": ((6, 8, 22), (22, 24, 56), None),
    "deep": ((14, 30, 70), (24, 56, 120), None),
    "ocean": ((40, 140, 205), (10, 46, 104), None),
    "forest": ((36, 74, 56), (64, 112, 74), None),
    "blueprint": ((28, 70, 140), (22, 56, 118), "grid"),
    "chalkboard": ((42, 72, 60), (32, 58, 48), None),
    "dark": ((24, 26, 34), (38, 42, 56), "dots"),
    "neon": ((26, 14, 50), (60, 20, 90), "grid"),
    # light
    "day": ((96, 170, 235), (200, 230, 250), None),
    "sunset": ((255, 150, 120), (255, 214, 160), None),
    "paper": ((251, 247, 238), (241, 234, 218), None),
    "studio": ((238, 242, 250), (212, 220, 238), None),
    "mint": ((224, 247, 238), (190, 234, 214), None),
    "peach": ((255, 236, 222), (252, 212, 192), None),
    "lavender": ((240, 236, 255), (214, 204, 246), None),
    "sky": ((206, 234, 255), (160, 210, 250), None),
    "lab": ((234, 246, 248), (206, 230, 236), "grid"),
    "notebook": ((252, 252, 246), (244, 244, 234), "lines"),
    "body": ((255, 226, 226), (242, 176, 180), None),
    "desert": ((255, 226, 170), (240, 186, 120), None),
    "kids": ((255, 244, 200), (255, 214, 226), "dots"),
}
BACKGROUNDS["theme"] = BACKGROUNDS["night"]
CAP_HL = (255, 210, 30)
TITLE_HL = (255, 205, 90)

# theme-dependent tokens (filled by set_ui_mode): usable as colour names in specs/SVG
UI = {}


def set_ui_mode(mode, accent=None):
    dark = mode == "dark"
    UI.clear()
    UI.update({
        "mode": mode,
        "outline": (226, 230, 248) if dark else (38, 42, 70),   # object outlines / strokes
        "ink": (245, 247, 255) if dark else (28, 32, 58),       # in-world text, ruler, leader lines
        "line": (230, 235, 250) if dark else (44, 50, 84),
        "pulse": (130, 195, 255) if dark else (52, 120, 220),
        "hl": (255, 210, 40) if dark else (245, 140, 30),        # travel dots, accents
        "accent": col(accent) if accent else ((255, 210, 40) if dark else (240, 110, 40)),
        "title1": (255, 255, 255) if dark else (28, 32, 58),
        "title2": TITLE_HL if dark else (col(accent) if accent else (240, 110, 40)),
        "band": (10, 12, 28) if dark else (255, 255, 255),
        "shadow": (12, 12, 26) if dark else (0, 0, 0),
    })


def col(c, default=(255, 255, 255)):
    if c is None:
        return default
    if isinstance(c, (list, tuple)):
        return tuple(int(v) for v in c[:3])
    c = str(c)
    if c.startswith("#") and len(c) == 7:
        return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
    if c.startswith("#") and len(c) == 4:
        return tuple(int(ch * 2, 16) for ch in c[1:])
    if c in UI and isinstance(UI[c], tuple):
        return UI[c]
    return PAL.get(c, default)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def shade(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


def luminance(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def resolve_background(spec, F):
    """-> dict(top, bottom, pattern, mode, accent). spec.background: preset name | {top,bottom|color,pattern,mode,accent}."""
    bg = F.get("background") or spec.get("background") or F.get("theme") or spec.get("theme") or "night"
    if isinstance(bg, str):
        top, bot, pat = BACKGROUNDS.get(bg.lower(), BACKGROUNDS["night"])
        d = {"top": top, "bottom": bot, "pattern": pat}
    else:
        c = bg.get("color")
        top = col(bg.get("top", c or "#16203f"))
        bot = col(bg.get("bottom", c or bg.get("top", "#2a3e78")))
        d = {"top": top, "bottom": bot, "pattern": bg.get("pattern"), "mode": bg.get("mode"), "accent": bg.get("accent")}
    if spec.get("pattern") is not None:
        d["pattern"] = spec["pattern"]
    if not d.get("mode"):
        d["mode"] = "light" if luminance(mix(d["top"], d["bottom"], 0.5)) > 140 else "dark"
    d["accent"] = d.get("accent") or spec.get("accent")
    return d


set_ui_mode("dark")


# ----------------------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease_io(p):
    p = clamp(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_back(p):
    p = clamp(p)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


EASES = {"io": ease_io, "out": ease_out, "back": ease_back, "linear": clamp}


def lerp(a, b, t):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [lerp(x, y, t) for x, y in zip(a, b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    return b if t >= 1 else a


# ----------------------------------------------------------------------------- ffmpeg
def ffmpeg_bin():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for cand in ["/mnt/data/ffmpeg", os.path.join(HERE, "ffmpeg")]:
        if os.path.exists(cand):
            os.chmod(cand, 0o755)
            return cand
    raise RuntimeError("ffmpeg not found")


def load_audio(path, sr=16000):
    cmd = [ffmpeg_bin(), "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


# ============================================================================= TIMING
STOP = set("""a an the and or but of to in on at for from by with as is are was were be been it its this that
these those you your we our they their he she his her i me my so if than then into onto about very just not no do does
did can will would could should has have had there here what when where which who how why also too all any some more most
yang dan di ke dari ini itu ada akan untuk dengan dalam pada atau juga karena jadi tapi tetapi lalu kalau jika bisa
sangat lebih paling sudah belum tidak bukan saja hanya kita kamu aku saya mereka dia ia nya pun lagi begitu seperti
adalah ialah yaitu oleh agar supaya sehingga maka para sang si tak apa mana kenapa mengapa bagaimana kapan siapa""".split())


def norm_word(w):
    return re.sub(r"[^0-9a-zA-ZÀ-ɏ%]+", "", w.lower())


def tokenize(script):
    toks = []
    for raw in re.findall(r"\S+", script.replace("—", " — ").replace("–", " – ")):
        if norm_word(raw) == "":
            if toks and raw.strip() in "—–-":
                toks[-1]["punct"] = max(toks[-1]["punct"], 1)
            continue
        p = 0
        if re.search(r"[.!?…]+[\"')\]]*$", raw):
            p = 2
        elif re.search(r"[,;:]+[\"')\]]*$", raw):
            p = 1
        toks.append({"w": raw, "n": norm_word(raw), "punct": p})
    return toks


def syllables(w):
    w = w.lower()
    digits = sum(ch.isdigit() for ch in w)
    if digits:
        return 1.6 * digits + 1
    v = len(re.findall(r"[aeiouyàáâäèéêëìíîïòóôöùúûü]+", w))
    return max(1, v) + (0.3 if len(w) > 7 else 0)


def estimate_timing(toks, rate=None, lang_hint="id", lead=0.35):
    """Script-only mode: synthetic timeline (~5.2 syll/s ID, ~4.6 syll/s EN)."""
    sps = rate or (5.2 if lang_hint == "id" else 4.6)
    t = lead
    out = []
    for tk in toks:
        d = syllables(tk["n"]) / sps + 0.04
        out.append({**tk, "start": round(t, 3), "end": round(t + d, 3)})
        t += d + (0.42 if tk["punct"] == 2 else 0.2 if tk["punct"] == 1 else 0.0)
    return out


def speech_map(x, sr=16000, hop=0.01, min_pause=0.14):
    h = int(sr * hop)
    n = len(x) // h
    fr = x[:n * h].reshape(n, h)
    db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
    lo, hi = np.percentile(db, 8), np.percentile(db, 92)
    thr = lo + 0.33 * (hi - lo)
    v = db > thr
    # close tiny gaps, drop tiny blips
    v = v.astype(np.uint8)
    k = max(1, int(0.05 / hop))
    v = cv2.morphologyEx(v.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)).ravel()
    v = cv2.morphologyEx(v.reshape(1, -1), cv2.MORPH_OPEN, np.ones((1, 3), np.uint8)).ravel().astype(bool)
    regions = []
    i = 0
    while i < n:
        if v[i]:
            j = i
            while j < n and v[j]:
                j += 1
            regions.append([i * hop, j * hop])
            i = j
        else:
            i += 1
    merged = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_pause:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def align(toks, audio_path):
    """Audio + script: pause-anchored alignment (no ASR model needed)."""
    x = load_audio(audio_path)
    regions = speech_map(x)
    if not regions:
        return estimate_timing(toks)
    # voiced-time axis
    vt = [0.0]
    for r in regions:
        vt.append(vt[-1] + r[1] - r[0])
    T = vt[-1]
    pauses = []  # (voiced_pos, real_start, real_end)
    for k in range(len(regions) - 1):
        pauses.append((vt[k + 1], regions[k][1], regions[k + 1][0]))
    N = len(toks)
    wts = np.array([syllables(t["n"]) for t in toks], float)
    cw = np.cumsum(wts) / wts.sum() * T
    gaps = list(range(N - 1))
    E = [cw[i] for i in gaps]
    bonus = [(2.0 if toks[i]["punct"] == 2 else 0.6 if toks[i]["punct"] == 1 else 0.0) for i in gaps]
    # a sentence end with no audible pause is unlikely -> skipping it costs
    gskip = [(1.0 if toks[i]["punct"] == 2 else 0.0) for i in gaps]
    K, G = len(pauses), len(gaps)
    INF = 1e18
    D = np.full((K + 1, G + 1), INF)
    B = np.zeros((K + 1, G + 1), np.int8)
    D[0, :] = 0
    for k in range(K + 1):
        for g in range(G + 1):
            if k == 0 and g == 0:
                continue
            best, bb = INF, 0
            if g > 0 and D[k, g - 1] + gskip[g - 1] < best:
                best, bb = D[k, g - 1] + gskip[g - 1], 1  # skip gap
            if k > 0:
                pd = pauses[k - 1][2] - pauses[k - 1][1]
                c = D[k - 1, g] + 0.6 + 2.5 * min(pd, 0.8)  # skip pause (unlikely mid-word)
                if c < best:
                    best, bb = c, 2
                if g > 0:
                    pk = pauses[k - 1][0]
                    c = D[k - 1, g - 1] + abs(pk - E[g - 1]) - bonus[g - 1] * (0.5 + min(pd, 0.8))
                    if c < best:
                        best, bb = c, 3
            D[k, g], B[k, g] = best, bb
    # backtrack
    k, g = K, G
    match = {}
    while k > 0 or g > 0:
        b = B[k, g]
        if b == 1:
            g -= 1
        elif b == 2:
            k -= 1
        else:
            match[g - 1] = k - 1
            k -= 1
            g -= 1

    def v2r(v):  # voiced time -> real time
        for idx in range(len(regions)):
            if v <= vt[idx + 1] + 1e-9:
                return regions[idx][0] + (v - vt[idx])
        return regions[-1][1]

    # anchors in voiced time: word index boundaries
    anchors = [(0, 0.0)] + [(gi + 1, pauses[pk][0]) for gi, pk in sorted(match.items())] + [(N, T)]
    out = []
    for (a, va), (b, vb) in zip(anchors[:-1], anchors[1:]):
        if b <= a:
            continue
        seg = wts[a:b]
        cs = np.concatenate([[0], np.cumsum(seg)]) / seg.sum() * (vb - va) + va
        for i in range(a, b):
            s, e = v2r(cs[i - a] + 1e-4), v2r(cs[i - a + 1] - 1e-4)
            out.append({**toks[i], "start": round(s, 3), "end": round(max(e, s + 0.05), 3)})
    return out


def transcribe(audio_path, model_dir, lang="id"):
    """Offline speech-to-text with OpenAI Whisper "base" (ONNX export from sherpa-onnx) run by onnxruntime
    (see whisper_ort.py). Audio is cut at pauses into <=25 s chunks (Whisper window is 30 s)."""
    from whisper_ort import WhisperORT
    sr = 16000
    x = load_audio(audio_path, sr)
    regions = speech_map(x, sr)
    chunks, cur = [], None
    for a, b in regions:
        if cur and b - cur[0] <= 25:
            cur[1] = b
        else:
            if cur:
                chunks.append(cur)
            cur = [a, b]
    if cur:
        chunks.append(cur)
    model = WhisperORT(model_dir)
    out = []
    for a, b in chunks:
        seg = x[int(max(0, a - 0.15) * sr):int((b + 0.15) * sr)]
        out.append(model.transcribe(seg, lang))
    return " ".join(t for t in out if t)


def compare_texts(transcript, script):
    """Word-level agreement between the Whisper transcript and the user's script (naskah)."""
    import difflib
    a = [norm_word(w) for w in transcript.split() if norm_word(w)]
    b = [norm_word(w) for w in script.split() if norm_word(w)]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    diffs = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            diffs.append((op, " ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return sm.ratio(), diffs


def from_words(words):
    toks = []
    for w in words:
        t = tokenize(w.get("word", w.get("w", "")))
        if not t:
            continue
        tk = t[0]
        tk.update(start=float(w["start"]), end=float(w["end"]))
        toks.append(tk)
    return toks


def parse_srt(path):
    txt = open(path, encoding="utf-8").read()
    out = []
    for blk in re.split(r"\n\s*\n", txt.strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", blk)
        if not m:
            continue
        g = [int(v) for v in m.groups()]
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(blk[m.end():].strip().splitlines())
        toks = tokenize(text)
        w = np.array([syllables(t["n"]) for t in toks], float)
        cs = np.concatenate([[0], np.cumsum(w)]) / max(w.sum(), 1e-6) * (e - s) + s
        for i, tk in enumerate(toks):
            out.append({**tk, "start": cs[i], "end": cs[i + 1]})
    return out


def resolve_words(spec, base):
    def P(p):
        return p if os.path.isabs(p) else os.path.join(base, p)
    lang = spec.get("lang", "id")
    if spec.get("words"):
        src = spec["words"]
        if isinstance(src, str):
            src = json.load(open(P(src)))
        return from_words(src), "words"
    if spec.get("srt"):
        return parse_srt(P(spec["srt"])), "srt"
    script = spec.get("script", "")
    toks = tokenize(script)
    if spec.get("audio"):
        if not toks:
            raise SystemExit("Audio given but no script/words: provide the transcript in spec.script")
        return align(toks, P(spec["audio"])), "audio"
    return estimate_timing(toks, spec.get("rate"), lang), "script"


# ----------------------------------------------------------------------------- captions
def build_captions(words, highlight=None, max_words=4, max_chars=20):
    hl = set(norm_word(h) for h in (highlight or []))
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        text = " ".join(words[j]["w"] for j in cur)
        nxt = words[i + 1] if i + 1 < len(words) else None
        brk = w["punct"] > 0 or len(cur) >= max_words or nxt is None
        if nxt is not None and len(text) + 1 + len(nxt["w"]) > max_chars:
            brk = True
        # don't strand a single punctuated word on its own
        if brk and w["punct"] == 0 and nxt is not None and nxt["punct"] > 0 and len(cur) <= 4 \
                and len(text) + 1 + len(nxt["w"]) <= max_chars + 6 and nxt["start"] - w["end"] < 0.35:
            brk = False
        if nxt is not None and nxt["start"] - w["end"] > 0.35:
            brk = True
        if brk:
            chunks.append(cur)
            cur = []
    caps = []
    for ci, ch in enumerate(chunks):
        ws = [words[j] for j in ch]
        flags = [norm_word(x["w"]) in hl for x in ws]
        if sum(flags) > 1:  # keep one keyword per chunk (the longest)
            best = max((len(ws[k]["n"]), k) for k in range(len(ws)) if flags[k])[1]
            flags = [k == best for k in range(len(ws))]
        if hl == set() or not any(flags):
            # auto: pick the most "contentful" word
            cands = [(len(x["n"]), k) for k, x in enumerate(ws) if x["n"] not in STOP and len(x["n"]) >= 3]
            if cands and not hl:
                flags[max(cands)[1]] = True
            elif cands and hl and not any(flags):
                pass
        start = ws[0]["start"]
        end = words[chunks[ci + 1][0]]["start"] if ci + 1 < len(chunks) else ws[-1]["end"] + 0.6
        if end - ws[-1]["end"] > 0.7:
            end = ws[-1]["end"] + 0.7
        caps.append({"words": [x["w"] for x in ws], "hl": flags, "start": start, "end": end})
    return caps


def write_srt(caps, path):
    def ts(t):
        t = max(t, 0)
        return "%02d:%02d:%02d,%03d" % (t // 3600, t % 3600 // 60, t % 60, (t - int(t)) * 1000)
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(caps, 1):
            f.write(f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{' '.join(c['words'])}\n\n")


# ============================================================================= DRAW HELPERS
SH = 4  # cv2 subpixel shift
SHF = 1 << SH


def ipts(pts):
    return np.round(np.asarray(pts, float) * SHF).astype(np.int32).reshape(-1, 1, 2)


class Xf:
    """world -> screen affine (uniform scale)."""
    def __init__(self, k, bx, by):
        self.k, self.bx, self.by = k, bx, by

    def p(self, x, y):
        return (x * self.k + self.bx, y * self.k + self.by)

    def ps(self, pts):
        a = np.asarray(pts, float).reshape(-1, 2)
        return np.stack([a[:, 0] * self.k + self.bx, a[:, 1] * self.k + self.by], 1)

    def s(self, v):
        return v * self.k

    def about(self, ax, ay, sc):
        # scale by sc around world anchor
        return Xf(self.k * sc, self.k * ax * (1 - sc) + self.bx, self.k * ay * (1 - sc) + self.by)


def fill_poly(img, pts, c):
    cv2.fillPoly(img, [ipts(pts)], col(c), cv2.LINE_AA, SH)


def stroke(img, pts, c, w=2, closed=False):
    w = max(1, int(round(w)))
    cv2.polylines(img, [ipts(pts)], closed, col(c), w, cv2.LINE_AA, SH)


def circle(img, x, y, r, c, w=-1):
    if r <= 0.3:
        return
    cv2.circle(img, (int(round(x * SHF)), int(round(y * SHF))), int(round(r * SHF)), col(c),
               w if w < 0 else max(1, int(round(w))), cv2.LINE_AA, SH)


def line(img, x0, y0, x1, y1, c, w=2):
    cv2.line(img, (int(x0 * SHF), int(y0 * SHF)), (int(x1 * SHF), int(y1 * SHF)), col(c),
             max(1, int(round(w))), cv2.LINE_AA, SH)


def blend_region(img, alpha, bbox, fn):
    if alpha <= 0.003:
        return
    if alpha >= 0.997:
        fn(img)
        return
    H, W = img.shape[:2]
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    save = img[y0:y1, x0:x1].copy()
    fn(img)
    img[y0:y1, x0:x1] = cv2.addWeighted(img[y0:y1, x0:x1], alpha, save, 1 - alpha, 0)


def paste(img, spr, x, y, alpha=1.0, center=True):
    """alpha-composite RGBA sprite (numpy HxWx4 uint8) at (x,y)."""
    if spr is None or alpha <= 0.003:
        return
    h, w = spr.shape[:2]
    if center:
        x, y = x - w / 2, y - h / 2
    x, y = int(round(x)), int(round(y))
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + s[:, :, :3].astype(np.float32) * a).astype(np.uint8)


_glow_cache = {}


def glow_sprite(r, c, strength=0.55):
    key = (int(r), col(c), round(strength, 2))
    if key not in _glow_cache:
        R = int(r)
        size = 2 * R + 2
        yy, xx = np.mgrid[0:size, 0:size]
        d = np.sqrt((xx - R) ** 2 + (yy - R) ** 2) / max(R, 1)
        a = np.clip(1 - d, 0, 1) ** 2 * 255 * strength
        spr = np.zeros((size, size, 4), np.uint8)
        spr[:, :, :3] = col(c)
        spr[:, :, 3] = a.astype(np.uint8)
        _glow_cache[key] = spr
    return _glow_cache[key]


_font_cache = {}


def font(path, size):
    key = (path, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, int(size))
    return _font_cache[key]


_text_cache = {}


def text_sprite(parts, size, fontpath=F_CAPTION, stroke_w=0, stroke_c=(20, 20, 40),
                shadow=0, spacing=0, max_w=None):
    """parts: list of (text, rgb). Returns RGBA numpy."""
    key = (tuple(parts), int(size), fontpath, stroke_w, stroke_c, shadow, spacing, max_w)
    if key in _text_cache:
        return _text_cache[key]
    sz = int(size)
    while True:
        f = font(fontpath, sz)
        space = f.getlength(" ")
        widths = []
        for t, _ in parts:
            if spacing:
                widths.append(sum(f.getlength(ch) + spacing for ch in t) - spacing)
            else:
                widths.append(f.getlength(t))
        total = sum(widths) + space * (len(parts) - 1)
        if not max_w or total <= max_w or sz < 12:
            break
        sz = int(sz * 0.93)
    asc, desc = f.getmetrics()
    pad = stroke_w + shadow + 4
    W = int(total + 2 * pad + 2)
    H = int(asc + desc + 2 * pad)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def draw_all(dx, dy, override=None, sw=0, sc=None):
        x = pad + dx
        for (t, c), w in zip(parts, widths):
            if spacing:
                xx = x
                for ch in t:
                    d.text((xx, pad + dy), ch, font=f, fill=override or tuple(c) + (255,),
                           stroke_width=sw, stroke_fill=sc)
                    xx += f.getlength(ch) + spacing
            else:
                d.text((x, pad + dy), t, font=f, fill=override or tuple(c) + (255,),
                       stroke_width=sw, stroke_fill=sc)
            x += w + space

    if shadow:
        draw_all(0, shadow, override=(10, 10, 25, 150), sw=stroke_w, sc=(10, 10, 25, 150))
    if stroke_w:
        draw_all(0, 0, override=tuple(stroke_c) + (255,), sw=stroke_w, sc=tuple(stroke_c) + (255,))
    draw_all(0, 0)
    arr = np.array(im)
    _text_cache[key] = arr
    return arr


def round_rect_sprite(w, h, r, fill, border=None, bw=0, fill_a=235):
    im = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, int(w) - 1, int(h) - 1], radius=int(r), fill=tuple(fill) + (fill_a,),
                        outline=(tuple(border) + (255,)) if border else None, width=int(bw))
    return np.array(im)


def shade_half(img, sx, sy, R, angle_deg, k=0.55, soft=0.12):
    """darken the half of a disk facing angle_deg (0=right, 90=down) — terminator line for planets/moons."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, sx - R - 2)), int(max(0, sy - R - 2)), int(min(W, sx + R + 3)), int(min(H, sy + R + 3))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - sx, yy - sy
    inside = np.clip((R - np.sqrt(dx * dx + dy * dy)) + 0.5, 0, 1)
    a = math.radians(angle_deg)
    side = np.clip(((dx * math.cos(a) + dy * math.sin(a)) / max(R, 1)) / soft + 0.5, 0, 1)
    m = (inside * side * k)[:, :, None]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - m) + np.array([12, 14, 32], np.float32) * m).astype(np.uint8)


def rng_for(*key):
    h = int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


# ============================================================================= OBJECTS
def anchor_of(o):
    if "x" in o and "y" in o:
        return (o["x"], o["y"])
    if o.get("type") in ("sea", "ground"):
        return (360, o.get("y", 900))
    if "points" in o and o["points"]:
        p = np.mean(np.asarray(o["points"], float), 0)
        return (p[0], p[1])
    if "from" in o:
        a, b = o["from"], o["to"]
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    if "area" in o:
        a = o["area"]
        return ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    return (360, 640)


def d_ground(img, o, X, t, ctx):
    y = o.get("y", 1000)
    c = col(o.get("color", "dirt"))
    x0, x1 = o.get("x0", -3000), o.get("x1", 4000)
    bot = o.get("bottom", 6000)
    pts = X.ps([[x0, y], [x1, y], [x1, bot], [x0, bot]])
    fill_poly(img, pts, c)
    stroke(img, X.ps([[x0, y], [x1, y]]), mix(c, (255, 255, 255), 0.18), X.s(3))
    if o.get("color", "dirt") == "grass":
        r = rng_for("tuft", y)
        xs = r.uniform(-400, 1200, 70)
        for xx in xs:
            pts = X.ps([[xx - 4, y + 1], [xx, y - 7], [xx + 4, y + 1]])
            fill_poly(img, pts, mix(c, (255, 255, 255), 0.15))


def d_sea(img, o, X, t, ctx):
    y = o.get("y", 850)
    bot = o.get("bottom", y + 260)
    x0, x1 = o.get("x0", -3000), o.get("x1", 4000)
    amp, wl = o.get("amp", 4), o.get("wave", 90)
    c = col(o.get("color", (40, 128, 212)))
    xs = np.linspace(x0, x1, 260)
    # only resolve visible part finely
    ys = y + amp * np.sin(xs / wl * 2 * math.pi + t * 1.6) + amp * 0.5 * np.sin(xs / (wl * 0.53) - t * 1.1)
    top = np.stack([xs, ys], 1)
    poly = np.concatenate([top, [[x1, bot], [x0, bot]]])
    fill_poly(img, X.ps(poly), c)
    band = np.concatenate([top + [0, 26], [[x1, bot], [x0, bot]]])
    fill_poly(img, X.ps(band), shade(c, 0.86))
    stroke(img, X.ps(top), mix(c, (255, 255, 255), 0.55), X.s(2.5))
    # floating sparkles
    if o.get("sparkle", True):
        r = rng_for("sea", y)
        n = int(o.get("sparkles", 70))
        px = r.uniform(-200, 920, n)
        py = r.uniform(y + 20, bot - 12, n)
        sp = r.uniform(4, 12, n)
        for i in range(n):
            xx = (px[i] + t * sp[i] + 1120) % 1120 - 200
            yy = py[i] + 3 * math.sin(t * 1.3 + i)
            sx, sy = X.p(xx, yy)
            circle(img, sx, sy, min(X.s(2.4), 4.0 * img.shape[1] / 720), (220, 235, 255))


MTN_SHAPE = [(-0.5, 0), (-0.33, -0.45), (-0.27, -0.43), (-0.05, -1.0), (0.17, -0.6), (0.23, -0.62), (0.5, 0)]


def d_mountain(img, o, X, t, ctx):
    x, y, w, h = o["x"], o["y"], o.get("w", 560), o.get("h", 520)
    pts = [[x + a * w, y + b * h] for a, b in MTN_SHAPE]
    c = col(o.get("color", "mountain"))
    fill_poly(img, X.ps(pts), c)
    # subtle shading on right face
    sh = [[x - 0.05 * w, y - h], [x + 0.17 * w, y - 0.6 * h], [x + 0.23 * w, y - 0.62 * h], [x + 0.5 * w, y], [x + 0.08 * w, y]]
    fill_poly(img, X.ps(sh), shade(c, 0.9))
    stroke(img, X.ps(pts), mix(c, (255, 255, 255), 0.62), X.s(3))
    if o.get("snow", True):
        px, py = x - 0.05 * w, y - h
        snow = [[px, py], [px - 0.055 * w, py + 0.13 * h], [px - 0.02 * w, py + 0.1 * h], [px + 0.01 * w, py + 0.14 * h],
                [px + 0.04 * w, py + 0.1 * h], [px + 0.062 * w, py + 0.13 * h]]
        fill_poly(img, X.ps(snow), (245, 248, 255))


def d_river(img, o, X, t, ctx):
    pts = np.asarray(o["points"], float)
    w = o.get("width", 26)
    c = col(o.get("color", (78, 160, 236)))
    sp = X.ps(pts)
    stroke(img, sp, (235, 242, 255), X.s(w + 6))
    stroke(img, sp, c, X.s(w))
    # flowing dots
    seg = np.diff(pts, axis=0)
    L = np.sqrt((seg ** 2).sum(1))
    total = L.sum()
    gap = o.get("dot_gap", 26)
    speed = o.get("speed", 40)
    n = int(total / gap)
    dc = col(o.get("dot_color", (255, 240, 170)))
    for i in range(n):
        d = (i * gap + t * speed) % total
        k = 0
        while k < len(L) - 1 and d > L[k]:
            d -= L[k]
            k += 1
        p = pts[k] + seg[k] * (d / max(L[k], 1e-6))
        sx, sy = X.p(*p)
        circle(img, sx, sy, X.s(w * 0.2), (250, 250, 255))
        circle(img, sx, sy, X.s(w * 0.13), dc)


def d_rocks(img, o, X, t, ctx):
    x, y, s = o["x"], o["y"], o.get("s", 1)
    r = rng_for("rocks", x, y)
    c = col(o.get("color", "rock"))
    n = int(o.get("n", 4))
    for i in range(n):
        cx = x + (i - n / 2) * 42 * s + r.uniform(-8, 8)
        cy = y + r.uniform(-30, 30) * s
        rw, rh = r.uniform(20, 34) * s, r.uniform(12, 20) * s
        pts = [[cx - rw, cy + rh * 0.5], [cx - rw * 0.6, cy - rh * 0.6], [cx + rw * 0.3, cy - rh * 0.8],
               [cx + rw, cy + rh * 0.1], [cx + rw * 0.7, cy + rh * 0.5]]
        fill_poly(img, X.ps(pts), c)
        stroke(img, X.ps(pts), mix(c, (255, 255, 255), 0.5), X.s(2), True)
    bits = o.get("bits", 0)
    if bits:
        bc = col(o.get("bits_color", (215, 190, 130)))
        for i in range(int(bits)):
            ph = (t * 0.25 + i / bits) % 1
            bx = x + (r.uniform(-1, 1) * 70 + ph * 60) * s
            by = y + (ph * 110 - 30) * s
            sx, sy = X.p(bx, by)
            q = X.s(4 * s)
            cv2.rectangle(img, (int(sx - q), int(sy - q)), (int(sx + q), int(sy + q)), bc, -1)


def d_cloud(img, o, X, t, ctx):
    s = o.get("s", 1.0) * 100
    x, y = o["x"] + 6 * math.sin(t * 0.4 + o["x"]), o["y"]
    dark = o.get("dark", False)
    c = col(o.get("color", (70, 76, 128) if dark else (196, 206, 232)))
    blobs = [(-0.55, 0.12, 0.33), (-0.2, -0.12, 0.45), (0.22, -0.05, 0.4), (0.55, 0.13, 0.3)]
    if dark:
        for bx, by, br in blobs:
            circle(img, *X.p(x + bx * s, y + by * s + 10), X.s(br * s * 1.05), shade(c, 0.78))
    for bx, by, br in blobs:
        circle(img, *X.p(x + bx * s, y + by * s), X.s(br * s), c)
    fill_poly(img, X.ps([[x - 0.55 * s, y + 0.12 * s], [x + 0.55 * s, y + 0.12 * s],
                         [x + 0.55 * s, y + 0.43 * s], [x - 0.55 * s, y + 0.43 * s]]), c)
    for bx, by, br in blobs[1:3]:
        circle(img, *X.p(x + bx * s - 0.06 * s, y + by * s - 0.08 * s), X.s(br * s * 0.55), mix(c, (255, 255, 255), 0.18))
    if o.get("rain"):
        d_rain(img, {"area": [x - 0.6 * s, y + 0.4 * s, x + 0.6 * s, o.get("rain_to", y + 600)],
                     "density": o.get("density", 1.0), "seed": o["x"]}, X, t, ctx)


def d_rain(img, o, X, t, ctx):
    x0, y0, x1, y1 = o["area"]
    dens = o.get("density", 1.0)
    n = int((x1 - x0) * (y1 - y0) / 5200 * dens)
    r = rng_for("rain", o.get("seed", 0), x0, y0)
    px = r.uniform(x0, x1, n)
    ph = r.uniform(0, 1, n)
    c = col(o.get("color", (120, 190, 255)))
    L = 26
    spd = o.get("speed", 1.6)
    for i in range(n):
        p = (ph[i] + t * spd * 700 / (y1 - y0 + 1)) % 1
        yy = y0 + p * (y1 - y0)
        xx = px[i] - p * 30
        a, b = X.p(xx, yy), X.p(xx - 5, yy + L)
        line(img, a[0], a[1], b[0], b[1], c, X.s(2.4))


def d_sun(img, o, X, t, ctx):
    x, y, r = o["x"], o["y"], o.get("r", 42)
    sx, sy = X.p(x, y)
    R = X.s(r)
    c = col(o.get("color", (245, 190, 50)))
    paste(img, glow_sprite(R * 2.3, mix(c, (255, 255, 255), 0.3), 0.35), sx, sy)
    circle(img, sx, sy, R * 1.18, mix(c, (40, 50, 90), 0.55))
    circle(img, sx, sy, R, (255, 240, 200))
    circle(img, sx, sy, R * 0.9, c)
    circle(img, sx - R * 0.15, sy - R * 0.15, R * 0.55, mix(c, (255, 240, 160), 0.45))


def d_moon(img, o, X, t, ctx):
    x, y, r = o["x"], o["y"], o.get("r", 38)
    sx, sy = X.p(x, y)
    R = X.s(r)
    paste(img, glow_sprite(R * 2.2, (200, 210, 255), 0.25), sx, sy)
    circle(img, sx, sy, R, (232, 236, 250))
    for dx, dy, rr in [(-0.3, -0.2, 0.18), (0.25, 0.25, 0.14), (0.1, -0.4, 0.1)]:
        circle(img, sx + dx * R, sy + dy * R, rr * R, (200, 205, 228))
    if "shade" in o:  # phase: dark half faces this angle (0=right, 90=down, 180=left)
        shade_half(img, sx, sy, R, o["shade"], o.get("shade_k", 0.8), 0.08)


def bolt_points(o):
    (ax, ay), (bx, by) = o["from"], o["to"]
    r = rng_for("bolt", ax, ay, o.get("seed", 1))
    n = o.get("segments", 7)
    amp = o.get("amp", 28)
    pts = []
    for i in range(n + 1):
        p = i / n
        x = ax + (bx - ax) * p
        y = ay + (by - ay) * p
        if 0 < i < n:
            x += (amp if i % 2 else -amp) * r.uniform(0.6, 1.2)
        pts.append([x, y])
    return pts


def d_lightning(img, o, X, t, ctx):
    pts = X.ps(bolt_points(o))
    hot = clamp(o.get("hot", 0))
    flick = o.get("_flash", 0)
    gc = mix((170, 185, 255), (240, 110, 60), hot)
    bb = (pts[:, 0].min() - 40, pts[:, 1].min() - 40, pts[:, 0].max() + 40, pts[:, 1].max() + 40)
    blend_region(img, 0.28 + 0.4 * hot + 0.3 * flick, bb, lambda im: stroke(im, pts, gc, X.s(16 + 16 * hot)))
    stroke(img, pts, mix((205, 210, 245), (250, 140, 90), hot), X.s(9 + 3 * hot))
    stroke(img, pts, (255, 255, 255), X.s(4.5))
    if hot > 0.05:
        r = rng_for("hotsp", o.get("seed", 1))
        for i in range(12):
            k = i % (len(pts) - 1)
            p = pts[k] + (pts[k + 1] - pts[k]) * r.uniform()
            off = X.s(r.uniform(18, 34)) * (1 if i % 2 else -1)
            circle(img, p[0] + off + 3 * math.sin(t * 3 + i), p[1] - 4 * math.sin(t * 2 + i),
                   X.s(r.uniform(4, 7)), mix((150, 110, 140), (235, 120, 70), hot))


def d_house(img, o, X, t, ctx):
    x, y, s = o["x"], o["y"], o.get("s", 1.0)
    w, h = 64 * s, 48 * s
    wall = col(o.get("color", (222, 150, 100)))
    roof = col(o.get("roof", (150, 56, 52)))
    fill_poly(img, X.ps([[x - w / 2, y], [x + w / 2, y], [x + w / 2, y - h], [x - w / 2, y - h]]), wall)
    fill_poly(img, X.ps([[x - w * 0.62, y - h], [x, y - h - 34 * s], [x + w * 0.62, y - h]]), roof)
    stroke(img, X.ps([[x - w * 0.62, y - h], [x, y - h - 34 * s], [x + w * 0.62, y - h]]), shade(roof, 0.7), X.s(2), True)
    lit = o.get("lit", 0)
    wc = mix((80, 104, 150), (250, 236, 150), clamp(lit))
    fill_poly(img, X.ps([[x - w * 0.36, y - h * 0.7], [x - w * 0.05, y - h * 0.7], [x - w * 0.05, y - h * 0.32], [x - w * 0.36, y - h * 0.32]]), (60, 40, 40))
    fill_poly(img, X.ps([[x - w * 0.33, y - h * 0.66], [x - w * 0.08, y - h * 0.66], [x - w * 0.08, y - h * 0.36], [x - w * 0.33, y - h * 0.36]]), wc)
    fill_poly(img, X.ps([[x + w * 0.12, y], [x + w * 0.36, y], [x + w * 0.36, y - h * 0.68], [x + w * 0.12, y - h * 0.68]]), (100, 58, 40))


def d_tree(img, o, X, t, ctx):
    x, y, s = o["x"], o["y"], o.get("s", 1.0)
    c = col(o.get("color", (60, 150, 90)))
    fill_poly(img, X.ps([[x - 5 * s, y], [x + 5 * s, y], [x + 4 * s, y - 40 * s], [x - 4 * s, y - 40 * s]]), (110, 70, 50))
    sw = 2 * math.sin(t * 1.2 + x)
    for dx, dy, r in [(-14, -52, 22), (14, -54, 22), (0, -72, 26)]:
        circle(img, *X.p(x + (dx + sw) * s, y + dy * s), X.s(r * s), c)
    circle(img, *X.p(x + (-6 + sw) * s, y - 78 * s), X.s(10 * s), mix(c, (255, 255, 255), 0.2))


def d_vent(img, o, X, t, ctx):
    x, y, h = o["x"], o["y"], o.get("h", 70)
    w = o.get("w", 46)
    pts = [[x - w / 2, y], [x - w * 0.32, y - h], [x + w * 0.32, y - h], [x + w / 2, y]]
    fill_poly(img, X.ps(pts), (30, 28, 46))
    stroke(img, X.ps(pts), (150, 150, 190), X.s(2), True)
    if o.get("smoke"):
        for i in range(8):
            p = (t * 0.35 + i / 8) % 1
            cx = x + 10 * math.sin(i * 2.1 + t) * p
            cy = y - h - p * 160
            sx, sy = X.p(cx, cy)
            r = X.s(10 + 22 * p)
            blend_region(img, 0.55 * (1 - p), (sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2),
                         lambda im, sx=sx, sy=sy, r=r: circle(im, sx, sy, r, (40, 40, 58)))


def d_vapor(img, o, X, t, ctx):
    x, y, h = o["x"], o["y"], o.get("h", 220)
    n = int(o.get("n", 3))
    c = col(o.get("color", (175, 195, 240)))
    for i in range(n):
        p = (t * 0.22 + i / n) % 1
        bx = x + (i - (n - 1) / 2) * o.get("spread", 50)
        yy = np.linspace(0, 70, 18)
        pts = [[bx + 6 * math.sin(v / 9 + t * 3 + i), y - p * h - v] for v in yy]
        sp = X.ps(pts)
        a = math.sin(math.pi * p)
        bb = (sp[:, 0].min() - 6, sp[:, 1].min() - 6, sp[:, 0].max() + 6, sp[:, 1].max() + 6)
        blend_region(img, 0.8 * a, bb, lambda im, sp=sp: stroke(im, sp, c, X.s(3)))


def d_ruler(img, o, X, t, ctx):
    x, y, L = o["x"], o["y"], o.get("len", 300)
    ticks = int(o.get("ticks", 4))
    c = col(o.get("color", "ink"))
    hl = clamp(o.get("highlight", 0))
    a, b = X.p(x, y), X.p(x + L, y)
    line(img, a[0], a[1], b[0], b[1], c, X.s(3))
    if hl > 0:
        e = X.p(x + L * hl, y)
        line(img, a[0], a[1], e[0], e[1], PAL["yellow"], X.s(9))
    for i in range(ticks + 1):
        tx = x + L * i / ticks
        p0, p1 = X.p(tx, y - 12), X.p(tx, y + 12)
        line(img, p0[0], p0[1], p1[0], p1[1], c, X.s(3))


PART_SHAPES = {"water", "plus", "minus", "dot", "square", "ring"}


def d_particles(img, o, X, t, ctx):
    x0, y0, x1, y1 = o["area"]
    for gi, g in enumerate(o.get("groups", [])):
        n = int(g.get("n", 30))
        r = rng_for("parts", o.get("id"), gi, x0, y0)
        px, py = r.uniform(x0, x1, n), r.uniform(y0, y1, n)
        ph = r.uniform(0, 6.28, n)
        lf = r.uniform(0.5, 1.4, n)
        lift_frac = g.get("lift_frac", 0.5)
        sel = r.uniform(0, 1, n) < lift_frac
        lift = g.get("lift", 0)
        motion = g.get("motion", "jiggle")
        spd = g.get("speed", 1.0)
        rad = g.get("r", 7)
        shape = g.get("shape", "dot")
        c = col(g.get("color", "red"))
        vis = clamp(g.get("show", 1))
        nn = int(round(n * vis))
        for i in range(nn):
            xx, yy = px[i], py[i]
            if motion == "jiggle":
                xx += 6 * math.sin(t * 1.7 * spd + ph[i])
                yy += 6 * math.cos(t * 1.3 * spd + ph[i] * 1.3)
            elif motion in ("rise", "fall", "flow"):
                span = (y1 - y0) if motion != "flow" else (x1 - x0)
                d = (t * 40 * spd * lf[i]) % span
                if motion == "rise":
                    yy = y1 - ((y1 - yy + d) % span)
                elif motion == "fall":
                    yy = y0 + ((yy - y0 + d) % span)
                else:
                    xx = x0 + ((xx - x0 + d) % span)
                xx += 4 * math.sin(t * 2 + ph[i])
            if lift and sel[i]:
                yy -= lift * lf[i]
                xx += 10 * math.sin(t + ph[i])
            sx, sy = X.p(xx, yy)
            R = X.s(rad)
            if shape == "water":
                ang = ph[i]
                for k in (-1, 1):
                    a = ang + k * 0.9
                    circle(img, sx + math.cos(a) * R * 0.95, sy + math.sin(a) * R * 0.95, R * 0.55, (245, 246, 255))
                circle(img, sx, sy, R, c)
            elif shape in ("plus", "minus"):
                circle(img, sx, sy, R, shade(c, 0.7))
                circle(img, sx, sy, R * 0.82, c)
                line(img, sx - R * 0.45, sy, sx + R * 0.45, sy, (255, 255, 255), max(1, R * 0.22))
                if shape == "plus":
                    line(img, sx, sy - R * 0.45, sx, sy + R * 0.45, (255, 255, 255), max(1, R * 0.22))
            elif shape == "square":
                q = R
                cv2.rectangle(img, (int(sx - q), int(sy - q)), (int(sx + q), int(sy + q)), c, -1)
            elif shape == "ring":
                circle(img, sx, sy, R, c, max(1, R * 0.3))
            else:
                circle(img, sx, sy, R, c)


def d_circle(img, o, X, t, ctx):
    sx, sy = X.p(o["x"], o["y"])
    R = X.s(o.get("r", 30))
    if o.get("glow"):
        paste(img, glow_sprite(R * 2.2, col(o.get("fill", "yellow")), 0.45), sx, sy)
    if o.get("fill", "white") != "none":
        circle(img, sx, sy, R, col(o.get("fill", "white")))
    if o.get("stroke"):
        circle(img, sx, sy, R, col(o["stroke"]), X.s(o.get("sw", 3)))


def d_rect(img, o, X, t, ctx):
    x, y, w, h = o["x"], o["y"], o.get("w", 100), o.get("h", 60)
    r = o.get("radius", 0)
    pts = [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]]
    if r:
        pts = []
        for cx, cy, a0 in [(x + w / 2 - r, y - h / 2 + r, -90), (x + w / 2 - r, y + h / 2 - r, 0),
                           (x - w / 2 + r, y + h / 2 - r, 90), (x - w / 2 + r, y - h / 2 + r, 180)]:
            for a in range(a0, a0 + 91, 15):
                pts.append([cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))])
    if o.get("fill", "white") != "none":
        fill_poly(img, X.ps(pts), col(o.get("fill", "white")))
    if o.get("stroke"):
        stroke(img, X.ps(pts), col(o["stroke"]), X.s(o.get("sw", 3)), True)


def d_poly(img, o, X, t, ctx):
    pts = X.ps(o["points"])
    if o.get("fill", "white") != "none":
        fill_poly(img, pts, col(o.get("fill", "white")))
    if o.get("stroke"):
        stroke(img, pts, col(o["stroke"]), X.s(o.get("sw", 3)), True)


def d_line(img, o, X, t, ctx):
    pts = np.asarray(o["points"], float)
    draw = clamp(o.get("draw", 1))
    if draw < 1:
        seg = np.diff(pts, axis=0)
        L = np.sqrt((seg ** 2).sum(1))
        tgt = L.sum() * draw
        out = [pts[0]]
        for k in range(len(L)):
            if tgt >= L[k]:
                out.append(pts[k + 1])
                tgt -= L[k]
            else:
                out.append(pts[k] + seg[k] * tgt / max(L[k], 1e-6))
                break
        pts = np.asarray(out)
    if len(pts) < 2:
        return
    sp = X.ps(pts)
    c = col(o.get("color", "ink"))
    w = X.s(o.get("width", 4))
    if o.get("dash"):
        seg = np.diff(sp, axis=0)
        for k in range(len(seg)):
            L = np.hypot(*seg[k])
            dl = X.s(o["dash"])
            m = int(L / (2 * dl)) + 1
            for j in range(m):
                a = sp[k] + seg[k] * (2 * j * dl / max(L, 1e-6))
                b = sp[k] + seg[k] * (min((2 * j + 1) * dl, L) / max(L, 1e-6))
                line(img, a[0], a[1], b[0], b[1], c, w)
    else:
        stroke(img, sp, c, w)
    if o.get("arrow"):
        draw_arrowhead(img, sp[-2], sp[-1], c, w)


def draw_arrowhead(img, a, b, c, w):
    d = np.asarray(b) - np.asarray(a)
    L = np.hypot(*d) + 1e-6
    u = d / L
    nrm = np.array([-u[1], u[0]])
    s = 4 * w + 6
    p = np.asarray(b)
    fill_poly(img, [p + u * s * 0.3, p - u * s + nrm * s * 0.7, p - u * s - nrm * s * 0.7], c)


def d_arrow(img, o, X, t, ctx):
    o2 = dict(o)
    o2["points"] = [o["from"], o["to"]]
    o2["arrow"] = True
    o2.setdefault("width", 6)
    o2.setdefault("color", "orange")
    d_line(img, o2, X, t, ctx)


def d_text(img, o, X, t, ctx):
    sx, sy = X.p(o["x"], o["y"])
    size = X.s(o.get("size", 40))
    light = UI.get("mode") == "light"
    spr = text_sprite([(o["text"], col(o.get("color", "ink")))], max(8, int(size)),
                      F_CAPTION if o.get("font") != "title" else F_TITLE, stroke_w=max(1, int(size / 12)) if not light or o.get("color") else 0,
                      stroke_c=(20, 20, 40))
    paste(img, spr, sx, sy)


def d_person(img, o, X, t, ctx):
    x, y, s = o["x"], o["y"], o.get("s", 1.0)
    c = col(o.get("color", (240, 190, 150)))
    shirt = col(o.get("shirt", (80, 160, 230)))
    fill_poly(img, X.ps([[x - 16 * s, y - 2], [x + 16 * s, y - 2], [x + 13 * s, y - 52 * s], [x - 13 * s, y - 52 * s]]), shirt)
    circle(img, *X.p(x, y - 68 * s), X.s(15 * s), c)
    fill_poly(img, X.ps([[x - 12 * s, y], [x - 3 * s, y], [x - 3 * s, y - 6], [x - 12 * s, y - 6]]), (50, 50, 70))
    fill_poly(img, X.ps([[x + 3 * s, y], [x + 12 * s, y], [x + 12 * s, y - 6], [x + 3 * s, y - 6]]), (50, 50, 70))


def d_planet(img, o, X, t, ctx):
    sx, sy = X.p(o["x"], o["y"])
    R = X.s(o.get("r", 80))
    c = col(o.get("color", (70, 140, 230)))
    if o.get("glow", True):
        paste(img, glow_sprite(R * 1.6, mix(c, (255, 255, 255), 0.3), 0.3), sx, sy)
    circle(img, sx, sy, R, c)
    if o.get("land", o.get("earth", False)):
        r = rng_for("land", o["x"])
        for i in range(5):
            a = r.uniform(0, 6.28) + t * 0.05
            d = r.uniform(0, 0.6)
            circle(img, sx + math.cos(a) * d * R, sy + math.sin(a) * d * R, R * r.uniform(0.18, 0.32), (90, 180, 110))
    shade_half(img, sx, sy, R, o.get("shade", 35), o.get("shade_k", 0.35), 0.5)
    circle(img, sx, sy, R, mix(c, (255, 255, 255), 0.5), 2)
    if o.get("ring"):
        cv2.ellipse(img, (int(sx), int(sy)), (int(R * 1.7), int(R * 0.45)), -15, 0, 360, (230, 210, 170), 3, cv2.LINE_AA)


def d_stars(img, o, X, t, ctx):
    H, W = img.shape[:2]
    r = rng_for("stars", o.get("seed", 0))
    n = int(o.get("n", 90))
    px, py, ph = r.uniform(0, W, n), r.uniform(0, H * o.get("height", 0.6), n), r.uniform(0, 6.28, n)
    for i in range(n):
        a = 0.5 + 0.5 * math.sin(t * 2 + ph[i])
        circle(img, px[i], py[i], 1 + 1.2 * a * W / 720, mix((90, 100, 150), (255, 255, 255), a))


def d_building(img, o, X, t, ctx):
    x, y, w, h = o["x"], o["y"], o.get("w", 90), o.get("h", 220)
    c = col(o.get("color", (70, 78, 120)))
    fill_poly(img, X.ps([[x - w / 2, y], [x + w / 2, y], [x + w / 2, y - h], [x - w / 2, y - h]]), c)
    r = rng_for("bld", x)
    for yy in np.arange(y - h + 18, y - 20, 26):
        for xx in np.arange(x - w / 2 + 14, x + w / 2 - 10, 22):
            on = r.uniform() < o.get("lit", 0.5)
            fill_poly(img, X.ps([[xx, yy], [xx + 10, yy], [xx + 10, yy + 12], [xx, yy + 12]]),
                      (250, 225, 140) if on else shade(c, 0.7))


# ----------------------------------------------------------------------------- general vector objects
_svg_cache = {}


def _svg_model(o):
    if svgkit is None:
        raise SystemExit("svg/prop objects need engine/svgkit.py next to explainer.py")
    src = o.get("svg")
    if not src and o.get("prop"):
        src = svgkit.PROPS.get(str(o["prop"]).lower())
        if src is None:
            print(f"  ! unknown prop '{o['prop']}' (available: {', '.join(sorted(svgkit.PROPS))})", file=sys.stderr)
            src = svgkit.PROPS["star"]
    key = (src, UI.get("mode"), tuple(sorted((o.get("recolor") or {}).items())))
    if key not in _svg_cache:
        rec = {str(k).lower(): v for k, v in (o.get("recolor") or {}).items()}

        def resolver(name):
            n = name.lower()
            if n in rec:
                return col(rec[n])
            if n in UI and isinstance(UI[n], tuple):
                return UI[n]
            return PAL.get(n)
        model = svgkit.parse_svg(src, resolver)
        if rec:  # recolor hex fills: {"#ef5a6f": "green"}
            for it in model["items"]:
                for k in ("fill", "stroke"):
                    c = it[k]
                    if c is not None:
                        hx = "#%02x%02x%02x" % c
                        if hx in rec:
                            it[k] = col(rec[hx])
        _svg_cache[key] = model
    return _svg_cache[key]


def d_svg(img, o, X, t, ctx):
    """Any flat illustration: {"type":"svg","svg":"<svg viewBox=...>…</svg>"} or {"type":"prop","prop":"lightbulb"}.
    x,y = centre, w = width in world units (default 120), rotate (deg), flip, bob (px of idle float)."""
    m = _svg_model(o)
    vx, vy, vw, vh = m["viewbox"]
    w = o.get("w", o.get("size", 120))
    k = w / max(vw, 1e-6)
    cx, cy = vx + vw / 2, vy + vh / 2
    ang = math.radians(o.get("rotate", 0) + o.get("wobble", 0) * math.sin(t * 2.2 + o.get("x", 0)))
    ca, sa = math.cos(ang), math.sin(ang)
    fx = -1 if o.get("flip") else 1
    ox = o.get("x", 360)
    oy = o.get("y", 640) + o.get("bob", 0) * math.sin(t * 1.8 + ox * 0.01)

    def tf(p):
        q = (p - [cx, cy]) * k
        q[:, 0] *= fx
        r = np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca], 1)
        return X.ps(r + [ox, oy])
    for it in m["items"]:
        subs = [(tf(p), c) for p, c in it["subs"]]

        def draw(im, it=it, subs=subs):
            if it["fill"] is not None:
                closed = [p for p, c in subs if len(p) >= 3]
                if closed:
                    cv2.fillPoly(im, [ipts(p) for p in closed], it["fill"], cv2.LINE_AA, SH)
            if it["stroke"] is not None and it["sw"] > 0:
                for p, c in subs:
                    stroke(im, p, it["stroke"], max(1, X.s(it["sw"] * k)), c)
        if it["opacity"] < 0.99:
            allp = np.concatenate([p for p, _ in subs])
            bb = (allp[:, 0].min() - 4, allp[:, 1].min() - 4, allp[:, 0].max() + 4, allp[:, 1].max() + 4)
            blend_region(img, it["opacity"], bb, draw)
        else:
            draw(img)


def d_ellipse(img, o, X, t, ctx):
    sx, sy = X.p(o["x"], o["y"])
    rx, ry = X.s(o.get("rx", 60)), X.s(o.get("ry", 40))
    ang = o.get("rotate", 0)
    if o.get("fill", "white") != "none":
        cv2.ellipse(img, (int(sx * SHF), int(sy * SHF)), (int(rx * SHF), int(ry * SHF)), ang, 0, 360,
                    col(o.get("fill", "white")), -1, cv2.LINE_AA, SH)
    if o.get("stroke"):
        cv2.ellipse(img, (int(sx * SHF), int(sy * SHF)), (int(rx * SHF), int(ry * SHF)), ang, 0, 360,
                    col(o["stroke"]), max(1, int(X.s(o.get("sw", 3)))), cv2.LINE_AA, SH)


def _catmull(pts, closed, n=10):
    p = np.asarray(pts, float)
    if len(p) < 3:
        return p
    P = np.concatenate([p[-1:], p, p[:2]]) if closed else np.concatenate([p[:1], p, p[-1:]])
    out = []
    segs = len(p) if closed else len(p) - 1
    for i in range(segs):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for j in range(n):
            t = j / n
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(p[-1])
    return np.asarray(out)


def d_path(img, o, X, t, ctx):
    """Smooth organic shape through points (blob, organ, hill, wave, road). closed=true fills."""
    closed = o.get("closed", True)
    pts = _catmull(o["points"], closed)
    if o.get("draw", 1) < 1 and not closed:
        n = max(2, int(len(pts) * clamp(o["draw"])))
        pts = pts[:n]
    sp = X.ps(pts)
    if closed and o.get("fill", "white") != "none":
        fill_poly(img, sp, col(o.get("fill", "white")))
    if o.get("stroke") or not closed:
        stroke(img, sp, col(o.get("stroke", o.get("color", "outline"))), X.s(o.get("sw", o.get("width", 3))), closed)


def d_room(img, o, X, t, ctx):
    """Indoor set: wall + floor (+ baseboard, window, picture). floor = world y of the floor line."""
    fy = o.get("floor", 900)
    x0, x1 = o.get("x0", -2000), o.get("x1", 2800)
    top = o.get("top", -2000)
    wall = col(o.get("wall", (244, 226, 204)))
    floor = col(o.get("floor_color", (196, 150, 110)))
    fill_poly(img, X.ps([[x0, top], [x1, top], [x1, fy], [x0, fy]]), wall)
    fill_poly(img, X.ps([[x0, fy], [x1, fy], [x1, fy + 3000], [x0, fy + 3000]]), floor)
    # floor boards
    if o.get("boards", True):
        for k in range(1, 8):
            yy = fy + k * k * 9
            a, b = X.p(x0, yy), X.p(x1, yy)
            line(img, a[0], a[1], b[0], b[1], shade(floor, 0.88), X.s(2))
    fill_poly(img, X.ps([[x0, fy - 22], [x1, fy - 22], [x1, fy], [x0, fy]]), mix(wall, (255, 255, 255), 0.5))
    stroke(img, X.ps([[x0, fy], [x1, fy]]), shade(floor, 0.7), X.s(3))
    win = o.get("window")
    if win:
        wx, wy, ww, wh = win if isinstance(win, (list, tuple)) else (o.get("x", 360) + 120, fy - 520, 220, 260)
        sky = col(o.get("window_sky", (150, 205, 245)))
        frame = col(o.get("frame", (255, 255, 255)))
        fill_poly(img, X.ps([[wx - ww / 2 - 12, wy - wh / 2 - 12], [wx + ww / 2 + 12, wy - wh / 2 - 12],
                             [wx + ww / 2 + 12, wy + wh / 2 + 12], [wx - ww / 2 - 12, wy + wh / 2 + 12]]), frame)
        fill_poly(img, X.ps([[wx - ww / 2, wy - wh / 2], [wx + ww / 2, wy - wh / 2], [wx + ww / 2, wy + wh / 2], [wx - ww / 2, wy + wh / 2]]), sky)
        a, b = X.p(wx, wy - wh / 2), X.p(wx, wy + wh / 2)
        line(img, a[0], a[1], b[0], b[1], frame, X.s(8))
        a, b = X.p(wx - ww / 2, wy), X.p(wx + ww / 2, wy)
        line(img, a[0], a[1], b[0], b[1], frame, X.s(8))


def d_table(img, o, X, t, ctx):
    x, y, w = o.get("x", 360), o.get("y", 900), o.get("w", 380)
    h = o.get("h", 150)
    c = col(o.get("color", (176, 116, 74)))
    out = col(o.get("stroke", "outline"))
    top = [[x - w / 2 - 10, y - h], [x + w / 2 + 10, y - h], [x + w / 2 + 10, y - h + 20], [x - w / 2 - 10, y - h + 20]]
    for lx in (x - w / 2 + 14, x + w / 2 - 30):
        leg = [[lx, y - h + 20], [lx + 16, y - h + 20], [lx + 16, y], [lx, y]]
        fill_poly(img, X.ps(leg), shade(c, 0.8))
        stroke(img, X.ps(leg), out, X.s(2.5), True)
    fill_poly(img, X.ps(top), c)
    stroke(img, X.ps(top), out, X.s(2.5), True)


DRAW = {
    "ground": d_ground, "sea": d_sea, "water": d_sea, "mountain": d_mountain, "river": d_river,
    "rocks": d_rocks, "cloud": d_cloud, "rain": d_rain, "sun": d_sun, "moon": d_moon,
    "lightning": d_lightning, "house": d_house, "tree": d_tree, "vent": d_vent, "vapor": d_vapor,
    "ruler": d_ruler, "particles": d_particles, "circle": d_circle, "rect": d_rect, "poly": d_poly,
    "line": d_line, "arrow": d_arrow, "text": d_text, "person": d_person, "planet": d_planet,
    "stars": d_stars, "building": d_building,
    "svg": d_svg, "prop": d_svg, "icon": d_svg, "ellipse": d_ellipse, "path": d_path, "blob": d_path,
    "room": d_room, "table": d_table,
}


# ============================================================================= FORMAT / SETTINGS
FORMAT_DEFAULTS = {
    "aspect": "9:16",          # 9:16 | 16:9 | 1:1 | 4:5 | 3:4  (aliases: portrait/potret, landscape/lanskap, square/persegi)
    "quality": 720,            # short side in px: 720 | 1080
    "fps": 30,                 # 24 | 30 | 60
    "captions": True,          # true/false or {show, position, words, highlight, size, uppercase}
    "title": True,             # hook title at the start (needs meta.title)
    "outro": True,             # takeaway + CTA at the end (needs meta.outro)
    "cta": None,               # CTA pill text; null = meta.cta or default; false/"" = no pill
    "sfx": True,
    "sfx_volume_db": 0,        # -6 = quieter, +3 = louder
    "ambience": True,          # allow rain/wind loops from spec.ambience
    "watermark": None,         # e.g. "@akunmu"
    "watermark_pos": "top-right",  # top-right | top-left | bottom-right | bottom-left
    "theme": None,             # alias of background
    "background": None,        # override spec.background (preset name or dict)
}
CAPTION_DEFAULTS = {"show": True, "position": "bottom", "words": 3, "highlight": True, "size": 1.0, "uppercase": False}
ASPECT_ALIASES = {"portrait": "9:16", "potret": "9:16", "vertical": "9:16", "vertikal": "9:16",
                  "landscape": "16:9", "lanskap": "16:9", "horizontal": "16:9", "horisontal": "16:9",
                  "square": "1:1", "persegi": "1:1", "kotak": "1:1", "feed": "4:5"}
SIZE_ALIASES = {"kecil": 0.82, "small": 0.82, "sedang": 1.0, "normal": 1.0, "medium": 1.0, "besar": 1.18, "large": 1.18, "big": 1.18}


def _truthy(v):
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "no", "tidak", "off", "nggak", "gak", "0", "none", "")
    return bool(v)


def resolve_format(spec, base_dir=".", quality=None, fps=None):
    F = dict(FORMAT_DEFAULTS)
    src = spec.get("format", {})
    if isinstance(src, str):  # path to a preset json
        pth = src if os.path.isabs(src) else os.path.join(base_dir, src)
        src = json.load(open(pth, encoding="utf-8"))
        src = src.get("format", src)
    # legacy top-level fields
    for k in ("sfx", "sfx_volume_db", "fps"):
        if k in spec and k not in src:
            F[k] = spec[k]
    F.update({k: v for k, v in src.items() if v is not None or k == "cta"})
    if "resolution" in src:
        F["quality"] = src["resolution"]
    a = str(F["aspect"]).strip().lower().replace("x", ":").replace("/", ":")
    a = ASPECT_ALIASES.get(a, a)
    try:
        aw, ah = [float(v) for v in a.split(":")]
    except Exception:
        aw, ah = 9.0, 16.0
    F["aspect"] = f"{int(aw)}:{int(ah)}"
    q = int(str(quality or F["quality"]).lower().replace("p", ""))
    if aw <= ah:
        W, H = q, q * ah / aw
    else:
        W, H = q * aw / ah, q
    F["W"], F["H"] = int(round(W / 2) * 2), int(round(H / 2) * 2)
    F["fps"] = int(fps or F["fps"])
    cap = dict(CAPTION_DEFAULTS)
    c = F.get("captions", True)
    if isinstance(c, dict):
        cap.update(c)
    else:
        cap["show"] = _truthy(c)
    if "caption_y" in spec:
        cap["y"] = spec["caption_y"]
    cap["size"] = SIZE_ALIASES.get(str(cap["size"]).lower(), cap["size"]) if isinstance(cap["size"], str) else float(cap["size"])
    cap["words"] = max(1, min(5, int(cap["words"])))
    if "y" not in cap:
        ratio = F["H"] / F["W"]
        pos = str(cap["position"]).lower()
        if pos in ("middle", "tengah", "center"):
            cap["y"] = 0.55
        elif pos in ("top", "atas"):
            cap["y"] = 0.22
        else:
            cap["y"] = 0.775 if ratio > 1.5 else 0.8 if ratio > 1.1 else 0.83 if ratio > 0.9 else 0.85
    F["captions"] = cap
    for k in ("title", "outro", "sfx", "ambience"):
        F[k] = _truthy(F[k])
    return F


# ============================================================================= ENGINE
class Engine:
    def __init__(self, spec, base_dir, res=None, fps=None):
        self.spec = spec
        self.base = base_dir
        self.F = F = resolve_format(spec, base_dir, res, fps)
        if "size" in spec and "format" not in spec and not res:
            F["W"], F["H"] = int(spec["size"][0]), int(spec["size"][1])
        self.W, self.H = F["W"], F["H"]
        # world + UI scale: area-preserving vs the 720x1280 design canvas
        self.S = math.sqrt(self.W * self.H / (DESIGN_W * DESIGN_H))
        self.fps = F["fps"]
        self.bg = resolve_background(spec, F)
        set_ui_mode(self.bg["mode"], self.bg.get("accent"))
        # auto-reframe: spec designed for one aspect (default 9:16) rendered in another -> pull camera back
        da = str(spec.get("design_aspect", "9:16")).replace("x", ":")
        try:
            dw, dh = [float(v) for v in ASPECT_ALIASES.get(da, da).split(":")]
        except Exception:
            dw, dh = 9.0, 16.0
        des_h = math.sqrt(DESIGN_W * DESIGN_H * dh / dw)          # world height visible in design aspect
        des_w = DESIGN_W * DESIGN_H / des_h
        vis_h, vis_w = self.H / self.S, self.W / self.S
        f = min(vis_h / des_h, vis_w / des_w)
        self.reframe = f ** 0.75 if (F.get("reframe", True) and f < 0.98) else 1.0
        self.words, self.mode = resolve_words(spec, base_dir)
        if not self.words:
            raise SystemExit("No words: give spec.script (and optionally audio/words/srt)")
        cf = F["captions"]
        wide = self.W > self.H * 1.2
        max_chars = {1: 99, 2: 15, 3: 20, 4: 24, 5: 30}[cf["words"]]
        if wide:
            max_chars = int(max_chars * 1.5)
        self.caps = build_captions(self.words, spec.get("highlight"), max_words=cf["words"], max_chars=max_chars)
        if not cf["highlight"]:
            for c in self.caps:
                c["hl"] = [False] * len(c["hl"])
        last = self.words[-1]["end"]
        self.duration = float(spec.get("duration", last + spec.get("tail", 1.2)))
        self.objects = []
        self.by_id = {}
        for i, o in enumerate(spec.get("world", [])):
            o = dict(o)
            if o.get("type") not in DRAW:
                print(f"  ! unknown object type: {o.get('type')} (id={o.get('id')})", file=sys.stderr)
            o.setdefault("id", f"_o{i}")
            o["_base"] = {k: v for k, v in o.items() if not k.startswith("_")}
            self.objects.append(o)
            self.by_id[o["id"]] = o
        self.tweens = {}   # id -> list of (t0, t1, key, to, ease)
        self.effects = []  # overlay effects
        self.cam = []      # camera keys (t0,t1, from?, to dict)
        self.sfx = []      # (t, name, gain_db)
        self.loops = []    # (t0, t1, name)
        self.events = []
        self._ptr = 0
        self._compile()
        self.sky = self.build_background()

    # ---------------------------------------------------------------- time anchors
    def T(self, at, default=0.0):
        if at is None:
            return default
        if isinstance(at, (int, float)):
            return float(at)
        s = str(at).strip()
        off = 0.0
        m = re.match(r"^(.*?)([+-]\d+(\.\d+)?)s$", s)
        if m:
            s, off = m.group(1).strip(), float(m.group(2))
        if s.lower() in ("start", "begin"):
            return 0.0 + off
        if s.lower() == "end":
            return self.duration + off
        if s.startswith("#"):
            return self.words[min(int(s[1:]), len(self.words) - 1)]["start"] + off
        end_mode = s.startswith("^")
        if end_mode:
            s = s[1:]
        q = [norm_word(x) for x in s.split() if norm_word(x)]
        if not q:
            return default
        n = [w["n"] for w in self.words]

        def find(frm):
            for i in range(frm, len(n) - len(q) + 1):
                if n[i:i + len(q)] == q:
                    return i
            # loose: prefix match
            for i in range(frm, len(n) - len(q) + 1):
                if all(n[i + k].startswith(q[k][:max(3, len(q[k]) - 2)]) for k in range(len(q))):
                    return i
            return None
        i = find(self._ptr)
        if i is None:
            i = find(0)
        if i is None:
            print(f"  ! anchor not found: '{at}' -> using {default:.2f}s", file=sys.stderr)
            return default + off
        self._ptr = i
        w = self.words[i + len(q) - 1] if end_mode else self.words[i]
        return (w["end"] if end_mode else w["start"]) + off

    # ---------------------------------------------------------------- compile beats
    def add_sfx(self, t, name, gain=None, act=None):
        if act is not None and "sfx" in act:
            name = act["sfx"]
        if not name or not self.F["sfx"]:
            return
        if name in SFX.LIB:
            self.sfx.append((max(0, t), name, gain))

    def _compile(self):
        sp = self.spec
        last_t = 0.0
        # camera initial
        c0 = sp.get("camera", {"x": 360, "y": 640, "zoom": 1.0})
        self.cam_init = {"x": c0.get("x", 360), "y": c0.get("y", 640), "zoom": c0.get("zoom", 1.0)}
        for o in self.objects:
            if o.get("hidden"):
                o["_base"]["alpha"] = 0.0
                o["_base"]["_pop"] = 0.0
        for b in sp.get("beats", []):
            t = self.T(b.get("at"), last_t)
            last_t = t
            acts = b.get("do")
            if isinstance(acts, dict):
                acts = [acts]
            if acts is None:
                acts = [{k: v for k, v in b.items() if k != "at"}]
            for a in acts:
                self._action(t + float(a.get("delay", 0)), a)
                kind = a.get("type") or next((k for k in a if k not in ("delay", "dur", "sfx")), "?")
                self.events.append((t, str(b.get("at")), kind))
        # loops for rain/wind ambience
        if sp.get("ambience") and self.F["ambience"] and self.F["sfx"]:
            amb = sp["ambience"]
            if isinstance(amb, str):
                amb = [{"name": amb}]
            for a in amb:
                self.loops.append((self.T(a.get("from"), 0), self.T(a.get("to"), self.duration), a["name"], a.get("gain")))
        # title / outro sfx
        meta = sp.get("meta", {})
        if meta.get("title") and self.F["title"]:
            self.add_sfx(0.05, "whoosh", -21)
            self.add_sfx(0.2, "pop", -18)
        if meta.get("outro") and self.F["outro"]:
            self.add_sfx(self.outro_start(), "chime", -20)

    def build_background(self):
        b = self.bg
        g = np.linspace(0, 1, self.H)[:, None]
        top, bot = np.array(b["top"], float), np.array(b["bottom"], float)
        img = np.repeat((top + (bot - top) * g)[:, None, :], self.W, 1).reshape(self.H, self.W, 3).astype(np.uint8)
        pat = b.get("pattern")
        if pat:
            S = self.S
            light = b["mode"] == "light"
            lc = shade(b["top"], 0.86) if light else mix(b["top"], (255, 255, 255), 0.12)
            ov = img.copy()
            if pat == "grid":
                step = int(48 * S)
                for x in range(step // 2, self.W, step):
                    cv2.line(ov, (x, 0), (x, self.H), lc, max(1, int(1.5 * S)), cv2.LINE_AA)
                for y in range(step // 2, self.H, step):
                    cv2.line(ov, (0, y), (self.W, y), lc, max(1, int(1.5 * S)), cv2.LINE_AA)
            elif pat == "dots":
                step = int(40 * S)
                for y in range(step // 2, self.H, step):
                    for x in range(step // 2 + (step // 2 if (y // step) % 2 else 0), self.W, step):
                        cv2.circle(ov, (x, y), max(1, int(2.6 * S)), lc, -1, cv2.LINE_AA)
            elif pat == "lines":
                step = int(44 * S)
                for y in range(int(120 * S), self.H, step):
                    cv2.line(ov, (0, y), (self.W, y), (190, 210, 235) if light else lc, max(1, int(2 * S)), cv2.LINE_AA)
                cv2.line(ov, (int(70 * S), 0), (int(70 * S), self.H), (240, 150, 150), max(1, int(2 * S)), cv2.LINE_AA)
            img = cv2.addWeighted(ov, 0.7, img, 0.3, 0)
        return img

    def outro_start(self):
        meta = self.spec.get("meta", {})
        if "outro_at" in meta:
            return self.T(meta["outro_at"], self.duration - 3.5)
        return max(0, self.duration - meta.get("outro_dur", 3.4))

    def tw(self, oid, t0, dur, key, to, ease="io"):
        self.tweens.setdefault(oid, []).append((t0, t0 + max(dur, 1e-3), key, to, ease))

    def target_pt(self, tgt):
        if isinstance(tgt, (list, tuple)):
            return ("pt", tuple(tgt))
        if isinstance(tgt, str) and tgt in self.by_id:
            return ("id", tgt)
        return ("pt", (360, 640))

    def _action(self, t, a):
        kind = a.get("type") or a.get("action") or a.get("kind")
        if kind is None:
            for k in ("camera", "show", "hide", "set", "label", "pulse", "card", "strike", "travel",
                      "burst", "bigtext", "shake", "flash", "sfx"):
                if k in a:
                    kind = k
                    break
        dur = float(a.get("dur", 0.8))
        until = a.get("until")
        known = ("camera", "show", "hide", "set", "label", "pulse", "card", "strike", "flash", "travel",
                 "burst", "bigtext", "shake", "sfx")
        if kind not in known:
            print(f"  ! unknown action: {a}", file=sys.stderr)
            return
        for key in ("id", "show", "hide", "set", "strike"):
            v = a.get(key)
            for vv in (v if isinstance(v, list) else [v]):
                if key in ("id", "show", "hide", "set", "strike") and isinstance(vv, str) and kind in ("show", "hide", "set", "strike") \
                        and vv not in self.by_id and vv not in known:
                    print(f"  ! action refers to unknown object id '{vv}'", file=sys.stderr)
        if kind == "camera":
            to = {}
            f = a.get("focus", a.get("camera"))
            if isinstance(f, str) and f in self.by_id:
                ax, ay = anchor_of(self.by_id[f])
                to["x"], to["y"] = ax, ay
            elif isinstance(f, (list, tuple)):
                to["x"], to["y"] = f[0], f[1]
            for k in ("x", "y", "zoom"):
                if k in a:
                    to[k] = a[k]
            if "zoom" in a:
                to["zoom"] = a["zoom"]
            d = float(a.get("dur", 1.2))
            self.cam.append((t, t + d, to, a.get("ease", "io")))
            if a.get("sfx", "auto") == "auto":
                self.add_sfx(t, "whoosh", -22 if d > 0.6 else -20)
            else:
                self.add_sfx(t, a.get("sfx"))
        elif kind == "show":
            ids = a["show"] if isinstance(a.get("show"), list) else [a.get("show", a.get("id"))]
            for i, oid in enumerate(ids):
                tt = t + i * a.get("stagger", 0.12)
                d = float(a.get("dur", 0.45))
                self.tw(oid, tt, d * 0.6, "alpha", 1.0, "out")
                self.tw(oid, tt, d, "_pop", 1.0, "back" if a.get("pop", True) else "out")
                if i < 3:
                    self.add_sfx(tt, "pop", None, a)
        elif kind == "hide":
            ids = a["hide"] if isinstance(a.get("hide"), list) else [a.get("hide", a.get("id"))]
            for oid in ids:
                self.tw(oid, t, float(a.get("dur", 0.4)), "alpha", 0.0, "io")
        elif kind == "set":
            oid = a.get("set", a.get("id"))
            for k, v in a.get("props", {}).items():
                self.tw(oid, t, dur, k, v, a.get("ease", "io"))
            if "sfx" in a:
                self.add_sfx(t, a["sfx"])
        elif kind == "label":
            t1 = self.T(until, t + float(a.get("hold", 2.6))) if until else t + float(a.get("hold", 2.6))
            self.effects.append({"k": "label", "t0": t, "t1": t1, "text": str(a["label"]).upper(),
                                 "tgt": self.target_pt(a.get("target", a.get("on"))),
                                 "at": a.get("point"), "off": a.get("offset", [50, -50])})
            self.add_sfx(t, "typing", None, a)
        elif kind == "pulse":
            self.effects.append({"k": "pulse", "t0": t, "t1": t + float(a.get("dur", 1.6)),
                                 "tgt": self.target_pt(a.get("pulse", a.get("target"))),
                                 "r": a.get("r", 110), "color": a.get("color", "pulse")})
            self.add_sfx(t, "blip", None, a)
        elif kind == "card":
            t1 = self.T(until, t + float(a.get("hold", 3.4))) if until else t + float(a.get("hold", 3.4))
            self.effects.append({"k": "card", "t0": t, "t1": t1, "rows": a.get("rows", []),
                                 "title": a.get("title"), "y": a.get("y", 0.28)})
            self.add_sfx(t, "swish", None, a)
            for i, r in enumerate(a.get("rows", [])):
                self.add_sfx(t + 0.35 + i * 0.55, "rise", -23)
            self.add_sfx(t + 0.35 + len(a.get("rows", [])) * 0.55, "ding", -21)
        elif kind == "strike":
            oid = a.get("strike", a.get("id"))
            n = int(a.get("count", 2))
            for k in range(n):
                tt = t + k * 0.18
                self.tw(oid, tt, 0.05, "_flash", 1.0, "linear")
                self.tw(oid, tt + 0.06, 0.25, "_flash", 0.0, "out")
            self.effects.append({"k": "flash", "t0": t, "t1": t + 0.35, "a": a.get("flash", 0.35)})
            self.add_sfx(t + float(a.get("thunder_delay", 0.05)), "thunder", None, a)
        elif kind == "flash":
            self.effects.append({"k": "flash", "t0": t, "t1": t + float(a.get("dur", 0.3)), "a": a.get("strength", 0.4)})
            self.add_sfx(t, a.get("sfx", None))
        elif kind == "travel":
            d = float(a.get("dur", 1.4))
            self.effects.append({"k": "travel", "t0": t, "t1": t + d + float(a.get("hold", 0.6)), "d": d,
                                 "a": self.target_pt(a.get("from")), "b": self.target_pt(a.get("to")),
                                 "color": a.get("color", "hl"), "n": a.get("n", 9), "arc": a.get("arc", 0.15),
                                 "r": a.get("r", 6)})
            self.add_sfx(t, "sparkle" if a.get("color", "hl") in ("hl", "accent", "yellow", "gold", "white") else "swish", None, a)
        elif kind == "burst":
            self.effects.append({"k": "burst", "t0": t, "t1": t + float(a.get("dur", 1.4)),
                                 "tgt": self.target_pt(a.get("burst", a.get("target"))),
                                 "color": a.get("color", (225, 150, 110)), "n": a.get("n", 4),
                                 "h": a.get("h", 500), "w": a.get("w", 120)})
            self.add_sfx(t, "boom", None, a)
        elif kind == "bigtext":
            self.effects.append({"k": "bigtext", "t0": t, "t1": t + float(a.get("hold", 1.8)), "text": a["bigtext"],
                                 "tgt": self.target_pt(a.get("target")) if a.get("target") else None,
                                 "y": a.get("y", 0.42), "color": a.get("color", "yellow"), "size": a.get("size", 92)})
            self.add_sfx(t, "ding", None, a)
        elif kind == "shake":
            self.effects.append({"k": "shake", "t0": t, "t1": t + float(a.get("dur", 0.45)), "amp": a.get("amp", 9)})
        elif kind == "sfx":
            self.add_sfx(t, a["sfx"], a.get("gain"))

    # ---------------------------------------------------------------- state eval
    def props(self, o, t):
        p = dict(o["_base"])
        p.setdefault("alpha", 1.0)
        p.setdefault("_pop", 1.0)
        p.setdefault("_flash", 0.0)
        for (t0, t1, k, to, e) in self.tweens.get(o["id"], []):
            if t < t0:
                continue
            q = EASES.get(e, ease_io)((t - t0) / (t1 - t0))
            if k.startswith("groups."):
                _, gi, gk = k.split(".")
                gi = int(gi)
                gs = [dict(g) for g in p["groups"]]
                cur = gs[gi].get(gk, 0)
                gs[gi][gk] = lerp(cur, to, q) if not isinstance(to, str) else to
                p["groups"] = gs
            else:
                cur = p.get(k, 0 if not isinstance(to, (list, str)) else to)
                p[k] = lerp(cur, to, q) if not isinstance(to, (str, bool)) else (to if q >= 1 or t >= t0 else cur)
        return p

    def camera(self, t):
        cur = dict(self.cam_init)
        for (t0, t1, to, e) in sorted(self.cam, key=lambda c: c[0]):
            if t < t0:
                break
            q = EASES.get(e, ease_io)((t - t0) / (t1 - t0))
            for k, v in to.items():
                cur[k] = cur[k] + (v - cur[k]) * q
        # gentle idle drift (keeps one-take shot alive)
        cur["zoom"] *= (1 + 0.012 * math.sin(t * 0.35)) * self.reframe
        cur["x"] += 3 * math.sin(t * 0.23)
        for ef in self.effects:
            if ef["k"] == "shake" and ef["t0"] <= t <= ef["t1"]:
                k = 1 - (t - ef["t0"]) / (ef["t1"] - ef["t0"])
                cur["x"] += ef["amp"] * k * math.sin(t * 90)
                cur["y"] += ef["amp"] * k * math.cos(t * 77)
        return cur

    def xf(self, t):
        c = self.camera(t)
        k = c["zoom"] * self.S
        return Xf(k, self.W / 2 - c["x"] * k, self.H / 2 - c["y"] * k), c

    def resolve(self, tgt, X, t, pt_off=None):
        kind, v = tgt
        if kind == "id":
            o = self.by_id[v]
            p = self.props(o, t)
            ax, ay = anchor_of(p)
        else:
            ax, ay = v
        if pt_off:
            ax, ay = ax + pt_off[0], ay + pt_off[1]
        return X.p(ax, ay)

    # ---------------------------------------------------------------- render one frame
    def frame(self, t):
        img = self.sky.copy()
        X, cam = self.xf(t)
        ctx = {"t": t}
        for o in self.objects:
            p = self.props(o, t)
            if p.get("alpha", 1) <= 0.003:
                continue
            fn = DRAW.get(p.get("type"))
            if not fn:
                continue
            pop = p.get("_pop", 1.0)
            Xo = X
            if pop < 0.999:
                ax, ay = anchor_of(p)
                Xo = X.about(ax, ay, max(0.01, pop))
            if p.get("screen"):  # fixed to screen (HUD)
                Xo = Xf(self.S, 0, 0)
            a = p.get("alpha", 1)
            mz = p.get("min_zoom")
            if mz:  # only visible when the camera is zoomed in (micro views)
                a *= clamp((cam["zoom"] - mz * 0.7) / (mz * 0.3))
                if a <= 0.003:
                    continue
            if a < 0.997:
                blend_region(img, a, None, lambda im: fn(im, p, Xo, t, ctx))
            else:
                fn(img, p, Xo, t, ctx)
        self.draw_effects(img, X, t)
        self.draw_captions(img, t)
        self.draw_titles(img, t)
        return img

    def draw_effects(self, img, X, t):
        S = self.S
        for ef in self.effects:
            if not (ef["t0"] <= t <= ef["t1"] + 0.4):
                continue
            k = ef["k"]
            lt = t - ef["t0"]
            out = clamp((ef["t1"] + 0.4 - t) / 0.4)
            if k == "pulse":
                sx, sy = self.resolve(ef["tgt"], X, t)
                d = ef["t1"] - ef["t0"]
                c = col(ef["color"])
                for j, rr in enumerate((0.55, 1.0)):
                    p = clamp(lt / 0.5)
                    R = X.s(ef["r"]) * rr * (0.7 + 0.3 * ease_out(p)) * (1 + 0.06 * lt)
                    a = 0.75 * ease_out(p) * clamp((d - lt) / 0.5 + 0.0)
                    blend_region(img, a, (sx - R - 4, sy - R - 4, sx + R + 4, sy + R + 4),
                                 lambda im, R=R: circle(im, sx, sy, R, c, 2 * S))
            elif k == "flash":
                d = ef["t1"] - ef["t0"]
                a = ef["a"] * (1 - clamp(lt / d)) ** 2 if lt <= d else 0
                if a > 0.01:
                    img[:] = cv2.addWeighted(img, 1 - a, np.full_like(img, 235), a, 0)
            elif k == "travel":
                A = np.array(self.resolve(ef["a"], X, t))
                B = np.array(self.resolve(ef["b"], X, t))
                c = col(ef["color"])
                d = ef["d"]
                head = clamp(lt / d)
                n = int(ef["n"])
                vec = B - A
                nrm = np.array([-vec[1], vec[0]])
                for i in range(n + 1):
                    q = i / n
                    if q > head:
                        break
                    P = A + vec * q + nrm * ef["arc"] * math.sin(math.pi * q) * -1
                    rr = X.s(ef["r"]) * (0.8 + 0.4 * (q == head))
                    blend_region(img, out, (P[0] - 20, P[1] - 20, P[0] + 20, P[1] + 20),
                                 lambda im, P=P, rr=rr: circle(im, P[0], P[1], rr, c))
                if head < 1:
                    P = A + vec * head + nrm * ef["arc"] * math.sin(math.pi * head) * -1
                    paste(img, glow_sprite(X.s(ef["r"]) * 4, c, 0.6), P[0], P[1], out)
                    circle(img, P[0], P[1], X.s(ef["r"]) * 1.3, c)
            elif k == "burst":
                sx, sy = self.resolve(ef["tgt"], X, t)
                c = col(ef["color"])
                n = ef["n"]
                p = ease_out(clamp(lt / 0.9))
                for side in (-1, 1):
                    for i in range(n):
                        yy = sy + min(X.s(ef["h"]), self.H * 0.8) * ((i + 0.5) / n - 0.5)
                        x0 = sx + side * (X.s(ef["w"]) * 0.5 + 60 * S * p + 20 * S)
                        x1 = x0 + side * 46 * S
                        def dr(im, x0=x0, x1=x1, yy=yy):
                            line(im, x0, yy, x1, yy, c, 7 * S)
                            draw_arrowhead(im, (x0, yy), (x1, yy), c, 6 * S)
                        blend_region(img, out * clamp(lt / 0.2), None, dr) if out < 1 or lt < 0.2 else dr(img)
            elif k == "label":
                self.draw_label(img, X, t, ef, lt, out)
            elif k == "card":
                self.draw_card(img, t, ef, lt, out)
            elif k == "bigtext":
                if ef["tgt"]:
                    sx, sy = self.resolve(ef["tgt"], X, t)
                else:
                    sx, sy = self.W / 2, self.H * ef["y"]
                sc = ease_back(clamp(lt / 0.35))
                size = ef["size"] * S * max(sc, 0.05)
                spr = text_sprite([(str(ef["text"]), col(ef["color"]))], int(size), F_TITLE,
                                  stroke_w=max(2, int(size / 14)), stroke_c=(18, 20, 40), shadow=int(4 * S))
                paste(img, spr, sx, sy, out)

    def draw_label(self, img, X, t, ef, lt, out):
        S = self.S
        sx, sy = self.resolve(ef["tgt"], X, t, ef.get("at"))
        ox, oy = ef["off"]
        bx, by = sx + ox * S, sy + oy * S
        a_in = clamp(lt / 0.25)
        alpha = a_in * out
        if alpha <= 0.01:
            return
        n = max(1, int(round(len(ef["text"]) * clamp(lt / 0.35))))
        txt = ef["text"][:n]
        full = text_sprite([(ef["text"], (255, 255, 255))], int(25 * S), F_LABEL, spacing=int(1 * S))
        spr = text_sprite([(txt, (255, 255, 255))], int(25 * S), F_LABEL, spacing=int(1 * S))
        w, h = full.shape[1] + 14 * S, full.shape[0] - 4 * S
        x0 = bx if ox >= 0 else bx - w
        y0 = by - h / 2
        # leader line + dot
        lx = x0 if ox >= 0 else x0 + w

        def dr(im):
            line(im, sx, sy, lx, by, UI["line"], 2 * S)
            circle(im, sx, sy, 4 * S, UI["line"] if UI["mode"] == "light" else (255, 255, 255))
        blend_region(img, alpha * 0.9, None, dr)
        box = round_rect_sprite(w, h, 3 * S, (18, 22, 44), (120, 130, 170), max(1, int(S)), 235)
        paste(img, box, x0, y0, alpha, center=False)
        line(img, x0 + 4 * S, y0 + 5 * S, x0 + 4 * S, y0 + h - 5 * S, (255, 255, 255), 2.5 * S) if alpha > 0.5 else None
        paste(img, spr, x0 + 7 * S, y0 + h / 2 - spr.shape[0] / 2 + 0.5 * S, alpha, center=False)

    def draw_card(self, img, t, ef, lt, out):
        S = self.S
        rows = ef["rows"]
        p_in = ease_back(clamp(lt / 0.4))
        alpha = clamp(lt / 0.2) * out
        if alpha <= 0.01:
            return
        has_label = any(r.get("label") for r in rows)
        cw = 640 * S
        rh = 92 * S
        ch = (len(rows) * rh + 40 * S) + (46 * S if ef.get("title") else 0)
        cx = self.W / 2
        cy = self.H * ef["y"] + (1 - p_in) * 40 * S
        x0, y0 = cx - cw / 2, cy - ch / 2
        card = round_rect_sprite(cw, ch, 18 * S, (17, 22, 48), (250, 205, 60), max(2, int(3 * S)), 245)
        paste(img, card, x0, y0, alpha, center=False)
        yy = y0 + 20 * S
        if ef.get("title"):
            spr = text_sprite([(ef["title"].upper(), (255, 255, 255))], int(28 * S), F_TITLE2, spacing=int(1 * S))
            paste(img, spr, cx, yy + 16 * S, alpha)
            yy += 46 * S
        for i, r in enumerate(rows):
            ry = yy + rh * i + rh / 2
            ix = x0 + 58 * S
            c = col(r.get("color", "blue" if i == 0 else "white"))
            icon = r.get("icon", "wave" if i == 0 else "box")

            def dicon(im, ix=ix, ry=ry, c=c, icon=icon):
                if icon == "wave":
                    pts = [[ix - 26 * S + k * 2 * S, ry + 7 * S * math.sin(k * 0.45)] for k in range(27)]
                    stroke(im, pts, c, 5 * S)
                elif icon == "box":
                    cv2.rectangle(im, (int(ix - 28 * S), int(ry - 20 * S)), (int(ix + 28 * S), int(ry + 20 * S)), (240, 245, 255), -1, cv2.LINE_AA)
                    cv2.rectangle(im, (int(ix - 25 * S), int(ry - 17 * S)), (int(ix + 25 * S), int(ry + 17 * S)), (52, 140, 225), -1, cv2.LINE_AA)
                else:
                    circle(im, ix, ry, 16 * S, c)
            blend_region(img, alpha, (ix - 40 * S, ry - 30 * S, ix + 40 * S, ry + 30 * S), dicon)
            tx0 = x0 + 108 * S
            if has_label:
                spr = text_sprite([(str(r.get("label", "")).upper(), (220, 226, 245))], int(21 * S), F_LABEL)
                paste(img, spr, tx0, ry - 22 * S - spr.shape[0] / 2 + 6 * S, alpha, center=False)
            tx1 = x0 + cw - 34 * S
            ty = ry + (8 * S if has_label else 0)
            v = clamp(r.get("value", 0.5)) * ease_out(clamp((lt - 0.35 - i * 0.55) / 0.7))
            bar_c = col(r.get("bar", "lightblue" if i == 0 else "white"))

            def dbar(im, ty=ty, v=v, bar_c=bar_c):
                line(im, tx0, ty, tx1, ty, (38, 50, 92), 26 * S)
                circle(im, tx0, ty, 13 * S, (38, 50, 92))
                circle(im, tx1, ty, 13 * S, (38, 50, 92))
                e = tx0 + (tx1 - tx0) * max(v, 0.012)
                line(im, tx0, ty, e, ty, bar_c, 22 * S)
                circle(im, tx0, ty, 11 * S, bar_c)
                circle(im, e, ty, 11 * S, bar_c)
            blend_region(img, alpha, (tx0 - 20 * S, ty - 20 * S, tx1 + 20 * S, ty + 20 * S), dbar)
            if r.get("text"):
                spr = text_sprite([(str(r["text"]), (255, 255, 255))], int(23 * S), F_LABEL, stroke_w=2)
                paste(img, spr, tx1 - spr.shape[1] / 2 - 6 * S, ty - 34 * S, alpha * clamp(v * 3))

    def draw_captions(self, img, t):
        S = self.S
        cf = self.F["captions"]
        if not cf["show"]:
            return
        y = self.H * cf["y"]
        for c in self.caps:
            if c["start"] <= t < c["end"]:
                up = cf.get("uppercase")
                parts = [(w.upper() if up else w, CAP_HL if h else (255, 255, 255)) for w, h in zip(c["words"], c["hl"])]
                lt = t - c["start"]
                sc = 0.9 + 0.1 * ease_back(clamp(lt / 0.16))
                size = int(50 * S * sc * cf["size"] * (1.15 if cf["words"] == 1 else 1.0))
                spr = text_sprite(parts, size, F_CAPTION, stroke_w=max(2, int(5 * S)), stroke_c=(22, 22, 40),
                                  shadow=int(3 * S), max_w=int(self.W * 0.9))
                paste(img, spr, self.W / 2, y)
                break

    def draw_titles(self, img, t):
        S = self.S
        meta = self.spec.get("meta", {})
        segs = []
        if meta.get("title") and self.F["title"]:
            t1 = float(meta.get("title_dur", min(3.2, self.duration * 0.12 + 1.2)))
            segs.append((0.0, t1, meta["title"], None))
        if meta.get("outro") and self.F["outro"]:
            cta = self.F.get("cta")
            if cta is None:
                cta = meta.get("cta", "FOLLOW FOR MORE")
            segs.append((self.outro_start(), self.duration + 1, meta["outro"], cta if cta else None))
        self.draw_watermark(img)
        for (t0, t1, lines, cta) in segs:
            if not (t0 <= t <= t1):
                continue
            lt = t - t0
            a = clamp(lt / 0.3) * clamp((t1 - t) / 0.35)
            if a <= 0.01:
                continue
            band_h = int(min(320 * S, 0.34 * self.H))
            g = (np.linspace(0.75, 0, band_h) ** 1.4)[:, None, None] * a
            img[:band_h] = (img[:band_h] * (1 - g) + np.array(UI["band"]) * g).astype(np.uint8)
            if isinstance(lines, str):
                lines = [lines]
            yy = 70 * S
            for i, ln in enumerate(lines):
                sc = ease_back(clamp((lt - i * 0.12) / 0.35))
                if sc <= 0.02:
                    continue
                c = UI["title1"] if i == 0 else UI["title2"]
                spr = text_sprite([(ln.upper(), c)], int(52 * S * (0.85 + 0.15 * sc)), F_TITLE,
                                  stroke_w=0, shadow=int(3 * S), spacing=int(1 * S), max_w=int(self.W * 0.9))
                paste(img, spr, self.W / 2, yy, a)
                yy += 58 * S
            if cta:
                sc = ease_back(clamp((lt - 0.35) / 0.35))
                if sc > 0.05:
                    tspr = text_sprite([(cta.upper(), (16, 24, 52))], int(27 * S), F_TITLE2, spacing=int(2 * S))
                    w, h = tspr.shape[1] + 34 * S, tspr.shape[0] + 6 * S
                    pill = round_rect_sprite(w * sc, h * sc, h * sc / 2, (120, 200, 255), None, 0, 255)
                    paste(img, pill, self.W / 2, yy + 8 * S, a)
                    if sc > 0.8:
                        paste(img, tspr, self.W / 2, yy + 8 * S, a)

    def draw_watermark(self, img):
        wm = self.F.get("watermark")
        if not wm:
            return
        S = self.S
        spr = text_sprite([(str(wm), (255, 255, 255))], int(20 * S), F_LABEL, stroke_w=max(1, int(2 * S)), stroke_c=(15, 15, 30))
        pos = str(self.F.get("watermark_pos", "top-right"))
        m = 22 * S
        x = self.W - spr.shape[1] - m if "right" in pos else m
        y = self.H - spr.shape[0] - m if "bottom" in pos else m
        if "top" in pos and self.H > self.W:
            y = m + 4 * S
        paste(img, spr, x, y, 0.6, center=False)

    # ---------------------------------------------------------------- audio
    def mix_audio(self, out_wav):
        sr = SFX.SR
        n = int((self.duration + 0.5) * sr)
        buf = np.zeros(n, np.float32)
        if self.spec.get("audio"):
            ap = self.spec["audio"]
            ap = ap if os.path.isabs(ap) else os.path.join(self.base, ap)
            v = load_audio(ap, sr)
            v = v / (np.max(np.abs(v)) + 1e-9) * 0.89
            m = min(n, len(v))
            buf[:m] += v[:m]
        gscale = 10 ** (float(self.F.get("sfx_volume_db", 0)) / 20)
        for (t, name, gain) in self.sfx:
            s = SFX.get(name)
            g = 10 ** ((gain if gain is not None else SFX.GAIN_DB.get(name, -20)) / 20) * gscale
            a = int(t * sr)
            if a >= n:
                continue
            m = min(len(s), n - a)
            buf[a:a + m] += s[:m] * g
        for (t0, t1, name, gain) in self.loops:
            d = max(0.5, t1 - t0)
            s = SFX.LOOPS[name](d)
            g = 10 ** ((gain if gain is not None else SFX.GAIN_DB.get(name, -27)) / 20) * gscale
            a = int(t0 * sr)
            m = min(len(s), n - a)
            if m > 0:
                buf[a:a + m] += s[:m] * g
        pk = np.max(np.abs(buf)) + 1e-9
        if pk > 0.98:
            buf = buf / pk * 0.98
        pcm = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())

    # ---------------------------------------------------------------- video
    def render_range(self, f0, f1, path, crf=20, preset="veryfast"):
        cmd = [ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.W}x{self.H}", "-r", str(self.fps), "-i", "-",
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", path]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(f0, f1):
            pr.stdin.write(self.frame(f / self.fps).tobytes())
        pr.stdin.close()
        pr.wait()
        if pr.returncode:
            raise RuntimeError("ffmpeg encode failed")

    def nframes(self):
        return int(math.ceil(self.duration * self.fps))

    def mux(self, video_parts, out):
        wav = out + ".mix.wav"
        self.mix_audio(wav)
        lst = out + ".parts.txt"
        with open(lst, "w") as f:
            for p in video_parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        cmd = [ffmpeg_bin(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-movflags", "+faststart", "-shortest", out]
        subprocess.run(cmd, check=True)
        os.remove(lst)
        os.remove(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--preview", default=None, help="comma list of seconds -> PNG stills (contact sheet)")
    ap.add_argument("--part", default=None, help="i/n : render only part i of n (video only)")
    ap.add_argument("--finalize", type=int, default=None, help="n : join n parts + audio")
    ap.add_argument("--timing", action="store_true", help="print captions + beat times")
    ap.add_argument("--check", action="store_true", help="validate spec: anchors, ids, beat times")
    ap.add_argument("--res", type=int, default=None, help="quality = short side in px (720 or 1080); overrides format.quality")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--draft", action="store_true", help="fast preview: 360 wide, 15 fps")
    ap.add_argument("--transcribe", default=None, help="model dir: treat SPEC arg as an audio file, print transcript")
    ap.add_argument("--lang", default="id", help="Whisper language code (id, en, …) or auto")
    ap.add_argument("--compare", default=None, help="with --transcribe: naskah .txt to compare against")
    a = ap.parse_args()
    if a.transcribe:
        txt = transcribe(a.spec, a.transcribe, a.lang)
        out = a.out or os.path.splitext(a.spec)[0] + ".transcript.txt"
        open(out, "w", encoding="utf-8").write(txt)
        print(txt)
        print("transcript ->", out, file=sys.stderr)
        if a.compare:
            ref = open(a.compare, encoding="utf-8").read()
            ratio, diffs = compare_texts(txt, ref)
            print(f"\n--- compare with {os.path.basename(a.compare)}: {ratio * 100:.0f}% word agreement ---")
            for op, heard, wrote in diffs[:40]:
                print(f"  {op:8s} audio: '{heard}'  |  naskah: '{wrote}'")
            if len(diffs) > 40:
                print(f"  … {len(diffs) - 40} more")
        return
    spec = json.load(open(a.spec, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(a.spec))
    res, fps = a.res, a.fps
    if a.draft:
        res, fps = 360, 15
    E = Engine(spec, base, res, fps)
    out = a.out or os.path.splitext(a.spec)[0] + ".mp4"
    stem = os.path.splitext(out)[0]
    F = E.F
    print(f"mode={E.mode} words={len(E.words)} duration={E.duration:.2f}s frames={E.nframes()} size={E.W}x{E.H} "
          f"aspect={F['aspect']} fps={E.fps} captions={'on' if F['captions']['show'] else 'off'}"
          f"({F['captions']['position']},{F['captions']['words']}w) title={F['title']} outro={F['outro']} "
          f"sfx={F['sfx']}({F['sfx_volume_db']:+}dB) watermark={F.get('watermark')}")
    write_srt(E.caps, stem + ".srt")
    json.dump([{k: w[k] for k in ("w", "start", "end")} for w in E.words], open(stem + ".words.json", "w"), ensure_ascii=False, indent=0)
    if a.timing or a.check:
        if a.timing:
            for c in E.caps:
                print(f"{c['start']:6.2f}-{c['end']:6.2f}  {' '.join(c['words'])}")
        print("--- beats ---")
        for t, at, kind in E.events:
            print(f"{t:6.2f}s  {kind:8s} at '{at}'")
        gaps = [b[0] - a_[0] for a_, b in zip(E.events, E.events[1:]) if b[0] - a_[0] > 6]
        if gaps:
            print(f"  ! {len(gaps)} gap(s) > 6s without any beat: add camera moves/labels so the video never stalls")
        return
    if a.preview:
        ts = [float(x) for x in a.preview.split(",")]
        ims = []
        for t in ts:
            im = E.frame(t)
            cv2.putText(im, f"{t:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 80), 2)
            pw = 360 if E.H >= E.W else 560
            ims.append(cv2.resize(im, (pw, E.H * pw // E.W)))
        cols = min(len(ims), 5)
        rows = (len(ims) + cols - 1) // cols
        h, w = ims[0].shape[:2]
        sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i, im in enumerate(ims):
            sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
        Image.fromarray(sheet).save(stem + "_preview.png")
        print("preview ->", stem + "_preview.png")
        return
    N = E.nframes()
    if a.part:
        i, n = [int(v) for v in a.part.split("/")]
        f0, f1 = N * (i - 1) // n, N * i // n
        E.render_range(f0, f1, f"{stem}.part{i}of{n}.mp4")
        print(f"part {i}/{n} done ({f1 - f0} frames)")
        return
    if a.finalize:
        n = a.finalize
        parts = [f"{stem}.part{i}of{n}.mp4" for i in range(1, n + 1)]
        E.mux(parts, out)
        for p in parts:
            os.remove(p)
        print("done ->", out)
        return
    tmp = stem + ".video.mp4"
    E.render_range(0, N, tmp)
    E.mux([tmp], out)
    os.remove(tmp)
    print("done ->", out)


if __name__ == "__main__":
    main()
