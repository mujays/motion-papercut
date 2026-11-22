#!/usr/bin/env python3
"""
PAPERCUT ENGINE — paper-cutout (papercraft) motion explainer videos.

Look (fixed): textured night-purple paper sky with tiny paper stars, layered paper hills with torn white edges and
pine trees, every element a cut-paper piece with a soft drop shadow, chunky sticker lettering (cream border,
coral / yellow / teal letters popping in one by one), hook title that shrinks into a topic chip, paper tags,
polaroid cards, counters, bar charts, starburst badges, takeaway strip. One continuous camera (pans / zooms,
parallax hills). Procedural music bed + paper SFX. Optional narration with auto-sync + paper-strip captions.

Usage:
  python papercut.py spec.json --check
  python papercut.py spec.json --preview 1,4,8,12
  python papercut.py spec.json -o out.mp4
  python papercut.py spec.json -o out.mp4 --part 1/3   ...  --finalize 3
  python papercut.py voice.mp3 --transcribe <whisper-model-dir> --lang id -o transcript.txt [--compare naskah.txt]
"""
import argparse, json, math, os, re, subprocess, sys, wave

import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explainer as EX  # noqa: E402  (timing, captions, format, ffmpeg helpers)
import paperkit as PK  # noqa: E402
import music as MU  # noqa: E402
import svgkit  # noqa: E402

FONT_DIR = os.path.join(HERE, "fonts")
F_STICKER = os.path.join(FONT_DIR, "Fredoka-Bold.ttf")
F_SEMI = os.path.join(FONT_DIR, "Fredoka-SemiBold.ttf")
F_HEAVY = os.path.join(FONT_DIR, "Montserrat-Black.ttf")
F_LABEL = os.path.join(FONT_DIR, "Rubik-Bold.ttf")
DESIGN_W, DESIGN_H = 720, 1280
col = PK.col
clamp, ease_io, ease_out, ease_back, lerp = EX.clamp, EX.ease_io, EX.ease_out, EX.ease_back, EX.lerp


def ease_in(p):
    p = clamp(p)
    return p ** 3


def ease_bounce(p):
    p = clamp(p)
    n1, d1 = 7.5625, 2.75
    if p < 1 / d1:
        return n1 * p * p
    if p < 2 / d1:
        p -= 1.5 / d1
        return n1 * p * p + 0.75
    if p < 2.5 / d1:
        p -= 2.25 / d1
        return n1 * p * p + 0.9375
    p -= 2.625 / d1
    return n1 * p * p + 0.984375


EASES = {"io": ease_io, "out": ease_out, "back": ease_back, "linear": clamp, "in": ease_in, "bounce": ease_bounce}

# ============================================================================= BACKGROUNDS
# sky gradient, star count, landscape colours (back, mid, front, fore), tree, grass
BACKGROUNDS = {
    "night": dict(top=(48, 42, 88), bottom=(92, 82, 138), stars=120,
                  land=[(116, 108, 168), (78, 108, 142), (46, 104, 92), (36, 86, 76)], tree=(38, 82, 78), grass=(28, 66, 60)),
    "dusk": dict(top=(62, 48, 102), bottom=(214, 138, 128), stars=45,
                 land=[(152, 112, 162), (108, 96, 142), (70, 112, 100), (52, 92, 84)], tree=(48, 80, 80), grass=(40, 70, 64)),
    "space": dict(top=(16, 16, 38), bottom=(44, 36, 82), stars=240,
                  land=[(90, 84, 140), (64, 74, 118), (52, 60, 98), (40, 46, 80)], tree=(36, 44, 74), grass=(30, 36, 64)),
    "deep": dict(top=(24, 50, 92), bottom=(40, 104, 140), stars=0,
                 land=[(60, 110, 150), (44, 92, 128), (34, 120, 116), (26, 96, 96)], tree=(24, 80, 84), grass=(20, 70, 74)),
    "day": dict(top=(132, 192, 230), bottom=(222, 238, 244), stars=0,
                land=[(150, 186, 212), (118, 178, 142), (86, 158, 110), (66, 134, 92)], tree=(46, 112, 80), grass=(44, 104, 72)),
    "kraft": dict(top=(216, 186, 142), bottom=(198, 162, 118), stars=0,
                  land=[(184, 146, 106), (150, 124, 92), (112, 142, 104), (92, 120, 88)], tree=(78, 104, 74), grass=(70, 94, 66)),
    "mint": dict(top=(212, 238, 226), bottom=(178, 222, 202), stars=0,
                 land=[(160, 206, 196), (120, 186, 166), (88, 160, 128), (68, 138, 108)], tree=(52, 116, 92), grass=(48, 106, 84)),
    "peach": dict(top=(252, 226, 204), bottom=(246, 196, 170), stars=0,
                  land=[(236, 170, 150), (206, 150, 140), (130, 160, 120), (104, 138, 100)], tree=(84, 120, 88), grass=(76, 110, 80)),
}
BACKGROUNDS["paper"] = BACKGROUNDS["kraft"]

# ============================================================================= PAPER PROPS (house style, no outlines)
PROPS = {
    "brain": '<svg viewBox="0 0 120 100"><path d="M30 86 C12 86 5 70 13 59 C3 49 9 30 24 28 C25 14 43 7 54 15 C62 5 82 7 88 19 C103 17 115 32 108 46 C118 56 113 77 98 79 C96 91 80 95 70 89 C62 97 40 97 30 86Z" fill="#EE9AA8"/><path d="M60 18 L60 88" stroke="#D9808F" stroke-width="2.5" fill="none"/><path d="M28 40 L41 33 L39 47 L50 42 M24 62 L36 58 L33 70 M42 76 L52 70 M44 22 L52 28 M72 26 L84 34 L74 40 M88 50 L98 44 M70 56 L82 52 L78 66 L90 64 M74 80 L86 76" stroke="#F8CDD4" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "cat": '<svg viewBox="0 0 100 100"><path d="M18 40 L22 8 L42 28 Z" fill="#E8955F"/><path d="M82 40 L78 8 L58 28 Z" fill="#E8955F"/><path d="M24 30 L26 16 L36 27 Z M76 30 L74 16 L64 27 Z" fill="#F6C9A8"/><ellipse cx="50" cy="56" rx="36" ry="32" fill="#EDA36A"/><ellipse cx="50" cy="70" rx="17" ry="12" fill="#F7D7B8"/><circle cx="36" cy="54" r="4.5" fill="#3B3A5C"/><circle cx="64" cy="54" r="4.5" fill="#3B3A5C"/><path d="M45 64 L55 64 L50 70 Z" fill="#C2645A"/></svg>',
    "dog": '<svg viewBox="0 0 100 100"><ellipse cx="50" cy="54" rx="33" ry="32" fill="#C8955F"/><path d="M20 28 C6 34 6 62 16 70 C24 66 26 48 28 34 Z" fill="#8A5E3C"/><path d="M80 28 C94 34 94 62 84 70 C76 66 74 48 72 34 Z" fill="#8A5E3C"/><ellipse cx="50" cy="70" rx="18" ry="13" fill="#F1DDBF"/><circle cx="38" cy="50" r="4.5" fill="#3B3A5C"/><circle cx="62" cy="50" r="4.5" fill="#3B3A5C"/><ellipse cx="50" cy="63" rx="6" ry="4.5" fill="#3B3A5C"/></svg>',
    "pine": '<svg viewBox="0 0 60 100"><rect x="26" y="78" width="8" height="18" fill="#6E4B36"/><path d="M30 4 L48 36 L12 36 Z M30 22 L52 58 L8 58 Z M30 40 L56 82 L4 82 Z" fill="#2F6B5E"/></svg>',
    "cloud": '<svg viewBox="0 0 120 70"><path d="M20 60 C6 60 4 42 16 38 C14 24 32 16 42 24 C48 8 76 8 80 26 C92 18 110 26 106 42 C118 46 114 60 102 60 Z" fill="#E9E3F2"/></svg>',
    "moon": '<svg viewBox="0 0 100 100"><path d="M62 8 C38 10 22 30 22 52 C22 76 42 94 66 92 C50 84 40 68 40 50 C40 32 50 16 62 8 Z" fill="#F6E7B8"/><circle cx="44" cy="40" r="4" fill="#E8D49A"/><circle cx="38" cy="64" r="6" fill="#E8D49A"/></svg>',
    "sun": '<svg viewBox="0 0 100 100"><path d="M50 2 L58 20 L76 10 L74 30 L94 30 L82 46 L98 58 L78 64 L84 84 L64 78 L58 98 L48 82 L34 96 L32 76 L12 80 L22 62 L4 52 L22 42 L12 24 L32 26 L34 6 L46 20 Z" fill="#F2C552"/><circle cx="50" cy="50" r="24" fill="#F6D776"/></svg>',
    "star": '<svg viewBox="0 0 100 100"><path d="M50 6 L62 36 L94 38 L69 58 L78 90 L50 72 L22 90 L31 58 L6 38 L38 36 Z" fill="#F2C552"/></svg>',
    "house": '<svg viewBox="0 0 100 100"><path d="M10 48 L50 12 L90 48 Z" fill="#E8775F"/><rect x="18" y="46" width="64" height="46" fill="#F4EEDC"/><rect x="42" y="64" width="16" height="28" fill="#74C6B4"/><rect x="24" y="56" width="14" height="12" fill="#F2C552"/><rect x="62" y="56" width="14" height="12" fill="#F2C552"/></svg>',
    "person": '<svg viewBox="0 0 100 100"><circle cx="50" cy="24" r="15" fill="#F2C4A0"/><path d="M24 96 C24 64 34 46 50 46 C66 46 76 64 76 96 Z" fill="#74C6B4"/><path d="M36 14 C40 4 62 4 66 16 C58 12 46 12 36 14 Z" fill="#3B3A5C"/></svg>',
    "heart": '<svg viewBox="0 0 100 100"><path d="M50 88 C20 66 8 50 8 32 C8 18 19 9 31 9 C40 9 46 14 50 21 C54 14 60 9 69 9 C81 9 92 18 92 32 C92 50 80 66 50 88 Z" fill="#E8775F"/></svg>',
    "bulb": '<svg viewBox="0 0 100 100"><path d="M50 8 C28 8 18 26 22 42 C25 54 36 60 37 72 L63 72 C64 60 75 54 78 42 C82 26 72 8 50 8 Z" fill="#F2C552"/><path d="M50 18 C40 18 32 26 32 36" fill="none" stroke="#FBE7A6" stroke-width="5" stroke-linecap="round"/><rect x="37" y="72" width="26" height="16" rx="3" fill="#A09EAC"/></svg>',
    "coin": '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="#ECB640"/><circle cx="50" cy="50" r="30" fill="#F2C552"/><path d="M50 32 L50 68" stroke="#D99A1E" stroke-width="7" stroke-linecap="round"/></svg>',
    "book": '<svg viewBox="0 0 100 100"><path d="M10 22 L48 28 L48 86 L10 80 Z" fill="#74C6B4"/><path d="M90 22 L52 28 L52 86 L90 80 Z" fill="#5FAF9E"/><path d="M16 34 L42 38 M16 46 L42 50 M58 38 L84 34 M58 50 L84 46" stroke="#F4EEDC" stroke-width="3" stroke-linecap="round"/></svg>',
    "phone": '<svg viewBox="0 0 100 100"><rect x="28" y="6" width="44" height="88" rx="8" fill="#3B3A5C"/><rect x="33" y="15" width="34" height="64" rx="3" fill="#A8D4E6"/><circle cx="50" cy="86" r="3.5" fill="#F4EEDC"/></svg>',
    "gear": '<svg viewBox="0 0 100 100"><path d="M44 6 L56 6 L58 18 L68 22 L78 14 L86 22 L78 32 L82 42 L94 44 L94 56 L82 58 L78 68 L86 78 L78 86 L68 78 L58 82 L56 94 L44 94 L42 82 L32 78 L22 86 L14 78 L22 68 L18 58 L6 56 L6 44 L18 42 L22 32 L14 22 L22 14 L32 22 L42 18 Z" fill="#A09EAC"/><circle cx="50" cy="50" r="14" fill="#5A5878"/></svg>',
    "rocket": '<svg viewBox="0 0 100 100"><path d="M50 6 C66 18 70 40 66 66 L34 66 C30 40 34 18 50 6 Z" fill="#F4EEDC"/><path d="M34 52 L18 74 L34 70 Z M66 52 L82 74 L66 70 Z" fill="#E8775F"/><circle cx="50" cy="36" r="8" fill="#74C6B4"/><path d="M40 68 L50 92 L60 68 Z" fill="#F2C552"/></svg>',
    "drop": '<svg viewBox="0 0 100 100"><path d="M50 6 C62 28 80 44 80 62 C80 80 66 92 50 92 C34 92 20 80 20 62 C20 44 38 28 50 6 Z" fill="#80BADE"/><path d="M34 62 C34 72 40 78 46 80" fill="none" stroke="#CDE7F4" stroke-width="5" stroke-linecap="round"/></svg>',
    "leaf": '<svg viewBox="0 0 100 100"><path d="M14 86 C14 40 44 12 90 10 C88 56 62 86 14 86 Z" fill="#5EAA6A"/><path d="M14 86 C34 62 54 42 78 22" fill="none" stroke="#3F8A50" stroke-width="4" stroke-linecap="round"/></svg>',
    "clock": '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="42" fill="#F4EEDC"/><path d="M50 50 L50 24 M50 50 L68 60" stroke="#3B3A5C" stroke-width="6" stroke-linecap="round"/><circle cx="50" cy="50" r="5" fill="#E8775F"/></svg>',
    "magnifier": '<svg viewBox="0 0 100 100"><path d="M62 62 L88 88" stroke="#966848" stroke-width="13" stroke-linecap="round"/><circle cx="42" cy="42" r="30" fill="#F4EEDC"/><circle cx="42" cy="42" r="22" fill="#A8D4E6"/></svg>',
    "laptop": '<svg viewBox="0 0 100 100"><rect x="18" y="20" width="64" height="44" rx="4" fill="#3B3A5C"/><rect x="23" y="25" width="54" height="34" rx="2" fill="#A8D4E6"/><path d="M8 68 L92 68 L86 78 L14 78 Z" fill="#A09EAC"/></svg>',
    "flag": '<svg viewBox="0 0 100 100"><rect x="18" y="8" width="6" height="88" rx="3" fill="#966848"/><path d="M24 12 L84 22 L24 46 Z" fill="#E8775F"/></svg>',
    "trophy": '<svg viewBox="0 0 100 100"><path d="M28 12 L72 12 L70 40 C68 54 60 60 50 60 C40 60 32 54 30 40 Z" fill="#F2C552"/><path d="M28 18 C12 18 12 40 30 40 M72 18 C88 18 88 40 70 40" fill="none" stroke="#ECB640" stroke-width="6"/><rect x="44" y="60" width="12" height="16" fill="#ECB640"/><rect x="30" y="76" width="40" height="12" rx="3" fill="#E8775F"/></svg>',
    "box": '<svg viewBox="0 0 100 100"><path d="M10 34 L50 18 L90 34 L50 50 Z" fill="#E2C08E"/><path d="M10 34 L50 50 L50 92 L10 76 Z" fill="#CEA674"/><path d="M90 34 L50 50 L50 92 L90 76 Z" fill="#B88F5E"/></svg>',
    "atom": '<svg viewBox="0 0 100 100"><g fill="none" stroke="#74C6B4" stroke-width="5"><ellipse cx="50" cy="50" rx="42" ry="15"/><ellipse cx="50" cy="50" rx="42" ry="15" transform="rotate(60 50 50)"/><ellipse cx="50" cy="50" rx="42" ry="15" transform="rotate(-60 50 50)"/></g><circle cx="50" cy="50" r="9" fill="#E8775F"/></svg>',
    "car": '<svg viewBox="0 0 100 100"><path d="M8 66 L8 52 L22 48 L32 32 L66 32 L78 48 L92 52 L92 66 Z" fill="#E8775F"/><path d="M36 38 L48 38 L48 48 L28 48 Z M54 38 L64 38 L72 48 L54 48 Z" fill="#A8D4E6"/><circle cx="28" cy="68" r="10" fill="#3B3A5C"/><circle cx="72" cy="68" r="10" fill="#3B3A5C"/></svg>',
    "tree": '<svg viewBox="0 0 100 100"><rect x="44" y="60" width="12" height="34" fill="#6E4B36"/><circle cx="50" cy="40" r="30" fill="#5EAA6A"/><circle cx="38" cy="32" r="10" fill="#7CC086"/></svg>',
    "fish": '<svg viewBox="0 0 100 60"><path d="M10 30 C24 6 62 4 78 30 C62 56 24 54 10 30 Z" fill="#E8955F"/><path d="M76 30 L96 12 L94 48 Z" fill="#E8775F"/><circle cx="28" cy="26" r="4" fill="#3B3A5C"/></svg>',
    "planet": '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="30" fill="#74C6B4"/><path d="M30 40 C40 34 52 44 62 36 M28 58 C40 62 54 54 70 60" stroke="#A6DDCE" stroke-width="6" fill="none" stroke-linecap="round"/><ellipse cx="50" cy="52" rx="48" ry="10" fill="none" stroke="#F2C552" stroke-width="5"/></svg>',
}


def prop_svg(name):
    n = str(name).lower()
    if n in PROPS:
        return PROPS[n]
    if n in svgkit.PROPS:
        return svgkit.PROPS[n]
    print(f"  ! unknown prop '{name}' (available: {', '.join(sorted(set(PROPS) | set(svgkit.PROPS)))})", file=sys.stderr)
    return PROPS["star"]


def _resolver(name):
    n = name.lower()
    if n in ("outline", "ink", "line"):
        return "none" if n == "outline" else PK.PAL["ink"]
    if n in PK.PAL:
        return PK.PAL[n]
    return None


_svg_models = {}


def svg_model(src):
    if src not in _svg_models:
        _svg_models[src] = svgkit.parse_svg(src, _resolver)
    return _svg_models[src]


# ============================================================================= SPRITE BUILDERS
def draw_svg(cv, src, cx, cy, w, recolor=None, lift=0.9):
    m = svg_model(src)
    vx, vy, vw, vh = m["viewbox"]
    k = w / max(vw, 1e-6)
    rec = {str(a).lower(): col(b) for a, b in (recolor or {}).items()}

    def tf(p):
        return (np.asarray(p, float) - [vx + vw / 2, vy + vh / 2]) * k + [cx, cy]
    for i, it in enumerate(m["items"]):
        lf = 0.0 if i == 0 else lift
        if it["fill"] is not None:
            c = it["fill"]
            c = rec.get("#%02x%02x%02x" % tuple(c), c)
            closed = [tf(p) for p, cl in it["subs"] if len(p) >= 3]
            if closed:
                cv.polys(closed, c, lf, it["opacity"])
        if it["stroke"] is not None and it["sw"] > 0:
            c = it["stroke"]
            c = rec.get("#%02x%02x%02x" % tuple(c), c)
            for p, cl in it["subs"]:
                cv.stroke(tf(p), it["sw"] * k, c, lf, cl, it["opacity"])


def svg_size(src, w):
    vx, vy, vw, vh = svg_model(src)["viewbox"]
    return w, w * vh / max(vw, 1e-6)


def text_w(txt, size, fp=F_STICKER):
    f = PK.font(fp, 200)
    return f.getlength(txt) / 200 * size


# ============================================================================= FORMAT
FORMAT_EXTRA = {"music": "auto", "music_mood": "playful", "music_volume_db": 0, "chip": True}


def resolve_format(spec, base, res=None, fps=None):
    src = spec.get("format", {})
    if isinstance(src, str):
        p = src if os.path.isabs(src) else os.path.join(base, src)
        src = json.load(open(p, encoding="utf-8"))
        src = src.get("format", src)
        spec = dict(spec, format=src)
    F = EX.resolve_format(spec, base, res, fps)
    for k, v in FORMAT_EXTRA.items():
        F[k] = src.get(k, v) if isinstance(src, dict) else v
    F["_captions_given"] = isinstance(src, dict) and "captions" in src
    return F


# ============================================================================= ENGINE
OBJ_TYPES = ("landscape", "hill", "shape", "prop", "svg", "text", "tag", "photo", "bubble", "arrow", "ring", "mark",
             "burst", "counter", "bars", "conveyor", "dials", "strip", "group")
ACTIONS = ("camera", "show", "hide", "set", "label", "pulse", "count", "flow", "bigtext", "draw", "grow",
           "takeaway", "shake", "sfx", "swap")


class Engine:
    def __init__(self, spec, base, res=None, fps=None):
        self.spec, self.base = spec, base
        self.F = F = resolve_format(spec, base, res, fps)
        self.W, self.H, self.fps = F["W"], F["H"], F["fps"]
        self.S = math.sqrt(self.W * self.H / (DESIGN_W * DESIGN_H))
        self.portrait = self.H > self.W * 1.15
        self.wide = self.W > self.H * 1.15
        bgname = F.get("background") or F.get("theme") or spec.get("background") or spec.get("theme") or "night"
        if isinstance(bgname, dict):
            bg = dict(BACKGROUNDS["night"])
            bg.update({k: (col(v) if k in ("top", "bottom", "tree", "grass") else v) for k, v in bgname.items()})
            if "land" in bgname:
                bg["land"] = [col(c) for c in bgname["land"]]
        else:
            bg = BACKGROUNDS.get(str(bgname).lower(), BACKGROUNDS["night"])
        self.bg = bg
        self.lang = spec.get("lang", "id")
        # auto-reframe when designed for another aspect
        da = str(spec.get("design_aspect", F["aspect"])).replace("x", ":")
        try:
            dw, dh = [float(v) for v in EX.ASPECT_ALIASES.get(da, da).split(":")]
        except Exception:
            dw, dh = 9.0, 16.0
        des_h = math.sqrt(DESIGN_W * DESIGN_H * dh / dw)
        des_w = DESIGN_W * DESIGN_H / des_h
        vis_h, vis_w = self.H / self.S, self.W / self.S
        f = min(vis_h / des_h, vis_w / des_w)
        self.reframe = f ** 0.75 if (F.get("reframe", True) and f < 0.98) else 1.0
        # words / timing
        self.has_voice = bool(spec.get("audio"))
        self.asr_info = None
        if self.has_voice and not (spec.get("script") or spec.get("srt") or spec.get("words")):
            raise SystemExit("Audio given but no script: transcribe first (papercut.py voice.mp3 --transcribe <model-dir>), "
                             "correct it, and put it in spec.script")
        if spec.get("script") or spec.get("srt") or spec.get("words"):
            self.words, self.mode = EX.resolve_words(spec, base)
        else:
            self.words, self.mode = [], "timeline"
        if not self.has_voice and self.mode != "timeline":
            self.mode = "script"
        cf = F["captions"]
        if not F["_captions_given"] and not self.has_voice:
            cf["show"] = False          # no narration -> stickers tell the story; captions only if asked
        max_chars = {1: 99, 2: 15, 3: 20, 4: 24, 5: 30}[cf["words"]]
        if self.wide:
            max_chars = int(max_chars * 1.5)
        self.caps = EX.build_captions(self.words, spec.get("highlight"), cf["words"], max_chars) if self.words else []
        if not cf["highlight"]:
            for c in self.caps:
                c["hl"] = [False] * len(c["hl"])
        meta = spec.get("meta", {})
        self.meta = meta
        self.outro_dur = float(meta.get("outro_dur", 3.6))
        if self.words:
            self.duration = float(spec.get("duration", self.words[-1]["end"] + spec.get("tail", 1.3)))
        else:
            ts = [b.get("at") for b in spec.get("beats", []) if isinstance(b.get("at"), (int, float))]
            est = (max(ts) if ts else 8) + 3.0 + (self.outro_dur if (meta.get("outro") and F["outro"]) else 0)
            self.duration = float(spec.get("duration", est))
        # camera zoom cap -> sprite resolution
        zooms = [spec.get("camera", {}).get("zoom", 1.0)]
        for b in spec.get("beats", []):
            acts = b.get("do", [])
            for a in (acts if isinstance(acts, list) else [acts]):
                if isinstance(a, dict) and a.get("type") == "camera" and "zoom" in a:
                    zooms.append(float(a["zoom"]))
        self.zcap = clamp(max(zooms) * self.reframe, 1.0, 3.0)
        self.ppu = self.S * self.zcap * 1.05
        self.hud_ppu = self.S * 1.2
        # objects
        self.objects, self.by_id = [], {}
        for i, o in enumerate(spec.get("world", [])):
            for oo in self.expand(o, i):
                self.add_object(oo)
        self.sfx, self.loops, self.cam, self.events, self.extra = [], [], [], [], []
        self.shakes = []
        self._ptr = 0
        self._sprites = {}
        self.warnings = []
        self.compile()
        self.sky = self.build_sky()
        self.stars = self.build_stars()

    # ---------------------------------------------------------------- objects
    def expand(self, o, i):
        if o.get("type") != "landscape":
            return [dict(o)]
        hz = o.get("y", 900)
        x0, x1 = o.get("x0", -700), o.get("x1", 1420)
        land = [col(c) for c in o.get("colors", [])] or self.bg["land"]
        seed = o.get("seed", 3)
        trees = o.get("trees", True)
        layers = int(o.get("layers", 4))
        pid = o.get("id", f"land{i}")
        out = [
            {"id": f"{pid}_back", "type": "hill", "x0": x0 - 200, "x1": x1 + 200, "y": hz - 70, "amp": 120, "wave": 980,
             "color": land[0], "parallax": 0.55, "seed": seed, "trees": 0, "edge": True},
            {"id": f"{pid}_mid", "type": "hill", "x0": x0 - 100, "x1": x1 + 100, "y": hz + 10, "amp": 60, "wave": 560,
             "color": land[1], "parallax": 0.78, "seed": seed + 1, "trees": (int((x1 - x0) / 120) if trees else 0),
             "tree_color": self.bg["tree"], "edge": True},
            {"id": f"{pid}_front", "type": "hill", "x0": x0, "x1": x1, "y": hz + 90, "amp": 12, "wave": 700,
             "color": land[2], "parallax": 1.0, "seed": seed + 2, "torn": True, "grass": True, "grass_color": self.bg["grass"]},
        ]
        if layers >= 4:
            out.append({"id": f"{pid}_fore", "type": "hill", "x0": x0, "x1": x1, "y": hz + 330, "amp": 8, "wave": 520,
                        "color": land[3], "parallax": 1.0, "seed": seed + 3, "torn": True, "grass": True,
                        "grass_color": PK.shade(land[3], 0.8)})
        if layers <= 2:
            out = out[1:3]
        for k in ("layer",):
            if k in o:
                for oo in out:
                    oo[k] = o[k]
        return out

    def add_object(self, o):
        o = dict(o)
        if o.get("type") not in OBJ_TYPES:
            self.warn(f"unknown object type: {o.get('type')} (id={o.get('id')})")
        o.setdefault("id", f"_o{len(self.objects)}")
        if o["id"] in self.by_id:
            self.warn(f"duplicate id '{o['id']}'")
        o["_base"] = {k: v for k, v in o.items() if not k.startswith("_")}
        o["_tw"], o["_ev"], o["_bumps"] = [], [], []
        o["_ord"] = len(self.objects)
        self.objects.append(o)
        self.by_id[o["id"]] = o
        return o

    def warn(self, msg):
        self.warnings.append(msg)
        print("  ! " + msg, file=sys.stderr)

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
        try:
            return float(s) + off
        except ValueError:
            pass
        if s.lower() in ("start", "begin"):
            return off
        if s.lower() == "end":
            return self.duration + off
        if s.lower() == "outro":
            return self.outro_start() + off
        if s.startswith("#") and self.words:
            return self.words[min(int(s[1:]), len(self.words) - 1)]["start"] + off
        end_mode = s.startswith("^")
        if end_mode:
            s = s[1:]
        q = [EX.norm_word(x) for x in s.split() if EX.norm_word(x)]
        n = [w["n"] for w in self.words]
        if not q or not n:
            self.warn(f"anchor '{at}' cannot be resolved (no script) -> {default:.2f}s; use seconds in timeline mode")
            return default + off

        def find(frm):
            for i in range(frm, len(n) - len(q) + 1):
                if n[i:i + len(q)] == q:
                    return i
            for i in range(frm, len(n) - len(q) + 1):
                if all(n[i + k].startswith(q[k][:max(3, len(q[k]) - 2)]) for k in range(len(q))):
                    return i
            return None
        i = find(self._ptr)
        if i is None:
            i = find(0)
        if i is None:
            self.warn(f"anchor not found: '{at}' -> using {default:.2f}s")
            return default + off
        self._ptr = i
        w = self.words[i + len(q) - 1] if end_mode else self.words[i]
        return (w["end"] if end_mode else w["start"]) + off

    def outro_start(self):
        if "outro_at" in self.meta:
            return self.T(self.meta["outro_at"], self.duration - self.outro_dur)
        return max(0.0, self.duration - self.outro_dur)

    def title_end(self):
        if "title_dur" in self.meta:
            return float(self.meta["title_dur"])
        if self.meta.get("title") and self.title_style() == "sign":
            ct = self.sign_geom()["confetti_t"]
            words = getattr(self, "words", None)
            if words:
                # narration: the sign leaves when the hook sentence ends (body starts), never before it is readable
                first = next((w["end"] for w in words if w["punct"] == 2), None)
                if first is not None and first < 4.5:
                    return max(ct + 0.45, first)
            return max(2.8, ct + 1.0)
        return 2.8

    # ---------------------------------------------------------------- compile
    def add_sfx(self, t, name, gain=None, act=None):
        if act is not None and "sfx" in act:
            name = act["sfx"]
        if not name or not self.F["sfx"]:
            return
        if name not in MU.LIB:
            self.warn(f"unknown sfx '{name}'")
            return
        self.sfx.append((max(0.0, t), name, gain))

    def tw(self, oid, t0, dur, key, to, ease="io"):
        o = self.by_id.get(oid)
        if o is None:
            self.warn(f"action refers to unknown object id '{oid}'")
            return
        o["_tw"].append((t0, t0 + max(dur, 1e-3), key, to, ease))

    def ev(self, oid, t0, dur, kind, mode):
        o = self.by_id.get(oid)
        if o is None:
            self.warn(f"action refers to unknown object id '{oid}'")
            return
        o["_ev"].append((t0, max(dur, 1e-3), kind, mode))

    def default_show_dur(self, o):
        if o.get("type") == "text" and o.get("letters", True):
            n = len(str(o.get("text", "")).replace(" ", "").replace("\n", ""))
            return 0.32 + 0.05 * n
        return 0.45

    def compile(self):
        sp = self.spec
        c0 = sp.get("camera", {})
        self.cam_init = {"x": c0.get("x", 360), "y": c0.get("y", 640), "zoom": c0.get("zoom", 1.0)}
        last = 0.0
        for b in sp.get("beats", []):
            t = self.T(b.get("at"), last)
            last = t
            acts = b.get("do")
            if isinstance(acts, dict):
                acts = [acts]
            if acts is None:
                acts = [{k: v for k, v in b.items() if k != "at"}]
            for a in acts:
                self.action(t + float(a.get("delay", 0)), a)
                kind = a.get("type") or next((k for k in a if k in ACTIONS), "?")
                self.events.append((t, str(b.get("at")), kind))
        if not self.words and "duration" not in sp:
            # timeline mode: make sure the last beat has room + outro
            need = last + 3.0 + (self.outro_dur if (self.meta.get("outro") and self.F["outro"]) else 0)
            self.duration = max(self.duration, need)
        if sp.get("ambience") and self.F["ambience"] and self.F["sfx"]:
            amb = sp["ambience"]
            amb = [{"name": amb}] if isinstance(amb, str) else amb
            for a in amb:
                self.loops.append((self.T(a.get("from"), 0), self.T(a.get("to"), self.duration), a["name"], a.get("gain")))
        if self.meta.get("title") and self.F["title"] and self.title_style() == "sign":
            g = self.sign_geom()
            self.add_sfx(0.02, "swish", -21)
            self.add_sfx(0.42, "stamp", -22)                      # board lands on its strings
            for wt in g["word_t"]:
                self.add_sfx(wt + 0.2, "tap", -19)
            self.add_sfx(g["confetti_t"], "sparkle", -22)
            self.add_sfx(self.title_end(), "swish", -21)
            if self.chip_text():
                self.add_sfx(self.title_end() + 0.55, "tap", -22)
        elif self.meta.get("title") and self.F["title"]:
            self.add_sfx(0.05, "paper", -22)
            n = len("".join(self.meta["title"] if isinstance(self.meta["title"], list) else [self.meta["title"]]).replace(" ", ""))
            for k in range(min(6, max(2, n // 3))):
                self.add_sfx(0.25 + k * 0.16, "softpop", -24)
            if self.chip_text():
                self.add_sfx(self.title_end(), "swish", -22)
        if self.meta.get("outro") and self.F["outro"]:
            self.add_sfx(self.outro_start(), "paper", -20)
            self.add_sfx(self.outro_start() + 0.5, "chime", -21)

    def target_xy(self, tgt):
        if isinstance(tgt, (list, tuple)):
            return float(tgt[0]), float(tgt[1])
        if isinstance(tgt, str) and tgt in self.by_id:
            b = self.by_id[tgt]["_base"]
            return float(b.get("x", 0)), float(b.get("y", 0))
        return 360.0, 640.0

    def action(self, t, a):
        kind = a.get("type") or next((k for k in a if k in ACTIONS), None)
        if kind not in ACTIONS:
            self.warn(f"unknown action: {a}")
            return
        if kind == "camera":
            to = {}
            f = a.get("focus", a.get("camera"))
            if isinstance(f, str) and f in self.by_id:
                to["x"], to["y"] = self.target_xy(f)
            elif isinstance(f, (list, tuple)):
                to["x"], to["y"] = f[0], f[1]
            for k in ("x", "y", "zoom"):
                if k in a:
                    to[k] = a[k]
            d = float(a.get("dur", 1.3))
            self.cam.append((t, t + d, to, a.get("ease", "io")))
            if a.get("sfx", "auto") == "auto":
                self.add_sfx(t, "whoosh", -25 if d > 0.6 else -23)
            else:
                self.add_sfx(t, a.get("sfx"))
        elif kind in ("show", "hide"):
            v = a.get(kind) if a.get(kind) not in (None, True) else a.get("id")
            ids = v if isinstance(v, list) else [v]
            for i, oid in enumerate(ids):
                o = self.by_id.get(oid)
                if o is None:
                    self.warn(f"action refers to unknown object id '{oid}'")
                    continue
                tt = t + i * float(a.get("stagger", 0.12))
                if kind == "show":
                    d = float(a.get("dur", self.default_show_dur(o)))
                    mode = a.get("enter", o.get("enter", "pop"))
                    self.ev(oid, tt, d, "show", mode)
                    if i < 4:
                        if o.get("type") == "text" and o.get("letters", True):
                            n = len(str(o.get("text", "")).replace(" ", ""))
                            for k in range(min(5, max(1, n // 3))):
                                self.add_sfx(tt + k * 0.12, "softpop", -24, a)
                        elif o.get("type") in ("tag", "strip", "bubble"):
                            self.add_sfx(tt, "tap", None, a)
                        elif str(mode).startswith("slide"):
                            self.add_sfx(tt, "paper", None, a)
                        elif o.get("type") == "burst":
                            self.add_sfx(tt, "stamp", None, a)
                        elif o.get("type") == "mark":
                            self.add_sfx(tt, "correct" if o.get("kind", "check") == "check" else "wrong", None, a)
                        else:
                            self.add_sfx(tt, "pop", None, a)
                else:
                    self.ev(oid, tt, float(a.get("dur", 0.35)), "hide", a.get("exit", "shrink"))
        elif kind == "set":
            oid = a.get("set") if isinstance(a.get("set"), str) else a.get("id")
            for k, v in a.get("props", {}).items():
                self.tw(oid, t, float(a.get("dur", 0.8)), k, v, a.get("ease", "io"))
            if "sfx" in a:
                self.add_sfx(t, a["sfx"])
        elif kind == "draw":
            oid = a.get("draw") if isinstance(a.get("draw"), str) else a.get("id")
            o = self.by_id.get(oid)
            if o is None:
                self.warn(f"draw: unknown id '{oid}'")
                return
            o["_base"].setdefault("draw", 0.0)
            if not any(e[2] == "show" for e in o["_ev"]) and o.get("hidden"):
                self.ev(oid, t, 0.01, "show", "fade")
            self.tw(oid, t, float(a.get("dur", 0.7)), "draw", 1.0, a.get("ease", "out"))
            self.add_sfx(t, "swish", -22, a)
        elif kind == "grow":
            oid = a.get("grow") if isinstance(a.get("grow"), str) else a.get("id")
            o = self.by_id.get(oid)
            if o is None:
                self.warn(f"grow: unknown id '{oid}'")
                return
            o["_base"].setdefault("grow", 0.0)
            d = float(a.get("dur", 2.0))
            self.tw(oid, t, d, "grow", float(a.get("to", 1.0)), a.get("ease", "linear"))
            n = len(o.get("values", [])) or 1
            for k in range(n):
                self.add_sfx(t + d * (k + 0.3) / n, "softpop", -23, a)
        elif kind == "count":
            oid = a.get("count") if isinstance(a.get("count"), str) else a.get("id")
            d = float(a.get("dur", 1.6))
            self.tw(oid, t, d, "value", float(a["to"]), a.get("ease", "out"))
            if self.F["sfx"] and a.get("sfx", "auto") == "auto":
                k = 0.0
                step = 0.045
                while k < d * 0.9:
                    self.add_sfx(t + k, "tick", -26)
                    step *= 1.09
                    k += step
                self.add_sfx(t + d, "ding", -22)
            elif a.get("sfx"):
                self.add_sfx(t, a["sfx"])
        elif kind == "pulse":
            oid = a.get("pulse") if isinstance(a.get("pulse"), str) else a.get("target", a.get("id"))
            o = self.by_id.get(oid)
            if o is None:
                self.warn(f"pulse: unknown id '{oid}'")
                return
            o["_bumps"].append((t, float(a.get("scale", 0.16))))
            self.add_sfx(t, "softpop", -22, a)
        elif kind == "label":
            tgt = a.get("target", a.get("on"))
            off = a.get("offset", [0, -70])
            t1 = self.T(a["until"], t + 2.6) if a.get("until") else t + float(a.get("hold", 2.6))
            o = {"type": "tag", "text": str(a["label"]), "hidden": True, "layer": a.get("layer", 50),
                 "size": a.get("size", 24), "rotate": a.get("rotate", -3), "shape": a.get("style", "pill"),
                 "bg": a.get("bg", "cream"), "ink": a.get("ink", "ink")}
            if isinstance(tgt, str) and tgt in self.by_id:
                o["follow"] = tgt
                o["follow_off"] = off
                o["parallax"] = self.by_id[tgt].get("parallax", 1.0)
                o["screen"] = self.by_id[tgt].get("screen", False)
            else:
                x, y = self.target_xy(tgt)
                o["x"], o["y"] = x + off[0], y + off[1]
            oo = self.add_object(o)
            self.ev(oo["id"], t, 0.4, "show", a.get("enter", "pop"))
            if a.get("hold") != 0:
                self.ev(oo["id"], t1, 0.3, "hide", "shrink")
            self.add_sfx(t, "tap", None, a)
        elif kind == "bigtext":
            pos = a.get("pos", [0.5, a.get("y", 0.42)])
            t1 = self.T(a["until"], t + 2.0) if a.get("until") else t + float(a.get("hold", 2.0))
            o = {"type": "text", "text": str(a["bigtext"]), "hidden": True, "screen": True, "pos": pos,
                 "size": a.get("size", 92), "colors": a.get("colors", a.get("color", "cycle")), "layer": 60}
            oo = self.add_object(o)
            self.ev(oo["id"], t, self.default_show_dur(o), "show", a.get("enter", "pop"))
            self.ev(oo["id"], t1, 0.3, "hide", "shrink")
            self.add_sfx(t, "stamp", -18, a)
        elif kind == "takeaway":
            pos = a.get("pos", [0.5, a.get("y", 0.84)])
            t1 = self.T(a["until"], t + 3.0) if a.get("until") else t + float(a.get("hold", 3.0))
            o = {"type": "strip", "text": a["takeaway"], "hidden": True, "screen": True, "pos": pos,
                 "size": a.get("size", 30), "layer": 60}
            oo = self.add_object(o)
            self.ev(oo["id"], t, 0.5, "show", "slide-up")
            self.ev(oo["id"], t1, 0.35, "hide", "fade")
            self.add_sfx(t, "paper", None, a)
        elif kind == "flow":
            items = a.get("items") or []
            if isinstance(items, dict):
                items = [items]
            if a.get("template"):
                tp = self.by_id.get(a["template"])
                if tp:
                    items = [{k: v for k, v in tp["_base"].items() if k not in ("id", "hidden")}]
            if not items:
                self.warn("flow: no items/template")
                return
            fx, fy = self.target_xy(a.get("from", [0, 640]))
            tx, ty = self.target_xy(a.get("to", [720, 640]))
            every = float(a.get("every", 0.5))
            n = int(a.get("count", 6))
            if a.get("until"):
                n = max(1, int((self.T(a["until"], t + n * every) - t) / every))
            d = float(a.get("dur", 2.4))
            ease = a.get("ease", "linear")
            for i in range(n):
                it = dict(items[i % len(items)])
                tt = t + i * every
                it.update({"x": fx, "y": fy, "hidden": True, "layer": a.get("layer", it.get("layer", 5))})
                if a.get("parallax") is not None:
                    it["parallax"] = a["parallax"]
                if a.get("rotate_jitter", 4):
                    rj = float(a.get("rotate_jitter", 4))
                    it.setdefault("rotate", float(np.random.default_rng(i + 7).uniform(-rj, rj)))
                oo = self.add_object(it)
                self.ev(oo["id"], tt, 0.35, "show", a.get("enter", "pop"))
                self.tw(oo["id"], tt, d, "x", tx, ease)
                self.tw(oo["id"], tt, d, "y", ty, ease)
                if a.get("arc"):
                    oo["_arc"] = (tt, tt + d, float(a["arc"]))
                if a.get("vanish", True):
                    self.ev(oo["id"], tt + d - 0.25, 0.3, "hide", a.get("exit", "shrink"))
                if i < 12 and i % max(1, int(a.get("sfx_every", 2))) == 0:
                    self.add_sfx(tt, a.get("sfx", "tap") if a.get("sfx") != "auto" else "tap", -23)
        elif kind == "swap":
            # hide one object and show another in its place (state change)
            self.action(t, {"type": "hide", "id": a["swap"][0], "dur": 0.25, "sfx": None})
            self.action(t + 0.1, {"type": "show", "id": a["swap"][1], "enter": a.get("enter", "pop")})
        elif kind == "shake":
            self.shakes.append((t, t + float(a.get("dur", 0.45)), float(a.get("amp", 8))))
        elif kind == "sfx":
            self.add_sfx(t, a["sfx"], a.get("gain"))

    # ---------------------------------------------------------------- per-frame state
    def props(self, o, t):
        p = dict(o["_base"])
        for (t0, t1, k, to, e) in o["_tw"]:
            if t < t0:
                continue
            q = EASES.get(e, ease_io)((t - t0) / (t1 - t0))
            if "." in k:
                head, idx = k.split(".", 1)
                arr = list(p.get(head, []))
                ii = int(idx)
                while len(arr) <= ii:
                    arr.append(0)
                arr[ii] = lerp(arr[ii], to, q)
                p[head] = arr
            else:
                cur = p.get(k, 0 if not isinstance(to, (list, str)) else to)
                p[k] = lerp(cur, to, q) if not isinstance(to, (str, bool)) else (to if q >= 1 else cur)
        if "_arc" in o:
            a0, a1, h = o["_arc"]
            if a0 <= t <= a1:
                p["y"] = p.get("y", 0) - h * math.sin(math.pi * (t - a0) / (a1 - a0))
        if p.get("follow") in self.by_id:
            tp = self.props(self.by_id[p["follow"]], t)
            off = p.get("follow_off", [0, -70])
            p["x"], p["y"] = tp.get("x", 0) + off[0], tp.get("y", 0) + off[1]
            if tp.get("screen"):
                p["pos"] = tp.get("pos")
        return p

    def vis(self, o, t):
        """-> (alpha, scale, rot, dx, dy, progress-for-letters or None, exiting)"""
        last = None
        for e in o["_ev"]:
            if e[0] <= t and (last is None or e[0] >= last[0]):
                last = e
        if last is None:
            return (0.0, 1, 0, 0, 0, None, False) if o.get("hidden") else (1.0, 1, 0, 0, 0, None, False)
        t0, d, kind, mode = last
        q = clamp((t - t0) / d)
        mode = str(mode)
        if kind == "show":
            if q >= 1:
                return (1.0, 1, 0, 0, 0, None, False)
            if mode == "pop":
                return (min(1, q * 4), max(0.02, ease_back(q)), (1 - ease_out(q)) * -10, 0, 0, q, False)
            if mode == "fade":
                return (q, 1, 0, 0, 0, q, False)
            if mode == "grow":
                return (min(1, q * 3), max(0.02, ease_out(q)), 0, 0, 0, q, False)
            if mode == "drop":
                return (min(1, q * 5), 1, (1 - q) * 6, 0, -(1 - ease_bounce(q)) * 260, q, False)
            dist = 260
            dx, dy = {"slide-up": (0, dist), "slide-down": (0, -dist), "slide-left": (dist, 0), "slide-right": (-dist, 0)}.get(mode, (0, dist))
            k = 1 - ease_out(q)
            return (min(1, q * 3), 1, k * 4, dx * k, dy * k, q, False)
        # hide
        if q >= 1:
            return (0.0, 1, 0, 0, 0, None, True)
        if mode == "fade":
            return (1 - q, 1, 0, 0, 0, None, True)
        if mode.startswith("slide"):
            dist = 260
            dx, dy = {"slide-up": (0, -dist), "slide-down": (0, dist), "slide-left": (-dist, 0), "slide-right": (dist, 0)}.get(mode, (0, dist))
            k = ease_in(q)
            return (1 - q, 1, 0, dx * k, dy * k, None, True)
        return (1 - ease_in(q), 1 - 0.7 * ease_in(q), q * 8, 0, 0, None, True)

    def camera(self, t):
        cur = dict(self.cam_init)
        for (t0, t1, to, e) in sorted(self.cam, key=lambda c: c[0]):
            if t < t0:
                break
            q = EASES.get(e, ease_io)((t - t0) / (t1 - t0))
            for k, v in to.items():
                cur[k] = cur[k] + (v - cur[k]) * q
        cur["zoom"] *= (1 + 0.01 * math.sin(t * 0.33)) * self.reframe
        cur["x"] += 2.5 * math.sin(t * 0.21)
        for (t0, t1, amp) in self.shakes:
            if t0 <= t <= t1:
                k = 1 - (t - t0) / (t1 - t0)
                cur["x"] += amp * k * math.sin(t * 90)
                cur["y"] += amp * k * math.cos(t * 77)
        return cur

    def to_screen(self, x, y, cam, par=1.0):
        z = 1 + (cam["zoom"] - 1) * par if par != 1.0 else cam["zoom"]
        cx = cam["x"] * par + 360 * (1 - par)
        cy = cam["y"] * par + 640 * (1 - par)
        k = z * self.S
        return self.W / 2 + (x - cx) * k, self.H / 2 + (y - cy) * k, k

    # ---------------------------------------------------------------- sprites
    def sprite(self, key, fn):
        s = self._sprites.get(key)
        if s is None:
            s = fn()
            self._sprites[key] = s
            if len(self._sprites) > 900:
                for k in list(self._sprites)[:200]:
                    del self._sprites[k]
        return s

    def parts(self, o, p, t, ppu):
        """list of (sprite, dx, dy, rot, sx, sy, alpha) in object-local world units."""
        ty = p.get("type")
        fn = getattr(self, "parts_" + str(ty), None)
        if fn is None:
            return []
        return fn(o, p, t, ppu)

    # --- builders ------------------------------------------------------------------------
    def parts_hill(self, o, p, t, ppu):
        keys = ("x0", "x1", "y", "amp", "wave", "bottom", "color", "torn", "trees", "tree_color", "grass", "seed", "edge", "tree_size")
        key = ("hill",) + tuple(json.dumps(p.get(k)) for k in keys) + (round(ppu, 3),)

        def build():
            x0, x1 = p.get("x0", -700), p.get("x1", 1420)
            y = p.get("y", 900)
            amp, wave = p.get("amp", 40), p.get("wave", 520)
            bottom = p.get("bottom", y + 900)
            seed = int(p.get("seed", 1))
            rng = np.random.default_rng(seed)
            ph = rng.uniform(0, 6.28, 3)
            xs = np.arange(x0, x1 + 10, 10.0)
            ys = y - amp * (0.55 * np.sin(2 * np.pi * xs / wave + ph[0]) + 0.3 * np.sin(2 * np.pi * xs / (wave * 0.53) + ph[1])
                            + 0.15 * np.sin(2 * np.pi * xs / (wave * 0.27) + ph[2]))
            top = np.stack([xs, ys], 1)
            tmax = max(p.get("tree_size", [30, 58])) if p.get("trees") else 0
            cv = PK.Canvas((x0, ys.min() - tmax - 20, x1, bottom), min(ppu, self.S * 2.2), pad=20)
            c = col(p.get("color"), (70, 110, 100))
            if p.get("torn"):
                wt = PK.jag(top - [0, 6], 2.4, 3.2, seed + 9)
                cv.poly(np.concatenate([wt, [[x1, bottom], [x0, bottom]]]), PK.PAL["cream"])
                ct = PK.jag(top, 1.3, 4.5, seed + 4)
                cv.poly(np.concatenate([ct, [[x1, bottom], [x0, bottom]]]), c, lift=1.2)
            else:
                cv.poly(np.concatenate([top, [[x1, bottom], [x0, bottom]]]), c)
                if p.get("edge", True):
                    cv.stroke(top - [0, 0.6], 1.6, PK.mix(c, (255, 255, 255), 0.28), 0, False, 0.8)
            ntree = int(p.get("trees", 0))
            if ntree:
                tc = col(p.get("tree_color"), (38, 82, 78))
                txs = np.sort(rng.uniform(x0 + 30, x1 - 30, ntree))
                ts = p.get("tree_size", [30, 58])
                for tx in txs:
                    s = rng.uniform(ts[0], ts[1])
                    ty0 = float(np.interp(tx, xs, ys)) + 6
                    trunk = [[tx - s * 0.07, ty0], [tx + s * 0.07, ty0], [tx + s * 0.07, ty0 - s * 0.25], [tx - s * 0.07, ty0 - s * 0.25]]
                    cv.poly(trunk, (92, 64, 52), lift=0.6)
                    for j in range(3):
                        yb = ty0 - s * (0.18 + 0.27 * j)
                        wj = s * (0.36 - 0.08 * j)
                        cv.poly([[tx, yb - s * 0.42], [tx + wj, yb], [tx - wj, yb]], PK.shade(tc, 1 + 0.06 * j), lift=0.9)
            if p.get("grass"):
                gc = col(p.get("grass_color"), PK.shade(c, 0.75))
                for _ in range(int((x1 - x0) / 26)):
                    gx = rng.uniform(x0, x1)
                    gy = float(np.interp(gx, xs, ys)) + rng.uniform(40, max(60, min(500, bottom - y - 20)))
                    for b in range(3):
                        dx = (b - 1) * 2.6
                        cv.stroke([[gx + dx, gy], [gx + dx * 2.2 + rng.uniform(-1, 1), gy - rng.uniform(7, 13)]], 1.6, gc, 0, False)
            return cv.finish(shadow=(0, -5, 8, 0.38), tex=0.085, edge=0.0, seed=seed, base_ppu=self.S, fine=False)
        return [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]

    def parts_shape(self, o, p, t, ppu):
        keys = ("shape", "w", "h", "r", "radius", "points", "fill", "color", "stroke", "sw", "torn", "closed")
        key = ("shape",) + tuple(json.dumps(p.get(k)) for k in keys) + (round(ppu, 3),)

        def build():
            sh = p.get("shape", "rect")
            c = col(p.get("fill", p.get("color", "cream")))
            w, h = p.get("w", 100), p.get("h", p.get("w", 100))
            r = p.get("r", w / 2)
            if sh in ("circle", "ellipse"):
                rx, ry = (r, r) if sh == "circle" else (w / 2, h / 2)
                cv = PK.Canvas((-rx, -ry, rx, ry), ppu)
                cv.ellipse(0, 0, rx, ry, c)
            elif sh in ("poly", "blob", "path"):
                pts = np.asarray(p.get("points", [[0, 0], [50, 0], [25, 40]]), float)
                if sh == "blob":
                    pts = PK.catmull(pts, True)
                mn, mx = pts.min(0), pts.max(0)
                sw = p.get("sw", 10)
                cv = PK.Canvas((mn[0] - sw, mn[1] - sw, mx[0] + sw, mx[1] + sw), ppu)
                if sh == "path":
                    cv.stroke(PK.catmull(pts, False) if len(pts) > 2 else pts, sw, c, 0, bool(p.get("closed")))
                else:
                    cv.poly(pts, c)
            else:
                rad = p.get("radius", 0 if sh == "rect" else 12)
                cv = PK.Canvas((-w / 2, -h / 2, w / 2, h / 2), ppu)
                if p.get("torn"):
                    pts = PK.rrect_pts(-w / 2, -h / 2, w / 2, h / 2, rad)
                    cv.poly(PK.jag(np.vstack([pts, pts[:1]]), 1.4, 4, 3), c)
                else:
                    cv.rrect(-w / 2, -h / 2, w / 2, h / 2, rad, c)
            return cv.finish(seed=hash(key) % 1000, base_ppu=self.S)
        return [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]

    def parts_prop(self, o, p, t, ppu):
        src = p.get("svg") or prop_svg(p.get("prop", "star"))
        w = p.get("w", 120)
        rec = p.get("recolor")
        key = ("svg", src, w, json.dumps(rec, sort_keys=True), round(ppu, 3))

        def build():
            ww, hh = svg_size(src, w)
            cv = PK.Canvas((-ww / 2, -hh / 2, ww / 2, hh / 2), ppu)
            draw_svg(cv, src, 0, 0, ww, rec)
            return cv.finish(seed=len(src) % 997, base_ppu=self.S)
        flip = -1 if p.get("flip") else 1
        return [(self.sprite(key, build), 0, 0, 0, flip, 1, 1)]

    parts_svg = parts_prop

    def sticker_letter(self, ch, size, color, outline, ppu, fp=F_STICKER, oc=None):
        oc = tuple(oc or PK.PAL["cream"])
        key = ("L", ch, round(size, 1), tuple(color), round(outline, 2), round(ppu, 3), fp, oc)

        def build():
            w = text_w(ch, size, fp) + 2 * outline + 8
            hgt = size * 1.3 + 2 * outline
            cv = PK.Canvas((-w / 2, -hgt / 2, w / 2, hgt / 2), ppu, pad=8)
            cv.text(ch, 0, 0, size, color, fp, "mm", outline=outline, outline_color=oc)
            return cv.finish(shadow=(2, 4, 3.5, 0.45), seed=ord(ch[0]) * 7, base_ppu=self.S)
        return self.sprite(key, build)

    def letter_layout(self, text, size, colors, fp=F_STICKER, track=0.05):
        """-> list of (char, x, y, color) centred on (0,0); lines separated by \\n. track = extra spacing (x size)."""
        lines = str(text).split("\n")
        cyc = colors if isinstance(colors, list) else (PK.STICKER_CYCLE if colors in (None, "cycle", "rainbow") else [colors])
        lh = size * 1.18
        out = []
        k = 0
        f = PK.font(fp, 200)
        for li, ln in enumerate(lines):
            wtot = f.getlength(ln) / 200 * size + track * size * max(0, len(ln) - 1)
            x = -wtot / 2
            y = (li - (len(lines) - 1) / 2) * lh
            for ch in ln:
                adv = f.getlength(ch) / 200 * size
                if ch.strip():
                    out.append((ch, x + adv / 2, y, col(cyc[k % len(cyc)])))
                    k += 1
                x += adv + track * size
        return out

    def parts_text(self, o, p, t, ppu):
        size = p.get("size", 64)
        outline = p.get("outline", size * 0.12)
        colors = p.get("colors", p.get("color", "cycle"))
        fp = {"heavy": F_HEAVY, "semi": F_SEMI}.get(p.get("font"), F_STICKER)
        lay = self.letter_layout(p.get("text", ""), size, colors, fp)
        rng = np.random.default_rng(len(str(p.get("text"))) * 31 + int(size))
        jit = p.get("jitter", 1.0)
        vis_q = p.get("_q")
        out = []
        n = len(lay)
        for i, (ch, x, y, c) in enumerate(lay):
            spr = self.sticker_letter(ch, size, c, outline, ppu, fp)
            rot = float(rng.uniform(-3.5, 3.5)) * jit
            yy = y + float(rng.uniform(-1.5, 1.5)) * jit * size / 40
            if p.get("wave"):
                yy += math.sin(t * 3 + i * 0.6) * p["wave"]
            s, a = 1.0, 1.0
            if vis_q is not None:
                D = p.get("_D", 0.6)
                st = min(0.05, (D - 0.3) / max(n, 1))
                lq = clamp((vis_q * D - i * st) / 0.3)
                s = max(0.01, ease_back(lq))
                a = min(1, lq * 4)
                rot += (1 - ease_out(lq)) * -14
            out.append((spr, x, yy, rot, s, s, a))
        return out

    def tag_sprite(self, text, size, bg, ink, shape, ppu, fp=F_STICKER):
        key = ("tag", text, size, tuple(bg), tuple(ink), shape, round(ppu, 3), fp)

        def build():
            tw_ = text_w(text, size, fp)
            padx, pady = size * 0.55, size * 0.32
            w, h = tw_ + 2 * padx, size * 1.05 + 2 * pady
            extra = h * 0.45 if shape == "tag" else 0
            cv = PK.Canvas((-w / 2 - 2, -h / 2 - 2, w / 2 + extra + 2, h / 2 + 2), ppu)
            if shape == "tag":
                pts = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2 + extra, 0], [w / 2, h / 2], [-w / 2, h / 2]]
                cv.poly(PK.rrect_pts(-w / 2, -h / 2, w / 2 + 1, h / 2, 4) if False else pts, bg)
                cv.circle(w / 2 + extra * 0.25, 0, h * 0.09, PK.shade(bg, 0.8))
            elif shape == "rect":
                cv.rrect(-w / 2, -h / 2, w / 2, h / 2, 3, bg)
            else:
                cv.rrect(-w / 2, -h / 2, w / 2, h / 2, h * 0.32, bg)
            cv.text(text, 0, 0, size, ink, fp, "mm")
            return cv.finish(shadow=(1.6, 3, 3, 0.42), seed=len(text) * 13, base_ppu=self.S)
        return self.sprite(key, build)

    def parts_tag(self, o, p, t, ppu):
        spr = self.tag_sprite(str(p.get("text", "tag")), p.get("size", 24), col(p.get("bg", "cream")),
                              col(p.get("ink", "ink")), p.get("shape", "pill"), ppu)
        return [(spr, 0, 0, 0, 1, 1, 1)]

    def parts_photo(self, o, p, t, ppu):
        w = p.get("w", 110)
        keys = ("w", "icon", "prop", "svg", "bg", "tag", "tag_color", "frame")
        key = ("photo",) + tuple(json.dumps(p.get(k)) for k in keys) + (round(ppu, 3),)

        def build():
            h = w * 1.22
            cv = PK.Canvas((-w / 2, -h / 2, w / 2, h / 2 + w * 0.12), ppu)
            cv.rrect(-w / 2, -h / 2, w / 2, h / 2, 3, col(p.get("frame", "cream")))
            m = w * 0.08
            sq = w - 2 * m
            cv.rrect(-w / 2 + m, -h / 2 + m, w / 2 - m, -h / 2 + m + sq, 2, col(p.get("bg", "lightblue")), lift=0.5)
            src = p.get("svg") or (prop_svg(p.get("icon", p.get("prop"))) if (p.get("icon") or p.get("prop")) else None)
            if src:
                draw_svg(cv, src, 0, -h / 2 + m + sq / 2, sq * 0.78, None, 0.8)
            if p.get("tag"):
                ts = w * 0.17
                tw_ = text_w(p["tag"], ts) + ts * 1.1
                ty = h / 2 - (h - sq - m) / 2 + w * 0.03
                pts = [[-tw_ / 2, ty - ts * 0.75], [tw_ / 2, ty - ts * 0.75], [tw_ / 2 + ts * 0.5, ty], [tw_ / 2, ty + ts * 0.75], [-tw_ / 2, ty + ts * 0.75]]
                cv.poly(pts, PK.PAL["white"], lift=1.0)
                cv.text(p["tag"], 0, ty, ts, col(p.get("tag_color", "red")), F_STICKER, "mm")
            return cv.finish(seed=w, base_ppu=self.S)
        return [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]

    def parts_bubble(self, o, p, t, ppu):
        txt = str(p.get("text", "?"))
        size = p.get("size", 34)
        tail = p.get("tail", "left")
        key = ("bubble", txt, size, tail, json.dumps(p.get("bg")), json.dumps(p.get("ink")), round(ppu, 3))

        def build():
            tw_ = text_w(txt, size)
            rx, ry = tw_ / 2 + size * 0.9, size * 0.95
            cv = PK.Canvas((-rx - size, -ry - 4, rx + size, ry + size * 0.9), ppu)
            bg = col(p.get("bg", "cream"))
            if tail != "none":
                if tail == "left":
                    tp = [[-rx * 0.55, ry * 0.5], [-rx * 0.95, ry + size * 0.7], [-rx * 0.2, ry * 0.75]]
                elif tail == "right":
                    tp = [[rx * 0.55, ry * 0.5], [rx * 0.95, ry + size * 0.7], [rx * 0.2, ry * 0.75]]
                else:
                    tp = [[-size * 0.4, ry * 0.8], [0, ry + size * 0.8], [size * 0.4, ry * 0.8]]
                cv.poly(tp, PK.shade(bg, 0.92))
            cv.ellipse(0, 0, rx, ry, bg)
            cv.text(txt, 0, 0, size, col(p.get("ink", "ink")), F_STICKER, "mm")
            return cv.finish(seed=len(txt), base_ppu=self.S)
        return [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]

    def parts_arrow(self, o, p, t, ppu):
        x, y = p.get("x", 0), p.get("y", 0)
        to = p.get("to", [x + 200, y])
        pts_abs = p.get("points")
        draw = clamp(p.get("draw", 1.0))
        dq = round(draw * 40) / 40
        width = p.get("width", 12)
        c = col(p.get("color", "yellow"))
        key = ("arrow", json.dumps(pts_abs), json.dumps(to), x, y, p.get("bend", 0.18), width, tuple(c), dq, p.get("head", True), round(ppu, 3))
        if dq <= 0.001:
            return []

        def build():
            if pts_abs:
                path = PK.catmull(np.asarray(pts_abs, float) - [x, y], False, 12)
            else:
                path = PK.bezier((0, 0), (to[0] - x, to[1] - y), p.get("bend", 0.18), 48)
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            L = cum[-1] * dq
            k = int(np.searchsorted(cum, L))
            part = path[:max(2, k)]
            if k < len(path) and k > 0:
                q = (L - cum[k - 1]) / max(seg[k - 1], 1e-6)
                part = np.vstack([path[:k], path[k - 1] + (path[k] - path[k - 1]) * q])
            hs = width * 2.0
            mn, mx = path.min(0) - hs * 1.5, path.max(0) + hs * 1.5
            cv = PK.Canvas((mn[0], mn[1], mx[0], mx[1]), ppu)
            if p.get("head", True) and len(part) >= 2:
                tip = part[-1]
                back = part[max(0, len(part) - 4)]
                d = tip - back
                d = d / max(np.linalg.norm(d), 1e-6)
                nrm = np.array([-d[1], d[0]])
                body_end = tip - d * hs * 0.9
                body = np.vstack([part[:-1], body_end]) if len(part) > 2 else np.vstack([part[0], body_end])
                cv.stroke(body, width, c, 0, False)
                cv.poly([tip + d * hs * 0.25, body_end + nrm * hs, body_end - nrm * hs], c)
            else:
                cv.stroke(part, width, c, 0, False)
            return cv.finish(seed=7, base_ppu=self.S)
        return [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]

    def parts_ring(self, o, p, t, ppu):
        rx, ry = p.get("rx", p.get("r", 80)), p.get("ry", p.get("r", 60))
        draw = round(clamp(p.get("draw", 1.0)) * 36) / 36
        c = col(p.get("color", "yellow"))
        width = p.get("width", 10)
        key = ("ring", rx, ry, draw, tuple(c), width, round(ppu, 3))
        if draw <= 0.001:
            return []

        def build():
            cv = PK.Canvas((-rx - width, -ry - width, rx + width, ry + width), ppu)
            a0 = -math.pi * 0.6
            pts = PK.ellipse_pts(0, 0, rx, ry, 90, 0, a0, a0 + 2 * math.pi * min(draw, 0.999) * 1.04)
            cv.stroke(pts, width, c, 0, draw >= 0.999)
            return cv.finish(seed=3, base_ppu=self.S)
        return [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]

    def parts_mark(self, o, p, t, ppu):
        kind = p.get("kind", "check")
        s = p.get("size", 60)
        c = col(p.get("color", "green" if kind == "check" else "red"))
        key = ("mark", kind, s, tuple(c), round(ppu, 3))

        def build():
            cv = PK.Canvas((-s / 2 - 8, -s / 2 - 8, s / 2 + 8, s / 2 + 8), ppu)
            w = s * 0.2
            if kind == "check":
                cv.stroke([[-s * 0.42, 0], [-s * 0.12, s * 0.3], [s * 0.45, -s * 0.38]], w, c, 0, False)
            else:
                cv.stroke([[-s * 0.38, -s * 0.38], [s * 0.38, s * 0.38]], w, c, 0, False)
                cv.stroke([[s * 0.38, -s * 0.38], [-s * 0.38, s * 0.38]], w, c, 0.8, False)
            return cv.finish(seed=5, base_ppu=self.S)
        return [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]

    def parts_burst(self, o, p, t, ppu):
        r = p.get("r", 80)
        n = int(p.get("points", 16))
        c = col(p.get("color", "yellow"))
        key = ("burst", r, n, tuple(c), round(ppu, 3))

        def build():
            cv = PK.Canvas((-r, -r, r, r), ppu)
            pts = []
            for i in range(2 * n):
                a = math.pi * i / n - math.pi / 2
                rr = r if i % 2 == 0 else r * p.get("inner", 0.74)
                pts.append((rr * math.cos(a), rr * math.sin(a)))
            cv.poly(pts, c)
            return cv.finish(seed=11, base_ppu=self.S)
        rot = t * p.get("spin", 6)
        out = [(self.sprite(key, build), 0, 0, rot, 1, 1, 1)]
        if p.get("text"):
            ts = p.get("text_size", r * 0.55)
            out.append((self.text_sprite(str(p["text"]), ts, col(p.get("text_color", "coral")), ppu), 0, 0, 0, 1, 1, 1))
        return out

    def text_sprite(self, txt, size, color, ppu, outline=None, fp=F_STICKER):
        outline = size * 0.12 if outline is None else outline
        key = ("T", txt, round(size, 1), tuple(color) if not isinstance(color, list) else json.dumps(color), round(outline, 2), round(ppu, 3), fp)

        def build():
            lay = self.letter_layout(txt, size, color if isinstance(color, list) else [color], fp, 0.02)
            w = text_w(txt, size, fp) * 1.05 + 2 * outline + 12
            hgt = size * 1.4 * (txt.count("\n") + 1) + 2 * outline
            cv = PK.Canvas((-w / 2, -hgt / 2, w / 2, hgt / 2), ppu, pad=8)
            # outline pass for all letters first (so borders merge), then letters
            for ch, x, y, c in lay:
                f = PK.font(fp, size * cv.ppu)
                px, py = cv.P([(x, y)])[0]
                im = Image.new("L", (cv.W, cv.H), 0)
                from PIL import ImageDraw
                ImageDraw.Draw(im).text((px, py), ch, font=f, fill=255, anchor="mm", stroke_width=int(round(outline * cv.ppu)), stroke_fill=255)
                cv.comp(np.array(im), PK.PAL["cream"])
            for ch, x, y, c in lay:
                cv.text(ch, x, y, size, c, fp, "mm")
            return cv.finish(shadow=(2, 4, 3.5, 0.45), seed=len(txt) * 3, base_ppu=self.S)
        return self.sprite(key, build)

    def fmt_num(self, v, p):
        dec = int(p.get("decimals", 0))
        sep = p.get("sep", "." if self.lang == "id" else ",")
        dsep = "," if sep == "." else "."
        s = f"{abs(v):,.{dec}f}"
        s = s.replace(",", "\0").replace(".", dsep).replace("\0", sep)
        return ("-" if v < 0 else "") + p.get("prefix", "") + s

    def parts_counter(self, o, p, t, ppu):
        size = p.get("size", 56)
        txt = self.fmt_num(float(p.get("value", 0)), p)
        spr = self.text_sprite(txt, size, col(p.get("color", "yellow")), ppu)
        out = []
        w_num = text_w(txt, size) * 1.02 + size * (0.12 if len(str(p.get("suffix") or "")) <= 1 else 0.22)
        align = p.get("align", "left")
        suf = p.get("suffix")
        ss = p.get("suffix_size", size * 0.62)
        w_suf = (text_w(suf, ss) + ss * (0.12 if len(str(suf)) <= 1 else 0.45)) if suf else 0
        total = w_num + w_suf
        x0 = 0 if align == "left" else -total / 2 if align == "center" else -total
        out.append((spr, x0 + w_num / 2, 0, 0, 1, 1, 1))
        if suf:
            ssp = self.text_sprite(suf, ss, col(p.get("suffix_color", "teal")), ppu)
            out.append((ssp, x0 + w_num + w_suf / 2, size * 0.08, 0, 1, 1, 1))
        return out

    def parts_bars(self, o, p, t, ppu):
        w, h = p.get("w", 440), p.get("h", 320)
        vals = p.get("values", [0.3, 0.5, 0.7, 0.9])
        n = max(1, len(vals))
        cyc = p.get("colors", ["yellow", "teal"])
        hl = p.get("highlight_last", True)
        axis_c = col(p.get("axis_color", "cream"))
        aw = p.get("axis_width", 7)
        akey = ("axes", w, h, tuple(axis_c), aw, round(ppu, 3))

        def axes():
            cv = PK.Canvas((-aw, -h - aw, w + aw, aw), ppu)
            cv.rrect(-aw / 2, -h, aw / 2, aw / 2, 2, axis_c)
            cv.rrect(-aw / 2, -aw / 2, w, aw / 2, 2, axis_c)
            return cv.finish(seed=2, base_ppu=self.S)
        out = [(self.sprite(akey, axes), 0, 0, 0, 1, 1, 1)]
        gap = p.get("gap", 0.32)
        slot = (w - aw * 2) / n
        bw = slot * (1 - gap)
        grow = p.get("grow", 1.0)
        for i, v in enumerate(vals):
            c = col(p.get("last_color", "coral")) if (hl and i == n - 1) else col(cyc[i % len(cyc)])
            bkey = ("bar", round(bw, 2), tuple(c), round(ppu, 3))

            def bar(c=c):
                cv = PK.Canvas((-bw / 2, -100, bw / 2, 0), ppu, pad=10)
                cv.rrect(-bw / 2, -100, bw / 2, 0, 1.5, c)
                return cv.finish(shadow=(2, 3, 3, 0.4), seed=int(bw), base_ppu=self.S)
            q = ease_out(clamp(grow * n - i))
            if q <= 0.002:
                continue
            hh = max(0.01, float(v) * (h - 10) * q)
            out.append((self.sprite(bkey, bar), aw + slot * (i + 0.5), -aw / 2, 0, 1, hh / 100, 1))
        if p.get("label"):
            out.append((self.tag_sprite(str(p["label"]), p.get("label_size", 20), col("cream"), col("ink"), "rect", ppu), w / 2, 34, 0, 1, 1, 1))
        return out

    def parts_conveyor(self, o, p, t, ppu):
        w = p.get("w", 600)
        hb = p.get("h", 22)
        c = col(p.get("color", (44, 92, 84)))
        key = ("conv", w, hb, tuple(c), int(p.get("legs", 3)), round(ppu, 3))

        def build():
            cv = PK.Canvas((-4, -hb / 2 - 4, w + 4, hb / 2 + 70), ppu)
            nl = int(p.get("legs", 3))
            for k in range(nl):
                lx = w * (k + 0.5) / nl
                cv.rrect(lx - 8, hb / 2 - 2, lx + 8, hb / 2 + 60, 2, PK.shade(c, 0.85))
                cv.rrect(lx - 16, hb / 2 + 52, lx + 16, hb / 2 + 62, 2, PK.shade(c, 0.85))
            cv.rrect(0, -hb / 2, w, hb / 2, hb / 2, c, lift=1.0)
            return cv.finish(seed=9, base_ppu=self.S)
        out = [(self.sprite(key, build), 0, 0, 0, 1, 1, 1)]
        rc = col(p.get("roller_color", "yellow"))
        rr = hb * 0.3
        dk = ("dot", round(rr, 2), tuple(rc), round(ppu, 3))

        def dot():
            cv = PK.Canvas((-rr, -rr, rr, rr), ppu, pad=6)
            cv.circle(0, 0, rr, rc)
            return cv.finish(shadow=(0.8, 1.2, 1.5, 0.35), seed=4, base_ppu=self.S)
        sp = p.get("spacing", 46)
        off = (t * p.get("speed", 60)) % sp
        x = hb / 2 + off
        while x < w - hb / 2:
            out.append((self.sprite(dk, dot), x, 0, 0, 1, 1, 1))
            x += sp
        return out

    def parts_dials(self, o, p, t, ppu):
        n = int(p.get("n", 3))
        r = p.get("r", 22)
        c = col(p.get("color", (60, 140, 132)))
        pw, ph = n * r * 2.6 + r * 0.8, r * 3.0
        key = ("dpanel", n, r, tuple(c), round(ppu, 3))

        def panel():
            cv = PK.Canvas((-pw / 2, -ph / 2, pw / 2, ph / 2), ppu)
            cv.rrect(-pw / 2, -ph / 2, pw / 2, ph / 2, 4, c)
            for k in range(n):
                cv.circle(-pw / 2 + r * 1.7 + k * r * 2.6, 0, r, PK.PAL["cream"], lift=0.8)
            return cv.finish(seed=6, base_ppu=self.S)
        nk = ("needle", r, round(ppu, 3))

        def needle():
            cv = PK.Canvas((-r * 0.2, -r * 0.85, r * 0.2, r * 0.85), ppu, pad=6)
            cv.poly([[0, -r * 0.8], [r * 0.16, 0], [0, r * 0.8], [-r * 0.16, 0]], col("red"))
            return cv.finish(shadow=(0.8, 1.4, 1.5, 0.4), seed=8, base_ppu=self.S)
        out = [(self.sprite(key, panel), 0, 0, 0, 1, 1, 1)]
        base = p.get("angle", 0)
        angs = p.get("angles", [30, -20, 50, 10, -40])
        for k in range(n):
            a = base + angs[k % len(angs)] + math.sin(t * 1.3 + k) * p.get("jitter", 3)
            out.append((self.sprite(nk, needle), -pw / 2 + r * 1.7 + k * r * 2.6, 0, a, 1, 1, 1))
        return out

    def strip_sprite(self, text, size, ppu, ink=None, hl=None, bg=None):
        ink = col(ink or "ink")
        hl = col(hl or "coral")
        bg = col(bg or "cream")
        key = ("strip", json.dumps(text), size, tuple(ink), tuple(hl), tuple(bg), round(ppu, 3))

        def build():
            if isinstance(text, list):
                segs = [str(s) for s in text]
            else:
                segs = re.split(r"\s*(?:->|→)\s*", str(text))
            arrow_w = size * 1.7
            widths = [text_w(s, size) for s in segs]
            total = sum(widths) + arrow_w * (len(segs) - 1)
            w, h = total + size * 1.8, size * 2.0
            cv = PK.Canvas((-w / 2, -h / 2, w / 2, h / 2), ppu)
            edge_l = PK.jag(np.array([[-w / 2, -h / 2], [-w / 2, h / 2]]), 1.6, 2.5, 4)
            edge_r = PK.jag(np.array([[w / 2, h / 2], [w / 2, -h / 2]]), 1.6, 2.5, 5)
            cv.poly(np.vstack([edge_l, edge_r]), bg)
            x = -total / 2
            for i, s in enumerate(segs):
                c = ink if (i == 0 or len(segs) == 1) else hl
                cv.text(s, x + widths[i] / 2, 0, size, c, F_STICKER, "mm")
                x += widths[i]
                if i < len(segs) - 1:
                    a0, a1 = x + size * 0.3, x + arrow_w - size * 0.3
                    cv.stroke([[a0, 0], [a1 - size * 0.35, 0]], size * 0.28, col("teal"), 0, False)
                    cv.poly([[a1, 0], [a1 - size * 0.55, -size * 0.38], [a1 - size * 0.55, size * 0.38]], col("teal"))
                    x += arrow_w
            return cv.finish(seed=12, base_ppu=self.S)
        return self.sprite(key, build)

    def parts_strip(self, o, p, t, ppu):
        spr = self.strip_sprite(p.get("text", ""), p.get("size", 30), ppu, p.get("ink"), p.get("hl"), p.get("bg"))
        return [(spr, 0, 0, 0, 1, 1, 1)]

    # ---------------------------------------------------------------- background
    def build_sky(self):
        bg = self.bg
        PW, PH = int(self.W * 1.3), int(self.H * 1.3)
        g = np.linspace(0, 1, PH, dtype=np.float32)[:, None, None]
        top, bot = np.array(bg["top"], np.float32), np.array(bg["bottom"], np.float32)
        img = np.repeat(top + (bot - top) * g, PW, 1)
        T = PK.tex_patch(PH, PW, scale=1.0 / max(self.S, 0.5), seed=21)
        rng = np.random.default_rng(2)
        mot = cv2.GaussianBlur(rng.normal(0, 1, (PH // 8 + 2, PW // 8 + 2)).astype(np.float32), (0, 0), 5)
        mot = cv2.resize(mot / (mot.std() + 1e-6), (PW, PH), interpolation=cv2.INTER_CUBIC)
        lum = img.mean(axis=2, keepdims=True)
        dark = float(lum.mean()) < 130
        img = img * (1 + 0.045 * T[:, :, None]) + (np.clip(mot, 0, None)[:, :, None] * (16 if dark else 10))
        yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
        d = np.sqrt(((xx - PW / 2) / (PW / 2)) ** 2 + ((yy - PH / 2) / (PH / 2)) ** 2)
        img *= (1 - 0.22 * np.clip(d - 0.45, 0, 1) ** 1.6)[:, :, None]
        return np.clip(img, 0, 255).astype(np.float32)

    def build_stars(self):
        n = int(self.F.get("stars", self.bg.get("stars", 0)))
        if self.spec.get("stars") is not None:
            n = int(self.spec["stars"])
        if n <= 0:
            return []
        PW, PH = self.sky.shape[1], self.sky.shape[0]
        rng = np.random.default_rng(17)
        sprites = []
        for k in range(4):
            r = (1.6 + k * 0.9) * self.S
            cv = PK.Canvas((-r * 1.6, -r * 1.6, r * 1.6, r * 1.6), 1.0, pad=4)
            if k >= 2:
                pts = []
                for i in range(10):
                    a = math.pi * i / 5 - math.pi / 2
                    rr = r * 1.5 if i % 2 == 0 else r * 0.6
                    pts.append((rr * math.cos(a), rr * math.sin(a)))
                cv.poly(pts, (250, 238, 196))
            else:
                cv.circle(0, 0, r, (248, 238, 206))
            sprites.append(cv.finish(shadow=(0.6, 1, 1.2, 0.3), tex=0, edge=0, seed=k))
        out = []
        for i in range(n):
            y = rng.uniform(0, 0.8) ** 1.3 * PH * 0.8
            out.append((rng.uniform(0, PW), y, sprites[int(rng.choice(4, p=[0.45, 0.3, 0.17, 0.08]))], rng.uniform(0, 6.3), rng.uniform(0.6, 1.8)))
        return out

    # ---------------------------------------------------------------- render one frame
    def frame(self, t):
        cam = self.camera(t)
        PW, PH = self.sky.shape[1], self.sky.shape[0]
        ox = int(clamp((PW - self.W) / 2 - (cam["x"] - 360) * 0.06 * self.S, 0, PW - self.W))
        oy = int(clamp((PH - self.H) / 2 - (cam["y"] - 640) * 0.04 * self.S, 0, PH - self.H))
        img = self.sky[oy:oy + self.H, ox:ox + self.W].copy()
        for (sx, sy, spr, ph, sp) in self.stars:
            x, y = sx - ox, sy - oy
            if -10 < x < self.W + 10 and -10 < y < self.H + 10:
                a = 0.55 + 0.45 * math.sin(t * sp + ph)
                PK.place(img, spr, PK.affine(x, y, 0, 1, 1, spr.ax, spr.ay), a)
        objs = sorted(self.objects, key=lambda o: (o.get("layer", 0), o["_ord"]))
        for o in objs:
            al, vs, vr, vdx, vdy, q, exiting = self.vis(o, t)
            if al <= 0.004:
                continue
            p = self.props(o, t)
            al *= float(p.get("alpha", 1.0))
            if al <= 0.004:
                continue
            if q is not None:
                p["_q"] = q
                p["_D"] = self.default_show_dur(o)
                if o.get("type") == "text" and o.get("letters", True):
                    vs, vr = 1.0, 0.0
            if exiting:
                p.pop("_q", None)
            bump = 1.0
            for (bt, amt) in o["_bumps"]:
                if bt <= t <= bt + 0.5:
                    bump *= 1 + amt * math.sin(math.pi * (t - bt) / 0.5)
            screen = p.get("screen")
            if screen:
                pos = p.get("pos")
                if pos:
                    X, Y = pos[0] * self.W, pos[1] * self.H
                else:
                    X, Y = self.W / 2 + (p.get("x", 360) - 360) * self.S, self.H / 2 + (p.get("y", 640) - 640) * self.S
                k = self.S
                ppu = self.hud_ppu
                X += vdx * k
                Y += vdy * k
            else:
                par = float(p.get("parallax", 1.0))
                bx = p.get("x", 0) + vdx
                by = p.get("y", 0) + vdy
                if p.get("bob"):
                    by += p["bob"] * math.sin(t * 1.8 + p.get("x", 0) * 0.013)
                X, Y, k = self.to_screen(bx, by, cam, par)
                ppu = self.ppu
                if o.get("type") == "hill":
                    X, Y, k = self.to_screen(0, 0, cam, par)
            rot = float(p.get("rotate", 0)) + vr
            if p.get("wobble"):
                rot += p["wobble"] * math.sin(t * 2.1 + o["_ord"])
            scl = float(p.get("scale", 1.0)) * vs * bump
            sxm, sym = float(p.get("sx", 1.0)), float(p.get("sy", 1.0))
            for (spr, dx, dy, r2, s2x, s2y, a2) in self.parts(o, p, t, ppu):
                if spr is None:
                    continue
                # local offset -> rotate/scale by object transform
                rr = math.radians(rot)
                lx, ly = dx * scl * sxm, dy * scl * sym
                px = X + (lx * math.cos(rr) - ly * math.sin(rr)) * k
                py = Y + (lx * math.sin(rr) + ly * math.cos(rr)) * k
                fx = k * scl * sxm * s2x / spr.ppu
                fy = k * scl * sym * s2y / spr.ppu
                M = PK.affine(px, py, rot + r2, fx, fy, spr.ax, spr.ay)
                PK.place(img, spr, M, al * a2)
        self.draw_hud(img, t)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------- HUD: title / chip / captions / outro
    def chip_text(self):
        c = self.meta.get("chip", True)
        if c is False or not self.F.get("chip", True):
            return None
        if isinstance(c, str):
            return c
        tt = self.meta.get("title")
        if not tt:
            return None
        return " ".join(tt) if isinstance(tt, list) else str(tt)

    def title_layout(self):
        lines = self.meta["title"]
        lines = lines if isinstance(lines, list) else [lines]
        S = self.S
        sizes = [80, 100] if len(lines) == 2 else [96] * len(lines)
        maxw = self.W * 0.88 / S
        out = []
        for i, ln in enumerate(lines):
            s = sizes[i] if i < len(sizes) else 80
            w = text_w(ln, s)
            if w > maxw:
                s *= maxw / w
            out.append((ln, s))
        return out

    def draw_hud(self, img, t):
        S = self.S
        F = self.F
        meta = self.meta
        ppu = self.hud_ppu
        o_start = self.outro_start()
        in_outro = meta.get("outro") and F["outro"] and t >= o_start
        # chip (persistent topic sticker, top-left)
        chip = self.chip_text() if (meta.get("title") and F["title"]) else (self.chip_text() if meta.get("chip") else None)
        te = self.title_end()
        if self.title_style() == "sign":
            self.draw_sign(img, t, chip, te, in_outro, o_start)
        else:
            if meta.get("title") and F["title"] and t < te + 0.6:
                lay = self.title_layout()
                cy = self.H * (0.30 if self.portrait else 0.36)
                gap = 1.08
                total_h = sum(s for _, s in lay) * gap
                y = cy - total_h / 2 * S
                fly = clamp((t - te) / 0.55) if chip else clamp((t - te) / 0.4)
                fe = ease_io(fly)
                k_idx = 0
                chip_spr = self.tag_sprite(chip, 24, col("coral"), col("cream"), "pill", ppu) if chip else None
                chx, chy = self.chip_pos(chip_spr)
                for (ln, s) in lay:
                    y_line = y + s * gap * S / 2
                    for (ch, lx, ly, c) in self.letter_layout(ln, s, "cycle"):
                        spr = self.sticker_letter(ch, s, c, s * 0.12, ppu)
                        st = 0.045
                        lq = clamp((t - 0.25 - k_idx * st) / 0.32)
                        k_idx += 1
                        if lq <= 0:
                            continue
                        rng = np.random.default_rng(k_idx * 13)
                        rot = float(rng.uniform(-3.5, 3.5)) + (1 - ease_out(lq)) * -14
                        sc = max(0.01, ease_back(lq))
                        X = self.W / 2 + lx * S
                        Y = y_line + ly * S + math.sin(t * 2.4 + k_idx * 0.5) * 1.2 * S
                        a = min(1, lq * 4)
                        if fly > 0:
                            if chip:
                                # shrink + fly toward the chip
                                tx = chx + (X - self.W / 2) * 0.18
                                ty_ = chy + (Y - cy) * 0.18
                                X, Y = X + (tx - X) * fe, Y + (ty_ - Y) * fe
                                sc *= 1 - 0.82 * fe
                                a *= 1 - clamp((fly - 0.45) / 0.4)
                            else:
                                a *= 1 - fly
                                sc *= 1 - 0.3 * fly
                        PK.place(img, spr, PK.affine(X, Y, rot, S * sc / spr.ppu, S * sc / spr.ppu, spr.ax, spr.ay), a)
                    y += s * gap * S
            if chip and t >= te + 0.3 and not in_outro:
                spr = self.tag_sprite(chip, 24, col("coral"), col("cream"), "pill", ppu)
                q = clamp((t - te - 0.3) / 0.35)
                x, y = self.chip_pos(spr)
                sc = max(0.01, ease_back(q))
                PK.place(img, spr, PK.affine(x, y, -2, S * sc / spr.ppu, S * sc / spr.ppu, spr.ax, spr.ay), min(1, q * 3))
            elif chip and in_outro:
                q = clamp((t - o_start) / 0.3)
                if q < 1:
                    spr = self.tag_sprite(chip, 24, col("coral"), col("cream"), "pill", ppu)
                    x, y = self.chip_pos(spr)
                    PK.place(img, spr, PK.affine(x, y, -2, S / spr.ppu, S / spr.ppu, spr.ax, spr.ay), 1 - q)
        # captions
        cf = F["captions"]
        if cf["show"]:
            cy = (min(self.H * cf["y"], self.H * 0.86 - 135 * S) if in_outro else self.H * cf["y"])
            for i, c in enumerate(self.caps):
                if c["start"] <= t < c["end"]:
                    up = cf.get("uppercase")
                    words = [w.upper() if up else w for w in c["words"]]
                    segs = " ".join(words)
                    hlw = [w for w, h in zip(words, c["hl"]) if h]
                    size = 40 * cf["size"] * (1.15 if cf["words"] == 1 else 1.0)
                    spr = self.caption_sprite(words, c["hl"], size, ppu)
                    lt = t - c["start"]
                    sc = 0.9 + 0.1 * ease_back(clamp(lt / 0.16))
                    maxw = self.W * 0.92
                    wpx = spr.levels[0].shape[1] / spr.ppu * S
                    if wpx > maxw:
                        sc *= maxw / wpx
                    rot = 1.2 if i % 2 else -1.2
                    PK.place(img, spr, PK.affine(self.W / 2, cy, rot, S * sc / spr.ppu, S * sc / spr.ppu, spr.ax, spr.ay), min(1, lt / 0.08))
                    break
        # outro: takeaway strip + CTA sticker
        if in_outro:
            lt = t - o_start
            txt = meta["outro"]
            if isinstance(txt, list):
                txt = txt if len(txt) > 1 else txt[0]
            size = 30 if self.portrait else 34
            spr = self.strip_sprite(txt, size, ppu)
            wpx = spr.levels[0].shape[1] / spr.ppu * S
            fit = min(1.0, self.W * 0.94 / max(wpx, 1))
            q = ease_out(clamp(lt / 0.5))
            y = self.H * (0.86 if self.portrait else 0.86) + (1 - q) * 160 * S
            PK.place(img, spr, PK.affine(self.W / 2, y, -0.8, S * fit / spr.ppu, S * fit / spr.ppu, spr.ax, spr.ay), min(1, lt / 0.2))
            cta = F.get("cta")
            if cta is None:
                cta = meta.get("cta", "FOLLOW UNTUK LAINNYA" if self.lang == "id" else "FOLLOW FOR MORE")
            if cta:
                cq = clamp((lt - 0.55) / 0.35)
                if cq > 0:
                    cs = self.tag_sprite(str(cta).upper(), 20, col("coral"), col("cream"), "pill", ppu, F_HEAVY)
                    sc = max(0.01, ease_back(cq))
                    hh = spr.levels[0].shape[0] / spr.ppu * S * fit
                    PK.place(img, cs, PK.affine(self.W / 2, y - hh / 2 - 34 * S, 2, S * sc / cs.ppu, S * sc / cs.ppu, cs.ax, cs.ay), min(1, cq * 3))
        self.draw_watermark(img)

    # ---------------------------------------------------------------- intro style "sign" (default)
    def title_style(self):
        st = self.meta.get("title_style") or self.F.get("title_style") or "sign"
        return "pop" if str(st).lower() in ("pop", "classic", "klasik") else "sign"

    def sign_geom(self):
        """layout of the hanging paper sign (design units) + timing of each word drop."""
        if hasattr(self, "_sign"):
            return self._sign
        lines = self.meta["title"]
        lines = lines if isinstance(lines, list) else [lines]
        maxw = (self.W / self.S) * 0.78
        base = [70, 88] if len(lines) == 2 else [84] * len(lines)
        words, rows, y = [], [], 0.0
        widest = 0
        sizes = []
        for li, ln in enumerate(lines):
            sz = base[li] if li < len(base) else 70
            w = text_w(ln, sz) * 1.12
            if w > maxw:
                sz *= maxw / w
                w = maxw
            sizes.append(sz)
            widest = max(widest, w)
        lh = [sz * 1.12 for sz in sizes]
        total = sum(lh)
        y = -total / 2
        cyc = ["teal", "coral", "lavender"]
        wi = 0
        nwords = sum(len(str(ln).split()) for ln in lines)
        for li, ln in enumerate(lines):
            sz = sizes[li]
            yc = y + lh[li] / 2
            toks = str(ln).split()
            sp = text_w(" ", sz) * 1.3
            wl = [text_w(tk, sz) * 1.12 for tk in toks]
            x = -(sum(wl) + sp * (len(toks) - 1)) / 2
            for tk, ww in zip(toks, wl):
                last = wi == nwords - 1
                c = col("coral") if last else col(cyc[wi % len(cyc)])
                letters = []
                lx = x
                for ch in tk:
                    adv = text_w(ch, sz) * 1.12
                    letters.append((ch, lx + adv / 2))
                    lx += adv
                words.append({"letters": letters, "y": yc, "size": sz * (1.06 if last else 1.0), "color": c,
                              "rot": (-4 if wi % 2 else 3.5), "t": 0.55 + wi * 0.24})
                x += ww + sp
                wi += 1
            y += lh[li]
        cw, ch_ = widest + 90, total + 70
        self._sign = {"words": words, "w": cw, "h": ch_, "word_t": [w["t"] for w in words],
                      "confetti_t": (words[-1]["t"] + 0.45) if words else 1.0}
        return self._sign

    def sign_sprite(self, w, h, ppu):
        key = ("sign", round(w, 1), round(h, 1), round(ppu, 3))

        def build():
            cv = PK.Canvas((-w / 2 - 6, -h / 2 - 6, w / 2 + 6, h / 2 + 6), ppu)
            pts = PK.rrect_pts(-w / 2, -h / 2, w / 2, h / 2, 6)
            cv.poly(PK.jag(np.vstack([pts, pts[:1]]), 1.5, 3.2, 21), PK.PAL["cream"])
            # a second, slightly smaller layer of kraft-tinted paper peeking at the bottom edge (two-ply card)
            cv.rrect(-w / 2 + 10, h / 2 - 12, w / 2 - 10, h / 2 - 4, 3, PK.mix(PK.PAL["cream"], PK.PAL["kraft"], 0.35), lift=0.6)
            for sx in (-1, 1):                      # punched holes with paper grommets
                hx, hy = sx * w * 0.32, -h / 2 + 15
                cv.circle(hx, hy, 7.5, PK.mix(PK.PAL["cream"], PK.PAL["kraft"], 0.5), lift=0.5)
                cv.circle(hx, hy, 3.6, PK.PAL["ink"])
            # tiny paper stars on the corners
            for (sx, sy, cc) in ((-w / 2 + 20, h / 2 - 26, "yellow"), (w / 2 - 22, h / 2 - 28, "pink")):
                st = []
                for i in range(10):
                    a = math.pi * i / 5 - math.pi / 2
                    rr = 10 if i % 2 == 0 else 4.2
                    st.append((sx + rr * math.cos(a), sy + rr * math.sin(a)))
                cv.poly(st, col(cc), lift=0.7)
            return cv.finish(shadow=(3, 7, 7, 0.45), tex=0.028, seed=31, base_ppu=self.S)
        return self.sprite(key, build)

    def draw_sign(self, img, t, chip, te, in_outro, o_start):
        S, ppu = self.S, self.hud_ppu
        meta, F = self.meta, self.F
        if meta.get("title") and F["title"] and t < te + 0.7:
            g = self.sign_geom()
            cw, chh = g["w"], g["h"]
            cy = self.H * (0.31 if self.portrait else 0.40)
            P = (self.W / 2, -40 * S)                       # string pivot (just above the frame)
            D0 = cy - P[1]
            drop = clamp((t - 0.02) / 0.62)
            yoff = -(1 - ease_bounce(drop)) * (cy + chh * S)
            yank = ease_in(clamp((t - te) / 0.5))
            yoff -= yank * (cy + chh * S * 1.2)
            ts = t - 0.5
            theta = (8 * math.exp(-2.0 * max(ts, 0)) * math.sin(6.5 * max(ts, 0)) if ts > 0 else 0) + 1.1 * math.sin(t * 1.6)
            th = math.radians(theta)
            c_, s_ = math.cos(th), math.sin(th)
            D = D0 + yoff

            def T(lx, ly):   # sign-local (screen px rel. to card centre) -> screen
                bx, by = lx, ly + D
                return P[0] + bx * c_ - by * s_, P[1] + bx * s_ + by * c_
            # strings (drawn first, behind the card)
            for sx in (-1, 1):
                hx, hy = T(sx * cw * 0.32 * S, -chh / 2 * S + 15 * S)
                cv2.line(img, (int(P[0]), int(P[1])), (int(hx), int(hy)), (232, 224, 204), max(1, int(round(2 * S))), cv2.LINE_AA)
            spr = self.sign_sprite(cw, chh, ppu)
            cx, cyy = T(0, 0)
            PK.place(img, spr, PK.affine(cx, cyy, theta, S / spr.ppu, S / spr.ppu, spr.ax, spr.ay))
            # words drop onto the card, bounce + squash
            for wd in g["words"]:
                lq0 = t - wd["t"]
                for j, (ch, lx) in enumerate(wd["letters"]):
                    lq = clamp((lq0 - j * 0.03) / 0.42)
                    if lq <= 0:
                        continue
                    yo = -(1 - ease_bounce(lq)) * 150
                    sq = math.sin(math.pi * clamp((lq - 0.34) / 0.22)) if lq > 0.34 else 0
                    sy_, sx_ = 1 - 0.22 * sq, 1 + 0.16 * sq
                    rot = wd["rot"] + (1 - lq) * 10 * (1 if j % 2 else -1)
                    sz = wd["size"]
                    sp = self.sticker_letter(ch, sz, wd["color"], sz * 0.07, ppu, F_STICKER, PK.PAL["white"])
                    X, Y = T(lx * S, (wd["y"] + yo + sz * 0.1 * (1 - sy_)) * S)
                    PK.place(img, sp, PK.affine(X, Y, theta + rot, S * sx_ / sp.ppu, S * sy_ / sp.ppu, sp.ax, sp.ay), min(1, lq * 5))
            # paper confetti burst when the last word lands
            tc = t - g["confetti_t"]
            if 0 <= tc <= 1.6:
                rng = np.random.default_rng(77)
                for k in range(18):
                    ang = rng.uniform(-math.pi * 0.95, -math.pi * 0.05)
                    v = rng.uniform(260, 520)
                    ox = rng.uniform(-cw / 2, cw / 2)
                    x0, y0 = T(ox * S, -chh / 2 * S)
                    x = x0 + math.cos(ang) * v * tc * S
                    y = y0 + (math.sin(ang) * v * tc + 0.5 * 900 * tc * tc) * S
                    c = ["coral", "yellow", "teal", "pink", "cream"][k % 5]
                    bit = self.confetti_sprite(c, k % 2, ppu)
                    PK.place(img, bit, PK.affine(x, y, rng.uniform(0, 360) + tc * rng.uniform(-500, 500), S / bit.ppu, S * abs(math.cos(tc * 9 + k)) / bit.ppu + 0.02, bit.ax, bit.ay),
                             clamp((1.6 - tc) / 0.5))
        # chip: a small tag hanging on its own string, top-left, gently swaying
        if chip and t >= te + 0.35 and not (in_outro and t > o_start + 0.4):
            spr = self.tag_sprite(chip, 24, col("coral"), col("cream"), "pill", ppu)
            x, y = self.chip_pos(spr)
            q = clamp((t - te - 0.35) / 0.55)
            yo = -(1 - ease_bounce(q)) * (y + 60 * S)
            if in_outro:
                yo -= ease_in(clamp((t - o_start) / 0.4)) * (y + 60 * S)
            tt = t - te - 0.35
            ang = 6 * math.exp(-2.4 * tt) * math.sin(7 * tt) + 1.4 * math.sin(t * 1.3)
            h = spr.levels[0].shape[0] / spr.ppu * S
            px, py = x, -10 * S
            L = (y + yo) - py
            r = math.radians(ang)
            cx, cyy = px - math.sin(r) * L, py + math.cos(r) * L
            top = (cx + math.sin(r) * h / 2, cyy - math.cos(r) * h / 2)
            cv2.line(img, (int(px), int(py)), (int(top[0]), int(top[1])), (232, 224, 204), max(1, int(round(1.6 * S))), cv2.LINE_AA)
            PK.place(img, spr, PK.affine(cx, cyy, ang - 2, S / spr.ppu, S / spr.ppu, spr.ax, spr.ay))

    def confetti_sprite(self, c, shape, ppu):
        key = ("conf", c, shape, round(ppu, 3))

        def build():
            cv = PK.Canvas((-7, -7, 7, 7), ppu, pad=5)
            if shape:
                cv.poly([[-6, -3], [6, -3], [6, 3], [-6, 3]], col(c))
            else:
                cv.poly([[0, -6], [6, 5], [-6, 5]], col(c))
            return cv.finish(shadow=(0.8, 1.5, 1.5, 0.3), seed=3, base_ppu=self.S)
        return self.sprite(key, build)

    def chip_pos(self, spr):
        S = self.S
        if spr is None:
            return 40 * S, 40 * S
        w = spr.levels[0].shape[1] / spr.ppu * S
        h = spr.levels[0].shape[0] / spr.ppu * S
        m = 22 * S
        top = (52 if self.portrait else 22) * S
        return m + w / 2, top + h / 2

    def caption_sprite(self, words, hl, size, ppu):
        key = ("cap", tuple(words), tuple(hl), round(size, 1), round(ppu, 3))

        def build():
            sp = text_w(" ", size)
            ws = [text_w(w, size) for w in words]
            total = sum(ws) + sp * (len(ws) - 1)
            w, h = total + size * 1.2, size * 1.75
            cv = PK.Canvas((-w / 2, -h / 2, w / 2, h / 2), ppu)
            el = PK.jag(np.array([[-w / 2, -h / 2], [-w / 2, h / 2]]), 1.4, 2.5, len(words))
            er = PK.jag(np.array([[w / 2, h / 2], [w / 2, -h / 2]]), 1.4, 2.5, len(words) + 1)
            cv.poly(np.vstack([el, er]), PK.PAL["cream"])
            x = -total / 2
            for wd, ww, h_ in zip(words, ws, hl):
                cv.text(wd, x + ww / 2, 0, size, col("coral") if h_ else col("ink"), F_STICKER, "mm")
                x += ww + sp
            return cv.finish(seed=len(words) * 5, base_ppu=self.S)
        return self.sprite(key, build)

    def draw_watermark(self, img):
        wm = self.F.get("watermark")
        if not wm:
            return
        S = self.S
        spr = self.tag_sprite(str(wm), 18, col("ink"), col("cream"), "pill", self.hud_ppu, F_SEMI)
        w = spr.levels[0].shape[1] / spr.ppu * S
        h = spr.levels[0].shape[0] / spr.ppu * S
        pos = str(self.F.get("watermark_pos", "top-right"))
        m = 22 * S
        x = self.W - w / 2 - m if "right" in pos else m + w / 2
        y = self.H - h / 2 - m if "bottom" in pos else m + h / 2 + (30 * S if self.portrait else 0)
        PK.place(img, spr, PK.affine(x, y, 0, S / spr.ppu, S / spr.ppu, spr.ax, spr.ay), 0.55)

    # ---------------------------------------------------------------- audio
    def mix_audio(self, out_wav):
        sr = MU.SR
        n = int((self.duration + 0.5) * sr)
        buf = np.zeros(n, np.float32)
        voice = None
        if self.spec.get("audio"):
            ap = self.spec["audio"]
            ap = ap if os.path.isabs(ap) else os.path.join(self.base, ap)
            v = EX.load_audio(ap, sr)
            v = v / (np.max(np.abs(v)) + 1e-9) * 0.89
            voice = np.zeros(n, np.float32)
            m = min(n, len(v))
            voice[:m] = v[:m]
            buf += voice
        mus = self.F.get("music", "auto")
        if isinstance(mus, str) and mus.lower() in ("auto", "ya", "yes", "on", "true"):
            mus = True
        elif isinstance(mus, str) and mus.lower() in ("tidak", "no", "off", "false", "none", ""):
            mus = False
        if mus:
            if isinstance(mus, str):
                mp = mus if os.path.isabs(mus) else os.path.join(self.base, mus)
                bed = EX.load_audio(mp, sr)
                if len(bed) < n:
                    bed = np.tile(bed, n // max(len(bed), 1) + 1)
                bed = bed[:n] / (np.max(np.abs(bed[:n])) + 1e-9) * 0.9
                f = min(n, int(1.5 * sr))
                bed[n - f:] *= np.linspace(1, 0, f)
            else:
                bed = MU.bed(self.duration + 0.3, self.F.get("music_mood", "playful"))
                bed = np.pad(bed, (0, max(0, n - len(bed))))[:n]
            base_db = (-23 if voice is not None else -10) + float(self.F.get("music_volume_db", 0))
            g = np.full(n, 10 ** (base_db / 20), np.float32)
            if voice is not None:
                hop = int(0.02 * sr)
                env = np.sqrt(np.convolve(voice ** 2, np.ones(hop * 5) / (hop * 5), mode="same"))
                env = np.clip(env / (np.percentile(env, 95) + 1e-6), 0, 1)
                g *= (1 - 0.55 * env).astype(np.float32)
            buf += bed * g
        gscale = 10 ** (float(self.F.get("sfx_volume_db", 0)) / 20)
        for (t, name, gain) in self.sfx:
            s = MU.get(name)
            gdb = gain if gain is not None else MU.GAIN_DB.get(name, -20)
            if voice is None:
                gdb += 2
            a = int(t * sr)
            if a >= n:
                continue
            m = min(len(s), n - a)
            buf[a:a + m] += s[:m] * 10 ** (gdb / 20) * gscale
        for (t0, t1, name, gain) in self.loops:
            d = max(0.5, t1 - t0)
            s = MU.LOOPS[name](d)
            g = 10 ** ((gain if gain is not None else MU.GAIN_DB.get(name, -27)) / 20) * gscale
            a = int(t0 * sr)
            m = min(len(s), n - a)
            if m > 0:
                buf[a:a + m] += s[:m] * g
        pk = np.max(np.abs(buf)) + 1e-9
        if pk > 0.97 or voice is None:
            buf = buf / pk * (0.95 if voice is None else 0.97)
        pcm = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())

    # ---------------------------------------------------------------- video
    def nframes(self):
        return int(math.ceil(self.duration * self.fps))

    def render_range(self, f0, f1, path, crf=20, preset="veryfast"):
        cmd = [EX.ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}",
               "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", path]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(f0, f1):
            pr.stdin.write(self.frame(f / self.fps).tobytes())
        pr.stdin.close()
        pr.wait()
        if pr.returncode:
            raise RuntimeError("ffmpeg encode failed")

    def mux(self, parts, out):
        wav = out + ".mix.wav"
        self.mix_audio(wav)
        lst = out + ".parts.txt"
        with open(lst, "w") as f:
            for p in parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        subprocess.run([EX.ffmpeg_bin(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        "-movflags", "+faststart", "-shortest", out], check=True)
        os.remove(lst)
        os.remove(wav)


# ============================================================================= CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("-o", "--out")
    ap.add_argument("--preview")
    ap.add_argument("--part")
    ap.add_argument("--finalize", type=int)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--res", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--transcribe", default=None, help="model dir: treat SPEC arg as an audio file, print transcript")
    ap.add_argument("--lang", default="id", help="Whisper language code (id, en, …) or auto")
    ap.add_argument("--compare", default=None, help="with --transcribe: naskah .txt to compare against")
    a = ap.parse_args()
    if a.transcribe:
        # same procedure as motion-2d-explainer: Whisper "base" (ONNX from sherpa-onnx) run by onnxruntime (whisper_ort.py)
        txt = EX.transcribe(a.spec, a.transcribe, a.lang)
        out = a.out or os.path.splitext(a.spec)[0] + ".transcript.txt"
        open(out, "w", encoding="utf-8").write(txt)
        print(txt)
        print("transcript ->", out, file=sys.stderr)
        if a.compare:
            ref = open(a.compare, encoding="utf-8").read()
            ratio, diffs = EX.compare_texts(txt, ref)
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
    print(f"mode={E.mode}{' (' + E.asr_info + ')' if E.asr_info else ''} duration={E.duration:.2f}s frames={E.nframes()} size={E.W}x{E.H} aspect={F['aspect']} fps={E.fps} "
          f"captions={'on' if F['captions']['show'] else 'off'} title={F['title']} outro={F['outro']} "
          f"music={F.get('music')}({F.get('music_mood')}) sfx={F['sfx']}({F['sfx_volume_db']:+}dB) bg={F.get('background') or spec.get('background', 'night')}")
    if E.caps:
        EX.write_srt(E.caps, stem + ".srt")
    if a.timing or a.check:
        if a.timing:
            for c in E.caps:
                print(f"{c['start']:6.2f}-{c['end']:6.2f}  {' '.join(c['words'])}")
        print("--- beats ---")
        for t, at, kind in E.events:
            print(f"{t:6.2f}s  {kind:8s} at '{at}'")
        ts = [0.0] + [e[0] for e in E.events] + [E.outro_start()]
        gaps = [b - a_ for a_, b in zip(ts, ts[1:]) if b - a_ > 5]
        if gaps:
            print(f"  ! {len(gaps)} gap(s) > 5s without any beat — add a show/label/camera/count so the video never stalls")
        if not spec.get("meta", {}).get("title"):
            print("  ! meta.title missing (hook title + topic chip)")
        print(f"warnings: {len(E.warnings)}")
        return
    if a.preview:
        ts = [float(x) for x in a.preview.split(",")]
        ims = []
        for t in ts:
            im = E.frame(t)
            cv2.putText(im, f"{t:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 80), 2)
            pw = 300 if E.H >= E.W else 520
            ims.append(cv2.resize(im, (pw, E.H * pw // E.W), interpolation=cv2.INTER_AREA))
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
        parts = [f"{stem}.part{i}of{a.finalize}.mp4" for i in range(1, a.finalize + 1)]
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
