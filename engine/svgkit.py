"""Minimal SVG -> polygon converter for flat-vector illustrations (no cairo needed).

Supports: svg(viewBox) g(transform) path rect circle ellipse polygon polyline line;
fill / stroke / stroke-width / opacity / fill-opacity / style="..."; transforms translate scale rotate matrix;
path commands M L H V C S Q T A Z (abs + rel). Colors: #rgb #rrggbb rgb(), basic names, engine palette names
(resolved by a callback), "none".
"""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

CURVE_STEPS = 14

BASIC = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (232, 92, 104), "green": (100, 215, 135),
         "blue": (52, 140, 225), "yellow": (255, 210, 40), "orange": (240, 130, 70), "purple": (185, 150, 235),
         "pink": (240, 150, 170), "grey": (150, 155, 180), "gray": (150, 155, 180), "brown": (150, 95, 60)}


# ----------------------------------------------------------------------------- transforms
def _mat_mul(a, b):
    return a @ b


def _parse_transform(s):
    M = np.eye(3)
    if not s:
        return M
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", args)]
        T = np.eye(3)
        if name == "translate":
            T[0, 2] = v[0]
            T[1, 2] = v[1] if len(v) > 1 else 0
        elif name == "scale":
            T[0, 0] = v[0]
            T[1, 1] = v[1] if len(v) > 1 else v[0]
        elif name == "rotate":
            a = math.radians(v[0])
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            if len(v) == 3:
                A = np.eye(3); A[0, 2], A[1, 2] = v[1], v[2]
                B = np.eye(3); B[0, 2], B[1, 2] = -v[1], -v[2]
                T = A @ R @ B
            else:
                T = R
        elif name == "matrix" and len(v) == 6:
            T = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif name == "skewX":
            T[0, 1] = math.tan(math.radians(v[0]))
        elif name == "skewY":
            T[1, 0] = math.tan(math.radians(v[0]))
        M = M @ T
    return M


def _apply(M, pts):
    p = np.asarray(pts, float)
    return p @ M[:2, :2].T + M[:2, 2]


# ----------------------------------------------------------------------------- path parsing
_TOK = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")


def _arc(p0, rx, ry, phi, fa, fs, p1):
    """SVG elliptical arc -> points (endpoint parameterization)."""
    x1, y1 = p0
    x2, y2 = p1
    if rx == 0 or ry == 0:
        return [p1]
    rx, ry = abs(rx), abs(ry)
    ph = math.radians(phi)
    c, s = math.cos(ph), math.sin(ph)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = c * dx + s * dy, -s * dx + c * dy
    lam = (x1p ** 2) / (rx ** 2) + (y1p ** 2) / (ry ** 2)
    if lam > 1:
        rx *= math.sqrt(lam)
        ry *= math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    co = math.sqrt(max(0, num / den)) if den else 0
    if fa == fs:
        co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = c * cxp - s * cyp + (x1 + x2) / 2
    cy = s * cxp + c * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a
    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dt > 0:
        dt -= 2 * math.pi
    elif fs and dt < 0:
        dt += 2 * math.pi
    n = max(4, int(abs(dt) / (math.pi / 16)))
    out = []
    for i in range(1, n + 1):
        t = t1 + dt * i / n
        out.append((cx + rx * math.cos(t) * c - ry * math.sin(t) * s, cy + rx * math.cos(t) * s + ry * math.sin(t) * c))
    return out


def parse_path(d):
    toks = [(m.group(1), m.group(2)) for m in _TOK.finditer(d)]
    i = 0
    subs = []          # list of (points, closed)
    cur = []
    pos = (0.0, 0.0)
    start = (0.0, 0.0)
    last_ctrl = None
    cmd = None

    def num():
        nonlocal i
        while i < len(toks) and toks[i][0]:
            break
        v = float(toks[i][1])
        i += 1
        return v

    def has_num():
        return i < len(toks) and toks[i][1] is not None

    while i < len(toks):
        if toks[i][0]:
            cmd = toks[i][0]
            i += 1
        elif cmd is None:
            i += 1
            continue
        rel = cmd.islower()
        C = cmd.upper()
        ox, oy = pos if rel else (0.0, 0.0)
        if C == "Z":
            if cur:
                subs.append((cur, True))
            cur = []
            pos = start
            last_ctrl = None
            continue
        if not has_num():
            continue
        if C == "M":
            x, y = num() + ox, num() + oy
            if cur:
                subs.append((cur, False))
            cur = [(x, y)]
            pos = start = (x, y)
            cmd = "l" if rel else "L"
            last_ctrl = None
        elif C == "L":
            x, y = num() + ox, num() + oy
            cur.append((x, y)); pos = (x, y); last_ctrl = None
        elif C == "H":
            x = num() + (pos[0] if rel else 0)
            cur.append((x, pos[1])); pos = (x, pos[1]); last_ctrl = None
        elif C == "V":
            y = num() + (pos[1] if rel else 0)
            cur.append((pos[0], y)); pos = (pos[0], y); last_ctrl = None
        elif C in "CS":
            if C == "C":
                c1 = (num() + ox, num() + oy)
            else:
                c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl and last_ctrl[2] == "C" else pos
            c2 = (num() + ox, num() + oy)
            p = (num() + ox, num() + oy)
            p0 = pos
            if not cur:
                cur = [p0]
            for k in range(1, CURVE_STEPS + 1):
                t = k / CURVE_STEPS
                mt = 1 - t
                cur.append((mt ** 3 * p0[0] + 3 * mt * mt * t * c1[0] + 3 * mt * t * t * c2[0] + t ** 3 * p[0],
                            mt ** 3 * p0[1] + 3 * mt * mt * t * c1[1] + 3 * mt * t * t * c2[1] + t ** 3 * p[1]))
            pos = p
            last_ctrl = (c2[0], c2[1], "C")
        elif C in "QT":
            if C == "Q":
                q = (num() + ox, num() + oy)
            else:
                q = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl and last_ctrl[2] == "Q" else pos
            p = (num() + ox, num() + oy)
            p0 = pos
            if not cur:
                cur = [p0]
            for k in range(1, CURVE_STEPS + 1):
                t = k / CURVE_STEPS
                mt = 1 - t
                cur.append((mt * mt * p0[0] + 2 * mt * t * q[0] + t * t * p[0], mt * mt * p0[1] + 2 * mt * t * q[1] + t * t * p[1]))
            pos = p
            last_ctrl = (q[0], q[1], "Q")
        elif C == "A":
            rx, ry, phi = num(), num(), num()
            fa, fs = int(num()), int(num())
            p = (num() + ox, num() + oy)
            if not cur:
                cur = [pos]
            cur.extend(_arc(pos, rx, ry, phi, fa, fs, p))
            pos = p
            last_ctrl = None
        else:
            i += 1
    if cur:
        subs.append((cur, False))
    return subs


def _ellipse_pts(cx, cy, rx, ry, n=48):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return list(zip(cx + rx * np.cos(a), cy + ry * np.sin(a)))


def _round_rect(x, y, w, h, rx, ry):
    rx = min(rx, w / 2)
    ry = min(ry, h / 2)
    if rx <= 0 or ry <= 0:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    pts = []
    for cx, cy, a0 in [(x + w - rx, y + ry, -90), (x + w - rx, y + h - ry, 0), (x + rx, y + h - ry, 90), (x + rx, y + ry, 180)]:
        for a in range(a0, a0 + 91, 10):
            r = math.radians(a)
            pts.append((cx + rx * math.cos(r), cy + ry * math.sin(r)))
    return pts


# ----------------------------------------------------------------------------- colors / style
def parse_color(v, resolver=None):
    if v is None:
        return None
    v = str(v).strip()
    if v in ("none", "transparent", ""):
        return "none"
    if v.startswith("#"):
        h = v[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    m = re.match(r"rgba?\(([^)]*)\)", v)
    if m:
        p = [float(x) for x in re.findall(r"[\d.]+", m.group(1))[:3]]
        return tuple(int(x) for x in p)
    if resolver:
        r = resolver(v)
        if r is not None:
            return r
    return BASIC.get(v.lower(), (128, 128, 128))


def _style(el, inherited):
    st = dict(inherited)
    for k in ("fill", "stroke", "stroke-width", "opacity", "fill-opacity", "stroke-opacity", "stroke-linejoin"):
        if el.get(k) is not None:
            st[k] = el.get(k)
    if el.get("style"):
        for part in el.get("style").split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                st[k.strip()] = v.strip()
    if "opacity" in (el.attrib if el is not None else {}) or "opacity" in st:
        pass
    return st


def _f(el, k, d=0.0):
    v = el.get(k)
    if v is None:
        return d
    m = re.match(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", v.strip())
    return float(m.group(0)) if m else d


# ----------------------------------------------------------------------------- main entry
def parse_svg(text, resolver=None):
    """Return dict(viewbox=(x,y,w,h), items=[dict(subs=[(Nx2 array, closed)], fill, stroke, sw, opacity)])."""
    text = text.strip()
    if not text.startswith("<svg"):
        text = f'<svg viewBox="0 0 100 100">{text}</svg>'
    text = re.sub(r'\sxmlns(:\w+)?="[^"]*"', "", text, count=0)
    root = ET.fromstring(text)
    vb = root.get("viewBox") or root.get("viewbox")
    if vb:
        vbx = [float(x) for x in re.split(r"[\s,]+", vb.strip())]
    else:
        vbx = [0, 0, _f(root, "width", 100), _f(root, "height", 100)]
    items = []

    def walk(el, M, st):
        tag = el.tag.split("}")[-1]
        st = _style(el, st)
        M = M @ _parse_transform(el.get("transform"))
        subs = None
        if tag in ("svg", "g"):
            for ch in el:
                walk(ch, M, st)
            return
        if tag == "path":
            subs = parse_path(el.get("d", ""))
        elif tag == "rect":
            x, y, w, h = _f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height")
            rx = _f(el, "rx", _f(el, "ry", 0))
            ry = _f(el, "ry", rx)
            subs = [(_round_rect(x, y, w, h, rx, ry), True)]
        elif tag == "circle":
            r = _f(el, "r")
            subs = [(_ellipse_pts(_f(el, "cx"), _f(el, "cy"), r, r), True)]
        elif tag == "ellipse":
            subs = [(_ellipse_pts(_f(el, "cx"), _f(el, "cy"), _f(el, "rx"), _f(el, "ry")), True)]
        elif tag in ("polygon", "polyline"):
            v = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", el.get("points", ""))]
            subs = [(list(zip(v[0::2], v[1::2])), tag == "polygon")]
        elif tag == "line":
            subs = [([(_f(el, "x1"), _f(el, "y1")), (_f(el, "x2"), _f(el, "y2"))], False)]
        if not subs:
            return
        fill = parse_color(st.get("fill", "#000000" if tag not in ("line", "polyline") else "none"), resolver)
        stroke = parse_color(st.get("stroke", "none"), resolver)
        sw = float(re.sub(r"[^\d.]", "", str(st.get("stroke-width", "1"))) or 1)
        scale = math.sqrt(abs(np.linalg.det(M[:2, :2]))) or 1
        op = float(st.get("opacity", 1)) if "opacity" in st else 1.0
        items.append({
            "subs": [(_apply(M, p), c) for p, c in subs if len(p) >= 2],
            "fill": None if fill == "none" else fill,
            "stroke": None if stroke == "none" else stroke,
            "sw": sw * scale,
            "opacity": op * float(st.get("fill-opacity", 1)),
        })

    walk(root, np.eye(3), {})
    return {"viewbox": tuple(vbx), "items": items}


# ----------------------------------------------------------------------------- built-in props
# Flat house style: base fill + one shade + one highlight, 3px "outline" stroke (theme-aware colour token).
O = 'stroke="outline" stroke-width="3" stroke-linejoin="round"'
PROPS = {
    "lightbulb": f'''<svg viewBox="0 0 100 100"><path d="M50 8 C28 8 18 26 22 42 C25 54 36 60 37 72 L63 72 C64 60 75 54 78 42 C82 26 72 8 50 8 Z" fill="#ffd84a" {O}/>
<path d="M50 16 C38 16 30 25 30 36" fill="none" stroke="#fff3b0" stroke-width="5" stroke-linecap="round"/>
<rect x="37" y="72" width="26" height="16" rx="3" fill="#9aa3c2" {O}/><rect x="42" y="88" width="16" height="6" rx="3" fill="#6a7090" {O}/></svg>''',
    "heart": f'<svg viewBox="0 0 100 100"><path d="M50 88 C20 66 8 50 8 32 C8 18 19 9 31 9 C40 9 46 14 50 21 C54 14 60 9 69 9 C81 9 92 18 92 32 C92 50 80 66 50 88 Z" fill="#ef5a6f" {O}/><path d="M24 24 C28 19 34 18 38 20" fill="none" stroke="#ffb3be" stroke-width="5" stroke-linecap="round"/></svg>',
    "book": f'<svg viewBox="0 0 100 100"><path d="M10 22 L48 28 L48 86 L10 80 Z" fill="#4f8ef0" {O}/><path d="M90 22 L52 28 L52 86 L90 80 Z" fill="#3a74d0" {O}/><path d="M16 30 L42 34 M16 40 L42 44 M16 50 L42 54" stroke="#cfe0ff" stroke-width="3" stroke-linecap="round"/><rect x="47" y="26" width="6" height="62" rx="2" fill="#2a4f9a" {O}/></svg>',
    "phone": f'<svg viewBox="0 0 100 100"><rect x="28" y="6" width="44" height="88" rx="8" fill="#2c3350" {O}/><rect x="33" y="15" width="34" height="64" rx="3" fill="#7fc4ff"/><circle cx="50" cy="86" r="3.5" fill="#9aa3c2"/><rect x="43" y="10" width="14" height="2.5" rx="1" fill="#9aa3c2"/></svg>',
    "coin": f'<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="#f2b632" {O}/><circle cx="50" cy="50" r="30" fill="#ffd25a" stroke="#d99a1e" stroke-width="3"/><path d="M50 32 L50 68 M58 39 C54 35 42 35 42 43 C42 51 58 49 58 57 C58 65 46 65 42 61" fill="none" stroke="#b77d10" stroke-width="5" stroke-linecap="round"/></svg>',
    "gear": f'<svg viewBox="0 0 100 100"><path d="M44 6 L56 6 L58 18 L68 22 L78 14 L86 22 L78 32 L82 42 L94 44 L94 56 L82 58 L78 68 L86 78 L78 86 L68 78 L58 82 L56 94 L44 94 L42 82 L32 78 L22 86 L14 78 L22 68 L18 58 L6 56 L6 44 L18 42 L22 32 L14 22 L22 14 L32 22 L42 18 Z" fill="#9aa3c2" {O}/><circle cx="50" cy="50" r="15" fill="#5b6280" {O}/></svg>',
    "rocket": f'<svg viewBox="0 0 100 100"><path d="M50 6 C66 18 70 40 66 66 L34 66 C30 40 34 18 50 6 Z" fill="#f2f4fb" {O}/><path d="M34 52 L18 72 L34 70 Z M66 52 L82 72 L66 70 Z" fill="#ef5a6f" {O}/><circle cx="50" cy="36" r="8" fill="#7fc4ff" {O}/><path d="M40 68 L50 92 L60 68 Z" fill="#ffb02e" stroke="#ff7b1c" stroke-width="3" stroke-linejoin="round"/></svg>',
    "leaf": f'<svg viewBox="0 0 100 100"><path d="M14 86 C14 40 44 12 90 10 C88 56 62 86 14 86 Z" fill="#5cc27a" {O}/><path d="M14 86 C34 62 54 42 78 22" fill="none" stroke="#2f8a4c" stroke-width="4" stroke-linecap="round"/></svg>',
    "drop": f'<svg viewBox="0 0 100 100"><path d="M50 6 C62 28 80 44 80 62 C80 80 66 92 50 92 C34 92 20 80 20 62 C20 44 38 28 50 6 Z" fill="#4fa8f0" {O}/><path d="M34 62 C34 72 40 78 46 80" fill="none" stroke="#c7e6ff" stroke-width="5" stroke-linecap="round"/></svg>',
    "star": f'<svg viewBox="0 0 100 100"><path d="M50 6 L62 36 L94 38 L69 58 L78 90 L50 72 L22 90 L31 58 L6 38 L38 36 Z" fill="#ffd25a" {O}/></svg>',
    "clock": f'<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="42" fill="#f2f4fb" {O}/><path d="M50 50 L50 24 M50 50 L68 60" stroke="#2c3350" stroke-width="5" stroke-linecap="round"/><circle cx="50" cy="50" r="4" fill="#ef5a6f"/></svg>',
    "cup": f'<svg viewBox="0 0 100 100"><path d="M18 34 L72 34 L66 86 L24 86 Z" fill="#f2f4fb" {O}/><path d="M72 44 C90 44 90 70 70 70" fill="none" stroke="outline" stroke-width="6"/><path d="M22 40 L68 40 L67 48 L23 48 Z" fill="#8a5a3c"/><path d="M36 26 C32 20 40 16 36 10 M50 26 C46 20 54 16 50 10" fill="none" stroke="#c9d0e6" stroke-width="3" stroke-linecap="round"/></svg>',
    "laptop": f'<svg viewBox="0 0 100 100"><rect x="18" y="20" width="64" height="44" rx="4" fill="#2c3350" {O}/><rect x="23" y="25" width="54" height="34" rx="2" fill="#7fc4ff"/><path d="M8 68 L92 68 L86 80 L14 80 Z" fill="#9aa3c2" {O}/></svg>',
    "cell": f'<svg viewBox="0 0 100 100"><path d="M50 8 C76 8 92 26 92 50 C92 74 74 92 50 92 C24 92 8 76 8 50 C8 24 26 8 50 8 Z" fill="#f7b0c4" {O}/><circle cx="52" cy="48" r="16" fill="#b05aa8" {O}/><circle cx="48" cy="44" r="5" fill="#e7a3d8"/><ellipse cx="28" cy="66" rx="7" ry="4" fill="#5cc27a"/><ellipse cx="72" cy="28" rx="7" ry="4" fill="#5cc27a"/><circle cx="74" cy="68" r="4" fill="#ffd25a"/></svg>',
    "magnifier": f'<svg viewBox="0 0 100 100"><circle cx="42" cy="42" r="28" fill="#c7e6ff" stroke="outline" stroke-width="6"/><path d="M62 62 L88 88" stroke="#8a5a3c" stroke-width="12" stroke-linecap="round"/><path d="M28 34 C30 26 36 22 42 22" fill="none" stroke="#ffffff" stroke-width="5" stroke-linecap="round"/></svg>',
    "battery": f'<svg viewBox="0 0 100 100"><rect x="10" y="28" width="72" height="44" rx="6" fill="#f2f4fb" {O}/><rect x="82" y="40" width="8" height="20" rx="2" fill="#9aa3c2" {O}/><rect x="17" y="35" width="40" height="30" rx="3" fill="#5cc27a"/></svg>',
    "house2": f'<svg viewBox="0 0 100 100"><path d="M12 48 L50 14 L88 48" fill="#c0504d" {O}/><rect x="20" y="46" width="60" height="44" fill="#f2c08a" {O}/><rect x="30" y="56" width="16" height="14" fill="#7fc4ff" {O}/><rect x="56" y="60" width="14" height="30" fill="#8a5a3c" {O}/></svg>',
    "tree2": f'<svg viewBox="0 0 100 100"><rect x="44" y="60" width="12" height="34" fill="#8a5a3c" {O}/><circle cx="50" cy="40" r="30" fill="#4caf6a" {O}/><circle cx="40" cy="30" r="9" fill="#7fd396"/></svg>',
    "sun2": f'<svg viewBox="0 0 100 100"><g stroke="#ffb02e" stroke-width="6" stroke-linecap="round"><path d="M50 4 L50 16 M50 84 L50 96 M4 50 L16 50 M84 50 L96 50 M17 17 L26 26 M74 74 L83 83 M17 83 L26 74 M74 26 L83 17"/></g><circle cx="50" cy="50" r="26" fill="#ffd25a" {O}/></svg>',
    "chart": f'<svg viewBox="0 0 100 100"><rect x="8" y="8" width="84" height="84" rx="8" fill="#f2f4fb" {O}/><rect x="20" y="56" width="12" height="26" rx="2" fill="#4fa8f0"/><rect x="40" y="40" width="12" height="42" rx="2" fill="#5cc27a"/><rect x="60" y="26" width="12" height="56" rx="2" fill="#ffb02e"/></svg>',
    "person2": f'<svg viewBox="0 0 100 100"><circle cx="50" cy="26" r="16" fill="#f2c4a0" {O}/><path d="M22 94 C22 66 34 50 50 50 C66 50 78 66 78 94 Z" fill="#4f8ef0" {O}/></svg>',
    "car": f'<svg viewBox="0 0 100 100"><path d="M8 66 L8 52 L22 48 L32 32 L66 32 L78 48 L92 52 L92 66 Z" fill="#ef5a6f" {O}/><path d="M36 38 L48 38 L48 48 L28 48 Z M54 38 L64 38 L72 48 L54 48 Z" fill="#c7e6ff"/><circle cx="28" cy="68" r="10" fill="#2c3350" {O}/><circle cx="72" cy="68" r="10" fill="#2c3350" {O}/></svg>',
    "atom": f'<svg viewBox="0 0 100 100"><g fill="none" stroke="#7fc4ff" stroke-width="4"><ellipse cx="50" cy="50" rx="42" ry="15"/><ellipse cx="50" cy="50" rx="42" ry="15" transform="rotate(60 50 50)"/><ellipse cx="50" cy="50" rx="42" ry="15" transform="rotate(120 50 50)"/></g><circle cx="50" cy="50" r="8" fill="#ef5a6f" {O}/></svg>',
    "food": f'<svg viewBox="0 0 100 100"><ellipse cx="50" cy="66" rx="42" ry="16" fill="#f2f4fb" {O}/><path d="M22 60 C24 40 38 30 50 30 C62 30 76 40 78 60 Z" fill="#ffb02e" {O}/><circle cx="40" cy="46" r="5" fill="#5cc27a"/><circle cx="58" cy="44" r="5" fill="#ef5a6f"/></svg>',
}
