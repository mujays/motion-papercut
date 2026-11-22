"""PAPERKIT — paper-cutout sprite toolkit for papercut.py.

Everything on screen is a piece of cut paper: a flat colour shape with a subtle fibre texture, a slightly lighter
cut edge and a soft drop shadow. Pieces are rendered once into RGBA sprites (with mip levels) and then placed
every frame with an affine transform (camera + object motion), like a cut-out puppet animation.
"""
import math

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

SH = 4
SHF = 1 << SH

# ----------------------------------------------------------------------------- palette (paper colours)
PAL = {
    "coral": (232, 119, 95), "yellow": (242, 197, 82), "teal": (116, 198, 180), "pink": (238, 154, 168),
    "cream": (244, 238, 222), "paper": (244, 238, 222), "ink": (59, 58, 92), "navy": (59, 58, 92),
    "red": (214, 84, 72), "green": (94, 170, 106), "blue": (128, 186, 222), "lightblue": (168, 212, 230),
    "lavender": (125, 118, 174), "purple": (120, 98, 168), "slate": (90, 122, 148), "forest": (47, 107, 94),
    "moss": (36, 82, 74), "white": (250, 247, 240), "orange": (238, 150, 74), "brown": (150, 104, 72),
    "grey": (160, 158, 172), "gray": (160, 158, 172), "sky": (150, 200, 232), "gold": (236, 182, 64),
    "mint": (170, 222, 196), "peach": (246, 196, 164), "kraft": (206, 170, 124), "dark": (40, 36, 62),
    "rose": (226, 98, 120), "sand": (226, 200, 150),
}
STICKER_CYCLE = ["coral", "yellow", "teal"]


def col(c, default=(244, 238, 222)):
    if c is None:
        return default
    if isinstance(c, (list, tuple)):
        return tuple(int(v) for v in c[:3])
    c = str(c).strip()
    if c.startswith("#") and len(c) == 7:
        return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
    if c.startswith("#") and len(c) == 4:
        return tuple(int(ch * 2, 16) for ch in c[1:])
    return PAL.get(c.lower(), default)


def shade(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ----------------------------------------------------------------------------- paper fibre texture
_TEX = None
_TEXF = None


def texture_fine():
    """same family as texture() but without the big blotches: for small paper pieces (letters, tags, props)."""
    global _TEXF
    if _TEXF is None:
        texture()
    return _TEXF


def texture():
    """768x768 float32 paper texture, zero mean / unit std: fibres + fine grain + low-frequency mottling."""
    global _TEX
    if _TEX is None:
        n = 768
        rng = np.random.default_rng(11)

        def nz(sig):
            pd = int(sig * 4) + 2
            a = np.pad(rng.normal(0, 1, (n, n)).astype(np.float32), pd, mode="wrap")
            a = cv2.GaussianBlur(a, (0, 0), sig)[pd:pd + n, pd:pd + n]   # seamless tile
            return a / (a.std() + 1e-6)
        fib = np.zeros((n, n), np.float32)
        for _ in range(1400):
            x, y = rng.uniform(0, n, 2)
            L = rng.uniform(5, 26)
            a = rng.uniform(0, math.pi)
            bend = rng.uniform(-0.4, 0.4)
            pts = []
            for k in range(6):
                q = k / 5
                aa = a + bend * q
                pts.append((x + math.cos(aa) * L * q, y + math.sin(aa) * L * q))
            cv2.polylines(fib, [np.int32(pts).reshape(-1, 1, 2)], False, float(rng.choice([-1.0, 1.0])), 1, cv2.LINE_AA)
        fib = cv2.GaussianBlur(fib, (0, 0), 0.6)
        fib /= fib.std() + 1e-6
        f07, f5, f28 = nz(0.7), nz(5), nz(28)
        t = 0.45 * f07 + 0.55 * f5 + 0.7 * f28 + 0.55 * fib
        _TEX = (t / (t.std() + 1e-6)).astype(np.float32)
        tf = 0.5 * f07 + 0.35 * f5 + 0.15 * f28 + 0.6 * fib
        global _TEXF
        _TEXF = (tf / (tf.std() + 1e-6)).astype(np.float32)
    return _TEX


def tex_patch(h, w, scale=1.0, seed=0, fine=False):
    """texture patch of size (h,w); `scale` = texture pixels per output pixel (<1 enlarges the grain)."""
    T = texture_fine() if fine else texture()
    n = T.shape[0]
    sh, sw = max(2, int(h * scale)), max(2, int(w * scale))
    rng = np.random.default_rng(seed)
    oy, ox = int(rng.integers(0, n)), int(rng.integers(0, n))
    reps_y, reps_x = (oy + sh) // n + 1, (ox + sw) // n + 1
    if reps_y > 1 or reps_x > 1:
        big = np.tile(T, (reps_y, reps_x))
    else:
        big = T
    p = big[oy:oy + sh, ox:ox + sw]
    if (sh, sw) != (h, w):
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    return p


# ----------------------------------------------------------------------------- fonts / text masks
_fonts = {}


def font(path, size):
    k = (path, int(size))
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(path, max(4, int(size)))
    return _fonts[k]


# ----------------------------------------------------------------------------- sprite
class Sprite:
    """premultiplied RGBA uint8 + mip levels. (ax, ay) = pixel of the object origin; ppu = pixels per world unit."""
    __slots__ = ("levels", "ax", "ay", "ppu")

    def __init__(self, rgba_premul, ax, ay, ppu):
        self.levels = [rgba_premul]
        self.ax, self.ay, self.ppu = ax, ay, ppu
        im = rgba_premul
        while min(im.shape[:2]) > 48 and len(self.levels) < 4:
            im = cv2.resize(im, (max(1, im.shape[1] // 2), max(1, im.shape[0] // 2)), interpolation=cv2.INTER_AREA)
            self.levels.append(im)

    def pick(self, need):
        """level whose pixel density is closest above `need` (pixels per world unit on screen)."""
        best = 0
        for i in range(len(self.levels)):
            if self.ppu / (2 ** i) >= need * 0.92:
                best = i
        return best


def place(frame, spr, M, alpha=1.0):
    """composite sprite onto float32 frame (H,W,3). M = 2x3 affine mapping sprite level-0 pixels -> screen."""
    if spr is None or alpha <= 0.004:
        return
    H, W = frame.shape[:2]
    sc = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    lv = spr.pick(sc * spr.ppu)
    img = spr.levels[lv]
    f = 2 ** lv
    if lv:
        # account for the reduced level resolution (exact ratio of sizes)
        fx = spr.levels[0].shape[1] / img.shape[1]
        fy = spr.levels[0].shape[0] / img.shape[0]
        M = M @ np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]], np.float64)
    h, w = img.shape[:2]
    M = M[:2] if M.shape[0] == 3 else M
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 1
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return
    Mr = M.copy()
    Mr[0, 2] -= x0
    Mr[1, 2] -= y0
    roi = cv2.warpAffine(img, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = roi[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    if a.max() <= 0:
        return
    dst = frame[y0:y1, x0:x1]
    dst *= (1 - a)
    dst += roi[:, :, :3].astype(np.float32) * alpha


def affine(sx, sy, rot_deg, scale_x, scale_y, ax, ay):
    """2x3 (as 3x3) matrix: translate(-ax,-ay) -> scale -> rotate -> translate(sx,sy)."""
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    M = np.array([[c * scale_x, -s * scale_y, 0], [s * scale_x, c * scale_y, 0], [0, 0, 1]], np.float64)
    M[0, 2] = sx - (M[0, 0] * ax + M[0, 1] * ay)
    M[1, 2] = sy - (M[1, 0] * ax + M[1, 1] * ay)
    return M


# ----------------------------------------------------------------------------- canvas (build a paper piece)
class Canvas:
    """Draw paper layers in world units inside bbox (x0,y0,x1,y1) at `ppu` px/unit. Origin of the object = world (0,0)."""

    def __init__(self, bbox, ppu, pad=14, max_px=5200):
        x0, y0, x1, y1 = bbox
        span = max(x1 - x0, y1 - y0) + 2 * pad
        if span * ppu > max_px:
            ppu = max_px / span
        self.ppu = ppu
        self.ox, self.oy = x0 - pad, y0 - pad
        self.W = int(math.ceil((x1 - x0 + 2 * pad) * ppu))
        self.H = int(math.ceil((y1 - y0 + 2 * pad) * ppu))
        self.rgb = np.zeros((self.H, self.W, 3), np.float32)
        self.a = np.zeros((self.H, self.W), np.float32)

    # world -> pixel
    def P(self, pts):
        a = np.asarray(pts, np.float64).reshape(-1, 2)
        return np.stack([(a[:, 0] - self.ox) * self.ppu, (a[:, 1] - self.oy) * self.ppu], 1)

    def _ip(self, pts):
        return np.round(self.P(pts) * SHF).astype(np.int32).reshape(-1, 1, 2)

    def new_mask(self):
        return np.zeros((self.H, self.W), np.uint8)

    def comp(self, mask, color, lift=0.0, alpha=1.0, texture_amt=None):
        """composite a coloured layer through `mask` (uint8 0..255). lift>0 drops a small shadow on the layers below."""
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return
        k = lift * self.ppu if lift > 0 else 0
        mg = int(k * 4 + 3)
        x0, y0 = max(0, x - mg), max(0, y - mg)
        x1, y1 = min(self.W, x + w + mg), min(self.H, y + h + mg)
        m = mask[y0:y1, x0:x1].astype(np.float32) * (alpha / 255.0)
        A = self.a[y0:y1, x0:x1]
        R = self.rgb[y0:y1, x0:x1]
        if lift > 0:
            sm = cv2.GaussianBlur(m, (0, 0), max(0.6, k * 0.9))
            Mt = np.float32([[1, 0, k * 0.5], [0, 1, k * 0.9]])
            sm = cv2.warpAffine(sm, Mt, (x1 - x0, y1 - y0))
            R *= (1 - (sm * 0.38 * A)[:, :, None])
        c = np.array(color, np.float32)[None, None, :]
        na = m + A * (1 - m)
        safe = np.maximum(na, 1e-6)[:, :, None]
        self.rgb[y0:y1, x0:x1] = (c * m[:, :, None] + R * (A * (1 - m))[:, :, None]) / safe
        self.a[y0:y1, x0:x1] = na

    # --- shapes (all world units)
    def poly(self, pts, color, lift=0.0, alpha=1.0):
        m = self.new_mask()
        if len(pts) >= 3:
            cv2.fillPoly(m, [self._ip(pts)], 255, cv2.LINE_AA, SH)
        self.comp(m, color, lift, alpha)
        return m

    def polys(self, list_pts, color, lift=0.0, alpha=1.0):
        m = self.new_mask()
        ps = [self._ip(p) for p in list_pts if len(p) >= 3]
        if ps:
            cv2.fillPoly(m, ps, 255, cv2.LINE_AA, SH)
        self.comp(m, color, lift, alpha)
        return m

    def stroke(self, pts, width, color, lift=0.0, closed=False, alpha=1.0, round_caps=True):
        m = self.new_mask()
        w = max(1, int(round(width * self.ppu)))
        cv2.polylines(m, [self._ip(pts)], closed, 255, w, cv2.LINE_AA, SH)
        if round_caps and not closed and len(pts) >= 2:
            for p in (pts[0], pts[-1]):
                q = self.P([p])[0]
                cv2.circle(m, (int(q[0] * SHF), int(q[1] * SHF)), int(w / 2 * SHF), 255, -1, cv2.LINE_AA, SH)
        self.comp(m, color, lift, alpha)
        return m

    def circle(self, cx, cy, r, color, lift=0.0, alpha=1.0):
        return self.poly(ellipse_pts(cx, cy, r, r), color, lift, alpha)

    def ellipse(self, cx, cy, rx, ry, color, lift=0.0, rot=0.0, alpha=1.0):
        return self.poly(ellipse_pts(cx, cy, rx, ry, rot=rot), color, lift, alpha)

    def rrect(self, x0, y0, x1, y1, r, color, lift=0.0, alpha=1.0):
        return self.poly(rrect_pts(x0, y0, x1, y1, r), color, lift, alpha)

    def text(self, txt, x, y, size, color, fontpath, anchor="mm", outline=0, outline_color=None, lift=0.0):
        """text in world units; outline (world units) draws a cream sticker border first."""
        f = font(fontpath, size * self.ppu)
        px, py = self.P([(x, y)])[0]
        if outline:
            im = Image.new("L", (self.W, self.H), 0)
            ImageDraw.Draw(im).text((px, py), txt, font=f, fill=255, anchor=anchor,
                                    stroke_width=int(round(outline * self.ppu)), stroke_fill=255)
            self.comp(np.array(im), outline_color or PAL["cream"], lift)
        im = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(im).text((px, py), txt, font=f, fill=255, anchor=anchor)
        self.comp(np.array(im), color, 0.0 if outline else lift)

    # --- finish into a sprite
    def finish(self, shadow=(2.5, 4.5, 4.0, 0.42), tex=0.055, edge=0.10, seed=0, base_ppu=1.0, shadow_color=(22, 14, 40), fine=True):
        """shadow=(dx, dy, blur, opacity) in world units. Returns Sprite (premultiplied)."""
        a = self.a
        rgb = self.rgb
        if tex:
            T = tex_patch(self.H, self.W, scale=base_ppu / self.ppu * 1.4, seed=seed, fine=fine)
            rgb = rgb * (1 + tex * T[:, :, None])
        if edge:
            k = max(1, int(round(1.6 * self.ppu)))
            er = cv2.erode(a, np.ones((2 * k + 1, 2 * k + 1), np.float32))
            e = np.clip(a - er, 0, 1)[:, :, None]
            rgb = rgb + (255 - rgb) * e * edge
        rgb = np.clip(rgb, 0, 255)
        out_rgb = rgb * a[:, :, None]          # premultiplied
        out_a = a.copy()
        if shadow and shadow[3] > 0:
            dx, dy, bl, op = shadow
            sm = cv2.GaussianBlur(a, (0, 0), max(0.8, bl * self.ppu * 0.6))
            Mt = np.float32([[1, 0, dx * self.ppu], [0, 1, dy * self.ppu]])
            sm = cv2.warpAffine(sm, Mt, (self.W, self.H)) * op
            sc = np.array(shadow_color, np.float32)[None, None, :]
            # paper over shadow
            out_rgb = out_rgb + sc * (sm * (1 - a))[:, :, None]
            out_a = a + sm * (1 - a)
        rgba = np.dstack([out_rgb, out_a * 255]).clip(0, 255).astype(np.uint8)
        # crop transparent margins
        ys, xs = np.where(rgba[:, :, 3] > 2)
        if len(xs) == 0:
            return Sprite(np.zeros((2, 2, 4), np.uint8), 0, 0, self.ppu)
        cx0, cx1, cy0, cy1 = max(xs.min() - 2, 0), min(xs.max() + 3, self.W), max(ys.min() - 2, 0), min(ys.max() + 3, self.H)
        rgba = np.ascontiguousarray(rgba[cy0:cy1, cx0:cx1])
        ax = (0 - self.ox) * self.ppu - cx0
        ay = (0 - self.oy) * self.ppu - cy0
        return Sprite(rgba, ax, ay, self.ppu)


# ----------------------------------------------------------------------------- geometry helpers
def ellipse_pts(cx, cy, rx, ry, n=64, rot=0.0, a0=0.0, a1=2 * math.pi):
    t = np.linspace(a0, a1, n)
    x, y = rx * np.cos(t), ry * np.sin(t)
    if rot:
        r = math.radians(rot)
        x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
    return np.stack([cx + x, cy + y], 1)


def rrect_pts(x0, y0, x1, y1, r, n=8):
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    if r <= 0.01:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def catmull(pts, closed=False, n=10):
    P = [np.array(p, float) for p in pts]
    if len(P) < 3:
        return np.array(P)
    if closed:
        P = [P[-1]] + P + [P[0], P[1]]
    else:
        P = [P[0]] + P + [P[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def bezier(a, b, bend=0.2, n=40):
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    nrm = np.array([d[1], -d[0]])
    c = (a + b) / 2 + nrm * bend
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def jag(pts, amp, step, seed):
    """add torn-paper jitter perpendicular to a polyline (world units)."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, float)
    out = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = np.linalg.norm(b - a)
        k = max(1, int(L / step))
        d = (b - a) / max(L, 1e-6)
        nrm = np.array([-d[1], d[0]])
        for j in range(k):
            q = a + (b - a) * j / k
            out.append(q + nrm * rng.uniform(-amp, amp))
    out.append(pts[-1])
    return np.array(out)
