"""
palette_check.py - developer tool: measure how distinguishable the colours
in theme.py are. Run it after editing any colour:

    python palette_check.py

For each colour set it prints
    * the smallest pairwise CIEDE2000 distance (dE) and which pair it is
      - roughly: dE < 10 = easy to confuse, 10-20 = distinguishable side by
        side, > 20 = clearly different
    * the same under simulated protanopia / deuteranopia / tritanopia
      (colour-vision deficiency, Machado et al. 2009, full severity)
    * the lowest contrast ratio against the plot and canvas backgrounds
      (WCAG; 3:1 is the minimum for graphics, higher is better)

It needs only NumPy (plus theme.py's own imports).
"""

import itertools

import numpy as np

import theme

# %% Colour maths

# ============================================================
# Colour maths (sRGB <-> CIELAB, CIEDE2000, WCAG contrast)
# ============================================================

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])

# Colour-vision-deficiency matrices (linear RGB), Machado et al. 2009.
CVD = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def hex_to_linear(hex_color):
    """'#rrggbb' -> linear-light RGB array in 0..1."""
    h = hex_color.lstrip("#")
    c = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0

    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_lab(linear_rgb):
    """Linear RGB -> CIELAB (L, a, b)."""
    xyz = (_M @ linear_rgb) / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz),
                 (24389 / 27 * xyz + 16) / 116)

    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]),
                     200 * (f[1] - f[2])])


def lab_of(hex_color, vision=None):
    """CIELAB of a hex colour, optionally as seen with a CVD type."""
    lin = hex_to_linear(hex_color)

    if vision is not None:
        lin = np.clip(CVD[vision] @ lin, 0.0, 1.0)

    return to_lab(lin)


def delta_e_2000(lab1, lab2):
    """CIEDE2000 colour difference between two CIELAB colours."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360

    dL, dC = L2 - L1, C2p - C1p
    dh = h2 - h1

    if C1p * C2p == 0:
        dh = 0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360

    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))

    Lb, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2

    if C1p * C2p == 0:
        hb = h1 + h2
    elif abs(h1 - h2) <= 180:
        hb = (h1 + h2) / 2
    elif h1 + h2 < 360:
        hb = (h1 + h2 + 360) / 2
    else:
        hb = (h1 + h2 - 360) / 2

    T = (1 - 0.17 * np.cos(np.radians(hb - 30))
         + 0.24 * np.cos(np.radians(2 * hb))
         + 0.32 * np.cos(np.radians(3 * hb + 6))
         - 0.20 * np.cos(np.radians(4 * hb - 63)))
    d_theta = 30 * np.exp(-(((hb - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc

    return float(np.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2
                         + Rt * (dC / Sc) * (dH / Sh)))


def contrast_ratio(hex_a, hex_b):
    """WCAG contrast ratio (1 = identical, 21 = black on white)."""
    la = float(hex_to_linear(hex_a) @ _M[1])
    lb = float(hex_to_linear(hex_b) @ _M[1])
    hi, lo = max(la, lb), min(la, lb)

    return (hi + 0.05) / (lo + 0.05)

# %% Reports

# ============================================================
# Reports
# ============================================================

def closest_pair(colors, vision=None):
    """(smallest dE, (colour_a, colour_b)) among `colors`."""
    labs = {c: lab_of(c, vision) for c in colors}

    return min(
        (delta_e_2000(labs[a], labs[b]), (a, b))
        for a, b in itertools.combinations(colors, 2)
    )


def report(name, colors, backgrounds):
    colors = list(colors)
    print(f"\n{name}  ({len(colors)} colours)")

    parts = []

    for vision in (None, "protan", "deutan", "tritan"):
        d, pair = closest_pair(colors, vision)
        label = "normal" if vision is None else vision
        parts.append(f"{label} {d:5.1f}")

        if vision is None:
            worst = pair

    print("  min dE:   " + " | ".join(parts))
    print(f"  closest pair (normal vision): {worst[0]} / {worst[1]}")

    worst_contrast = min(
        (contrast_ratio(c, bg), c, bg) for c in colors for bg in backgrounds
    )
    print(f"  min contrast vs background: {worst_contrast[0]:.1f}:1 "
          f"({worst_contrast[1]} on {worst_contrast[2]})")


def main():
    backgrounds = (theme.PLOT_BG, theme.CANVAS_BG)

    report("NEURON_COLORS", theme.NEURON_COLORS, backgrounds)

    for n in (3, 6):
        report(f"  first {n} neuron colours", theme.NEURON_COLORS[:n],
               backgrounds)

    report("CURRENT_COLORS + threshold line",
           list(theme.CURRENT_COLORS.values()) + [theme.THRESHOLD_COLOR],
           backgrounds)
    report("EDGE_COLORS", theme.EDGE_COLORS.values(), backgrounds)


if __name__ == "__main__":
    main()
