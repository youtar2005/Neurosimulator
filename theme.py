"""
theme.py - dark "midnight indigo" theme shared by the editor and the
simulation window.

Backgrounds are a desaturated blue-black (easy on the eyes, no pure black
or pure white); accents are a blue -> purple pair used for selection,
hover and pressed states. Colours of data (neurons, currents, edges) are
soft pastels so they stay readable without glaring.

Usage:
    theme.apply_theme(app)          # once, right after creating QApplication
    theme.neuron_color(i)           # i-th neuron colour (QColor)
    theme.series_color(i)           # generic plot series colour (QColor)
"""

from string import Template

import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets

# %% Colours

# ============================================================
# Colours
# ============================================================

# Surfaces
BG = "#12121c"              # window background
PANEL = "#181826"           # toolbars, side panels
SURFACE = "#1f2033"         # inputs, buttons
HOVER = "#2a2b45"
PRESSED = "#34355a"
BORDER = "#2e2f4a"
BORDER_LIGHT = "#45476d"

# Text
TEXT = "#d6d8f0"
MUTED = "#8b8eb5"

# Accents (blue -> purple)
BLUE = "#6c8cff"
BLUE_DIM = "#3f5bd6"
PURPLE = "#a37bff"
PURPLE_DIM = "#7a4fe0"
PURPLE_LIGHT = "#c3a8ff"
HIGHLIGHT = "#5b4fd1"       # text-selection / list highlight

# Status
ERROR = "#ff6b8a"

# Editor canvas
CANVAS_BG = "#0f0f19"
NEURON_OUTLINE = "#2b2c48"
SELECTION = "#f4f4ff"       # selected neuron outline / selected edge
EDGE_LABEL = "#9a9dc4"

# Plots
PLOT_BG = "#0e0e18"
PLOT_FG = "#9a9dc4"

# Categorical data colours. Chosen with palette_check.py (CIEDE2000):
#   * every colour has >= 5.5:1 contrast against the plot/canvas background
#   * the first 6 are >= 23 apart in normal vision and >= 10 apart under
#     red-green colour blindness; all 10 are >= 17 apart (normal vision)
#   * the order is deliberate: any prefix (first 3, first 6, ...) is the most
#     distinguishable set of that size, so small networks get the best colours.
NEURON_COLORS = (
    "#5c8dff",   # blue
    "#e6632d",   # vermilion
    "#d7a9fe",   # lavender
    "#c9d53f",   # yellow-green
    "#2dc7c1",   # teal
    "#cb6f8d",   # rose
    "#12e992",   # mint
    "#6ca062",   # moss
    "#f924d2",   # magenta
    "#e19d48",   # amber
)

# Connection types. Also distinguished by shape (arrowhead / dashed line).
EDGE_COLORS = {
    "exc": "#55d98d",    # green
    "inh": "#ff5a6e",    # red
    "gap": "#ffb43b",    # amber
    "veto": "#b78cff",   # purple
}

# Membrane-current traces: the six most distinguishable palette colours.
CURRENT_COLORS = {
    "Na": "#e6632d",     # vermilion
    "K": "#5c8dff",      # blue
    "Cl": "#2dc7c1",     # teal
    "pump": "#d7a9fe",   # lavender
    "syn": "#c9d53f",    # yellow-green
    "stim": "#cb6f8d",   # rose
}

THRESHOLD_COLOR = "#f4f4ff"   # dashed near-white: distinct from every trace


def neuron_color(i):
    return QtGui.QColor(NEURON_COLORS[i % len(NEURON_COLORS)])


def series_color(i):
    return QtGui.QColor(NEURON_COLORS[i % len(NEURON_COLORS)])


def _luminance(color):
    """WCAG relative luminance (0 = black, 1 = white)."""
    def lin(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    return (0.2126 * lin(color.red()) + 0.7152 * lin(color.green())
            + 0.0722 * lin(color.blue()))


def contrast_text(color):
    """Black or white text, whichever is more readable on `color`."""
    L = _luminance(QtGui.QColor(color))
    black_ratio = (L + 0.05) / 0.05
    white_ratio = 1.05 / (L + 0.05)

    return QtGui.QColor("#0b0b14" if black_ratio >= white_ratio else "#ffffff")

# %% Palette and stylesheet

# ============================================================
# Palette and stylesheet
# ============================================================

def _build_palette():
    c = QtGui.QColor
    P = QtGui.QPalette
    p = P()

    p.setColor(P.Window, c(BG))
    p.setColor(P.WindowText, c(TEXT))
    p.setColor(P.Base, c(SURFACE))
    p.setColor(P.AlternateBase, c(PANEL))
    p.setColor(P.ToolTipBase, c(SURFACE))
    p.setColor(P.ToolTipText, c(TEXT))
    p.setColor(P.Text, c(TEXT))
    p.setColor(P.Button, c(SURFACE))
    p.setColor(P.ButtonText, c(TEXT))
    p.setColor(P.BrightText, c("#ffffff"))
    p.setColor(P.Link, c(BLUE))
    p.setColor(P.Highlight, c(HIGHLIGHT))
    p.setColor(P.HighlightedText, c("#ffffff"))

    if hasattr(P, "PlaceholderText"):
        p.setColor(P.PlaceholderText, c(MUTED))

    for role in (P.WindowText, P.Text, P.ButtonText):
        p.setColor(P.Disabled, role, c(MUTED))

    return p


_STYLESHEET = Template("""
QToolTip {
    background-color: $surface; color: $text;
    border: 1px solid $purple; padding: 4px;
}

QToolBar {
    background: $panel; border: none;
    border-bottom: 1px solid $border;
    spacing: 4px; padding: 4px;
}
QToolBar::separator { background: $border; width: 1px; margin: 4px 6px; }

QToolButton {
    background: transparent; color: $text;
    border: 1px solid transparent; border-radius: 6px;
    padding: 5px 10px;
}
QToolButton:hover { background: $hover; border-color: $border; }
QToolButton:pressed { background: $pressed; }
QToolButton:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 $blue_dim, stop:1 $purple_dim);
    border: 1px solid $purple; color: #ffffff;
}

QPushButton {
    background: $surface; color: $text;
    border: 1px solid $border; border-radius: 6px;
    padding: 5px 10px;
}
QPushButton:hover { background: $hover; border-color: $purple; }
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 $blue_dim, stop:1 $purple_dim);
    border-color: $purple_light; color: #ffffff;
}

QGroupBox {
    border: 1px solid $border; border-radius: 8px;
    margin-top: 12px; padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: $purple_light;
}

QStatusBar {
    background: $panel; color: $muted;
    border-top: 1px solid $border;
}

QScrollArea { border: none; background: transparent; }
QGraphicsView { border: 1px solid $border; border-radius: 6px; }

QScrollBar:vertical { background: $bg; width: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: $border_light; min-height: 24px; border-radius: 5px;
}
QScrollBar::handle:vertical:hover { background: $purple_dim; }
QScrollBar:horizontal { background: $bg; height: 10px; margin: 0; }
QScrollBar::handle:horizontal {
    background: $border_light; min-width: 24px; border-radius: 5px;
}
QScrollBar::handle:horizontal:hover { background: $purple_dim; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
""")


def stylesheet():
    return _STYLESHEET.substitute(
        bg=BG, panel=PANEL, surface=SURFACE, hover=HOVER, pressed=PRESSED,
        border=BORDER, border_light=BORDER_LIGHT, text=TEXT, muted=MUTED,
        blue_dim=BLUE_DIM, purple=PURPLE, purple_dim=PURPLE_DIM,
        purple_light=PURPLE_LIGHT,
    )


def apply_theme(app):
    """Apply the dark theme to the whole application (call once)."""
    app.setStyle("Fusion")
    app.setPalette(_build_palette())
    app.setStyleSheet(stylesheet())

    # Must be set before any PlotWidget is created.
    pg.setConfigOptions(background=PLOT_BG, foreground=PLOT_FG)
