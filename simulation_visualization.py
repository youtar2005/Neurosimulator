"""
simulation_visualization.py - live oscilloscope for a running Network.

Contents
    * display settings
    * min/max plot decimation (spikes are never lost)
    * SimulationWindow: voltage, per-connection PSP, currents, weights,
      spike raster and the period readout, plus stimulation controls

Depends on simulation.py (engine + buffers) and analytics.py (period
estimate). It does not know about the network editor.
"""

import threading
import traceback

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

import theme
from analytics import estimate_period
from simulation import (
    CHANNELS,
    ConnectionPlotBuffer,
    PlotBuffer,
    SimulationWorker,
    SpikeLog,
    WEIGHT_CAPACITY,
    WEIGHT_SPAN_S,
    WeightHistory,
)

# %% Display settings

# ============================================================
# Display settings
# ============================================================

window = 50.0               # Display window (ms)

GUI_INTERVAL_MS = 33        # ~30 FPS
YRANGE_EVERY = 10           # Recompute current-plot Y range every N frames
MAX_PLOT_POINTS = 1000      # Max points drawn per curve (after decimation)

V_Na_threshold = -40.0      # Display threshold (mV)

RASTER_SPAN_MS = 2000.0     # Time span shown in the spike raster
RASTER_UPDATE_MS = 250      # Raster / period readout refresh interval

# %% Plot decimation

# ============================================================
# Plot decimation (min/max, so spikes are never lost)
# ============================================================

def make_bins(t, dt_, max_points=MAX_PLOT_POINTS):
    """
    Choose bins for min/max decimation, or None if not needed.

    Bin edges are aligned to absolute sample indices, so the bins do
    not shift as the window scrolls (this avoids shimmering).
    """
    n = len(t)

    if n == 0:
        return None

    per_bin = max(1, max_points // 2)    # each bin yields 2 points
    k = -(-n // per_bin)                 # ceil(n / per_bin)

    if k <= 1:
        return None

    start = (-int(round(t[0] / dt_))) % k
    m = (n - start) // k

    if m < 1:
        return None

    return start, k, m


def decimate(t, y, bins):
    """Reduce (t, y) to <= MAX_PLOT_POINTS, keeping each bin's extremes."""
    if bins is None:
        return t, y

    start, k, m = bins
    end = start + m * k

    yr = y[start:end].reshape(m, k)
    tr = t[start:end].reshape(m, k)

    lo = yr.min(axis=1)
    hi = yr.max(axis=1)

    # Emit the two extremes in the order they occurred in time.
    swap = yr.argmin(axis=1) > yr.argmax(axis=1)
    first = np.where(swap, hi, lo)
    second = np.where(swap, lo, hi)

    xs = np.empty(2 * m)
    ys = np.empty(2 * m)
    xs[0::2] = tr[:, 0]
    xs[1::2] = tr[:, -1]
    ys[0::2] = first
    ys[1::2] = second

    return xs, ys

# %% Simulation window (oscilloscope)

# ============================================================
# Simulation window (oscilloscope)
# ============================================================

_WATCHED_EVENTS = (
    QtCore.QEvent.KeyPress,
    QtCore.QEvent.KeyRelease,
    QtCore.QEvent.WindowDeactivate,
)

SPEEDS = (("0.01x", 0.01), ("0.1x", 0.1), ("0.5x", 0.5),
          ("1x", 1.0), ("2x", 2.0))


class SimulationWindow(QtWidgets.QMainWindow):
    def __init__(self, network, colors, window_ms=window):
        super().__init__()

        self.network = network
        self.neurons = network.neurons
        self.colors = colors
        self.window_ms = window_ms
        self.dt = network.dt
        self.selected = 0

        self.last_version = -1
        self.last_x_range = None
        self.frame = 0
        self.error_shown = False

        capacity = max(2, int(np.ceil(window_ms / self.dt)) + 2)
        self.buffers = [PlotBuffer(capacity) for _ in self.neurons]
        self.connection_buffer = ConnectionPlotBuffer(len(network.synapses), capacity)

        self.plastic_idx = [
            k for k, s in enumerate(network.synapses) if s.plastic
        ]
        self.weight_hist = (
            WeightHistory(len(self.plastic_idx), WEIGHT_CAPACITY)
            if self.plastic_idx else None
        )
        self.last_weight_version = -1

        self.spike_log = SpikeLog(len(self.neurons))

        self.stop_event = threading.Event()
        self.worker = SimulationWorker(
            network, self.buffers, self.stop_event,
            weight_hist=self.weight_hist, plastic_idx=self.plastic_idx,
            spike_log=self.spike_log, connection_buffer=self.connection_buffer,
        )

        self.setWindowTitle(
            f"Simulation - {len(self.neurons)} neuron(s)"
        )
        self.resize(1250, 800)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        self.setCentralWidget(central)

        # ---------------- Plots ----------------

        plots = QtWidgets.QVBoxLayout()
        root.addLayout(plots, stretch=1)

        self.voltage_plot = pg.PlotWidget(title="Membrane Voltage")
        self.voltage_plot.setLabel("left", "Voltage", units="mV")
        self.voltage_plot.setLabel("bottom", "Time", units="ms")
        self.voltage_plot.showGrid(x=True, y=True, alpha=0.3)
        self.voltage_plot.addLegend()

        self.voltage_lines = []

        for neuron, color in zip(self.neurons, self.colors):
            self.voltage_lines.append(self.voltage_plot.plot(
                pen=pg.mkPen(color, width=2.0), name=neuron.name,
            ))

        self.voltage_plot.addLine(
            y=V_Na_threshold,
            pen=pg.mkPen(theme.THRESHOLD_COLOR, style=QtCore.Qt.DashLine),
        )
        plots.addWidget(self.voltage_plot, stretch=1)

        # Connection-specific synaptic voltage contributions, replacing the
        # former zoomed/filtered membrane-voltage plot.
        self.connection_plot = pg.PlotWidget(
            title="Estimated Postsynaptic Potential by Synaptic Connection"
        )
        self.connection_plot.setLabel("left", "Estimated voltage contribution", units="mV")
        self.connection_plot.setLabel("bottom", "Time", units="ms")
        self.connection_plot.showGrid(x=True, y=True, alpha=0.3)
        self.connection_plot.addLegend()
        self.connection_lines = []
        for j, syn in enumerate(self.network.synapses):
            color = theme.series_color(j)
            kind = "EPSP" if syn.E_rev > -40.0 else "IPSP"
            self.connection_lines.append(self.connection_plot.plot(
                pen=pg.mkPen(color, width=1.8),
                name=f"{syn.pre.name} → {syn.post.name} ({kind})",
            ))
        plots.addWidget(self.connection_plot, stretch=1)
        self.connection_plot.setXLink(self.voltage_plot)

        self.current_plot = pg.PlotWidget(title="")
        self.current_plot.setLabel("left", "Current", units="uA/cm²")
        self.current_plot.setLabel("bottom", "Time", units="ms")
        self.current_plot.showGrid(x=True, y=True, alpha=0.3)
        self.current_plot.addLegend()

        specs = (
            ("Na", "Na current", theme.CURRENT_COLORS["Na"]),
            ("K", "K current", theme.CURRENT_COLORS["K"]),
            ("Cl", "Cl current", theme.CURRENT_COLORS["Cl"]),
            ("pump", "Na/K pump current", theme.CURRENT_COLORS["pump"]),
            ("syn", "Synaptic / gap current", theme.CURRENT_COLORS["syn"]),
            ("stim", "Stimulus current", theme.CURRENT_COLORS["stim"]),
        )

        self.current_lines = {}

        for key, label, color in specs:
            self.current_lines[key] = self.current_plot.plot(
                pen=pg.mkPen(color, width=1.6), name=label,
            )

        plots.addWidget(self.current_plot, stretch=1)

        self.current_plot.setXLink(self.voltage_plot)

        self.weight_plot = None
        self.weight_lines = []

        if self.plastic_idx:
            wp = pg.PlotWidget(title="Hebbian synaptic weights")
            wp.setLabel("left", "g", units="mS/cm²")
            wp.setLabel("bottom", "Time", units="s")
            wp.showGrid(x=True, y=True, alpha=0.3)
            wp.addLegend()
            wp.disableAutoRange()

            top = max(network.synapses[k].w_max for k in self.plastic_idx)
            wp.setYRange(0, top * 1.05, padding=0)

            for j, k in enumerate(self.plastic_idx):
                syn = network.synapses[k]
                self.weight_lines.append(wp.plot(
                    pen=pg.mkPen(theme.series_color(j), width=2.0),
                    name=f"{syn.pre.name} → {syn.post.name}",
                ))

            plots.addWidget(wp, stretch=1)
            self.weight_plot = wp

        # ---------------- Spike raster + period readout ----------------

        raster_widget = QtWidgets.QWidget()
        raster_widget.setFixedWidth(340)
        raster_col = QtWidgets.QVBoxLayout(raster_widget)
        raster_col.setContentsMargins(0, 0, 0, 0)
        root.addWidget(raster_widget)

        self.raster_plot = pg.PlotWidget(title="Spike raster")
        self.raster_plot.setLabel("bottom", "Time", units="s")
        self.raster_plot.showGrid(x=True, y=False, alpha=0.3)
        self.raster_plot.disableAutoRange()
        self.raster_plot.setYRange(
            -0.6, len(self.neurons) - 0.4, padding=0
        )
        self.raster_plot.getAxis("left").setTicks(
            [[(i, n.name) for i, n in enumerate(self.neurons)]]
        )

        self.raster_lines = []

        for neuron, color in zip(self.neurons, self.colors):
            self.raster_lines.append(self.raster_plot.plot(
                pen=pg.mkPen(color, width=2), connect="pairs",
            ))

        raster_col.addWidget(self.raster_plot, stretch=1)

        raster_col.addWidget(QtWidgets.QLabel(
            "Period (last 3 s of spikes):"
        ))
        self.period_label = QtWidgets.QLabel("waiting for spikes...")
        self.period_label.setStyleSheet(
            f"font-family: monospace; color: {theme.PURPLE_LIGHT};"
        )
        self.period_label.setWordWrap(True)
        raster_col.addWidget(self.period_label)

        # Manual ranges only: no autorange, clip-to-view or auto
        # downsampling (we decimate ourselves).
        self.voltage_plot.disableAutoRange()
        self.current_plot.disableAutoRange()
        self.voltage_plot.setYRange(-100, 70, padding=0)
        self.current_plot.setYRange(-400, 400, padding=0)
        self.connection_plot.setYRange(-20, 20, padding=0)

        # Plots never see the mouse or keyboard.
        for plot in [p for p in (self.voltage_plot, self.connection_plot,
                                 self.current_plot, self.weight_plot,
                                 self.raster_plot)
                     if p is not None]:
            plot.setMouseEnabled(x=False, y=False)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.setFocusPolicy(QtCore.Qt.NoFocus)
            plot.setMouseTracking(False)
            plot.viewport().setMouseTracking(False)
            plot.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            plot.viewport().setAttribute(
                QtCore.Qt.WA_TransparentForMouseEvents, True
            )

        # ---------------- Side panel ----------------

        side = QtWidgets.QVBoxLayout()
        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(200)
        root.addWidget(side_widget)

        side.addWidget(QtWidgets.QLabel("Currents of:"))
        self.selector = QtWidgets.QComboBox()
        self.selector.addItems([n.name for n in self.neurons])
        self.selector.setFocusPolicy(QtCore.Qt.NoFocus)
        self.selector.currentIndexChanged.connect(self.on_select)
        side.addWidget(self.selector)

        side.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_box = QtWidgets.QComboBox()
        for label, _ in SPEEDS:
            self.speed_box.addItem(label)
        self.speed_box.setCurrentIndex(3)
        self.speed_box.setFocusPolicy(QtCore.Qt.NoFocus)
        self.speed_box.currentIndexChanged.connect(self.on_speed)
        side.addWidget(self.speed_box)

        side.addWidget(QtWidgets.QLabel(
            "Stimulate (hold a button,\nor hold key 1-9 / Space=first):"
        ))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(QtCore.Qt.NoFocus)
        holder = QtWidgets.QWidget()
        holder_layout = QtWidgets.QVBoxLayout(holder)
        holder_layout.setContentsMargins(2, 2, 2, 2)

        for i, neuron in enumerate(self.neurons):
            text = f"[{i + 1}] {neuron.name}" if i < 9 else neuron.name
            button = QtWidgets.QPushButton(text)
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            button.pressed.connect(
                lambda n=neuron: setattr(n, "stim_active", True)
            )
            button.released.connect(
                lambda n=neuron: setattr(n, "stim_active", False)
            )
            holder_layout.addWidget(button)

        holder_layout.addStretch(1)
        scroll.setWidget(holder)
        side.addWidget(scroll, stretch=1)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {theme.ERROR};")
        side.addWidget(self.status)

        # ---------------- Keys ----------------

        self.key_map = {QtCore.Qt.Key_Space: 0}

        for i in range(min(9, len(self.neurons))):
            self.key_map[getattr(QtCore.Qt, f"Key_{i + 1}")] = i

        QtWidgets.QApplication.instance().installEventFilter(self)

        self.on_select(0)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(GUI_INTERVAL_MS)

        self.raster_timer = QtCore.QTimer(self)
        self.raster_timer.timeout.connect(self.update_raster)
        self.raster_timer.start(RASTER_UPDATE_MS)

        self.worker.start()

    # ---------------- controls ----------------

    def on_select(self, index):
        self.selected = max(0, index)
        self.current_plot.setTitle(
            f"Membrane Currents - {self.neurons[self.selected].name}"
        )
        self.last_version = -1
        self.frame = 0

    def on_speed(self, index):
        self.worker.speed = SPEEDS[index][1]

    # ---------------- drawing ----------------

    def update_plots(self):
        try:
            self._update_plots()
            self._update_weights()
        except Exception:
            traceback.print_exc()
            self.timer.stop()

    def _update_weights(self):
        if self.weight_plot is None:
            return

        version = self.weight_hist.version

        if version == self.last_weight_version:
            return

        self.last_weight_version = version

        t, w = self.weight_hist.snapshot()

        if len(t) == 0:
            return

        ts = t / 1000.0

        for line, row in zip(self.weight_lines, w):
            line.setData(ts, row, skipFiniteCheck=True)

        self.weight_plot.setXRange(
            max(0.0, ts[-1] - WEIGHT_SPAN_S),
            max(WEIGHT_SPAN_S, ts[-1]),
            padding=0,
        )

    def _update_plots(self):
        if self.worker.error and not self.error_shown:
            self.status.setText(self.worker.error)
            self.error_shown = True

        version = self.buffers[0].version

        if version == self.last_version:
            return

        self.last_version = version

        # Voltages of all neurons (only t and V are copied).
        right = None

        for line, buffer in zip(self.voltage_lines, self.buffers):
            snap = buffer.snapshot(("t", "V", "syn"))
            t_data = snap["t"]

            if len(t_data) == 0:
                continue

            bins = make_bins(t_data, self.dt)
            voltage = snap["V"]
            x, y = decimate(t_data, voltage, bins)
            line.setData(x, y, skipFiniteCheck=True)

            if right is None or t_data[-1] > right:
                right = t_data[-1]

        if right is None:
            return

        # Plot per-connection estimated voltage contributions.
        if self.connection_lines:
            ct, cv = self.connection_buffer.snapshot()
            if len(ct):
                cbins = make_bins(ct, self.dt)
                vals = []
                for line, row in zip(self.connection_lines, cv):
                    cx, cy = decimate(ct, row, cbins)
                    line.setData(cx, cy, skipFiniteCheck=True)
                    vals.append(row)
                if vals and self.frame % YRANGE_EVERY == 0:
                    lo = min(float(v.min()) for v in vals)
                    hi = max(float(v.max()) for v in vals)
                    if hi - lo < 0.1:
                        mid = 0.5 * (hi + lo)
                        lo, hi = mid - 0.05, mid + 0.05
                    pad = 0.1 * (hi - lo)
                    self.connection_plot.setYRange(lo - pad, hi + pad, padding=0)

        # Currents of the selected neuron.
        selected = self.buffers[self.selected].snapshot()
        t_sel = selected["t"]

        if len(t_sel) > 0:
            bins = make_bins(t_sel, self.dt)

            for key, line in self.current_lines.items():
                x, y = decimate(t_sel, selected[key], bins)
                line.setData(x, y, skipFiniteCheck=True)

            if self.frame % YRANGE_EVERY == 0:
                lo = min(selected[k].min() for k in self.current_lines)
                hi = max(selected[k].max() for k in self.current_lines)

                if hi - lo < 1.0:
                    mid = 0.5 * (hi + lo)
                    lo, hi = mid - 0.5, mid + 0.5

                pad = 0.1 * (hi - lo)
                self.current_plot.setYRange(lo - pad, hi + pad, padding=0)

            self.frame += 1

        left = max(0.0, right - self.window_ms)
        x_range = (left, max(self.window_ms, right))

        if x_range != self.last_x_range:
            self.voltage_plot.setXRange(
                x_range[0], x_range[1], padding=0
            )
            self.last_x_range = x_range

    def update_raster(self):
        try:
            self._update_raster()
        except Exception:
            traceback.print_exc()
            self.raster_timer.stop()

    def _update_raster(self):
        now = self.network.t
        spikes = self.spike_log.snapshot()

        lo = max(0.0, now - RASTER_SPAN_MS)
        hi = max(RASTER_SPAN_MS, now)

        lines = []

        for i, (curve, ts) in enumerate(zip(self.raster_lines, spikes)):
            visible = ts[ts >= lo]
            x = np.repeat(visible / 1000.0, 2)
            y = np.tile((i - 0.4, i + 0.4), len(visible))
            curve.setData(x, y, connect="pairs")

            name = self.neurons[i].name
            est = estimate_period(ts, now)

            if est is None:
                lines.append(f"{name}: no rhythm yet")
            else:
                note = "" if est["cv"] < 0.1 else "  (irregular)"
                lines.append(
                    f"{name}: {est['P']:.1f} ms, "
                    f"{est['spc']:.0f} spike(s)/cycle{note}"
                )

        self.raster_plot.setXRange(lo / 1000.0, hi / 1000.0, padding=0)
        self.period_label.setText("\n".join(lines))

    # ---------------- keyboard ----------------

    def eventFilter(self, obj, event):
        etype = event.type()

        # Cheap early exit: this filter sees every event in the app.
        if etype not in _WATCHED_EVENTS:
            return False

        if etype == QtCore.QEvent.WindowDeactivate:
            if obj is self:
                for neuron in self.neurons:
                    neuron.stim_active = False
            return False

        # Only react while this window is active (the editor has text
        # fields that need the digit keys).
        if QtWidgets.QApplication.activeWindow() is not self:
            return False

        index = self.key_map.get(event.key())

        if index is None:
            return False

        if not event.isAutoRepeat():
            self.neurons[index].stim_active = (
                etype == QtCore.QEvent.KeyPress
            )

        return True

    def closeEvent(self, event):
        QtWidgets.QApplication.instance().removeEventFilter(self)
        self.stop_event.set()
        self.timer.stop()
        self.raster_timer.stop()
        self.worker.join(timeout=2.0)
        super().closeEvent(event)
