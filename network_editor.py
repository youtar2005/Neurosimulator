"""
network_editor.py - graphical editor for building a neuron network, and
the application entry point.

Run this file to start the app:

    python network_editor.py

Contents
    * graphics items (NeuronItem, EdgeItem)
    * EditorScene (editing, build_network, save/load)
    * PropertiesPanel
    * EditorWindow (toolbar, file ops, launches the SimulationWindow)
    * main()
"""

import dataclasses
import faulthandler
import json
import math
import sys
import traceback
from dataclasses import asdict

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from simulation import (
    AMPA,
    GABA_A,
    MAX_DELAY_MS,
    VETO,
    Network,
    Neuron,
    NeuronParams,
    dt,
)
from simulation_visualization import SimulationWindow
import theme

faulthandler.enable()


def excepthook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)


sys.excepthook = excepthook

# %% Network editor: graphics items

# ============================================================
# Network editor: graphics items
# ============================================================

EDGE_COLORS = {
    kind: QtGui.QColor(color) for kind, color in theme.EDGE_COLORS.items()
}

EDGE_LABELS = {
    "exc": "Excitatory synapse",
    "inh": "Inhibitory synapse",
    "gap": "Gap junction",
    "veto": "Veto synapse",
}

EDGE_DEFAULTS = {
    "exc": dict(g=0.5, **AMPA),
    "inh": dict(g=0.5, **GABA_A),
    "gap": dict(g=0.05, E_rev=0.0, alpha=0.0, beta=0.0),
    "veto": dict(g=1.0, E_rev=0.0, **VETO),
}

PLASTICITY_DEFAULTS = dict(
    plastic=False, A_plus=0.005, A_minus=0.006,
    tau_plus=20.0, tau_minus=20.0, w_max=2.0, tau_decay=0.0,
)

PLASTIC_FIELDS = [
    ("A+ (LTP rate)", "A_plus", 0, 1, 4, 0.001),
    ("A- (LTD rate)", "A_minus", 0, 1, 4, 0.001),
    ("tau+ (ms)", "tau_plus", 0.1, 1000, 1, 1.0),
    ("tau- (ms)", "tau_minus", 0.1, 1000, 1, 1.0),
    ("g max (mS/cm²)", "w_max", 0, 20, 3, 0.1),
    ("Decay tau (ms, 0=off)", "tau_decay", 0, 1e6, 0, 1000.0),
]


class NeuronItem(QtWidgets.QGraphicsEllipseItem):
    RADIUS = 26

    def __init__(self, name, color, x, y):
        R = self.RADIUS
        super().__init__(-R, -R, 2 * R, 2 * R)

        self.edges = []
        self.name = name
        self.color = QtGui.QColor(color)
        self.params = NeuronParams()
        self.stim_amplitude = 15.0
        self.bias_current = 0.0

        self.setBrush(QtGui.QBrush(self.color))
        self.setPen(QtGui.QPen(QtGui.QColor(theme.NEURON_OUTLINE), 2))
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setZValue(1)

        self.label = QtWidgets.QGraphicsSimpleTextItem(name, self)
        self.label.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        font = self.label.font()
        font.setBold(True)
        self.label.setFont(font)
        self.label.setBrush(QtGui.QBrush(theme.contrast_text(self.color)))
        self._center_label()

        self.setPos(x, y)

    def _center_label(self):
        rect = self.label.boundingRect()
        self.label.setPos(-rect.width() / 2, -rect.height() / 2)

    def set_name(self, name):
        self.name = name
        self.label.setText(name)
        self._center_label()

    def paint(self, painter, option, widget=None):
        # Selection is shown by the outline, not Qt's dashed rectangle.
        option.state &= ~QtWidgets.QStyle.State_Selected
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        super().paint(painter, option, widget)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            for edge in self.edges:
                edge.refresh()
        elif change == QtWidgets.QGraphicsItem.ItemSelectedHasChanged:
            if value:
                self.setPen(QtGui.QPen(QtGui.QColor(theme.SELECTION), 4))
            else:
                self.setPen(QtGui.QPen(QtGui.QColor(theme.NEURON_OUTLINE), 2))

        return super().itemChange(change, value)


class EdgeItem(QtWidgets.QGraphicsItem):
    def __init__(self, kind, src, dst):
        super().__init__()

        self.kind = kind
        self.src = src
        self.dst = dst

        defaults = EDGE_DEFAULTS[kind]
        self.g = defaults["g"]
        self.E_rev = defaults["E_rev"]
        self.alpha = defaults["alpha"]
        self.beta = defaults["beta"]

        for key, value in PLASTICITY_DEFAULTS.items():
            setattr(self, key, value)

        self.gate_ext = False
        self.delay = 0.0

        self._curve = QtGui.QPainterPath()
        self._head = QtGui.QPainterPath()
        self._shape = QtGui.QPainterPath()
        self._label_pos = QtCore.QPointF()

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.setZValue(0)

        src.edges.append(self)
        dst.edges.append(self)
        self.refresh()

    @staticmethod
    def _unit(v):
        length = math.hypot(v.x(), v.y()) or 1.0
        return QtCore.QPointF(v.x() / length, v.y() / length)

    def refresh(self):
        self.prepareGeometryChange()

        R = NeuronItem.RADIUS
        p1 = self.src.pos()
        p2 = self.dst.pos()

        u = self._unit(p2 - p1)
        normal = QtCore.QPointF(-u.y(), u.x())

        # Chemical synapses bow to one side so that A->B and B->A
        # do not overlap. Gap junctions are straight.
        offset = 0.0 if self.kind == "gap" else 24.0
        ctrl = (p1 + p2) / 2 + normal * offset

        ua = self._unit(ctrl - p1)
        ub = self._unit(p2 - ctrl)

        start = p1 + ua * R
        end = p2 - ub * R

        curve = QtGui.QPainterPath(start)
        curve.quadTo(ctrl, end)

        head = QtGui.QPainterPath()
        perp = QtCore.QPointF(-ub.y(), ub.x())

        if self.kind == "exc":
            base = end - ub * 11
            head.moveTo(end)
            head.lineTo(base + perp * 5.5)
            head.lineTo(base - perp * 5.5)
            head.closeSubpath()
        elif self.kind == "inh":
            head.addEllipse(end - ub * 5, 5, 5)
        elif self.kind == "veto":
            back = end - ub * 4
            head.moveTo(end + perp * 9)
            head.lineTo(end - perp * 9)
            head.lineTo(back - perp * 9)
            head.lineTo(back + perp * 9)
            head.closeSubpath()
        else:
            head.addEllipse(start, 3.5, 3.5)
            head.addEllipse(end, 3.5, 3.5)

        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(12)
        shape = stroker.createStroke(curve)
        shape.addPath(head)

        self._curve = curve
        self._head = head
        self._shape = shape
        self._label_pos = start * 0.25 + ctrl * 0.5 + end * 0.25

        self.update()

    def plasticity(self):
        return {key: getattr(self, key) for key in PLASTICITY_DEFAULTS}

    def boundingRect(self):
        lp = self._label_pos
        label_rect = QtCore.QRectF(lp.x() - 40, lp.y() - 14, 80, 28)
        return self._shape.boundingRect().united(label_rect)

    def shape(self):
        return self._shape

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        color = EDGE_COLORS[self.kind]
        selected = self.isSelected()
        draw_color = QtGui.QColor(theme.SELECTION) if selected else color

        pen = QtGui.QPen(draw_color, 3 if selected else 2)

        if self.kind == "gap":
            pen.setStyle(QtCore.Qt.DashLine)

        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(self._curve)

        painter.setPen(QtGui.QPen(draw_color, 1))
        painter.setBrush(QtGui.QBrush(draw_color))
        painter.drawPath(self._head)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.EDGE_LABEL)))
        lp = self._label_pos
        painter.drawText(
            QtCore.QRectF(lp.x() - 40, lp.y() - 14, 80, 28),
            QtCore.Qt.AlignCenter,
            f"{self.g:g}" + (" H" if self.plastic else "")
            + (f" {self.delay:g}ms"
               if self.delay > 0 and self.kind != "gap" else ""),
        )

# %% Network editor: scene

# ============================================================
# Network editor: scene
# ============================================================

MODE_HINTS = {
    "select": "Click to select, drag to move. Delete removes the selection.",
    "add": "Click on empty canvas to place a neuron.",
    "exc": "Drag from the presynaptic neuron to the postsynaptic neuron.",
    "inh": "Drag from the presynaptic neuron to the postsynaptic neuron.",
    "gap": "Drag between two neurons to couple them electrically.",
    "veto": "Drag from the vetoing neuron to the neuron it silences.",
}


class EditorScene(QtWidgets.QGraphicsScene):
    status = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.setSceneRect(-1000, -700, 2000, 1400)

        self.mode = "select"
        self.neurons = []
        self.edges = []

        self._palette_counter = 0
        self._pending_src = None
        self._rubber = None

    # ---------------- model operations ----------------

    def unique_name(self):
        names = {n.name for n in self.neurons}
        i = 1

        while f"N{i}" in names:
            i += 1

        return f"N{i}"

    def add_neuron(self, x, y, name=None, color=None):
        if color is None:
            color = theme.neuron_color(self._palette_counter)

        self._palette_counter += 1

        item = NeuronItem(name or self.unique_name(), color, x, y)
        self.addItem(item)
        self.neurons.append(item)

        return item

    def add_edge(self, kind, src, dst, **overrides):
        for edge in self.edges:
            same = edge.kind == kind and edge.src is src and edge.dst is dst
            reverse_gap = (
                kind == "gap" and edge.kind == "gap"
                and edge.src is dst and edge.dst is src
            )

            if same or reverse_gap:
                return None

        edge = EdgeItem(kind, src, dst)

        for key, value in overrides.items():
            setattr(edge, key, value)

        edge.refresh()
        self.addItem(edge)
        self.edges.append(edge)

        return edge

    def remove_edge(self, edge):
        if edge not in self.edges:
            return

        self.edges.remove(edge)

        for neuron in (edge.src, edge.dst):
            if edge in neuron.edges:
                neuron.edges.remove(edge)

        self.removeItem(edge)

    def remove_neuron(self, neuron):
        if neuron not in self.neurons:
            return

        for edge in list(neuron.edges):
            self.remove_edge(edge)

        self.neurons.remove(neuron)
        self.removeItem(neuron)

    def delete_selected(self):
        selected = self.selectedItems()

        for item in selected:
            if isinstance(item, EdgeItem):
                self.remove_edge(item)

        for item in selected:
            if isinstance(item, NeuronItem):
                self.remove_neuron(item)

    def clear_all(self):
        self.clearSelection()
        self.neurons = []
        self.edges = []
        self._pending_src = None
        self._rubber = None
        self._palette_counter = 0
        self.clear()

    def neuron_at(self, pos):
        for item in self.items(pos):
            while item is not None and not isinstance(item, NeuronItem):
                item = item.parentItem()

            if item is not None:
                return item

        return None

    # ---------------- build a runnable Network ----------------

    def build_network(self):
        net = Network(dt=dt)
        mapping = {}
        colors = []

        for item in self.neurons:
            neuron = Neuron(
                item.name,
                params=NeuronParams(**asdict(item.params)),
                stim_amplitude=item.stim_amplitude,
                bias_current=item.bias_current,
            )
            net.add(neuron)
            mapping[item] = neuron
            colors.append(QtGui.QColor(item.color))

        for edge in self.edges:
            a = mapping[edge.src]
            b = mapping[edge.dst]

            if edge.kind == "gap":
                net.connect_gap(a, b, edge.g)
            elif edge.kind == "veto":
                net.connect_veto(
                    a, b, strength=edge.g, alpha=edge.alpha,
                    beta=edge.beta, gate_ext=edge.gate_ext,
                    delay=edge.delay,
                )
            else:
                net.connect_chemical(
                    a, b, edge.g, edge.E_rev, edge.alpha, edge.beta,
                    delay=edge.delay, **edge.plasticity(),
                )

        return net, colors

    # ---------------- save / load ----------------

    def to_dict(self):
        index = {item: i for i, item in enumerate(self.neurons)}

        return {
            "version": 1,
            "neurons": [
                {
                    "name": n.name,
                    "x": n.pos().x(),
                    "y": n.pos().y(),
                    "color": [n.color.red(), n.color.green(), n.color.blue()],
                    "stim_amplitude": n.stim_amplitude,
                    "bias_current": n.bias_current,
                    "params": asdict(n.params),
                }
                for n in self.neurons
            ],
            "edges": [
                {
                    "kind": e.kind,
                    "src": index[e.src],
                    "dst": index[e.dst],
                    "g": e.g,
                    "E_rev": e.E_rev,
                    "alpha": e.alpha,
                    "beta": e.beta,
                    "gate_ext": e.gate_ext,
                    "delay": e.delay,
                    "plasticity": e.plasticity(),
                }
                for e in self.edges
            ],
        }

    def from_dict(self, data):
        valid = {f.name for f in dataclasses.fields(NeuronParams)}

        self.clear_all()
        created = []

        for entry in data["neurons"]:
            item = self.add_neuron(
                entry["x"], entry["y"],
                name=entry["name"],
                color=QtGui.QColor(*entry["color"]),
            )
            item.stim_amplitude = entry.get("stim_amplitude", 15.0)
            item.bias_current = entry.get("bias_current", 0.0)
            item.params = NeuronParams(**{
                k: v for k, v in entry.get("params", {}).items()
                if k in valid
            })
            created.append(item)

        for entry in data["edges"]:
            self.add_edge(
                entry["kind"],
                created[entry["src"]],
                created[entry["dst"]],
                g=entry["g"],
                E_rev=entry.get("E_rev", 0.0),
                alpha=entry.get("alpha", 0.0),
                beta=entry.get("beta", 0.0),
                gate_ext=entry.get("gate_ext", False),
                delay=entry.get("delay", 0.0),
                **{k: v for k, v in entry.get("plasticity", {}).items()
                   if k in PLASTICITY_DEFAULTS},
            )

    # ---------------- mouse handling ----------------

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.scenePos()

            if self.mode == "add":
                if not self.items(pos):
                    item = self.add_neuron(pos.x(), pos.y())
                    self.clearSelection()
                    item.setSelected(True)
                    event.accept()
                    return

            elif self.mode in ("exc", "inh", "gap", "veto"):
                src = self.neuron_at(pos)

                if src is not None:
                    self._pending_src = src

                    pen = QtGui.QPen(EDGE_COLORS[self.mode], 2)
                    pen.setStyle(QtCore.Qt.DashLine)
                    self._rubber = QtWidgets.QGraphicsLineItem(
                        QtCore.QLineF(src.pos(), pos)
                    )
                    self._rubber.setPen(pen)
                    self._rubber.setZValue(2)
                    self._rubber.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                    self.addItem(self._rubber)

                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._rubber is not None and self._pending_src is not None:
            self._rubber.setLine(QtCore.QLineF(
                self._pending_src.pos(), event.scenePos()
            ))
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (self._pending_src is not None
                and event.button() == QtCore.Qt.LeftButton):
            src = self._pending_src
            dst = self.neuron_at(event.scenePos())

            self.removeItem(self._rubber)
            self._rubber = None
            self._pending_src = None

            if dst is not None and dst is not src:
                edge = self.add_edge(self.mode, src, dst)

                if edge is None:
                    self.status.emit("That connection already exists.")
                else:
                    self.clearSelection()
                    edge.setSelected(True)

            event.accept()
            return

        super().mouseReleaseEvent(event)

# %% Network editor: properties panel

# ============================================================
# Network editor: properties panel
# ============================================================

NEURON_FIELDS = [
    ("Stimulus amplitude (µA/cm²)", "stim_amplitude", -500, 500, 2, 1.0),
    ("Bias current (µA/cm²)", "bias_current", -500, 500, 2, 0.5),
    ("C_m (µF/cm²)", "params.C_m", 0.1, 10, 2, 0.1),
    ("g_Na max (mS/cm²)", "params.g_Na_max", 0, 1000, 1, 5.0),
    ("g_K max (mS/cm²)", "params.g_K_max", 0, 500, 1, 1.0),
    ("g_Cl (mS/cm²)", "params.g_Cl", 0, 50, 3, 0.05),
    ("E_Na (mV)", "params.E_Na", -100, 150, 1, 1.0),
    ("E_K (mV)", "params.E_K", -150, 0, 1, 1.0),
    ("E_Cl (mV)", "params.E_Cl", -150, 50, 1, 1.0),
    ("Pump max (µA/cm²)", "params.I_pump_max", 0, 100, 2, 0.5),
]

CHEMICAL_FIELDS = [
    ("E_rev (mV)", "E_rev", -120, 50, 1, 1.0),
    ("alpha (1/ms)", "alpha", 0, 100, 3, 0.1),
    ("beta (1/ms)", "beta", 0, 100, 3, 0.05),
]


def get_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)

    return obj


def set_path(obj, path, value):
    *head, last = path.split(".")

    for part in head:
        obj = getattr(obj, part)

    setattr(obj, last, value)


def make_spin(lo, hi, decimals, step):
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setKeyboardTracking(False)

    return spin


class PropertiesPanel(QtWidgets.QStackedWidget):
    def __init__(self, scene):
        super().__init__()

        self.scene = scene
        self.item = None
        self._loading = False

        self.setFixedWidth(300)

        # ---- Page 0: help ----
        help_label = QtWidgets.QLabel(
            "Select a single neuron or connection to edit its "
            "properties.\n\n"
            "Toolbar:\n"
            "• Add Neuron: click the canvas\n"
            "• Synapse / Gap / Veto: drag from one neuron to another\n"
            "• Select: click or drag items; Delete removes them\n\n"
            "Then press Simulate."
        )
        help_label.setWordWrap(True)
        help_label.setAlignment(QtCore.Qt.AlignTop)
        self.addWidget(help_label)

        # ---- Page 1: neuron ----
        neuron_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(neuron_page)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name)
        form.addRow("Name", self.name_edit)

        self.neuron_spins = {}

        for label, path, lo, hi, decimals, step in NEURON_FIELDS:
            spin = make_spin(lo, hi, decimals, step)
            spin.valueChanged.connect(
                lambda v, p=path: self._on_neuron_value(p, v)
            )
            self.neuron_spins[path] = spin
            form.addRow(label, spin)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(neuron_page)
        self.addWidget(scroll)

        # ---- Page 2: connection ----
        edge_page = QtWidgets.QWidget()
        edge_layout = QtWidgets.QVBoxLayout(edge_page)

        self.edge_title = QtWidgets.QLabel("")
        self.edge_title.setWordWrap(True)
        edge_layout.addWidget(self.edge_title)

        edge_form = QtWidgets.QFormLayout()
        self.g_spin = make_spin(0, 20, 4, 0.05)
        self.g_spin.valueChanged.connect(
            lambda v: self._on_edge_value("g", v)
        )
        self.g_label = QtWidgets.QLabel("g (mS/cm²)")
        edge_form.addRow(self.g_label, self.g_spin)

        self.delay_spin = make_spin(0, MAX_DELAY_MS, 2, 0.5)
        self.delay_spin.valueChanged.connect(
            lambda v: self._on_edge_value("delay", v)
        )
        self.delay_label = QtWidgets.QLabel("Delay (ms)")
        edge_form.addRow(self.delay_label, self.delay_spin)
        edge_layout.addLayout(edge_form)

        self.chem_group = QtWidgets.QGroupBox("Synapse kinetics")
        chem_form = QtWidgets.QFormLayout(self.chem_group)
        self.edge_spins = {}

        for label, attr, lo, hi, decimals, step in CHEMICAL_FIELDS:
            spin = make_spin(lo, hi, decimals, step)
            spin.valueChanged.connect(
                lambda v, a=attr: self._on_edge_value(a, v)
            )
            self.edge_spins[attr] = spin
            chem_form.addRow(label, spin)

        edge_layout.addWidget(self.chem_group)

        self.plastic_group = QtWidgets.QGroupBox("Hebbian plasticity (STDP)")
        plastic_form = QtWidgets.QFormLayout(self.plastic_group)

        self.plastic_box = QtWidgets.QCheckBox("Enable")
        self.plastic_box.toggled.connect(
            lambda v: self._on_edge_value("plastic", v)
        )
        plastic_form.addRow(self.plastic_box)

        for label, attr, lo, hi, decimals, step in PLASTIC_FIELDS:
            spin = make_spin(lo, hi, decimals, step)
            spin.valueChanged.connect(
                lambda v, a=attr: self._on_edge_value(a, v)
            )
            self.edge_spins[attr] = spin
            plastic_form.addRow(label, spin)

        edge_layout.addWidget(self.plastic_group)

        self.ext_box = QtWidgets.QCheckBox("Also veto external drive (stimulus + bias)")
        self.ext_box.toggled.connect(
            lambda v: self._on_edge_value("gate_ext", v)
        )
        edge_layout.addWidget(self.ext_box)

        edge_layout.addStretch(1)

        self.addWidget(edge_page)

        scene.selectionChanged.connect(self.on_selection)

    def on_selection(self):
        selected = self.scene.selectedItems()

        if len(selected) != 1:
            self.item = None
            self.setCurrentIndex(0)
            return

        item = selected[0]
        self._loading = True

        try:
            if isinstance(item, NeuronItem):
                self.name_edit.setText(item.name)

                for path, spin in self.neuron_spins.items():
                    spin.setValue(get_path(item, path))

                self.setCurrentIndex(1)

            elif isinstance(item, EdgeItem):
                arrow = "↔" if item.kind == "gap" else "→"
                self.edge_title.setText(
                    f"{EDGE_LABELS[item.kind]}\n"
                    f"{item.src.name} {arrow} {item.dst.name}"
                )
                self.g_spin.setValue(item.g)
                self.delay_spin.setValue(item.delay)

                for attr, spin in self.edge_spins.items():
                    spin.setValue(getattr(item, attr))

                is_veto = item.kind == "veto"
                self.g_label.setText(
                    "Strength (0-1)" if is_veto else "g (mS/cm²)"
                )
                self.edge_spins["E_rev"].setEnabled(not is_veto)
                self.chem_group.setVisible(item.kind != "gap")
                self.plastic_box.setChecked(item.plastic)
                self.plastic_group.setVisible(item.kind in ("exc", "inh"))
                self.ext_box.setChecked(item.gate_ext)
                self.ext_box.setVisible(is_veto)
                self.delay_label.setVisible(item.kind != "gap")
                self.delay_spin.setVisible(item.kind != "gap")

                self.setCurrentIndex(2)

            else:
                self.setCurrentIndex(0)
                return
        finally:
            self._loading = False

        self.item = item

    def _on_name(self):
        item = self.item

        if self._loading or not isinstance(item, NeuronItem):
            return

        new = self.name_edit.text().strip()
        taken = any(n.name == new for n in self.scene.neurons if n is not item)

        if new and not taken:
            item.set_name(new)
        else:
            self.name_edit.setText(item.name)

    def _on_neuron_value(self, path, value):
        if self._loading or not isinstance(self.item, NeuronItem):
            return

        set_path(self.item, path, value)

    def _on_edge_value(self, attr, value):
        if self._loading or not isinstance(self.item, EdgeItem):
            return

        setattr(self.item, attr, value)
        self.item.refresh()

# %% Network editor: main window

# ============================================================
# Network editor: main window
# ============================================================

class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.sim = None

        self.setWindowTitle("Neuron Network Editor")
        self.resize(1200, 800)

        self.scene = EditorScene()
        self.scene.status.connect(
            lambda text: self.statusBar().showMessage(text, 4000)
        )

        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(theme.CANVAS_BG)))
        self.view.centerOn(0, 0)

        self.panel = PropertiesPanel(self.scene)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.panel)
        self.setCentralWidget(central)

        self._build_toolbar()
        self._fit_toolbar_width()
        self.set_mode("select")
        self.load_demo()

    # ---------------- toolbar ----------------

    def _build_toolbar(self):
        bar = self.addToolBar("Tools")
        bar.setMovable(False)
        self.toolbar = bar

        group = QtWidgets.QActionGroup(self)
        group.setExclusive(True)
        self.mode_actions = {}

        modes = (
            ("select", "Select / Move"),
            ("add", "Add Neuron"),
            ("exc", "Excitatory Synapse"),
            ("inh", "Inhibitory Synapse"),
            ("gap", "Gap Junction"),
            ("veto", "Veto Synapse"),
        )

        for mode, text in modes:
            action = QtWidgets.QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, m=mode: self.set_mode(m)
            )
            group.addAction(action)
            bar.addAction(action)
            self.mode_actions[mode] = action

        bar.addSeparator()

        delete = QtWidgets.QAction("Delete Selected", self)
        delete.setShortcuts([
            QtGui.QKeySequence(QtCore.Qt.Key_Delete),
            QtGui.QKeySequence(QtCore.Qt.Key_Backspace),
        ])
        delete.triggered.connect(self.scene.delete_selected)
        bar.addAction(delete)

        bar.addSeparator()

        new = QtWidgets.QAction("New", self)
        new.setShortcut(QtGui.QKeySequence.New)
        new.triggered.connect(self.new_network)
        bar.addAction(new)

        open_action = QtWidgets.QAction("Open...", self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_network)
        bar.addAction(open_action)

        save = QtWidgets.QAction("Save...", self)
        save.setShortcut(QtGui.QKeySequence.Save)
        save.triggered.connect(self.save_network)
        bar.addAction(save)

        bar.addSeparator()

        simulate = QtWidgets.QAction("▶ Simulate", self)
        simulate.setShortcut("Ctrl+R")
        simulate.triggered.connect(self.simulate)
        bar.addAction(simulate)

    def _fit_toolbar_width(self):
        """
        Make the window at least as wide as the toolbar needs, so every
        button is visible (Qt otherwise hides the overflow behind a ">>"
        menu).
        """
        self.toolbar.ensurePolished()
        needed = self.toolbar.sizeHint().width() + 24

        self.setMinimumWidth(max(self.minimumWidth(), needed))

        if self.width() < needed:
            self.resize(needed, self.height())

    def showEvent(self, event):
        # Button widths are only final once the stylesheet is polished,
        # so measure again now that the window is about to appear.
        self._fit_toolbar_width()
        super().showEvent(event)

    def set_mode(self, mode):
        self.scene.mode = mode
        self.mode_actions[mode].setChecked(True)
        self.view.setCursor(
            QtCore.Qt.ArrowCursor if mode == "select"
            else QtCore.Qt.CrossCursor
        )
        self.statusBar().showMessage(MODE_HINTS[mode])

    # ---------------- file operations ----------------

    def load_demo(self):
        a = self.scene.add_neuron(-200, 0, name="A")
        b = self.scene.add_neuron(0, -120, name="B")
        c = self.scene.add_neuron(200, 0, name="C")

        self.scene.add_edge("exc", a, b)
        self.scene.add_edge("exc", b, c)
        self.scene.add_edge("inh", c, a)

    def new_network(self):
        self.scene.clear_all()

    def save_network(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save network", "network.json", "JSON (*.json)"
        )

        if not path:
            return

        try:
            with open(path, "w") as f:
                json.dump(self.scene.to_dict(), f, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))

    def open_network(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open network", "", "JSON (*.json)"
        )

        if not path:
            return

        try:
            with open(path) as f:
                data = json.load(f)

            self.scene.from_dict(data)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(exc))

    # ---------------- simulation ----------------

    def simulate(self):
        if not self.scene.neurons:
            QtWidgets.QMessageBox.information(
                self, "Nothing to simulate", "Add at least one neuron first."
            )
            return

        net, colors = self.scene.build_network()

        if self.sim is not None:
            self.sim.close()
            self.sim = None

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        try:
            net.warmup()     # JIT-compile (cached after the first run)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.sim = SimulationWindow(net, colors)
        self.sim.show()

    def closeEvent(self, event):
        if self.sim is not None:
            self.sim.close()

        super().closeEvent(event)

# %% Application startup

# ============================================================
# Application startup
# ============================================================

def main():
    pg.setConfigOptions(antialias=False, useOpenGL=False)

    app = QtWidgets.QApplication.instance()

    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    theme.apply_theme(app)

    editor = EditorWindow()
    editor.show()

    app.exec_()


if __name__ == "__main__":
    main()
