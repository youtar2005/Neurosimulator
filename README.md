# Hodgkin-Huxley Network Simulator

An interactive tool for building small networks of Hodgkin-Huxley neurons,
connecting them with chemical synapses, gap junctions and "veto" gates, and
watching them run live on a multi-trace oscilloscope with spike raster and
rhythm analysis.

```
python network_editor.py
```

Requirements: Python 3.9+, `numpy`, `numba`, `PyQt5`, `pyqtgraph`.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [File reference](#3-file-reference)
   - [simulation.py](#31-simulationpy)
   - [analytics.py](#32-analyticspy)
   - [theme.py](#33-themepy)
   - [simulation_visualization.py](#34-simulation_visualizationpy)
   - [network_editor.py](#35-network_editorpy)
   - [palette_check.py](#36-palette_checkpy)
4. [Settings reference](#4-settings-reference-where-each-constant-lives)
5. [Where to make changes (recipes)](#5-where-to-make-changes-recipes)
6. [Conventions, units and gotchas](#6-conventions-units-and-gotchas)

---

## 1. Quick start

1. `python network_editor.py` opens the **network editor** with a demo
   network: `A -> B -> C -| A` (excitatory, excitatory, inhibitory).
2. Pick a tool from the toolbar (Add Neuron, Excitatory / Inhibitory
   Synapse, Gap Junction, Veto Synapse) and click or drag on the canvas.
3. Click a neuron or connection and edit its properties in the right-hand
   panel (stimulus, bias current, conductances, delays, plasticity, ...).
4. Press **Simulate** (`Ctrl+R`). The first run compiles the numba kernel
   (a few seconds; cached afterwards), then the **simulation window** opens.
5. In the simulation window, hold a neuron's button (or key `1`-`9`,
   `Space` = first neuron) to inject its stimulus current.
6. Save / open networks as JSON with the toolbar or `Ctrl+S` / `Ctrl+O`.

---

## 2. Architecture at a glance

```
                       theme.py  (colours, Qt palette + stylesheet)
                          ^                ^
                          |                |
 network_editor.py -------+---> simulation_visualization.py ----> analytics.py
   (editor + main)        |        (oscilloscope window)          (pure NumPy)
        |                 |                 |
        |                 |                 | uses buffers + worker
        v                 |                 v
        +-----------> simulation.py  (engine, no GUI imports)
```

Dependency rules (keep them one-directional):

| File | May import | Must NOT import |
|---|---|---|
| `analytics.py` | `numpy`, `math` | anything GUI, `simulation` |
| `simulation.py` | `numpy`, `numba`, stdlib | anything GUI |
| `theme.py` | `PyQt5`, `pyqtgraph` | the other project files |
| `simulation_visualization.py` | `simulation`, `analytics`, `theme`, Qt | `network_editor` |
| `network_editor.py` | everything above | - |

**Run-time data flow**

```
GUI thread                                   Worker thread (SimulationWorker)
----------                                   ---------------------------------
EditorScene.build_network() -> Network
Network.warmup()  (JIT compile)
SimulationWindow(net, colors) ------------->  loop:
                                                out = network.step_chunk(100)
                                                spike detection -> SpikeLog
                                                PlotBuffer.append_batch (per neuron)
                                                ConnectionPlotBuffer.append_batch
                                                WeightHistory.append (every 10 chunks)
                                                sleep to keep real-time x speed
QTimer (33 ms)  -> update_plots()   <---- snapshot() of buffers (locks)
QTimer (250 ms) -> update_raster()  <---- SpikeLog.snapshot() + estimate_period()
```

One *chunk* = `CHUNK_STEPS` (100) steps of `dt` (0.01 ms) = 1 ms of model time.
The kernel returns an array `out[neuron, channel, step]`, where channels are
named by `CHANNELS`.

---

## 3. File reference

### 3.1 `simulation.py`

The numerical core. No GUI imports; it can be used headless (see
[recipe 5.2](#52-run-a-simulation-headless-and-analyse-it)).

#### Module constants

| Name | Value | Meaning |
|---|---|---|
| `dt` | `0.01` | Integration timestep (ms) |
| `CHUNK_STEPS` | `100` | Steps per compiled kernel call |
| `SPIKE_THRESHOLD` | `0.0` | Upward crossing of this voltage (mV) = a spike |
| `MAX_DELAY_MS` | `50.0` | Longest supported transmission delay |
| `WEIGHT_SAMPLE_CHUNKS` | `10` | Record synaptic weights every N chunks |
| `WEIGHT_CAPACITY` | `3000` | Number of weight samples kept |
| `WEIGHT_SPAN_S` | derived | Time span of the weight plot (s) |
| `SYN_VT`, `SYN_KP` | `2.0`, `5.0` | Transmitter release sigmoid midpoint / slope (mV) |
| `AMPA`, `GABA_A`, `VETO` | dicts | Default kinetics: `E_rev`, `alpha`, `beta` (veto has no `E_rev`) |
| `CHANNELS` | tuple | Row names of the kernel output: `("t","V","Na","K","Cl","pump","syn","stim")` |

#### Gating kinetics

| Signature | What it does |
|---|---|
| `vtrap(x, y)` *(numba)* | Numerically stable `x / (exp(x/y) - 1)`; returns `y` when `x/y` is ~0 (removes the 0/0 singularity in the alpha rates). |
| `hh_rates(V_m)` | Classical HH rate constants `(alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n)` for membrane voltage `V_m` (mV; internally shifted so rest is 0). |
| `steady_state_gates(V_m)` | Steady-state `(m_inf, h_inf, n_inf)` at `V_m`; used to initialise a neuron at rest. |

> The kernel contains its own inline copy of the rate equations (for speed).
> If you change the kinetics, change **both** `hh_rates` and the kernel.

#### Neuron model

**`NeuronParams`** *(dataclass)* - biophysical parameters of one membrane patch.

| Field | Default | Unit |
|---|---|---|
| `C_m` | 1.0 | uF/cm^2 |
| `g_Na_max`, `g_K_max`, `g_Cl` | 120, 36, 0.3 | mS/cm^2 |
| `E_Na`, `E_K`, `E_Cl` | 60, -90, -70 | mV |
| `Na_i`, `Na_o`, `K_i`, `K_o`, `K_Na`, `K_K` | pump concentrations / affinities | mM |
| `I_pump_max` | 10.0 | uA/cm^2 |

| Method | What it does |
|---|---|
| `pump_current(self)` | Outward electrogenic Na+/K+ pump current: `I_pump_max * (Na_i/(Na_i+K_Na))^3 * (K_o/(K_o+K_K))^2`. |
| `as_array(self)` | Packs `[C_m, g_Na_max, g_K_max, g_Cl, E_Na, E_K, E_Cl, I_pump]` into a float64 vector. **This order defines the column indices read by the kernel.** |

**`Neuron`**

| Signature | What it does |
|---|---|
| `Neuron(name="neuron", V0=-65.0, params=None, stim_amplitude=15.0, bias_current=0.0)` | One HH neuron. Holds `params`, state vector `state = [V, m, h, n]`, `stim_amplitude` (applied while `stim_active`), `stim_active` (toggled by the GUI) and a constant `bias_current`. |
| `V` *(property)* | Current membrane voltage `state[0]`. |
| `external_current(self)` | `bias_current` plus `stim_amplitude` if `stim_active`. Read by the network each chunk. |

#### Connections (plain dataclasses)

| Class | Fields | Meaning |
|---|---|---|
| `Synapse` | `pre, post, g_max, E_rev, alpha, beta, delay=0.0`, plasticity: `plastic, A_plus, A_minus, tau_plus, tau_minus, w_max, tau_decay`, live `weight` | Kinetic chemical synapse: `ds/dt = alpha*T(V_pre)*(1-s) - beta*s`, `I = g*s*(V_post - E_rev)`. Optional pair-based STDP. |
| `GapJunction` | `a, b, g` | Ohmic electrical coupling `g*(V_a - V_b)`. |
| `VetoSynapse` | `pre, post, strength, alpha, beta, gate_ext, delay` | Multiplicative gate: scales the postsynaptic synaptic input (and optionally external drive) by `1 - strength*s`. |

#### The compiled kernel

```python
simulate_network_chunk(
    states, params, i_ext,
    syn_pre, syn_post, syn_g, syn_E, syn_alpha, syn_beta, syn_s,
    syn_g0, syn_plast, syn_trace, syn_delay,
    gap_a, gap_b, gap_g,
    veto_pre, veto_post, veto_strength, veto_ext,
    veto_alpha, veto_beta, veto_s, veto_delay,
    vhist, hist_pos, syn_v_response, syn_v_out,
    n_steps, t0, dt, out)          # numba @njit, no return value
```

Advances the whole network `n_steps` steps **in place**. Per step it:

1. writes start-of-step voltages into the ring buffer `vhist` (for delays);
2. computes chemical synaptic currents and updates each `syn_s`;
3. adds gap-junction currents;
4. computes veto gates (`gate_syn`, `gate_ext`) and scales `i_syn`;
5. updates a per-connection EPSP/IPSP *estimate* (`syn_v_response`,
   written to `syn_v_out[k, step]`) - a passive-membrane approximation, not
   an exact decomposition of the HH voltage;
6. updates every neuron: forward-Euler for `V`, exact exponential (Rush-Larsen)
   updates for `m`, `h`, `n`; detects spikes; writes `out[i, :, step]`;
7. applies pair-based STDP to plastic synapses (soft bounds, optional decay
   to `syn_g0`);
8. advances the ring-buffer position.

| Array | Shape | Contents |
|---|---|---|
| `states` | `(N, 4)` | `[V, m, h, n]` per neuron (mutated) |
| `params` | `(N, 8)` | rows from `NeuronParams.as_array()` |
| `i_ext` | `(N,)` | external current per neuron |
| `syn_*` | `(S,)` | chemical synapse arrays; `syn_g` is the live weight |
| `syn_plast` | `(S, 7)` | `[enabled, A+, A-, tau+, tau-, w_max, tau_decay]` |
| `syn_trace` | `(S, 2)` | `[pre trace, post trace]` |
| `gap_*` | `(G,)` | gap-junction endpoints and conductances |
| `veto_*` | `(K,)` | veto arrays |
| `syn_delay`, `veto_delay` | `(S,)`, `(K,)` int64 | delays in steps (< history length) |
| `vhist` | `(N, L)` | voltage ring buffer, `L = MAX_DELAY_MS/dt + 1` |
| `out` | `(N, len(CHANNELS), n_steps)` | recorded channels |

#### `Network`

Couples neurons and advances them together. **Topology is frozen on first
run**; parameters (conductances, weights' baselines, delays, stimulus, ...)
stay live-tunable because `_sync_parameters` re-reads them every chunk.

| Signature | What it does |
|---|---|
| `Network(dt=0.01)` | Empty network. |
| `add(neuron)` | Add a neuron; returns it. |
| `connect_chemical(pre, post, g_max, E_rev, alpha, beta, delay=0.0, **plasticity)` | Add a `Synapse`; `**plasticity` accepts `plastic, A_plus, A_minus, tau_plus, tau_minus, w_max, tau_decay`. |
| `connect_excitatory(pre, post, g_max=0.5, **plasticity)` | Chemical synapse with `AMPA` kinetics. |
| `connect_inhibitory(pre, post, g_max=0.5, **plasticity)` | Chemical synapse with `GABA_A` kinetics. |
| `connect_gap(a, b, g=0.05)` | Add a `GapJunction`. |
| `connect_veto(pre, post, strength=1.0, alpha, beta, gate_ext=False, delay=0.0)` | Add a `VetoSynapse`. |
| `weights()` | Copy of the live chemical-synapse weights (one per synapse). |
| `warmup(n_steps=CHUNK_STEPS)` | Triggers JIT compilation by running the kernel on **copies** of all state, so nothing is learned or advanced. |
| `step_chunk(n_steps)` | Advances the network and returns `out` (shape `(N, len(CHANNELS), n_steps)`). **The returned array is reused on the next call - copy it if you keep it.** Also updates `Synapse.weight`. |
| `t` *(property)* | Model time in ms (`step_count * dt`). |

Internal helpers: `_check_open()` (raises if topology is frozen),
`_check_member(*neurons)`, `_index(neuron)`, `_compile()` (builds all arrays
and freezes topology), `_delay_steps(delay_ms)` (ms -> clamped step count),
`_sync_parameters()` (copies live Python-side values into the arrays),
`_run(...)` (thin wrapper that calls the kernel with the right arguments).

#### Thread-safe buffers

All are written by the worker thread and read by the GUI thread; every
`snapshot()` returns **copies** in chronological order. `version` increases
on every write, so the GUI can skip redraws when nothing changed.

| Class / signature | What it does |
|---|---|
| `PlotBuffer(capacity)` | Per-neuron ring buffer holding every channel in `CHANNELS`. |
| &nbsp;&nbsp;`append_batch(block)` | Append a `(len(CHANNELS), n)` block (e.g. `out[i]`). |
| &nbsp;&nbsp;`snapshot(keys=None)` | Dict `{channel: ndarray}` for the requested channels (all if `None`). |
| `ConnectionPlotBuffer(n_connections, capacity)` | Ring buffer for the per-synapse PSP estimates plus a shared time axis. |
| &nbsp;&nbsp;`append_batch(t, values)` | `t` is `(n,)`, `values` is `(n_connections, n)`. |
| &nbsp;&nbsp;`snapshot()` | `(t, values)`. |
| `WeightHistory(n, capacity)` | Slow ring buffer of `n` plastic-synapse weights. |
| &nbsp;&nbsp;`append(t, w)` | Store one time-point. |
| &nbsp;&nbsp;`snapshot()` | `(t, w)` with `w` shaped `(n, samples)`. |
| `SpikeLog(n, keep_ms=10000.0)` | Recent spike times (ms) per neuron, trimmed to the last `keep_ms`. |
| &nbsp;&nbsp;`add(neuron_idx, spike_times)` | Append spikes (parallel arrays). |
| &nbsp;&nbsp;`snapshot()` | List of `n` float arrays of spike times. |

#### `SimulationWorker(threading.Thread)`

```python
SimulationWorker(network, buffers, stop_event, chunk_steps=CHUNK_STEPS,
                 speed=1.0, weight_hist=None, plastic_idx=(),
                 spike_log=None, connection_buffer=None)
```

`run()` loops until `stop_event` is set: step one chunk, abort with a
message on NaN/inf (`self.error`), detect spikes (including the boundary
between chunks), feed all buffers, and sleep so that model time advances at
`speed` x real time. Change `worker.speed` at any time (the speed selector
does). Any exception is stored in `self.error`.

---

### 3.2 `analytics.py`

Pure NumPy analysis of simulation output. It takes plain arrays (spike times
in ms), so it works live in the GUI and offline in scripts.

| Signature | What it does |
|---|---|
| `estimate_period(times, now, span_ms=PERIOD_SPAN_MS, gap_factor=1.6)` | Cycle period of a spike train from the last `span_ms`. Interspike intervals longer than `gap_factor x` the 10th-percentile ISI separate bursts; the period is the mean burst-to-burst interval. With no such gaps the neuron fires once per cycle and the period is the median ISI. Returns `dict(P, spc, cv)` - period (ms), spikes per cycle, coefficient of variation (0 = perfectly regular) - or `None` if fewer than 4 spikes are in the window. |
| `predict_loop_period(I, d_ab, d_bc, d_ca)` | Analytic period prediction for the `A -> B -> C -| A` loop (fitted constants `LOOP_*` and derivation are in the comment block above it). Returns `(period_ms, spikes_per_cycle)` or `None` outside the validated drive range `LOOP_I_MIN < I < LOOP_I_MAX`. |

Constant: `PERIOD_SPAN_MS = 3000.0` - spans used by the live readout.

`analytics.py` is where to add new analyses; see
[recipe 5.1](#51-add-a-new-analysis-to-analyticspy).

---

### 3.3 `theme.py`

The dark "midnight indigo" look: blue-black surfaces, a blue-to-purple accent
gradient for selected / hovered / pressed states, and a measured categorical
palette for data.

#### Colour constants

| Group | Names | Used for |
|---|---|---|
| Surfaces | `BG`, `PANEL`, `SURFACE`, `HOVER`, `PRESSED`, `BORDER`, `BORDER_LIGHT` | Windows, toolbars, inputs, buttons, borders |
| Text | `TEXT`, `MUTED` | Normal / secondary text |
| Accents | `BLUE`, `BLUE_DIM`, `PURPLE`, `PURPLE_DIM`, `PURPLE_LIGHT`, `HIGHLIGHT` | Active tool gradient, hover borders, group titles, text selection |
| Status | `ERROR` | Error text |
| Editor canvas | `CANVAS_BG`, `NEURON_OUTLINE`, `SELECTION`, `EDGE_LABEL` | Canvas background, unselected neuron outline, selection outline, edge value labels |
| Plots | `PLOT_BG`, `PLOT_FG` | Plot background; axes/labels/grid |
| Data | `NEURON_COLORS`, `EDGE_COLORS`, `CURRENT_COLORS`, `THRESHOLD_COLOR` | Neuron/voltage/raster/PSP/weight traces; edge types; membrane-current traces; the -40 mV dashed line |

**Data palette design** (verified by `palette_check.py`):

| Set | Count | Min pairwise dE (normal vision) | Min dE under colour-blindness | Min contrast vs. background |
|---|---|---|---|---|
| `NEURON_COLORS` | 10 | 17.3 | 7.1 - 8.0 | 5.6 : 1 |
| first 6 / first 3 | 6 / 3 | 23.6 / 23.6 | 9.6+ / 10.9+ | 5.6 : 1 |
| `CURRENT_COLORS` + threshold | 7 | 23.4 | 9.6+ | 5.6 : 1 |
| `EDGE_COLORS` | 4 | 32.7 | 8.5+ | 6.3 : 1 |

(dE = CIEDE2000; below ~10 colours are easy to confuse, above ~20 they are
clearly different.) The *order* of `NEURON_COLORS` is deliberate: any prefix
is the most distinguishable set of that size, so small networks get the
best colours. Edge types additionally differ by arrowhead shape and line
style, so they never rely on colour alone.

#### Functions

| Signature | What it does |
|---|---|
| `neuron_color(i)` | `QColor` of the i-th neuron colour (wraps around after 10). Used for new neurons in the editor. |
| `series_color(i)` | `QColor` for the i-th generic plot series (same palette); used for PSP and weight traces. |
| `contrast_text(color)` | Black or white (`QColor`), whichever is more readable on `color` (WCAG luminance). Used for neuron labels. |
| `stylesheet()` | Returns the Qt stylesheet string (toolbar, buttons, group boxes, scrollbars, tooltips, status bar). |
| `apply_theme(app)` | Call once after creating the `QApplication`: sets the Fusion style, the dark `QPalette`, the stylesheet, and pyqtgraph's `background` / `foreground` defaults (these must be set **before** any `PlotWidget` is created). |

Private: `_luminance(color)` (WCAG relative luminance), `_build_palette()`
(the `QPalette`; spin boxes, combo boxes and check boxes are themed via the
palette rather than the stylesheet because styling them via CSS can remove
their arrows).

---

### 3.4 `simulation_visualization.py`

The live oscilloscope for a running `Network`. It knows nothing about the
editor.

#### Constants

| Name | Value | Meaning |
|---|---|---|
| `window` | `50.0` | Visible time window of the scrolling plots (ms) |
| `GUI_INTERVAL_MS` | `33` | Plot refresh interval (~30 FPS) |
| `YRANGE_EVERY` | `10` | Re-fit the Y range of the current/PSP plots every N frames |
| `MAX_PLOT_POINTS` | `1000` | Max points drawn per curve after decimation |
| `V_Na_threshold` | `-40.0` | Dashed reference line on the voltage plot (mV) |
| `RASTER_SPAN_MS` | `2000.0` | Time span shown in the raster |
| `RASTER_UPDATE_MS` | `250` | Raster / period readout refresh interval |
| `SPEEDS` | tuple | `(label, factor)` choices in the Speed selector |

#### Functions

| Signature | What it does |
|---|---|
| `make_bins(t, dt_, max_points=MAX_PLOT_POINTS)` | Chooses min/max decimation bins for a time array, or `None` if no decimation is needed. Bin edges are aligned to absolute sample indices, so the plot does not shimmer as it scrolls. Returns `(start, k, m)`. |
| `decimate(t, y, bins)` | Reduces `(t, y)` to at most `MAX_PLOT_POINTS` points, keeping each bin's minimum and maximum (in time order) so spikes are never dropped. |

#### `SimulationWindow(network, colors, window_ms=window)`

Owns the buffers, the `SimulationWorker`, two timers and all plots. Layout:
left column of plots, middle column raster + period readout, right column
controls.

| Plot (attribute) | Shows | Data source |
|---|---|---|
| `voltage_plot` | Membrane voltage of all neurons + threshold line | `PlotBuffer` `"V"` |
| `connection_plot` | Estimated PSP per chemical synapse (EPSP/IPSP) | `ConnectionPlotBuffer` |
| `current_plot` | Na, K, Cl, pump, synaptic, stimulus currents of the *selected* neuron | `PlotBuffer` channels |
| `weight_plot` *(only if any synapse is plastic)* | STDP weights over time | `WeightHistory` |
| `raster_plot` | Spike raster, last 2 s | `SpikeLog` |

The voltage, PSP and current plots share an X axis (`setXLink`).

| Method | What it does |
|---|---|
| `on_select(index)` | Changes which neuron's currents are plotted. |
| `on_speed(index)` | Sets `worker.speed` from `SPEEDS`. |
| `update_plots()` | Timer slot; wraps `_update_plots()` and `_update_weights()` and stops the timer on error. |
| `_update_plots()` | Redraws voltage, PSP and current plots from buffer snapshots (skips if the buffer `version` did not change), re-fits Y ranges, scrolls the X window, shows worker errors. |
| `_update_weights()` | Redraws the weight plot. |
| `update_raster()` / `_update_raster()` | Redraws the raster and recomputes the period readout with `estimate_period`. |
| `eventFilter(obj, event)` | App-wide key filter: while this window is active, keys `1`-`9` / `Space` hold the matching neuron's stimulus; deactivating the window releases all stimuli. |
| `closeEvent(event)` | Stops timers and worker, removes the event filter. |

Key attributes you may want in extensions: `self.neurons`, `self.buffers`
(one `PlotBuffer` per neuron), `self.connection_buffer`, `self.spike_log`,
`self.selected`, `self.frame`, `self.worker`.

---

### 3.5 `network_editor.py`

The graphical network editor **and the program entry point**.

#### Module-level data

| Name | Meaning |
|---|---|
| `EDGE_COLORS` | `QColor` per connection kind (built from `theme.EDGE_COLORS`) |
| `EDGE_LABELS` | Human-readable name per kind |
| `EDGE_DEFAULTS` | Default `g`, `E_rev`, `alpha`, `beta` per kind (`exc`, `inh`, `gap`, `veto`) |
| `PLASTICITY_DEFAULTS` | Default STDP fields for every edge |
| `PLASTIC_FIELDS`, `NEURON_FIELDS`, `CHEMICAL_FIELDS` | Spin-box specs: `(label, attribute path, min, max, decimals, step)`; the properties panel is generated from these |
| `MODE_HINTS` | Status-bar hint per tool mode |

Functions: `excepthook(exc_type, exc, tb)` prints uncaught exceptions
instead of silently aborting (installed as `sys.excepthook`);
`get_path(obj, path)` / `set_path(obj, path, value)` read/write dotted
attributes such as `"params.g_Na_max"`; `make_spin(lo, hi, decimals, step)`
creates a configured `QDoubleSpinBox`; `main()` creates the `QApplication`,
applies the theme and shows the editor.

#### Graphics items

| Class | What it does |
|---|---|
| `NeuronItem(name, color, x, y)` | A draggable, selectable circle. Holds the editable model data (`params`, `stim_amplitude`, `bias_current`) and its `edges`. Methods: `set_name(name)`, `_center_label()`, `paint(...)` (hides Qt's dashed selection box), `itemChange(change, value)` (re-routes attached edges on move; changes the outline when selected). |
| `EdgeItem(kind, src, dst)` | A curved arrow for one connection. Holds `g`, `E_rev`, `alpha`, `beta`, `delay`, `gate_ext` and plasticity fields. `refresh()` recomputes the curve, arrowhead (triangle = excitatory, dot = inhibitory, block = veto, end dots = gap junction), hit-shape and label position; `plasticity()` returns the STDP fields as a dict; `boundingRect()`, `shape()`, `paint()` implement drawing. |

#### `EditorScene(QGraphicsScene)`

Signal: `status(str)` - messages for the status bar.

| Method | What it does |
|---|---|
| `unique_name()` | Next free `N1`, `N2`, ... |
| `add_neuron(x, y, name=None, color=None)` | Add a neuron (palette colour if none given). |
| `add_edge(kind, src, dst, **overrides)` | Add a connection; returns `None` for duplicates (and reversed gap junctions). |
| `remove_edge(edge)`, `remove_neuron(neuron)` | Remove an item (a neuron removes its edges). |
| `delete_selected()` | Delete the current selection. |
| `clear_all()` | Reset the scene. |
| `neuron_at(pos)` | Neuron under a scene position, or `None`. |
| `build_network()` | Converts the scene to a runnable `(Network, colors)`; this is the bridge from the editor to the simulation. |
| `to_dict()` / `from_dict(data)` | Serialize / restore the network (versioned JSON structure; unknown keys are ignored on load). |
| `mousePressEvent / mouseMoveEvent / mouseReleaseEvent` | Implement the tools: click to place a neuron; drag with a rubber-band line to create a connection. |

#### `PropertiesPanel(scene)`

A `QStackedWidget`: page 0 = help, page 1 = neuron form, page 2 = connection
form. `on_selection()` fills the right page when exactly one item is
selected; `_on_name()`, `_on_neuron_value(path, value)` and
`_on_edge_value(attr, value)` write edits straight back to the item (guarded
by `_loading` so populating the form does not trigger edits).

#### `EditorWindow`

| Method | What it does |
|---|---|
| `_build_toolbar()` | Creates mode buttons (exclusive), Delete, New / Open / Save, and Simulate. Stores the bar as `self.toolbar`. |
| `_fit_toolbar_width()` | Makes the window at least as wide as the toolbar needs so **no button is hidden behind the `>>` overflow menu**; called after building and again in `showEvent`, once stylesheet-dependent button widths are final. |
| `showEvent(event)` | Re-measures the toolbar just before the window appears. |
| `set_mode(mode)` | Switches tool; updates cursor and hint. |
| `load_demo()` | Builds the default `A -> B -> C -| A` network. |
| `new_network()`, `save_network()`, `open_network()` | File operations (JSON). |
| `simulate()` | Builds the network, runs `warmup()` under a wait cursor, opens a `SimulationWindow` (closing any previous one). |
| `closeEvent(event)` | Closes the simulation window too. |

---

### 3.6 `palette_check.py`

Developer tool - run after changing any colour in `theme.py`:

```
python palette_check.py
```

| Signature | What it does |
|---|---|
| `hex_to_linear(hex_color)` | `'#rrggbb'` -> linear-light RGB. |
| `to_lab(linear_rgb)` | Linear RGB -> CIELAB. |
| `lab_of(hex_color, vision=None)` | CIELAB of a colour, optionally as seen with `"protan"`, `"deutan"` or `"tritan"` colour-vision deficiency. |
| `delta_e_2000(lab1, lab2)` | CIEDE2000 colour difference. |
| `contrast_ratio(hex_a, hex_b)` | WCAG contrast ratio. |
| `closest_pair(colors, vision=None)` | `(smallest dE, (colour_a, colour_b))` in a set. |
| `report(name, colors, backgrounds)` | Prints min dE (normal and CVD) and min contrast for a set. |
| `main()` | Reports on neuron colours (all, first 3, first 6), currents + threshold, and edges. |

---

## 4. Settings reference (where each constant lives)

| I want to change... | Edit | Name |
|---|---|---|
| Integration timestep | `simulation.py` | `dt` |
| Chunk size (GUI/thread granularity) | `simulation.py` | `CHUNK_STEPS` |
| What counts as a spike | `simulation.py` | `SPIKE_THRESHOLD` |
| Max transmission delay | `simulation.py` | `MAX_DELAY_MS` |
| Weight-history length / rate | `simulation.py` | `WEIGHT_CAPACITY`, `WEIGHT_SAMPLE_CHUNKS` |
| Default synapse kinetics | `simulation.py` | `AMPA`, `GABA_A`, `VETO` |
| Visible time window | `simulation_visualization.py` | `window` |
| Frame rate / decimation | `simulation_visualization.py` | `GUI_INTERVAL_MS`, `MAX_PLOT_POINTS` |
| Raster span / refresh | `simulation_visualization.py` | `RASTER_SPAN_MS`, `RASTER_UPDATE_MS` |
| Speed choices | `simulation_visualization.py` | `SPEEDS` |
| Period-estimate window | `analytics.py` | `PERIOD_SPAN_MS` |
| Any colour | `theme.py` | see [3.3](#33-themepy) |
| Default edge values | `network_editor.py` | `EDGE_DEFAULTS`, `PLASTICITY_DEFAULTS` |
| Editable fields in the panel | `network_editor.py` | `NEURON_FIELDS`, `CHEMICAL_FIELDS`, `PLASTIC_FIELDS` |
| Starting network | `network_editor.py` | `EditorWindow.load_demo` |

---

## 5. Where to make changes (recipes)

### 5.1 Add a new analysis to `analytics.py`

Analyses are plain functions of NumPy data. Put each one in its own section
of `analytics.py`.

```python
# analytics.py  -- new section

# %% Firing statistics

def firing_rate(times, now, span_ms=1000.0):
    """Mean firing rate (Hz) over the last span_ms."""
    t = np.asarray(times, dtype=float)
    return 1000.0 * np.count_nonzero(t > now - span_ms) / span_ms


def isi_cv(times, now, span_ms=PERIOD_SPAN_MS):
    """Coefficient of variation of interspike intervals (None if < 3 spikes)."""
    t = np.asarray(times, dtype=float)
    t = t[t > now - span_ms]
    if len(t) < 3:
        return None
    isi = np.diff(t)
    return float(isi.std() / isi.mean())
```

**Show it in the live readout** - the readout text is built in
`SimulationWindow._update_raster` (`simulation_visualization.py`):

```python
from analytics import estimate_period, firing_rate      # extend the import

# inside _update_raster, in the loop over neurons:
rate = firing_rate(ts, now)
lines.append(f"{name}: {est['P']:.1f} ms, {est['spc']:.0f} spike(s)/cycle, {rate:.0f} Hz")
```

**Conventions to follow:** times are in **ms**; accept `times` and `now`
as arguments instead of reading global state; return `None` when there is
not enough data; keep the file free of GUI and `simulation` imports.

### 5.2 Run a simulation headless and analyse it

Because `simulation.py` and `analytics.py` have no GUI dependencies, you can
script experiments:

```python
import numpy as np
from simulation import Network, Neuron, CHUNK_STEPS, SPIKE_THRESHOLD, dt
from analytics import estimate_period

net = Network(dt=dt)
a = net.add(Neuron("A", bias_current=26.0))
b = net.add(Neuron("B"))
c = net.add(Neuron("C"))
net.connect_excitatory(a, b)
net.connect_excitatory(b, c)
net.connect_veto(c, a, strength=1.0, beta=0.02, gate_ext=True, delay=5.0)

net.warmup()

spikes = [[] for _ in net.neurons]
prev = np.array([n.V for n in net.neurons])

for _ in range(5000):                       # 5000 chunks = 5 s of model time
    out = net.step_chunk(CHUNK_STEPS)       # (N, len(CHANNELS), 100); reused!
    t, V = out[:, 0, :], out[:, 1, :]
    ext = np.concatenate((prev[:, None], V), axis=1)
    hit = (ext[:, :-1] < SPIKE_THRESHOLD) & (ext[:, 1:] >= SPIKE_THRESHOLD)
    for i, k in zip(*np.nonzero(hit)):
        spikes[i].append(t[i, k])
    prev = V[:, -1].copy()

for n, s in zip(net.neurons, spikes):
    print(n.name, estimate_period(s, net.t))
```

To change a parameter between runs (a parameter sweep) just set it on the
neuron or connection objects (`a.bias_current = 28.0`,
`net.vetos[0].delay = 8.0`); `step_chunk` re-reads them every chunk.
Topology (adding neurons / connections) is frozen after the first run.

### 5.3 Add a new plot to the simulation window

*Example: a phase-plane plot (voltage vs. Na current) of the selected neuron,
using data that is already buffered.*

1. **Create the widget** in `SimulationWindow.__init__`, next to the other
   plots (before the "Plots never see the mouse" loop):

   ```python
   self.phase_plot = pg.PlotWidget(title="Phase plane (selected neuron)")
   self.phase_plot.setLabel("left", "I_Na", units="uA/cm²")
   self.phase_plot.setLabel("bottom", "Voltage", units="mV")
   self.phase_plot.showGrid(x=True, y=True, alpha=0.3)
   self.phase_line = self.phase_plot.plot(
       pen=pg.mkPen(theme.series_color(0), width=1.5))
   plots.addWidget(self.phase_plot, stretch=1)
   # do NOT setXLink: its x axis is voltage, not time
   ```

2. **Include it in the mouse/keyboard lock-down list** (the tuple of plots in
   the `for plot in [...]` loop) so it behaves like the others.

3. **Update it** in `_update_plots`, right after the `selected = ...snapshot()`
   line (the data is already there):

   ```python
   if len(t_sel) > 0:
       self.phase_line.setData(selected["V"], selected["Na"],
                               skipFiniteCheck=True)
   ```

4. Use `theme.series_color(i)` (or a colour from `theme.py`) for pens so the
   plot matches the palette; add new colours to `theme.py` and run
   `palette_check.py`.

Plots driven by the **spike log** (histograms, firing-rate curves,
cross-correlations) should be refreshed from `_update_raster` (250 ms
timer) instead, using `self.spike_log.snapshot()`. Anything that needs heavy
computation belongs in `analytics.py`, with the window only calling it.

### 5.4 Record a new quantity (new channel), e.g. the `m` gate

1. `simulation.py` - append the name to `CHANNELS`:
   `CHANNELS = ("t", "V", "Na", "K", "Cl", "pump", "syn", "stim", "m")`
2. `simulation.py` - in `simulate_network_chunk`, write it after the update
   (the index is the position in `CHANNELS`):
   `out[i, 8, step] = m`
3. That is all the plumbing: `Network` allocates `out` from
   `len(CHANNELS)`, `PlotBuffer` allocates a row per channel, and the
   selected neuron's snapshot (`self.buffers[self.selected].snapshot()`)
   contains `"m"`. Plot it as in recipe 5.3. (Gating variables are
   unitless, so give them their own plot rather than adding them to the
   current plot, which has units of uA/cm^2.)

If you add the channel to the *current plot*, also add it to `specs` in
`SimulationWindow.__init__` and a colour to `theme.CURRENT_COLORS`.

### 5.5 Add a new per-neuron parameter

*Example: an extra potassium leak conductance `g_KL`.*

1. `NeuronParams` - add the field (`g_KL: float = 0.0`).
2. `NeuronParams.as_array` - append it to the array (it becomes column 8)
   and widen `self._params = np.empty((N, 8))` to 9 in `Network._compile`.
3. `simulate_network_chunk` - read `g_KL = params[i, 8]` and add
   `g_KL * (V - E_K)` to the current balance.
4. `network_editor.py` - add one row to `NEURON_FIELDS`:
   `("g_KL (mS/cm²)", "params.g_KL", 0, 10, 3, 0.05)`.

Saving and loading need no change: parameters are saved with `asdict` and
unknown / missing keys are handled on load, so old files still open.

### 5.6 Add a new connection type

Touch these places, in order:

1. `simulation.py` - a new dataclass (like `VetoSynapse`), a
   `Network.connect_<kind>` method, its arrays in `_compile`, their
   per-chunk refresh in `_sync_parameters`, the new arguments in
   `simulate_network_chunk`, in `Network._run` **and** in the call inside
   `warmup` (they must match positionally).
2. `network_editor.py` - an entry in `EDGE_DEFAULTS`, `EDGE_LABELS`,
   `MODE_HINTS`, and the `modes` tuple in `_build_toolbar`; an arrowhead in
   `EdgeItem.refresh`; a branch in `EditorScene.build_network` (and the
   `("exc", "inh", "gap", "veto")` tuple in `mousePressEvent`); visibility
   rules in `PropertiesPanel.on_selection`.
3. `theme.py` - a colour in `EDGE_COLORS`; run `palette_check.py`.

Saving / loading already stores `kind` and the generic edge fields.

### 5.7 Change the look (colours, fonts, spacing)

- **Colours:** edit the constants in `theme.py` only. To add a colour to the
  data palette, append it to `NEURON_COLORS` and run
  `python palette_check.py`; keep minimum dE (normal vision) above ~15 and
  contrast against `PLOT_BG` above ~4.5. Put the most important colours
  first.
- **Widget styling:** edit the CSS in `_STYLESHEET` (`theme.py`); widgets not
  mentioned there are themed through the palette in `_build_palette()`.
- **Line widths** are in `simulation_visualization.py` where each
  `pg.mkPen(..., width=...)` is created.
- Colours of neurons are **stored in saved JSON files**; files saved before a
  palette change keep their old colours.

### 5.8 Add a toolbar button

In `EditorWindow._build_toolbar`:

```python
export = QtWidgets.QAction("Export CSV...", self)
export.triggered.connect(self.export_csv)       # your new method
bar.addAction(export)
```

No width bookkeeping is needed: `_fit_toolbar_width` measures the toolbar and
widens the window so the new button is visible.

### 5.9 Add a new editable field to a neuron or connection

Add a tuple to `NEURON_FIELDS` / `CHEMICAL_FIELDS` / `PLASTIC_FIELDS` in
`network_editor.py`. The panel creates the spin box, loads it on selection
and writes edits back automatically. If the value must reach the kernel, also
follow recipe 5.5 / 5.6.

---

## 6. Conventions, units and gotchas

- **Units:** time in ms, voltage in mV, conductance in mS/cm^2, current in
  uA/cm^2, capacitance in uF/cm^2, rates (`alpha`, `beta`) in 1/ms.
- **Kernel argument order is positional.** `simulate_network_chunk`,
  `Network._run` and the call in `Network.warmup` must agree exactly.
- **Rates are duplicated** in `hh_rates` (Python) and inline in the kernel;
  edit both when changing kinetics.
- **Topology is frozen** after the first `warmup()` / `step_chunk()`;
  parameters are live. To change topology, build a new `Network` (the
  editor does this on every Simulate).
- **`step_chunk` returns a reused array** - copy what you keep.
- **Thread safety:** the worker thread writes the buffers, the GUI thread
  only reads via `snapshot()`. Don't touch buffer internals from the GUI.
  Stimulus flags (`stim_active`) are simple attribute writes from the GUI.
- **numba cache:** kernels use `cache=True`. If a change to the kernel does
  not seem to take effect, delete the `__pycache__` folder.
- **PSP traces are estimates:** the "Estimated Postsynaptic Potential" plot
  treats each synapse as an independent input to a passive membrane; it is a
  visual aid, not an exact decomposition of the nonlinear HH voltage.
- **Loop-period prediction** (`predict_loop_period`) is only valid for the
  fitted setup described in `analytics.py` (default neurons, AMPA
  synapses, a veto with `beta = 0.02` and "also veto external drive", drive
  `24.5 < I < 32`).
- **Divergence:** if the simulation produces NaN/inf the worker stops and
  the window shows "Diverged: ... Try weaker coupling."
- **Colour-blind safety:** the palette is checked, but connection kinds are
  also distinguished by shape; keep that property when adding new kinds.
