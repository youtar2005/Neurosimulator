"""
simulation.py - the numerical core (no GUI imports).

Contents
    * simulation settings
    * Hodgkin-Huxley gating kinetics
    * Neuron / NeuronParams
    * connection types (chemical, gap, veto)
    * the numba-compiled network kernel
    * Network (couples neurons and advances them)
    * thread-safe data buffers (voltage, connection, weights, spikes)
    * SimulationWorker (background thread that runs the Network)
"""

import math
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import numpy as np
from numba import njit

# %% Simulation settings

# ============================================================
# Simulation settings
# ============================================================

dt = 0.01                   # Simulation timestep (ms)

CHUNK_STEPS = 100           # Steps per compiled call (100 * 0.01 ms = 1 ms)

SPIKE_THRESHOLD = 0.0       # mV, upward crossing counts as a spike

MAX_DELAY_MS = 50.0         # Longest transmission delay supported (ms)

WEIGHT_SAMPLE_CHUNKS = 10   # record weights every 10 chunks (10 ms)
WEIGHT_CAPACITY = 3000      # samples kept
WEIGHT_SPAN_S = (
    WEIGHT_CAPACITY * WEIGHT_SAMPLE_CHUNKS * CHUNK_STEPS * dt / 1000.0
)

# %% Hodgkin-Huxley gating kinetics

# ============================================================
# Hodgkin-Huxley gating kinetics
# ============================================================

@njit(cache=True, nogil=True)
def vtrap(x, y):
    """Numerically stable x / (exp(x/y) - 1)."""
    z = x / y

    if abs(z) < 1e-6:
        return y

    return x / math.expm1(z)


def hh_rates(V_m):
    """Classical HH rates, with voltage referenced to -65 mV."""
    V = V_m + 65.0

    alpha_m = 0.1 * vtrap(25.0 - V, 10.0)
    beta_m = 4.0 * math.exp(-V / 18.0)

    alpha_h = 0.07 * math.exp(-V / 20.0)
    beta_h = 1.0 / (math.exp((30.0 - V) / 10.0) + 1.0)

    alpha_n = 0.01 * vtrap(10.0 - V, 10.0)
    beta_n = 0.125 * math.exp(-V / 80.0)

    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def steady_state_gates(V_m):
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = hh_rates(V_m)

    return (
        alpha_m / (alpha_m + beta_m),
        alpha_h / (alpha_h + beta_h),
        alpha_n / (alpha_n + beta_n),
    )

# %% Neuron entity

# ============================================================
# Neuron entity
# ============================================================

@dataclass
class NeuronParams:
    """Biophysical parameters of one (homogeneous) membrane patch."""

    C_m: float = 1.0             # uF/cm^2

    g_Na_max: float = 120.0      # mS/cm^2
    g_K_max: float = 36.0        # mS/cm^2
    g_Cl: float = 0.3            # mS/cm^2

    E_Na: float = 60.0           # mV
    E_K: float = -90.0           # mV
    E_Cl: float = -70.0          # mV

    # Na+/K+ pump
    Na_i: float = 12.0
    Na_o: float = 145.0
    K_i: float = 140.0
    K_o: float = 4.0
    K_Na: float = 10.0
    K_K: float = 1.5
    I_pump_max: float = 10.0     # uA/cm^2

    def pump_current(self):
        """Outward electrogenic Na+/K+ pump current."""
        na_factor = self.Na_i / (self.Na_i + self.K_Na)
        k_factor = self.K_o / (self.K_o + self.K_K)

        return self.I_pump_max * na_factor**3 * k_factor**2

    def as_array(self):
        # Layout must match the indices used in simulate_network_chunk.
        return np.array([
            self.C_m,
            self.g_Na_max,
            self.g_K_max,
            self.g_Cl,
            self.E_Na,
            self.E_K,
            self.E_Cl,
            self.pump_current(),
        ], dtype=np.float64)


class Neuron:
    """
    A single-compartment Hodgkin-Huxley neuron.

    Owns its parameters, state vector [V, m, h, n] and externally
    applied current. Stepping is done by the Network.
    """

    def __init__(self, name="neuron", V0=-65.0, params=None,
                 stim_amplitude=15.0, bias_current=0.0):
        self.name = name
        self.params = params if params is not None else NeuronParams()

        m, h, n = steady_state_gates(V0)
        self.state = np.array([V0, m, h, n], dtype=np.float64)

        self.stim_amplitude = stim_amplitude   # applied while stim_active
        self.stim_active = False
        self.bias_current = bias_current       # always applied

    @property
    def V(self):
        return self.state[0]

    def external_current(self):
        extra = self.stim_amplitude if self.stim_active else 0.0
        return self.bias_current + extra


# ============================================================
# Connections between neurons
# ============================================================

# Kinetic synapse model (Destexhe et al.):
#   ds/dt = alpha * T(V_pre) * (1 - s) - beta * s
#   T(V_pre) = 1 / (1 + exp(-(V_pre - SYN_VT) / SYN_KP))
#   I = g_max * s * (V_post - E_rev)
SYN_VT = 2.0     # mV
SYN_KP = 5.0     # mV

AMPA = dict(E_rev=0.0, alpha=1.1, beta=0.19)
GABA_A = dict(E_rev=-80.0, alpha=5.0, beta=0.18)
VETO = dict(alpha=10.0, beta=0.1)   # fast onset, ~10 ms decay


@dataclass
class Synapse:
    pre: Neuron
    post: Neuron
    g_max: float                 # mS/cm^2 (initial / baseline weight)
    E_rev: float                 # mV
    alpha: float                 # 1/ms
    beta: float                  # 1/ms
    delay: float = 0.0           # ms, transmission delay

    # Hebbian STDP (off by default)
    plastic: bool = False
    A_plus: float = 0.005        # LTP rate per pairing
    A_minus: float = 0.006       # LTD rate per pairing
    tau_plus: float = 20.0       # ms, presynaptic trace
    tau_minus: float = 20.0      # ms, postsynaptic trace
    w_max: float = 2.0           # mS/cm^2, upper bound
    tau_decay: float = 0.0       # ms, relaxation to g_max (0 = off)

    weight: float = None         # live weight, updated by the Network

    def __post_init__(self):
        self.weight = self.g_max


@dataclass
class GapJunction:
    a: Neuron
    b: Neuron
    g: float                     # mS/cm^2


@dataclass
class VetoSynapse:
    pre: Neuron
    post: Neuron
    strength: float = 1.0        # 0..1, fraction of input removed at s = 1
    alpha: float = 10.0          # 1/ms
    beta: float = 0.1            # 1/ms
    gate_ext: bool = False       # also gate the external drive current
    delay: float = 0.0           # ms, transmission delay

# %% Compiled network kernel

# ============================================================
# Compiled network kernel
# ============================================================

CHANNELS = ("t", "V", "Na", "K", "Cl", "pump", "syn", "stim")


@njit(cache=True, nogil=True)
def simulate_network_chunk(
    states, params, i_ext,
    syn_pre, syn_post, syn_g, syn_E, syn_alpha, syn_beta, syn_s,
    syn_g0, syn_plast, syn_trace, syn_delay,
    gap_a, gap_b, gap_g,
    veto_pre, veto_post, veto_strength, veto_ext,
    veto_alpha, veto_beta, veto_s, veto_delay,
    vhist, hist_pos, syn_v_response, syn_v_out,
    n_steps, t0, dt, out,
):
    """
    syn_g     : (S,)   live weights, updated in place by plasticity
    syn_g0    : (S,)   baseline weights
    syn_plast : (S, 7) [enabled, A+, A-, tau+, tau-, w_max, tau_decay]
    syn_trace : (S, 2) [pre trace, post trace], updated in place
    veto_*    : (K,)   multiplicative veto synapses; veto_s updated in place
    syn_delay, veto_delay : (S,), (K,) int64 delays in steps (< vhist.shape[1])
    vhist     : (N, L) ring buffer of past voltages; hist_pos = [write index]
    """
    N = states.shape[0]
    S = syn_pre.shape[0]
    G = gap_a.shape[0]
    K = veto_pre.shape[0]
    L = vhist.shape[1]
    pos = hist_pos[0]

    i_syn = np.zeros(N)
    syn_i = np.zeros(S)
    gate_syn = np.ones(N)
    gate_ext = np.ones(N)
    spiked = np.zeros(N, dtype=np.bool_)

    for step in range(n_steps):
        # Record start-of-step voltages; connections read them delayed.
        for i in range(N):
            vhist[i, pos] = states[i, 0]
            i_syn[i] = 0.0

        for k in range(S):
            V_pre = vhist[syn_pre[k], (pos - syn_delay[k] + L) % L]
            V_post = states[syn_post[k], 0]

            T = 1.0 / (1.0 + math.exp(-(V_pre - SYN_VT) / SYN_KP))
            s = syn_s[k]

            syn_i[k] = syn_g[k] * s * (V_post - syn_E[k])
            i_syn[syn_post[k]] += syn_i[k]

            syn_s[k] = s + dt * (
                syn_alpha[k] * T * (1.0 - s) - syn_beta[k] * s
            )

        for k in range(G):
            I_gap = gap_g[k] * (states[gap_a[k], 0] - states[gap_b[k], 0])
            i_syn[gap_a[k]] += I_gap
            i_syn[gap_b[k]] -= I_gap

        # ---- Veto gates (multiplicative, from start-of-step voltages).
        for i in range(N):
            gate_syn[i] = 1.0
            gate_ext[i] = 1.0

        for k in range(K):
            V_pre = vhist[veto_pre[k], (pos - veto_delay[k] + L) % L]
            T = 1.0 / (1.0 + math.exp(-(V_pre - SYN_VT) / SYN_KP))
            s = veto_s[k]

            gate = 1.0 - veto_strength[k] * s
            if gate < 0.0:
                gate = 0.0

            gate_syn[veto_post[k]] *= gate
            if veto_ext[k] > 0.5:
                gate_ext[veto_post[k]] *= gate

            veto_s[k] = s + dt * (
                veto_alpha[k] * T * (1.0 - s) - veto_beta[k] * s
            )

        for i in range(N):
            i_syn[i] *= gate_syn[i]

        # Per-connection estimate of the postsynaptic voltage contribution.
        # Treat each synapse's gated current as an independent input to a
        # passive membrane (C_m plus the model's Cl/leak conductance). This
        # makes connection-specific EPSP/IPSP-like traces visible; it is an
        # estimate, not a decomposition of the full nonlinear HH voltage.
        for k in range(S):
            post = syn_post[k]
            C_m = params[post, 0]
            g_leak = params[post, 3]
            I_k = syn_i[k] * gate_syn[post]
            dv = -I_k / C_m - (g_leak / C_m) * syn_v_response[k]
            syn_v_response[k] += dt * dv
            syn_v_out[k, step] = syn_v_response[k]

        for i in range(N):
            V = states[i, 0]
            V_prev = V
            m = states[i, 1]
            h = states[i, 2]
            n = states[i, 3]

            C_m = params[i, 0]
            g_Na_max = params[i, 1]
            g_K_max = params[i, 2]
            g_Cl = params[i, 3]
            E_Na = params[i, 4]
            E_K = params[i, 5]
            E_Cl = params[i, 6]
            I_pump = params[i, 7]
            I_ext = i_ext[i] * gate_ext[i]

            Vr = V + 65.0

            a_m = 0.1 * vtrap(25.0 - Vr, 10.0)
            b_m = 4.0 * math.exp(-Vr / 18.0)
            a_h = 0.07 * math.exp(-Vr / 20.0)
            b_h = 1.0 / (math.exp((30.0 - Vr) / 10.0) + 1.0)
            a_n = 0.01 * vtrap(10.0 - Vr, 10.0)
            b_n = 0.125 * math.exp(-Vr / 80.0)

            I_Na = g_Na_max * m**3 * h * (V - E_Na)
            I_K = g_K_max * n**4 * (V - E_K)
            I_Cl = g_Cl * (V - E_Cl)

            V += dt * (
                I_ext - I_Na - I_K - I_Cl - I_pump - i_syn[i]
            ) / C_m

            r = a_m + b_m
            x_inf = a_m / r
            m = x_inf + (m - x_inf) * math.exp(-r * dt)

            r = a_h + b_h
            x_inf = a_h / r
            h = x_inf + (h - x_inf) * math.exp(-r * dt)

            r = a_n + b_n
            x_inf = a_n / r
            n = x_inf + (n - x_inf) * math.exp(-r * dt)

            states[i, 0] = V
            states[i, 1] = m
            states[i, 2] = h
            states[i, 3] = n

            spiked[i] = (V_prev < SPIKE_THRESHOLD) and (V >= SPIKE_THRESHOLD)

            out[i, 0, step] = t0 + (step + 1) * dt
            out[i, 1, step] = V
            out[i, 2, step] = I_Na
            out[i, 3, step] = I_K
            out[i, 4, step] = I_Cl
            out[i, 5, step] = I_pump
            out[i, 6, step] = i_syn[i]
            out[i, 7, step] = I_ext

        # ---- Hebbian plasticity (pair-based STDP with soft bounds).
        for k in range(S):
            if syn_plast[k, 0] == 0.0:
                continue

            A_plus = syn_plast[k, 1]
            A_minus = syn_plast[k, 2]
            tau_p = syn_plast[k, 3]
            tau_m = syn_plast[k, 4]
            w_max = syn_plast[k, 5]
            tau_d = syn_plast[k, 6]

            x_pre = syn_trace[k, 0] * math.exp(-dt / tau_p)
            x_post = syn_trace[k, 1] * math.exp(-dt / tau_m)
            w = syn_g[k]

            if spiked[syn_pre[k]]:
                w -= A_minus * w * x_post          # post-before-pre: LTD
                x_pre += 1.0

            if spiked[syn_post[k]]:
                w += A_plus * (w_max - w) * x_pre  # pre-before-post: LTP
                x_post += 1.0

            if tau_d > 0.0:
                w += dt * (syn_g0[k] - w) / tau_d

            if w < 0.0:
                w = 0.0
            elif w > w_max:
                w = w_max

            syn_g[k] = w
            syn_trace[k, 0] = x_pre
            syn_trace[k, 1] = x_post

        pos = (pos + 1) % L

    hist_pos[0] = pos

# %% Network: the interface between neurons

# ============================================================
# Network: the interface between neurons
# ============================================================

class Network:
    """
    Couples neurons and advances them together. Topology is frozen on
    first use; parameters stay live-tunable.
    """

    def __init__(self, dt=0.01):
        self.dt = dt
        self.neurons = []
        self.synapses = []
        self.gap_junctions = []
        self.vetos = []

        self.step_count = 0
        self._compiled = False
        self._out = None

    def _check_open(self):
        if self._compiled:
            raise RuntimeError("Topology is frozen after the first run.")

    def _check_member(self, *neurons):
        for neuron in neurons:
            if neuron not in self.neurons:
                raise ValueError(f"{neuron.name} is not in this network.")

    def add(self, neuron):
        self._check_open()
        self.neurons.append(neuron)
        return neuron

    def connect_chemical(self, pre, post, g_max, E_rev, alpha, beta,
                         delay=0.0, **plasticity):
        self._check_open()
        self._check_member(pre, post)
        synapse = Synapse(pre, post, g_max, E_rev, alpha, beta, delay,
                          **plasticity)
        self.synapses.append(synapse)
        return synapse

    def connect_excitatory(self, pre, post, g_max=0.5, **plasticity):
        return self.connect_chemical(
            pre, post, g_max, **AMPA, **plasticity)

    def connect_inhibitory(self, pre, post, g_max=0.5, **plasticity):
        return self.connect_chemical(
            pre, post, g_max, **GABA_A, **plasticity)

    def connect_gap(self, a, b, g=0.05):
        self._check_open()
        self._check_member(a, b)
        junction = GapJunction(a, b, g)
        self.gap_junctions.append(junction)
        return junction

    def connect_veto(self, pre, post, strength=1.0, alpha=VETO["alpha"],
                     beta=VETO["beta"], gate_ext=False, delay=0.0):
        self._check_open()
        self._check_member(pre, post)
        veto = VetoSynapse(pre, post, strength, alpha, beta, gate_ext, delay)
        self.vetos.append(veto)
        return veto

    def _index(self, neuron):
        return self.neurons.index(neuron)

    def _compile(self):
        N = len(self.neurons)

        self._states = np.empty((N, 4), dtype=np.float64)

        for i, neuron in enumerate(self.neurons):
            self._states[i] = neuron.state
            neuron.state = self._states[i]

        self._params = np.empty((N, 8), dtype=np.float64)
        self._i_ext = np.zeros(N, dtype=np.float64)

        S = len(self.synapses)

        self._syn_pre = np.array(
            [self._index(s.pre) for s in self.synapses], dtype=np.int64)
        self._syn_post = np.array(
            [self._index(s.post) for s in self.synapses], dtype=np.int64)
        self._syn_g = np.array(
            [s.g_max for s in self.synapses], dtype=np.float64)
        self._syn_g0 = self._syn_g.copy()
        self._syn_E = np.zeros(S)
        self._syn_alpha = np.zeros(S)
        self._syn_beta = np.zeros(S)
        self._syn_s = np.zeros(S)
        self._syn_plast = np.zeros((S, 7))
        self._syn_trace = np.zeros((S, 2))
        self._syn_delay = np.zeros(S, dtype=np.int64)
        self._syn_v_response = np.zeros(S, dtype=np.float64)
        self._syn_v_out = None

        self._gap_a = np.array(
            [self._index(g.a) for g in self.gap_junctions], dtype=np.int64)
        self._gap_b = np.array(
            [self._index(g.b) for g in self.gap_junctions], dtype=np.int64)
        self._gap_g = np.zeros(len(self.gap_junctions))

        K = len(self.vetos)

        self._veto_pre = np.array(
            [self._index(v.pre) for v in self.vetos], dtype=np.int64)
        self._veto_post = np.array(
            [self._index(v.post) for v in self.vetos], dtype=np.int64)
        self._veto_strength = np.zeros(K)
        self._veto_ext = np.zeros(K)
        self._veto_alpha = np.zeros(K)
        self._veto_beta = np.zeros(K)
        self._veto_s = np.zeros(K)
        self._veto_delay = np.zeros(K, dtype=np.int64)

        # Voltage history for transmission delays (initialised at rest).
        self._hist_len = int(round(MAX_DELAY_MS / self.dt)) + 1
        self._vhist = np.empty((N, self._hist_len), dtype=np.float64)

        for i in range(N):
            self._vhist[i, :] = self._states[i, 0]

        self._hist_pos = np.zeros(1, dtype=np.int64)

        self._compiled = True

    def _delay_steps(self, delay_ms):
        steps = int(round(delay_ms / self.dt))
        return max(0, min(steps, self._hist_len - 1))

    def _sync_parameters(self):
        for i, neuron in enumerate(self.neurons):
            self._params[i] = neuron.params.as_array()
            self._i_ext[i] = neuron.external_current()

        for k, syn in enumerate(self.synapses):
            self._syn_E[k] = syn.E_rev
            self._syn_alpha[k] = syn.alpha
            self._syn_beta[k] = syn.beta
            self._syn_g0[k] = syn.g_max
            self._syn_delay[k] = self._delay_steps(syn.delay)
            self._syn_plast[k] = (
                1.0 if syn.plastic else 0.0,
                syn.A_plus, syn.A_minus,
                syn.tau_plus, syn.tau_minus,
                syn.w_max, syn.tau_decay,
            )

            # Plastic weights are state owned by the kernel; only
            # non-plastic synapses follow g_max live.
            if not syn.plastic:
                self._syn_g[k] = syn.g_max

        for k, gap in enumerate(self.gap_junctions):
            self._gap_g[k] = gap.g

        for k, v in enumerate(self.vetos):
            self._veto_strength[k] = v.strength
            self._veto_ext[k] = 1.0 if v.gate_ext else 0.0
            self._veto_alpha[k] = v.alpha
            self._veto_beta[k] = v.beta
            self._veto_delay[k] = self._delay_steps(v.delay)

    def _run(self, states, syn_s, syn_g, syn_trace, veto_s,
             vhist, hist_pos, syn_v_response, syn_v_out,
             n_steps, t0, out):
        simulate_network_chunk(
            states, self._params, self._i_ext,
            self._syn_pre, self._syn_post, syn_g,
            self._syn_E, self._syn_alpha, self._syn_beta, syn_s,
            self._syn_g0, self._syn_plast, syn_trace, self._syn_delay,
            self._gap_a, self._gap_b, self._gap_g,
            self._veto_pre, self._veto_post, self._veto_strength,
            self._veto_ext, self._veto_alpha, self._veto_beta, veto_s,
            self._veto_delay, vhist, hist_pos, syn_v_response, syn_v_out,
            n_steps, t0, self.dt, out,
        )

    def weights(self):
        """Copy of the live synaptic weights (one per chemical synapse)."""
        return self._syn_g.copy()

    def warmup(self, n_steps=CHUNK_STEPS):
        if not self._compiled:
            self._compile()

        self._sync_parameters()
        out = np.empty((len(self.neurons), len(CHANNELS), n_steps))
        # Everything the kernel mutates is copied, so warmup learns nothing.
        syn_v_out = np.empty((len(self.synapses), n_steps), dtype=np.float64)
        self._run(
            self._states.copy(), self._syn_s.copy(),
            self._syn_g.copy(), self._syn_trace.copy(),
            self._veto_s.copy(), self._vhist.copy(), self._hist_pos.copy(),
            self._syn_v_response.copy(), syn_v_out,
            n_steps, 0.0, out,
        )

    def step_chunk(self, n_steps):
        if not self._compiled:
            self._compile()

        if self._out is None or self._out.shape[2] != n_steps:
            self._out = np.empty(
                (len(self.neurons), len(CHANNELS), n_steps),
                dtype=np.float64,
            )

        if self._syn_v_out is None or self._syn_v_out.shape[1] != n_steps:
            self._syn_v_out = np.empty(
                (len(self.synapses), n_steps), dtype=np.float64
            )

        self._sync_parameters()
        self._run(
            self._states, self._syn_s, self._syn_g, self._syn_trace,
            self._veto_s, self._vhist, self._hist_pos,
            self._syn_v_response, self._syn_v_out,
            n_steps, self.t, self._out,
        )
        self.step_count += n_steps

        for k, syn in enumerate(self.synapses):
            syn.weight = float(self._syn_g[k])

        return self._out

    @property
    def t(self):
        return self.step_count * self.dt

# %% Thread-safe data buffers

# ============================================================
# Thread-safe circular buffers (written by the worker thread,
# read by the GUI thread)
# ============================================================

class PlotBuffer:
    """Per-neuron ring buffer of all CHANNELS."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.lock = threading.Lock()

        self.data = {
            name: np.zeros(capacity, dtype=np.float64)
            for name in CHANNELS
        }

        self.write_pos = 0
        self.count = 0
        self.version = 0

    def append_batch(self, block):
        """Append a (len(CHANNELS), n) array of samples."""
        n = block.shape[1]

        if n == 0:
            return

        if n > self.capacity:
            block = block[:, -self.capacity:]
            n = self.capacity

        with self.lock:
            first = min(n, self.capacity - self.write_pos)
            remaining = n - first

            for row, key in enumerate(CHANNELS):
                self.data[key][
                    self.write_pos:self.write_pos + first
                ] = block[row, :first]

                if remaining:
                    self.data[key][:remaining] = block[row, first:]

            self.write_pos = (self.write_pos + n) % self.capacity
            self.count = min(self.count + n, self.capacity)
            self.version += 1

    def snapshot(self, keys=None):
        """Ordered copy of the requested channels (all if keys is None)."""
        keys = CHANNELS if keys is None else keys

        with self.lock:
            if self.count < self.capacity:
                return {
                    k: self.data[k][:self.count].copy() for k in keys
                }

            wp = self.write_pos

            return {
                k: np.concatenate((self.data[k][wp:], self.data[k][:wp]))
                for k in keys
            }


class ConnectionPlotBuffer:
    """Ring buffer for time series of per-chemical-synapse voltage estimates."""
    def __init__(self, n_connections, capacity):
        self.capacity = capacity
        self.n_connections = n_connections
        self.lock = threading.Lock()
        self.t = np.zeros(capacity, dtype=np.float64)
        self.values = np.zeros((n_connections, capacity), dtype=np.float64)
        self.write_pos = 0
        self.count = 0
        self.version = 0

    def append_batch(self, t, values):
        n = len(t)
        if n == 0 or self.n_connections == 0:
            return
        if n > self.capacity:
            t = t[-self.capacity:]
            values = values[:, -self.capacity:]
            n = self.capacity
        with self.lock:
            first = min(n, self.capacity - self.write_pos)
            remaining = n - first
            self.t[self.write_pos:self.write_pos + first] = t[:first]
            self.values[:, self.write_pos:self.write_pos + first] = values[:, :first]
            if remaining:
                self.t[:remaining] = t[first:]
                self.values[:, :remaining] = values[:, first:]
            self.write_pos = (self.write_pos + n) % self.capacity
            self.count = min(self.count + n, self.capacity)
            self.version += 1

    def snapshot(self):
        with self.lock:
            if self.count < self.capacity:
                return self.t[:self.count].copy(), self.values[:, :self.count].copy()
            p = self.write_pos
            return (np.concatenate((self.t[p:], self.t[:p])),
                    np.concatenate((self.values[:, p:], self.values[:, :p]), axis=1))


class WeightHistory:
    """Slow ring buffer of plastic synapse weights."""

    def __init__(self, n, capacity):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.t = np.zeros(capacity)
        self.w = np.zeros((n, capacity))
        self.pos = 0
        self.count = 0
        self.version = 0

    def append(self, t, w):
        with self.lock:
            self.t[self.pos] = t
            self.w[:, self.pos] = w
            self.pos = (self.pos + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)
            self.version += 1

    def snapshot(self):
        with self.lock:
            if self.count < self.capacity:
                return (self.t[:self.count].copy(),
                        self.w[:, :self.count].copy())

            p = self.pos
            return (np.concatenate((self.t[p:], self.t[:p])),
                    np.concatenate((self.w[:, p:], self.w[:, :p]), axis=1))


class SpikeLog:
    """Thread-safe record of recent spike times (ms), one list per neuron."""

    def __init__(self, n, keep_ms=10000.0):
        self.lock = threading.Lock()
        self.keep_ms = keep_ms
        self.times = [deque() for _ in range(n)]

    def add(self, neuron_idx, spike_times):
        with self.lock:
            for i, t in zip(neuron_idx, spike_times):
                self.times[i].append(float(t))

            cutoff = float(np.max(spike_times)) - self.keep_ms

            for dq in self.times:
                while dq and dq[0] < cutoff:
                    dq.popleft()

    def snapshot(self):
        with self.lock:
            return [np.array(dq, dtype=float) for dq in self.times]

# %% Simulation thread

# ============================================================
# Simulation thread
# ============================================================

class SimulationWorker(threading.Thread):
    def __init__(self, network, buffers, stop_event,
                 chunk_steps=CHUNK_STEPS, speed=1.0,
                 weight_hist=None, plastic_idx=(), spike_log=None,
                 connection_buffer=None):
        super().__init__(daemon=True)

        self.network = network
        self.buffers = buffers
        self.stop_event = stop_event
        self.chunk_steps = chunk_steps
        self.speed = speed
        self.error = None
        self.weight_hist = weight_hist
        self.plastic_idx = list(plastic_idx)
        self.spike_log = spike_log
        self.connection_buffer = connection_buffer

    def run(self):
        try:
            chunk = self.chunk_steps
            deadline = time.perf_counter()
            chunks = 0                      # counts chunks, not neurons
            prev_v = np.array([n.V for n in self.network.neurons])

            while not self.stop_event.is_set():
                out = self.network.step_chunk(chunk)

                if not np.isfinite(out).all():
                    bad = np.argwhere(~np.isfinite(out))[0]
                    self.error = (
                        f"Diverged: neuron "
                        f"{self.network.neurons[bad[0]].name}, "
                        f"{CHANNELS[bad[1]]}, t = {self.network.t:.1f} ms. "
                        f"Try weaker coupling."
                    )
                    print(self.error)
                    self.stop_event.set()
                    break

                if self.spike_log is not None:
                    v = out[:, 1, :]
                    ext = np.concatenate((prev_v[:, None], v), axis=1)
                    hit = (
                        (ext[:, :-1] < SPIKE_THRESHOLD)
                        & (ext[:, 1:] >= SPIKE_THRESHOLD)
                    )
                    ni, ki = np.nonzero(hit)

                    if len(ni):
                        self.spike_log.add(ni, out[ni, 0, ki])

                    prev_v = v[:, -1].copy()

                for i, buffer in enumerate(self.buffers):
                    buffer.append_batch(out[i])

                if self.connection_buffer is not None and len(self.network.synapses):
                    self.connection_buffer.append_batch(
                        out[0, 0, :], self.network._syn_v_out
                    )

                chunks += 1

                if (self.weight_hist is not None
                        and chunks % WEIGHT_SAMPLE_CHUNKS == 0):
                    self.weight_hist.append(
                        self.network.t,
                        self.network.weights()[self.plastic_idx],
                    )

                period = chunk * self.network.dt / 1000.0 / self.speed
                deadline += period
                delay = deadline - time.perf_counter()

                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.1:
                    deadline = time.perf_counter()

        except Exception as exc:
            traceback.print_exc()
            self.error = f"Error: {exc}"
            self.stop_event.set()
