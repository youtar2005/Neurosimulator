"""
analytics.py - analysis of simulation output (no GUI, no simulation deps).

Every function here takes plain NumPy data (e.g. spike times in ms), so
it can be used live from the visualization or offline from scripts.

Contents
    * estimate_period      - cycle period / spikes per cycle / regularity
    * predict_loop_period  - analytic period prediction for A -> B -> C -| A

Add further analyses (firing rate, ISI statistics, phase relationships,
synchrony measures, ...) in new sections below.
"""

import math

import numpy as np

# %% Settings

PERIOD_SPAN_MS = 3000.0     # Spikes used for the period estimate

# %% Rhythm analysis

# ============================================================
# Rhythm analysis
# ============================================================

def estimate_period(times, now, span_ms=PERIOD_SPAN_MS, gap_factor=1.6):
    """
    Cycle period of a spike train (times in ms) from the last span_ms.

    Unusually long interspike intervals (> gap_factor x the typical short
    interval) separate bursts; the period is the mean burst-to-burst
    interval. With no such gaps the neuron fires once per cycle and the
    period is the median interspike interval.

    Returns dict(P, spc, cv) or None:
        P   period (ms)
        spc spikes per cycle
        cv  coefficient of variation of the period (0 = perfectly regular)
    """
    t = np.asarray(times, dtype=float)
    t = t[t > now - span_ms]

    if len(t) < 4:
        return None

    isi = np.diff(t)
    gap = gap_factor * np.percentile(isi, 10)

    starts = [t[0]]
    counts = [1]

    for a, b in zip(t[:-1], t[1:]):
        if b - a > gap:
            starts.append(b)
            counts.append(1)
        else:
            counts[-1] += 1

    if len(starts) >= 4:
        per = np.diff(starts)
        return dict(
            P=float(per.mean()),
            spc=float(np.median(counts[1:-1])),
            cv=float(per.std() / per.mean()),
        )

    return dict(
        P=float(np.median(isi)),
        spc=1.0,
        cv=float(isi.std() / isi.mean()),
    )


# %% Loop period prediction

# ---- Period prediction for the A -> B -> C -| A loop ----
#
# Fitted for: default HH neurons, AMPA synapses A->B and B->C with
# g = 0.5, and a veto C -> A with strength 1, alpha 10, beta 0.02 and
# "Also veto external drive" ticked. A is driven by a constant current I
# (bias + held stimulus, uA/cm^2).
#
#   T_A(I)  = 11.45 * (I / 25) ** -0.357       A's own tonic interval
#   t_loop  = d_AB + d_BC + d_CA + 2 * 2.07    A's spike -> veto reaches A
#   n       = ceil(t_loop / T_A)               A spikes per cycle
#   P       = (n - 1) * T_A + t_loop + K / (I - I0) + L0
#
# Reading it: A fires n spikes before the veto lands, the loop transit
# takes t_loop, and then A needs a recovery time K / (I - I0) before the
# fading veto lets it restart. The recovery time grows without bound as
# I falls toward I0 ~ 24 (below ~24.5 the rhythm dies out). Only the SUM
# of the three delays matters.
#
# Measured accuracy for 24.5 < I < 32: mean error 3.6 %, max 8.5 % on
# settings not used in the fit. Above I ~ 32 the veto no longer silences
# A and the formula fails. K, I0 and L0 are not separately identifiable
# (different triples fit equally well), and everything was fit at
# beta = 0.02 only; the dependence on beta is NOT captured.

LOOP_TA_REF = 11.45
LOOP_TA_EXP = -0.357
LOOP_LATENCY = 2.07
LOOP_I0 = 24.05
LOOP_K = 820.0
LOOP_OFFSET = -10.0
LOOP_I_MIN = 24.5
LOOP_I_MAX = 32.0


def predict_loop_period(I, d_ab, d_bc, d_ca):
    """
    Predicted period (ms) and spikes per cycle of the 3-neuron loop, or
    None if I is outside the validated range (see above).
    """
    if I <= LOOP_I_MIN or I >= LOOP_I_MAX:
        return None

    t_a = LOOP_TA_REF * (I / 25.0) ** LOOP_TA_EXP
    t_loop = d_ab + d_bc + d_ca + 2.0 * LOOP_LATENCY
    n = math.ceil(t_loop / t_a)

    period = (
        (n - 1) * t_a + t_loop
        + LOOP_K / (I - LOOP_I0)
        + LOOP_OFFSET
    )

    return period, n
