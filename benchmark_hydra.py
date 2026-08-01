#!/usr/bin/env python
"""
benchmark_hydra.py
==================
Profile and benchmark the hydra-pspec Gibbs sampler. Reports how
computational (memory) and temporal (wall-clock time) costs scale with
increasing input data sizes.

Run with the py10 conda environment:
    conda run -n py10 python benchmark_hydra.py

Four dimensions are swept independently, all others held at BASE:
    Ntimes     : number of time samples          (rows of vis)
    Nfreqs     : number of frequency channels    (cols of vis)
    Nfgmodes   : number of foreground modes
    Nsys_modes : number of multiplicative systematic modes

Outputs (written to ./benchmark_results/):
    scaling_overview.png   - time + memory vs each parameter
    subfunc_breakdown.png  - per-sub-function timing vs Ntimes
    cprofile_report.txt    - cProfile hot-path table
    benchmark_summary.txt  - full numerical results table
"""

import sys
import os
import time
import cProfile
import pstats
import io
import tracemalloc
import contextlib

import numpy as np
import scipy.special
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt

# ── Import from installed package ─────────────────────────────────────────────
import hydra_pspec as hp
from hydra_pspec.pspec import (
    gibbs_step,
    gcr_fg_and_signal,
    sample_pspec,
    covariance_from_pspec,
)
from hydra_pspec.sys_solver import gcr_systematics
from hydra_pspec.sys_solver import sys_modes as build_sys_modes
from hydra_pspec.utils import fourier_operator


# =============================================================================
# Configuration
# =============================================================================

# Baseline (reference) configuration
BASE = dict(Ntimes=80, Nfreqs=60, Nfgmodes=10, Nsys_modes=4)

# How many repeated timing calls to average over for each data point
N_REPEATS = 3

# Values to sweep for each parameter (others held at BASE)
SWEEPS = {
    "Ntimes":     [10, 20, 40, 80, 120, 160],
    "Nfreqs":     [20, 40, 60, 80, 100, 120],
    "Nfgmodes":   [2,  5,  10, 15, 20,  30],
    "Nsys_modes": [1,  2,  4,  6,  8],
}

OUTPUT_DIR = "benchmark_results"


# =============================================================================
# Utilities
# =============================================================================

@contextlib.contextmanager
def suppress_stdout():
    """Redirect stdout to /dev/null for the duration of the context."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def theoretical_memory_MB(Ntimes, Nfreqs, Nfgmodes, Nsys_modes):
    """
    Estimate the theoretical peak memory footprint (in MB) for one
    gibbs_step() call, based on the dominant arrays.

    All dominant arrays are complex128 (16 bytes/element) unless noted.
    """
    c = 16   # bytes per complex128
    d = 8    # bytes per float64

    # Visibility data
    vis  = Ntimes * Nfreqs * c
    # Block-operator matrix A built per time: (Nfreqs + Nfgmodes)^2
    A_mat = (Nfreqs + Nfgmodes)**2 * c
    # E, Einv, sqrtE: all (Nfreqs, Nfreqs)
    E_mats = 3 * Nfreqs**2 * c
    # Foreground modes
    fg    = Nfreqs * Nfgmodes * c
    # Systematic modes (Ntimes*Nfreqs, Nsys_modes)
    sm    = Ntimes * Nfreqs * Nsys_modes * c
    # GCR systematics: M_tilde is (2*Ntimes*Nfreqs, 2*Nsys_modes)
    mt    = 2 * Ntimes * Nfreqs * 2 * Nsys_modes * d
    # Ninv (Nfreqs, Nfreqs)
    ni    = Nfreqs**2 * d
    # Output arrays from one Gibbs step
    out   = Ntimes * Nfreqs * c + Nfreqs * d + Ntimes * Nfgmodes * c + Nsys_modes * c

    total = vis + A_mat + E_mats + fg + sm + mt + ni + out
    return total / 1e6


def memory_estimate(Ntimes, Nfreqs, Nfgmodes, Nsys_modes, Niter=1):
    """
    Print a complete, itemised memory estimate for a gibbs_sample() run.

    Two categories are reported:
      - Per-step compute arrays  : constant overhead, independent of Niter.
        These are the working arrays allocated and (mostly) freed each step.
      - Output sample arrays     : grow linearly with Niter.
        gibbs_sample() pre-allocates all Niter samples upfront in RAM.

    Parameters
    ----------
    Ntimes, Nfreqs, Nfgmodes, Nsys_modes : int
        Problem dimensions (must match what you pass to gibbs_sample / sys_sampler_wrapper).
    Niter : int
        Number of Gibbs iterations (default 1 for a per-step estimate).

    Usage
    -----
    From the command line (py10 conda env):
        conda run -n py10 python -c "
        from benchmark_hydra import memory_estimate
        memory_estimate(Ntimes=80, Nfreqs=60, Nfgmodes=10, Nsys_modes=4, Niter=100000)
        "

    Or interactively:
        from benchmark_hydra import memory_estimate
        memory_estimate(80, 60, 10, 4, 100000)
    """
    c = 16    # bytes per complex128 element
    d = 8     # bytes per float64 element

    # ── Per-step compute arrays (constant, independent of Niter) ────────────
    # These exist during each gibbs_step() call and are reused/freed each iter.

    items_compute = [
        # (label, shape description, bytes)
        ("vis  – visibility data",
         f"({Ntimes} × {Nfreqs})",
         Ntimes * Nfreqs * c),

        ("Ninv – inverse noise cov",
         f"({Nfreqs} × {Nfreqs})",
         Nfreqs * Nfreqs * d),

        ("fg_modes – foreground basis",
         f"({Nfreqs} × {Nfgmodes})",
         Nfreqs * Nfgmodes * d),

        ("sys_modes – systematics basis",
         f"({Ntimes * Nfreqs} × {Nsys_modes})",
         Ntimes * Nfreqs * Nsys_modes * c),

        ("E, Einv, sqrtE – signal covariances (×3)",
         f"3 × ({Nfreqs} × {Nfreqs})",
         3 * Nfreqs * Nfreqs * c),

        ("A – GCR block operator (per time step)",
         f"({Nfreqs + Nfgmodes} × {Nfreqs + Nfgmodes})",
         (Nfreqs + Nfgmodes) ** 2 * c),

        ("M_tilde – systematics real-split matrix",
         f"({2 * Ntimes * Nfreqs} × {2 * Nsys_modes})",
         2 * Ntimes * Nfreqs * 2 * Nsys_modes * d),

        ("Ainv_sys – systematics preconditioner",
         f"({2 * Nsys_modes} × {2 * Nsys_modes})",
         (2 * Nsys_modes) ** 2 * d),
    ]

    # ── Output sample arrays (scale linearly with Niter) ─────────────────────
    # gibbs_sample() allocates all of these upfront before the loop starts.

    items_output = [
        ("signal_amps – EoR signal samples",
         f"({Niter} × {Ntimes} × {Nfreqs})",
         Niter * Ntimes * Nfreqs * c),

        ("signal_ps   – power spectrum samples",
         f"({Niter} × {Nfreqs})",
         Niter * Nfreqs * d),

        ("fg_amps     – foreground amplitude samples",
         f"({Niter} × {Ntimes} × {Nfgmodes})",
         Niter * Ntimes * Nfgmodes * c),

        ("sys_amps    – systematics amplitude samples",
         f"({Niter} × {Nsys_modes})",
         Niter * Nsys_modes * c),

        ("chisq       – chi-squared per iteration",
         f"({Niter} × {Ntimes} × {Nfreqs})",
         Niter * Ntimes * Nfreqs * d),

        ("ln_post     – log-posterior per iteration",
         f"({Niter},)",
         Niter * d),
    ]

    def _fmt(b):
        """Format bytes as MB or GB."""
        if b >= 1e9:
            return f"{b / 1e9:>9.2f} GB"
        return f"{b / 1e6:>9.2f} MB"

    W = 74
    print("=" * W)
    print(f" Memory estimate for gibbs_sample()")
    print(f"   Ntimes={Ntimes}, Nfreqs={Nfreqs}, Nfgmodes={Nfgmodes}, "
          f"Nsys_modes={Nsys_modes}, Niter={Niter:,}")
    print("=" * W)

    # Compute section
    compute_total = sum(b for _, _, b in items_compute)
    print(f"\n{'─'*W}")
    print(f" PER-STEP COMPUTE ARRAYS  (constant, independent of Niter)")
    print(f"{'─'*W}")
    print(f"  {'Array':<44} {'Shape':<28} {'Size':>10}")
    print(f"  {'-'*44} {'-'*28} {'-'*10}")
    for label, shape, b in items_compute:
        print(f"  {label:<44} {shape:<28} {_fmt(b)}")
    print(f"  {'─'*44} {'─'*28} {'─'*10}")
    print(f"  {'SUBTOTAL':<44} {'':<28} {_fmt(compute_total)}")

    # Output section
    output_total = sum(b for _, _, b in items_output)
    print(f"\n{'─'*W}")
    print(f" OUTPUT SAMPLE ARRAYS  (scale linearly with Niter={Niter:,})")
    print(f"{'─'*W}")
    print(f"  {'Array':<44} {'Shape':<28} {'Size':>10}")
    print(f"  {'-'*44} {'-'*28} {'-'*10}")
    for label, shape, b in items_output:
        print(f"  {label:<44} {shape:<28} {_fmt(b)}")
    print(f"  {'─'*44} {'─'*28} {'─'*10}")
    print(f"  {'SUBTOTAL':<44} {'':<28} {_fmt(output_total)}")

    grand_total = compute_total + output_total
    print(f"\n{'─'*W}")
    print(f"  {'TOTAL ESTIMATED RAM':<44} {'':<28} {_fmt(grand_total)}")
    print(f"  {'  of which Niter-dependent':<44} {'':<28} {_fmt(output_total)}")
    print(f"  {'  of which fixed overhead':<44} {'':<28} {_fmt(compute_total)}")

    # Dominant scaling terms for the user
    dominant_output = max(items_output, key=lambda x: x[2])
    print(f"\n  Dominant output array : {dominant_output[0].split('–')[0].strip()}")
    print(f"  Scaling law (output)  : {Ntimes} × {Nfreqs} × 16 × Niter bytes")
    print(f"                        = {Ntimes * Nfreqs * 16 / 1e6:.4f} MB per 1000 iterations")
    print("=" * W)


def make_inputs(Ntimes, Nfreqs, Nfgmodes, Nsys_modes, rng_seed=42):
    """
    Build fully synthetic inputs for gibbs_step(), no file I/O required.

    Parameters
    ----------
    Ntimes, Nfreqs, Nfgmodes, Nsys_modes : int
        Problem dimensions.
    rng_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        All arrays needed to call gibbs_step() and the main sub-functions.
    """
    rng = np.random.default_rng(rng_seed)

    # Frequency and time grids
    freqs_Hz = np.linspace(100e6, 120e6, Nfreqs)
    times_s  = np.linspace(0.0, 1.0, Ntimes) * (24.0 / (2.0 * np.pi)) * 3600.0

    # ── Fourier operator ────────────────────────────────────────────────────
    F = fourier_operator(Nfreqs, unitary=True)

    # ── EoR power spectrum + covariance ────────────────────────────────────
    ps_true = 0.0012 * (1.0 + 0.3 * np.sin(3.0 * np.linspace(0.0, 1.0, Nfreqs)))
    S_true  = covariance_from_pspec(ps_true, F)
    sqrtS   = np.linalg.cholesky(S_true)

    # EoR signal (Ntimes, Nfreqs)
    eor = (
        sqrtS
        @ (rng.standard_normal((Nfreqs, Ntimes)) + 1j * rng.standard_normal((Nfreqs, Ntimes)))
        / np.sqrt(2.0)
    ).T

    # ── Foreground modes (Legendre polynomials) ─────────────────────────────
    x_leg   = np.linspace(-1.0, 1.0, Nfreqs)
    fg_modes = np.column_stack(
        [scipy.special.legendre(i)(x_leg) for i in range(Nfgmodes)]
    )  # (Nfreqs, Nfgmodes)

    # Random FG amplitudes → foreground visibility
    fg_amps_t = (
        rng.standard_normal((Ntimes, Nfgmodes))
        + 1j * rng.standard_normal((Ntimes, Nfgmodes))
    )
    fg_true = fg_amps_t @ fg_modes.T   # (Ntimes, Nfreqs)

    # ── Noise ───────────────────────────────────────────────────────────────
    noise_ps = 0.0004 * np.ones(Nfreqs)
    N_true   = covariance_from_pspec(noise_ps, F)
    Ninv     = np.diag(1.0 / np.diag(N_true))
    noise    = (
        np.sqrt(N_true)
        @ (rng.standard_normal((Nfreqs, Ntimes)) + 1j * rng.standard_normal((Nfreqs, Ntimes)))
        / np.sqrt(2.0)
    ).T   # (Ntimes, Nfreqs)

    # ── Systematics ─────────────────────────────────────────────────────────
    # Mode pairs: (delay_index, fringe_rate_index = 0)
    nm_list = [(3 + i, 0) for i in range(Nsys_modes)]
    with suppress_stdout():
        sm = build_sys_modes(freqs_Hz=freqs_Hz, times_sec=times_s, modes=nm_list)
    # shape: (Ntimes * Nfreqs, Nsys_modes)

    sys_amps_true = (
        rng.standard_normal(Nsys_modes) + 1j * rng.standard_normal(Nsys_modes)
    ) * 0.05   # small amplitudes so sampler doesn't diverge

    sys_prior = 100.0**2 * np.eye(Nsys_modes)

    gain_true = (1.0 + (sm @ sys_amps_true).reshape((Nfreqs, Ntimes)).T)

    # ── Data ────────────────────────────────────────────────────────────────
    vis   = gain_true * (fg_true + eor) + noise
    flags = np.ones(Nfreqs, dtype=int)

    # ── Power spectrum prior ─────────────────────────────────────────────────
    # shape (2, Nfreqs): row 0 = lower bound, row 1 = upper bound
    ps_prior = np.vstack([
        1e-7 * np.ones(Nfreqs),
        1e-1 * np.ones(Nfreqs),
    ])

    return dict(
        vis=vis, flags=flags, Ninv=Ninv,
        signal_ps=ps_true, signal_ps_prior=ps_prior,
        fg_modes=fg_modes, sys_modes=sm,
        sys_amps=sys_amps_true.copy(), sys_prior=sys_prior,
        freqs_Hz=freqs_Hz, times_s=times_s,
        fourier_op=F, eor=eor, fg_true=fg_true,
        Ntimes=Ntimes, Nfreqs=Nfreqs,
        Nfgmodes=Nfgmodes, Nsys_modes=Nsys_modes,
    )


# =============================================================================
# Timing and memory helpers
# =============================================================================

def time_fn(fn, *args, n_repeats=N_REPEATS, **kwargs):
    """
    Time fn(*args, **kwargs) n_repeats times with stdout suppressed.

    Returns
    -------
    mean_s, std_s : float
        Mean and standard deviation of elapsed wall-clock time (seconds).
    """
    times = []
    with suppress_stdout():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def peak_mem_tracemalloc(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) once under tracemalloc and return the
    peak Python-level memory increment in bytes.
    """
    tracemalloc.start()
    with suppress_stdout():
        fn(*args, **kwargs)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# =============================================================================
# Gibbs-step wrapper
# =============================================================================

def run_gibbs_step(inps):
    """Run one gibbs_step() call from a pre-built inputs dict."""
    return gibbs_step(
        vis=inps["vis"],
        flags=inps["flags"],
        Ninv=inps["Ninv"],
        signal_ps=inps["signal_ps"],
        signal_ps_prior=inps["signal_ps_prior"],
        fg_modes=inps["fg_modes"],
        sys_modes=inps["sys_modes"],
        sys_amps=inps["sys_amps"].copy(),   # copy so repeated calls don't corrupt
        sys_prior=inps["sys_prior"],
        iter=0,
        verbose=False,
    )


# =============================================================================
# Parameter sweeps: full Gibbs step
# =============================================================================

def sweep_parameter(param_name, values, log_lines):
    """
    Vary one dimension over `values`, hold others at BASE.

    Returns
    -------
    t_means, t_stds, mem_measured_MB, mem_theory_MB : ndarray
    """
    header = f"Sweeping {param_name}: {values}"
    print(f"\n{'─'*60}\n{header}\n{'─'*60}")
    log_lines.append(f"\n{header}")

    t_means, t_stds, mem_meas, mem_theory = [], [], [], []

    for v in values:
        cfg = {**BASE, param_name: v}
        Nt, Nf, Nfg, Ns = (cfg["Ntimes"], cfg["Nfreqs"],
                            cfg["Nfgmodes"], cfg["Nsys_modes"])

        inps = make_inputs(Nt, Nf, Nfg, Ns)

        # Warm-up call (avoids first-call overhead in plots)
        with suppress_stdout():
            run_gibbs_step(inps)

        # Timing
        t_mean, t_std = time_fn(run_gibbs_step, inps, n_repeats=N_REPEATS)

        # Memory (single call under tracemalloc)
        mem_b = peak_mem_tracemalloc(run_gibbs_step, inps)
        mem_mb = mem_b / 1e6

        # Theoretical estimate
        th_mb = theoretical_memory_MB(Nt, Nf, Nfg, Ns)

        t_means.append(t_mean)
        t_stds.append(t_std)
        mem_meas.append(mem_mb)
        mem_theory.append(th_mb)

        line = (
            f"  {param_name}={v:<5d} "
            f"(Nt={Nt}, Nf={Nf}, Nfg={Nfg}, Ns={Ns})  "
            f"time={t_mean:.3f}±{t_std:.3f}s  "
            f"mem_meas={mem_mb:.1f}MB  mem_theory={th_mb:.1f}MB"
        )
        print(line)
        log_lines.append(line)

    return (
        np.array(t_means), np.array(t_stds),
        np.array(mem_meas), np.array(mem_theory),
    )


# =============================================================================
# Sub-function timing breakdown
# =============================================================================

def sweep_subfunctions(ntimes_vals, log_lines):
    """
    For each Ntimes value (Nfreqs/Nfgmodes/Nsys_modes held at BASE),
    time each major sub-function individually.

    Sub-functions timed:
        gcr_fg_and_signal   – GCR solver for EoR + foreground
        gcr_systematics     – GCR solver for multiplicative systematics
        sample_pspec        – Inverse-gamma power spectrum sampling
        covariance_from_pspec – Fourier-domain covariance construction
    """
    header = "Sub-function breakdown (varying Ntimes, others at BASE)"
    print(f"\n{'─'*60}\n{header}\n{'─'*60}")
    log_lines.append(f"\n{header}")

    fn_keys   = ["gcr_fg", "gcr_sys", "sample_ps", "cov_from_ps"]
    fn_labels = {
        "gcr_fg":       "GCR EoR+FG solver",
        "gcr_sys":      "GCR systematics solver",
        "sample_ps":    "Power spectrum sampling",
        "cov_from_ps":  "Covariance from P(k)",
    }
    results = {k: {"mean": [], "std": []} for k in fn_keys}

    for nt in ntimes_vals:
        cfg  = {**BASE, "Ntimes": nt}
        inps = make_inputs(**cfg)

        vis      = inps["vis"]
        flags    = inps["flags"]
        fg_modes = inps["fg_modes"]
        Ninv     = inps["Ninv"]
        ps       = inps["signal_ps"]
        F        = inps["fourier_op"]
        sm       = inps["sys_modes"]
        sa       = inps["sys_amps"]
        sp       = inps["sys_prior"]
        Nf       = inps["Nfreqs"]
        Nfg      = inps["Nfgmodes"]
        eor      = inps["eor"]
        ps_prior = inps["signal_ps_prior"]

        Nparams  = Nf + Nfg
        sys_model = (1.0 + (sm @ sa).reshape((Nf, nt)).T)

        # Closures for each sub-function
        def fn_gcr_fg():
            return gcr_fg_and_signal(
                vis=vis, flags=flags, fg_modes=fg_modes, Nparams=Nparams,
                sys_model=sys_model, signal_ps=ps, Ninv=Ninv,
                fourier_op=F, verbose=False,
            )

        def fn_gcr_sys():
            return gcr_systematics(
                data=vis, Ninv=Ninv, sky_model=vis,
                sys_modes=sm, sys_prior=sp,
            )

        def fn_sample_ps():
            return sample_pspec(s=eor, prior=ps_prior)

        def fn_cov():
            return covariance_from_pspec(ps, F)

        fn_map = {
            "gcr_fg":      fn_gcr_fg,
            "gcr_sys":     fn_gcr_sys,
            "sample_ps":   fn_sample_ps,
            "cov_from_ps": fn_cov,
        }

        row_parts = [f"  Ntimes={nt:<5d}"]
        for key in fn_keys:
            m, s = time_fn(fn_map[key], n_repeats=N_REPEATS)
            results[key]["mean"].append(m)
            results[key]["std"].append(s)
            row_parts.append(f"{fn_labels[key]}={m:.4f}±{s:.4f}s")

        line = "  " + "   ".join(row_parts)
        print(line)
        log_lines.append(line)

    return results, fn_labels


# =============================================================================
# cProfile reference run
# =============================================================================

def run_cprofile(inps, top_n=25):
    """
    Run cProfile on one gibbs_step() call.

    Returns
    -------
    str
        Formatted cProfile stats (sorted by cumulative time).
    """
    pr = cProfile.Profile()
    pr.enable()
    with suppress_stdout():
        run_gibbs_step(inps)
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative")
    ps.print_stats(top_n)
    return buf.getvalue()


# =============================================================================
# Plotting
# =============================================================================

def make_plots(sweep_results, subfn_results, subfn_labels, ntimes_vals):
    """Produce and save all benchmark plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    params     = list(SWEEPS.keys())
    n_params   = len(params)
    param_colors = {p: f"C{i}" for i, p in enumerate(params)}

    # ── Figure 1: scaling overview (time & memory vs each parameter) ─────────
    fig, axes = plt.subplots(n_params, 3, figsize=(16, 4 * n_params))
    if n_params == 1:
        axes = axes[np.newaxis, :]

    for row, param in enumerate(params):
        vals              = SWEEPS[param]
        t_mean, t_std, mem_meas, mem_theory = sweep_results[param]
        c = param_colors[param]

        # Time panel
        ax = axes[row, 0]
        ax.errorbar(vals, t_mean, yerr=t_std, fmt="o-", capsize=4,
                    color=c, linewidth=1.8, markersize=6)
        ax.axvline(BASE[param], color="grey", linestyle="--", alpha=0.5,
                   label=f"baseline={BASE[param]}")
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("Wall-clock time (s)", fontsize=10)
        ax.set_title(f"Time vs {param}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Measured memory panel
        ax = axes[row, 1]
        ax.plot(vals, mem_meas, "s-", color=c, linewidth=1.8, markersize=6,
                label="tracemalloc peak")
        ax.axvline(BASE[param], color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("Peak memory (MB)", fontsize=10)
        ax.set_title(f"Measured peak memory vs {param}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Theoretical memory panel
        ax = axes[row, 2]
        ax.plot(vals, mem_theory, "^--", color=c, linewidth=1.8, markersize=6,
                label="theoretical estimate")
        ax.axvline(BASE[param], color="grey", linestyle="--", alpha=0.5)
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("Theoretical memory (MB)", fontsize=10)
        ax.set_title(f"Theoretical memory vs {param}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "hydra-pspec — Gibbs step scaling with input dimensions\n"
        f"(baseline: {BASE}, {N_REPEATS} repeats per point)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "scaling_overview.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Figure 2: sub-function breakdown ────────────────────────────────────
    fn_keys   = list(subfn_results.keys())
    fn_colors = {k: f"C{i}" for i, k in enumerate(fn_keys)}

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for key in fn_keys:
        means = np.array(subfn_results[key]["mean"])
        stds  = np.array(subfn_results[key]["std"])
        label = subfn_labels[key]
        c     = fn_colors[key]

        axes2[0].errorbar(
            ntimes_vals, means, yerr=stds,
            fmt="o-", capsize=3, label=label, color=c, linewidth=1.5,
        )

    axes2[0].set_xlabel("Ntimes", fontsize=11)
    axes2[0].set_ylabel("Time (s)", fontsize=11)
    axes2[0].set_title("Sub-function wall-clock time vs Ntimes", fontsize=11)
    axes2[0].legend(fontsize=8)
    axes2[0].grid(True, alpha=0.3)

    # Stacked area chart: fraction of total time each sub-function takes
    total = np.zeros(len(ntimes_vals))
    for key in fn_keys:
        total += np.array(subfn_results[key]["mean"])

    bottoms = np.zeros(len(ntimes_vals))
    for key in fn_keys:
        frac = np.array(subfn_results[key]["mean"]) / np.where(total > 0, total, 1)
        axes2[1].bar(
            ntimes_vals, frac, bottom=bottoms, width=np.array(ntimes_vals) * 0.08,
            label=subfn_labels[key], color=fn_colors[key], alpha=0.8,
        )
        bottoms += frac

    axes2[1].set_xlabel("Ntimes", fontsize=11)
    axes2[1].set_ylabel("Fraction of total sub-function time", fontsize=11)
    axes2[1].set_title("Relative cost of each sub-function", fontsize=11)
    axes2[1].legend(fontsize=8)
    axes2[1].grid(True, alpha=0.3, axis="y")
    axes2[1].set_ylim(0, 1.05)

    fig2.suptitle(
        "hydra-pspec — Sub-function timing breakdown (Nfreqs=60, Nfgmodes=10, Nsys_modes=4)",
        fontsize=12,
    )
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "subfunc_breakdown.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")


# =============================================================================
# Summary table
# =============================================================================

def write_summary(sweep_results, subfn_results, subfn_labels,
                  ntimes_vals, profile_str, log_lines):
    """Write all results to a plain-text file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []
    lines.append("=" * 80)
    lines.append("hydra-pspec Benchmark Summary")
    lines.append(f"Baseline configuration : {BASE}")
    lines.append(f"Timing repeats         : {N_REPEATS}")
    lines.append("=" * 80)

    # ── Parameter sweep table ────────────────────────────────────────────────
    lines.append("\n── Full Gibbs-step scaling ──────────────────────────────────────────────")
    header = f"{'Param':<14} {'Value':>7}  {'Time mean (s)':>14}  {'Time std (s)':>13}  "
    header += f"{'Mem meas (MB)':>14}  {'Mem theory (MB)':>16}  {'Baseline?':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for param in SWEEPS:
        vals = SWEEPS[param]
        t_mean, t_std, mem_meas, mem_theory = sweep_results[param]
        for i, v in enumerate(vals):
            marker = "<-- baseline" if v == BASE[param] else ""
            lines.append(
                f"{param:<14} {v:>7d}  {t_mean[i]:>14.4f}  {t_std[i]:>13.4f}  "
                f"{mem_meas[i]:>14.1f}  {mem_theory[i]:>16.1f}  {marker}"
            )
        lines.append("")

    # ── Sub-function table ───────────────────────────────────────────────────
    lines.append("\n── Sub-function timing vs Ntimes ────────────────────────────────────────")
    col_header = f"{'Ntimes':>8}" + "".join(
        f"  {subfn_labels[k][:22]:>24}" for k in subfn_results
    )
    lines.append(col_header)
    lines.append("-" * len(col_header))
    for j, nt in enumerate(ntimes_vals):
        row = f"{nt:>8}"
        for k in subfn_results:
            m = subfn_results[k]["mean"][j]
            s = subfn_results[k]["std"][j]
            row += f"  {m:>10.4f}±{s:<10.4f}"
        lines.append(row)

    # ── cProfile ─────────────────────────────────────────────────────────────
    lines.append("\n\n── cProfile (cumulative, reference config) ──────────────────────────────")
    lines.append(profile_str)

    text = "\n".join(lines)

    path = os.path.join(OUTPUT_DIR, "benchmark_summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved: {path}")

    path2 = os.path.join(OUTPUT_DIR, "cprofile_report.txt")
    with open(path2, "w") as f:
        f.write(f"cProfile — reference config: {BASE}\n\n")
        f.write(profile_str)
    print(f"Saved: {path2}")


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lines = []

    print("=" * 70)
    print("hydra-pspec Benchmarking Suite")
    print(f"Base configuration : {BASE}")
    print(f"Timing repeats     : {N_REPEATS}")
    print(f"Output directory   : {OUTPUT_DIR}/")
    print("=" * 70)

    # ── 1. Parameter sweeps (full Gibbs step) ────────────────────────────────
    sweep_results = {}
    for param, vals in SWEEPS.items():
        sweep_results[param] = sweep_parameter(param, vals, log_lines)

    # ── 2. Sub-function breakdown vs Ntimes ──────────────────────────────────
    ntimes_vals = SWEEPS["Ntimes"]
    subfn_results, subfn_labels = sweep_subfunctions(ntimes_vals, log_lines)

    # ── 3. cProfile reference run ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"cProfile: base configuration {BASE}")
    print(f"{'─'*60}")
    ref_inps    = make_inputs(**BASE)
    profile_str = run_cprofile(ref_inps, top_n=25)
    print(profile_str)

    # ── 4. Plots ─────────────────────────────────────────────────────────────
    make_plots(sweep_results, subfn_results, subfn_labels, ntimes_vals)

    # ── 5. Text summary ──────────────────────────────────────────────────────
    write_summary(
        sweep_results, subfn_results, subfn_labels,
        ntimes_vals, profile_str, log_lines,
    )

    # ── 6. Console summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"Results written to: {OUTPUT_DIR}/")
    print(f"  scaling_overview.png   — time + memory vs each dimension")
    print(f"  subfunc_breakdown.png  — sub-function timing vs Ntimes")
    print(f"  cprofile_report.txt    — hot-path cProfile table")
    print(f"  benchmark_summary.txt  — full numerical results")
    print("=" * 70)


if __name__ == "__main__":
    main()
