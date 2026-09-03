"""
predict_sampler_runtime.py
--------------------------
Predict the total wall-clock time of a Hydra-pspec Gibbs sampler run by
calibrating against the *current* machine, rather than counting FLOPs.

The script builds exactly the inputs that ``sys_sampler_wrapper.py`` builds,
runs a short chain of real Gibbs iterations through ``hp.pspec.gibbs_sample``,
times it with ``time.perf_counter``, and extrapolates the measured
per-iteration cost to a target sample count.

It also reports the machine it measured on, checks the prediction against
known reference runs, and checks whether the target run's preallocated arrays
will actually fit in RAM.

Usage
-----
    # default: reference config (Ntimes=80, Nfreqs=60, Case II, 4 sys params)
    python predict_sampler_runtime.py

    # calibrate the truncated config currently set in sys_sampler_wrapper.py
    python predict_sampler_runtime.py --case III-trunc --ntimes 20 --nfreqs 15

    # longer calibration, different target
    python predict_sampler_runtime.py --cal-iters 100 --target-samples 100000

    # machine/config report only, no sampling
    python predict_sampler_runtime.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.special

REPO_ROOT = Path(__file__).resolve().parent


# =============================================================================
# Configuration mirrored from sys_sampler_wrapper.py
# =============================================================================
# Systematics mode pairs (delay index n, fringe-rate index m), verbatim from
# the comment block in sys_sampler_wrapper.py.
CASES = {
    "I":         [(3, 0),  (4, 0),  (5, 0),  (6, 0)],
    "II":        [(10, 0), (11, 0), (12, 0), (13, 0)],
    "III":       [(3, 20), (4, 20), (5, 20), (6, 20)],
    "I-trunc":   [(3, 0),  (4, 0)],
    "II-trunc":  [(5, 0),  (6, 0)],
    "III-trunc": [(3, 8),  (4, 8)],
}
# The combined case is all three full cases together (12 modes).
CASES["combined"] = CASES["I"] + CASES["II"] + CASES["III"]

# True systematics amplitudes used by the wrapper, keyed by mode count.
SYS_AMPS_TRUE = {
    4: np.array([1. + 4j, 2. + 3j, 3. + 2j, 4. + 1j]),
    2: np.array([12. + 5j, 4. + 20j]),
}

# Defaults from sys_sampler_wrapper.py
DEFAULT_NFGMODES = 10
DEFAULT_TARGET_SAMPLES = 250_000     # sys_sampler_wrapper.py:44  (Niter)
DEFAULT_NOISE_PS_VAL = 0.0004        # sys_sampler_wrapper.py:69
DEFAULT_DATA_SEED = 11               # np.random.seed(11)
DEFAULT_SAMPLER_SEED = 10            # gibbs_sample(seed=10)
DEFAULT_SOLVER_TOL = 1e-13
DEFAULT_NPROC = 1

# Reference runs supplied by the user, keyed by number of systematics
# parameters. These are for sanity-checking only and are never used to build
# the prediction itself. They were produced with the release configuration
# (Ntimes=80, Nfreqs=60), on an unspecified machine.
REFERENCE_RUNS = {
    4: {
        "hours": {"Case I": 38.5, "Case II": 33.4, "Case III": 36.2},
        "samples": 100_000,
        "ntimes": 80,
        "nfreqs": 60,
    },
    12: {
        "hours": {"combined": 55.8},
        "samples": 100_000,
        "ntimes": 80,
        "nfreqs": 60,
    },
}

# Ratio beyond which the reference cross-check is escalated to a warning.
REFERENCE_WARN_FACTOR = 2.0


# =============================================================================
# Machine context
# =============================================================================
def _cpu_model() -> str:
    """Best-effort CPU model string for the current platform."""
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _blas_info() -> str:
    """Name and threading of the BLAS numpy is linked against, if discoverable."""
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        name = blas.get("name", "unknown")
        version = blas.get("version", "")
        return f"{name} {version}".strip()
    except Exception:
        return "unknown"


def gcr_solver_kind() -> dict:
    """
    Detect whether the current code solves the GCR system directly or
    iteratively.

    This matters more than anything else for cross-run comparisons: commit
    9057d57 replaced an iterative gmres solve (maxiter=8000) with a direct LU
    solve, which changes per-iteration cost by a large factor. Any reference
    time logged before that commit is not comparable to a time measured after
    it.
    """
    try:
        src = (REPO_ROOT / "hydra_pspec" / "pspec.py").read_text(errors="ignore")
    except Exception:
        return {"kind": "unknown", "detail": "could not read hydra_pspec/pspec.py"}
    if "np.linalg.solve(A, b)" in src:
        return {"kind": "direct",
                "detail": "GCR uses a direct LU solve (np.linalg.solve)"}
    if "sp.sparse.linalg.gmres" in src or "sparse.linalg.cgs" in src:
        return {"kind": "iterative",
                "detail": "GCR uses an iterative solve (gmres/cgs)"}
    return {"kind": "unknown", "detail": "could not classify the GCR solve"}


def _gpu_status() -> dict:
    """
    Determine whether the sampler can use a GPU.

    hydra_pspec is pure numpy/scipy, so the answer is normally "no". We still
    probe for the common GPU array libraries so the report says whether one is
    merely installed (and therefore might be picked up by a future change)
    versus actually in use.
    """
    installed = []
    for mod in ("cupy", "jax", "torch"):
        try:
            __import__(mod)
            installed.append(mod)
        except Exception:
            pass

    # Does the sampler code itself reference any GPU library?
    used_by_sampler = False
    pkg = REPO_ROOT / "hydra_pspec"
    if pkg.is_dir():
        for src in pkg.glob("*.py"):
            try:
                text = src.read_text(errors="ignore")
            except Exception:
                continue
            if any(f"import {m}" in text for m in ("cupy", "jax", "torch")):
                used_by_sampler = True
                break

    return {
        "gpu_libraries_installed": installed,
        "sampler_uses_gpu": used_by_sampler,
        "summary": (
            "GPU in use by sampler"
            if used_by_sampler
            else "CPU-only (hydra_pspec is pure numpy/scipy; no GPU code path)"
        ),
    }


def machine_context() -> dict:
    """Collect the machine facts needed to reproduce/interpret a prediction."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_total = vm.total
        ram_available = vm.available
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
    except Exception:
        ram_total = ram_available = None
        physical_cores = None
        logical_cores = os.cpu_count()

    thread_env = {
        var: os.environ[var]
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        if var in os.environ
    }

    return {
        "cpu_model": _cpu_model(),
        "cores_physical": physical_cores,
        "cores_logical": logical_cores,
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_available,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": _blas_info(),
        "blas_thread_env": thread_env or None,
        "gpu": _gpu_status(),
        "hostname": platform.node(),
        "gcr_solver": gcr_solver_kind(),
    }


# =============================================================================
# Input construction (mirrors sys_sampler_wrapper.py exactly)
# =============================================================================
def calc_ps(s):
    """Delay power spectrum of visibility data (sys_sampler_wrapper.calc_ps)."""
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    _, nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / nfreqs


def _sys_amps_for(nm_list):
    """
    True systematics amplitudes for a given mode list.

    Uses the wrapper's literal values where they exist for that mode count, and
    otherwise tiles them deterministically. These only set the truth and the
    chain's initial point; they do not change the shape of any operator, so
    per-iteration cost is insensitive to them.
    """
    n = len(nm_list)
    if n in SYS_AMPS_TRUE:
        return SYS_AMPS_TRUE[n].copy()
    base = SYS_AMPS_TRUE[4]
    return np.resize(base, n).astype(complex)


def _import_hydra_pspec():
    """Import hydra_pspec, with a pointed message if the env is wrong."""
    try:
        import hydra_pspec as hp
    except ImportError as exc:
        raise SystemExit(
            f"Could not import hydra_pspec ({exc}).\n"
            f"This script must run in the environment the sampler runs in, so "
            f"that it measures the same numpy/BLAS stack. On this machine that "
            f"is the 'py10' conda env:\n"
            f"    conda run -n py10 python {Path(__file__).name} ...\n"
            f"    # or: ~/miniconda3/envs/py10/bin/python {Path(__file__).name} ..."
        ) from exc
    return hp


def build_inputs(ntimes, nfreqs, nfgmodes, nm_list, noise_ps_val=DEFAULT_NOISE_PS_VAL,
                 data_seed=DEFAULT_DATA_SEED):
    """
    Build the full set of ``gibbs_sample`` inputs, reproducing the ordering of
    sys_sampler_wrapper.py.

    The ordering matters: ``sys_modes`` is built from the synthetic 100-120 MHz
    grid *before* ``freqs`` is overwritten by the loaded array (wrapper lines
    112-116 vs 142). Reproducing that keeps the operators bit-identical to the
    real run.
    """
    hp = _import_hydra_pspec()

    np.random.seed(data_seed)

    # --- Frequency and LST grids (wrapper lines 102-103) ---
    freqs = np.linspace(100., 120., 120)[:nfreqs]      # MHz
    lsts = np.linspace(0., 1., ntimes)

    # --- Systematics model (built on the synthetic grid, before reload) ---
    sys_amps_true = _sys_amps_for(nm_list)
    sys_modes = hp.sys_solver.sys_modes(
        freqs_Hz=freqs * 1e6,
        times_sec=lsts * 24. / (2. * np.pi) * 3600.,
        modes=nm_list,
    )
    sys_prior = 100.**2 * np.eye(sys_amps_true.size)
    gain_true = (1. + (sys_modes @ sys_amps_true).reshape([nfreqs, ntimes]).T)

    # --- EoR field, loaded (dummy_flag = False branch) ---
    data = REPO_ROOT / "res" / "npy_data"
    eor_true = np.load(data / "eor_true.npy")[:ntimes, :nfreqs]
    lsts = np.load(data / "lsts_full.npy")[:ntimes]
    freqs = np.load(data / "freqs_full.npy")[:nfreqs] * 10e-6
    ps_true = calc_ps(eor_true)

    # --- Foregrounds ---
    fg_true = np.load(data / "fg_true.npy")[:ntimes, :nfreqs]
    fgmodes = np.array([
        scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
        for i in range(nfgmodes)
    ]).T

    # --- Priors ---
    ps_prior = np.column_stack((
        1e-7 * np.ones(nfreqs),
        1e-1 * np.ones(nfreqs),
    ))

    # --- Noise ---
    noise_ps_true = noise_ps_val * np.ones(nfreqs)
    N_true = hp.pspec.covariance_from_pspec(noise_ps_true, hp.utils.fourier_operator(nfreqs, unitary=True))
    Ninv = np.diag(1. / np.diag(N_true))
    n = (
        np.sqrt(N_true)
        @ (np.random.randn(freqs.size, ntimes) + 1.j * np.random.randn(freqs.size, ntimes))
        / np.sqrt(2.)
    )

    # --- Data ---
    d = gain_true * (fg_true + eor_true) + n.T

    return dict(
        vis=d,
        flags=np.ones((len(freqs),), dtype=int),
        signal_ps_initial=ps_true,
        fg_modes=fgmodes,
        Ninv=Ninv,
        signal_ps_prior=ps_prior.T,
        seed=DEFAULT_SAMPLER_SEED,
        freqs=freqs,
        lsts=np.linspace(0., 1., ntimes),
        map_estimate=False,
        nproc=DEFAULT_NPROC,
        sys_modes=sys_modes,
        sys_prior=sys_prior,
        sys_initial=sys_amps_true,
        solver_tol=DEFAULT_SOLVER_TOL,
        sample_systematics=True,
        sample_eor_fg=True,
        sample_signal_ps=True,
        sky_model_initial=fg_true + eor_true,
    )


# =============================================================================
# Calibration
# =============================================================================
def _run_chain(kwargs, niter, out_dir, sampler_verbose, log_path):
    """
    Run ``niter`` real Gibbs iterations and return elapsed wall-clock seconds.

    ``write_Niter`` is set above ``niter`` so that no bulk .npy dump happens
    inside the timed region -- that write is a one-off at the end of a real run,
    not a per-iteration cost. The per-iteration HDF5 append and flush *are*
    included, because a real run pays them on every iteration.
    """
    hp = _import_hydra_pspec()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stdout_saved = sys.stdout
    log_handle = open(log_path, "a") if log_path is not None else None
    try:
        if log_handle is not None:
            sys.stdout = log_handle
        t0 = time.perf_counter()
        hp.pspec.gibbs_sample(
            Niter=niter,
            verbose=sampler_verbose,
            write_Niter=niter + 1,   # suppress the bulk .npy dump
            out_dir=str(out_dir),
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
    finally:
        sys.stdout = stdout_saved
        if log_handle is not None:
            log_handle.close()
    return elapsed


def calibrate(kwargs, cal_iters, warmup_iters, out_root, sampler_verbose,
              time_budget, log_path, min_cal_seconds=0.):
    """
    Time real Gibbs iterations on this machine.

    A short warm-up chain is run and discarded: it absorbs first-touch page
    faults, BLAS thread spin-up and HDF5 file creation, none of which recur
    across a long run. The warm-up is also timed, and its per-iteration cost is
    reported as an independent second measurement -- if the two disagree
    badly, the machine's timing is not stable enough to extrapolate from.
    """
    result = {"warmup_iters": warmup_iters, "requested_cal_iters": cal_iters}

    warm_dir = Path(out_root) / "warmup"
    warm_elapsed = _run_chain(kwargs, warmup_iters, warm_dir, sampler_verbose, log_path)
    warm_per_iter = warm_elapsed / warmup_iters
    result["warmup_seconds"] = warm_elapsed
    result["warmup_per_iter_seconds"] = warm_per_iter

    # Size the timed phase from the warm-up estimate: long enough to be a
    # meaningful sample, short enough to stay within the time budget.
    requested = cal_iters
    if min_cal_seconds > 0:
        needed = int(np.ceil(min_cal_seconds / max(warm_per_iter, 1e-9)))
        if needed > cal_iters:
            cal_iters = needed
    if time_budget is not None:
        affordable = max(1, int(time_budget / max(warm_per_iter, 1e-9)))
        cal_iters = min(cal_iters, affordable)
    if cal_iters != requested:
        result["cal_iters_adjusted_from"] = requested
    result["cal_iters"] = cal_iters

    cal_dir = Path(out_root) / "calibration"
    cal_elapsed = _run_chain(kwargs, cal_iters, cal_dir, sampler_verbose, log_path)

    result["cal_seconds"] = cal_elapsed
    result["per_iter_seconds"] = cal_elapsed / cal_iters
    result["warmup_vs_cal_ratio"] = (
        warm_per_iter / result["per_iter_seconds"] if result["per_iter_seconds"] > 0 else float("nan")
    )
    return result


# =============================================================================
# Prediction, memory and reference checks
# =============================================================================
def memory_footprint(target_samples, ntimes, nfreqs, nfgmodes):
    """
    Bytes preallocated by ``gibbs_sample`` for the full chain.

    gibbs_sample allocates every output array at full length up front
    (pspec.py:852-859), so this is committed before the first iteration runs.
    """
    c, f = 16, 8  # complex128, float64
    parts = {
        "signal_amps": target_samples * ntimes * nfreqs * c,
        "signal_ps": target_samples * nfreqs * f,
        "fg_amps": target_samples * ntimes * nfgmodes * c,
        "sys_amps": None,   # filled by caller (needs nsys)
        "chisq": target_samples * ntimes * nfreqs * f,
        "ln_post": target_samples * f,
    }
    return parts


def predict(per_iter_seconds, target_samples):
    total_seconds = per_iter_seconds * target_samples
    return {
        "target_samples": target_samples,
        "per_iter_seconds": per_iter_seconds,
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600.,
        "total_days": total_seconds / 86400.,
    }


def reference_check(per_iter_seconds, nsys, ntimes, nfreqs, solver_kind=None):
    """
    Cross-check the measured per-iteration cost against the logged reference
    runs with the same systematics parameter count.

    Returns a dict with a verdict and the messages to print. A large deviation
    is reported, not silently absorbed: it means the machine, the configuration
    or the code has changed relative to when those numbers were logged.
    """
    ref = REFERENCE_RUNS.get(nsys)
    if ref is None:
        return {
            "status": "no_reference",
            "messages": [
                f"No reference run logged for {nsys} systematics parameters "
                f"(references exist for {sorted(REFERENCE_RUNS)}). "
                "Prediction reported without cross-check."
            ],
        }

    ref_samples = ref["samples"]
    predicted_hours = per_iter_seconds * ref_samples / 3600.
    ref_hours = list(ref["hours"].values())
    ref_lo, ref_hi = min(ref_hours), max(ref_hours)
    ref_mean = sum(ref_hours) / len(ref_hours)
    ratio = predicted_hours / ref_mean

    dims_match = (ntimes == ref["ntimes"] and nfreqs == ref["nfreqs"])

    messages = [
        f"Extrapolated to {ref_samples:,} samples at {nsys} systematics params: "
        f"{predicted_hours:.2f} h",
        "Reference runs: "
        + ", ".join(f"{k} {v} h" for k, v in ref["hours"].items())
        + f"  (range {ref_lo}-{ref_hi} h, mean {ref_mean:.1f} h)",
        f"Ratio predicted/reference-mean: {ratio:.2f}x",
    ]

    if not dims_match:
        messages.append(
            f"NOTE: calibration ran at Ntimes={ntimes}, Nfreqs={nfreqs}, but the "
            f"reference runs used Ntimes={ref['ntimes']}, Nfreqs={ref['nfreqs']}. "
            "The comparison is NOT like-for-like; a large ratio is expected and "
            "does not by itself indicate a machine difference."
        )

    if ratio > REFERENCE_WARN_FACTOR or ratio < 1. / REFERENCE_WARN_FACTOR:
        status = "warn"
        if ratio > 1:
            factor, direction = ratio, "SLOWER"
        else:
            factor, direction = 1. / ratio, "FASTER"
        messages.insert(
            0,
            f"WARNING: this configuration is {factor:.1f}x {direction} than the "
            f"reference runs at the same systematics parameter count -- beyond "
            f"the {REFERENCE_WARN_FACTOR:g}x threshold. Do not use the reference "
            f"times to sanity-check this prediction until the cause is known.",
        )
        causes = []
        if not dims_match:
            causes.append(
                "(a) DIMENSIONS DIFFER from the reference run (see the NOTE "
                "above) -- this alone explains a large ratio."
            )
        if solver_kind and solver_kind.get("kind") == "direct":
            causes.append(
                "(b) THE GCR SOLVER CHANGED: this code solves the GCR system "
                "directly (np.linalg.solve), whereas the reference runs predate "
                "commit 9057d57 and used an iterative gmres solve with "
                "maxiter=8000. A direct LU solve on a system this small is far "
                "cheaper per iteration, so a large speed-up here is expected "
                "and legitimate."
            )
        causes.append(
            "(c) a different machine, or different BLAS threading "
            "(OMP_NUM_THREADS et al.) than the reference runs used."
        )
        messages.append("Candidate causes:")
        messages.extend("  " + c for c in causes)
        if dims_match and solver_kind and solver_kind.get("kind") == "direct":
            messages.append(
                "Assessment: dimensions match the reference runs, so (a) is "
                "ruled out. (b) is the most likely explanation. The prediction "
                "itself is still a direct measurement of THIS code on THIS "
                "machine and remains the number to plan with."
            )
    else:
        status = "ok"
        messages.insert(
            0,
            f"Consistent with the reference runs ({ratio:.2f}x, within "
            f"{REFERENCE_WARN_FACTOR:g}x).",
        )

    return {
        "status": status,
        "predicted_hours_at_reference": predicted_hours,
        "reference_mean_hours": ref_mean,
        "reference_hours": ref["hours"],
        "ratio": ratio,
        "dimensions_match_reference": dims_match,
        "messages": messages,
    }


# =============================================================================
# Reporting
# =============================================================================
def _gb(nbytes):
    return nbytes / 1024.**3


def _fmt_duration(seconds):
    h = seconds / 3600.
    if h < 1:
        return f"{seconds / 60.:.1f} min"
    if h < 48:
        return f"{h:.2f} h"
    return f"{h:.1f} h ({h / 24.:.2f} days)"


def print_report(cfg, machine, cal, pred, refchk, mem):
    w = 78
    print("=" * w)
    print("Hydra-pspec Gibbs sampler -- measured runtime prediction")
    print("=" * w)

    print("\n-- Machine (measured on) " + "-" * (w - 25))
    print(f"  Host              : {machine['hostname']}")
    print(f"  CPU               : {machine['cpu_model']}")
    print(f"  Cores             : {machine['cores_physical']} physical / "
          f"{machine['cores_logical']} logical")
    if machine["ram_total_bytes"]:
        print(f"  RAM               : {_gb(machine['ram_total_bytes']):.1f} GiB total, "
              f"{_gb(machine['ram_available_bytes']):.1f} GiB available")
    print(f"  GPU               : {machine['gpu']['summary']}")
    if machine["gpu"]["gpu_libraries_installed"]:
        print(f"                      (installed but unused: "
              f"{', '.join(machine['gpu']['gpu_libraries_installed'])})")
    print(f"  Platform          : {machine['platform']}")
    print(f"  Python / numpy    : {machine['python']} / {machine['numpy']}")
    print(f"  BLAS              : {machine['blas']}")
    print(f"  BLAS thread env   : {machine['blas_thread_env'] or 'unset (library default)'}")
    print(f"  GCR solve         : {machine['gcr_solver']['detail']}")

    print("\n-- Configuration calibrated " + "-" * (w - 28))
    print(f"  Case              : {cfg['case']}  {cfg['nm_list']}")
    print(f"  Ntimes x Nfreqs   : {cfg['ntimes']} x {cfg['nfreqs']}")
    print(f"  Nfgmodes          : {cfg['nfgmodes']}")
    print(f"  Systematics params: {cfg['nsys']}")
    print(f"  GCR system size   : {cfg['nfreqs'] + cfg['nfgmodes']} params x "
          f"{cfg['ntimes']} times")
    print(f"  nproc             : {DEFAULT_NPROC}")

    print("\n-- Calibration (real Gibbs iterations, wall clock) " + "-" * (w - 51))
    print(f"  Warm-up           : {cal['warmup_iters']} iters in "
          f"{cal['warmup_seconds']:.2f} s "
          f"({cal['warmup_per_iter_seconds']:.4f} s/iter, discarded)")
    if "cal_iters_adjusted_from" in cal:
        direction = ("raised" if cal["cal_iters"] > cal["cal_iters_adjusted_from"]
                     else "reduced")
        print(f"  Note              : {direction} from "
              f"{cal['cal_iters_adjusted_from']} iters "
              f"(--min-cal-seconds / --time-budget)")
    print(f"  Measured          : {cal['cal_iters']} iters in "
          f"{cal['cal_seconds']:.2f} s")
    print(f"  Per iteration     : {cal['per_iter_seconds']:.4f} s")
    print(f"  Warm-up/measured  : {cal['warmup_vs_cal_ratio']:.2f}x "
          + ("(stable)" if 0.7 <= cal["warmup_vs_cal_ratio"] <= 1.4
             else "(UNSTABLE -- timing varies between chains; treat the "
                  "prediction as approximate)"))

    print("\n-- Prediction " + "-" * (w - 14))
    print(f"  Target samples    : {pred['target_samples']:,}")
    print(f"  PREDICTED TOTAL   : {_fmt_duration(pred['total_seconds'])}")
    print(f"                      ({pred['total_seconds']:.0f} s, "
          f"{pred['total_hours']:.2f} h, {pred['total_days']:.2f} days)")

    print("\n-- Cross-check against logged reference runs " + "-" * (w - 45))
    for msg in refchk["messages"]:
        print(f"  {msg}")

    print("\n-- Memory footprint of the target run " + "-" * (w - 38))
    print(f"  gibbs_sample preallocates all output arrays at full length before "
          f"iteration 1:")
    for name, nbytes in mem["parts"].items():
        print(f"    {name:<14s}: {_gb(nbytes):8.2f} GiB")
    print(f"  {'TOTAL':<16s}: {_gb(mem['total']):8.2f} GiB")
    if mem.get("warning"):
        print(f"  {mem['warning']}")

    print("\n-- Costs NOT included in the per-iteration figure " + "-" * (w - 50))
    print("  * One-off setup before the chain (input build, array allocation).")
    print(f"  * The final bulk .npy dump: sys_sampler_wrapper.py sets "
          f"write_Niter = Niter,")
    print(f"    so ~{_gb(mem['total']):.1f} GiB is written once at the end. "
          f"Add minutes, not hours.")
    print("  * The per-iteration HDF5 append and flush ARE included (measured).")
    print("=" * w)


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Predict Gibbs sampler total runtime by timing real "
                    "iterations on this machine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", default="II", choices=sorted(CASES),
                   help="Systematics mode set. Full cases have 4 params, "
                        "'combined' has 12, '-trunc' variants have 2.")
    p.add_argument("--nm-list", default=None,
                   help="Override --case with an explicit mode list, "
                        'e.g. "3,0 4,0 5,0 6,0".')
    p.add_argument("--ntimes", type=int, default=80,
                   help="Number of times. 80 = reference config; the wrapper "
                        "is currently set to 20.")
    p.add_argument("--nfreqs", type=int, default=60,
                   help="Number of frequencies. 60 = reference config; the "
                        "wrapper is currently set to 15.")
    p.add_argument("--nfgmodes", type=int, default=DEFAULT_NFGMODES,
                   help="Number of foreground modes.")
    p.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES,
                   help="Sample count to extrapolate to (wrapper's Niter).")
    p.add_argument("--cal-iters", type=int, default=30,
                   help="Timed calibration iterations.")
    p.add_argument("--warmup-iters", type=int, default=3,
                   help="Discarded warm-up iterations.")
    p.add_argument("--min-cal-seconds", type=float, default=15.,
                   help="Grow cal-iters until the timed phase lasts at least "
                        "this long, so the per-iteration figure is not read "
                        "off a handful of fast iterations. Use 0 to disable.")
    p.add_argument("--time-budget", type=float, default=300.,
                   help="Approximate seconds allowed for the timed phase; "
                        "cal-iters is reduced to fit. Use 0 to disable.")
    p.add_argument("--out-dir", default=None,
                   help="Scratch directory for calibration outputs "
                        "(default: a temporary directory, deleted afterwards).")
    p.add_argument("--sampler-verbose", action="store_true",
                   help="Let gibbs_sample print its per-iteration table "
                        "(captured to a log file, not the terminal).")
    p.add_argument("--json", dest="json_path", default=None,
                   help="Also write the full result as JSON to this path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report machine and configuration without sampling.")
    args = p.parse_args(argv)

    if args.nm_list:
        nm_list = [tuple(int(x) for x in tok.split(",")) for tok in args.nm_list.split()]
        case_name = "custom"
    else:
        nm_list = CASES[args.case]
        case_name = args.case

    nsys = len(nm_list)
    cfg = {
        "case": case_name,
        "nm_list": nm_list,
        "ntimes": args.ntimes,
        "nfreqs": args.nfreqs,
        "nfgmodes": args.nfgmodes,
        "nsys": nsys,
    }

    machine = machine_context()

    # Memory footprint of the *target* run.
    parts = memory_footprint(args.target_samples, args.ntimes, args.nfreqs, args.nfgmodes)
    parts["sys_amps"] = args.target_samples * nsys * 16
    total_bytes = sum(parts.values())
    mem = {"parts": parts, "total": total_bytes}
    if machine["ram_total_bytes"] and total_bytes > 0.8 * machine["ram_total_bytes"]:
        mem["warning"] = (
            f"WARNING: {_gb(total_bytes):.1f} GiB exceeds 80% of this machine's "
            f"{_gb(machine['ram_total_bytes']):.1f} GiB of RAM. The run will "
            f"swap or be killed before it finishes, whatever the time "
            f"prediction says. Reduce Niter, Ntimes/Nfreqs, or stream samples "
            f"to disk instead of preallocating."
        )

    if args.dry_run:
        print(json.dumps({"config": cfg, "machine": machine,
                          "memory_gib": {k: _gb(v) for k, v in parts.items()},
                          "memory_total_gib": _gb(total_bytes),
                          "memory_warning": mem.get("warning")},
                         indent=2, default=str))
        return 0

    print(f"Building inputs (Ntimes={args.ntimes}, Nfreqs={args.nfreqs}, "
          f"case {case_name}, {nsys} sys params)...", flush=True)
    kwargs = build_inputs(args.ntimes, args.nfreqs, args.nfgmodes, nm_list)

    tmp_root = args.out_dir or tempfile.mkdtemp(prefix="gibbs_cal_")
    log_path = Path(tmp_root) / "sampler_stdout.log"
    Path(tmp_root).mkdir(parents=True, exist_ok=True)

    print(f"Calibrating: {args.warmup_iters} warm-up + up to {args.cal_iters} "
          f"timed iterations...", flush=True)
    try:
        cal = calibrate(
            kwargs,
            cal_iters=args.cal_iters,
            warmup_iters=args.warmup_iters,
            out_root=tmp_root,
            sampler_verbose=args.sampler_verbose,
            time_budget=(args.time_budget if args.time_budget > 0 else None),
            log_path=log_path,
            min_cal_seconds=args.min_cal_seconds,
        )
    finally:
        if args.out_dir is None:
            shutil.rmtree(tmp_root, ignore_errors=True)

    pred = predict(cal["per_iter_seconds"], args.target_samples)
    refchk = reference_check(cal["per_iter_seconds"], nsys, args.ntimes,
                             args.nfreqs, solver_kind=machine["gcr_solver"])

    print()
    print_report(cfg, machine, cal, pred, refchk, mem)

    if args.json_path:
        payload = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": cfg,
            "machine": machine,
            "calibration": cal,
            "prediction": pred,
            "reference_check": refchk,
            "memory_bytes": {**parts, "total": total_bytes},
            "memory_warning": mem.get("warning"),
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nJSON written to {args.json_path}")

    return 2 if refchk["status"] == "warn" else 0


if __name__ == "__main__":
    sys.exit(main())
