# Changelog

---

## 2026-04 — Script and module cleanup

### `sys_solver.py`

**Added**
- Module-level docstring listing all public symbols.
- Full NumPy-style docstrings for every function: `fourier_mode_2d`,
  `sys_modes`, `gcr_systematics`, `sq_mat_tr`, `sq_mat_tr2`, `inv_mat`,
  and `cholesky_inverse`.
- `gcr_systematics`: documented the `Raises` section (both `ValueError`
  paths for solver failure).

**Removed**
- Unused imports: `matplotlib.pylab`, `sklearn.metrics`,
  `scipy.linalg.fractional_matrix_power`, and
  `plotting_functions.master_plotter`.
- Debug `print(nf, nt)` statement inside `fourier_mode_2d` loop.
- `scipy.linalg as sl` alias (retained `scipy.sparse.linalg` which is
  actually used by the solvers).

**Fixed**
- `assert` error messages in `fourier_mode_2d` converted to f-strings.
- `ValueError` messages in `gcr_systematics` converted to f-strings.
- `sq_mat_tr2`: quadrant assignments simplified from four-element index
  notation to two-element slice notation (no logic change).
- `inv_mat`: removed intermediate `diag_el` variable; simplified to a
  single `diag_inv` computation.
- Consistent `===` section banners replacing ad-hoc inline comments.

---

### `sohini_test.py`

**Added**
- Module-level docstring clarifying this is the production 100 k-iteration
  run script and distinguishing it from `sys_sampler_wrapper.py`.
- `calc_ps` docstring with parameters and returns.
- `===` section banners throughout.

**Removed**
- Unused imports: `pylab`, `UVData`, `Quantity`, `units`, `ticker`,
  `cmcrameri`, `sys`.
- All commented-out `op_dir` and `nm_list` alternatives moved to inline
  comments on the active lines (case labels preserved).
- Commented-out uvh5 foreground loading block.
- Commented-out dummy foreground fitting block.
- Redundant `flags_i` variable — inlined directly into `gibbs_sample`.

**Fixed**
- f-strings replace all `.format()` calls.
- `lsts` pre-defined array now passed to `gibbs_sample` instead of
  recomputing `np.linspace(0., 1., Ntimes)` at the call site.
- Consistent spacing and argument alignment in `gibbs_sample` call.

---

## 2026-03-26 — Second version of HERA test notebook

**Added**

- Added new cable reflection notebook in tools/hera_val/
- New notebook loads npy data, creates sky visibilities by doing eor+fg and returns reflection systematics at +/- 1200ns
- Notebooks also plots DL-FR plot for paper

## 2026-03-26 — HERA validation notebook refactor

## 2026-03 — `sohini_test.py` cleanup (first pass)

### `sohini_test.py`

**Fixed**
- Removed spurious `exit()` call (original line 130) that halted the run before
  completing any Gibbs iterations.
- Made all data paths robust via `Path(__file__).parent` — no longer depends on
  the shell's current working directory.
- Added `os.makedirs(op_dir, exist_ok=True)` so the output directory is created
  automatically on first run.
- Passed the pre-defined `lsts` array to `gibbs_sample` instead of recomputing it.

**Removed**
- Unused imports: `pylab`, `UVData`, `Quantity`, `units`, `ticker`, `cmcrameri`, `sys`.

---

## 2026-03 — Package and scripts restructure

### `hydra_pspec/__init__.py`

**Added**
- `sys_solver` to explicit module exports so `from hydra_pspec import sys_solver`
  works without a separate import.

### `scripts/` (new directory)

**Moved**
- `hydra_pspec/calc-vis-cov-matrices.py` → `scripts/calc-vis-cov-matrices.py`.
  The hyphenated filename made it non-importable as a package member.

**Added**
- `scripts/extract_hera_val_data.py` — loads `res/test_data/*.uvh5`, forms
  pseudo-Stokes I, computes the 2D delay–fringe-rate transform, fits a Gaussian
  mask to identify systematic modes, and writes all pre-processed arrays to
  `res/hera_val_npy/`.
- `scripts/hera_val_gibbs_wrapper.py` — Gibbs sampler wrapper configured for
  the HERA validation data in `res/hera_val_npy/`, with all parameter choices
  documented inline.

---

## 2026-02-16 — February 2026 update

General development and run updates. See commit `629730e`.

---

## 2025-09-25 — Sanity save

See commit `985f1b0`.

---

## 2025 — Core development milestones

See `git log` for full details. Selected milestones:

| Commit | Description |
|---|---|
| `3c5598d`, `339b6a3`, `95c25e2` | Fixed critical reshape bug in `sys_modes()` in `sys_solver.py` |
| `238235b` | Final run version before paper |
| `5d98bb9` | Running with bigger priors and fitted foregrounds |
| `4379ea2` | Added masked-data and filtered-data run modes |
| `009d108` | Results after 1 k-iteration test runs |
| `ae9560b` | Saving changes after systematic gain model integration |
| `d3b5178` | Added plotting notebook |
| `28cc1bf` | Repository restructure |
| `b0b592a` | Added notebooks from Andromeda cluster |