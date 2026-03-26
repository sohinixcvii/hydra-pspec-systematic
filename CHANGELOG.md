# Changelog

---

## 2026-03-26 — HERA validation notebook refactor

### `tools/hera_val/test-3.2.0.ipynb` (rewrite, 68 → 16 cells)

**Decluttered**
- Removed abstract, description, summary, and software-version markdown sections.
- Removed hardcoded server paths (`/nvme2/scratch/sohini/...`, `/users/heramgr/...`).
- Removed `sim_prep` import and its `sys.path.append` hack.
- Removed `UVData`/`pyuvdata`/`h5py`/`hera_pspec`/`yaml` imports.
- Removed EW baseline selection cells — data is already a pre-extracted single-baseline 2D array.
- Removed broken `{{print(datetime.now())}}` template cell.
- Removed `type()`/`pwd`/empty debug cells.
- Removed Gaussian mask generation and paper-plotting cells.
- Removed all uvh5 file-save cells and `nm_list` scratch/debug cells (cells 45–67).

**Changed to `.npy` file system**
- Data loaded from `res/hera_val_npy/` via `numpy.load`; no UVData/pyuvdata file I/O.
- Added LST → fake-JD time conversion cell for correct fringe-rate axis in `plot_waterfalls`.

**Bug fixes in `plot_waterfalls`**
- `limit_drng == 'all'` was mutated inside the per-panel loop, causing incorrect
  dynamic-range clipping from the second panel onward. Fixed by computing
  `clip_drng` as a single boolean expression outside the mutation path.

**Added**
- Third plot panel: **Systematics = Corrupted − Clean** diagnostic waterfall.
- `wfall_kw` dict for DRY waterfall calls across all three figures.

---

## 2026-03 — `sohini_test.py` cleanup

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
