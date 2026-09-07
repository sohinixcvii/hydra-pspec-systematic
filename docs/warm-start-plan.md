# Warm-start / chain-resume plan

**Status:** implemented. See [`warm-start.md`](warm-start.md) for the
resulting feature, the API and how each blocker below was resolved. This
document is kept as the original survey and rationale.
**Target:** implement on a branch off `main`.

## Context

The 250k-iteration Case III run (`paper_plots/250k_run/low_dl_fr_20`) completed
its chain but died in the post-chain `.npy` dump with
`OSError: Not enough free space to write 19200000000 bytes`. The samples
survived only because `GibbsSampleH5Writer` flushes every iteration. There is
currently no way to continue a chain, so any interruption means restarting from
iteration zero.

This document is the survey of what has to change to support a warm start:
stopping a chain and resuming it from its last recorded state.

Two things it depends on:

- **The seed fix** (removal of the per-call `np.random.seed(None)` in
  `gcr_fg_and_signal_per_time`, and seeding once in `gibbs_sample`). Without a
  deterministic RNG stream there is no way to tell a correct resume from a
  buggy one. Line numbers below refer to the working tree **with that fix
  applied**.
- The observation that the Gibbs state is tiny — see next section.

---

## Finding: the chain state is two small arrays

Iteration *i* of `gibbs_step()` consumes exactly two things from iteration
*i−1*, set at `pspec.py:911-912`:

```python
signal_ps_current = signal_ps[i]
sys_amps_current  = sys_amps[i]
```

`signal_amps` and `fg_amps` are regenerated from those two by the GCR step;
`chisq` and `ln_post` are diagnostics. So the Markov state is
`(signal_ps, sys_amps)` — one `(Nfreqs,)` float array and one
`(Nsys_modes,)` complex array, a few hundred bytes.

Both are already arguments of `gibbs_sample()`: `signal_ps_initial` and
`sys_initial`. A crude warm start is therefore possible today by reading the
last row of `gibbs_samples.h5` and passing it back in. Everything below is
about making that correct, safe, and non-destructive.

**Exception:** `sky_model_initial` also becomes state, but only when
`sample_eor_fg=False`, where `gibbs_step` passes it through untouched. The
wrapper currently supplies the truth (`fg_true + eor_true`), which is harmless
only because production runs use `sample_eor_fg=True`.

---

## Blockers

### B1 — The HDF5 writer truncates on every run

`pspec.py:878`:

```python
h5_writer = utils.GibbsSampleH5Writer(fp=out_dir, overwrite=True)
```

Hardcoded `overwrite=True`; `_open_gibbs_sample_h5` then does
`os.remove(h5_path)`. A resume pointed at the same `out_dir` **deletes the
chain it is resuming from, before reading it.** This is the most dangerous
single line for warm start and should be the first thing changed.

### B2 — There is no reader

`utils.py` has three write paths (`write_numpy_files`,
`append_gibbs_sample_h5`, `GibbsSampleH5Writer`) and no read path. A resume
needs to read the last sample and the current chain length.

### B3 — Crash-truncated datasets must be repaired on open

`_append_arrays_to_h5` (`utils.py:378-414`) iterates over the six arrays and
resizes each one independently. A crash between two of those resizes leaves,
say, `signal_amps` with N+1 rows and `ln_post` with N.

Any resume must compute `n = min(len(d) for d in datasets)` and truncate all
six to `n` before reading the last sample. Skip this and the datasets
silently desynchronise — every downstream index is off by one for the
remainder of the chain, with no error raised.

### B4 — RNG state is not persisted

This is the statistical-correctness issue, not just a reproducibility one.

After the seed fix the chain is a single deterministic MT19937 stream seeded
once at `pspec.py:830`. A resume that calls `np.random.seed(seed)` again
**replays the same random numbers from the top**, so the resumed segment
reuses the exact draws of the first segment. That is a genuine correlation
artefact in the posterior samples.

Options:

1. **Exact continuation (recommended).** Store `np.random.get_state()` — a
   624-element `uint32` array plus position and Gaussian cache — in the h5
   next to the samples; restore with `np.random.set_state()` on resume.
   Segmented and uninterrupted runs then produce bit-identical chains.
2. **Derived reseed (fallback).** Reseed with e.g. `seed + start_iter`.
   Simpler; still reproducible given the resume points, but the chain no
   longer matches an uninterrupted run.

Either way, the resume path must **not** call `np.random.seed()`.

### B5 — `write_numpy_files` will clobber the chain with the new segment only

`utils.py:325-330` writes fixed filenames via `np.save` — no append, no
sequence number. Resuming 250k → 300k leaves `dps-eor.npy` holding 50k
samples.

This matters more than it first appears, because **all post-processing reads
the `.npy` files, not the h5**:

- `post_processing_funcs.py:612-616`
- `plotting_functions.py:503-504`

A truncated chain would be analysed with no error — `post_processing_funcs.py:637`
only prints a WARNING when the length disagrees with `args.Niter`.

Fix: make the `.npy` files a *derived export of the full h5*, written once at
the end of a run, rather than the primary output.

### B6 — No provenance to validate a resume against

Nothing in the h5 records `seed`, `Ntimes`, `Nfreqs`, `nm_list`, the priors, or
the data. Nothing prevents resuming a Case III chain against Case I
`sys_modes`.

The wrapper rebuilds the data `d` from scratch on every run
(`sys_sampler_wrapper.py:196`). That is deterministic under
`np.random.seed(11)`, so it currently happens to work — but change `Ntimes`,
`Nfreqs`, or `dummy_flag` and a resumed chain silently starts sampling
*different data*. Store the config plus a hash of `vis` / `Ninv` / `sys_modes`
as h5 attributes and refuse a mismatched resume.

---

## Secondary benefit: memory

Preallocation at `pspec.py:856-864` sizes every array to the full `Niter`. For
the 250k run at 80 × 60:

| array | size |
|---|---|
| `signal_amps` | 19.2 GB |
| `chisq` | 9.6 GB |
| `fg_amps` | 3.2 GB |
| **total resident** | **~32 GB** |

With warm start, `Niter` becomes the *segment* length: ~1.3 GB for
10k-iteration chunks. This is an argument for building the feature even
ignoring crash recovery — and it is the same 19.2 GB array that exhausted the
disk, since `write_numpy_files` dumps it verbatim.

---

## Change list

### `hydra_pspec/utils.py` — most of the work, ~120 lines

- `read_gibbs_sample_h5(fp, index=-1)` → dict of the requested sample plus the
  total chain length.
- `h5_chain_length(fp)` → min over datasets; `truncate_h5_to(fp, n)` for B3.
- `save_rng_state(f)` / `load_rng_state(f)` — one small dataset plus attrs.
- `write_chain_metadata(f, **kw)` / `check_chain_metadata(f, **kw)` for B6.
- `export_npy_from_h5(fp)` — becomes the way `.npy` files are produced,
  replacing the in-loop `write_numpy_files` call.
- `GibbsSampleH5Writer`: add `truncate_to()`. `overwrite` is already a
  parameter and only needs plumbing.

### `hydra_pspec/pspec.py` — ~40 lines, all inside `gibbs_sample`

- New `resume=False` argument.
- `pspec.py:825-831` — restore RNG state when resuming; seed only when not.
- `pspec.py:866-867` — initialise `signal_ps_current` / `sys_amps_current`
  from the h5 last sample when resuming; ignore (or assert absent)
  `signal_ps_initial` / `sys_initial`.
- `pspec.py:878` — `overwrite=not resume`.
- `pspec.py:913` — persist RNG state with each append, or every `write_Niter`
  (accepting rollback to the last checkpoint on a crash).
- `pspec.py:924, 938` — do not write partial `.npy` on a resumed run; export
  from the h5 instead.
- Decide `Niter` semantics. **Recommend: additional iterations, not total** —
  it composes with a job scheduler and does not require the caller to know the
  current chain length.

### `sys_sampler_wrapper.py`

- `--resume` flag.
- Load `data_true.npy` / `eor_true.npy` / `fg_true.npy` from `op_dir` instead
  of regenerating them; skip the `np.save` calls that would overwrite them.

### Post-processing

No change needed **if** the `.npy` files remain a full-chain export. To read
the h5 directly instead, `post_processing_funcs.py:611-617` is the single
place to add a branch.

---

## Acceptance test

Write this before the implementation:

> Run 10 iterations straight. Separately run 5, resume for 5 more. Assert all
> six outputs are bit-identical.

This test is only meaningful because of the seed fix — before it, the two runs
would have differed regardless, and a resume bug would have been
indistinguishable from RNG noise.

---

## Suggested order

1. B1 + B2 + B3 — crash-safe reader and `overwrite=not resume`. This alone
   recovers a chain killed by a full disk.
2. B4 — RNG persistence. Required before any resumed chain is used for
   science.
3. B5 — `.npy` as a derived export. Removes the silent-truncation trap and the
   duplicate 19.2 GB write.
4. B6 — metadata validation.
5. Segment-length `Niter` and the memory win.

## Effort

Roughly 150–200 lines, concentrated in `utils.py`. No change to the sampler
maths; the public API grows by one keyword argument. About a day including the
acceptance test.

The riskiest parts are B1 and B3: resume code that reads a file it has already
deleted, or reads desynchronised datasets, fails in ways that look like
physics.
