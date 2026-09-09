# Warm start — stopping and resuming a Gibbs chain

**Status:** implemented.
**Plan / survey:** [`warm-start-plan.md`](warm-start-plan.md).
**Acceptance test:** `tests/test_warm_start.py`.

A chain can now be stopped and continued from its last recorded sample. A
segmented run produces a chain **bit-identical** to the uninterrupted run it
replaces, so a resume is not a statistical compromise.

---

## Quick start

```bash
# Start a chain: 10,000 iterations
python sys_sampler_wrapper.py --Niter 10000 --out-dir ./run_dir

# Continue it: 10,000 MORE iterations (total 20,000)
python sys_sampler_wrapper.py --Niter 10000 --out-dir ./run_dir --resume
```

`--Niter` is the number of *additional* iterations under `--resume`, not the
target total. That composes with a job scheduler: every segment is the same
command, and no caller has to know the current chain length.

From Python:

```python
from hydra_pspec.pspec import gibbs_sample

# segment 1
gibbs_sample(..., Niter=5000, out_dir='run_dir', seed=10)

# segment 2 — signal_ps_initial / sys_initial / seed are ignored
gibbs_sample(..., Niter=5000, out_dir='run_dir', seed=10, resume=True)
```

After both calls `run_dir/gibbs_samples.h5` holds 10,000 samples and the
`.npy` files hold all 10,000.

### Recovering a chain that was killed

A resume is also the recovery path for an interrupted run — including the
failure that motivated this work, where the 250k Case III run finished its
chain and then died writing the 19.2 GB `.npy` dump. Point `--resume` at the
output directory and the chain continues from the last sample that reached
disk. `GibbsSampleH5Writer` flushes every iteration, so at most the iteration
in progress is lost.

If the process died *during* an append, the HDF5 datasets are left at
different lengths; the resume repairs that automatically (see B3 below).

---

## What is actually saved and restored

Iteration *i* of `gibbs_step()` consumes exactly two things from iteration
*i−1*: `signal_ps` and `sys_amps`. `signal_amps` and `fg_amps` are regenerated
from that pair by the GCR step, and `chisq` / `ln_post` are diagnostics. The
Markov state is therefore one `(Nfreqs,)` float array and one
`(Nsys_modes,)` complex array — a few hundred bytes.

Restoring it is not sufficient on its own. A resume also restores the NumPy
global RNG state, without which the resumed segment would replay the random
draws the first segment already used. Both live in `gibbs_samples.h5`:

| In the HDF5 file | Contents |
|---|---|
| `signal_amps`, `signal_ps`, `fg_amps`, `sys_amps`, `chisq`, `ln_post` | the samples, one row per iteration |
| `rng_state/keys` + attrs | MT19937 state as of the end of the last sample |
| `meta_*` file attributes | run configuration and input-data hashes |

**Exception.** `sky_model_initial` is also part of the state when
`sample_eor_fg=False`, where `gibbs_step` passes it through untouched. It is
not read back from the file; the caller must pass the same array on resume,
and the metadata check (below) enforces that it hashes the same. Production
runs use `sample_eor_fg=True`, where the argument is overwritten in the first
conditional sampling step and does not matter.

---

## How each blocker was resolved

The plan identified six blockers. Each is listed here with the code that
addresses it, because each one fails in a way that looks like physics rather
than like a bug.

### B1 — the HDF5 writer no longer truncates on resume

`pspec.py` previously opened the writer with a hardcoded `overwrite=True`,
which deletes the file. A resume pointed at the same directory would have
destroyed the chain it was about to read. It is now
`overwrite=not resume`.

### B2 — there is a reader

`utils.read_gibbs_sample_h5(fp, index=-1)` returns the requested sample plus
the chain length. `utils.h5_chain_length(fp)` returns the length alone.

### B3 — crash-truncated datasets are repaired before anything is read

`_append_arrays_to_h5` resizes the six datasets one at a time. A crash between
two of those resizes leaves, say, `signal_amps` with N+1 rows and `ln_post`
with N. Reading the "last" sample of such a file mixes rows from two
iterations, and every subsequent index is off by one for the rest of the
chain, with nothing raised.

`utils.repair_h5_chain(fp)` computes `n = min(len(d) for d in datasets)` and
truncates all of them to `n`. `gibbs_sample()` calls it as the very first step
of a resume, before the reader runs.

### B4 — the RNG state is persisted

After the seed fix the chain is a single deterministic MT19937 stream seeded
once at the top of `gibbs_sample()`. Calling `np.random.seed(seed)` again on
resume would replay that stream from the start, so the resumed segment would
reuse the exact draws of the first — a genuine correlation artefact in the
posterior samples, not merely a reproducibility wart.

The implementation takes **option 1 of the plan, exact continuation**:
`utils.save_rng_state()` writes the 624-element MT19937 key array plus the
stream position and Gaussian cache alongside every sample;
`utils.load_rng_state()` restores it with `np.random.set_state()`. The resume
path never calls `np.random.seed()`.

The state is tagged with the chain length at which it was recorded. If the
repair of B3 discarded the row that state belongs to, the tags disagree, the
state is refused, and the run falls back to **option 2, a derived reseed** of
`seed + start_iter`, with a `RuntimeWarning`. That fallback is reproducible but
no longer matches an uninterrupted chain, so the warning is worth reading.

Cost: one ~2.5 kB dataset write per iteration, against the ~76 kB of
`signal_amps` written per iteration at 80 × 60.

### B5 — the `.npy` files are now a derived export of the full HDF5

`np.save` has fixed filenames and no append path, so dumping the in-memory
arrays of a resumed run would leave `dps-eor.npy` holding the latest segment
in place of the whole chain. That would be silent: **all post-processing reads
the `.npy` files, not the HDF5** (`post_processing_funcs.py:612-616`,
`plotting_functions.py:503-504`), and `post_processing_funcs.py` only prints a
WARNING when the sample count disagrees with `args.Niter`.

`utils.export_npy_from_h5(fp)` writes the six files from the HDF5 instead,
streaming rows through `np.lib.format.open_memmap` so the export never holds a
full array in memory. `gibbs_sample()` calls it in place of both former
`write_numpy_files` calls — the periodic one every `write_Niter` iterations
and the final one — so the exported files always cover every segment.

`write_numpy_files()` itself is unchanged and still available; it is simply no
longer what the sampler uses.

**Post-processing needs no change.** The `.npy` files remain a full-chain
export with the same names, shapes and dtypes as before.

### B6 — a resume is validated against the chain it is joining

Nothing previously prevented resuming a Case III chain against Case I
`sys_modes`. `gibbs_sample()` now records, as HDF5 attributes at the start of a
run: `seed`, `Ntimes`, `Nfreqs`, `Nmodes`, `Nsys_modes`, the three sampling
toggles, `map_estimate`, `solver`, and SHA-256 hashes of `vis`, `flags`,
`Ninv`, `fg_modes`, `sys_modes`, `sys_prior`, `signal_ps_prior` and
`sky_model_initial`. A resume compares all of them and raises `ValueError`
listing every mismatch. Pass `check_metadata=False` to override.

The hashes matter because `sys_sampler_wrapper.py` rebuilt the data `d` from
scratch on every run. That is deterministic under `np.random.seed(11)`, so it
happened to work — but changing `Ntimes`, `Nfreqs` or `dummy_flag` would have
had a resumed chain silently sampling *different data*. The wrapper now loads
`data_true.npy`, `eor_true.npy`, `fg_true.npy` and `gain_true.npy` from the
output directory under `--resume`, and skips the `np.save` calls that would
overwrite them, so the dependency is gone rather than merely detected.

---

## Memory

`gibbs_sample()` preallocates every sample array to the full `Niter`. At
80 × 60, a 250k-iteration run needs:

| array | size |
|---|---|
| `signal_amps` | 19.2 GB |
| `chisq` | 9.6 GB |
| `fg_amps` | 3.2 GB |
| **total resident** | **~32 GB** |

With warm start, `Niter` is the *segment* length, so the same chain run in
10k-iteration segments needs ~1.3 GB resident. This is a reason to use the
feature even on a machine that never crashes.

The same 19.2 GB array is what exhausted the disk on the 250k run. Passing
`export_npy=False` (`--no-export-npy` in the wrapper) skips the `.npy` export
entirely; every sample is still in `gibbs_samples.h5`, which is gzip
compressed, and the export can be run later against a disk with room for it:

```python
from hydra_pspec import utils
utils.export_npy_from_h5('run_dir')
```

---

## API reference

### `gibbs_sample()` — three new keyword arguments

| Argument | Default | Meaning |
|---|---|---|
| `resume` | `False` | Continue the chain in `out_dir`. `Niter` becomes the number of additional iterations. `signal_ps_initial`, `sys_initial` and (given a stored RNG state) `seed` are ignored. Requires `out_dir`. |
| `export_npy` | `True` | Write the `.npy` files as an export of the full HDF5 chain. `False` skips them. |
| `check_metadata` | `True` | Validate a resume against the metadata of the original run; raise on a mismatch. |

The returned arrays cover the iterations run by *that call*. On a resumed run
that is the new segment only — the complete chain is in the HDF5 file and in
the `.npy` export.

### `utils.py` — new functions

| Function | Description |
|---|---|
| `gibbs_sample_h5_path(fp)` | Resolve an output directory or file path to the `.h5` file |
| `h5_chain_length(fp)` | Number of complete samples (minimum over datasets) |
| `truncate_h5_to(fp, n)` | Truncate every sample dataset to `n` rows |
| `repair_h5_chain(fp)` | Truncate all datasets to their common length; returns it |
| `read_gibbs_sample_h5(fp, index=-1)` | One sample as a dict, plus `chain_length` |
| `save_rng_state(f, chain_length)` | Persist the NumPy global RNG state |
| `load_rng_state(fp, chain_length)` | Restore it, or return None if unusable |
| `array_hash(arr)` | Stable SHA-256 of dtype, shape and bytes |
| `write_chain_metadata(f, **kw)` | Record run provenance as `meta_*` attributes |
| `read_chain_metadata(fp)` | Read it back |
| `check_chain_metadata(fp, **kw)` | Compare and raise (or warn) on mismatch |
| `export_npy_from_h5(fp)` | Write the six `.npy` files from the HDF5 chain |

Every one of these accepts an output directory, a path to the `.h5` file, or
an already-open `h5py.File`. The last form exists because HDF5's file lock
forbids a second handle to a file a running chain is holding open.

### `GibbsSampleH5Writer` — new members

`save_rng=True` constructor flag (persist the RNG state with every append),
and the `file`, `chain_length`, `truncate_to()`, `repair()`,
`write_metadata()` and `export_npy()` members.

### `sys_sampler_wrapper.py` — new flags

| Flag | Meaning |
|---|---|
| `--resume` | Continue the chain in the output directory |
| `--Niter N` | Iterations to run (additional iterations under `--resume`) |
| `--out-dir DIR` | Output directory, overriding the value in the Configuration section |
| `--no-export-npy` | Skip the `.npy` export (`export_npy=False`) |

---

## Acceptance test

From the plan:

> Run 10 iterations straight. Separately run 5, resume for 5 more. Assert all
> six outputs are bit-identical.

`tests/test_warm_start.py` implements exactly that, plus the supporting
behaviour for B1, B3, B4, B5 and B6:

```bash
pytest tests/test_warm_start.py -v
```

It runs in a few seconds on a 4 × 8 problem. The test is only meaningful
because `gibbs_sample()` seeds once per chain and draws from that single
stream in a fixed serial order — before that fix the two runs would have
differed regardless, and a resume bug would have been indistinguishable from
RNG noise.

A three-segment split (2 + 3 + 4) and a full-shape run of the wrapper
(80 × 60, 2 + 2 against 4) were also verified bit-identical during
implementation.

---

## No mathematical logic changed

The sampler maths is untouched. Every edit is I/O, state plumbing or
validation. Two changes are worth naming because they are visible but not
mathematical:

- The two assertions on the initial state (`sys_initial` shape, and
  `signal_ps_initial` lying within `signal_ps_prior`) now run against the
  arrays actually used. For a fresh run those are the caller's arguments,
  exactly as before; for a resume they are the values read back from the
  chain, which is the only version of the check that means anything there.
- The verbose iteration counter prints the global iteration number
  (`start_iter + i + 1`) rather than the segment index, so a resumed segment
  continues the numbering. Identical for a fresh chain.

---

## Limitations and follow-ups

- **`out_dir=None` does not work**, despite the docstring saying samples are
  simply not written to disk. `GibbsSampleH5Writer` is constructed
  unconditionally and `Path(None)` raises `TypeError`. This predates warm
  start and is unchanged; every caller in the repository passes `out_dir`.
- **The periodic export re-writes the whole chain.** Exporting every
  `write_Niter` iterations costs O(chain length) each time, so a small
  `write_Niter` on a long chain is expensive. This is the same cost profile as
  the `write_numpy_files(signal_amps[:i+1], ...)` call it replaced. All
  in-repository callers set `write_Niter = Niter`, i.e. export once at the end.
- **`predict_sampler_runtime.py:404`** sets `write_Niter = niter + 1` with the
  comment "suppress the bulk .npy dump", but the end-of-run write fires anyway
  (`Niter % write_Niter > 0` is true). It could now pass `export_npy=False` to
  achieve what the comment intends. Left unchanged so as not to perturb the
  existing runtime calibration data.
- **Resuming a `map_estimate=True` run** is possible but meaningless:
  `map_estimate` forces `Niter = 1`.
