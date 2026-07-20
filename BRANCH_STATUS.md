# Branch status: `master` vs `main` (2026-07-19, corrected after `git fetch`)

*Correction note: an earlier version of this file was written from a local clone that hadn't fetched `origin/main` in some time (last-known tip: `890d270`). A `git fetch origin` run on 2026-07-19 on the NAS showed `main` has moved to `73b7056`, 5 commits ahead of what this note originally described. Numbers below are corrected accordingly.*

## Summary

`master` and `origin/main` are **unrelated histories** at the git level — no common ancestor commit between them — but they are not independent in practice: `main` receives periodic **manual backports** of specific changes from `master` (confirmed by commit message, e.g. `f75fd36 Sync src_clean rename work ... into main`), plus its own independent maintenance commits. Reconciling them still means manual file-by-file porting, not a `git merge`, but `main` is actively tended, not simply stale/abandoned.

## What's on each

**`master`** (this working branch, active development) — the messy, authoritative research tree: loose top-level scripts from earlier iterations sitting alongside the current, canonical `src_clean/` layer that the thesis actually cites. Currently at `3d1bebb`.

**`origin/main`** — a clean, flattened, GitHub/Kaggle-facing layout (`core/`, `analysis/`, `training/`, `deployment/`, `legacy/`, `models/`, `patches/`). Currently at `73b7056` (fetched 2026-07-19). This is the branch the public Kaggle notebook and the `v1.0-thesis` release checkpoints are built from.

## Divergence

No common ancestor (`git merge-base` returns nothing) — `main` was built as an independent flatten, not branched from `master`. As of this fetch: `master` is 8 commits ahead of `main`'s last-synced point per GitHub's compare view; `main` has 5 commits (`890d270..73b7056`) that are not on `master`:

- `f75fd36` — Sync src_clean rename work (`drift_wrapper`/`drift_type`/`drift_schedules` → `perturbation`/`transition` vocabulary) into main — **manual backport of `master`'s `53f9a1d`.**
- `2196822` — Record publish commit hash in manifest (routine housekeeping)
- `0f8c07d` — Remove stray `.write_test6` sandbox artifacts
- `00e67d4` — Fix grid2op demo: warm up grid2op's dynamic Action class before checkpoint load
- `73b7056` — Archive grid2op + walker2d-zoo experiments (scope decision 2026-07-12)

**Two areas worth checking before any future sync, not yet verified:**
1. `main`'s `00e67d4` (grid2op Action-class warmup fix) may overlap with or be superseded by `master`'s later `3d1bebb` ("Grid2op retry-pass fixes: chronic RNG diversity, class-pickling bypass, w/s recalibration to grid2op timescale") — both are independent grid2op fixes; unclear if they touch the same code path or are complementary.
2. `main`'s `73b7056` ("Archive grid2op + walker2d-zoo experiments, scope decision 2026-07-12") and `master`'s `731442c` ("Prognostics demo + walker2d deployment archive") both archive walker2d around the same date — worth confirming they're not contradicting each other (main archiving something master still actively used).

Neither of these was diffed against the other branch — flagged for a closer look at sync time, not resolved here.

## What's new on `master` since `main`'s last synced point (`890d270`)

- Grid2op prognostics demo (240 lifetimes, 3-way replication) and the retry-pass fixes (chronic RNG diversity, class-pickling bypass, w/s recalibration) — `731442c`, `3d1bebb`.
- The `perturbation`/`transition` vocabulary rename (`53f9a1d`) — **already backported to main** via `f75fd36`, so this one is not a gap.
- **This session (2026-07-19)**: RUL-coverage LOO pool extended to 4 subjects (Hopper raw/VecNorm, Walker2d, QuadX-Hover), `viz6_dk_dims_heatmap.png` reconstructed from raw data (original generator unrecoverable — see script docstring), `environment/requirements.txt` synced to the real NAS `audit_env` freeze, and the SAC 5M FAR figure/numbers updated from the completed 5M rerun (91.9% → 80.1%, run-level FA 8/9). Not yet backported to `main`.

## Recommendation

Not urgent for thesis submission — the thesis draws on `master`/`src_clean` exclusively, and `main` is the public-release/Kaggle-consumption artifact, not upstream of the thesis. But since `main` is being actively (if irregularly) maintained rather than frozen, a full "re-flatten `src_clean/` onto `main`" approach risks clobbering `main`-only fixes like `00e67d4` and `73b7056` that don't exist on `master`. Any future sync should diff those two flagged commits first, then selectively backport `master`'s new work (same pattern as `f75fd36`) rather than doing a blind overwrite. Best done once the thesis numbers are final and there's no more risk of further correction.
