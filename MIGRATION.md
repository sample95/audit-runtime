# MIGRATION.md — what was kept, dropped, and why

Source: ThesisStorage root, snapshot 2026-06-12. Rule: keep the newest canonical version of each script (all root copies, May–June 2026); omit stale duplicates and non-code artifacts. **Nothing at root was modified or deleted.**

## Omitted: stale duplicates (older versions of kept scripts)

| Omitted (root-relative) | Date | Superseded by (in src_clean) |
|---|---|---|
| `training/train_hopper.py` | 2026-04-17 | `training/train_hopper.py` (2026-05-03) |
| `training/train_hover.py` | 2026-05-17 | `training/train_hover.py` (2026-06-08) |
| `training/train_subject.py` | 2026-04-17 | early Hopper trainer → `training/train_hopper.py` |
| `deployment/deployment_mujoco.py` | 2026-04-13 | `legacy/deployment_mujoco.py` (2026-05-01), itself superseded by `core/audit_deploy.py` |
| `deployment/deployment_pyflyt.py` | 2026-04-13 | domain-agnostic `legacy/deployment.py` (2026-05-01) → `core/audit_deploy.py` |
| `analysis/evaluation_mujoco.py`, `analysis/evaluation_pyflyt.py` | 2026-04-13 | `analysis/evaluation.py` (2026-05-11) |
| `training/configs/hopper.json`, `instrument.json` | 2026-04 | `configs/` (root versions) |

## Omitted: not code

Data and outputs (stay at root, shared): `results/` (1.6G), `runs/`, `models/`, `logs/`, `figures/` (generated), `videos/`, `papers/`, `archive/`, `t13_calib/`, `training/mujoco/` (run dirs), `calibration_thresholds.json`, `t11_calibration_report.txt`, `tea_debug.log`, `t.txt`, `__pycache__/`.

Docs (excluded by chosen scope): `docs/`, `OVERNIGHT_RUN.md`, root `README.md` (whitespace-only), `.docx`/`.pdf` files.

Empty dirs: `scripts/`, `command/`, `training/hopper/`.

## Kept deliberately despite overlap

- All trainer variants (`train_hopper{,_nas,_sac,_vecnorm}.py`, `train_mujoco.py`, `train_walker2d{,_nas}.py`): they differ by algorithm/normalization/infrastructure (VecNorm is a studied condition; NAS scripts are the compute-node versions), so none is provably redundant.
- `legacy/deployment.py` + `legacy/deployment_mujoco.py`: almost certainly superseded by `audit_deploy.py` (2026-06-09) but kept until confirmed, since they may document the pre-v12.6 deployment protocol.
- Both patch scripts: β_TS (T-15) is live; per-dim D_k patch feeds the planned re-run.

## Code edits

Listed in README §"Code changes" — 2 fix types, 5 files, all tagged `[src_clean]`, no logic changes.

## Additions 2026-07-07 (NeurIPS code-completeness pass)

- `LICENSE` — MIT.
- `models/` — the 55 audited checkpoints + 40 VecNormalize stats, hardlinked from `results/` (same bytes; see `models/README.md`); `make_sha256sums.sh` for the integrity manifest.
- `analysis/rul_coverage_validation.py` — T-40 leave-one-out RUL coverage script, copied unmodified from the project folder (2026-07-04); was missing from the 06-12 snapshot.
- `README.md` — added results-reproduction table (config → thesis result), pre-trained models section, compute section, license note; corrected the data-location claim (post-June results live inside `src_clean/results/`, per R21).

## Additions 2026-07-07 (Kaggle / external-subjects pass) — DRAFT, UNTESTED where marked

- `core/grid2op_gym_env.py` — gymnasium adapter (Grid2opChronics-v0) making Grid2op runnable through the unmodified `audit_deploy.py` v12.9; blackout→terminated, end-of-chronic→truncated; chronic selection via the existing `ChronicPerturbationWrapper` (renamed from `ChronicDriftWrapper` in the perturbation/transition terminology cleanup). **DRAFT-UNTESTED**: two VERIFY points; smoke-test before any thesis run.
- `core/run_grid2op_demo.py` — driver: registers the env, swaps `PERTURBATION_REGISTRY["ChronicPerturbation"]` (formerly `WRAPPER_REGISTRY["ChronicDrift"]`) for the pass-through shim, runs `audit_deploy.py` via runpy. Zero changes to core instrument code.
- `configs/grid2op_demo.json` — n=1 live-monitor demo (G1/seed0, D1, t0=300k).
- `configs/walker2d_zoo_ppo.json` / `configs/walker2d_zoo_sac.json` — external SB3-Zoo subjects (HF `sb3/{ppo,sac}-Walker2d-v3`), n=1 each, declared v3→v5 gap, t0=510k (off the window grid).
- `training/fetch_walker2d_zoo.py` — HF download + rename to framework convention + PROVENANCE.md.
- `training/notebooks/kaggle_zoo_grid2op.ipynb` — end-to-end Kaggle session: private-repo clone via secret PAT, pinned installs, zoo fetch, Walker2d audits, Grid2op G1 training, demo smoke test (60k), 600k demo, results zip.

## Archived 2026-07-12 (post-scope-decision cleanup)

- `legacy/grid2op_live_demo/` — the entire Grid2op workstream (training, adapter, demo driver, feasibility tests, configs, Kaggle notebook, trained G1_seed0 checkpoint). Grid2op removed from the thesis by author decision; kept as Learning Agreement evidence (see grid2op_feasibility_2026-07-12.md in the thesis folder).
- `legacy/walker2d_zoo/` — the external-checkpoint experiment (zoo configs, fetch script, checkpoints, audit logs). Superseded by the author's own Walker2d campaign (in the thesis); both zoo policies failed the admission gate (v3→v4 gap).
- Active tree now contains only thesis-supporting code: Hopper (PPO/VecNorm/SAC), Walker2d (own policies), QuadX hover, and the analysis/test-sequence pipeline.
