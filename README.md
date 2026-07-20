# RL Policy Audit Framework: Runtime Monitoring with RUL Prediction

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)

Environment-agnostic runtime audit for frozen reinforcement-learning policies: detects distributional drift from the observation–reward–termination stream alone, raises a calibrated alarm, and accompanies every alarm with a survival-model prediction range for the remaining useful life (RUL) of the deployment.

## Abstract

A reinforcement-learning policy is deployed frozen, but its environment is not: distributional drift eventually invalidates the conditions under which the policy was validated. Standard evaluation reports scalar returns, and existing monitors detect drift without stating how long the operator has before failure. This work develops and evaluates an environment-agnostic audit instrument that consumes only the observation–reward–termination stream of a deployed policy, detects drift as a windowed divergence from a frozen pilot baseline with a per-run calibrated threshold, and accompanies every alarm with a survival-model prediction range for the remaining useful life of the deployment. The instrument was evaluated over 150 controlled deployments of frozen PPO and SAC policies on two continuous-control benchmarks (MuJoCo Hopper-v5 and PyFlyt QuadX-Hover) under baseline, gradual, and abrupt transition schedules. Under gradual drift the instrument warned before hard failure in 13/15 (raw PPO) and 15/15 (observation-normalised) runs, with median lead times of 29,410 and 160,561 steps against a window-level false-alarm rate of 1.14% and 10.86% respectively; the leave-one-out 90th-percentile prediction range covered 82.1% of observed failures (n=28), upgrading the alarm from a reference indicator to a calibrated prediction. Abrupt onsets expose a geometric response-time bound of one analysis window, and multi-million-step null deployments show that any false-alarm rate is meaningful only at a stated horizon. A cross-algorithm replication (SAC) confirms detection transfers while the policy itself survives the same perturbation. The framework demonstrates that accountability for deployed RL is achievable as a measurable compound of detection, false-alarm burden, traceable provenance, and calibrated lead time.

## Layout

| Folder | Contents |
|---|---|
| `core/` | The audit framework itself: `audit_deploy.py`, `drift_wrappers.py`, `launch_audit.py` (3-worker launcher), `pre_deployment.py` |
| `configs/` | All experiment configs (Hopper, Walker2d, Hover, Grid2op, T-series, locked instrument params) |
| `training/` | Policy training scripts (PPO/SAC/NAS variants per environment) + `notebooks/` (Kaggle + validation) |
| `deployment/` | Pilot calibration + rendering: `action_pilot.py`, `render_deployment.py` |
| `test_sequence/` | T-series scripts: T-01 report, T-08 R_k, T-09 E_k, T-10 survival deploy, T-11 calibrate, T-12 rerun, T-13 analysis |
| `analysis/` | Post-hoc analysis: `analysis.py`, `eval_*`, `deploy_results.py`, `exploratory_analysis.py`, `k0_sensitivity.py`, `diag_alive_bonus.py`, `plot_framework_monitor.py`, `rul_coverage_validation.py` |
| `figures/` | Figure generation + docx insertion |
| `patches/` | One-shot data patches: `patch_beta_ts.py` (β_TS), `patch_hopper_channels.py` (per-dimension D_k) |
| `legacy/` | Superseded by `core/audit_deploy.py` — kept for provenance: `deployment.py`, `deployment_mujoco.py` |
| `environment/` | Dockerfile + pinned `requirements.txt` |
| `experiment_artifacts/` | Locked instrument config + commit-hash manifest proving the results were not tuned after the fact — see [Traceability](#traceability) |

Pre-trained checkpoints are **not** included in this repository (see [Pre-trained models](#pre-trained-models) below).

## Installation

```bash
git clone https://github.com/sample95/audit-runtime.git
cd audit-runtime
pip install -r environment/requirements.txt
```

Or via Docker:

```bash
docker build -t audit -f environment/Dockerfile .
docker run --rm -it audit
```

## Quickstart

```python
import sys, json
sys.path.insert(0, "core")
from stable_baselines3 import PPO
from audit_deploy import run_deployment, calibrate_pilot, compute_windowed_metrics

model = PPO.load("path/to/your_policy.zip", device="cpu")

obs_arr, act_arr, reward_arr, reward_forward_arr, T_hard = run_deployment(
    model, env_id="Hopper-v5", transition="gradual", seed=0,
)

# Calibrate the pilot baseline from the first K0 windows, then score the
# whole run against it (see instrument.json for K0/w/s/alpha/etc.).
calib = calibrate_pilot(obs_arr, K0=4, w=50_000, s=25_000, B=21, alpha=0.5,
                        clip_lo_p=0.5, clip_hi_p=99.5)
D_k, y_k, starts = compute_windowed_metrics(obs_arr, reward_arr, calib,
                                            T_eval=len(obs_arr), w=50_000, s=25_000,
                                            B=21, alpha=0.5)
print(f"T_hard={T_hard}  windows scored={len(D_k)}")
```

For a full run with alarm thresholds, RUL prediction, and JSON output, use the CLI entry point instead of calling the internals directly (see below).

## Reproduce results

```bash
python core/launch_audit.py configs/hopper_vc.json
python analysis/deploy_results.py --results-dir results/mujoco/hopper_vc
```

Each headline thesis result maps to one config:

| Result | Config | Expected |
|---|---|---|
| Hopper PPO, gradual: 13/15 timely, mean lead 34,751 (median 29,410) | `configs/hopper_t10_v2.json` | FAR 6/525 = 1.14%, AUC 0.9986 |
| Hopper PPO, abrupt (t₀=500k): 0/15 (geometric bound) | `configs/hopper_t10_v2.json` | 3 pre-onset alarms |
| Hopper VecNorm, gradual: 15/15, mean lead 186,850 | `configs/hopper_vc.json` | FAR 57/525 = 10.86%, AUC 0.9828 |
| Hopper VecNorm, abrupt: 6/15 straddling-window, mean lead 13,623 | `configs/hopper_vc.json` | 6 pre-onset, 3 structural misses |
| Hopper SAC (M1+M3): TPR 10/10 gradual, 8/10 abrupt, FAR 0/280 | `configs/hopper_sac.json` | M2 gated out (pre-deployment) |
| QuadX Hover Q2: 5/5 both schedules | `configs/hover.json` | window FAR 0% at 600k |
| Weibull fits: κ=7.91/λ=214,830 (PPO), κ=5.19/λ=393,692 (VC) | via `analysis/deploy_results.py` | — |
| RUL p90 coverage 23/28 = 82.1% (LOO) | T-40 protocol on gradual runs | CI [0.631, 0.939] |
| 5M-horizon FAR: 93.7% / 75.5% / 91.9% | `configs/*_none_5M.json` | T-41/T-43 |

Seeds {0–4} are fixed in every config; per-run thresholds calibrate from that run's own pilot window only (no cross-seed information leaks into the threshold).

## Output interpretation

Each deployment run writes one `*_metrics_summary.json` (or `*_t10_metrics.json` for reset-free runs). Key fields:

| Field | Meaning |
|---|---|
| `D_k` | Per-window Jensen–Shannon divergence of standardized observations vs. the frozen pilot reference PMF — the primary drift signal. |
| `D_k_dims` | Same JSD, broken out per observation dimension (`D_k_dims[k].mean() == D_k[k]`) — identifies *which* state variables are drifting. |
| `y_k` | Per-window mean reward. |
| `tau_warn` / `tau_watch` | Per-run calibrated alarm / watch thresholds, set from the leave-one-out quantile of the pilot window's own D_k values. |
| `T_warn` | Step at which the earliest alarm channel fired (window end — the actual observable detection time in a live deployment). |
| `T_drift`, `T_deg`, `T_cusum`, `T_dyn`, `T_surplus`, `T_mk`, `T_bts` | First-fire step for each individual detection channel (JSD hard threshold, reward degradation, CUSUM, step-size norm, OLS/RF reward-residual surplus, Mahalanobis composite, rolling trend slope). |
| `T_hard` | Step of actual policy failure (`terminated=True`), or `null` if the deployment completed without failing. |
| `RUL_p50` / `RUL_p90` | Conditional-Weibull remaining-useful-life estimate (steps) at the 50th/90th percentile, computed at `T_warn`. |
| `T_fail_p50` / `T_fail_p90` | `T_warn + RUL_p50` / `T_warn + RUL_p90` — the predicted absolute failure step. |
| `A_drift`, `A_watch`, `A_deg` | Per-window boolean alarm arrays for each channel. |

## Pre-trained models

Checkpoints are **not** committed to this repository's git history (kept out via `.gitignore` — see the `.Trash-1000`/large-file incident this repo's history was cleaned of). The 55 frozen checkpoints (~115 MB, + 40 VecNormalize stats files) are instead attached to the [`v1.0-thesis` release](https://github.com/sample95/audit-runtime/releases/tag/v1.0-thesis) as downloadable assets:

| Asset | Subject | Contents |
|---|---|---|
| [`hopper.zip`](https://github.com/sample95/audit-runtime/releases/download/v1.0-thesis/hopper.zip) | Hopper raw PPO | M1/M2/M3 × seeds {0–4} |
| [`hopper_vc.zip`](https://github.com/sample95/audit-runtime/releases/download/v1.0-thesis/hopper_vc.zip) | Hopper PPO + VecNormalize | M1/M2/M3 × seeds {0–4} + `.pkl` stats |
| [`hopper_sac.zip`](https://github.com/sample95/audit-runtime/releases/download/v1.0-thesis/hopper_sac.zip) | Hopper SAC (cross-algorithm) | M1/M2/M3 × seeds {0–4} + `.pkl` stats |
| [`hover.zip`](https://github.com/sample95/audit-runtime/releases/download/v1.0-thesis/hover.zip) | QuadX Hover PPO | Q1/Q2 × seeds {0–4} + `.pkl` stats |
| [`SHA256SUMS`](https://github.com/sample95/audit-runtime/releases/download/v1.0-thesis/SHA256SUMS) | — | Integrity manifest for all of the above |

Download and unzip into `models/` (e.g. `unzip hopper.zip -d models/`), then verify integrity:

```bash
cd models
sha256sum -c SHA256SUMS
```

See `models/README.md` for the loading pattern (`PPO.load`/`SAC.load`, `VecNormalize.load` for the VecNorm/SAC/hover subjects) and which config maps to which checkpoint folder. Walker2d zoo checkpoints are handled separately — they're public upstream, fetched at runtime by `training/fetch_walker2d_zoo.py` rather than mirrored here.

## Compute

- **Training + 5M/extended runs:** Kaggle (4 vCPU / T4 GPU, 12h sessions, ~30h/week GPU quota) — see `training/notebooks/` and `kaggle_launch_checklist.md`.
- **Standard 1M deployments:** CPU-only, 3-worker launcher (`core/launch_audit.py`); a 15-run subject completes overnight on a single machine.
- **Environment:** `environment/Dockerfile` (base `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`) + pinned `environment/requirements.txt`.

## Traceability

Since this framework's entire premise is that alarm thresholds must be locked *before* seeing results, `experiment_artifacts/` contains:

- `instrument.lock.json` — exact copy of the locked instrument config (K0, window/stride, quantile thresholds, persistence counts) used for every result reported in the thesis.
- `manifest.txt` — the commit hash the reported experiments actually ran under.

This is the traceability proof: it lets anyone verify the thresholds were not adjusted after seeing the failure/detection outcomes.

## Citation

If you use this framework, please cite:

```bibtex
@mastersthesis{sumares2026audit,
  author = {Sumares, Duarte},
  title  = {Accountability and Traceability in Reinforcement Learning: An Audit Framework for Runtime Drift Detection with Calibrated Remaining-Useful-Life Prediction},
  school = {TODO: institution},
  year   = {2026}
}
```

## License

MIT — see `LICENSE`.

## Contact

Duarte Sumares — `djssumares@gmail.com` · ORCID: `TODO`
