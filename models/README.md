# Pre-Trained Models — Frozen Audit Subjects

The 55 policy checkpoints audited in the thesis (+ 40 VecNormalize stats files). All are Stable-Baselines3 saves, hardlinked verbatim from their canonical training outputs under `results/` — same bytes, no duplication (integrity: no seed was selected or excluded post-hoc; every trained seed is here). Generate the integrity manifest once with `bash make_sha256sums.sh`, then verify with `sha256sum -c SHA256SUMS`.

| Folder | Subject | Algo | Env | Conditions × seeds | Trained by |
|---|---|---|---|---|---|
| `hopper/` | Hopper raw PPO | PPO | Hopper-v5 | M1/M2/M3 × {0–4} | `training/train_hopper.py` |
| `hopper_vc/` | Hopper PPO+VecNormalize | PPO | Hopper-v5 | M1/M2/M3 × {0–4} (+ per-seed `*_vecnormalize.pkl` frozen stats) | `training/train_hopper_vecnorm.py` |
| `hopper_sac/` | Hopper SAC (cross-algorithm) | SAC | Hopper-v5 | M1/M2/M3 × {0–4} (+ `.pkl` stats). M2 excluded from audit by pre-deployment gate (60% none-condition failure rate); checkpoints kept per integrity rule R37 | `training/train_hopper_sac.py` |
| `hover/` | QuadX Hover | PPO | PyFlyt/QuadX-Hover-v4 | Q1/Q2 × {0–4} (+ `.pkl` stats) | `training/train_hover.py` |

Training regimes: M1 = time limit 1000 / reset noise 0.005 · M2 = time limit 250 · M3 = reset noise 0.02 · Q1 = 10.0 s episodes · Q2 = 2.5 s episodes.

Loading: `PPO.load(path)` / `SAC.load(path)`; for VecNormalize subjects, wrap the env with `VecNormalize.load(pkl, venv)` and set `training=False, norm_reward=False` (frozen statistics — the audit protocol never updates them). Exact versions: `environment/requirements.txt` / `environment/Dockerfile`.

Deployment configs under `configs/` reference these via `ckpt_dir`; the canonical copies remain in `results/` alongside their deployment outputs.

Public release: attached as assets on the [`v1.0-thesis` GitHub release](https://github.com/sample95/audit-runtime/releases/tag/v1.0-thesis) (`hopper.zip`, `hopper_vc.zip`, `hopper_sac.zip`, `hover.zip`, `SHA256SUMS`) — see the main `README.md` "Pre-trained models" section for direct links. Not committed to git history; download and unzip into this folder, then `sha256sum -c SHA256SUMS` to verify.
