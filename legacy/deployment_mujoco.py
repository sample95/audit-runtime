"""
deployment_mujoco.py — Persistent deployment audit for MuJoCo policies
Methodology: v12.2
Environments: Hopper-v4, Walker2d-v4

ARCHIVED: superseded by core/audit_deploy.py (v12.3+). Kept as a frozen
historical reference — internals are NOT updated to current terminology.
Current vocabulary (see core/drift_wrappers.py): "perturbation" = the
mechanism (e.g. GravityPerturbation), "transition" = the schedule
("baseline"/"abrupt"/"gradual"). This file's "drift"-prefixed names
(DRIFT_SCHEDULES, GravityDriftWrapper, "no_drift") reflect the pre-rename
vocabulary in use when v12.2 was current and are left as-is.

Failure criteria (rely on env defaults — do NOT override):
  Hopper-v4   : terminated=True when z < 0.7 OR |angle| > 0.2
  Walker2d-v4 : terminated=True when z ∉ [0.8, 2.0] OR |angle| > 1.0
  (Walker2d range distinct from Hopper; must not substitute 0.7 threshold)

Drift perturbation: gravity modification
  g_nominal = -9.81  m/s²
  g_drifted = -14.715 m/s² (1.5× nominal; documented perturbation choice)
  t0 = 0.5 × T_eval (abrupt switch point)
  Gradual: linear ramp from g_nominal to g_drifted over steps [0, t0]

Outputs (per condition × seed × drift_schedule):
  {cond}_seed{seed}_{drift}_metrics_summary.json   — curves, thresholds, event times
  {cond}_seed{seed}_{drift}_rollouts_summary.json  — per-run KM input row
  all_rollouts_summary.json                        — aggregated across all runs
"""

import os
import json
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
import math

def first_eligible_warning_start(K0: int, W: int, S: int) -> int:
    pilot_end = (K0 - 1) * S + W
    return math.ceil(pilot_end / S) * S

torch.set_num_threads(1)

# ── Locked instrument parameters (v12.2 §2) ────────────────────────────────────
T_EVAL    = 1_000_000   # total deployment steps
B         = 21          # histogram bins per feature
W         = 50_000      # window size (steps)
S         = 25_000      # stride (steps)
K0        = 4           # pilot windows
ALPHA     = 0.5         # pseudocount
CLIP_LO_P = 0.5         # lower clip percentile
CLIP_HI_P = 99.5        # upper clip percentile
C_FAIL    = 2.0         # soft-failure σ multiplier
M_FAIL    = 2           # soft-failure persistence (consecutive windows)
Q_WARN    = 0.95        # drift warning quantile
Q_WATCH   = 0.90        # drift watch quantile (descriptive only)
EPS_STD   = 1e-8        # standardization epsilon

G_NOMINAL  = -9.81
G_DRIFTED  = -14.715
T0         = T_EVAL // 2   # 500,000

ENVS = {
    "hopper": {
        "env_id":      "Hopper-v5",
        "results_dir": "./results/training/hopper",
        "conditions":  ["M1", "M2", "M3"],
    },
    "walker2d": {
        "env_id":      "Walker2d-v4",
        "results_dir": "./results/training/walker2d",
        "conditions":  ["M1", "M2", "M3"],
    },
}

SEEDS           = [0, 1, 2, 3, 4]
DRIFT_SCHEDULES = ["no_drift", "abrupt", "gradual"]


class GravityDriftWrapper(gym.Wrapper):
    """
    Injects gravity perturbation into MuJoCo environment.
    Modifies model.opt.gravity[2] in-place at each step.
    Global step counter is NOT reset on episode resets (persistent deployment).
    """
    def __init__(self, env, drift_type,
                 t0=T0, g_nominal=G_NOMINAL, g_drifted=G_DRIFTED):
        super().__init__(env)
        self.drift_type = drift_type
        self.t0         = t0
        self.g_nominal  = g_nominal
        self.g_drifted  = g_drifted
        self._global_t  = 0   # persists across episode resets

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_gravity()  # re-apply after MuJoCo model reset restores defaults
        return obs, info

    def step(self, action):
        self._global_t += 1
        self._apply_gravity()
        return self.env.step(action)

    def _apply_gravity(self):
        t, t0 = self._global_t, self.t0
        if self.drift_type == "no_drift":
            g = self.g_nominal
        elif self.drift_type == "abrupt":
            g = self.g_drifted if t >= t0 else self.g_nominal
        else:  # gradual
            pilot_end = (K0 - 1) * S + W
            if t < pilot_end:
                frac = 0.0
            else:  # gradual
                pilot_end = (K0 - 1) * S + W
                if t < pilot_end:
                    frac = 0.0
                else:
                    frac = min((t - pilot_end) / (t0 - pilot_end), 1.0) if t0 > pilot_end else 1.0
                g = self.g_nominal + frac * (self.g_drifted - self.g_nominal)


def get_window_starts(T_eval, w, s):
    """Return list of 0-indexed window start steps."""
    starts, k = [], 0
    while True:
        t_start = k * s
        if t_start + w > T_eval:
            break
        starts.append(t_start)
        k += 1
    return starts


# ── PMF and JSD ───────────────────────────────────────────────────────────────
def compute_pmf(std_obs, bin_edges_per_feat, alpha, n_bins):
    """
    Compute PMF [d, B] with pseudocount alpha.
    std_obs       : [T, d] standardized + clipped observations
    bin_edges_per_feat : [d, B+1] per-feature bin edges
    """
    d = std_obs.shape[1]
    pmf = np.zeros((d, n_bins), dtype=np.float64)
    for j in range(d):
        counts, _ = np.histogram(std_obs[:, j], bins=bin_edges_per_feat[j])
        pmf[j] = (counts + alpha) / (counts.sum() + alpha * n_bins)
    return pmf


def jsd_single(p, q):
    """Jensen-Shannon divergence between two 1-D PMFs (nats)."""
    eps = 1e-12
    m   = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def jsd_aggregate(pmf_k, ref_pmf):
    """D_k = (1/d) * sum_j JSD(ref_j, pmf_k_j)."""
    d = pmf_k.shape[0]
    return float(np.mean([jsd_single(ref_pmf[j], pmf_k[j]) for j in range(d)]))


def calibrate_pilot(obs_arr, K0, w, s, B, alpha, clip_lo_p, clip_hi_p):
    """
    Compute pilot statistics from first K0 windows.
    All calibration artifacts are frozen after this call.

    Returns dict with:
      mu, sigma         : standardization statistics [d]
      clip_lo, clip_hi  : per-feature clip bounds [d] (on standardized space)
      bin_edges         : per-feature bin edges [d, B+1]
      ref_pmf           : reference PMF [d, B] (mean over K0 pilot window PMFs)
      pilot_D           : list of K0 JSD values (used for tau_warn/tau_watch)
      pilot_y           : list of K0 windowed mean rewards (for mu_ref/sigma_ref)
    """
    T, d = obs_arr.shape
    starts = get_window_starts(T, w, s)
    assert len(starts) >= K0, f"Insufficient windows for K0={K0}: got {len(starts)}"

    pilot_starts = starts[:K0]

    # Collect all pilot-window observations (steps covered by first K0 windows)
    pilot_mask = np.zeros(T, dtype=bool)
    for ps in pilot_starts:
        pilot_mask[ps:ps + w] = True
    pilot_obs = obs_arr[pilot_mask]  # [N_pilot, d]

    # Standardization stats
    mu    = pilot_obs.mean(axis=0)
    sigma = pilot_obs.std(axis=0) + EPS_STD

    # Standardize pilot observations
    std_pilot = (pilot_obs - mu) / sigma

    # Clip bounds (percentiles of standardized pilot obs, per feature)
    clip_lo = np.percentile(std_pilot, clip_lo_p, axis=0)
    clip_hi = np.percentile(std_pilot, clip_hi_p, axis=0)

    # Bin edges: uniform between clip bounds per feature
    bin_edges = np.array([
        np.linspace(clip_lo[j], clip_hi[j], B + 1)
        for j in range(d)
    ])  # [d, B+1]

    # Compute PMF for each pilot window
    pilot_pmfs = []
    for ps in pilot_starts:
        win_std = np.clip((obs_arr[ps:ps + w] - mu) / sigma, clip_lo, clip_hi)
        pilot_pmfs.append(compute_pmf(win_std, bin_edges, alpha, B))
    pilot_pmfs = np.array(pilot_pmfs)  # [K0, d, B]

    # Reference PMF for deployment
    ref_pmf = pilot_pmfs.mean(axis=0)

    # Leave-one-out pilot variability
    pilot_D = []
    for k in range(K0):
        loo_pmfs = np.delete(pilot_pmfs, k, axis=0)
        loo_ref  = loo_pmfs.mean(axis=0)
        pilot_D.append(jsd_aggregate(pilot_pmfs[k], loo_ref))

    return {
        "mu": mu, "sigma": sigma,
        "clip_lo": clip_lo, "clip_hi": clip_hi,
        "bin_edges": bin_edges, "ref_pmf": ref_pmf,
        "pilot_D": pilot_D,
    }


def compute_windowed_metrics(obs_arr, reward_arr, calib, T_eval, w, s, B, alpha):
    """
    Compute D_k (drift) and y_k (windowed mean reward) for all windows.
    Returns D_k [n_windows], y_k [n_windows], starts [n_windows].
    """
    d       = obs_arr.shape[1]
    starts  = get_window_starts(T_eval, w, s)
    mu      = calib["mu"]
    sigma   = calib["sigma"]
    clip_lo = calib["clip_lo"]
    clip_hi = calib["clip_hi"]
    be      = calib["bin_edges"]
    ref_pmf = calib["ref_pmf"]

    D_k, y_k = [], []
    for ps in starts:
        win_obs = obs_arr[ps:ps + w]
        win_rew = reward_arr[ps:ps + w]
        std_obs = np.clip((win_obs - mu) / sigma, clip_lo, clip_hi)
        pmf     = compute_pmf(std_obs, be, alpha, B)
        D_k.append(jsd_aggregate(pmf, ref_pmf))
        y_k.append(float(win_rew.mean()))

    return np.array(D_k), np.array(y_k), starts


def theil_sen_slope(y_k, starts):
    """β_TS = median_{i<j} (y_j - y_i) / (t_start_j - t_start_i)."""
    slopes = []
    n = len(y_k)
    for i in range(n):
        for j in range(i + 1, n):
            dx = starts[j] - starts[i]
            if dx > 0:
                slopes.append((y_k[j] - y_k[i]) / dx)
    return float(np.median(slopes)) if slopes else 0.0


def compute_early_warning(D_k, y_k, calib, starts, K0, q_warn, q_watch, c_fail):
    pilot_D   = calib["pilot_D"]
    tau_warn  = float(np.quantile(pilot_D, q_warn))
    tau_watch = float(np.quantile(pilot_D, q_watch))

    pilot_y   = y_k[:K0]
    mu_ref    = float(np.mean(pilot_y))
    sigma_ref = float(np.std(pilot_y, ddof=1)) if K0 > 1 else 1.0
    sigma_ref = max(sigma_ref, 0.01 * abs(mu_ref))

    first_warn_start = first_eligible_warning_start(K0, W, S)
    eligible = np.array([ps >= first_warn_start for ps in starts], dtype=bool)

    A_deg   = ((y_k < mu_ref - c_fail * sigma_ref) & eligible).astype(int)
    A_drift = ((D_k > tau_warn) & eligible).astype(int)
    A_watch = ((D_k > tau_watch) & eligible).astype(int)

    T_deg   = next((int(starts[k]) for k in range(len(A_deg)) if A_deg[k]), None)
    T_drift = next((int(starts[k]) for k in range(len(A_drift)) if A_drift[k]), None)

    candidates = [t for t in [T_deg, T_drift] if t is not None]
    T_warn = int(min(candidates)) if candidates else None

    return {
        "tau_warn":  tau_warn,
        "tau_watch": tau_watch,
        "mu_ref":    mu_ref,
        "sigma_ref": sigma_ref,
        "A_deg":     A_deg.tolist(),
        "A_drift":   A_drift.tolist(),
        "A_watch":   A_watch.tolist(),
        "T_deg":     T_deg,
        "T_drift":   T_drift,
        "T_warn":    T_warn,
    }

def compute_soft_failure(y_k, starts, mu_ref, sigma_ref, c_fail, m_fail):
    """
    T_soft = t_start(k*) where k* = min{k : F_k AND F_{k+1} ... AND F_{k+m-1}}.
    F_k = I[y_k < μ_ref − c·σ_ref].
    Returns step index or None.
    """
    threshold = mu_ref - c_fail * sigma_ref
    n = len(y_k)
    for k in range(n - m_fail + 1):
        if all(y_k[k + i] < threshold for i in range(m_fail)):
            return int(starts[k])
    return None


def run_deployment(model, env_id, drift_type, seed):
    """
    Run T_EVAL steps of persistent deployment under specified drift schedule.

    Policy resets after hard failure (terminated=True) but the deployment
    continues; T_hard records the FIRST hard failure step.
    Truncation (truncated=True from TimeLimit) does NOT count as failure —
    consistent with persistent deployment semantics (Pardo et al., 2018).

    Returns:
      obs_arr        : [T_EVAL, d] float32
      reward_arr     : [T_EVAL] float32
      T_hard         : int or None (first hard failure global step)
      episode_starts : list of global steps where episodes started
    """
    env = GravityDriftWrapper(gym.make(env_id), drift_type=drift_type)

    obs, _    = env.reset(seed=seed)
    obs_list  = []
    rew_list  = []
    T_hard    = None
    ep_starts = [0]
    ep_seed   = seed

    for t in range(T_EVAL):
        obs_list.append(obs.copy())
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rew_list.append(float(reward))

        if terminated:
            if T_hard is None:
                T_hard = t + 1  # 1-indexed global failure step
            ep_seed += 1
            obs, _ = env.reset(seed=ep_seed)
            ep_starts.append(t + 1)
        elif truncated:
            # Truncation: reset without recording as failure
            ep_seed += 1
            obs, _ = env.reset(seed=ep_seed)
            ep_starts.append(t + 1)
        else:
            obs = next_obs

    env.close()
    return (
        np.array(obs_list, dtype=np.float32),
        np.array(rew_list,  dtype=np.float32),
        T_hard,
        ep_starts,
    )

_parser = argparse.ArgumentParser(description="MuJoCo deployment audit (v12.2)")
_parser.add_argument("--condition", type=str, default=None,
                     help="Run only this condition (e.g. M1, M2, M3). "
                          "Omit to run all conditions sequentially.")
_parser.add_argument("--env", type=str, default=None,
                     help="Run only this env key (e.g. hopper, walker2d). "
                          "Omit to run all envs.")
_args = _parser.parse_args()
CONDITION_FILTER = _args.condition   # None → no filter
ENV_FILTER       = _args.env         # None → no filter

print("=" * 60)
print("MUJOCO DEPLOYMENT AUDIT (v12.2)")
print(f"T_eval={T_EVAL:,} | w={W:,} | s={S:,} | K0={K0} | B={B}")
print(f"Drift: gravity {G_NOMINAL} → {G_DRIFTED} m/s² at t0={T0:,}")
print(f"Drift schedules: {DRIFT_SCHEDULES}")
if CONDITION_FILTER:
    print(f"[worker mode] condition={CONDITION_FILTER}")
print("=" * 60)

all_rollouts = []

for env_key, env_cfg in ENVS.items():
    if ENV_FILTER and env_key != ENV_FILTER:
        continue
    env_id      = env_cfg["env_id"]
    results_dir = env_cfg["results_dir"]
    conditions  = env_cfg["conditions"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n── {env_id} ──")

    for cond in conditions:
        if CONDITION_FILTER and cond != CONDITION_FILTER:
            continue
        for seed in SEEDS:
            ckpt_path = os.path.join(results_dir, f"{cond}_seed{seed}.zip")
            if not os.path.exists(ckpt_path):
                print(f"  [skip] {cond} seed={seed} — checkpoint not found")
                continue

            model = PPO.load(ckpt_path, device="cpu")

            for drift_type in DRIFT_SCHEDULES:
                prefix        = f"{cond}_seed{seed}_{drift_type}"
                metrics_path  = os.path.join(results_dir, f"{prefix}_metrics_summary.json")
                rollouts_path = os.path.join(results_dir, f"{prefix}_rollouts_summary.json")

                if os.path.exists(metrics_path) and os.path.exists(rollouts_path):
                    print(f"  [skip] {cond} seed={seed} {drift_type}")
                    # Re-load rollout for aggregation
                    with open(rollouts_path) as f:
                        all_rollouts.append(json.load(f))
                    continue

                print(f"  [run]  {cond} seed={seed} {drift_type} ...", flush=True)

                # 1. Persistent deployment run
                obs_arr, reward_arr, T_hard, ep_starts = run_deployment(
                    model, env_id, drift_type, seed
                )

                # 2. Pilot calibration (first K0 windows)
                calib = calibrate_pilot(
                    obs_arr, K0, W, S, B, ALPHA, CLIP_LO_P, CLIP_HI_P
                )

                # 3. Windowed drift and degradation
                D_k, y_k, starts = compute_windowed_metrics(
                    obs_arr, reward_arr, calib, T_EVAL, W, S, B, ALPHA
                )

                # 4. Theil-Sen slope
                beta_TS = theil_sen_slope(y_k, starts)

                # 5. Early warning
                ew = compute_early_warning(
                    D_k, y_k, calib, starts, K0, Q_WARN, Q_WATCH, C_FAIL
                )
                mu_ref    = ew["mu_ref"]
                sigma_ref = ew["sigma_ref"]

                # 6. Soft failure (survival instrument, requires m=2 persistence)
                T_soft = compute_soft_failure(
                    y_k, starts, mu_ref, sigma_ref, C_FAIL, M_FAIL
                )

                # 7. Composite failure (hard failure takes precedence on tie)
                T_comp      = T_soft
                delta       = 1 if T_soft is not None else 0
                T_comp_obs  = T_soft if T_soft is not None else T_EVAL

                # 8. Timely warning
                T_warn = ew["T_warn"]
                timely = bool(delta == 1 and T_warn is not None and T_warn < T_comp_obs)

                # 9. Detection delay after drift onset (abrupt only)
                if drift_type == "abrupt" and T_warn is not None:
                    delta_delay = int(T_warn - T0)
                else:
                    delta_delay = None

                # ── Metrics summary ───────────────────────────────────────────
                metrics = {
                    # Identity
                    "env_id":         env_id,
                    "condition":      cond,
                    "seed":           seed,
                    "drift_type":     drift_type,
                    "T_eval":         T_EVAL,
                    # Pilot calibration artifacts (logged for reproducibility)
                    "mu":             calib["mu"].tolist(),
                    "sigma":          calib["sigma"].tolist(),
                    "clip_lo":        calib["clip_lo"].tolist(),
                    "clip_hi":        calib["clip_hi"].tolist(),
                    "pilot_D":        calib["pilot_D"],
                    "tau_warn":       ew["tau_warn"],
                    "tau_watch":      ew["tau_watch"],
                    "mu_ref":         mu_ref,
                    "sigma_ref":      sigma_ref,
                    # Curves
                    "window_starts":  [int(x) for x in starts],
                    "D_k":            D_k.tolist(),
                    "y_k":            y_k.tolist(),
                    "A_deg":          ew["A_deg"],
                    "A_drift":        ew["A_drift"],
                    "A_watch":        ew["A_watch"],
                    "beta_TS":        beta_TS,
                    "T_hard": T_hard,
                    "T_soft":         T_soft,
                    "T_comp":         T_comp,
                    "T_comp_obs":     T_comp_obs,
                    "delta":          delta,
                    "T_deg":          ew["T_deg"],
                    "T_drift":        ew["T_drift"],
                    "T_warn":         T_warn,
                    "timely_warn":    timely,
                    "delta_delay":    delta_delay,
                    "episode_starts": ep_starts[:50],
                    "n_episodes":     len(ep_starts),
                }

                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

                # ── Rollouts summary (KM input row) ───────────────────────────
                rollout = {
                    "env_id":       env_id,
                    "condition":    cond,
                    "seed":         seed,
                    "drift_type":   drift_type,
                    "T_comp":       T_comp_obs,
                    "delta":        delta,
                    "T_warn":       T_warn,
                    "timely_warn":  timely,
                    "delta_delay":  delta_delay,
                    "T_hard": T_hard,
                    "T_soft":       T_soft,
                    "beta_TS":      beta_TS,
                    "mu_ref":       mu_ref,
                    "sigma_ref":    sigma_ref,
                }

                with open(rollouts_path, "w") as f:
                    json.dump(rollout, f, indent=2)

                all_rollouts.append(rollout)

                print(f"  [done] {cond} seed={seed} {drift_type} | "
                      f"δ={delta} T_comp={T_comp} T_warn={T_warn} "
                      f"timely={timely} Δ_delay={delta_delay}")

# In worker mode (--condition X) write a per-condition slice so parallel
# workers don't clobber each other.  Run without --condition afterwards to
# merge (or just cat the slices).

os.makedirs("./results/mujoco", exist_ok=True)
if CONDITION_FILTER:
    agg_path = f"./results/mujoco/rollouts_{CONDITION_FILTER}.json"
else:
    agg_path = "./results/mujoco/all_rollouts_summary.json"

with open(agg_path, "w") as f:
    json.dump(all_rollouts, f, indent=2)

print(f"\nAggregate rollouts → {agg_path}")
print("Deployment audit complete.")
