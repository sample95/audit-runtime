"""
deployment.py — Domain-agnostic persistent deployment audit for frozen RL policies
Methodology: v12.6 | Framework: Empirical Audit Framework for RL Policy Reliability

ARCHIVED: superseded by core/audit_deploy.py + core/launch_audit.py. Kept as a
frozen historical reference — internals still use the pre-rename vocabulary
(DRIFT_SCHEDULES, DRIFT_WRAPPER, WRAPPER_REGISTRY, "GravityDrift" etc.). These
still resolve correctly since drift_wrappers.py keeps them as backward-compat
aliases for PERTURBATION_REGISTRY / "perturbation_wrapper" / "transition_schedules",
the current terminology used by the active pipeline.

Usage:
  python deployment.py --domain hopper
  python deployment.py --domain walker2d
  python deployment.py --domain hover
  python deployment.py --domain grid2op
  python deployment.py --domain hopper_sac
  python deployment.py --domain walker2d_sac

  --domain resolves to configs/{domain}.json
  Instrument locks are read from configs/instrument.json (v12.6, frozen)

To add a new domain: create configs/{newdomain}.json — no changes to this script.

Instrument fixes applied (all domains):
  Fix 1 — LOO pilot_D          : leave-one-out JSD for pilot threshold calibration
  Fix 2 — Relative sigma_ref   : max(sigma_ref, 0.01 * |mu_ref|)
  Fix 3 — Gradual ramp delay   : drift ramp starts at pilot_end = K0 * w
  Fix 4 — Post-pilot scan      : T_warn cannot fire during pilot period (k < K0)
  Fix 5 — Quantile threshold   : tau_warn = Q_{q_warn}(D^pilot), tau_watch = Q_{q_watch}(D^pilot)
                                 Empirical quantile of the K0 pilot D values.
                                 Backed by conformal prediction theory (Prashant 2025) and
                                 healthy-only calibration + freeze (Li-2026; Siffer 2017).

Domain-specific decisions (declared in domain JSON, not hardcoded here):
  hard_failure_counts      : false for MuJoCo (falls recoverable), true for PyFlyt/Grid2Op
  gradual_ramp_after_pilot : true for all current domains
  delta_delay_schedule     : schedule name where Δ_delay is meaningful (abrupt / D1)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "core"))  # [src_clean] import shim
from drift_wrappers import WRAPPER_REGISTRY

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--domain",  required=True,
                    help="Domain name — resolves to configs/{domain}.json")
parser.add_argument("--configs", default="configs",
                    help="Path to configs directory (default: configs/)")
parser.add_argument("--conditions",   type=str, nargs="+", default=None,
                    help="Override conditions from config (e.g. --conditions M1 M2)")
args = parser.parse_args()

domain_cfg_path     = os.path.join(args.configs, f"{args.domain}.json")
instrument_cfg_path = os.path.join(args.configs, "instrument.json")

for path in [domain_cfg_path, instrument_cfg_path]:
    if not os.path.exists(path):
        print(f"ERROR: config not found: {path}")
        sys.exit(1)

with open(domain_cfg_path) as f:
    dcfg = json.load(f)
with open(instrument_cfg_path) as f:
    icfg = json.load(f)

# ── Instrument parameters (v12.6 locks) ──────────────────────────────────────
K0                 = icfg["K0"]
ALPHA              = icfg["alpha"]
CLIP_LO_P          = icfg["clip_lo_p"]
CLIP_HI_P          = icfg["clip_hi_p"]
C_FAIL             = icfg["c_fail"]
M_FAIL             = icfg["m_fail"]
Q_WARN             = icfg["q_warn"]           # tau_warn quantile (default 0.95)
Q_WATCH            = icfg["q_watch"]          # tau_watch quantile (default 0.90)
EPS_STD            = icfg["eps_std"]
SIGMA_FLOOR_FACTOR = icfg["sigma_ref_relative_floor"]

ENV_ID                   = dcfg["env_id"]
RESULTS_DIR              = dcfg["results_dir"]
CONDITIONS               = args.conditions if args.conditions is not None else dcfg["conditions"]
SEEDS                    = dcfg["seeds"]
T_EVAL                   = dcfg["T_eval"]
B                        = dcfg.get("B", None)   # None → auto-determined from pilot (FD rule)
W                        = dcfg["w"]
S                        = dcfg["s"]
ALGO                     = dcfg.get("algo", "PPO")
CKPT_SUFFIX              = dcfg.get("ckpt_suffix", "")
DRIFT_SCHEDULES          = dcfg["drift_schedules"]
HARD_FAILURE_COUNTS      = dcfg["hard_failure_counts"]
GRADUAL_RAMP_AFTER_PILOT = dcfg["gradual_ramp_after_pilot"]
DELTA_DELAY_SCHEDULE     = dcfg["delta_delay_schedule"]
DRIFT_WRAPPER            = dcfg["drift_wrapper"]
DRIFT_PARAMS             = {k: v for k, v in dcfg.get("drift_params", {}).items()
                            if not k.startswith("_")}

T0        = T_EVAL // 2
PILOT_END = ((K0 - 1) * S + W) if GRADUAL_RAMP_AFTER_PILOT else 0

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(ckpt_path):
    if ALGO == "SAC":
        from stable_baselines3 import SAC
        return SAC.load(ckpt_path, device="cpu")
    from stable_baselines3 import PPO
    return PPO.load(ckpt_path, device="cpu")


def make_env(condition, drift_type, seed):
    """
    Constructs the wrapped environment for a single deployment run.
    Returns (env, chronic_mgr_or_None).
    All domain-specific construction is driven by the domain JSON.
    """
    chronic_mgr = None

    if DRIFT_WRAPPER == "GravityDrift":
        import gymnasium as gym
        env = WRAPPER_REGISTRY["GravityDrift"](
            gym.make(ENV_ID),
            drift_type=drift_type,
            t0=T0, pilot_end=PILOT_END,
            **DRIFT_PARAMS,
        )

    elif DRIFT_WRAPPER == "ObsNoiseDrift":
        import gymnasium as gym
        import PyFlyt.gym_envs  # noqa: F401
        env_kwargs     = dcfg.get("env_kwargs", {})
        cond_durations = dcfg.get("condition_durations", {})
        max_dur        = cond_durations.get(condition, 10.0)
        base_env = gym.make(ENV_ID, max_duration_seconds=max_dur, **env_kwargs)
        env = WRAPPER_REGISTRY["ObsNoiseDrift"](
            base_env,
            drift_type=drift_type,
            t0=T0, pilot_end=PILOT_END,
            rng_seed=seed + 1000,
            **DRIFT_PARAMS,
        )

    elif DRIFT_WRAPPER == "ChronicDrift":
        import grid2op
        from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace
        from lightsim2grid import LightSimBackend
        from gymnasium.wrappers import TimeLimit

        action_attrs = dcfg.get("action_attrs", ["redispatch", "curtail"])
        condition_h  = dcfg.get("condition_h", {})
        h_train      = condition_h.get(condition, 1000)

        g2op_env = grid2op.make(ENV_ID, backend=LightSimBackend())
        gym_env  = GymEnv(g2op_env)
        gym_env.observation_space = BoxGymObsSpace(gym_env.init_env.observation_space)
        gym_env.action_space      = BoxGymActSpace(
            gym_env.init_env.action_space,
            attr_to_keep=action_attrs,
        )
        env = TimeLimit(gym_env, max_episode_steps=h_train)
        chronic_mgr = WRAPPER_REGISTRY["ChronicDrift"](
            g2op_env,
            drift_type=drift_type,
            t0=T0, pilot_end=PILOT_END,
            rng_seed=seed + 500,
        )

    else:
        raise ValueError(
            f"Unknown drift_wrapper '{DRIFT_WRAPPER}'. "
            f"Add it to drift_wrappers.py and WRAPPER_REGISTRY."
        )

    return env, chronic_mgr


def get_window_starts(T_eval, w, s):
    starts, k = [], 0
    while True:
        t_start = k * s
        if t_start + w > T_eval:
            break
        starts.append(t_start)
        k += 1
    return starts


# ── Auto bin-count (Freedman–Diaconis rule on standardised pilot obs) ─────────
def _auto_bin_count(std_pilot_obs, clip_lo, clip_hi, b_min=5, b_max=100):
    """
    Freedman–Diaconis rule applied per dimension on standardised pilot
    observations, then median across dimensions.

    h_FD = 2 · IQR · n^{-1/3}   →   B_j = ceil((clip_hi_j − clip_lo_j) / h_FD_j)
    Final B = median(B_j), clipped to [b_min, b_max].

    Observations are already z-scored so all dimensions share the same scale;
    IQR variation is driven purely by the shape of each marginal.
    """
    n      = std_pilot_obs.shape[0]
    iqr    = (np.percentile(std_pilot_obs, 75, axis=0)
              - np.percentile(std_pilot_obs, 25, axis=0))
    ranges = np.maximum(clip_hi - clip_lo, 1e-8)
    h_fd   = 2.0 * np.maximum(iqr, 1e-8) * (n ** (-1.0 / 3.0))
    B_per  = np.clip(np.ceil(ranges / h_fd).astype(int), b_min, b_max)
    return int(np.clip(int(np.median(B_per)), b_min, b_max))


# ── PMF and JSD ───────────────────────────────────────────────────────────────
def compute_pmf(std_obs, bin_edges_per_feat, alpha, n_bins):
    d   = std_obs.shape[1]
    pmf = np.zeros((d, n_bins), dtype=np.float64)
    for j in range(d):
        counts, _ = np.histogram(std_obs[:, j], bins=bin_edges_per_feat[j])
        pmf[j] = (counts + alpha) / (counts.sum() + alpha * n_bins)
    return pmf


def jsd_single(p, q):
    eps = 1e-12
    m   = 0.5 * (p + q)
    return float(0.5 * (
        np.sum(p * np.log((p + eps) / (m + eps))) +
        np.sum(q * np.log((q + eps) / (m + eps)))
    ))


def jsd_aggregate(pmf_k, ref_pmf):
    return float(np.mean([
        jsd_single(ref_pmf[j], pmf_k[j]) for j in range(pmf_k.shape[0])
    ]))


# ── Pilot calibration — Fix 1: LOO pilot_D ───────────────────────────────────
def calibrate_pilot(obs_arr, K0, w, s, B_cfg, alpha, clip_lo_p, clip_hi_p):
    """
    Fix 1 — Leave-one-out pilot_D:
      Each pilot window JSD is computed against the mean of the other K0-1
      windows. Prevents self-referential near-zero tau_warn.
      Deployment ref_pmf remains mean of all K0 pilot windows.

    B_cfg : explicit bin count (int) or None to auto-determine via
            Freedman–Diaconis rule on standardised pilot observations.
    """
    T, d   = obs_arr.shape
    starts = get_window_starts(T, w, s)
    assert len(starts) >= K0, f"Need >= K0={K0} windows, got {len(starts)}"

    pilot_starts = starts[:K0]
    pilot_mask   = np.zeros(T, dtype=bool)
    for ps in pilot_starts:
        pilot_mask[ps:ps + w] = True
    pilot_obs = obs_arr[pilot_mask]

    mu    = pilot_obs.mean(axis=0)
    sigma = pilot_obs.std(axis=0) + EPS_STD
    std_p = (pilot_obs - mu) / sigma

    clip_lo = np.percentile(std_p, clip_lo_p, axis=0)
    clip_hi = np.percentile(std_p, clip_hi_p, axis=0)

    # Auto-determine B from pilot data if not supplied in config
    if B_cfg is None:
        B = _auto_bin_count(std_p, clip_lo, clip_hi)
    else:
        B = int(B_cfg)

    bin_edges = np.array([
        np.linspace(clip_lo[j], clip_hi[j], B + 1) for j in range(d)
    ])

    pilot_pmfs = []
    for ps in pilot_starts:
        ws = np.clip((obs_arr[ps:ps + w] - mu) / sigma, clip_lo, clip_hi)
        pilot_pmfs.append(compute_pmf(ws, bin_edges, alpha, B))
    pilot_pmfs = np.array(pilot_pmfs)

    ref_pmf = pilot_pmfs.mean(axis=0)

    pilot_D = []
    for k in range(K0):
        loo_ref = np.delete(pilot_pmfs, k, axis=0).mean(axis=0)
        pilot_D.append(jsd_aggregate(pilot_pmfs[k], loo_ref))

    return {
        "mu": mu, "sigma": sigma,
        "clip_lo": clip_lo, "clip_hi": clip_hi,
        "bin_edges": bin_edges, "ref_pmf": ref_pmf,
        "pilot_D": pilot_D,
        "B": B,                          # auto-determined or explicit
        "B_source": "auto" if B_cfg is None else "config",
    }


def compute_windowed_metrics(obs_arr, reward_arr, calib, T_eval, w, s, alpha):
    """B is derived from calib["B"] — set by calibrate_pilot (auto or config)."""
    starts           = get_window_starts(T_eval, w, s)
    mu, sigma        = calib["mu"], calib["sigma"]
    clip_lo, clip_hi = calib["clip_lo"], calib["clip_hi"]
    B                = calib["B"]

    D_k, y_k = [], []
    for ps in starts:
        std_obs = np.clip((obs_arr[ps:ps + w] - mu) / sigma, clip_lo, clip_hi)
        D_k.append(jsd_aggregate(
            compute_pmf(std_obs, calib["bin_edges"], alpha, B),
            calib["ref_pmf"],
        ))
        y_k.append(float(reward_arr[ps:ps + w].mean()))
    return np.array(D_k), np.array(y_k), starts


def theil_sen_slope(y_k, starts):
    slopes = [
        (y_k[j] - y_k[i]) / (starts[j] - starts[i])
        for i in range(len(y_k))
        for j in range(i + 1, len(y_k))
        if starts[j] - starts[i] > 0
    ]
    return float(np.median(slopes)) if slopes else 0.0


# ── Early warning — Fix 2: sigma floor; Fix 4: post-pilot scan; Fix 5: quantile threshold ───
def compute_early_warning(D_k, y_k, calib, starts, K0,
                          q_warn, q_watch, c_fail, sigma_floor_factor):
    # Fix 5 — empirical quantile threshold (Prashant 2025; Li-2026; Siffer 2017)
    # tau_warn = Q_{q_warn}(D^pilot),  tau_watch = Q_{q_watch}(D^pilot)
    tau_warn  = float(np.quantile(calib["pilot_D"], q_warn))
    tau_watch = float(np.quantile(calib["pilot_D"], q_watch))

    mu_ref    = float(np.mean(y_k[:K0]))
    sigma_ref = float(np.std(y_k[:K0], ddof=1)) if K0 > 1 else 1.0
    sigma_ref = max(sigma_ref, sigma_floor_factor * abs(mu_ref), EPS_STD)

    first_warn_start = PILOT_END
    eligible = np.array([ps >= first_warn_start for ps in starts], dtype=bool)

    A_deg   = ((y_k < mu_ref - c_fail * sigma_ref) & eligible).astype(int)
    A_drift = ((D_k > tau_warn) & eligible).astype(int)
    A_watch = ((D_k > tau_watch) & eligible).astype(int)

    T_deg   = next((int(starts[k]) for k in range(len(A_deg))   if A_deg[k]),   None)
    T_drift = next((int(starts[k]) for k in range(len(A_drift)) if A_drift[k]), None)
    cands   = [t for t in [T_deg, T_drift] if t is not None]

    return {
        "tau_warn": tau_warn, "tau_watch": tau_watch,
        "mu_ref": mu_ref,     "sigma_ref": sigma_ref,
        "A_deg":   A_deg.tolist(),
        "A_drift": A_drift.tolist(),
        "A_watch": A_watch.tolist(),
        "T_deg": T_deg, "T_drift": T_drift,
        "T_warn": int(min(cands)) if cands else None,
    }


def compute_soft_failure(y_k, starts, mu_ref, sigma_ref, c_fail, m_fail):
    threshold = mu_ref - c_fail * sigma_ref
    for k in range(len(y_k) - m_fail + 1):
        if all(y_k[k + i] < threshold for i in range(m_fail)):
            return int(starts[k])
    return None


def run_deployment(model, condition, drift_type, seed):
    env, chronic_mgr = make_env(condition, drift_type, seed)

    if chronic_mgr is not None:
        chronic_id = chronic_mgr.select_chronic(0)
        env.unwrapped.set_id(chronic_id)

    obs, _    = env.reset(seed=seed)
    obs_list, rew_list = [], []
    T_hard, ep_starts, ep_term_types, ep_seed = None, [0], [], seed
    global_t = 0

    for t in range(T_EVAL):
        obs_list.append(obs.copy())
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rew_list.append(float(reward))
        global_t += 1

        if terminated or truncated:
            if terminated and T_hard is None:
                T_hard = t + 1
            # Record per-episode termination type for survival analysis
            ep_term_types.append("T" if terminated else "C")
            ep_seed += 1
            if chronic_mgr is not None:
                chronic_id = chronic_mgr.select_chronic(global_t)
                env.unwrapped.set_id(chronic_id)
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
        ep_term_types,
    )


print("=" * 60)
print(f"DEPLOYMENT AUDIT — {args.domain.upper()} ({ALGO})")
print(f"Config : {domain_cfg_path}")
print(f"T_eval={T_EVAL:,} | w={W:,} | s={S:,} | K0={K0} | B={B}")
print(f"t0={T0:,} | pilot_end={PILOT_END:,}")
print(f"hard_failure_counts={HARD_FAILURE_COUNTS} | "
      f"gradual_ramp_after_pilot={GRADUAL_RAMP_AFTER_PILOT}")
print(f"Drift schedules: {DRIFT_SCHEDULES}")
print("=" * 60)

for cond in CONDITIONS:
    print(f"\n── Condition {cond} ──")

    for seed in SEEDS:
        ckpt_name = f"{cond}{CKPT_SUFFIX}_seed{seed}.zip"
        ckpt_path = os.path.join(RESULTS_DIR, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"  [skip] {cond} seed={seed} — checkpoint not found ({ckpt_name})")
            continue

        model = load_model(ckpt_path)

        for drift_type in DRIFT_SCHEDULES:
            prefix        = f"{cond}{CKPT_SUFFIX}_seed{seed}_{drift_type}"
            metrics_path  = os.path.join(RESULTS_DIR, f"{prefix}_metrics_summary.json")
            rollouts_path = os.path.join(RESULTS_DIR, f"{prefix}_rollouts_summary.json")

            if os.path.exists(metrics_path) and os.path.exists(rollouts_path):
                print(f"  [skip] {cond} seed={seed} {drift_type}")
                continue

            print(f"  [run]  {cond} seed={seed} {drift_type} ...", flush=True)

            obs_arr, reward_arr, T_hard, ep_starts, ep_term_types = run_deployment(
                model, cond, drift_type, seed
            )
            calib   = calibrate_pilot(obs_arr, K0, W, S, B, ALPHA, CLIP_LO_P, CLIP_HI_P)
            D_k, y_k, starts = compute_windowed_metrics(
                obs_arr, reward_arr, calib, T_EVAL, W, S, ALPHA
            )
            beta_TS = theil_sen_slope(y_k, starts)
            ew      = compute_early_warning(
                D_k, y_k, calib, starts, K0,
                Q_WARN, Q_WATCH, C_FAIL, SIGMA_FLOOR_FACTOR,
            )
            mu_ref, sigma_ref = ew["mu_ref"], ew["sigma_ref"]
            T_soft  = compute_soft_failure(y_k, starts, mu_ref, sigma_ref, C_FAIL, M_FAIL)

            if HARD_FAILURE_COUNTS:
                cands  = [t for t in [T_hard, T_soft] if t is not None]
                T_comp = int(min(cands)) if cands else None
            else:
                T_comp = T_soft

            delta       = 1 if T_comp is not None else 0
            T_comp_obs  = T_comp if T_comp is not None else T_EVAL
            T_warn      = ew["T_warn"]
            timely      = bool(delta == 1 and T_warn is not None and T_warn < T_comp_obs)
            delta_delay = (
                int(T_warn - T0)
                if drift_type == DELTA_DELAY_SCHEDULE and T_warn is not None
                else None
            )

            metrics = {
                "env_id": ENV_ID, "domain": args.domain,
                "condition": cond, "seed": seed, "algo": ALGO,
                "drift_type": drift_type, "T_eval": T_EVAL,
                "K0": K0,
                "B": calib["B"], "B_source": calib["B_source"],
                "w": W, "s": S,
                "c_fail": C_FAIL, "m_fail": M_FAIL,
                "hard_failure_counts": HARD_FAILURE_COUNTS,
                "gradual_ramp_after_pilot": GRADUAL_RAMP_AFTER_PILOT,
                "mu": calib["mu"].tolist(), "sigma": calib["sigma"].tolist(),
                "clip_lo": calib["clip_lo"].tolist(), "clip_hi": calib["clip_hi"].tolist(),
                "pilot_D": calib["pilot_D"],
                "tau_warn": ew["tau_warn"], "tau_watch": ew["tau_watch"],
                "mu_ref": mu_ref, "sigma_ref": sigma_ref,
                "window_starts": [int(x) for x in starts],
                "D_k": D_k.tolist(), "y_k": y_k.tolist(),
                "A_deg": ew["A_deg"], "A_drift": ew["A_drift"],
                "A_watch": ew["A_watch"], "beta_TS": beta_TS,
                # Alarm burden (number of windows where each alarm fired)
                "n_alarm_windows_deg":   int(sum(ew["A_deg"])),
                "n_alarm_windows_drift": int(sum(ew["A_drift"])),
                "n_alarm_windows_watch": int(sum(ew["A_watch"])),
                "T_hard": T_hard, "T_soft": T_soft,
                "T_comp": T_comp, "T_comp_obs": T_comp_obs, "delta": delta,
                "T_deg": ew["T_deg"], "T_drift": ew["T_drift"], "T_warn": T_warn,
                "timely_warn": timely, "delta_delay": delta_delay,
                # Full episode data for survival analysis on episode lengths
                "episode_starts":    ep_starts,
                "episode_term_types": ep_term_types,   # "T"=terminated, "C"=truncated
                "n_episodes": len(ep_term_types),
            }
            rollout = {
                "env_id": ENV_ID, "domain": args.domain,
                "condition": cond, "seed": seed, "algo": ALGO,
                "drift_type": drift_type,
                "T_comp": T_comp_obs, "delta": delta,
                "T_warn": T_warn, "timely_warn": timely,
                "delta_delay": delta_delay,
                "T_hard": T_hard, "T_soft": T_soft,
                "beta_TS": beta_TS, "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            }

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            with open(rollouts_path, "w") as f:
                json.dump(rollout, f, indent=2)

            print(f"  [done] {cond} seed={seed} {drift_type} | "
                  f"δ={delta} T_comp={T_comp} T_warn={T_warn} "
                  f"timely={timely} Δ_delay={delta_delay} "
                  f"tau_warn={ew['tau_warn']:.6f} sigma_ref={sigma_ref:.6f} "
                  f"B={calib['B']}({calib['B_source']}) "
                  f"n_ep={len(ep_term_types)}")

# ── Aggregate rollouts (scan files — robust to re-runs) ───────────────────────
agg_path = os.path.join(RESULTS_DIR, f"all_rollouts_{args.domain}_summary.json")
all_rollouts = []
for fname in sorted(os.listdir(RESULTS_DIR)):
    if (fname.endswith("_rollouts_summary.json")
            and "all_rollouts" not in fname):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            all_rollouts.append(json.load(f))

with open(agg_path, "w") as f:
    json.dump(all_rollouts, f, indent=2)

print(f"\nAggregate rollouts ({len(all_rollouts)} runs) → {agg_path}")
print("Deployment audit complete.")
