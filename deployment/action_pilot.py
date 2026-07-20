#!/usr/bin/env python3
"""
action_pilot.py — Short deployment pilot with action logging
=============================================================
Runs the 2× gravity pilot (configs/hopper_2g_pilot.json) and records:
  - obs_arr, reward_arr (same as deployment.py)
  - act_arr            (NEW: raw action at every timestep)

Per-window action statistics added to metrics output:
  act_mean_k[k]  — mean action per dimension in window k           shape (w, d_act)→(d_act,)
  act_std_k[k]   — std  action per dimension in window k           shape (d_act,)
  act_ac1_k[k]   — lag-1 autocorrelation per dimension in window k shape (d_act,)
  act_std_agg_k[k] — mean of act_std_k across action dims (scalar) — summary volatility

These are then compared against D_k (obs shift) and y_k/A_deg (reward degradation)
to find the missing link: does obs-distribution shift translate to action volatility?

Usage:
    # Single worker (all conditions):
    python action_pilot.py

    # 3-worker parallel — one condition per worker, staggered I/O:
    python action_pilot.py --condition M1 --start_delay 0  &
    python action_pilot.py --condition M2 --start_delay 30 &
    python action_pilot.py --condition M3 --start_delay 60 &

    # --start_delay N  sleeps N seconds before the first rollout so workers
    # desync naturally and never hit the filesystem at the same moment.

Scale-up criterion (printed at end):
    A_deg fires in >= 2/3 seeds → scale up to full 5-seed × 3-transition run
    Otherwise                   → try ObsNoisePerturbation with noise_std_max=0.3
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t as t_dist

import gymnasium as gym
try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    sys.exit("stable-baselines3 not found. pip install stable-baselines3")

sys.path.insert(0, str(Path(__file__).parent))
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "core"))  # [src_clean] import shim
from drift_wrappers import PERTURBATION_REGISTRY

parser = argparse.ArgumentParser()
parser.add_argument("--config",      default="configs/hopper_2g_pilot.json")
parser.add_argument("--instrument",  default="configs/instrument.json")
parser.add_argument("--out",         default=None,
                    help="Override results_dir from config")
parser.add_argument("--condition",   default=None,
                    help="Run only this condition (e.g. M1). "
                         "One worker per condition for 3-way parallelism.")
parser.add_argument("--start_delay", type=int, default=0,
                    help="Sleep N seconds before starting rollouts. "
                         "Desync workers so filesystem writes never overlap. "
                         "Recommended: 0, 30, 60 for workers 1, 2, 3.")
args = parser.parse_args()

with open(args.config)     as f: dcfg = json.load(f)
with open(args.instrument) as f: icfg = json.load(f)

# ── constants ─────────────────────────────────────────────────────────────────
ENV_ID       = dcfg["env_id"]
RESULTS_DIR  = Path(args.out or dcfg["results_dir"])
CONDITIONS   = [args.condition] if args.condition else dcfg["conditions"]
SEEDS        = dcfg["seeds"]
T_EVAL       = dcfg["T_eval"]
W            = dcfg["w"]
S            = dcfg["s"]
B            = dcfg["B"]
ALGO         = dcfg["algo"]
CKPT_SUFFIX  = dcfg.get("ckpt_suffix", "")
PERTURBATION_WRAPPER = dcfg.get("perturbation_wrapper", dcfg.get("drift_wrapper"))
TRANSITION_SCHEDULES = dcfg.get("transition_schedules", dcfg.get("drift_schedules"))
DRIFT_PARAMS = dcfg.get("drift_params", {})
HARD_FAILURE_COUNTS = dcfg.get("hard_failure_counts", False)
GRADUAL_RAMP = dcfg.get("gradual_ramp_after_pilot", True)
SAFETY_MARGINS = dcfg.get("safety_margins", [])

K0              = icfg.get("K0", 4)
ALPHA           = icfg.get("alpha", 0.5)
CLIP_LO_P       = icfg.get("clip_lo_p", 0.5)
CLIP_HI_P       = icfg.get("clip_hi_p", 99.5)
C_FAIL          = icfg.get("c_fail", 2.0)
M_FAIL          = icfg.get("m_fail", 2)
Q_WARN          = icfg.get("q_warn", 0.95)
Q_WATCH         = icfg.get("q_watch", 0.90)
# tolerate both "lambda_FA" (v12.6 Cópia) and "lambda_fa" (NAS original)
LAMBDA_FA       = icfg.get("lambda_FA", icfg.get("lambda_fa", 0.05))
EPS_STD         = icfg.get("eps_std", 1e-8)
SIGMA_FLOOR_FAC = icfg.get("sigma_ref_relative_floor", 0.01)

T0        = T_EVAL // 2
PILOT_END = ((K0 - 1) * S + W) if GRADUAL_RAMP else 0

assert PILOT_END < T0, (
    f"PILOT_END={PILOT_END} >= T0={T0}. "
    f"Increase T_eval (need T_eval > 2×PILOT_END = {2*PILOT_END}). "
    f"Current T_eval={T_EVAL}."
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Config     : {args.config}")
print(f"Results    : {RESULTS_DIR}")
print(f"T_eval={T_EVAL:,}  T0={T0:,}  pilot_end={PILOT_END:,}")
print(f"K0={K0}  w={W:,}  s={S:,}  B={B}")
print(f"Perturbation : {PERTURBATION_WRAPPER}  params={DRIFT_PARAMS}")
print(f"Conditions : {CONDITIONS}  Seeds: {SEEDS}  Transition schedules: {TRANSITION_SCHEDULES}")
print()

# ── env / model ───────────────────────────────────────────────────────────────
# Model checkpoints are in the ORIGINAL hopper results dir (same models)
ORIG_RESULTS = Path(dcfg.get("_model_source", "results/mujoco/hopper"))

def load_model(path):
    try:
        return SAC.load(path, device="cpu")
    except Exception:
        return PPO.load(path, device="cpu")

def make_env(seed, transition):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    WrapperClass = PERTURBATION_REGISTRY[PERTURBATION_WRAPPER]
    # Build wrapper kwargs from drift_params — strip comment keys, add fixed args
    wrapper_kwargs = {k: v for k, v in DRIFT_PARAMS.items()
                      if not k.startswith("_")}
    # NOTE: the constructor kwarg is "transition" (see core/drift_wrappers.py) —
    # this call previously used the stale "drift_type" kwarg name, which does
    # not exist on the current wrapper classes and would raise a TypeError.
    env = WrapperClass(env, transition=transition, t0=T0, pilot_end=PILOT_END,
                       **wrapper_kwargs)
    return env

# ── window helpers (from deployment.py) ──────────────────────────────────────
def get_window_starts(T_eval, w, s):
    starts = []
    t = 0
    while True:
        if t + w > T_eval:
            break
        starts.append(t)
        t += s
    return starts

def jsd_1d(p, q, eps=1e-10):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b): return np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def compute_jsd_window(std_obs_window, ref_pmf, bin_edges_per_feat, alpha, B):
    d = std_obs_window.shape[1]
    jsds = []
    for j in range(d):
        counts, _ = np.histogram(std_obs_window[:, j], bins=bin_edges_per_feat[j])
        p = counts.astype(float)
        jsds.append(jsd_1d(p, ref_pmf[j]))
    return float(np.mean(jsds))

def calibrate_pilot(obs_arr):
    mu    = obs_arr.mean(axis=0)
    sigma = obs_arr.std(axis=0) + EPS_STD
    std_p = np.clip((obs_arr - mu) / sigma,
                    np.percentile((obs_arr - mu) / sigma, CLIP_LO_P, axis=0),
                    np.percentile((obs_arr - mu) / sigma, CLIP_HI_P, axis=0))

    clip_lo = np.percentile(std_p, CLIP_LO_P, axis=0)
    clip_hi = np.percentile(std_p, CLIP_HI_P, axis=0)

    starts = get_window_starts(T_EVAL, W, S)[:K0]
    bin_edges = []
    for j in range(obs_arr.shape[1]):
        edges = np.linspace(clip_lo[j], clip_hi[j], B + 1)
        bin_edges.append(edges)

    pilot_pmfs = []
    for ps in starts:
        w_obs = np.clip((obs_arr[ps:ps+W] - mu) / sigma, clip_lo, clip_hi)
        pmf_j = []
        for j in range(obs_arr.shape[1]):
            counts, _ = np.histogram(w_obs[:, j], bins=bin_edges[j])
            pmf_j.append(counts.astype(float) / (counts.sum() + 1e-10))
        pilot_pmfs.append(pmf_j)

    pilot_D = []
    for k in range(K0):
        loo_ref = [
            np.mean([pilot_pmfs[i][j] for i in range(K0) if i != k], axis=0)
            for j in range(obs_arr.shape[1])
        ]
        pilot_D.append(compute_jsd_window(
            np.clip((obs_arr[starts[k]:starts[k]+W] - mu) / sigma, clip_lo, clip_hi),
            loo_ref, bin_edges, ALPHA, B
        ))

    ref_pmf = [
        np.mean([pilot_pmfs[i][j] for i in range(K0)], axis=0)
        for j in range(obs_arr.shape[1])
    ]

    return {
        "mu": mu, "sigma": sigma, "clip_lo": clip_lo, "clip_hi": clip_hi,
        "bin_edges": bin_edges, "ref_pmf": ref_pmf, "pilot_D": pilot_D, "B": B
    }

def evt_tau(pilot_D):
    n = len(pilot_D)
    if n < 2: return float(np.max(pilot_D))
    mu = np.mean(pilot_D); s = np.std(pilot_D, ddof=1)
    tc = float(t_dist.ppf(1.0 - LAMBDA_FA, df=n - 1))
    return float(mu + tc * s * np.sqrt(1.0 + 1.0 / n))

def action_window_stats(act_window):
    """act_window shape: (w, d_act)"""
    mean = act_window.mean(axis=0)          # (d_act,)
    std  = act_window.std(axis=0)           # (d_act,)
    # lag-1 autocorrelation per dimension
    ac1 = np.array([
        float(np.corrcoef(act_window[:-1, j], act_window[1:, j])[0, 1])
        if act_window.shape[0] > 1 else 0.0
        for j in range(act_window.shape[1])
    ])
    std_agg = float(std.mean())             # scalar summary
    return mean, std, ac1, std_agg

# ── rollout with action logging ───────────────────────────────────────────────
def run_pilot_deployment(model, env, seed):
    obs, _ = env.reset(seed=seed)
    obs_list, rew_list, act_list = [], [], []
    ep_starts = [0]
    T_hard = None

    for t in range(T_EVAL):
        obs_list.append(obs.copy())
        action, _ = model.predict(obs, deterministic=True)
        act_list.append(action.copy())
        obs, reward, terminated, truncated, _ = env.step(action)
        rew_list.append(float(reward))

        if (terminated or truncated):
            if T_hard is None:
                T_hard = t + 1
            obs, _ = env.reset()   # no seed — let env advance naturally
            if t + 1 < T_EVAL:
                ep_starts.append(t + 1)

    obs_arr = np.array(obs_list, dtype=np.float32)
    rew_arr = np.array(rew_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)
    return obs_arr, rew_arr, act_arr, T_hard, ep_starts

# ── windowed metrics (obs + reward + actions) ─────────────────────────────────
def compute_all_metrics(obs_arr, rew_arr, act_arr, calib):
    starts  = get_window_starts(T_EVAL, W, S)
    mu, sigma, clip_lo, clip_hi = (
        calib["mu"], calib["sigma"], calib["clip_lo"], calib["clip_hi"]
    )
    D_k = []; y_k = []; act_std_k = []; act_ac1_k = []; act_std_agg_k = []

    for ps in starts:
        std_obs = np.clip((obs_arr[ps:ps+W] - mu) / sigma, clip_lo, clip_hi)
        D_k.append(compute_jsd_window(std_obs, calib["ref_pmf"],
                                      calib["bin_edges"], ALPHA, B))
        y_k.append(float(rew_arr[ps:ps+W].mean()))
        a_win = act_arr[ps:ps+W]
        _, std, ac1, std_agg = action_window_stats(a_win)
        act_std_k.append(std.tolist())
        act_ac1_k.append(ac1.tolist())
        act_std_agg_k.append(std_agg)

    tau_warn  = evt_tau(calib["pilot_D"])
    tau_watch = float(np.quantile(calib["pilot_D"], Q_WATCH))

    mu_ref    = float(np.mean(y_k[:K0]))
    sigma_ref = float(np.std(y_k[:K0], ddof=1)) if K0 > 1 else 1.0
    sigma_ref = max(sigma_ref, SIGMA_FLOOR_FAC * abs(mu_ref), EPS_STD)

    eligible = np.array([s >= PILOT_END for s in starts])
    D_arr    = np.array(D_k)
    y_arr    = np.array(y_k)

    A_drift = ((D_arr > tau_warn) & eligible).astype(int)
    A_watch = ((D_arr > tau_watch) & eligible).astype(int)

    # A_deg: y_k < mu_ref - c_fail * sigma_ref for m_fail consecutive windows
    deg_flag = (y_arr < mu_ref - C_FAIL * sigma_ref) & eligible
    A_deg = np.zeros(len(starts), dtype=int)
    consec = 0
    for i, flag in enumerate(deg_flag):
        if flag:
            consec += 1
            if consec >= M_FAIL:
                A_deg[i - M_FAIL + 1:i + 1] = 1
        else:
            consec = 0

    T_drift = next((starts[i] for i, v in enumerate(A_drift) if v), None)
    T_deg   = next((starts[i] for i, v in enumerate(A_deg)   if v), None)

    # beta_TS (trend slope over all post-pilot windows)
    post = np.where(eligible)[0]
    if len(post) >= 2:
        x = np.array([starts[i] for i in post], dtype=float)
        y = y_arr[post]
        beta_TS = float(np.polyfit(x, y, 1)[0])
    else:
        beta_TS = 0.0

    return {
        "window_starts": [int(s) for s in starts],
        "D_k": [float(v) for v in D_k],
        "y_k": [float(v) for v in y_k],
        "A_drift": A_drift.tolist(),
        "A_watch": A_watch.tolist(),
        "A_deg":   A_deg.tolist(),
        "act_std_agg_k": [float(v) for v in act_std_agg_k],
        "act_std_k":     act_std_k,
        "act_ac1_k":     act_ac1_k,
        "tau_warn":  tau_warn,
        "tau_watch": tau_watch,
        "mu_ref":    mu_ref,
        "sigma_ref": sigma_ref,
        "beta_TS":   beta_TS,
        "T_drift":   T_drift,
        "T_deg":     T_deg,
    }

# ── figure: D_k + y_k + act_std per run ──────────────────────────────────────
def plot_run(metrics, T_hard, cond, seed, transition, out_path):
    ws      = np.array(metrics["window_starts"])
    mids    = (ws + W // 2) / 1000      # x-axis in ×10³
    D_k     = np.array(metrics["D_k"])
    y_k     = np.array(metrics["y_k"])
    A_deg   = np.array(metrics["A_deg"])
    act_std = np.array(metrics["act_std_agg_k"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax1, ax2, ax3 = axes

    # Panel 1: D_k
    ax1.plot(mids, D_k, lw=1.8, color="#1f77b4", label="D_k (JSD)")
    ax1.axhline(metrics["tau_warn"],  ls="--", lw=1.2, color="red",    label=f"τ_warn={metrics['tau_warn']:.2e}")
    ax1.axhline(metrics["tau_watch"], ls=":",  lw=1.0, color="orange",  label=f"τ_watch={metrics['tau_watch']:.2e}")
    ax1.axvline(T0/1000, color="black", lw=1.5, ls="-", label=f"t₀={T0//1000}k")
    ax1.set_ylabel("D_k (JSD)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: y_k + A_deg shading
    ax2.plot(mids, y_k, lw=1.8, color="#2ca02c", label="y_k (mean reward)")
    ax2.axhline(metrics["mu_ref"],                       ls="--", lw=1, color="gray")
    ax2.axhline(metrics["mu_ref"] - C_FAIL * metrics["sigma_ref"],
                ls=":", lw=1.2, color="purple", label="A_deg threshold")
    for i, flag in enumerate(A_deg):
        if flag:
            ax2.axvspan(mids[i] - S/2000, mids[i] + S/2000, color="purple", alpha=0.2)
    if T_hard is not None:
        ax2.axvline(T_hard/1000, color="red", lw=1.5, ls="--", label=f"T_hard={T_hard//1000}k")
    ax2.set_ylabel("y_k (mean reward)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: action std (aggregated across dims)
    ax3.plot(mids, act_std, lw=1.8, color="#d62728", label="act_std (mean over dims)")
    ax3.axvline(T0/1000, color="black", lw=1.5, ls="-")
    ax3.set_ylabel("Action σ")
    ax3.set_xlabel("Steps (×10³)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    t_drift_str = f"{metrics['T_drift']//1000}k" if metrics['T_drift'] else "—"
    t_deg_str   = f"{metrics['T_deg']//1000}k"   if metrics['T_deg']   else "—"
    t_hard_str  = f"{T_hard//1000}k"              if T_hard else "—"
    drift_label = (f"g={DRIFT_PARAMS.get('g_drifted')}" if "g_drifted" in DRIFT_PARAMS
                   else f"noise_std={DRIFT_PARAMS.get('noise_std_max')}" if "noise_std_max" in DRIFT_PARAMS
                   else PERTURBATION_WRAPPER)
    title = (f"{cond} seed{seed} | {transition} | {drift_label} "
             f"| T_drift={t_drift_str}  T_deg={t_deg_str}  T_hard={t_hard_str}")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
# ── desync I/O across parallel workers ───────────────────────────────────────
# Each worker sleeps --start_delay seconds before doing anything.
# This staggers the initial env.reset(), model load, and all subsequent writes
# so workers never compete on the filesystem at the same moment.
# Recommended stagger: worker 1 → 0s, worker 2 → 30s, worker 3 → 60s.
if args.start_delay > 0:
    print(f"[desync] sleeping {args.start_delay}s before starting "
          f"(worker for condition {CONDITIONS}) ...")
    time.sleep(args.start_delay)

# helper: atomic JSON write — write to .tmp then rename so a concurrent
# reader never sees a half-written file if two workers finish close together.
def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)

summary_rows = []

for cond in CONDITIONS:
    for seed in SEEDS:
        ckpt_name = f"{cond}{CKPT_SUFFIX}_seed{seed}.zip"
        ckpt_path = Path("results/mujoco/hopper") / ckpt_name
        if not ckpt_path.exists():
            print(f"[skip] {ckpt_name} not found")
            continue

        model = load_model(str(ckpt_path))
        print(f"\n── {cond} seed{seed} ──")

        for transition in TRANSITION_SCHEDULES:
            prefix = f"{cond}_seed{seed}_{transition}"
            print(f"  Running {prefix} | T_eval={T_EVAL:,} | t0={T0:,}")

            env = make_env(seed=seed, transition=transition)
            obs_arr, rew_arr, act_arr, T_hard, ep_starts = \
                run_pilot_deployment(model, env, seed=seed)
            env.close()

            pilot_end_idx = PILOT_END   # actual step index
            calib = calibrate_pilot(obs_arr[:pilot_end_idx + W])

            metrics = compute_all_metrics(obs_arr, rew_arr, act_arr, calib)

            out_json = RESULTS_DIR / f"{prefix}_pilot_metrics.json"
            save_dict = {k: v for k, v in metrics.items()
                         if not isinstance(v, np.ndarray)}
            save_dict.update({
                "condition": cond, "seed": seed, "transition": transition,
                "T_eval": T_EVAL, "T0": T0, "pilot_end": PILOT_END,
                "T_hard": T_hard, "K0": K0, "w": W, "s": S,
                "g_drifted": DRIFT_PARAMS.get("g_drifted"),
                "pilot_D": calib["pilot_D"],
                "n_episodes": len(ep_starts),
            })
            atomic_write_json(out_json, save_dict)

            plot_run(metrics, T_hard, cond, seed, transition,
                     RESULTS_DIR / f"{prefix}_pilot.png")

            A_deg_fires = any(metrics["A_deg"])
            summary_rows.append({
                "run":         prefix,
                "condition":   cond,
                "seed":        seed,
                "transition":  transition,
                "T_hard":      T_hard,
                "T_drift":     metrics["T_drift"],
                "T_deg":       metrics["T_deg"],
                "A_deg_fires": A_deg_fires,
                "beta_TS":     metrics["beta_TS"],
                "tau_warn":    metrics["tau_warn"],
                "mu_ref":      metrics["mu_ref"],
                "sigma_ref":   metrics["sigma_ref"],
            })
            print(f"    T_hard={T_hard}  T_drift={metrics['T_drift']}  "
                  f"T_deg={metrics['T_deg']}  A_deg={A_deg_fires}  "
                  f"β_TS={metrics['beta_TS']:.3e}")

# ── summary ──────────────────────────────────────────────────────────────────
sdf = pd.DataFrame(summary_rows)
print("\n" + "="*60)
print("PILOT SUMMARY")
print("="*60)
print(sdf.to_string(index=False))
sdf.to_csv(RESULTS_DIR / "pilot_summary.csv", index=False)

n_adeg = sdf["A_deg_fires"].sum()
n_runs = len(sdf)
print(f"\nA_deg fires: {n_adeg}/{n_runs} runs")
print()
if n_adeg >= 2:
    print("✓ SCALE UP: A_deg fires in ≥2/3 runs.")
    print("  Next: run full 5-seed × 3-transition-type deployment run.")
else:
    print(f"✗ A_deg did not fire in ≥2/3 runs ({n_adeg}/{n_runs}).")
    print("  Options: increase perturbation magnitude or check env.reset() fix is applied.")
print(f"\nResults: {RESULTS_DIR}/")
