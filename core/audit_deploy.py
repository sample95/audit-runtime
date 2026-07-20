"""
audit_deploy.py — Config-driven persistent deployment audit for RL policies
Methodology: v12.9

Usage (from audit-framework/):
    python audit_deploy.py --domain hopper          # full 45-run evaluation
    python audit_deploy.py --domain hopper_t01      # T-01 FAR diagnostic
    python audit_deploy.py --domain hopper --condition M1   # worker mode

Config resolution:
    domain config  : {configs_dir}/{domain}.json        (env, seeds, T_eval, drift, ...)
    instrument cfg : {configs_dir}/instrument.json      (K0, alpha, tau_warn_floor, ...)

Checkpoint format:
    ./models/{COND}_seed{SEED}/   (SB3 directory format; each dir contains model.zip)
    Override with "ckpt_dir" in domain config.

NAS compatibility:
    Sets MUJOCO_GL=osmesa if not already set. Safe to override before launching.

Data extraction design (see tuning_journal.md §Data Extraction):
    CONFIRMATORY fields (primary evaluation, locked v12.9):
        pilot_D, D_k, y_k, A_drift, A_deg, A_watch, tau_warn, tau_watch,
        T_warn, T_comp, T_soft, T_hard, delta, timely_warn, mu_ref, sigma_ref
        mean_ep_len_k, A_len, mu_len_ref, sigma_len_ref, T_len   [A_len channel, T-06+]
        beta_ts_series, T_bts, tau_bts, beta_ts_window             [β_TS channel, T-15+]
        rul_p50_series, rul_p90_series, T_pred_p50_series, T_pred_p90_series
            [conditional Weibull RUL, opt-in via domain cfg "rul_weibull"; all None if disabled]

    AUXILIARY fields (exploratory EDA only — do NOT feed back to confirmatory method):
        aux_action_windows: per-window action statistics
            act_mean_norm   — mean ||a_t|| per window
            act_std_norm    — std of ||a_t|| per window
            act_smoothness  — mean ||a_t − a_{t−1}|| per window (action change rate)

    OCSVM COMPARISON CHANNEL (opt-in, set ocsvm_enabled=true in domain config):
        Provides per-step anomaly scoring as a cross-validation channel for D_k.
        Trained on pilot obs (same standardisation as JSD); scores each deployment
        step; aggregated per window as O_k (anomaly fraction). Confirms D_k alarm
        concordance without requiring new runs.
        O_k        — per-window fraction of steps classified as anomalous by OCSVM
        A_ocsvm    — per-window alarm flag (O_k > tau_ocsvm, post-pilot only)
        T_ocsvm    — window end of first A_ocsvm==1 alarm
        tau_ocsvm  — calibrated threshold = Q_{q_warn}(pilot_O)
        pilot_O    — per-pilot-window anomaly fractions (calibration basis)

    TRAJECTORY files (*_trajectory.npz, optional, set save_trajectories=true in domain cfg):
        obs [T_eval × obs_dim], actions [T_eval × act_dim],
        rewards [T_eval], ep_starts [n_episodes]
        Used only by exploratory_analysis.py, never by deploy_results.py.
"""

import os
import sys
import json
import argparse
import math
import glob

# ── Headless rendering (NAS / no display) ─────────────────────────────────────
os.environ.setdefault("MUJOCO_GL", "osmesa")

import pickle
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

_parser = argparse.ArgumentParser(description="Config-driven deployment audit (v12.9)")
_parser.add_argument("--domain",    required=True, help="Domain config name, e.g. hopper_t01")
_parser.add_argument("--configs",   default="./configs", help="Path to configs directory")
_parser.add_argument("--condition",  default=None, help="Run only this condition (worker mode)")
_parser.add_argument("--transition", default=None, help="Run only this transition schedule (worker mode)")
_args = _parser.parse_args()

CONFIGS_DIR       = _args.configs
DOMAIN            = _args.domain
COND_FILTER       = _args.condition
TRANSITION_FILTER = _args.transition

def _load_json(path):
    with open(path) as f:
        return json.load(f)

domain_cfg = _load_json(os.path.join(CONFIGS_DIR, f"{DOMAIN}.json"))
instr_cfg  = _load_json(os.path.join(CONFIGS_DIR, "instrument.json"))

# ── Instrument parameters (v12.6 locked) ─────────────────────────────────────
K0              = int(instr_cfg["K0"])
ALPHA           = float(instr_cfg["alpha"])
CLIP_LO_P       = float(instr_cfg.get("clip_lo_p", 0.5))
CLIP_HI_P       = float(instr_cfg.get("clip_hi_p", 99.5))
C_FAIL          = float(instr_cfg["c_fail"])
M_FAIL          = int(instr_cfg["m_fail"])
Q_WARN          = float(instr_cfg["q_warn"])
Q_WATCH         = float(instr_cfg["q_watch"])
EPS_STD         = float(instr_cfg.get("eps_std", 1e-8))
SIGMA_REL_FLOOR = float(instr_cfg.get("sigma_ref_relative_floor", 0.01))
TAU_WARN_FLOOR  = float(domain_cfg.get("tau_warn_floor",          # domain override
                        instr_cfg.get("tau_warn_floor", 0.0)))    # else instrument default
C_LEN           = float(instr_cfg.get("c_len", 2.0))    # A_len threshold multiplier
K_CUSUM_FACTOR  = float(instr_cfg.get("k_cusum_factor", 0.5))   # CUSUM slack
H_CUSUM_FACTOR  = float(instr_cfg.get("h_cusum_factor", 5.0))   # CUSUM threshold
BETA_TS_WINDOW  = int(instr_cfg.get("beta_ts_window", 5))        # rolling OLS window (windows)
TAU_BTS         = float(instr_cfg.get("tau_bts", -5e-7))          # β_TS alarm threshold; −5e-7 ≈ 25% reward loss over 1M steps

# OCSVM comparison channel — opt-in, disabled by default
OCSVM_ENABLED = bool(domain_cfg.get("ocsvm_enabled", False))
OCSVM_NU      = float(domain_cfg.get("ocsvm_nu",   instr_cfg.get("ocsvm_nu",   0.05)))   # upper bound on training anomaly fraction
OCSVM_KERNEL  = str(domain_cfg.get("ocsvm_kernel", instr_cfg.get("ocsvm_kernel", "rbf")))

ENV_ID      = domain_cfg["env_id"]
RESULTS_DIR = domain_cfg["results_dir"]
CONDITIONS  = domain_cfg["conditions"]
SEEDS       = domain_cfg["seeds"]
T_EVAL      = int(domain_cfg["T_eval"])
B           = int(domain_cfg.get("B", 21))
W           = int(domain_cfg.get("w", 50000))
S           = int(domain_cfg.get("s", 25000))
ENV_KWARGS          = domain_cfg.get("env_kwargs", {})          # e.g. hover agent_hz / flight_mode
CONDITION_DURATIONS = domain_cfg.get("condition_durations", {}) # e.g. Q1=10.0s, Q2=2.5s
OBS_WRAPPER = domain_cfg.get("obs_wrapper")  # e.g. "TimeFeatureWrapper" for externally-trained zoo policies

CKPT_DIR        = domain_cfg.get("ckpt_dir", RESULTS_DIR)   # default: same dir as results
CKPT_SUFFIX       = domain_cfg.get("ckpt_suffix", "")
ALGO_NAME         = domain_cfg.get("algo", "PPO")            # "PPO" or "SAC"
_ALGO_MAP         = {"PPO": PPO, "SAC": SAC}
AlgoCls           = _ALGO_MAP.get(ALGO_NAME, PPO)
# "perturbation_wrapper" names the MECHANISM (what physically changes); "transition_schedules"
# lists the SCHEDULE values (when/how it activates). Both accept the pre-rename config keys
# ("drift_wrapper" / "drift_schedules") as fallbacks so older configs keep working unchanged.
PERTURBATION_WRAPPER = domain_cfg.get("perturbation_wrapper", domain_cfg.get("drift_wrapper", "GravityPerturbation"))
TRANSITION_SCHEDULES = domain_cfg.get("transition_schedules", domain_cfg.get("drift_schedules", ["baseline", "abrupt", "gradual"]))
DRIFT_PARAMS      = domain_cfg.get("drift_params", {})
HARD_FAIL_COUNTS  = bool(domain_cfg.get("hard_failure_counts", False))
SUPPRESS_HF_WARN  = bool(domain_cfg.get("suppress_hard_failure_warning", False))
SAVE_TRAJECTORIES = bool(domain_cfg.get("save_trajectories", False))   # EDA only

PILOT_END = (K0 - 1) * S + W          # last step of final pilot window
T0        = int(domain_cfg.get("t0_abrupt", T_EVAL // 2))  # abrupt switch point (configurable)

# Conditional RUL Weibull parameters (opt-in; both kappa and lambda must be > 0 to enable)
# Add "rul_weibull": {"kappa": k, "lambda": l} to domain config to activate per-run RUL series.
# PPO gradual: kappa=7.91, lambda=214830   (from T-16 fit, n=13/15 detected)
# VecNorm gradual: kappa=5.19, lambda=393692 (from T-16 fit, n=15/15 detected)
_rul_wb    = domain_cfg.get("rul_weibull", {})
RUL_KAPPA  = float(_rul_wb.get("kappa",  0.0))
RUL_LAMBDA = float(_rul_wb.get("lambda", 0.0))

torch.set_num_threads(1)
os.makedirs(RESULTS_DIR, exist_ok=True)

from drift_wrappers import PERTURBATION_REGISTRY
import gymnasium as gym

if PERTURBATION_WRAPPER in ("ObsNoisePerturbation", "ObsNoiseDrift"):
    import PyFlyt.gym_envs  # noqa: F401 — registers PyFlyt namespaces with gymnasium

PerturbationWrapper = PERTURBATION_REGISTRY[PERTURBATION_WRAPPER]

def first_eligible_warning_start(K0: int, W: int, S: int) -> int:
    pilot_end = (K0 - 1) * S + W
    return math.ceil(pilot_end / S) * S


def get_window_starts(T_eval, w, s):
    starts, k = [], 0
    while True:
        t_start = k * s
        if t_start + w > T_eval:
            break
        starts.append(t_start)
        k += 1
    return starts


def compute_pmf(std_obs, bin_edges_per_feat, alpha, n_bins):
    d   = std_obs.shape[1]
    pmf = np.zeros((d, n_bins), dtype=np.float64)
    for j in range(d):
        counts, _ = np.histogram(std_obs[:, j], bins=bin_edges_per_feat[j])
        pmf[j]    = (counts + alpha) / (counts.sum() + alpha * n_bins)
    return pmf


def jsd_single(p, q):
    eps = 1e-12
    m   = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def jsd_aggregate(pmf_k, ref_pmf):
    d = pmf_k.shape[0]
    return float(np.mean([jsd_single(ref_pmf[j], pmf_k[j]) for j in range(d)]))


def calibrate_pilot(obs_arr, K0, w, s, B, alpha, clip_lo_p, clip_hi_p):
    T, d   = obs_arr.shape
    starts = get_window_starts(T, w, s)
    assert len(starts) >= K0, f"Insufficient windows for K0={K0}: got {len(starts)}"

    pilot_starts = starts[:K0]
    pilot_mask   = np.zeros(T, dtype=bool)
    for ps in pilot_starts:
        pilot_mask[ps:ps + w] = True
    pilot_obs = obs_arr[pilot_mask]

    mu    = pilot_obs.mean(axis=0)
    sigma = pilot_obs.std(axis=0) + EPS_STD
    std_pilot = (pilot_obs - mu) / sigma

    clip_lo = np.percentile(std_pilot, clip_lo_p, axis=0)
    clip_hi = np.percentile(std_pilot, clip_hi_p, axis=0)

    bin_edges = np.array([
        np.linspace(clip_lo[j], clip_hi[j], B + 1)
        for j in range(d)
    ])

    pilot_pmfs = []
    for ps in pilot_starts:
        win_std = np.clip((obs_arr[ps:ps + w] - mu) / sigma, clip_lo, clip_hi)
        pilot_pmfs.append(compute_pmf(win_std, bin_edges, alpha, B))
    pilot_pmfs = np.array(pilot_pmfs)   # [K0, d, B]

    ref_pmf = pilot_pmfs.mean(axis=0)

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
    starts  = get_window_starts(T_eval, w, s)
    mu      = calib["mu"];      sigma   = calib["sigma"]
    clip_lo = calib["clip_lo"]; clip_hi = calib["clip_hi"]
    be      = calib["bin_edges"]; ref_pmf = calib["ref_pmf"]
    d       = ref_pmf.shape[0]   # number of observation dimensions

    D_k, y_k, D_k_dims = [], [], []
    for ps in starts:
        win_obs = obs_arr[ps:ps + w]
        win_rew = reward_arr[ps:ps + w]
        if len(win_obs) < w:
            break   # partial window at end of truncated deployment — discard
        std_obs = np.clip((win_obs - mu) / sigma, clip_lo, clip_hi)
        pmf     = compute_pmf(std_obs, be, alpha, B)
        per_dim = [jsd_single(ref_pmf[j], pmf[j]) for j in range(d)]
        D_k.append(float(np.mean(per_dim)))
        D_k_dims.append(per_dim)
        y_k.append(float(win_rew.mean()))

    # Trim starts to match the number of complete windows computed
    starts = starts[:len(D_k)]
    return np.array(D_k), np.array(y_k), starts, D_k_dims


# ── Auxiliary: windowed action features (EDA only) ────────────────────────────
def compute_windowed_action_features(act_arr: np.ndarray, starts: list, w: int) -> dict:
    """
    Compute per-window action statistics for EXPLORATORY analysis only.
    These are AUXILIARY fields — they do not feed into any confirmatory metric.
    See exploratory_analysis.py for their use.

    Returns a dict of lists (one value per window):
        act_mean_norm   mean ||a_t||        — overall action magnitude
        act_std_norm    std  ||a_t||        — within-window magnitude variability
        act_smoothness  mean ||a_t−a_{t−1}||— step-to-step action change rate
    """
    act_mean_norm, act_std_norm, act_smoothness = [], [], []
    for ps in starts:
        win = act_arr[ps : ps + w]                           # [w, act_dim]
        norms = np.linalg.norm(win, axis=1)                  # [w]
        act_mean_norm.append(float(norms.mean()))
        act_std_norm.append(float(norms.std()))
        if len(win) > 1:
            diffs = np.linalg.norm(np.diff(win, axis=0), axis=1)
            act_smoothness.append(float(diffs.mean()))
        else:
            act_smoothness.append(0.0)
    return {
        "act_mean_norm":  act_mean_norm,
        "act_std_norm":   act_std_norm,
        "act_smoothness": act_smoothness,
    }


# ── A_len: episode-length degradation channel ────────────────────────────────
def compute_ep_len_per_window(ep_starts_full: list, starts: list, w: int) -> np.ndarray:
    """
    Mean episode length for episodes starting within each window [ws, ws+w).
    Uses the FULL ep_starts list (before the [:50] save truncation).
    Returns NaN for windows where no episode starts (handles sparse policies).

    Episode length = gap to next episode start (last episode extends to
    max(ep_starts) + estimated_len, capped so it doesn't inflate the window mean).
    We use a robust cap: lengths > 10 × median are replaced with median to prevent
    the last episode's open-ended length from corrupting the window mean.
    """
    arr = np.array(ep_starts_full, dtype=np.int64)
    if len(arr) == 0:
        return np.full(len(starts), np.nan)

    raw_lengths = np.diff(arr)                       # length[i] = starts[i+1] - starts[i]
    if len(raw_lengths) == 0:
        return np.full(len(starts), np.nan)

    # Robust cap for the implicit last episode (open-ended)
    med = float(np.median(raw_lengths))
    cap = 10.0 * med if med > 0 else 1e9
    lengths = np.append(raw_lengths, min(cap, raw_lengths[-1]))  # conservative last-ep estimate

    result = np.full(len(starts), np.nan)
    for i, ws in enumerate(starts):
        mask = (arr >= ws) & (arr < ws + w)
        if mask.any():
            result[i] = float(lengths[mask].mean())
    return result


def compute_len_failure(mean_ep_len_k: np.ndarray, starts: list,
                        K0: int, c_len: float, sigma_rel_floor: float,
                        m_fail: int) -> dict:
    """
    Calibrate A_len from pilot windows, then find T_len on evaluation windows.

    A_len_k = I[mean_ep_len_k < mu_len_ref - c_len * sigma_len_ref]
    T_len   = first window start where m_fail consecutive A_len_k == 1.

    Immune to healthy_reward masking: episode length shortens under drift
    regardless of per-step alive bonuses (Pardo 2018; Greenberg & Mannor 2021).
    Motivated by reward-corruption framework (Everitt et al. 2017; Skalse et al. 2022).
    """
    pilot_lens = mean_ep_len_k[:K0]
    valid      = pilot_lens[~np.isnan(pilot_lens)]

    if len(valid) == 0:
        return {
            "mu_len_ref": None, "sigma_len_ref": None, "threshold_len": None,
            "A_len": [0] * len(starts), "T_len": None,
        }

    mu_len    = float(valid.mean())
    sigma_len = float(valid.std(ddof=0))
    # Floor prevents near-zero sigma when pilot episodes are all identical length
    sigma_len = max(sigma_len, sigma_rel_floor * abs(mu_len)) if mu_len != 0 else max(sigma_len, 1.0)
    threshold = mu_len - c_len * sigma_len

    # A_len only eligible after pilot phase (same first_eligible logic as A_drift/A_deg)
    first_warn_start = first_eligible_warning_start(K0, W, S)
    A_len  = []
    T_len  = None
    consec = 0

    for k, (ep_len_w, ws) in enumerate(zip(mean_ep_len_k, starts)):
        if ws < first_warn_start:
            A_len.append(0)
            continue
        fires = (not np.isnan(ep_len_w)) and (ep_len_w < threshold)
        A_len.append(int(fires))
        if fires:
            consec += 1
            if consec >= m_fail and T_len is None:
                T_len = int(starts[k - m_fail + 1])
        else:
            consec = 0

    return {
        "mu_len_ref":    mu_len,
        "sigma_len_ref": sigma_len,
        "threshold_len": threshold,
        "A_len":         A_len,
        "T_len":         T_len,
    }


# ── CUSUM: cumulative sum change-point detector on D_k ───────────────────────
def compute_cusum_dk(D_k, pilot_D, starts, K0, k_factor, h_factor, w=0):
    """
    One-sided upper CUSUM on D_k for sustained drift detection.

    S_k = max(0, S_{k-1} + (D_k[k] - mu_pilot) - k_cusum)  for k >= K0
    T_cusum = first window start where S_k > h_cusum.

    k_cusum = k_factor × sigma(pilot_D)  — slack (insensitivity to noise)
    h_cusum = h_factor × sigma(pilot_D)  — decision threshold

    CUSUM accumulates evidence of a sustained upward shift in D_k rather than
    requiring m_fail consecutive threshold crossings. More sensitive to small
    gradual shifts; shares the same S=25k window granularity floor as T_warn.
    Pilot windows (k < K0) are excluded from accumulation.

    Returns dict with cusum_S, T_cusum, k_cusum, h_cusum, mu_pilot_D.
    """
    mu_pilot    = float(np.mean(pilot_D)) if pilot_D else 0.0
    sigma_pilot = float(np.std(pilot_D, ddof=1)) if len(pilot_D) > 1 else 1e-8
    k_cusum     = k_factor * sigma_pilot
    h_cusum     = h_factor * sigma_pilot

    cusum_S = []
    s       = 0.0
    T_cusum = None
    for i, d in enumerate(D_k):
        if i < K0:          # pilot phase — reset accumulator, no alarm
            cusum_S.append(0.0)
            continue
        s = max(0.0, s + (d - mu_pilot) - k_cusum)
        cusum_S.append(round(s, 8))
        if T_cusum is None and h_cusum > 0 and s > h_cusum:
            T_cusum = int(starts[i]) + w   # window end: earliest observable alarm

    return {
        "cusum_S":    cusum_S,
        "T_cusum":    T_cusum,
        "k_cusum":    round(k_cusum, 8),
        "h_cusum":    round(h_cusum, 8),
        "mu_pilot_D": round(mu_pilot, 8),
    }


def theil_sen_slope(y_k, starts):
    slopes = []
    n = len(y_k)
    for i in range(n):
        for j in range(i + 1, n):
            dx = starts[j] - starts[i]
            if dx > 0:
                slopes.append((y_k[j] - y_k[i]) / dx)
    return float(np.median(slopes)) if slopes else 0.0


# ── Early warning (with tau_warn_floor) ───────────────────────────────────────
def compute_early_warning(D_k, y_k, calib, starts, K0, q_warn, q_watch, c_fail,
                          tau_warn_floor=0.0, w=0):
    """
    Compute τ_warn = max(Q_{q_warn}(pilot_D), tau_warn_floor).
    τ_watch is floored proportionally: max(Q_{q_watch}(pilot_D), floor × q_watch/q_warn).

    tau_warn_floor (default 0.0 = off) prevents near-zero calibration from
    K0=4 pilot windows on a stable policy where LOO JSD ≈ 0. Set via
    instrument.json "tau_warn_floor". See tuning_journal.md T-01.
    """
    tau_warn_raw  = float(np.quantile(calib["pilot_D"], q_warn))  if calib["pilot_D"] else 0.0
    tau_watch_raw = float(np.quantile(calib["pilot_D"], q_watch)) if calib["pilot_D"] else 0.0

    tau_warn_floor_watch = tau_warn_floor * (q_watch / q_warn) if q_warn > 0 else tau_warn_floor
    tau_warn  = max(tau_warn_raw,  tau_warn_floor)
    tau_watch = max(tau_watch_raw, tau_warn_floor_watch)

    pilot_y   = y_k[:K0]
    mu_ref    = float(np.mean(pilot_y))
    sigma_ref = float(np.std(pilot_y, ddof=1)) if K0 > 1 else 1.0
    sigma_ref = max(sigma_ref, SIGMA_REL_FLOOR * abs(mu_ref) if mu_ref != 0.0 else EPS_STD)

    first_warn_start = first_eligible_warning_start(K0, W, S)
    eligible = np.array([ps >= first_warn_start for ps in starts], dtype=bool)

    A_deg   = ((y_k < mu_ref - c_fail * sigma_ref) & eligible).astype(int)
    A_drift = ((D_k > tau_warn)  & eligible).astype(int)
    A_watch = ((D_k > tau_watch) & eligible).astype(int)

    # Alarm timestamps at window END: the earliest step a live monitor could
    # raise an alarm is after observing the complete window [starts[k], starts[k]+w).
    T_deg   = next((int(starts[k]) + w for k in range(len(A_deg))   if A_deg[k]),   None)
    T_drift = next((int(starts[k]) + w for k in range(len(A_drift)) if A_drift[k]), None)

    candidates = [t for t in [T_deg, T_drift] if t is not None]
    T_warn = int(min(candidates)) if candidates else None

    return {
        "tau_warn":       tau_warn,
        "tau_watch":      tau_watch,
        "tau_warn_raw":   tau_warn_raw,
        "tau_watch_raw":  tau_watch_raw,
        "tau_warn_floor": tau_warn_floor,
        "mu_ref":         mu_ref,
        "sigma_ref":      sigma_ref,
        "A_deg":          A_deg.tolist(),
        "A_drift":        A_drift.tolist(),
        "A_watch":        A_watch.tolist(),
        "T_deg":          T_deg,
        "T_drift":        T_drift,
        "T_warn":         T_warn,
    }


def compute_soft_failure(y_k, starts, mu_ref, sigma_ref, c_fail, m_fail, w=0):
    """
    T_soft = window end of the last window in the first run of m_fail consecutive
    windows below threshold. Reported at window end (starts[k+m_fail-1]+w) —
    the earliest step a live monitor could confirm the m_fail-th consecutive alarm.
    """
    threshold = mu_ref - c_fail * sigma_ref
    n = len(y_k)
    for k in range(n - m_fail + 1):
        if all(y_k[k + i] < threshold for i in range(m_fail)):
            return int(starts[k + m_fail - 1]) + w   # window end of last triggering window
    return None


# ── Conditional Weibull RUL ───────────────────────────────────────────────────
def compute_conditional_rul(starts, w, T_warn, kappa, lambda_):
    """
    Per-window conditional Weibull RUL estimator.

    After T_warn fires, each subsequent window emits an updated RUL_p50 and
    RUL_p90 (remaining steps to predicted failure) plus T_pred_p50/T_pred_p90
    (absolute predicted failure step), conditioned on having survived to the
    current window end.

    Derivation (conditional Weibull):
        S(t) = exp(-(t/λ)^κ),  t = absolute steps from deployment start
        Conditioning on T > t_e (survived to window end):
        P(T > t_e + Δ | T > t_e) = 0.5 (p50) or 0.1 (p90):
        → Δ = λ · ((t_e/λ)^κ + ln(q))^(1/κ) − t_e
          where q = 2 for p50, q = 10 for p90

    NOTE: t_e = win_end (absolute step from deployment start), NOT steps since
    T_warn.  The Weibull was fit to T_hard measured from deployment start, so
    the conditioning point must use the same time origin.  RUL shrinks as the
    policy survives past its expected failure region.

    Parameters
    ----------
    starts  : list[int]  — window start steps, index-aligned with D_k / y_k
    w       : int        — window length (steps)
    T_warn  : int|None   — step at which the D_k alarm was raised (window end)
    kappa   : float      — Weibull shape κ  (from population survival fit)
    lambda_ : float      — Weibull scale λ  (from population survival fit)

    Returns
    -------
    dict with per-window lists (None for pre-alarm windows or if disabled):
        rul_p50_series    — RUL_p50  (steps remaining, 50th percentile)
        rul_p90_series    — RUL_p90  (steps remaining, 90th percentile)
        T_pred_p50_series — absolute predicted failure step (p50)
        T_pred_p90_series — absolute predicted failure step (p90)
    """
    n = len(starts)
    rul_p50    = [None] * n
    rul_p90    = [None] * n
    t_pred_p50 = [None] * n
    t_pred_p90 = [None] * n

    if T_warn is None or kappa <= 0.0 or lambda_ <= 0.0:
        return {
            "rul_p50_series":    rul_p50,
            "rul_p90_series":    rul_p90,
            "T_pred_p50_series": t_pred_p50,
            "T_pred_p90_series": t_pred_p90,
        }

    for k, ps in enumerate(starts):
        win_end = int(ps) + w          # absolute step from deployment start
        if win_end < T_warn:
            continue                   # pre-alarm window — leave None

        # t_e = absolute steps survived to this window end (conditioning point).
        # The Weibull was fit to T_hard measured from deployment start, so
        # conditioning on T > win_end requires t_e = win_end (not win_end - T_warn).
        t_e = float(win_end)
        base      = (t_e / lambda_) ** kappa
        delta_p50 = lambda_ * (base + math.log(2))  ** (1.0 / kappa) - t_e
        delta_p90 = lambda_ * (base + math.log(10)) ** (1.0 / kappa) - t_e

        rul_p50[k]    = int(round(max(delta_p50, 0.0)))
        rul_p90[k]    = int(round(max(delta_p90, 0.0)))
        t_pred_p50[k] = win_end + rul_p50[k]
        t_pred_p90[k] = win_end + rul_p90[k]

    return {
        "rul_p50_series":    rul_p50,
        "rul_p90_series":    rul_p90,
        "T_pred_p50_series": t_pred_p50,
        "T_pred_p90_series": t_pred_p90,
    }


# ── β_TS rolling slope alarm ─────────────────────────────────────────────────
def compute_beta_ts_alarm(y_k, starts, K0, tau_bts, beta_ts_window, w=0):
    """
    Rolling OLS slope of the reward signal y_k.  Mirrors the live-monitor view:
    at each eligible window i the slope is computed over the last beta_ts_window
    eligible (post-pilot) windows.

    T_bts = end of the first eligible window where rolling slope < tau_bts.

    tau_bts = 0.0 (default) → alarm fires when the reward trend first turns
    negative.  Set tau_bts < 0 in instrument.json for a less sensitive threshold
    (e.g. -1e-6 for ~½ Hopper-reward drop per 500 k steps).

    beta_ts_series: list of length len(starts); None for pilot/ineligible/
    insufficient-data windows, float OLS slope otherwise.  Index-aligned with
    window_starts so downstream analysis can pair slope with step number.
    """
    first_warn = first_eligible_warning_start(K0, W, S)
    n = len(y_k)
    beta_ts_series = [None] * n
    T_bts = None

    eligible_idx = [i for i, s in enumerate(starts) if int(s) >= first_warn]

    for rank, i in enumerate(eligible_idx):
        lo = max(0, rank - beta_ts_window + 1)
        win_idx = eligible_idx[lo : rank + 1]
        if len(win_idx) < 2:
            continue   # leave as None until enough windows accumulated

        win_x = np.array([float(starts[j]) for j in win_idx])
        win_y = np.array([float(y_k[j])    for j in win_idx])
        x_mean = win_x.mean()
        y_mean = win_y.mean()
        denom  = float(((win_x - x_mean) ** 2).sum())
        slope  = (float(((win_x - x_mean) * (win_y - y_mean)).sum() / denom)
                  if denom > 0 else 0.0)
        beta_ts_series[i] = round(slope, 12)

        if T_bts is None and slope < tau_bts:
            T_bts = int(starts[i]) + w   # window end — earliest observable alarm

    return T_bts, beta_ts_series


# ── VecNormalize obs helper ───────────────────────────────────────────────────
def _normalize_obs(obs: np.ndarray, vecnorm) -> np.ndarray:
    """
    Apply saved VecNormalize obs statistics to a single observation vector.
    Reads obs_rms directly from the loaded pickle — no VecEnv attachment needed.
    vecnorm.training must be False (set at load time) to keep stats frozen.

    If vecnorm is None (policy trained without VecNorm), returns obs unchanged.
    This makes the audit framework agnostic to whether VecNormalize was used.
    """
    if vecnorm is None:
        return obs
    rms = vecnorm.obs_rms
    return np.clip(
        (obs - rms.mean) / np.sqrt(rms.var + vecnorm.epsilon),
        -vecnorm.clip_obs,
        vecnorm.clip_obs,
    ).astype(np.float32)


# ── OCSVM: one-class SVM comparison channel ───────────────────────────────────
def calibrate_ocsvm(obs_arr, calib, K0, W, S, nu, kernel):
    """
    Train a one-class SVM on pilot observations (same standardisation as JSD).

    nu controls both the upper bound on the training anomaly fraction and the
    lower bound on the support vector fraction.  nu=0.05 → ≤5% of pilot steps
    are treated as outliers, giving pilot_O ≈ 0.05 per window.

    The SVM is fit on the concatenated K0-window pilot block (K0*W steps × obs_dim)
    so it sees the full pilot distribution, not just one window.  Per-window
    anomaly fractions (pilot_O) are then computed separately to calibrate tau_ocsvm
    via the same Q_{q_warn} quantile procedure used for tau_warn.

    Returns dict with 'clf' (fitted OneClassSVM) and 'pilot_O' (list of K0 floats).
    """
    from sklearn.svm import OneClassSVM

    mu, sigma   = calib["mu"], calib["sigma"]
    all_starts  = get_window_starts(len(obs_arr), W, S)
    pilot_starts = all_starts[:K0]

    # Stack pilot obs — standardised (same normalisation as JSD), not bin-clipped
    parts = []
    for ps in pilot_starts:
        win = obs_arr[ps : ps + W]
        parts.append((win - mu) / sigma)
    pilot_obs = np.vstack(parts)   # [K0*W, obs_dim]

    clf = OneClassSVM(kernel=kernel, nu=nu)
    clf.fit(pilot_obs)

    # Per-pilot-window anomaly fraction — basis for tau_ocsvm calibration
    pilot_O = []
    for ps in pilot_starts:
        std_obs = (obs_arr[ps : ps + W] - mu) / sigma
        pilot_O.append(float(np.mean(clf.predict(std_obs) == -1)))

    return {"clf": clf, "pilot_O": pilot_O}


def compute_ocsvm_channel(obs_arr, calib, calib_ocsvm, starts, K0, W, S, q_warn):
    """
    Score deployment windows with the fitted OCSVM.

    O_k        = fraction of steps in window k predicted anomalous (predict == -1).
    tau_ocsvm  = Q_{q_warn}(pilot_O)  — same quantile procedure as tau_warn.
    A_ocsvm_k  = I[O_k > tau_ocsvm]  for eligible (post-pilot) windows only.
    T_ocsvm    = window end of the first A_ocsvm==1 window.

    Single-window alarm (no m_fail consecutive check) — matches the A_drift /
    T_drift alarm design so lead-time comparisons are directly interpretable.
    """
    clf      = calib_ocsvm["clf"]
    pilot_O  = calib_ocsvm["pilot_O"]
    mu, sigma = calib["mu"], calib["sigma"]

    tau_ocsvm        = float(np.quantile(pilot_O, q_warn))
    first_warn_start = first_eligible_warning_start(K0, W, S)

    O_k, A_ocsvm = [], []
    T_ocsvm = None

    for ps in starts:
        win = obs_arr[int(ps) : int(ps) + W]
        if len(win) < W:
            break
        std_obs  = (win - mu) / sigma
        o        = float(np.mean(clf.predict(std_obs) == -1))
        eligible = int(ps) >= first_warn_start
        fires    = eligible and (o > tau_ocsvm)

        O_k.append(o)
        A_ocsvm.append(int(fires))
        if fires and T_ocsvm is None:
            T_ocsvm = int(ps) + W   # window end — earliest observable alarm

    return {
        "O_k":       O_k,
        "A_ocsvm":   A_ocsvm,
        "T_ocsvm":   T_ocsvm,
        "tau_ocsvm": round(tau_ocsvm, 6),
        "pilot_O":   pilot_O,
    }


def run_deployment(model, env_id, transition, seed, vecnorm=None, cond=None):
    """Run one reset-free deployment rollout until first hard failure or T_EVAL steps.

    Deployment ends at the FIRST terminated=True (policy failure) — no episode
    resets. TimeLimit is disabled (max_episode_steps=None) so truncated never
    fires mid-deployment; on PyFlyt environments an internal truncation (e.g.
    battery/duration limit) triggers a plain env.reset() and the deployment
    continues, since it is not a policy failure. Runs that fail before
    PILOT_END are discarded by the caller (too little pilot data to calibrate).

    Args:
        model: A loaded Stable-Baselines3 policy (PPO or SAC) exposing
            `.predict(obs, deterministic=True)`.
        env_id: Gymnasium environment ID to construct via `gym.make`.
        transition: Schedule name for this run ("baseline", "abrupt",
            "gradual", ...) — passed through to the module-level `PerturbationWrapper`.
        seed: Base RNG seed for the initial `env.reset()` and for any
            mid-deployment resets triggered by a non-failure truncation.
        vecnorm: Optional loaded VecNormalize statistics object used to
            normalize observations before they reach the policy; pass None
            if the policy was trained without observation normalization.
        cond: Condition label (e.g. "M1"), used only for logging by the caller.

    Returns:
        tuple: (obs_arr, act_arr, reward_arr, reward_forward_arr, T_hard)
            obs_arr (np.ndarray): Raw (unnormalized) observations, shape
                (T_hard or T_EVAL, obs_dim), float32.
            act_arr (np.ndarray): Actions taken at each step, same length
                as obs_arr, float32.
            reward_arr (np.ndarray): Per-step environment reward, float32.
            reward_forward_arr (np.ndarray): Per-step `info["reward_forward"]`
                component (F_k channel, T-14), float32.
            T_hard (int | None): 1-indexed step of the first hard failure, or
                None if the rollout reached T_EVAL without terminating.
    """
    drift_kwargs = {}
    if PERTURBATION_WRAPPER in ("GravityPerturbation", "GravityDrift"):
        drift_kwargs = {
            "g_nominal": float(DRIFT_PARAMS.get("g_nominal", -9.81)),
            "g_drifted": float(DRIFT_PARAMS.get("g_drifted", -14.715)),
        }
    elif PERTURBATION_WRAPPER in ("ObsNoisePerturbation", "ObsNoiseDrift"):
        drift_kwargs = {
            "noise_std_max": float(DRIFT_PARAMS.get("noise_std_max", 0.3)),
            "rng_seed":      int(DRIFT_PARAMS.get("rng_seed", 42)),
        }

    # Build env kwargs for AUDIT (reset-free persistent deployment).
    # max_episode_steps=None disables the gym TimeLimit wrapper.
    #
    # CONDITION_DURATIONS (e.g. Q1=10s, Q2=2.5s) are TRAINING parameters — they set
    # the truncation horizon the policy was trained under. They must NOT be applied
    # here: passing max_duration_seconds=10 would cause PyFlyt to fire truncated=True
    # every 400 steps, producing thousands of episode resets and violating the
    # reset-free deployment assumption.
    #
    # For PyFlyt environments (detected by the presence of "agent_hz" in ENV_KWARGS):
    # we set max_duration_seconds = T_eval / agent_hz so that PyFlyt's internal clock
    # never triggers a reset before the deployment budget is exhausted. The only
    # episode-ending event is terminated=True (crash/out-of-bounds), caught by
    # HARD_FAIL_COUNTS. See tuning_journal.md for design rationale.
    #
    # For MuJoCo/other environments: max_episode_steps=None is sufficient; those envs
    # do not use max_duration_seconds and passing it would raise a TypeError.
    _env_kw = dict(ENV_KWARGS)
    if "agent_hz" in _env_kw:                              # PyFlyt-only parameter
        _agent_hz = float(_env_kw["agent_hz"])
        _env_kw["max_duration_seconds"] = T_EVAL / _agent_hz   # full deployment = one episode
    base_env = gym.make(env_id, max_episode_steps=None, **_env_kw)

    # Optional observation-space wrapper for externally-trained policies whose
    # saved VecNormalize stats (and network input layer) expect features beyond
    # the raw env observation -- e.g. RL-Zoo's TimeFeatureWrapper, used to train
    # sb3/ppo-Walker2d-v3 (appends a 1-dim "fraction of episode remaining"
    # feature: 17-dim Walker2d obs -> 18-dim, matching the zoo checkpoint's
    # VecNormalize rms shape). Set via domain cfg "obs_wrapper".
    # NOTE: this wrapper sits *inside* PerturbationWrapper, so its output also becomes
    # the audit's own "raw_env_obs" stream -- the drift detector is therefore
    # also exposed to the synthetic time-remaining channel, not just physical
    # env state. This is part of the same declared v3-trained/v4-deployed
    # external-checkpoint gap already noted in walker2d_zoo_*.json (a
    # demonstration subject, not a Hopper-protocol replication).
    if OBS_WRAPPER == "TimeFeatureWrapper":
        from sb3_contrib.common.wrappers import TimeFeatureWrapper
        base_env = TimeFeatureWrapper(base_env)
    elif OBS_WRAPPER:
        raise ValueError(f"unknown obs_wrapper: {OBS_WRAPPER!r}")

    env = PerturbationWrapper(base_env, transition=transition,
                       t0=T0, pilot_end=PILOT_END, **drift_kwargs)

    obs, _ = env.reset(seed=seed)
    obs_list, act_list, rew_list, rf_list = [], [], [], []
    T_hard = None

    for t in range(T_EVAL):
        obs_list.append(obs.copy())
        action, _ = model.predict(_normalize_obs(obs, vecnorm), deterministic=True)
        act_list.append(action.copy())                       # collect for EDA
        obs, reward, terminated, truncated, info = env.step(action)
        rew_list.append(float(reward))
        rf_list.append(float(info.get("reward_forward", 0.0)))   # F_k channel (T-14)

        if terminated:
            T_hard = t + 1
            break   # stop-on-first-failure — reset-free deployment, no episode resets
        elif truncated:
            # Episode time-limit reached (e.g. PyFlyt max_duration_seconds).
            # Not a failure — reset for the next flight and continue deployment.
            obs, _ = env.reset(seed=seed * 10000 + t)

    env.close()
    return (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        np.array(rew_list, dtype=np.float32),
        np.array(rf_list,  dtype=np.float32),   # F_k: per-step forward reward
        T_hard,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"AUDIT DEPLOY v12.9 — domain={DOMAIN}")
print(f"  env={ENV_ID} | T_eval={T_EVAL:,} | K0={K0} | w={W:,} | s={S:,} | B={B}")
print(f"  tau_warn_floor={TAU_WARN_FLOOR:.2e} | perturbation_mechanism={PERTURBATION_WRAPPER}")
print(f"  ckpt_dir={CKPT_DIR} | results_dir={RESULTS_DIR}")
if COND_FILTER:
    print(f"  [worker mode] condition={COND_FILTER}")
if TRANSITION_FILTER:
    print(f"  [worker mode] transition={TRANSITION_FILTER}")
print("=" * 65)

all_rollouts = []

for cond in CONDITIONS:
    if COND_FILTER and cond != COND_FILTER:
        continue

    for seed in SEEDS:
        ckpt_path = os.path.join(CKPT_DIR, f"{cond}{CKPT_SUFFIX}_seed{seed}.zip")
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {cond} seed={seed} — checkpoint not found: {ckpt_path}")
            continue

        model = AlgoCls.load(ckpt_path, device="cpu")
        print(f"\n  [{cond} seed={seed}] loaded from {ckpt_path} (algo={ALGO_NAME})")

        # Auto-detect VecNormalize stats — transparent to framework
        # Try both naming conventions: _vecnormalize.pkl (SB3 default) and _vecnorm.pkl (grid2op convention)
        _vn_candidates = [
            os.path.join(CKPT_DIR, f"{cond}{CKPT_SUFFIX}_seed{seed}_vecnormalize.pkl"),
            os.path.join(CKPT_DIR, f"{cond}{CKPT_SUFFIX}_seed{seed}_vecnorm.pkl"),
        ]
        _vecnorm_path = next((p for p in _vn_candidates if os.path.exists(p)), None)
        if _vecnorm_path is not None:
            with open(_vecnorm_path, "rb") as _f:
                _vecnorm = pickle.load(_f)
            _vecnorm.training    = False   # freeze running stats during audit
            _vecnorm.norm_reward = False   # never normalize rewards at eval time
            print(f"  [vecnorm] stats loaded from {os.path.basename(_vecnorm_path)} — obs normalized during rollout")
        else:
            _vecnorm = None
            print(f"  [vecnorm] no stats found — raw obs (policy trained without VecNorm)")

        for transition in TRANSITION_SCHEDULES:
            if TRANSITION_FILTER and transition != TRANSITION_FILTER:
                continue
            prefix        = f"{cond}_seed{seed}_{transition}"
            metrics_path  = os.path.join(RESULTS_DIR, f"{prefix}_metrics_summary.json")
            rollouts_path = os.path.join(RESULTS_DIR, f"{prefix}_rollouts_summary.json")

            if os.path.exists(metrics_path) and os.path.exists(rollouts_path):
                print(f"  [skip] {cond} seed={seed} {transition} (results exist)")
                with open(rollouts_path) as f:
                    all_rollouts.append(json.load(f))
                continue

            print(f"  [run]  {cond} seed={seed} {transition} ...", flush=True)

            obs_arr, act_arr, reward_arr, rf_arr, T_hard = run_deployment(
                model, env_id=ENV_ID, transition=transition, seed=seed,
                vecnorm=_vecnorm, cond=cond,
            )

            # Discard runs that fail before the pilot phase ends — not enough
            # data to calibrate the reference distribution.
            if T_hard is not None and T_hard < PILOT_END:
                print(f"  [discard] {cond} seed={seed} {transition} — "
                      f"failed at step {T_hard} before pilot end ({PILOT_END})")
                continue

            calib = calibrate_pilot(
                obs_arr, K0, W, S, B, ALPHA, CLIP_LO_P, CLIP_HI_P
            )
            D_k, y_k, starts, D_k_dims = compute_windowed_metrics(
                obs_arr, reward_arr, calib, T_EVAL, W, S, B, ALPHA
            )

            # ── OCSVM comparison channel (opt-in) ────────────────────────────
            ocsvm_result = None
            if OCSVM_ENABLED:
                _calib_ocsvm = calibrate_ocsvm(obs_arr, calib, K0, W, S,
                                                OCSVM_NU, OCSVM_KERNEL)
                ocsvm_result = compute_ocsvm_channel(
                    obs_arr, calib, _calib_ocsvm, starts, K0, W, S, Q_WARN
                )

            beta_TS    = theil_sen_slope(y_k, starts)
            T_bts, beta_ts_series = compute_beta_ts_alarm(
                y_k, starts, K0, TAU_BTS, BETA_TS_WINDOW, w=W
            )
            aux_action = compute_windowed_action_features(act_arr, starts, W)
            ew = compute_early_warning(
                D_k, y_k, calib, starts, K0, Q_WARN, Q_WATCH, C_FAIL,
                tau_warn_floor=TAU_WARN_FLOOR, w=W
            )

            cusum_result = compute_cusum_dk(
                D_k, calib["pilot_D"], starts, K0,
                K_CUSUM_FACTOR, H_CUSUM_FACTOR, w=W
            )
            T_cusum = cusum_result["T_cusum"]

            mu_ref    = ew["mu_ref"]
            sigma_ref = ew["sigma_ref"]
            T_soft    = compute_soft_failure(
                y_k, starts, mu_ref, sigma_ref, C_FAIL, M_FAIL, w=W
            )

            # A_len: episode-length degradation channel — disabled in no-reset
            # deployment design. With a single continuous run (no episode resets),
            # there are no episode boundaries to compute mean episode length from.
            # T_len is always None; A_len is all zeros.
            mean_ep_len_k = np.full(len(starts), np.nan)
            len_result    = compute_len_failure(
                mean_ep_len_k, starts, K0, C_LEN, SIGMA_REL_FLOOR, M_FAIL
            )
            T_len = len_result["T_len"]   # always None

            # Reset-free competing risks: T_comp = T_hard.
            # In the no-reset design the first terminated=True IS the deployment
            # end event — there are no recoverable resets. T_hard is always the
            # competing risk. T_soft/T_cusum are alarm instruments, not failure
            # events; they do not participate in the survival model.
            # delta=0 (censored) only when the rollout runs to T_EVAL without fall.
            T_comp     = T_hard
            delta      = 1 if T_hard is not None else 0
            T_comp_obs = T_hard if T_hard is not None else T_EVAL

            T_warn = ew["T_warn"]
            timely = bool(
                delta == 1
                and T_warn is not None
                and T_warn < T_comp_obs
            )

            delta_delay = int(T_warn - T0) if (transition == "abrupt" and T_warn is not None) else None
            # Baseline runs (transition in {"baseline","none","no_drift"}) always have
            # delta_delay=None, since delta_delay is only defined for the abrupt schedule.

            # ── Window-by-window conditional Weibull RUL (opt-in) ────────────────
            # Enabled when domain cfg contains "rul_weibull": {"kappa": k, "lambda": l}.
            # All series are lists of None if disabled or T_warn is None.
            # RUL_p50/p90: remaining steps to predicted failure conditioned on
            # having survived to window end. T_pred_p50/p90: absolute step.
            rul_result = compute_conditional_rul(starts, W, T_warn, RUL_KAPPA, RUL_LAMBDA)

            # ── Optional: save raw trajectory for EDA (never used by confirmatory eval) ──
            if SAVE_TRAJECTORIES:
                traj_path = os.path.join(RESULTS_DIR, f"{prefix}_trajectory.npz")
                np.savez_compressed(
                    traj_path,
                    obs            = obs_arr,
                    actions        = act_arr,
                    rewards        = reward_arr,
                    reward_forward = rf_arr,     # F_k channel (T-14)
                    # ep_starts omitted: no episode resets in persistent deployment
                )

            metrics = {
                # Identity
                "env_id":     ENV_ID,
                "domain":     DOMAIN,
                "condition":  cond,
                "seed":       seed,
                "transition": transition,
                # Instrument snapshot
                "K0":              K0,
                "W":               W,
                "S":               S,
                "tau_warn_floor":  TAU_WARN_FLOOR,
                "T_eval":          T_EVAL,
                # Calibration
                "mu":              calib["mu"].tolist(),
                "sigma":           calib["sigma"].tolist(),
                "clip_lo":         calib["clip_lo"].tolist(),
                "clip_hi":         calib["clip_hi"].tolist(),
                "pilot_D":         calib["pilot_D"],
                # Thresholds
                "tau_warn":        ew["tau_warn"],
                "tau_watch":       ew["tau_watch"],
                "tau_warn_raw":    ew["tau_warn_raw"],
                "tau_watch_raw":   ew["tau_watch_raw"],
                "mu_ref":          mu_ref,
                "sigma_ref":       sigma_ref,
                # Time-series
                "window_starts":   [int(x) for x in starts],
                "D_k":             D_k.tolist(),
                "D_k_dims":        D_k_dims,   # per-dimension JSD [n_windows × obs_dim]
                "y_k":             y_k.tolist(),
                "F_k":             [float(np.mean(rf_arr[ps:ps+W]))
                                    for ps in starts],   # F_k: forward-reward channel (T-14)
                # Alarm flags
                "A_deg":           ew["A_deg"],
                "A_drift":         ew["A_drift"],
                "A_watch":         ew["A_watch"],
                "beta_TS":         beta_TS,
                "beta_ts_series":  beta_ts_series,   # rolling OLS slopes, aligned with window_starts
                "T_bts":           T_bts,
                "tau_bts":         TAU_BTS,
                "beta_ts_window":  BETA_TS_WINDOW,
                # Event times
                "T_hard":          T_hard,
                "T_soft":          T_soft,
                "T_comp":          T_comp,
                "T_comp_obs":      T_comp_obs,
                "delta":           delta,
                "T_deg":           ew["T_deg"],
                "T_drift":         ew["T_drift"],
                "T_warn":          T_warn,
                "timely_warn":     timely,
                "delta_delay":     delta_delay,
                # RUL: per-window conditional Weibull remaining useful life (opt-in)
                # All series are None-filled when rul_weibull not in domain config.
                "rul_weibull_kappa":  RUL_KAPPA  if (RUL_KAPPA > 0 and RUL_LAMBDA > 0) else None,
                "rul_weibull_lambda": RUL_LAMBDA if (RUL_KAPPA > 0 and RUL_LAMBDA > 0) else None,
                "rul_p50_series":    rul_result["rul_p50_series"],
                "rul_p90_series":    rul_result["rul_p90_series"],
                "T_pred_p50_series": rul_result["T_pred_p50_series"],
                "T_pred_p90_series": rul_result["T_pred_p90_series"],
                # A_len: episode-length degradation channel
                "mean_ep_len_k":   [round(float(v), 2) if not np.isnan(v) else None
                                    for v in mean_ep_len_k],
                "A_len":           len_result["A_len"],
                "mu_len_ref":      round(len_result["mu_len_ref"], 2)
                                   if len_result["mu_len_ref"] is not None else None,
                "sigma_len_ref":   round(len_result["sigma_len_ref"], 4)
                                   if len_result["sigma_len_ref"] is not None else None,
                "threshold_len":   round(len_result["threshold_len"], 2)
                                   if len_result["threshold_len"] is not None else None,
                "T_len":           T_len,
                # CUSUM: cumulative sum change-point detector on D_k (T-07)
                "cusum_S":         cusum_result["cusum_S"],
                "T_cusum":         T_cusum,
                "k_cusum":         cusum_result["k_cusum"],
                "h_cusum":         cusum_result["h_cusum"],
                "mu_pilot_D":      cusum_result["mu_pilot_D"],
                # Deployment length info (no episode resets in this design)
                "deployment_len":  len(obs_arr),   # T_hard if failed, T_EVAL if completed
                # ── Auxiliary action-response features (EDA only) ──────────────────
                # These are per-window summaries of the action trajectory.
                # IMPORTANT: Do NOT use these fields to tune confirmatory thresholds.
                # They are inputs for exploratory_analysis.py only.
                "aux_action_windows": aux_action,
                # ── OCSVM comparison channel (opt-in, ocsvm_enabled in domain cfg) ──
                # Cross-validates D_k alarms via per-step anomaly scoring.
                # All fields are None when OCSVM_ENABLED=False (default).
                "ocsvm_enabled": OCSVM_ENABLED,
                "O_k":           ocsvm_result["O_k"]       if ocsvm_result else None,
                "A_ocsvm":       ocsvm_result["A_ocsvm"]   if ocsvm_result else None,
                "T_ocsvm":       ocsvm_result["T_ocsvm"]   if ocsvm_result else None,
                "tau_ocsvm":     ocsvm_result["tau_ocsvm"] if ocsvm_result else None,
                "pilot_O":       ocsvm_result["pilot_O"]   if ocsvm_result else None,
            }

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            rollout = {
                "env_id":       ENV_ID,
                "domain":       DOMAIN,
                "condition":    cond,
                "seed":         seed,
                "transition":   transition,
                "T_comp":       T_comp_obs,
                "delta":        delta,
                "T_warn":       T_warn,
                "timely_warn":  timely,
                "delta_delay":  delta_delay,
                "T_hard":       T_hard,
                "T_soft":       T_soft,
                "T_len":        T_len,
                "T_cusum":      T_cusum,
                "beta_TS":      beta_TS,
                "T_bts":        T_bts,
                "tau_bts":      TAU_BTS,
                "T_ocsvm":      ocsvm_result["T_ocsvm"] if ocsvm_result else None,
                "tau_ocsvm":    ocsvm_result["tau_ocsvm"] if ocsvm_result else None,
                "mu_ref":       mu_ref,
                "sigma_ref":    sigma_ref,
                "tau_warn":     ew["tau_warn"],
            }

            with open(rollouts_path, "w") as f:
                json.dump(rollout, f, indent=2)

            all_rollouts.append(rollout)

            _T_ocsvm_str = str(ocsvm_result["T_ocsvm"]) if ocsvm_result else "-"
            print(f"  [done] {cond} seed={seed} {transition} | "
                  f"δ={delta} T_comp={T_comp} T_soft={T_soft} T_len={T_len} "
                  f"T_warn={T_warn} T_cusum={T_cusum} T_bts={T_bts} T_ocsvm={_T_ocsvm_str} "
                  f"tau_warn={ew['tau_warn']:.3e} timely={timely}")

# ── Aggregate rollouts ─────────────────────────────────────────────────────────
os.makedirs(os.path.join(RESULTS_DIR, ".."), exist_ok=True)

agg_path = os.path.join(RESULTS_DIR, "..", f"{DOMAIN}_rollouts_summary.json")
with open(agg_path, "w") as f:
    json.dump(all_rollouts, f, indent=2)

print(f"\nAggregate rollouts → {agg_path} ({len(all_rollouts)} runs)")
