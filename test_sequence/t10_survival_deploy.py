"""
t10_survival_deploy.py — T-10: Reset-Free Survival Diagnostic & Live-Monitor Demonstration
============================================================================================

TWO-PART STRUCTURE
------------------
Part 1  Retroactive survival diagnostic (no new NAS run).
        Reads existing *_metrics_summary.json files and reports, per seed, how
        many steps and windows the agent would survive after the pilot ends if
        episodes are NOT restarted.  Classifies each seed as viable or not for
        Part 2.

        Minimum viable survival threshold:
            n_surv_windows >= K0 + 1   post-pilot windows
        where n_surv_windows = floor((T_hard - PILOT_END) / S).
        Seeds with T_hard=None (ran full T_eval without termination) are always
        viable.

Part 2  Reset-free deployment with all detection channels.
        Unlike audit_deploy.py, which resets the episode on every termination,
        this script stops the rollout on the first terminated=True event.

        Three-outcome classification per run:
            Outcome 1 — episode reaches T_eval with no termination
                        → reliability report / no-failure forecast
            Outcome 2 — episode terminates AND ≥1 alarm fired before T_hard
                        → drift correctly flagged before failure (PASS)
            Outcome 3 — episode terminates AND 0 alarms fired before T_hard
                        → BROKEN FRAMEWORK (audit failure)
            Pre-pilot — episode terminates before PILOT_END
                        → cannot assess, pilot did not complete

        Detection channels included (all from locked instrument.json params):
            D_k / A_drift / A_watch  — JSD distributional drift   (T-05)
            y_k / A_deg              — reward degradation          (T-05)
            CUSUM on D_k / T_cusum   — change-point detector       (T-07)
            R_k / A_dyn              — obs step-size norm          (T-08)
            E_k / A_surplus          — OLS reward residual         (T-09)
            M_k / A_mk               — Mahalanobis [y_k,D_k] joint (Ameur 2026)

        Additionally saved per run:
            *_t10_buf1k.npz          — 1k-step monitoring buffer (reward, obs
                                       norm, step-size norm) for cross-policy
                                       calibration and fine-grained comparison.

        A_len is NOT included — episode length is not defined in a single-
        episode reset-free run.

        Worker mode: --condition M1  (runs only that condition; allows
                     parallel NAS workers per condition)

USAGE
-----
    # Full run — Part 1 (if --results-dir provided) then Part 2 (default, no --part needed)
    python t10_survival_deploy.py --domain hopper_t10 --configs configs
                                  --results-dir results/mujoco/hopper/subjectB

    # Part 2 only — NAS, full run (skip diagnostic)
    python t10_survival_deploy.py --domain hopper_t10 --configs configs

    # Part 1 only — retroactive diagnostic, any machine
    python t10_survival_deploy.py --part 1
                                  --results-dir results/mujoco/hopper/subjectB

    # Worker mode — one condition at a time (for parallel NAS workers)
    python t10_survival_deploy.py --domain hopper_t10 --configs configs
                                  --condition M1

DOMAIN CONFIG TEMPLATE  (save as configs/hopper_t10.json)
----------------------------------------------------------
{
  "env_id": "Hopper-v5",
  "results_dir": "./results/mujoco/hopper/t10_rf",
  "episodic_results_dir": "./results/mujoco/hopper/subjectB",
  "conditions": ["M1", "M2", "M3"],
  "seeds": [0, 1, 2, 3, 4],
  "T_eval": 1000000,
  "B": 21,
  "w": 50000,
  "s": 25000,
  "algo": "PPO",
  "ckpt_suffix": "",
  "perturbation_wrapper": "GravityPerturbation",
  "transition_schedules": ["baseline", "abrupt", "gradual"],
  "drift_params": {
    "g_nominal": -9.81,
    "g_drifted": -14.715
  }
}
"""

import os
import sys
import json
import argparse
import math
import glob
import csv

os.environ.setdefault("MUJOCO_GL", "osmesa")

import pickle
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
EPS_STD         = 1e-8
Q_RK            = 0.95   # τ_R quantile  (T-08)
Q_EK            = 0.95   # τ_E quantile  (T-09, retry at 0.99 if FAR > 0.10)
C_SURPLUS       = 3.0    # surplus alarm multiplier (T-09)
R2_FLOOR        = 0.30   # min R² for primary E_k path (T-09)
M_FAIL_RK       = 2      # persistence for A_dyn (T-08)
M_FAIL_EK       = 2      # persistence for A_surplus (T-09)
# M_k (Ameur CDI-style Mahalanobis on [y_k, D_k] joint KPI vector)
Q_MK            = 0.95   # τ_M quantile for LOO calibration
K_MK_SENS       = 2.0    # adaptive threshold sensitivity  τ_M = μ_M + K_MK_SENS * σ_M
M_FAIL_MK       = 2      # persistence for A_mk
# 1k monitoring buffer
BUFFER_BLOCK    = 1000   # step resolution for saved monitoring buffer
Q_BTS           = 0.99   # upper-tail quantile for tau_TS (trend slope alarm)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE UTILITY — window / PMF / JSD  (mirrors audit_deploy.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_window_starts(T_avail, w, s):
    """All window starts [0, s, 2s, ...] such that start+w <= T_avail."""
    starts, k = [], 0
    while True:
        t = k * s
        if t + w > T_avail:
            break
        starts.append(t)
        k += 1
    return starts


def first_eligible_warning_start(K0, w, s):
    pilot_end = (K0 - 1) * s + w
    return math.ceil(pilot_end / s) * s


def compute_pmf(std_obs, bin_edges_per_feat, alpha, n_bins):
    d, pmf = std_obs.shape[1], np.zeros((std_obs.shape[1], n_bins), dtype=np.float64)
    for j in range(d):
        counts, _ = np.histogram(std_obs[:, j], bins=bin_edges_per_feat[j])
        pmf[j] = (counts + alpha) / (counts.sum() + alpha * n_bins)
    return pmf


def jsd_single(p, q):
    eps = 1e-12
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log((p + eps) / (m + eps))) +
                        np.sum(q * np.log((q + eps) / (m + eps)))))


def jsd_aggregate(pmf_k, ref_pmf):
    return float(np.mean([jsd_single(ref_pmf[j], pmf_k[j])
                          for j in range(pmf_k.shape[0])]))


def calibrate_pilot(obs_arr, K0, w, s, B, alpha, clip_lo_p, clip_hi_p):
    T, d   = obs_arr.shape
    starts = get_window_starts(T, w, s)
    assert len(starts) >= K0, f"Too few windows for K0={K0}: got {len(starts)}"

    pilot_mask = np.zeros(T, dtype=bool)
    for ps in starts[:K0]:
        pilot_mask[ps:ps + w] = True
    pilot_obs = obs_arr[pilot_mask]

    mu    = pilot_obs.mean(axis=0)
    sigma = pilot_obs.std(axis=0) + EPS_STD
    std_pilot = (pilot_obs - mu) / sigma

    clip_lo = np.percentile(std_pilot, clip_lo_p, axis=0)
    clip_hi = np.percentile(std_pilot, clip_hi_p, axis=0)
    bin_edges = np.array([np.linspace(clip_lo[j], clip_hi[j], B + 1) for j in range(d)])

    pilot_pmfs = []
    for ps in starts[:K0]:
        wstd = np.clip((obs_arr[ps:ps + w] - mu) / sigma, clip_lo, clip_hi)
        pilot_pmfs.append(compute_pmf(wstd, bin_edges, alpha, B))
    pilot_pmfs = np.array(pilot_pmfs)
    ref_pmf = pilot_pmfs.mean(axis=0)

    pilot_D = []
    for k in range(K0):
        loo_ref = np.delete(pilot_pmfs, k, axis=0).mean(axis=0)
        pilot_D.append(jsd_aggregate(pilot_pmfs[k], loo_ref))

    return {"mu": mu, "sigma": sigma, "clip_lo": clip_lo, "clip_hi": clip_hi,
            "bin_edges": bin_edges, "ref_pmf": ref_pmf, "pilot_D": pilot_D}


def compute_windowed_metrics(obs_arr, reward_arr, calib, T_avail, w, s, B, alpha):
    """Return D_k (aggregate JSD), y_k (mean reward), window starts, and
    D_k_dims (per-observation-dimension JSD, shape [n_windows, obs_dim]).
    D_k[k] == D_k_dims[k].mean() by construction."""
    starts = get_window_starts(T_avail, w, s)
    mu, sigma  = calib["mu"], calib["sigma"]
    clip_lo, clip_hi = calib["clip_lo"], calib["clip_hi"]
    be, ref_pmf = calib["bin_edges"], calib["ref_pmf"]
    D_k, y_k, D_k_dims = [], [], []
    for ps in starts:
        wstd  = np.clip((obs_arr[ps:ps + w] - mu) / sigma, clip_lo, clip_hi)
        pmf_k = compute_pmf(wstd, be, alpha, B)
        dims  = [jsd_single(ref_pmf[j], pmf_k[j]) for j in range(pmf_k.shape[0])]
        D_k.append(float(np.mean(dims)))
        D_k_dims.append(dims)
        y_k.append(float(reward_arr[ps:ps + w].mean()))
    return np.array(D_k), np.array(y_k), starts, np.array(D_k_dims)


def compute_early_warning(D_k, y_k, calib, starts, K0, q_warn, q_watch, c_fail,
                          tau_warn_floor=0.0, sigma_rel_floor=0.01, w=50000, s=25000):
    tau_warn_raw  = float(np.quantile(calib["pilot_D"], q_warn))
    tau_watch_raw = float(np.quantile(calib["pilot_D"], q_watch))
    tau_warn  = max(tau_warn_raw, tau_warn_floor)
    tau_watch = max(tau_watch_raw, tau_warn_floor * (q_watch / q_warn) if q_warn > 0 else tau_warn_floor)

    pilot_y  = y_k[:K0]
    mu_ref   = float(np.mean(pilot_y))
    sigma_ref = float(np.std(pilot_y, ddof=1)) if K0 > 1 else 1.0
    sigma_ref = max(sigma_ref, sigma_rel_floor * abs(mu_ref) if mu_ref != 0 else EPS_STD)

    first_warn = first_eligible_warning_start(K0, w, s)
    eligible   = np.array([ps >= first_warn for ps in starts], dtype=bool)

    A_deg   = ((y_k < mu_ref - c_fail * sigma_ref) & eligible).astype(int)
    A_drift = ((D_k > tau_warn)  & eligible).astype(int)
    A_watch = ((D_k > tau_watch) & eligible).astype(int)

    # Report at window END (start + w): that is when detection is made in a live setting.
    T_deg   = next((int(starts[k]) + w for k in range(len(A_deg))   if A_deg[k]),   None)
    T_drift = next((int(starts[k]) + w for k in range(len(A_drift)) if A_drift[k]), None)
    T_warn  = int(min(t for t in [T_deg, T_drift] if t is not None)) \
              if any(t is not None for t in [T_deg, T_drift]) else None

    return {"tau_warn": tau_warn, "tau_watch": tau_watch,
            "tau_warn_raw": tau_warn_raw, "tau_watch_raw": tau_watch_raw,
            "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            "A_deg": A_deg.tolist(), "A_drift": A_drift.tolist(),
            "A_watch": A_watch.tolist(),
            "T_deg": T_deg, "T_drift": T_drift, "T_warn": T_warn}


def compute_soft_failure(y_k, starts, mu_ref, sigma_ref, c_fail, m_fail, w=50000):
    thr = mu_ref - c_fail * sigma_ref
    n   = len(y_k)
    for k in range(n - m_fail + 1):
        if all(y_k[k + i] < thr for i in range(m_fail)):
            # Report at end of the LAST window in the consecutive run (when persistence confirmed)
            return int(starts[k + m_fail - 1]) + w
    return None


def compute_cusum_dk(D_k, pilot_D, starts, K0, k_factor, h_factor, w=50000):
    mu_p    = float(np.mean(pilot_D)) if pilot_D else 0.0
    sigma_p = float(np.std(pilot_D, ddof=1)) if len(pilot_D) > 1 else 1e-8
    k_c, h_c = k_factor * sigma_p, h_factor * sigma_p
    cusum_S, s, T_cusum = [], 0.0, None
    for i, d in enumerate(D_k):
        if i < K0:
            cusum_S.append(0.0)
            continue
        s = max(0.0, s + (d - mu_p) - k_c)
        cusum_S.append(round(s, 8))
        if T_cusum is None and h_c > 0 and s > h_c:
            T_cusum = int(starts[i]) + w   # report at window end
    return {"cusum_S": cusum_S, "T_cusum": T_cusum,
            "k_cusum": round(k_c, 8), "h_cusum": round(h_c, 8),
            "mu_pilot_D": round(mu_p, 8)}


def theil_sen_slope(y_k, starts):
    slopes = [(y_k[j] - y_k[i]) / (starts[j] - starts[i])
              for i in range(len(y_k)) for j in range(i + 1, len(y_k))
              if starts[j] != starts[i]]
    return float(np.median(slopes)) if slopes else 0.0


def _normalize_obs(obs, vecnorm):
    if vecnorm is None:
        return obs
    rms = vecnorm.obs_rms
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + vecnorm.epsilon),
                   -vecnorm.clip_obs, vecnorm.clip_obs).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  T-08 — R_k  OBS STEP-SIZE NORM  (inlined from t08_compute_rk.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_windowed_rk(obs_arr, starts, w, obs_dims=None):
    if obs_dims is not None:
        obs_arr = obs_arr[:, obs_dims]
    R_k = []
    for ps in starts:
        diffs = np.diff(obs_arr[ps:ps + w], axis=0)
        R_k.append(float(np.linalg.norm(diffs, axis=1).mean()))
    return np.array(R_k)


def calibrate_rk(obs_arr, K0, w, s, obs_dims=None, q=Q_RK):
    pilot_starts = [i * s for i in range(K0)]
    R_full = compute_windowed_rk(obs_arr, pilot_starts, w, obs_dims)
    loo = []
    for i in range(K0):
        loo_vals = np.delete(R_full, i)
        if len(loo_vals):
            loo.append(float(np.quantile(loo_vals, q)))
    tau_R = float(np.quantile(loo, q)) if loo else float(np.max(R_full))
    return {"tau_R": tau_R, "R_pilot": R_full.tolist()}


def compute_dyn_alarm(R_k, tau_R, m_fail=M_FAIL_RK):
    raw   = (R_k > tau_R).astype(int)
    alarm = np.zeros_like(raw)
    streak = 0
    for i, a in enumerate(raw):
        streak = streak + 1 if a else 0
        if streak >= m_fail:
            alarm[i] = 1
    return alarm


# ══════════════════════════════════════════════════════════════════════════════
#  T-09 — E_k  OLS REWARD RESIDUAL  (inlined from t09_compute_ek.py)
# ══════════════════════════════════════════════════════════════════════════════

def ols_fit_pilot(obs_arr, reward_arr, K0, w, s):
    from numpy.linalg import lstsq
    pilot_obs, pilot_rew = [], []
    for i in range(K0):
        ps = i * s
        pilot_obs.append(obs_arr[ps:ps + w])
        pilot_rew.append(reward_arr[ps:ps + w])
    X = np.vstack(pilot_obs)
    y = np.concatenate(pilot_rew)
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    coef, _, _, _ = lstsq(X_aug, y, rcond=None)
    w_vec, b = coef[:-1], coef[-1]
    y_hat    = X_aug @ coef
    ss_res   = float(np.sum((y - y_hat) ** 2))
    ss_tot   = float(np.sum((y - y.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return w_vec, float(b), float(r2)


def compute_windowed_ek(obs_arr, y_k, starts, w, w_vec, b, obs_dims=None):
    if obs_dims is not None:
        obs_arr = obs_arr[:, obs_dims]
        w_vec   = w_vec[obs_dims]
    E_k, mu_obs = [], []
    for i, ps in enumerate(starts):
        mu_o = obs_arr[ps:ps + w].mean(axis=0)
        mu_obs.append(mu_o)
        E_k.append(float(y_k[i] - (w_vec @ mu_o + b)))
    return np.array(E_k), mu_obs


def calibrate_ek(obs_arr, reward_arr, K0, w, s, w_vec, b, obs_dims=None, q=Q_EK):
    pilot_starts = [i * s for i in range(K0)]
    # Per-step residuals on pilot
    all_res = []
    for ps in pilot_starts:
        o = obs_arr[ps:ps + w] if obs_dims is None else obs_arr[ps:ps + w][:, obs_dims]
        wv = w_vec if obs_dims is None else w_vec[obs_dims]
        r  = reward_arr[ps:ps + w]
        y_hat = o @ wv + b
        all_res.append(np.mean(r) - np.mean(y_hat))    # per-window mean residual
    all_res = np.array(all_res)
    loo_tau = []
    for i in range(K0):
        loo = np.delete(all_res, i)
        if len(loo):
            loo_tau.append(float(np.quantile(np.abs(loo), q)))
    tau_E    = float(np.quantile(loo_tau, q)) if loo_tau else float(np.max(np.abs(all_res)))
    sigma_E  = float(np.std(all_res, ddof=1)) if len(all_res) > 1 else float(np.abs(all_res).max())
    return {"tau_E": tau_E, "sigma_E": sigma_E, "E_pilot": all_res.tolist()}


def compute_surplus_alarm(E_k, tau_E, sigma_E, c_surplus=C_SURPLUS,
                          r2=None, r2_floor=R2_FLOOR, A_drift=None,
                          m_fail=M_FAIL_EK):
    """Primary path if R²>=r2_floor: E_k > tau_E for m_fail windows.
       Fallback: A_surplus AND (A_drift) when R²<r2_floor."""
    path = "primary" if (r2 is None or r2 >= r2_floor) else "fallback"
    raw  = (E_k > tau_E).astype(int)
    if path == "fallback" and A_drift is not None:
        A_drift_arr = np.array(A_drift)
        raw = (raw & A_drift_arr).astype(int)
    alarm, streak = np.zeros(len(raw), dtype=int), 0
    for i, a in enumerate(raw):
        streak = streak + 1 if a else 0
        if streak >= m_fail:
            alarm[i] = 1
    return alarm, path


# ══════════════════════════════════════════════════════════════════════════════
#  T-09b — E_k_rf  RANDOM-FOREST REWARD RESIDUAL  (supervised-learning variant)
#
#  Parallel channel to the OLS E_k.  Replaces the linear pilot predictor with a
#  RandomForestRegressor to capture non-linear obs→reward structure.
#  Identical calibration and alarm logic; results stored as E_k_rf / A_surplus_rf
#  so both channels can be compared post-hoc.
#
#  Key difference vs OLS: RF cannot exploit linearity of expectation, so the
#  per-window residual is  mean(r_t) − mean(rf.predict(o_t))  evaluated over
#  all steps in the window, not just the window-mean observation.
# ══════════════════════════════════════════════════════════════════════════════

def rf_fit_pilot(obs_arr, reward_arr, K0, w, s,
                 n_estimators=100, random_state=0):
    """Fit a RandomForestRegressor on pilot per-step obs→reward data.

    Returns (rf_model, r2_train) where r2_train is in-sample R² on pilot.
    Note: in-sample R² for RF will be inflated vs OLS; use for reference only.
    """
    from sklearn.ensemble import RandomForestRegressor
    pilot_obs, pilot_rew = [], []
    for i in range(K0):
        ps = i * s
        pilot_obs.append(obs_arr[ps:ps + w])
        pilot_rew.append(reward_arr[ps:ps + w])
    X = np.vstack(pilot_obs)
    y = np.concatenate(pilot_rew)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X, y)
    y_hat  = rf.predict(X)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rf, float(r2)


def compute_windowed_ek_rf(obs_arr, y_k, starts, w, rf_model):
    """Per-window E_k residuals using RF: E_k_rf[i] = mean(r_t) - mean(rf(o_t)).

    Unlike OLS, RF cannot be evaluated on the window-mean observation, so we
    predict all per-step observations and take their mean.
    """
    E_k_rf = []
    for i, ps in enumerate(starts):
        obs_win   = obs_arr[ps:ps + w]
        pred_mean = float(rf_model.predict(obs_win).mean())
        E_k_rf.append(float(y_k[i] - pred_mean))
    return np.array(E_k_rf)


def calibrate_ek_rf(obs_arr, reward_arr, K0, w, s, rf_model, q=Q_EK):
    """LOO calibration for RF E_k channel — mirrors calibrate_ek for OLS."""
    pilot_starts = [i * s for i in range(K0)]
    all_res = []
    for ps in pilot_starts:
        obs_win    = obs_arr[ps:ps + w]
        rew_win    = reward_arr[ps:ps + w]
        pred_mean  = float(rf_model.predict(obs_win).mean())
        all_res.append(float(np.mean(rew_win) - pred_mean))
    all_res = np.array(all_res)
    loo_tau = []
    for i in range(K0):
        loo = np.delete(all_res, i)
        if len(loo):
            loo_tau.append(float(np.quantile(np.abs(loo), q)))
    tau_E_rf  = (float(np.quantile(loo_tau, q)) if loo_tau
                 else float(np.max(np.abs(all_res))))
    sigma_E_rf = (float(np.std(all_res, ddof=1)) if len(all_res) > 1
                  else float(np.abs(all_res).max()))
    return {"tau_E": tau_E_rf, "sigma_E": sigma_E_rf, "E_pilot": all_res.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
#  Ameur-CDI M_k — MAHALANOBIS KPI COMPOSITE  (from Ameur et al. 2026)
#
#  KPI vector per window k:  v_k = [ỹ_k, D̃_k]
#    ỹ_k  = (y_k - μ_y) / σ_y    (normalised reward)
#    D̃_k  = (D_k - μ_D) / σ_D    (normalised JSD drift)
#
#  M_k = ||v_k||₂  in normalised space  (= diagonal Mahalanobis)
#
#  Captures joint drift: a simultaneous moderate y_k drop AND moderate D_k
#  rise fires M_k even when neither channel individually exceeds its own
#  threshold. Addresses the Ameur CDI motivation directly.
#
#  Calibration: LOO Q_{Q_MK} over pilot M values → τ_M.
#  Alarm A_mk fires when M_k > τ_M for M_FAIL_MK consecutive windows.
#  NOTE: D_U (epistemic uncertainty) from Ameur requires evidential model
#  heads (Dirichlet/NIG). NOT included — would require model retraining.
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_mk(y_k, D_k, K0, q=Q_MK, k_sens=K_MK_SENS):
    """
    Calibrate M_k composite channel from pilot windows [0 .. K0-1].

    Returns dict with keys: mu_y, sigma_y, mu_D, sigma_D, tau_M, M_pilot
    """
    y_pilot = np.array(y_k[:K0])
    D_pilot = np.array(D_k[:K0])

    mu_y, sigma_y = float(y_pilot.mean()), float(y_pilot.std(ddof=1)) + EPS_STD
    mu_D, sigma_D = float(D_pilot.mean()), float(D_pilot.std(ddof=1)) + EPS_STD

    # Pilot M values
    M_pilot = np.sqrt(((y_pilot - mu_y) / sigma_y) ** 2 +
                      ((D_pilot - mu_D) / sigma_D) ** 2)

    # LOO calibration: for each pilot window i, use Q_q of the other K0-1
    loo_thresholds = []
    for i in range(K0):
        loo = np.delete(M_pilot, i)
        if len(loo):
            loo_thresholds.append(float(np.quantile(loo, q)))
    tau_M_loo = float(np.quantile(loo_thresholds, q)) if loo_thresholds else float(M_pilot.max())

    # Adaptive threshold (Ameur eq.14/15 style): μ + k·σ over pilot
    tau_M_adapt = float(M_pilot.mean() + k_sens * M_pilot.std(ddof=1))

    # Use whichever is larger — conservative
    tau_M = max(tau_M_loo, tau_M_adapt)

    return {
        "mu_y": mu_y, "sigma_y": sigma_y,
        "mu_D": mu_D, "sigma_D": sigma_D,
        "tau_M": tau_M,
        "M_pilot": M_pilot.tolist(),
    }


def compute_mk(y_k, D_k, calib_mk):
    """Compute M_k for all windows given calibration from pilot."""
    mu_y, sigma_y = calib_mk["mu_y"], calib_mk["sigma_y"]
    mu_D, sigma_D = calib_mk["mu_D"], calib_mk["sigma_D"]
    y_arr = np.array(y_k)
    D_arr = np.array(D_k)
    return np.sqrt(((y_arr - mu_y) / sigma_y) ** 2 +
                   ((D_arr - mu_D) / sigma_D) ** 2)


def compute_mk_alarm(M_k, tau_M, m_fail=M_FAIL_MK):
    """A_mk: fires when M_k > τ_M for m_fail consecutive windows."""
    raw   = (M_k > tau_M).astype(int)
    alarm = np.zeros_like(raw)
    streak = 0
    for i, a in enumerate(raw):
        streak = streak + 1 if a else 0
        if streak >= m_fail:
            alarm[i] = 1
    return alarm




# ══════════════════════════════════════════════════════════════════════════════
#  T-15 — β_TS ROLLING TREND SLOPE
#
#  Rolling OLS slope of y_k[0..k] computed at each window close.
#  Converts the run-level Theil-Sen scalar into a per-window signal that
#  can raise a real-time alarm when the cumulative reward trend turns negative
#  (positive slope in the index domain indicates gradual drift in this env).
#
#  tau_TS = LOO Q_{Q_BTS}(pilot rolling slopes) — upper-tail threshold.
#  Alarm fires when beta_TS_k > tau_TS for M_FAIL consecutive post-pilot windows.
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_bts(y_k_arr):
    """Per-window rolling OLS slope of y_k[0..k] over window indices.

    Returns array of length len(y_k_arr).
    Index 0 is always 0.0 (single point — undefined slope).
    """
    bts = np.zeros(len(y_k_arr))
    for k in range(1, len(y_k_arr)):
        x = np.arange(k + 1, dtype=float)
        bts[k] = float(np.polyfit(x, y_k_arr[:k + 1], 1)[0])
    return bts


def calibrate_bts(y_k_arr, K0, q=Q_BTS):
    """LOO upper-tail calibration of tau_TS from pilot rolling OLS slopes.

    tau_TS = LOO Q_{q}(|pilot_bts|) — upper-tail threshold for alarm.
    """
    pilot_bts = compute_rolling_bts(np.array(y_k_arr[:K0]))
    loo_tau = []
    for i in range(K0):
        loo = [abs(pilot_bts[j]) for j in range(K0) if j != i]
        if loo:
            loo_tau.append(float(np.quantile(loo, q)))
    return float(np.quantile(loo_tau, q)) if loo_tau else float(np.max(np.abs(pilot_bts)) + 1e-12)


def compute_bts_alarm(bts_k, tau_TS, m_fail=2):
    """A_bts: fires when rolling OLS slope > tau_TS for m_fail consecutive windows."""
    raw   = (bts_k > tau_TS).astype(int)
    alarm = np.zeros_like(raw)
    streak = 0
    for i, a in enumerate(raw):
        streak = streak + 1 if a else 0
        if streak >= m_fail:
            alarm[i] = 1
    return alarm


# ══════════════════════════════════════════════════════════════════════════════
#  T-16: Weibull RUL annotation
#
#  Fit from T-16 gradual n=15 MLE: k=7.91, λ=214,830 steps.
#  Conditional Weibull quantile of remaining useful life given T_hard > T_warn.
#
#  Formula:
#    P(RUL > rul | T_hard > T_warn) = S(T_warn + rul) / S(T_warn)
#    → rul_q = λ · ((T_warn/λ)^k − ln(1−q))^(1/k) − T_warn
#
#  Returns integer step count.
# ══════════════════════════════════════════════════════════════════════════════
_WEIBULL_K   = 7.91
_WEIBULL_LAM = 214_830.0


def weibull_conditional_rul(t_warn, q, k=_WEIBULL_K, lam=_WEIBULL_LAM):
    """q-th quantile of remaining steps conditional on surviving to t_warn."""
    import math
    base = (t_warn / lam) ** k - math.log(1.0 - q)
    if base <= 0:
        return 0
    return max(0, int(round((base ** (1.0 / k)) * lam - t_warn)))


# ══════════════════════════════════════════════════════════════════════════════
#  1k MONITORING BUFFER
#
#  Saves a lightweight .npz alongside each metrics JSON with 1k-step
#  resolution traces for:
#    - reward_mean   : mean reward per 1k block
#    - obs_norm_mean : L2 norm of per-block mean observation
#    - rk_1k         : mean step-size norm (||Δo||₂) within each 1k block
#    - step          : start step index of each block
#
#  Used for cross-policy calibration and fine-grained comparison.
# ══════════════════════════════════════════════════════════════════════════════

def save_1k_buffer(obs_arr, reward_arr, out_path, block_size=BUFFER_BLOCK):
    """
    Compute per-block statistics and save as compressed .npz.

    Parameters
    ----------
    obs_arr     : (T, d) float32 observation array
    reward_arr  : (T,)   float32 reward array
    out_path    : file path for .npz (without extension — np.savez adds .npz)
    block_size  : steps per block (default 1000)
    """
    T = len(reward_arr)
    n_blocks = T // block_size
    if n_blocks == 0:
        return   # run too short to bin

    steps, rew_mean, obs_norm_mean, rk_buf = [], [], [], []
    for b in range(n_blocks):
        s0 = b * block_size
        s1 = s0 + block_size
        steps.append(s0)
        rew_mean.append(float(reward_arr[s0:s1].mean()))
        obs_norm_mean.append(float(np.linalg.norm(obs_arr[s0:s1].mean(axis=0))))
        diffs = np.diff(obs_arr[s0:s1], axis=0)
        rk_buf.append(float(np.linalg.norm(diffs, axis=1).mean())
                      if len(diffs) > 0 else 0.0)

    np.savez_compressed(
        out_path,
        step          = np.array(steps,         dtype=np.int32),
        reward_mean   = np.array(rew_mean,       dtype=np.float32),
        obs_norm_mean = np.array(obs_norm_mean,  dtype=np.float32),
        rk_1k         = np.array(rk_buf,         dtype=np.float32),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1  —  RETROACTIVE SURVIVAL DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

def part1_survival_diagnostic(results_dir: str):
    """
    Read all *_metrics_summary.json in results_dir.
    For each run, compute:
        PILOT_END    = (K0-1)*S + W
        surv_steps   = T_hard - PILOT_END  (None → agent ran full T_eval, viable)
        n_surv_win   = floor(surv_steps / S)  if surv_steps > 0 else 0
        viable       = T_hard is None  OR  n_surv_win >= K0 + 1
    """
    pattern = os.path.join(results_dir, "*_metrics_summary.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no *_metrics_summary.json found in {results_dir}")
        sys.exit(1)

    rows = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        K0     = d["K0"]
        starts = d["window_starts"]
        s      = int(starts[1] - starts[0])
        w      = int(d["T_eval"] - starts[-1])
        pilot_end   = (K0 - 1) * s + w
        min_surv    = (K0 + 1) * s       # minimum viable post-pilot steps
        T_hard      = d.get("T_hard")
        n_ep        = d.get("n_episodes", "?")

        if T_hard is None:
            surv_steps = None          # ran full T_eval
            n_surv_win = "∞"
            viable     = True
        else:
            surv_steps = T_hard - pilot_end
            n_surv_win = max(0, surv_steps // s)
            viable     = surv_steps >= min_surv

        rows.append({
            "file":        os.path.basename(f),
            "condition":   d.get("condition", "?"),
            "seed":        d.get("seed", "?"),
            # "baseline" is canonical going forward; "none"/"no_drift" are
            # pre-rename aliases still present in older result JSONs.
            "transition":  {"none": "baseline", "no_drift": "baseline"}.get(
                                d.get("transition", "?"), d.get("transition", "?")),
            "T_hard":      T_hard,
            "PILOT_END":   pilot_end,
            "surv_steps":  surv_steps,
            "n_surv_win":  n_surv_win,
            "n_episodes":  n_ep,
            "viable":      viable,
            "_K0":         K0,
            "_S":          s,
            "min_surv_steps": min_surv,
        })

    # ── Print table ───────────────────────────────────────────────────────────
    hdr = (f"{'FILE':<46} {'T_hard':>10} {'PILOT_END':>10} "
           f"{'surv_steps':>12} {'n_surv_win':>10} {'viable':>7}")
    sep = "─" * len(hdr)
    print(f"\nT-10  Part 1 — Reset-Free Survival Diagnostic\n{sep}\n{hdr}\n{sep}")
    for r in rows:
        th = str(r["T_hard"]) if r["T_hard"] is not None else "None(full)"
        sv = str(r["surv_steps"]) if r["surv_steps"] is not None else "∞"
        nw = str(r["n_surv_win"])
        v  = "✓ YES" if r["viable"] else "  NO "
        print(f"{r['file']:<46} {th:>10} {r['PILOT_END']:>10} {sv:>12} {nw:>10} {v:>7}")
    print(sep)

    # ── Summary by condition ─────────────────────────────────────────────────
    print("\n── Viability summary ────────────────────────────────────────────────")
    for drift in ["baseline", "abrupt", "gradual"]:
        sub = [r for r in rows if r["transition"] == drift]
        if not sub:
            continue
        n_v = sum(r["viable"] for r in sub)
        print(f"  {drift:<10}: {n_v}/{len(sub)} seeds viable")
    total_viable = sum(r["viable"] for r in rows)
    print(f"  TOTAL     : {total_viable}/{len(rows)} runs viable  "
          f"(threshold: n_surv_win ≥ {rows[0]['min_surv_steps'] // (rows[0]['PILOT_END'] // (rows[0]['min_surv_steps'] // (rows[0]['K0']+1) if 'K0' in rows[0] else 1)):d} post-pilot windows)")

    if total_viable == 0:
        print("\n  ⚠  0 viable seeds — Part 2 cannot run with this policy/environment.")
        print("     Recommendation: test a longer-horizon environment (e.g. Grid2op)")
        print("     or a policy trained for sustained operation, not episodic resets.")
    else:
        print(f"\n  → Proceed to Part 2 with the {total_viable} viable seed(s).")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(results_dir, "t10_survival_diagnostic.csv")
    flat_rows = [{k: v for k, v in r.items() if k not in ("min_surv_steps",)}
                 for r in rows]
    if flat_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            w.writerows(flat_rows)
        print(f"\nCSV saved → {out_csv}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2  —  RESET-FREE DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_reset_free(model, env_id, transition, seed, T_eval, T0, pilot_end,
                   perturbation_wrapper_cls, drift_kwargs, vecnorm=None):
    """
    Run up to T_eval steps in reset-free mode.

    TimeLimit is disabled (max_episode_steps=None) to prevent the gymnasium
    TimeLimit wrapper from triggering truncated=True mid-episode.  The only
    stop condition is terminated=True (genuine hard failure) or reaching T_eval.

    Returns (obs_arr, act_arr, reward_arr, T_hard, T_eff)
        T_hard : step index of first termination, or None
        T_eff  : actual number of steps collected (= T_hard or T_eval)
    """
    import gymnasium as gym
    base_env = gym.make(env_id, max_episode_steps=None)
    env      = perturbation_wrapper_cls(base_env, transition=transition,
                                 t0=T0, pilot_end=pilot_end, **drift_kwargs)

    obs, _ = env.reset(seed=seed)
    obs_list, act_list, rew_list = [], [], []
    T_hard = None

    for t in range(T_eval):
        obs_list.append(obs.copy())
        action, _ = model.predict(_normalize_obs(obs, vecnorm), deterministic=True)
        act_list.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        rew_list.append(float(reward))

        if terminated:
            T_hard = t + 1   # 1-indexed step of failure
            break
        # truncated should not fire (TimeLimit disabled); if it somehow does,
        # treat as clean completion of that stretch and continue.

    env.close()
    T_eff = T_hard if T_hard is not None else len(obs_list)
    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32),
            np.array(rew_list, dtype=np.float32),
            T_hard, T_eff)


def classify_outcome(T_hard, T_eff, T_eval, T_warn, T_drift, T_deg, T_cusum,
                     pilot_end, T0=None, transition="baseline"):
    """
    Outcome classification per the live-monitor framework spec.

    T_onset is computed from transition type:
      - "abrupt"   : T_onset = T0   (transition starts at apex = T0)
      - "gradual"  : T_onset = pilot_end  (ramp starts at pilot_end, apex at T0)
      - "baseline" : T_onset = T_eval    (no transition)

    Returns (outcome_code, outcome_label)
        0  → Pre-pilot termination      (died before pilot_end — cannot assess)
       -1  → Pre-transition termination (abrupt only: died before T0 — cannot assess)
        1  → Ran to T_eval, no termination
        2  → Terminated, ≥1 alarm fired before T_hard  (PASS)
        3  → Terminated, 0 alarms fired before T_hard  (FRAMEWORK MISS)
    """
    # Compute transition onset
    if transition == "gradual":
        T_onset = pilot_end          # ramp begins right after pilot
    elif transition == "abrupt":
        T_onset = T0 if T0 is not None else T_eval
    else:
        T_onset = T_eval             # baseline: no transition window to miss

    if T_hard is None or T_eff >= T_eval:
        return 1, "OUTCOME-1: ran to T_eval — reliability report"

    if T_hard < pilot_end:
        return 0, f"PRE-PILOT TERMINATION at step {T_hard} (PILOT_END={pilot_end})"

    # Abrupt only: died during nominal phase before transition onset
    if transition == "abrupt" and T_hard < T_onset:
        return -1, (f"PRE-TRANSITION TERMINATION at step {T_hard} "
                    f"(T_onset={T_onset}) — cannot assess framework")

    # Collect all alarm times that fired before T_hard
    alarms_before = [t for t in [T_warn, T_drift, T_deg, T_cusum]
                     if t is not None and t < T_hard]
    if alarms_before:
        first = min(alarms_before)
        phase = ("during ramp" if transition == "gradual" and T_hard < T0
                 else "post-apex") if T0 is not None else ""
        return 2, (f"OUTCOME-2: terminated at {T_hard} {phase}, "
                   f"first alarm at {first} (lead={T_hard - first} steps)")
    else:
        return 3, f"OUTCOME-3: terminated at {T_hard}, NO alarm fired — FRAMEWORK MISS"


def part2_reset_free_deploy(domain_cfg, instr_cfg, cond_filter=None):
    from stable_baselines3 import PPO, SAC
    import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "core"))  # [src_clean] import shim
    from drift_wrappers import PERTURBATION_REGISTRY

    ENV_ID      = domain_cfg["env_id"]
    RESULTS_DIR = domain_cfg["results_dir"]
    EPI_DIR     = domain_cfg.get("episodic_results_dir", None)
    CONDITIONS  = domain_cfg["conditions"]
    SEEDS       = domain_cfg["seeds"]
    T_EVAL      = int(domain_cfg["T_eval"])
    B           = int(domain_cfg.get("B", 21))
    W           = int(domain_cfg.get("w", 50000))
    S           = int(domain_cfg.get("s", 25000))
    CKPT_DIR    = domain_cfg.get("ckpt_dir", RESULTS_DIR)
    CKPT_SUFFIX = domain_cfg.get("ckpt_suffix", "")
    ALGO_NAME   = domain_cfg.get("algo", "PPO")
    PERTURBATION_WRAPPER = domain_cfg.get("perturbation_wrapper", domain_cfg.get("drift_wrapper", "GravityPerturbation"))
    TRANSITION_SCHEDULES = domain_cfg.get("transition_schedules", domain_cfg.get("drift_schedules", ["baseline", "abrupt", "gradual"]))
    DRIFT_PARAMS    = domain_cfg.get("drift_params", {})

    K0              = int(instr_cfg["K0"])
    ALPHA           = float(instr_cfg["alpha"])
    CLIP_LO_P       = float(instr_cfg.get("clip_lo_p", 0.5))
    CLIP_HI_P       = float(instr_cfg.get("clip_hi_p", 99.5))
    C_FAIL          = float(instr_cfg["c_fail"])
    M_FAIL          = int(instr_cfg["m_fail"])
    Q_WARN          = float(instr_cfg["q_warn"])
    Q_WATCH         = float(instr_cfg["q_watch"])
    TAU_WARN_FLOOR  = float(instr_cfg.get("tau_warn_floor", 0.0))
    SIGMA_REL_FLOOR = float(instr_cfg.get("sigma_ref_relative_floor", 0.01))
    K_CUSUM         = float(instr_cfg.get("k_cusum_factor", 0.5))
    H_CUSUM         = float(instr_cfg.get("h_cusum_factor", 5.0))

    PILOT_END = (K0 - 1) * S + W
    T0        = int(domain_cfg.get("t0_abrupt", T_EVAL // 2))

    AlgoCls        = {"PPO": PPO, "SAC": SAC}.get(ALGO_NAME, PPO)
    PerturbationWrapperCls = PERTURBATION_REGISTRY[PERTURBATION_WRAPPER]

    drift_kwargs = {}
    if PERTURBATION_WRAPPER in ("GravityPerturbation", "GravityDrift"):
        drift_kwargs = {"g_nominal":            float(DRIFT_PARAMS.get("g_nominal",            -9.81)),
                        "g_drifted":            float(DRIFT_PARAMS.get("g_drifted",            -14.715)),
                        "abrupt_ramp_duration": int(  DRIFT_PARAMS.get("abrupt_ramp_duration",  50_000))}
    elif PERTURBATION_WRAPPER in ("ObsNoisePerturbation", "ObsNoiseDrift"):
        drift_kwargs = {"noise_std_max":        float(DRIFT_PARAMS.get("noise_std_max",         0.3)),
                        "rng_seed":             int(  DRIFT_PARAMS.get("rng_seed",              42)),
                        "abrupt_ramp_duration": int(  DRIFT_PARAMS.get("abrupt_ramp_duration",  50_000))}

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print(f"T-10 Part 2 — Reset-Free Deployment  (LIVE MONITOR MODE)")
    print(f"  env={ENV_ID} | T_eval={T_EVAL:,} | K0={K0} | W={W:,} | S={S:,}")
    print(f"  PILOT_END={PILOT_END:,} | T0={T0:,}  (drift onset for abrupt/gradual)")
    print(f"  channels: D_k A_drift A_deg CUSUM R_k E_k  [A_len: N/A in RF mode]")
    if cond_filter:
        print(f"  [worker mode] condition={cond_filter}")
    print("=" * 70)

    all_results = []

    for cond in CONDITIONS:
        if cond_filter and cond != cond_filter:
            continue

        ckpt_path = os.path.join(CKPT_DIR, f"{cond}{CKPT_SUFFIX}_seed{{}}.zip")
        # Load model once per condition if seeds share a checkpoint
        for seed in SEEDS:
            cp = ckpt_path.format(seed)
            if not os.path.exists(cp):
                print(f"  [skip] {cond} seed={seed} — checkpoint not found: {cp}")
                continue

            model = AlgoCls.load(cp, device="cpu")
            print(f"\n  [{cond} seed={seed}] checkpoint loaded")

            vn_candidates = [
                os.path.join(CKPT_DIR, f"{cond}{CKPT_SUFFIX}_seed{seed}_vecnormalize.pkl"),
                os.path.join(CKPT_DIR, f"{cond}{CKPT_SUFFIX}_seed{seed}_vecnorm.pkl"),
            ]
            vecnorm_path = next((p for p in vn_candidates if os.path.exists(p)), None)
            vecnorm = None
            if vecnorm_path:
                with open(vecnorm_path, "rb") as vf:
                    vecnorm = pickle.load(vf)
                vecnorm.training    = False
                vecnorm.norm_reward = False
                print(f"  [vecnorm] loaded from {os.path.basename(vecnorm_path)}")

            for transition in TRANSITION_SCHEDULES:
                prefix       = f"{cond}_seed{seed}_{transition}"
                metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_t10_metrics.json")

                if os.path.exists(metrics_path):
                    print(f"  [skip] {cond} seed={seed} {transition} (results exist)")
                    with open(metrics_path) as fp:
                        all_results.append(json.load(fp))
                    continue

                print(f"  [run ] {cond} seed={seed} {transition} ...", flush=True)

                obs_arr, act_arr, reward_arr, T_hard, T_eff = run_reset_free(
                    model, ENV_ID, transition, seed, T_EVAL, T0, PILOT_END,
                    PerturbationWrapperCls, drift_kwargs, vecnorm)

                # ── Pre-pilot termination check ───────────────────────────────
                if T_hard is not None and T_hard < PILOT_END:
                    outcome_code = 0
                    outcome_label = (f"PRE-PILOT TERMINATION at step {T_hard} "
                                     f"(PILOT_END={PILOT_END}) — cannot assess")
                    result = {
                        "condition": cond, "seed": seed, "transition": transition,
                        "T_hard": T_hard, "T_eff": T_eff, "T_eval": T_EVAL,
                        "PILOT_END": PILOT_END, "outcome_code": outcome_code,
                        "outcome": outcome_label,
                    }
                    with open(metrics_path, "w") as fp:
                        json.dump(result, fp, indent=2)
                    all_results.append(result)
                    print(f"  [done] {cond} seed={seed} {transition} | ⚠  {outcome_label}")
                    continue

                # ── Full metric computation ───────────────────────────────────
                # Windowed metrics use T_eff (actual episode length) not T_eval
                calib   = calibrate_pilot(obs_arr, K0, W, S, B, ALPHA,
                                          CLIP_LO_P, CLIP_HI_P)
                D_k, y_k, starts, D_k_dims = compute_windowed_metrics(
                    obs_arr, reward_arr, calib, T_eff, W, S, B, ALPHA)
                beta_TS = theil_sen_slope(y_k.tolist(), starts)
                ew      = compute_early_warning(
                    D_k, y_k, calib, starts, K0, Q_WARN, Q_WATCH, C_FAIL,
                    tau_warn_floor=TAU_WARN_FLOOR,
                    sigma_rel_floor=SIGMA_REL_FLOOR, w=W, s=S)
                T_soft  = compute_soft_failure(
                    y_k.tolist(), starts, ew["mu_ref"], ew["sigma_ref"],
                    C_FAIL, M_FAIL, w=W)
                cusum   = compute_cusum_dk(
                    D_k.tolist(), calib["pilot_D"], starts, K0, K_CUSUM, H_CUSUM, w=W)


                # ── T-08: R_k (reset-free — no obs discontinuities from resets)
                calib_rk = calibrate_rk(obs_arr, K0, W, S)
                R_k      = compute_windowed_rk(obs_arr, starts, W)
                A_dyn    = compute_dyn_alarm(R_k, calib_rk["tau_R"], M_FAIL)
                T_dyn    = next((int(starts[i]) + W for i in range(len(A_dyn))
                                 if A_dyn[i] and starts[i] >= PILOT_END), None)

                # ── T-09: E_k (reset-free — clean pilot obs→reward mapping)
                ols_w, ols_b, r2 = ols_fit_pilot(obs_arr, reward_arr, K0, W, S)
                calib_ek = calibrate_ek(obs_arr, reward_arr, K0, W, S, ols_w, ols_b)
                E_k, _   = compute_windowed_ek(obs_arr, y_k, starts, W, ols_w, ols_b)
                A_surplus, ek_path = compute_surplus_alarm(
                    E_k, calib_ek["tau_E"], calib_ek["sigma_E"],
                    C_SURPLUS, r2=r2, r2_floor=R2_FLOOR,
                    A_drift=np.array(ew["A_drift"]), m_fail=M_FAIL)
                T_surplus = next((int(starts[i]) + W for i in range(len(A_surplus))
                                  if A_surplus[i] and starts[i] >= PILOT_END), None)

                # ── T-09b: E_k_rf (RF parallel — supervised-learning predictor)
                rf_model, r2_rf  = rf_fit_pilot(
                    obs_arr, reward_arr, K0, W, S, random_state=seed)
                calib_ek_rf      = calibrate_ek_rf(
                    obs_arr, reward_arr, K0, W, S, rf_model)
                E_k_rf           = compute_windowed_ek_rf(
                    obs_arr, y_k, starts, W, rf_model)
                A_surplus_rf, ek_rf_path = compute_surplus_alarm(
                    E_k_rf, calib_ek_rf["tau_E"], calib_ek_rf["sigma_E"],
                    C_SURPLUS, r2=r2_rf, r2_floor=R2_FLOOR,
                    A_drift=np.array(ew["A_drift"]), m_fail=M_FAIL)
                T_surplus_rf = next(
                    (int(starts[i]) + W for i in range(len(A_surplus_rf))
                     if A_surplus_rf[i] and starts[i] >= PILOT_END), None)

                # ── Ameur M_k: Mahalanobis KPI composite [y_k, D_k] ──────────
                calib_mk = calibrate_mk(y_k.tolist(), D_k.tolist(), K0)
                M_k      = compute_mk(y_k.tolist(), D_k.tolist(), calib_mk)
                A_mk     = compute_mk_alarm(M_k, calib_mk["tau_M"], M_FAIL_MK)
                T_mk     = next((int(starts[i]) + W for i in range(len(A_mk))
                                 if A_mk[i] and starts[i] >= PILOT_END), None)

                # ── T-15: β_TS rolling trend slope ───────────────────────────
                bts_k    = compute_rolling_bts(y_k.tolist())
                tau_TS   = calibrate_bts(y_k.tolist(), K0)
                A_bts    = compute_bts_alarm(bts_k, tau_TS, M_FAIL)
                T_bts    = next((int(starts[i]) + W for i in range(len(A_bts))
                                 if A_bts[i] and starts[i] >= PILOT_END), None)

                # ── Global threshold overrides (injected by t12_rerun) ────────
                # Per-run calibrations above use pilot-LOO thresholds.  When
                # instr_cfg contains  tau_*_override  keys (set by t12_rerun from
                # t11 pooled calibration), those values replace the per-run
                # thresholds and all dependent alarms/times are recomputed here.
                _ov = {k: instr_cfg[k] for k in (
                    "tau_warn_override", "tau_watch_override",
                    "tau_R_override", "tau_E_override", "tau_E_rf_override",
                    "tau_M_override", "tau_cusum_override",
                    "tau_TS_override",
                ) if k in instr_cfg}
                if _ov:
                    first_warn = first_eligible_warning_start(K0, W, S)
                    eligible   = np.array([ps >= first_warn for ps in starts],
                                          dtype=bool)

                    if "tau_warn_override" in _ov:
                        tw  = float(_ov["tau_warn_override"])
                        if "tau_watch_override" in _ov:
                            tww = float(_ov["tau_watch_override"])
                        elif ew["tau_warn"] > 0:
                            tww = tw * (ew["tau_watch"] / ew["tau_warn"])
                        else:
                            tww = tw * 0.95
                        A_drift_ov = ((D_k > tw)  & eligible).astype(int)
                        A_watch_ov = ((D_k > tww) & eligible).astype(int)
                        T_drift_ov = next(
                            (int(starts[k]) + W for k in range(len(A_drift_ov))
                             if A_drift_ov[k] and starts[k] >= PILOT_END), None)
                        ew["tau_warn"]  = tw
                        ew["tau_watch"] = tww
                        ew["A_drift"]   = A_drift_ov.tolist()
                        ew["A_watch"]   = A_watch_ov.tolist()
                        ew["T_drift"]   = T_drift_ov
                        _tl = [t for t in [ew["T_deg"], T_drift_ov] if t is not None]
                        ew["T_warn"] = min(_tl) if _tl else None

                    if "tau_R_override" in _ov:
                        calib_rk["tau_R"] = float(_ov["tau_R_override"])
                        A_dyn    = compute_dyn_alarm(R_k, calib_rk["tau_R"], M_FAIL)
                        T_dyn    = next(
                            (int(starts[i]) + W for i in range(len(A_dyn))
                             if A_dyn[i] and starts[i] >= PILOT_END), None)

                    if "tau_E_override" in _ov:
                        calib_ek["tau_E"] = float(_ov["tau_E_override"])
                        A_surplus, ek_path = compute_surplus_alarm(
                            E_k, calib_ek["tau_E"], calib_ek["sigma_E"],
                            C_SURPLUS, r2=r2, r2_floor=R2_FLOOR,
                            A_drift=np.array(ew["A_drift"]), m_fail=M_FAIL)
                        T_surplus = next(
                            (int(starts[i]) + W for i in range(len(A_surplus))
                             if A_surplus[i] and starts[i] >= PILOT_END), None)

                    if "tau_E_rf_override" in _ov:
                        calib_ek_rf["tau_E"] = float(_ov["tau_E_rf_override"])
                        A_surplus_rf, ek_rf_path = compute_surplus_alarm(
                            E_k_rf, calib_ek_rf["tau_E"], calib_ek_rf["sigma_E"],
                            C_SURPLUS, r2=r2_rf, r2_floor=R2_FLOOR,
                            A_drift=np.array(ew["A_drift"]), m_fail=M_FAIL)
                        T_surplus_rf = next(
                            (int(starts[i]) + W for i in range(len(A_surplus_rf))
                             if A_surplus_rf[i] and starts[i] >= PILOT_END), None)

                    if "tau_M_override" in _ov:
                        calib_mk["tau_M"] = float(_ov["tau_M_override"])
                        A_mk  = compute_mk_alarm(M_k, calib_mk["tau_M"], M_FAIL_MK)
                        T_mk  = next(
                            (int(starts[i]) + W for i in range(len(A_mk))
                             if A_mk[i] and starts[i] >= PILOT_END), None)

                    if "tau_cusum_override" in _ov:
                        _sigma_p = float(np.std(calib["pilot_D"], ddof=1)) \
                                   if len(calib["pilot_D"]) > 1 else 1e-8
                        _h_abs   = float(_ov["tau_cusum_override"])
                        _h_fac   = _h_abs / _sigma_p if _sigma_p > 0 else H_CUSUM
                        cusum = compute_cusum_dk(
                            D_k.tolist(), calib["pilot_D"], starts,
                            K0, K_CUSUM, _h_fac, w=W)

                    if "tau_TS_override" in _ov:
                        tau_TS = float(_ov["tau_TS_override"])
                        A_bts  = compute_bts_alarm(bts_k, tau_TS, M_FAIL)
                        T_bts  = next(
                            (int(starts[i]) + W for i in range(len(A_bts))
                             if A_bts[i] and starts[i] >= PILOT_END), None)

                # ── 1k monitoring buffer ──────────────────────────────────────
                buf_path = os.path.join(RESULTS_DIR, f"{prefix}_t10_buf1k")
                save_1k_buffer(obs_arr, reward_arr, buf_path)

                # ── Outcome classification ────────────────────────────────────
                T_warn_all = ew["T_warn"]
                # Extend T_warn to include R_k, E_k, and M_k alarms
                extra_alarms = [t for t in [T_dyn, T_surplus, T_mk, T_bts] if t is not None]
                if extra_alarms:
                    if T_warn_all is None:
                        T_warn_all = min(extra_alarms)
                    else:
                        T_warn_all = min(T_warn_all, *extra_alarms)

                # ── T-16: Weibull RUL annotation at alarm onset ───────────────
                if T_warn_all is not None:
                    rul_p50 = weibull_conditional_rul(T_warn_all, 0.50)
                    rul_p90 = weibull_conditional_rul(T_warn_all, 0.90)
                    T_fail_p50 = T_warn_all + rul_p50
                    T_fail_p90 = T_warn_all + rul_p90
                else:
                    rul_p50 = rul_p90 = T_fail_p50 = T_fail_p90 = None

                outcome_code, outcome_label = classify_outcome(
                    T_hard, T_eff, T_EVAL,
                    T_warn_all, ew["T_drift"], ew["T_deg"], cusum["T_cusum"],
                    PILOT_END, T0=T0, transition=transition)

                # ── Load episodic result for comparison ───────────────────────
                episodic_T_warn = None
                if EPI_DIR:
                    epi_path = os.path.join(EPI_DIR,
                                            f"{prefix}_metrics_summary.json")
                    if os.path.exists(epi_path):
                        with open(epi_path) as fp:
                            ep = json.load(fp)
                        episodic_T_warn = ep.get("T_warn")

                # ── Build result dict ─────────────────────────────────────────
                result = {
                    # Identity
                    "condition": cond, "seed": seed, "transition": transition,
                    "env_id": ENV_ID,
                    # Run info
                    "T_eval": T_EVAL, "T_eff": T_eff, "T_hard": T_hard,
                    "PILOT_END": PILOT_END, "T0": T0,
                    # Outcome
                    "outcome_code": outcome_code, "outcome": outcome_label,
                    # Calibration
                    "K0": K0, "tau_warn": ew["tau_warn"],
                    "tau_watch": ew["tau_watch"],
                    "mu_ref": ew["mu_ref"], "sigma_ref": ew["sigma_ref"],
                    # Time-series (lists)
                    "window_starts": [int(x) for x in starts],
                    "D_k": D_k.tolist(), "y_k": y_k.tolist(),
                    # T-19: per-observation-dimension JSD (shape: [n_windows, obs_dim])
                    # D_k_dims[k].mean() == D_k[k] by construction
                    "D_k_dims": D_k_dims.tolist(),
                    # Alarms
                    "A_deg":   ew["A_deg"],
                    "A_drift": ew["A_drift"],
                    "A_watch": ew["A_watch"],
                    "beta_TS": beta_TS,
                    # Event times — primary channels
                    "T_warn":    T_warn_all,
                    "T_drift":   ew["T_drift"],
                    "T_deg":     ew["T_deg"],
                    "T_soft":    T_soft,
                    "T_cusum":   cusum["T_cusum"],
                    "cusum_S":   cusum["cusum_S"],
                    # T-08 R_k channel
                    "tau_R":     calib_rk["tau_R"],
                    "R_pilot":   calib_rk["R_pilot"],
                    "R_k":       R_k.tolist(),
                    "A_dyn":     A_dyn.tolist(),
                    "T_dyn":     T_dyn,
                    # T-09 E_k OLS channel
                    "ols_r2":    round(r2, 4),
                    "ols_path":  ek_path,
                    "tau_E":     calib_ek["tau_E"],
                    "sigma_E":   calib_ek["sigma_E"],
                    "E_k":       E_k.tolist(),
                    "A_surplus": A_surplus.tolist(),
                    "T_surplus": T_surplus,
                    # T-09b E_k_rf RF channel (supervised-learning comparison)
                    "rf_r2":        round(r2_rf, 4),
                    "rf_path":      ek_rf_path,
                    "tau_E_rf":     calib_ek_rf["tau_E"],
                    "sigma_E_rf":   calib_ek_rf["sigma_E"],
                    "E_k_rf":       E_k_rf.tolist(),
                    "A_surplus_rf": A_surplus_rf.tolist(),
                    "T_surplus_rf": T_surplus_rf,
                    # Ameur M_k composite channel
                    "tau_M":     calib_mk["tau_M"],
                    "mu_y_mk":   calib_mk["mu_y"],
                    "sigma_y_mk": calib_mk["sigma_y"],
                    "mu_D_mk":   calib_mk["mu_D"],
                    "sigma_D_mk": calib_mk["sigma_D"],
                    "M_k":       M_k.tolist(),
                    "A_mk":      A_mk.tolist(),
                    "T_mk":      T_mk,
                    # T-15 β_TS rolling trend slope (live version of Theil-Sen scalar)
                    "tau_TS":    tau_TS,
                    "beta_TS_k": bts_k.tolist(),
                    "A_bts":     A_bts.tolist(),
                    "T_bts":     T_bts,
                    # T-16 Weibull RUL annotation (conditional on T_hard > T_warn_all)
                    # k=7.91, λ=214830 (MLE fit from gradual n=15)
                    "RUL_p50":    rul_p50,
                    "RUL_p90":    rul_p90,
                    "T_fail_p50": T_fail_p50,
                    "T_fail_p90": T_fail_p90,
                    # Comparison with episodic
                    "episodic_T_warn": episodic_T_warn,
                    "warn_delta_vs_episodic": (
                        (T_warn_all - episodic_T_warn)
                        if T_warn_all is not None and episodic_T_warn is not None
                        else None),
                }

                with open(metrics_path, "w") as fp:
                    json.dump(result, fp, indent=2)
                all_results.append(result)

                print(f"  [done] {cond} seed={seed} {transition} | "
                      f"T_hard={T_hard} T_warn={T_warn_all} "
                      f"T_dyn={T_dyn} T_surplus={T_surplus}(ols) "
                      f"T_surplus_rf={T_surplus_rf}(rf) T_mk={T_mk} T_bts={T_bts} | "
                      f"{outcome_label[:45]}")

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_results:
        _print_outcome_table(all_results)
        _save_results_csv(all_results, RESULTS_DIR)


def _print_outcome_table(results):
    print("\n" + "=" * 80)
    print("T-10 Part 2 — Outcome Summary")
    print("=" * 80)
    hdr = (f"{'COND':<5} {'SEED':>4} {'DRIFT':<10} "
           f"{'T_warn':>9} {'T_dyn':>8} {'T_surplus':>10} {'T_mk':>8} {'T_bts':>8} {'OUT':>3}")
    print(hdr)
    print("─" * 100)
    for r in results:
        if r.get("outcome_code", -1) == 0:
            print(f"  {r['condition']:<4} {r['seed']:>4} {r['transition']:<10}  PRE-PILOT TERMINATION")
            continue
        print(f"  {r['condition']:<4} {r['seed']:>4} {r['transition']:<10} "
              f"{str(r.get('T_warn',    'None')):>9} "
              f"{str(r.get('T_dyn',     'None')):>8} "
              f"{str(r.get('T_surplus', 'None')):>10} "
              f"{str(r.get('T_mk',      'None')):>8} "
              f"{str(r.get('T_bts',     'None')):>8} "
              f"  {r.get('outcome_code', '?')}")
    print("─" * 100)
    counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        c = r.get("outcome_code", -99)
        counts[c] = counts.get(c, 0) + 1
    print(f"  Outcome -1 (pre-T0 fail):   {counts[-1]}  |  "
          f"Outcome 0 (pre-pilot fail): {counts[0]}  |  "
          f"Outcome 1 (full run):       {counts[1]}  |  "
          f"Outcome 2 (PASS):           {counts[2]}  |  "
          f"Outcome 3 (FRAMEWORK MISS): {counts[3]}")
    print("=" * 80)


def _save_results_csv(results, out_dir):
    flat = []
    skip = {"D_k", "y_k", "D_k_dims", "A_deg", "A_drift", "A_watch", "R_k", "A_dyn",
            "E_k", "A_surplus", "E_k_rf", "A_surplus_rf", "M_k", "A_mk",
            "beta_TS_k", "A_bts",
            "cusum_S", "R_pilot", "M_pilot", "window_starts"}
    for r in results:
        flat.append({k: v for k, v in r.items() if k not in skip and not isinstance(v, list)})
    if not flat:
        return
    out_path = os.path.join(out_dir, "t10_rf_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f"\nCSV saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", type=int, choices=[1, 2], default=None,
                        help="1 = retroactive survival diagnostic only; "
                             "2 = reset-free deployment only; "
                             "omit to run both in sequence (default)")
    # Part 1
    parser.add_argument("--results-dir", default=None,
                        help="Part 1: path to folder with *_metrics_summary.json. "
                             "Also used as episodic reference in Part 2 if "
                             "'episodic_results_dir' is absent from domain config.")
    # Part 2
    parser.add_argument("--domain",  default=None,
                        help="Part 2: domain config name (e.g. hopper_t10)")
    parser.add_argument("--configs", default="./configs",
                        help="Part 2: path to configs directory (default: ./configs)")
    parser.add_argument("--condition", default=None,
                        help="Part 2: worker mode -- run only this condition "
                             "(e.g. M1, M2, M3)")
    args = parser.parse_args()

    run_part1 = args.part in (1, None)
    run_part2 = args.part in (2, None)

    if run_part1:
        if not args.results_dir:
            if args.part == 1:
                parser.error("--results-dir is required for --part 1")
            run_part1 = False
        else:
            part1_survival_diagnostic(args.results_dir)

    if run_part2:
        if not args.domain:
            parser.error("--domain is required to run Part 2")
        domain_cfg = json.load(open(
            os.path.join(args.configs, f"{args.domain}.json")))
        instr_cfg  = json.load(open(
            os.path.join(args.configs, "instrument.json")))
        if args.results_dir and "episodic_results_dir" not in domain_cfg:
            domain_cfg["episodic_results_dir"] = args.results_dir
        part2_reset_free_deploy(domain_cfg, instr_cfg, cond_filter=args.condition)


if __name__ == "__main__":
    main()
