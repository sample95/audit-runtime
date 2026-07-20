"""
patch_hopper_channels.py
========================
Post-hoc computation of missing channels for hopper audit results.

  hopper_vc  (gravity ×1.5) — add OCSVM (trajectory.npz files exist)
  hopper t10_v2 (gravity ×2.0) — add beta_TS (y_k + window_starts exist in JSON)
                                  OCSVM NOT possible here (no trajectory files)

Run on NAS (or Steam Deck via ThesisStorage mount):
    ~/msc-thesis/audit_env/bin/python3 patch_hopper_channels.py

Patches metrics_summary.json files in-place. Existing keys are NOT overwritten
if already present (safe to re-run). Prints per-file status.
"""

import json
import math
import glob
import os

import numpy as np
from sklearn.svm import OneClassSVM

# ── Config (matches audit_deploy.py / instrument.json / hopper_vc.json) ───────
W       = 50_000     # window width (steps)
S       = 25_000     # window stride
K0      = 4          # pilot windows
Q_WARN  = 0.95       # quantile for tau_ocsvm (matches q_warn in instrument.json)
NU      = 0.05       # OCSVM nu — upper bound on training anomaly fraction
KERNEL  = "rbf"      # OCSVM kernel
# Subsampling for SVM feasibility: consecutive hopper steps are highly correlated;
# every 5th step preserves the distribution while reducing pilot_obs from 200k to
# 40k points (K0=4, W=50k → 4×10k). Matches the effective resolution for OCSVM
# scoring (anomaly fraction is insensitive to sub-step correlations).
SUBSAMPLE = 5        # take every Nth step from each window (1 = no subsampling)

# beta_TS params (matches instrument.json defaults used for hopper_vc)
BETA_TS_WINDOW = 5        # rolling OLS window (in windows)
TAU_BTS        = -5e-7    # alarm threshold (negative slope = reward decay)

BASE = os.path.dirname(os.path.abspath(__file__))


# ── Helpers (copied from audit_deploy.py — must stay in sync) ─────────────────
def get_window_starts(T_eval, w, s):
    starts, k = [], 0
    while True:
        t_start = k * s
        if t_start + w > T_eval:
            break
        starts.append(t_start)
        k += 1
    return starts


def first_eligible_warning_start(K0, W, S):
    pilot_end = (K0 - 1) * S + W
    return math.ceil(pilot_end / S) * S


def calibrate_ocsvm(obs_arr, mu, sigma, K0, W, S, nu, kernel, subsample=SUBSAMPLE):
    """Fit OCSVM on pilot windows; return clf + pilot_O.

    subsample: take every Nth step from each window to keep training set tractable.
    Consecutive hopper steps are highly correlated (dt=0.005 s); every 5th step
    preserves the distribution while cutting pilot_obs from 200k to 40k points.
    pilot_O is computed on the same subsampled windows for consistency.
    """
    all_starts   = get_window_starts(len(obs_arr), W, S)
    pilot_starts = all_starts[:K0]

    parts = []
    for ps in pilot_starts:
        win = obs_arr[ps : ps + W : subsample]
        parts.append((win - mu) / sigma)
    pilot_obs = np.vstack(parts)

    clf = OneClassSVM(kernel=kernel, nu=nu)
    clf.fit(pilot_obs)

    pilot_O = []
    for ps in pilot_starts:
        std_obs = (obs_arr[ps : ps + W : subsample] - mu) / sigma
        pilot_O.append(float(np.mean(clf.predict(std_obs) == -1)))

    return clf, pilot_O


def compute_ocsvm_channel(obs_arr, mu, sigma, clf, pilot_O, starts, K0, W, S, q_warn,
                          subsample=SUBSAMPLE):
    """Score deployment windows; return T_ocsvm + tau_ocsvm."""
    tau_ocsvm        = float(np.quantile(pilot_O, q_warn))
    first_warn_start = first_eligible_warning_start(K0, W, S)

    T_ocsvm = None
    for ps in starts:
        win = obs_arr[int(ps) : int(ps) + W : subsample]
        if len(win) < W // subsample:
            break
        std_obs  = (win - mu) / sigma
        o        = float(np.mean(clf.predict(std_obs) == -1))
        eligible = int(ps) >= first_warn_start
        if eligible and o > tau_ocsvm and T_ocsvm is None:
            T_ocsvm = int(ps) + W

    return T_ocsvm, round(tau_ocsvm, 6)


def compute_beta_ts(y_k, window_starts, K0, W, S, beta_ts_window, tau_bts):
    """Rolling OLS slope on y_k; return T_bts, beta_ts_series, tau_bts, beta_ts_window."""
    n = len(y_k)
    first_warn_start = first_eligible_warning_start(K0, W, S)
    beta_ts_series = [None] * n
    T_bts = None

    for i, ps in enumerate(window_starts):
        if int(ps) < first_warn_start:
            continue
        lo   = max(0, i - beta_ts_window + 1)
        xs   = np.arange(lo, i + 1, dtype=float)
        ys   = np.array([y_k[j] for j in range(lo, i + 1)], dtype=float)
        if len(xs) < 2:
            continue
        # OLS slope
        xm, ym = xs.mean(), ys.mean()
        denom  = ((xs - xm) ** 2).sum()
        if denom == 0:
            continue
        slope = float(((xs - xm) * (ys - ym)).sum() / denom)
        beta_ts_series[i] = round(slope, 12)

        if T_bts is None and slope < tau_bts:
            T_bts = int(window_starts[i]) + W

    return T_bts, beta_ts_series


# ── Patch hopper_vc — OCSVM from trajectory.npz ───────────────────────────────
def patch_hopper_vc():
    results_dir = os.path.join(BASE, "results", "mujoco", "hopper_vc")
    metrics_files = sorted(glob.glob(os.path.join(results_dir, "*_metrics_summary.json")))
    print(f"\n=== hopper_vc OCSVM patch  ({len(metrics_files)} files) ===")

    patched = skipped = errors = 0
    for mf in metrics_files:
        stem = mf.replace("_metrics_summary.json", "")
        traj_path = stem + "_trajectory.npz"

        if not os.path.exists(traj_path):
            print(f"  [MISSING traj] {os.path.basename(mf)}")
            errors += 1
            continue

        with open(mf) as f:
            d = json.load(f)

        if "T_ocsvm" in d:
            skipped += 1
            continue   # already patched

        # Load trajectory obs
        traj    = np.load(traj_path)
        obs_arr = traj["obs"].astype(np.float64)

        mu    = np.array(d["mu"],    dtype=np.float64)
        sigma = np.array(d["sigma"], dtype=np.float64)
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)

        starts = d["window_starts"]

        try:
            clf, pilot_O = calibrate_ocsvm(obs_arr, mu, sigma, K0, W, S, NU, KERNEL)
            T_ocsvm, tau_ocsvm = compute_ocsvm_channel(
                obs_arr, mu, sigma, clf, pilot_O, starts, K0, W, S, Q_WARN
            )
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(mf)}: {e}")
            errors += 1
            continue

        d["T_ocsvm"]   = T_ocsvm
        d["tau_ocsvm"] = tau_ocsvm

        with open(mf, "w") as f:
            json.dump(d, f, indent=2)

        status = f"T_ocsvm={T_ocsvm}"
        print(f"  [patch] {os.path.basename(mf)}  {status}")
        patched += 1

    print(f"  → patched={patched}  skipped={skipped}  errors={errors}")


# ── Patch hopper t10_v2 — beta_TS from y_k ────────────────────────────────────
def patch_hopper_t10v2():
    results_dir = os.path.join(BASE, "results", "mujoco", "hopper", "t10_v2")
    metrics_files = sorted(glob.glob(os.path.join(results_dir, "*_metrics_summary.json")))
    print(f"\n=== hopper t10_v2  beta_TS patch  ({len(metrics_files)} files) ===")
    print("    NOTE: OCSVM NOT patched here — trajectory files were not saved.")
    print("          Queue a full NAS rerun with ocsvm_enabled=true in hopper.json.")

    patched = skipped = errors = 0
    for mf in metrics_files:
        with open(mf) as f:
            d = json.load(f)

        if "T_bts" in d:
            skipped += 1
            continue   # already patched

        y_k           = d.get("y_k")
        window_starts = d.get("window_starts")

        if not y_k or not window_starts:
            print(f"  [MISSING y_k/window_starts] {os.path.basename(mf)}")
            errors += 1
            continue

        try:
            T_bts, beta_ts_series = compute_beta_ts(
                y_k, window_starts, K0, W, S, BETA_TS_WINDOW, TAU_BTS
            )
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(mf)}: {e}")
            errors += 1
            continue

        d["T_bts"]           = T_bts
        d["tau_bts"]         = TAU_BTS
        d["beta_ts_series"]  = beta_ts_series
        d["beta_ts_window"]  = BETA_TS_WINDOW

        with open(mf, "w") as f:
            json.dump(d, f, indent=2)

        print(f"  [patch] {os.path.basename(mf)}  T_bts={T_bts}")
        patched += 1

    print(f"  → patched={patched}  skipped={skipped}  errors={errors}")


if __name__ == "__main__":
    patch_hopper_vc()
    patch_hopper_t10v2()
    print("\nDone. Regenerate hopper figures after verifying patch results.")
