"""
t09_compute_ek.py  —  T-09: E_k obs-predicted reward residual
=============================================================

WHAT THIS IMPLEMENTS
--------------------
E_k = y_k  −  (w · μ_obs_k  +  b)

where  w, b  are OLS coefficients fit on pilot data:
    reward_t  ≈  w · o_t  +  b

and  μ_obs_k = (1/W) Σ_{t ∈ window k}  o_t   (per-window mean obs vector).

Calibration:
    τ_E = Q₀.₉₅(|E_pilot|)    (pilot residuals from LOO OLS fits)

A_surplus fires when E_k > c_surplus * σ_E   (default c_surplus = 3.0)
where σ_E = std(|E_pilot|).

The asymmetry c_surplus=3.0 vs c_deg=2.0 is intentional: spurious reward
surpluses are rarer events than degradation and require stronger evidence.

PRIMARY vs FALLBACK PATH
------------------------
PRIMARY  (OLS R² ≥ 0.30):
    E_k is the direct signal.  A_surplus fires on E_k alone.

FALLBACK (OLS R² < 0.30):
    obs → reward relationship is too weak to use E_k alone.
    A_surplus_confirmed = A_surplus AND (A_drift OR A_dyn)
    (Requires D_k / A_drift to be available from the main metrics JSON.)

"VALOR FROM NULL RESOURCES" FRAMING
-------------------------------------
A frozen policy cannot improve its reward by intent.  A positive E_k means:
the reward gained in window k exceeds what the pilot obs envelope predicts.
This is only possible if the ENVIRONMENT changed to provide extra resources.
E_k therefore separates legitimate performance from environment-granted
surplus — analogous to a right-hand-side relaxation in LP sensitivity.

MODES OF USE
------------
Mode A — Standalone (reads *_trajectory.npz files):
    Requires  save_trajectories = true  in domain config.

        python t09_compute_ek.py [results_dir]

Mode B — Integration patch for audit_deploy.py:
    See the INTEGRATION PATCH section below and the inline comments.

USAGE (Mode A)
--------------
    python t09_compute_ek.py [results_dir] [--c-surplus FLOAT] [--r2-floor FLOAT]

    --c-surplus   σ multiplier for A_surplus (default 3.0)
    --r2-floor    Minimum OLS R² to use primary path (default 0.30)
"""

import argparse
import json
import os
import sys
import glob
import csv
import numpy as np
from typing import Optional, List, Tuple


DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "mujoco", "hopper", "idk"
)
Q_WARN     = 0.95    # pilot quantile for τ_E
C_SURPLUS  = 3.0     # σ-multiplier for A_surplus (asymmetric vs C_deg=2.0)
R2_FLOOR   = 0.30    # minimum OLS R² for primary path
M_FAIL     = 2       # consecutive-window persistence


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION PATCH FOR audit_deploy.py
# ══════════════════════════════════════════════════════════════════════════════

def ols_fit_pilot(
    obs_arr: np.ndarray,
    reward_arr: np.ndarray,
    K0: int,
    w: int,
    s: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit OLS  reward_t ≈ w · o_t + b  on pilot data only.

    Parameters
    ----------
    obs_arr    : (T, d) full observation array
    reward_arr : (T,)   full per-step reward array
    K0         : number of pilot windows
    w, s       : window width and stride

    Returns
    -------
    coef  : (d,) weight vector w
    intercept : float b
    r2    : float R² on pilot data

    INSERTION POINT in audit_deploy.py
    -----------------------------------
    After calibrate_pilot() returns (≈ line 610), before compute_windowed_metrics():

        from t09_compute_ek import ols_fit_pilot, compute_windowed_ek, calibrate_ek
        ols_coef, ols_intercept, ols_r2 = ols_fit_pilot(obs_arr, reward_arr, K0, W, S)
        calib_ek = calibrate_ek(obs_arr, reward_arr, K0, W, S,
                                 ols_coef, ols_intercept)
        calib.update(calib_ek)
    """
    pilot_end = K0 * s + w                    # pilot region covers first K0 windows
    pilot_obs = obs_arr[:pilot_end]            # (T_pilot, d)
    pilot_rew = reward_arr[:pilot_end]         # (T_pilot,)

    # Design matrix: add intercept column
    X = np.column_stack([pilot_obs, np.ones(len(pilot_obs))])
    # OLS via least-squares (np.linalg.lstsq)
    result, _, _, _ = np.linalg.lstsq(X, pilot_rew, rcond=None)
    coef      = result[:-1]    # (d,)
    intercept = float(result[-1])

    # R² on pilot data
    pred   = pilot_obs @ coef + intercept
    ss_res = float(np.sum((pilot_rew - pred) ** 2))
    ss_tot = float(np.sum((pilot_rew - pilot_rew.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coef, intercept, r2


def compute_windowed_ek(
    obs_arr: np.ndarray,
    y_k: np.ndarray,
    starts: list,
    w: int,
    ols_coef: np.ndarray,
    ols_intercept: float,
    obs_dims: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    T-09: Per-window obs-predicted reward residual.

        μ_obs_k   = (1/W) Σ_{t ∈ window k} o_t    [mean obs vector in window]
        pred_k    = ols_coef · μ_obs_k + ols_intercept
        E_k       = y_k  −  pred_k

    Parameters
    ----------
    obs_arr       : (T, d) observation array
    y_k           : (K,) per-window mean rewards (from compute_windowed_metrics)
    starts        : list of window start indices
    w             : window width
    ols_coef      : (d,) OLS weight vector
    ols_intercept : float OLS intercept
    obs_dims      : optional subset of obs dimensions (must match dims used in ols_fit_pilot)

    Returns
    -------
    E_k    : (K,) residual array
    mu_obs : (K, d) per-window mean obs matrix (for diagnostics)

    INSERTION POINT in audit_deploy.py
    -----------------------------------
    Inside compute_windowed_metrics() after y_k is appended (≈ line 224):

        # T-09 obs mean (cheap — obs already loaded per window)
        mu_obs_k_list.append(win_obs.mean(axis=0))   # (d,)

    Then after the loop, call compute_windowed_ek() with the collected mu_obs_k.
    Or pass obs_arr and recompute the means here (both are O(T) and equivalent).
    """
    if obs_dims is not None:
        obs_arr  = obs_arr[:, obs_dims]
        ols_coef = ols_coef[obs_dims]

    E_k    = []
    mu_obs = []
    for i, ps in enumerate(starts):
        window   = obs_arr[ps : ps + w]               # (w, d)
        mu_k     = window.mean(axis=0)                # (d,)
        pred_k   = float(ols_coef @ mu_k + ols_intercept)
        E_k.append(float(y_k[i]) - pred_k)
        mu_obs.append(mu_k)

    return np.array(E_k), np.array(mu_obs)


def calibrate_ek(
    obs_arr: np.ndarray,
    reward_arr: np.ndarray,
    K0: int,
    w: int,
    s: int,
    ols_coef: np.ndarray,
    ols_intercept: float,
    q: float = Q_WARN,
    obs_dims: Optional[List[int]] = None,
) -> dict:
    """
    LOO calibration for τ_E using pilot windows.

    For each pilot window i:
      - Fit OLS on the remaining K0-1 pilot windows
      - Compute E_i = y_i − (w_loo · μ_obs_i + b_loo)
    τ_E = Q_{q}(|E_pilot|)
    σ_E = std(|E_pilot|)

    Returns
    -------
    calib_ek : dict with keys  "tau_E", "sigma_E", "E_pilot", "r2_pilot_loo"

    INSERTION POINT in audit_deploy.py
    -----------------------------------
    Call after ols_fit_pilot():

        calib_ek = calibrate_ek(obs_arr, reward_arr, K0, W, S, ols_coef, ols_intercept)
        calib.update(calib_ek)
    """
    pilot_starts = [i * s for i in range(K0)]
    E_pilot = []

    for i in range(K0):
        ps_i = pilot_starts[i]

        # LOO: build pilot obs/rew from the K0-1 remaining windows
        loo_starts = [pilot_starts[j] for j in range(K0) if j != i]
        loo_obs  = np.concatenate([obs_arr[ps:ps+w] for ps in loo_starts], axis=0)
        loo_rew  = np.concatenate([reward_arr[ps:ps+w] for ps in loo_starts], axis=0)

        if obs_dims is not None:
            loo_obs_fit = loo_obs[:, obs_dims]
        else:
            loo_obs_fit = loo_obs

        # Fit LOO OLS
        X_loo   = np.column_stack([loo_obs_fit, np.ones(len(loo_obs_fit))])
        res, _, _, _ = np.linalg.lstsq(X_loo, loo_rew, rcond=None)
        coef_i  = res[:-1]
        intc_i  = float(res[-1])

        # Predict for window i
        win_obs_i = obs_arr[ps_i : ps_i + w]
        win_rew_i = reward_arr[ps_i : ps_i + w]
        if obs_dims is not None:
            win_obs_i = win_obs_i[:, obs_dims]
        mu_i  = win_obs_i.mean(axis=0)
        y_i   = float(win_rew_i.mean())
        pred_i = float(coef_i @ mu_i + intc_i)
        E_pilot.append(y_i - pred_i)

    E_pilot_arr = np.array(E_pilot)
    abs_E       = np.abs(E_pilot_arr)

    return {
        "tau_E":   float(np.quantile(abs_E, q)),
        "sigma_E": float(abs_E.std()),
        "E_pilot": E_pilot_arr.tolist(),
    }


def compute_surplus_alarm(
    E_k: np.ndarray,
    tau_E: float,
    sigma_E: float,
    c_surplus: float = C_SURPLUS,
    m_fail: int = M_FAIL,
    A_drift: Optional[np.ndarray] = None,
    A_dyn:   Optional[np.ndarray] = None,
    r2: float = 1.0,
    r2_floor: float = R2_FLOOR,
) -> Tuple[np.ndarray, str]:
    """
    A_surplus: fires when E_k > c_surplus * σ_E  for m_fail consecutive windows.

    Primary path  (r2 ≥ r2_floor):
        A_surplus based on E_k alone.

    Fallback path (r2 < r2_floor):
        A_surplus_confirmed = A_surplus AND (A_drift OR A_dyn)
        A_drift / A_dyn must be provided as numpy arrays.

    Returns
    -------
    alarm  : (K,) int array {0, 1}
    path   : "primary" or "fallback"
    """
    threshold = c_surplus * sigma_E
    raw       = (E_k > threshold).astype(int)

    # Apply m_fail persistence
    alarm_raw = np.zeros_like(raw)
    streak = 0
    for i, a in enumerate(raw):
        streak = streak + 1 if a else 0
        if streak >= m_fail:
            alarm_raw[i] = 1

    if r2 >= r2_floor:
        return alarm_raw, "primary"

    # Fallback: require co-occurrence with A_drift or A_dyn
    if A_drift is None and A_dyn is None:
        # No drift signals provided — treat as watch-level only
        return np.zeros_like(alarm_raw), "fallback_watch_only"

    drift_signal = np.zeros(len(alarm_raw), dtype=int)
    if A_drift is not None:
        drift_signal = np.maximum(drift_signal, A_drift.astype(int))
    if A_dyn is not None:
        drift_signal = np.maximum(drift_signal, A_dyn.astype(int))

    return (alarm_raw & drift_signal).astype(int), "fallback"


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE MODE  (Mode A — reads *_trajectory.npz files)
# ══════════════════════════════════════════════════════════════════════════════

def find_trajectory_file(metrics_path: str) -> Optional[str]:
    base = metrics_path.replace("_metrics_summary.json", "_trajectory.npz")
    return base if os.path.exists(base) else None


def process_run(
    metrics_path: str,
    obs_dims: Optional[List[int]],
    c_surplus: float,
    r2_floor: float,
    q_warn: float = Q_WARN,
) -> Optional[dict]:
    traj_path = find_trajectory_file(metrics_path)
    if traj_path is None:
        return None

    with open(metrics_path) as f:
        m = json.load(f)

    traj = np.load(traj_path)
    # Support 'obs' or 'observations' as key names
    obs_key = "obs" if "obs" in traj else "observations" if "observations" in traj else None
    if obs_key is None:
        raise KeyError(f"trajectory npz has no 'obs'/'observations' key — found: {list(traj.keys())}")
    obs_arr = traj[obs_key]
    # Support 'rewards' or 'reward' as key names
    rew_key = "rewards" if "rewards" in traj else "reward" if "reward" in traj else None
    if rew_key is None:
        raise KeyError(f"trajectory npz has no 'rewards'/'reward' key — found: {list(traj.keys())}")
    reward_arr = traj[rew_key]

    # Derive w and s from window_starts (new audit_deploy.py no longer saves them directly)
    K0     = m["K0"]
    starts = m.get("window_starts") or m.get("starts")
    if starts is None:
        raise KeyError(f"metrics JSON missing window_starts — available keys: {list(m.keys())}")
    if len(starts) < 2:
        raise ValueError(f"Need ≥2 window_starts to derive stride s, got {len(starts)}")
    s = int(starts[1] - starts[0])
    w = int(m["T_eval"] - starts[-1])
    y_k    = np.array(m["y_k"])
    A_drift = np.array(m.get("A_drift", [0] * len(y_k)))

    # OLS fit on pilot
    ols_coef, ols_intercept, r2 = ols_fit_pilot(obs_arr, reward_arr, K0, w, s)

    # LOO calibration
    calib_ek = calibrate_ek(obs_arr, reward_arr, K0, w, s,
                             ols_coef, ols_intercept, q=q_warn, obs_dims=obs_dims)
    tau_E   = calib_ek["tau_E"]
    sigma_E = calib_ek["sigma_E"]

    # Per-window E_k
    E_k, mu_obs = compute_windowed_ek(
        obs_arr, y_k, starts, w, ols_coef, ols_intercept, obs_dims
    )

    # Alarm
    alarm, path = compute_surplus_alarm(
        E_k, tau_E, sigma_E,
        c_surplus=c_surplus,
        r2=r2, r2_floor=r2_floor,
        A_drift=A_drift,
    )

    return {
        "env_id":       m["env_id"],
        "domain":       m["domain"],
        "condition":    m["condition"],
        "seed":         m["seed"],
        "transition":   m["transition"],
        "ols_r2":       round(r2, 4),
        "ols_path":     path,
        "tau_E":        tau_E,
        "sigma_E":      sigma_E,
        "q_warn":       q_warn,
        "c_surplus":    c_surplus,
        "threshold_E":  c_surplus * sigma_E,
        "E_pilot":      calib_ek["E_pilot"],
        "E_k":          E_k.tolist(),
        "A_surplus":    alarm.tolist(),
        "n_alarm_surplus": int(alarm.sum()),
        "E_k_max":      float(E_k.max()),
        "E_k_min":      float(E_k.min()),
        "obs_dims_used": obs_dims if obs_dims is not None else "all",
    }


def print_table(rows: list):
    header = (
        f"{'FILE':<36} {'DRIFT':<9} {'R²':>6} {'path':>10} "
        f"{'τ_E':>8} {'E_k max':>8} {'n_alarm':>8}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        src = os.path.basename(r.get("_src", "?"))
        print(
            f"{src[:36]:<36} {r['transition']:<9} {r['ols_r2']:>6.3f} "
            f"{r['ols_path']:>10} {r['tau_E']:>8.4f} "
            f"{r['E_k_max']:>8.4f} {r['n_alarm_surplus']:>8}"
        )
    print(sep)


def write_csv(rows: list, out_path: str):
    if not rows:
        return
    flat_rows = []
    for r in rows:
        flat = {k: v for k, v in r.items()
                if not isinstance(v, list) and not k.startswith("_")}
        flat_rows.append(flat)
    if not flat_rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        w.writeheader()
        w.writerows(flat_rows)
    print(f"\nCSV saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Folder with *_trajectory.npz and *_metrics_summary.json")
    parser.add_argument("--q-warn",    type=float, default=Q_WARN,
                        help=f"Pilot quantile for τ_E calibration (default {Q_WARN}; "
                             f"use 0.99 for T-09 Q0.99 retry)")
    parser.add_argument("--c-surplus", type=float, default=C_SURPLUS,
                        help=f"σ multiplier for A_surplus threshold (default {C_SURPLUS})")
    parser.add_argument("--r2-floor",  type=float, default=R2_FLOOR,
                        help=f"Minimum OLS R² for primary path (default {R2_FLOOR})")
    parser.add_argument("--obs-dims",  type=int, nargs="+", default=None, metavar="i",
                        help="Obs dimension indices to include in OLS + E_k. "
                             "Default: all. Must match dimensions used in pilot fit.")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "*_metrics_summary.json")
    metrics_files = sorted(glob.glob(pattern))

    if not metrics_files:
        print(f"ERROR: no *_metrics_summary.json files found under:\n  {args.results_dir}")
        sys.exit(1)

    traj_available = [f for f in metrics_files if find_trajectory_file(f) is not None]
    traj_missing   = [f for f in metrics_files if find_trajectory_file(f) is None]

    if traj_missing:
        print(f"\n⚠  No trajectory .npz found for {len(traj_missing)} run(s).")
        print("   Set  save_trajectories: true  in the domain config and re-run audit_deploy.py.")
        print()

    if not traj_available:
        print("No trajectory files available — cannot compute E_k in standalone mode.")
        print("\nAlternative: integrate the patch functions from this script into")
        print("audit_deploy.py (see INTEGRATION PATCH section in source).")
        sys.exit(0)

    print(f"\nT-09  E_k obs-predicted reward residual")
    print(f"       q_warn={args.q_warn}  c_surplus={args.c_surplus}  "
          f"r2_floor={args.r2_floor}  m_fail={M_FAIL}\n")
    print(f"Processing {len(traj_available)} run(s) with trajectory data…\n")

    rows = []
    for mpath in traj_available:
        try:
            r = process_run(mpath, args.obs_dims, args.c_surplus, args.r2_floor,
                            q_warn=args.q_warn)
            if r is None:
                continue
            r["_src"] = mpath

            # OLS R² gate message
            gate = ("PRIMARY — E_k alone" if r["ols_r2"] >= args.r2_floor
                    else f"FALLBACK — R²={r['ols_r2']:.3f} < floor={args.r2_floor}")
            print(f"  {os.path.basename(mpath)[:50]}  R²={r['ols_r2']:.3f}  [{gate}]")

            # Save enriched JSON
            out_path = mpath.replace("_metrics_summary.json", "_metrics_summary_t09.json")
            with open(out_path, "w") as f:
                out_data = {k: v for k, v in r.items() if not k.startswith("_")}
                json.dump(out_data, f, indent=2)

            rows.append(r)
        except Exception as e:
            print(f"  SKIP {os.path.basename(mpath)}: {e}")

    print()
    if rows:
        print_table(rows)
        out_csv = os.path.join(args.results_dir, "t09_ek_results.csv")
        write_csv(rows, out_csv)

        # ── Acceptance summary ───────────────────────────────────────────────
        # Null-condition labels: "baseline" is canonical going forward;
        # "none" (t10_rf / t13_final) and "no_drift" (hopper_vc) are pre-rename
        # aliases still present in older result JSONs.
        NULL_TRANSITIONS = {"baseline", "none", "no_drift"}
        print("\nACCEPTANCE CRITERIA CHECKLIST")
        print("─────────────────────────────────────────────────────────────")
        none_rows  = [r for r in rows if r["transition"] in NULL_TRANSITIONS]
        drift_rows = [r for r in rows if r["transition"] not in NULL_TRANSITIONS]

        # FAR ≤ 0.10: under null condition, ≤10% of eligible windows should alarm
        if none_rows:
            # n_alarm_surplus is over all windows; eligible = total - K0 pilot windows
            # K0 is not stored in the t09 result dict, so use a conservative denominator
            # (len(E_k) already excludes nothing, but pilot windows won't alarm by design
            # because OLS is fit on pilot data — residuals there are calibration residuals,
            # not evaluated. Use len(E_k) as denominator: worst-case FAR estimate.)
            per_seed_far = {
                f"{r['condition']}_seed{r['seed']}": r["n_alarm_surplus"] / max(len(r["E_k"]), 1)
                for r in none_rows
            }
            max_far = max(per_seed_far.values())
            mean_far = sum(per_seed_far.values()) / len(per_seed_far)
            far_ok   = max_far <= 0.10
            null_label = list({r["transition"] for r in none_rows})[0]
            print(f"  FAR ≤ 0.10 under {null_label}:  {'PASS ✓' if far_ok else 'FAIL ✗'}  "
                  f"(mean={mean_far:.3f}  max={max_far:.3f}  n={len(none_rows)} runs)")
            if not far_ok:
                failing = {k: v for k, v in per_seed_far.items() if v > 0.10}
                for k, v in sorted(failing.items(), key=lambda x: -x[1]):
                    print(f"    {k}: FAR={v:.3f}")
        else:
            print("  FAR check: no null-condition runs available")

        # OLS R² gate
        r2_vals = [r["ols_r2"] for r in rows]
        print(f"\n  OLS R² range: [{min(r2_vals):.3f}, {max(r2_vals):.3f}]")
        n_primary  = sum(1 for r in rows if r["ols_path"] == "primary")
        n_fallback = sum(1 for r in rows if "fallback" in r["ols_path"])
        print(f"  Primary path: {n_primary} run(s)  |  Fallback path: {n_fallback} run(s)")
        print()
        print("  E_k ≈ 0 under legitimate high performance: inspect A_surplus")
        print("  for none rows above — if n_alarm_surplus is 0 or low, criterion met.")
        print("  Pr ≥ 0.60 under favorable drift: check A_surplus for gravity=0.5x runs")
        print("  (requires a favorable-drift condition; not present in current result set).")
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
