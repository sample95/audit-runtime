"""
t08_compute_rk.py  —  T-08: R_k step-size norm (dynamic drift channel)
=======================================================================

WHAT THIS IMPLEMENTS
--------------------
R_k = (1/W) × Σ_{t ∈ window k} ‖o_{t+1} − o_t‖₂

R_k measures the mean per-step displacement in observation space within each
window.  Unlike D_k (distributional drift vs the pilot reference), R_k is:

  • obs-based, not policy-extracted
  • env-agnostic (does not assume any reward structure)
  • sensitive to *dynamic* drift (sudden increased motion, oscillation) rather
    than slow distributional shift

Calibration follows the same LOO pilot scheme as D_k:
  τ_R = Q₀.₉₅(R_pilot)   where R_pilot is computed over the K0 pilot windows.

A_dyn fires when R_k > τ_R for m_fail = 2 consecutive windows.

MODES OF USE
------------
Mode A — Standalone (reads *_trajectory.npz files):
    Requires re-running audit_deploy.py with  save_trajectories = true
    in the domain config.  Then run:

        python t08_compute_rk.py [results_dir]

    The script reads every *_trajectory.npz and the matching
    *_metrics_summary.json, computes R_k, and writes
    *_metrics_summary_t08.json with the new fields appended.

Mode B — Integration patch for audit_deploy.py:
    Copy the functions in the "INTEGRATION PATCH" section below into
    audit_deploy.py (see inline comments for exact insertion points).
    On the next deployment run, R_k / A_dyn will be computed and saved
    alongside D_k / y_k in the metrics JSON.

GRID2OP CAVEAT
--------------
Grid2op observations contain many binary dimensions (line status etc.).
‖Δo‖₂ over binary dims fires every time a line trips regardless of agent
behaviour.  Before applying R_k to Grid2op runs, consider restricting the
L2 norm to continuous observation dimensions only (e.g. load p, gen p,
rho values).  Pass  --obs-dims  to select a subset (see CLI below).

USAGE (Mode A)
--------------
    python t08_compute_rk.py [results_dir] [--alive-bonus FLOAT]
                              [--obs-dims i j k ...]

    results_dir   Folder with *_trajectory.npz + *_metrics_summary.json
                  Default: ./results/mujoco/hopper/idk
    --obs-dims    Integer indices of obs dimensions to include in the norm.
                  Default: all dimensions.

OUTPUT (Mode A)
---------------
Per-run JSON files:  *_metrics_summary_t08.json
Aggregate CSV:       t08_rk_results.csv
Console table showing τ_R, A_dyn alarm count per condition/seed/drift.
"""

import argparse
import json
import os
import sys
import glob
import csv
import numpy as np
from typing import Optional, List


DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "mujoco", "hopper", "idk"
)
Q_WARN   = 0.95   # pilot quantile for τ_R (mirrors τ_warn calibration)
M_FAIL   = 2      # consecutive-window persistence for A_dyn


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION PATCH FOR audit_deploy.py
#  ─────────────────────────────────────
#  These four functions belong inside audit_deploy.py.
#  Insertion points are noted in the docstrings.
# ══════════════════════════════════════════════════════════════════════════════

def compute_windowed_rk(
    obs_arr: np.ndarray,
    starts: list,
    w: int,
    obs_dims: Optional[List[int]] = None,
) -> np.ndarray:
    """
    T-08: Per-window mean step-size norm.

        R_k = (1/W) × Σ_{t=ps}^{ps+w-2} ‖obs[t+1] − obs[t]‖₂

    Parameters
    ----------
    obs_arr   : (T, d) observation array from run_deployment
    starts    : list of window start indices (from get_window_starts)
    w         : window width in steps
    obs_dims  : optional subset of observation dimensions to use in the norm.
                None → use all dimensions.

    Returns
    -------
    R_k : (K,) float array, one value per window

    INSERTION POINT in audit_deploy.py
    -----------------------------------
    Add this function after compute_windowed_metrics() (≈ line 227).
    Then inside the main loop (≈ line 613), after the metrics computation:

        rk_arr = compute_windowed_rk(obs_arr, starts, W, obs_dims=OBS_DIMS)

    And include rk_arr in the metrics dict (≈ line 700):

        "R_k":       rk_arr.tolist(),
        "tau_R":     calib["tau_R"],
        "A_dyn":     compute_dyn_alarm(rk_arr, calib["tau_R"], M_FAIL).tolist(),
    """
    if obs_dims is not None:
        obs_arr = obs_arr[:, obs_dims]

    R_k = []
    for ps in starts:
        window = obs_arr[ps : ps + w]             # (w, d)
        diffs  = np.diff(window, axis=0)          # (w-1, d)
        norms  = np.linalg.norm(diffs, axis=1)    # (w-1,)
        R_k.append(float(norms.mean()))
    return np.array(R_k)


def calibrate_rk(
    obs_arr: np.ndarray,
    K0: int,
    w: int,
    s: int,
    obs_dims: Optional[List[int]] = None,
    q: float = Q_WARN,
) -> dict:
    """
    LOO calibration for τ_R using the K0 pilot windows.

    Mirrors calibrate_pilot() in audit_deploy.py: excludes each pilot
    window in turn and takes Q_{q} over the remaining K0-1 values.
    The floor is 0 (R_k is always non-negative).

    Parameters
    ----------
    obs_arr  : full obs array (pilot + deployment)
    K0       : number of pilot windows
    w, s     : window width and stride
    obs_dims : optional dimension subset
    q        : quantile for τ_R (default 0.95)

    Returns
    -------
    calib_rk : dict with keys  "tau_R", "R_pilot"

    INSERTION POINT in audit_deploy.py
    -----------------------------------
    Call inside calibrate_pilot() at the end (≈ line 208), or as a
    separate call right after calibrate_pilot() returns:

        from t08_compute_rk import calibrate_rk
        calib_rk = calibrate_rk(obs_arr, K0, W, S, obs_dims=OBS_DIMS)
        calib.update(calib_rk)   # merges tau_R and R_pilot into calib dict
    """
    from math import ceil
    pilot_starts = [i * s for i in range(K0)]
    R_pilot_full = compute_windowed_rk(obs_arr, pilot_starts, w, obs_dims)

    # LOO: for each pilot window i, use the other K0-1 to set the threshold
    loo_thresholds = []
    for i in range(K0):
        leave_one_out = np.delete(R_pilot_full, i)
        if len(leave_one_out) == 0:
            continue
        loo_thresholds.append(float(np.quantile(leave_one_out, q)))

    tau_R = float(np.quantile(loo_thresholds, q)) if loo_thresholds else float(np.max(R_pilot_full))

    return {
        "tau_R":    tau_R,
        "R_pilot":  R_pilot_full.tolist(),
    }


def compute_dyn_alarm(
    R_k: np.ndarray,
    tau_R: float,
    m_fail: int = M_FAIL,
) -> np.ndarray:
    """
    A_dyn: fires when R_k > τ_R for m_fail consecutive windows.

    Same persistence logic as A_deg and A_drift in the framework.

    Returns
    -------
    A_dyn : (K,) int array  {0, 1}
    """
    raw   = (R_k > tau_R).astype(int)
    alarm = np.zeros_like(raw)
    streak = 0
    for i, a in enumerate(raw):
        streak = streak + 1 if a else 0
        if streak >= m_fail:
            alarm[i] = 1
    return alarm


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE MODE  (Mode A — reads *_trajectory.npz files)
# ══════════════════════════════════════════════════════════════════════════════

def find_trajectory_file(metrics_path: str) -> Optional[str]:
    """
    Given  .../prefix_metrics_summary.json  looks for
           .../prefix_trajectory.npz
    """
    base = metrics_path.replace("_metrics_summary.json", "_trajectory.npz")
    return base if os.path.exists(base) else None


def process_run(metrics_path: str, obs_dims: Optional[List[int]]) -> Optional[dict]:
    """
    Load metrics JSON + trajectory npz for one run and compute R_k / A_dyn.
    Returns enriched dict or None if trajectory is not available.
    """
    traj_path = find_trajectory_file(metrics_path)
    if traj_path is None:
        return None

    with open(metrics_path) as f:
        m = json.load(f)

    traj    = np.load(traj_path)
    # Support both 'obs' and 'observations' as key names
    obs_key = "obs" if "obs" in traj else "observations" if "observations" in traj else None
    if obs_key is None:
        raise KeyError(f"trajectory npz has no 'obs'/'observations' key — found: {list(traj.keys())}")
    obs_arr = traj[obs_key]       # (T, d)

    # Derive w and s from window_starts (new audit_deploy.py no longer saves them directly)
    K0     = m["K0"]
    starts = m.get("window_starts") or m.get("starts")
    if starts is None:
        raise KeyError(f"metrics JSON missing window_starts — available keys: {list(m.keys())}")
    if len(starts) < 2:
        raise ValueError(f"Need ≥2 window_starts to derive stride s, got {len(starts)}")
    s = int(starts[1] - starts[0])
    w = int(m["T_eval"] - starts[-1])

    # Calibrate τ_R from pilot windows
    calib_rk = calibrate_rk(obs_arr, K0, w, s, obs_dims)
    tau_R    = calib_rk["tau_R"]
    R_pilot  = calib_rk["R_pilot"]

    # Compute R_k for all windows
    R_k   = compute_windowed_rk(obs_arr, starts, w, obs_dims)
    A_dyn = compute_dyn_alarm(R_k, tau_R, M_FAIL)

    # Build result
    n_dyn = int(A_dyn.sum())
    result = {
        **{k: m[k] for k in ("env_id", "domain", "condition", "seed", "transition",
                               "mu_ref", "sigma_ref", "T_eval", "K0")},
        "w":               w,
        "s":               s,
        "tau_R":           tau_R,
        "R_pilot":         R_pilot,
        "R_k":             R_k.tolist(),
        "A_dyn":           A_dyn.tolist(),
        "n_alarm_dyn":     n_dyn,
        "obs_dims_used":   obs_dims if obs_dims is not None else "all",
    }
    return result


def print_table(rows: list):
    header = (
        f"{'FILE':<38} {'DRIFT':<9} {'tau_R':>8} {'R_k min':>8} "
        f"{'R_k max':>8} {'n_A_dyn':>8}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        rk_vals = r["R_k"]
        print(
            f"{os.path.basename(r.get('_src','?'))[:38]:<38} "
            f"{r['transition']:<9} {r['tau_R']:>8.5f} "
            f"{min(rk_vals):>8.5f} {max(rk_vals):>8.5f} "
            f"{r['n_alarm_dyn']:>8}"
        )
    print(sep)


def write_csv(rows: list, out_path: str):
    if not rows:
        return
    flat_rows = []
    for r in rows:
        flat = {k: v for k, v in r.items()
                if not isinstance(v, list)}
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
    parser.add_argument("--obs-dims", type=int, nargs="+", default=None,
                        metavar="i",
                        help="Observation dimension indices to include in L2 norm. "
                             "Default: all. Tip: exclude binary dims for Grid2op.")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "*_metrics_summary.json")
    metrics_files = sorted(glob.glob(pattern))

    if not metrics_files:
        print(f"ERROR: no *_metrics_summary.json files found under:\n  {args.results_dir}")
        sys.exit(1)

    # ── Check for trajectory files ────────────────────────────────────────────
    traj_available = [f for f in metrics_files if find_trajectory_file(f) is not None]
    traj_missing   = [f for f in metrics_files if find_trajectory_file(f) is None]

    if traj_missing:
        print(f"\n⚠  No trajectory .npz found for {len(traj_missing)} run(s).")
        print("   To generate them, set  save_trajectories: true  in the domain config")
        print("   and re-run audit_deploy.py.  Missing files:")
        for f in traj_missing[:5]:
            print(f"     {os.path.basename(f)}")
        if len(traj_missing) > 5:
            print(f"     ... and {len(traj_missing)-5} more")
        print()

    if not traj_available:
        print("No trajectory files available — cannot compute R_k in standalone mode.")
        print("\nAlternative: integrate the patch functions from this script into")
        print("audit_deploy.py (see INTEGRATION PATCH section in source).")
        sys.exit(0)

    print(f"\nT-08  R_k step-size norm  (Q={Q_WARN:.2f}, m_fail={M_FAIL})\n")
    print(f"Processing {len(traj_available)} run(s) with trajectory data…\n")

    rows = []
    for mpath in traj_available:
        try:
            r = process_run(mpath, args.obs_dims)
            if r is None:
                continue
            r["_src"] = mpath

            # Save enriched metrics JSON
            out_path = mpath.replace("_metrics_summary.json", "_metrics_summary_t08.json")
            with open(out_path, "w") as f:
                out_data = {k: v for k, v in r.items() if not k.startswith("_")}
                json.dump(out_data, f, indent=2)

            rows.append(r)
        except Exception as e:
            print(f"  SKIP {os.path.basename(mpath)}: {e}")

    if rows:
        print_table(rows)
        out_csv = os.path.join(args.results_dir, "t08_rk_results.csv")
        write_csv(rows, out_csv)
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
