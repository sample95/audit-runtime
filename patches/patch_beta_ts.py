"""
patch_beta_ts.py
Post-processes existing *_metrics_summary.json and *_rollouts_summary.json files
to add T_bts and beta_ts_series — the β_TS rolling-slope alarm fields introduced
in audit_deploy.py v12.9.

No simulation required: all inputs (y_k, window_starts, K0) are already in the
existing metrics JSONs.  Same pattern as patch_convergence.py.

Usage:
    python3 patch_beta_ts.py                          # hopper_vc default
    python3 patch_beta_ts.py results/mujoco/hopper_vc
    python3 patch_beta_ts.py results/mujoco/hopper_vc --tau-bts -1e-6
    python3 patch_beta_ts.py results/mujoco/hopper_vc --window 8
"""
import argparse
import glob
import json
import math
import os

import numpy as np

# ── Defaults (must match audit_deploy.py v12.9) ────────────────────────────────
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "mujoco", "hopper_vc"
)
DEFAULT_BETA_TS_WINDOW = 5
# tau_bts = 0.0 fires on any momentary negative slope (high FAR on null condition).
# −5e-7 requires a sustained trend of ~25% reward loss over 1M steps before alarming;
# passes null-condition FAR while still catching genuine gradual degradation.
DEFAULT_TAU_BTS        = -5e-7


# ── Replicate first_eligible_warning_start from audit_deploy.py ───────────────
def first_eligible_warning_start(K0: int, W: int, S: int) -> int:
    pilot_end = (K0 - 1) * S + W
    return math.ceil(pilot_end / S) * S


# ── Rolling OLS β_TS alarm (mirrors compute_beta_ts_alarm in audit_deploy.py) ─
def compute_beta_ts_alarm(y_k, starts, K0, tau_bts, beta_ts_window, W, S, w=0):
    first_warn = first_eligible_warning_start(K0, W, S)
    n = len(y_k)
    beta_ts_series = [None] * n
    T_bts = None

    eligible_idx = [i for i, s in enumerate(starts) if int(s) >= first_warn]

    for rank, i in enumerate(eligible_idx):
        lo = max(0, rank - beta_ts_window + 1)
        win_idx = eligible_idx[lo : rank + 1]
        if len(win_idx) < 2:
            continue

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Folder containing *_metrics_summary.json files")
    parser.add_argument("--window", type=int, default=DEFAULT_BETA_TS_WINDOW,
                        help=f"Rolling window width in windows (default {DEFAULT_BETA_TS_WINDOW})")
    parser.add_argument("--tau-bts", type=float, default=DEFAULT_TAU_BTS,
                        help=f"Alarm threshold: slope < tau_bts fires T_bts "
                             f"(default {DEFAULT_TAU_BTS})")
    parser.add_argument("--force", action="store_true",
                        help="Re-patch files that already have T_bts (e.g. to change tau_bts)")
    args = parser.parse_args()

    beta_ts_window = args.window
    tau_bts        = args.tau_bts

    metrics_files = sorted(glob.glob(
        os.path.join(args.results_dir, "*_metrics_summary.json")
    ))
    if not metrics_files:
        print(f"ERROR: no *_metrics_summary.json files found in:\n  {args.results_dir}")
        return

    print(f"patch_beta_ts.py  window={beta_ts_window}  tau_bts={tau_bts}")
    print(f"Found {len(metrics_files)} metrics file(s) in {args.results_dir}\n")

    patched = skipped = 0

    for mpath in metrics_files:
        with open(mpath) as f:
            m = json.load(f)

        if "T_bts" in m and not args.force:
            print(f"  [skip]  {os.path.basename(mpath)} — T_bts already present (--force to overwrite)")
            skipped += 1
            continue

        y_k    = m.get("y_k")
        starts = m.get("window_starts")
        K0     = m.get("K0")

        if y_k is None or starts is None or K0 is None:
            print(f"  [warn]  {os.path.basename(mpath)} — missing y_k/window_starts/K0, skip")
            skipped += 1
            continue

        if len(starts) < 2:
            print(f"  [warn]  {os.path.basename(mpath)} — need ≥2 window_starts, skip")
            skipped += 1
            continue
        S = int(m.get("S", starts[1] - starts[0]))
        # W is stored explicitly in audit_deploy.py v12.9+.
        # Older metrics JSONs omit it; fall back to 2×S (standard hopper_vc config).
        # Do NOT derive as T_eval - starts[-1]: that gives wrong values when
        # deployment_len < T_eval (early termination due to T_hard).
        W = int(m.get("W", 2 * S))

        T_bts, beta_ts_series = compute_beta_ts_alarm(
            y_k, starts, K0, tau_bts, beta_ts_window, W, S, w=W
        )

        # Patch metrics JSON in-place
        m["T_bts"]           = T_bts
        m["beta_ts_series"]  = beta_ts_series
        m["tau_bts"]         = tau_bts
        m["beta_ts_window"]  = beta_ts_window

        with open(mpath, "w") as f:
            json.dump(m, f, indent=2)

        # Patch matching rollout JSON
        rpath = mpath.replace("_metrics_summary.json", "_rollouts_summary.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                r = json.load(f)
            r["T_bts"]   = T_bts
            r["tau_bts"] = tau_bts
            with open(rpath, "w") as f:
                json.dump(r, f, indent=2)

        print(f"  [patch] {os.path.basename(mpath)}  T_bts={T_bts}")
        patched += 1

    # ── Rebuild aggregate rollouts summary ─────────────────────────────────────
    agg_candidates = glob.glob(os.path.join(args.results_dir, "..", "*_rollouts_summary.json"))
    if agg_candidates:
        # Reload all per-run rollout JSONs and rebuild
        rollout_files = sorted(glob.glob(
            os.path.join(args.results_dir, "*_rollouts_summary.json")
        ))
        all_rollouts = []
        for rpath in rollout_files:
            with open(rpath) as f:
                all_rollouts.append(json.load(f))
        # Use the domain from the first entry to pick the right aggregate file
        if all_rollouts:
            domain = all_rollouts[0].get("domain", "unknown")
            agg_path = os.path.join(args.results_dir, "..", f"{domain}_rollouts_summary.json")
            with open(agg_path, "w") as f:
                json.dump(all_rollouts, f, indent=2)
            print(f"\nAggregate rollouts rebuilt → {agg_path} ({len(all_rollouts)} runs)")

    print(f"\nDone — patched={patched}  skipped={skipped}")


if __name__ == "__main__":
    main()
