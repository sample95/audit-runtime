"""
T-11  Threshold Calibration
============================
Loads all *_none_t10_metrics.json files, pools post-pilot signal values
across every model × seed run, and computes empirical percentile-based
thresholds for each channel.

Outputs
-------
  calibration_thresholds.json   — new tau_* values
  t11_calibration_report.txt    — per-channel FAR table (old vs new)

Usage
-----
  python t11_calibrate.py --results-dir ./t10_rf --out-dir ./t11_calib
"""

import argparse, json, os, numpy as np
from pathlib import Path

CHANNELS = [
    ("D_k",       "tau_warn",  "tau_watch"),   # JSD distributional drift
    ("R_k",       "tau_R",     None),           # obs step-size norm (T-08)
    ("E_k",       "tau_E",     None),           # OLS reward residual (T-09)
    ("M_k",       "tau_M",     None),           # Mahalanobis KPI composite — retroactively computed from D_k+y_k if absent
    ("cusum_S",   "tau_cusum", None),           # CUSUM on D_k (T-07)
    ("beta_TS_k", "tau_TS",   None),            # rolling OLS trend slope (T-15) — recomputed from y_k if absent
]
# Desired max FAR: fraction of post-pilot windows that may fire
FAR_TARGETS = [0.01, 0.05]   # 1 % and 5 %

def load_none_files(results_dir):
    pattern = Path(results_dir)
    # Primary: t10_survival_deploy.py naming
    files = sorted(pattern.glob("*_none_t10_metrics.json"))
    if not files:
        # Fallback: audit_deploy.py naming (has D_k, cusum_S, F_k; R_k/E_k/M_k absent)
        files = sorted(pattern.glob("*_none_metrics_summary.json"))
    if not files:
        raise FileNotFoundError(
            f"No *_none_t10_metrics.json or *_none_metrics_summary.json in {results_dir}")
    print(f"Found {len(files)} none runs ({files[0].name} …)")
    return [json.load(open(f)) for f in files]

def _rolling_bts_from_yk(record):
    """
    Recompute per-window rolling OLS slopes from stored y_k array.
    Mirrors compute_rolling_bts() in t10_survival_deploy.py exactly.
    """
    y_k = np.array(record["y_k"], dtype=float)
    K   = len(y_k)
    bts = np.zeros(K)
    for k in range(1, K):
        x = np.arange(k + 1, dtype=float)
        bts[k] = float(np.polyfit(x, y_k[:k + 1], 1)[0])
    return bts


def _mk_from_dk_yk(record):
    """
    Retroactively compute M_k from D_k and y_k when M_k is not stored.
    Mirrors calibrate_mk() / compute_mk() in t10_survival_deploy.py exactly.
    M_k = ||[ỹ_k, D̃_k]||₂  in pilot-normalised space (diagonal Mahalanobis).
    """
    EPS = 1e-8
    K0  = record.get("K0", 4)
    D_k = np.array(record["D_k"], dtype=float)
    y_k = np.array(record["y_k"], dtype=float)
    mu_y    = y_k[:K0].mean();  sigma_y = y_k[:K0].std(ddof=1) + EPS
    mu_D    = D_k[:K0].mean();  sigma_D = D_k[:K0].std(ddof=1) + EPS
    return np.sqrt(((y_k - mu_y) / sigma_y) ** 2 + ((D_k - mu_D) / sigma_D) ** 2)


def pool_signal(records, key, pilot_end_idx):
    """Collect all post-pilot values of signal `key` across all records.

    Special case: if key == 'M_k' and the field is absent, compute it
    retroactively from D_k and y_k (both always present).
    """
    vals = []
    for r in records:
        arr = r.get(key)
        if arr is None:
            if key == "M_k" and "D_k" in r and "y_k" in r:
                arr = _mk_from_dk_yk(r)
            elif key == "beta_TS_k" and "y_k" in r:
                arr = _rolling_bts_from_yk(r)
            else:
                continue
        arr = np.array(arr, dtype=float)
        K0 = r.get("K0", 4)
        arr = arr[K0:]
        vals.append(arr)
    return np.concatenate(vals) if vals else np.array([])

def compute_thresholds(vals, targets):
    if len(vals) == 0:
        return {}
    return {f"p{int(t*100):02d}": float(np.percentile(np.abs(vals), (1-t)*100))
            for t in targets}

def far_at_threshold(vals, tau, absolute=True):
    if len(vals) == 0:
        return float("nan")
    v = np.abs(vals) if absolute else vals
    return float((v > tau).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./t10_rf")
    ap.add_argument("--out-dir",     default="./t11_calib")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_none_files(args.results_dir)

    report_lines = [
        "T-11 Calibration Report",
        "=" * 60,
        f"None runs used: {len(records)}",
        "",
    ]

    new_taus = {}

    for sig_key, tau_key, tau_watch_key in CHANNELS:
        vals = pool_signal(records, sig_key, pilot_end_idx=4)
        if len(vals) == 0:
            report_lines.append(f"{sig_key}: NO DATA")
            continue

        # Current tau (median across runs, since each run has its own)
        current_taus = [r[tau_key] for r in records if tau_key in r and r[tau_key] is not None]
        tau_current  = float(np.median(current_taus)) if current_taus else None

        # New thresholds at FAR targets
        thresholds = {}
        for far_t in FAR_TARGETS:
            pct = (1 - far_t) * 100
            # For E_k use absolute value; for R_k threshold is upper tail
            v = np.abs(vals) if sig_key in ("E_k", "cusum_S") else vals
            thresholds[far_t] = float(np.percentile(v, pct))

        new_tau = thresholds[FAR_TARGETS[0]]   # use 1% FAR target as default
        new_taus[tau_key] = new_tau

        # Current FAR
        if tau_current is not None:
            v = np.abs(vals) if sig_key in ("E_k", "cusum_S") else vals
            cur_far = float((v > tau_current).mean())
        else:
            cur_far = float("nan")

        new_far = float((vals > new_tau).mean()) if sig_key not in ("E_k","cusum_S") \
                  else float((np.abs(vals) > new_tau).mean())

        report_lines += [
            f"Channel : {sig_key}  (threshold key: {tau_key})",
            f"  N post-pilot windows   : {len(vals)}",
            f"  Signal range           : [{vals.min():.6f}, {vals.max():.6f}]",
            f"  Current tau (median)   : {tau_current:.6f}"  if tau_current else "  Current tau: N/A",
            f"  Current FAR            : {cur_far*100:.1f}%",
            f"  New tau (FAR≤1%)       : {thresholds[0.01]:.6f}",
            f"  New tau (FAR≤5%)       : {thresholds[0.05]:.6f}",
            f"  FAR at new tau (1%)    : {new_far*100:.1f}%",
            "",
        ]

        # Also handle watch threshold if present
        if tau_watch_key:
            watch_taus = [r[tau_watch_key] for r in records
                          if tau_watch_key in r and r[tau_watch_key] is not None]
            if watch_taus:
                tau_w_cur = float(np.median(watch_taus))
                w_far_cur = float((vals > tau_w_cur).mean())
                # watch threshold = 95% of hard threshold
                new_taus[tau_watch_key] = new_tau * 0.95
                report_lines.append(
                    f"  Watch tau current/FAR  : {tau_w_cur:.6f} / {w_far_cur*100:.1f}%  "
                    f"→ new: {new_tau*0.95:.6f}"
                )
                report_lines.append("")

    # CUSUM needs its own tau field (not in CHANNELS list above)
    cusum_vals = pool_signal(records, "cusum_S", pilot_end_idx=4)
    if len(cusum_vals):
        new_taus["tau_cusum"] = float(np.percentile(np.abs(cusum_vals), 99))

    # Save
    out_json = os.path.join(args.out_dir, "calibration_thresholds.json")
    out_txt  = os.path.join(args.out_dir, "t11_calibration_report.txt")

    json.dump(new_taus, open(out_json, "w"), indent=2)
    open(out_txt, "w").write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nThresholds saved → {out_json}")
    print(f"Report saved    → {out_txt}")

if __name__ == "__main__":
    main()
