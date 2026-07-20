#!/usr/bin/env python3
"""
evaluation.py  ·  v2.0
Entry point for the RL audit evaluation pipeline.

Usage:
    python evaluation.py --domain hopper
    python evaluation.py --domain hopper --results_dir ./results/mujoco/hopper/idk
    python evaluation.py --domain hopper --sections 1,2,8,9

Outputs (saved to --out_dir, default: <results_dir>/eval/):
    01_overview_heatmap.png
    02_signal_trajectories.png
    03_warning_quality.png
    04_policy_comparison.png
    05_failure_anatomy.png
    06_seed_consistency.png
    07_similarities.png
    08_km_rul_forecast.png
    09_seed_video.png
    summary.txt
    rul_forecast.json   (optional, if --save_forecast)

Module layout:
    eval_utils.py  — helpers, palettes, statistical functions, data loading
    eval_plots.py  — sections 1–9 (all plot functions)
    evaluation.py  — write_summary + main()  ← this file
"""

import argparse
import json
import os
import sys

import numpy as np

from eval_utils import (
    load_metrics, all_conditions, all_transitions, safe_float,
)
from eval_plots import (
    plot_overview_heatmap,
    plot_signal_trajectories,
    plot_warning_quality,
    plot_policy_comparison,
    plot_failure_anatomy,
    plot_seed_consistency,
    plot_similarities,
    plot_km_rul,
    plot_seed_video,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY TEXT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def write_summary(recs, out_path, domain, forecast_records=None):
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)
    T_eval = max(safe_float(r.get("T_eval", 1e6)) for r in recs)

    lines = [
        "=" * 64,
        f"  DEPLOYMENT AUDIT SUMMARY  ·  domain={domain}",
        "=" * 64,
        f"  Total runs analysed : {len(recs)}",
        f"  Conditions          : {', '.join(conds)}",
        f"  Transition schedules: {', '.join(transitions)}",
        f"  T_eval              : {T_eval:,.0f} steps",
        "",
    ]

    for c in conds:
        for d in transitions:
            sub = [r for r in recs if r.get("condition")==c and r.get("transition")==d]
            if not sub: continue
            n         = len(sub)
            warned    = sum(1 for r in sub if r.get("T_warn") is not None)
            timely    = sum(1 for r in sub if r.get("timely_warn"))
            hard      = sum(1 for r in sub if r.get("T_hard") is not None)
            t_warn_v  = [safe_float(r.get("T_warn")) for r in sub if r.get("T_warn") is not None]
            dd_v      = [safe_float(r.get("delta_delay")) for r in sub
                         if r.get("timely_warn") and r.get("delta_delay") is not None]
            lines += [
                f"  [{c}  ×  {d}]  n={n}",
                f"    Pr(warn)          = {warned/n:.2f}   ({warned}/{n})",
                f"    Pr(timely|warned) = {timely/max(warned,1):.2f}   ({timely}/{warned})",
                f"    Pr(hard_fail)     = {hard/n:.2f}   ({hard}/{n})",
            ]
            if t_warn_v:
                lines.append(f"    T_warn  med/IQR   = {np.median(t_warn_v):,.0f} / "
                             f"[{np.percentile(t_warn_v,25):,.0f} – {np.percentile(t_warn_v,75):,.0f}]")
            if dd_v:
                lines.append(f"    delta_delay med   = {np.median(dd_v):,.0f} steps")
            lines.append("")

    if forecast_records:
        lines += ["  REMAINING USEFUL LIFE (Theil–Sen):", ""]
        for fr in forecast_records:
            rul  = fr.get("RUL_steps")
            lo   = fr.get("RUL_CI_lo_frac")
            hi   = fr.get("RUL_CI_hi_frac")
            frac = fr.get("RUL_T_eval_frac")
            ci   = f"90% CI: [{lo:.2f}, {hi:.2f}] × T_eval" if lo and hi else "CI: n/a"
            lines.append(
                f"    {fr['condition']:3s} / {fr['transition']:10s}  "
                f"RUL = {rul:,.0f} steps  ({frac:.2f} × T_eval)  {ci}"
            )
        lines.append("")

    lines += ["=" * 64]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [sum] {out_path}")
    print("\n".join(lines))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="RL deployment audit evaluation")
    parser.add_argument("--domain",      required=True, help="Domain key (e.g. hopper)")
    parser.add_argument("--config",      default=None,
                        help="Path to domain JSON config (auto-detected if omitted)")
    parser.add_argument("--results_dir", default=None,
                        help="Override results directory from config")
    parser.add_argument("--out_dir",     default=None,
                        help="Directory for output plots (default: results_dir/eval)")
    parser.add_argument("--save_forecast", action="store_true",
                        help="Save RUL forecast to rul_forecast.json")
    parser.add_argument("--sections",    default="all",
                        help="Comma-separated section numbers to run, e.g. 1,2,8,9 (default: all)")
    args = parser.parse_args()

    # ── Resolve config ────────────────────────────────────────────────────
    cfg_path = args.config
    if cfg_path is None:
        candidates = [
            f"{args.domain}.json",
            os.path.join(os.path.dirname(__file__), f"{args.domain}.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                cfg_path = c; break
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            dcfg = json.load(f)
        print(f"  [config] loaded {cfg_path}")
    else:
        dcfg = {}
        print(f"  [config] no config found, using defaults")

    results_dir = args.results_dir or dcfg.get("results_dir", f"./results/{args.domain}")
    out_dir     = args.out_dir or os.path.join(results_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading metrics from: {results_dir}")
    recs = load_metrics(results_dir, args.domain)
    if not recs:
        print("  ERROR: no records found. Run deployment.py first.")
        sys.exit(1)
    print(f"  Loaded {len(recs)} records | "
          f"conditions={all_conditions(recs)} | transitions={all_transitions(recs)}")

    # ── Resolve sections ──────────────────────────────────────────────────
    if args.sections.strip().lower() == "all":
        run_sections = set(range(1, 10))
    else:
        run_sections = set(int(x.strip()) for x in args.sections.split(",") if x.strip())

    def out(name):
        return os.path.join(out_dir, name)

    print(f"\nRunning sections: {sorted(run_sections)}")
    print(f"Output directory: {out_dir}\n")

    forecast_records = None

    if 1 in run_sections:
        plot_overview_heatmap(recs, out("01_overview_heatmap.png"))

    if 2 in run_sections:
        plot_signal_trajectories(recs, out("02_signal_trajectories.png"))

    if 3 in run_sections:
        plot_warning_quality(recs, out("03_warning_quality.png"))

    if 4 in run_sections:
        plot_policy_comparison(recs, out("04_policy_comparison.png"))

    if 5 in run_sections:
        plot_failure_anatomy(recs, out("05_failure_anatomy.png"))

    if 6 in run_sections:
        plot_seed_consistency(recs, out("06_seed_consistency.png"))

    if 7 in run_sections:
        plot_similarities(recs, out("07_similarities.png"))

    if 8 in run_sections:
        fc_path = out("rul_forecast.json") if args.save_forecast else None
        forecast_records = plot_km_rul(recs, out("08_km_rul_forecast.png"),
                                       save_forecast=fc_path)

    if 9 in run_sections:
        plot_seed_video(recs, out("09_seed_video.png"))

    write_summary(recs, out("summary.txt"), args.domain, forecast_records)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
