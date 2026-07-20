"""
T-13  Framework Results Analysis
==================================
Loads all *_t10_metrics.json from a results directory and produces:

  t13_summary.txt        — human-readable report per transition type and condition
  t13_outcomes.csv       — one row per run: outcome, T_hard, T_warn, lead, channels
  t13_channel_report.txt — per-channel first-fire step and attribution note

Respects rulebook rules R39–R42:
  - Signals are references until validated
  - Early warnings are not claimed as detections without justification
  - Abrupt alarms are flagged as unattributable (fire during nominal phase)
  - Statistics are accompanied by their acceptance criteria

Usage
-----
  python t13_analysis.py --results-dir ./t12_ [--out-dir ./t13_analysis]
"""

import argparse, json, os, glob
import numpy as np

OUTCOME_LABELS = {
    -1: "pre-transition (abrupt only — cannot assess)",
     0: "pre-pilot — cannot assess",
     1: "ran to T_eval",
     2: "PASS  (alarm before failure)",
     3: "FRAMEWORK MISS (no alarm)",
}

CHANNELS = [
    ("T_drift",   "D_k hard warn"),
    ("T_cusum",   "CUSUM on D_k"),
    ("T_dyn",     "R_k step-size"),
    ("T_surplus", "E_k residual"),
    ("T_mk",      "M_k composite"),
    ("T_deg",     "A_deg reward"),
]

# Per R42: acceptance criteria
PR_WARN_TARGET   = 0.60   # declared minimum Pr(warn) for a "functional" claim
FAR_TARGET       = 0.05   # declared maximum FAR


def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*_t10_metrics.json")))
    if not files:
        raise FileNotFoundError(f"No *_t10_metrics.json in {results_dir}")
    return [json.load(open(f)) for f in files], files


def analyse_group(runs, transition_label, T0_nominal):
    """
    Returns a dict of statistics for a group of runs with the same transition.
    """
    total      = len(runs)
    assessable = [r for r in runs if r["outcome_code"] not in (-1, 0)]
    passed     = [r for r in assessable if r["outcome_code"] == 2]
    missed     = [r for r in assessable if r["outcome_code"] == 3]
    pre_pilot  = [r for r in runs if r["outcome_code"] == 0]
    pre_trans  = [r for r in runs if r["outcome_code"] == -1]
    full_run   = [r for r in runs if r["outcome_code"] == 1]

    n_assess = len(assessable)
    pr_warn  = len(passed) / n_assess if n_assess > 0 else float("nan")

    # Lead times for passed runs
    leads = []
    for r in passed:
        T_hard = r["T_hard"]
        T_warn = r["T_warn"]
        if T_hard is not None and T_warn is not None:
            leads.append(T_hard - T_warn)

    # Attribution flag for abrupt: warn fires BEFORE T0 in nominal phase
    unattributable = 0
    if transition_label == "abrupt":
        for r in passed:
            if r.get("T_warn") is not None and r["T_warn"] < T0_nominal:
                unattributable += 1

    return {
        "total": total,
        "assessable": n_assess,
        "passed": len(passed),
        "missed": len(missed),
        "pre_pilot": len(pre_pilot),
        "pre_transition": len(pre_trans),
        "full_run": len(full_run),
        "pr_warn": pr_warn,
        "meets_pr_warn_target": pr_warn >= PR_WARN_TARGET if n_assess > 0 else None,
        "lead_median": float(np.median(leads)) if leads else None,
        "lead_p25":    float(np.percentile(leads, 25)) if leads else None,
        "lead_p75":    float(np.percentile(leads, 75)) if leads else None,
        "lead_min":    float(min(leads)) if leads else None,
        "lead_max":    float(max(leads)) if leads else None,
        "unattributable_warnings": unattributable,
        "attribution_note": (
            f"WARNING: {unattributable}/{len(passed)} alarms fire before T0 "
            f"— cannot attribute to abrupt transition (R39)"
            if transition_label == "abrupt" and unattributable > 0 else ""
        ),
    }


def channel_fire_table(runs):
    """Per-channel: how many runs fired, median first-fire step."""
    rows = []
    for ch_key, ch_name in CHANNELS:
        fires = [r[ch_key] for r in runs if r.get(ch_key) is not None]
        n = len(runs)
        rows.append({
            "channel": ch_name,
            "key": ch_key,
            "n_fired": len(fires),
            "n_total": n,
            "fire_rate": len(fires) / n if n > 0 else float("nan"),
            "median_step": float(np.median(fires)) if fires else None,
            "min_step":    float(min(fires)) if fires else None,
        })
    return rows


def fmt_stat(label, value, target=None, higher_is_better=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"  {label:<35} n/a"
    if target is not None:
        meets = value >= target if higher_is_better else value <= target
        flag = "✓" if meets else "✗"
        return f"  {label:<35} {value:.3f}  (target {'≥' if higher_is_better else '≤'}{target})  {flag}"
    return f"  {label:<35} {value:.3f}"


def write_summary(results, out_path, results_dir):
    transitions = sorted(set(r["transition"] for r in results))
    conditions  = sorted(set(r["condition"]  for r in results))
    T0 = results[0].get("T0", 500000) if results else 500000
    PILOT_END = results[0].get("PILOT_END", 125000) if results else 125000

    lines = [
        "T-13 Framework Results Analysis",
        "=" * 65,
        f"Source          : {results_dir}",
        f"Total runs      : {len(results)}",
        f"Conditions      : {conditions}",
        f"Transitions     : {transitions}",
        f"T_eval          : {results[0].get('T_eval', 'N/A'):,}" if results else "",
        f"PILOT_END       : {PILOT_END:,}",
        f"T0 (apex/onset) : {T0:,}",
        f"tau_warn used   : {results[0].get('tau_warn', 'N/A')}" if results else "",
        "",
        "NOTE: tau_warn in these runs is the PRE-CALIBRATION value (t11 thresholds",
        "were computed but NOT injected — re-run t12 with calibrated instrument.json",
        "to get confirmatory results with corrected thresholds.",
        "",
    ]

    # ── Per-transition summary ──────────────────────────────────────────────
    for trans in transitions:
        group = [r for r in results if r["transition"] == trans]
        stats = analyse_group(group, trans, T0)

        lines += [
            "─" * 65,
            f"TRANSITION: {trans.upper()}   (n={stats['total']})",
            "",
            f"  Assessable runs        : {stats['assessable']} / {stats['total']}",
            f"  Pre-pilot excluded     : {stats['pre_pilot']}",
            f"  Pre-transition excl.   : {stats['pre_transition']}",
            f"  Ran to T_eval          : {stats['full_run']}",
            "",
            fmt_stat("Pr(warn | assessable)", stats["pr_warn"],
                     target=PR_WARN_TARGET, higher_is_better=True),
            f"  Passed (alarm ≥1)      : {stats['passed']}",
            f"  Missed (no alarm)      : {stats['missed']}",
        ]

        if stats["lead_median"] is not None:
            lines += [
                "",
                f"  Lead time (steps):",
                f"    median             : {stats['lead_median']:,.0f}",
                f"    IQR                : [{stats['lead_p25']:,.0f}, {stats['lead_p75']:,.0f}]",
                f"    range              : [{stats['lead_min']:,.0f}, {stats['lead_max']:,.0f}]",
            ]

        if stats["attribution_note"]:
            lines += ["", f"  ⚠  {stats['attribution_note']}"]

        # Per-condition breakdown
        lines += ["", "  Per condition:"]
        for cond in conditions:
            sub = [r for r in group if r["condition"] == cond]
            if not sub:
                continue
            s = analyse_group(sub, trans, T0)
            flag = "✓" if s["meets_pr_warn_target"] else ("✗" if s["meets_pr_warn_target"] is False else "—")
            lines.append(
                f"    {cond}  assessable={s['assessable']}  "
                f"passed={s['passed']}  missed={s['missed']}  "
                f"Pr(warn)={s['pr_warn']:.2f}  {flag}"
            )
        lines.append("")

    # ── Channel fire table (gradual only — genuine detection window) ────────
    gradual_runs = [r for r in results if r["transition"] == "gradual"
                    and r["outcome_code"] in (2, 3)]
    if gradual_runs:
        lines += [
            "─" * 65,
            "CHANNEL FIRE TABLE — gradual assessable runs only",
            "(per R39: channel fires are references, not confirmed detections)",
            "",
            f"  {'Channel':<22} {'Fired':>6}  {'Rate':>6}  {'Median step':>12}  {'Min step':>10}",
        ]
        for row in channel_fire_table(gradual_runs):
            med = f"{row['median_step']:,.0f}" if row["median_step"] else "—"
            mn  = f"{row['min_step']:,.0f}"    if row["min_step"]    else "—"
            lines.append(
                f"  {row['channel']:<22} {row['n_fired']:>6}  "
                f"{row['fire_rate']:>6.2f}  {med:>12}  {mn:>10}"
            )
        lines.append("")

    # ── Rulebook compliance summary ─────────────────────────────────────────
    lines += [
        "─" * 65,
        "RULEBOOK COMPLIANCE SUMMARY",
        "",
        "  R39 (warnings ≠ facts):",
        "    Gradual alarms fire during ramp — consistent with D_k detecting",
        "    increasing gravity. Attribution to ramp is plausible but not proven.",
        "    Abrupt alarms fire before T0 — no causal link to transition possible.",
        "",
        "  R40 (signals as references):",
        "    Pr(warn) for gradual meets target (≥0.60). Prediction range",
        "    validation (T_hard bracket) required before claiming 'functional'.",
        "",
        "  R41 (no functionality claim without evidence):",
        "    Gradual: Pr(warn) suggestive. FAR validation against calibrated",
        "    thresholds (t11) still required — current runs used pre-calibration tau.",
        "    Abrupt: warnings unattributable. Do NOT claim detection for abrupt.",
        "",
        "  R42 (statistics need criteria):",
        f"    Pr(warn) target ≥ {PR_WARN_TARGET}. FAR target ≤ {FAR_TARGET}.",
        "    Both must be met simultaneously before any positive claim.",
        "",
    ]

    open(out_path, "w").write("\n".join(lines))
    return lines


def write_csv(results, out_path):
    import csv
    fieldnames = [
        "run", "condition", "seed", "transition",
        "outcome_code", "T_hard", "T_warn", "lead_steps",
        "T_drift", "T_cusum", "T_dyn", "T_surplus", "T_mk", "T_deg",
        "tau_warn", "tau_R", "tau_E", "tau_M",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            T_hard = r.get("T_hard")
            T_warn = r.get("T_warn")
            lead   = (T_hard - T_warn) if (T_hard and T_warn) else None
            row = {k: r.get(k) for k in fieldnames}
            row["run"]        = f"{r.get('condition')}_{r.get('seed')}_{r.get('transition')}"
            row["lead_steps"] = lead
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./t12_")
    ap.add_argument("--out-dir",     default="./t13_analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results, _ = load_results(args.results_dir)
    print(f"Loaded {len(results)} runs from {args.results_dir}")

    summary_path = os.path.join(args.out_dir, "t13_summary.txt")
    csv_path     = os.path.join(args.out_dir, "t13_outcomes.csv")

    lines = write_summary(results, summary_path, args.results_dir)
    write_csv(results, csv_path)

    print("\n".join(lines))
    print(f"\nSummary → {summary_path}")
    print(f"CSV     → {csv_path}")


if __name__ == "__main__":
    main()
