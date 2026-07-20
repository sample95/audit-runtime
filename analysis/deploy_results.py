"""
deploy_results.py — Domain-agnostic ETL + evaluation table for deployment audit results
========================================================================================
SCOPE: CONFIRMATORY evaluation only (locked v12.6 methodology).
This script computes and reports the primary thesis metrics. It does NOT
read aux_action_windows or trajectory data — those belong in exploratory_analysis.py.
See tuning_journal.md §Confirmatory vs Exploratory Boundary.

Loads all *_metrics_summary.json from a results dir into a pandas DataFrame,
computes the standard thesis evaluation table, and outputs CSV + printed tables.

Usage (from audit-framework/):
    python deploy_results.py --domain hopper          # full evaluation
    python deploy_results.py --domain hopper_t01      # T-01 FAR diagnostic
    python deploy_results.py --domain hopper --latex  # also emit LaTeX table

Output files (in results_dir/):
    eval_table.csv          main condition × drift evaluation table
    per_seed_table.csv      per-seed tau_warn / FAR diagnostics
    far_summary.csv         false alarm rate summary
    all_runs_flat.csv       full flat dataframe (scalar cols only)
"""

import argparse
import json
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)

parser = argparse.ArgumentParser(description="Deploy results ETL + evaluation table")
parser.add_argument("--domain",  required=True,
                    help="Domain name (must match a config in --configs dir)")
parser.add_argument("--configs", default="./configs",
                    help="Path to configs directory (default: ./configs)")
parser.add_argument("--latex",      action="store_true",
                    help="Also print LaTeX table rows")
parser.add_argument("--no-save",    action="store_true",
                    help="Print only, do not write CSV/txt files")
parser.add_argument("--no-figures", action="store_true",
                    help="Skip KM/RUL figure generation (useful on headless systems without matplotlib)")
args = parser.parse_args()

def _load_json(path):
    with open(path) as f:
        return json.load(f)

cfg_path = Path(args.configs) / f"{args.domain}.json"
if not cfg_path.exists():
    sys.exit(f"Config not found: {cfg_path}")

cfg         = _load_json(cfg_path)
RESULTS_DIR = Path(cfg["results_dir"])
T_EVAL      = int(cfg.get("T_eval", 1_000_000))
T0          = T_EVAL // 2
CONDITIONS  = cfg.get("conditions", ["M1", "M2", "M3"])
SEEDS       = cfg.get("seeds",      [0, 1, 2, 3, 4])


# ════════════════════════════════════════════════════════════════════════════
# ETL — load all JSON results into a flat DataFrame
# ════════════════════════════════════════════════════════════════════════════

SERIES_KEYS = ["pilot_D", "D_k", "y_k", "A_deg", "A_drift", "A_watch",
               "window_starts"]

def _flatten_record(d: dict) -> dict:
    """Flatten one metrics JSON into a row dict.
    Series (list-valued) fields → '<key>_list' columns.
    Nested safety_margins dict → flat sm_<name>_<metric> columns.
    """
    row = {}
    for k, v in d.items():
        if isinstance(v, list):
            if k in SERIES_KEYS:
                row[f"{k}_list"] = v
            # else drop (e.g. mu, sigma arrays)
        elif isinstance(v, dict):
            if k == "safety_margins":
                for sm_name, sm_vals in v.items():
                    if isinstance(sm_vals, dict):
                        for metric, val in sm_vals.items():
                            row[f"sm_{sm_name}_{metric}"] = val
        else:
            row[k] = v
    return row


def load_results(results_dir: Path):
    """Returns (df, raw_recs).
    df        — flattened DataFrame for tabular analysis
    raw_recs  — list of raw JSON dicts needed by plot_km_rul
    """
    files = sorted(results_dir.glob("*_metrics_summary.json"))
    if not files:
        sys.exit(f"No *_metrics_summary.json found in {results_dir}")

    raw_recs = [json.loads(f.read_text()) for f in files]
    # Normalize the null-condition label: "baseline" is canonical going forward;
    # "none"/"no_drift" are pre-rename aliases still present in older result JSONs.
    _BASELINE_ALIASES = ("none", "no_drift")
    for r in raw_recs:
        if r.get("transition") in _BASELINE_ALIASES:
            r["transition"] = "baseline"
    records  = [_flatten_record(r) for r in raw_recs]
    df = pd.DataFrame(records)

    # canonical dtypes
    for col in ["condition", "transition", "domain"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in ["seed", "K0", "B", "T_eval", "T_hard", "T_soft",
                "T_comp", "T_comp_obs", "T_warn", "T_deg", "T_drift",
                "delta", "n_episodes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_key = lambda r: (r.get("condition",""), r.get("seed", 0), r.get("transition",""))
    raw_recs = sorted(raw_recs, key=sort_key)
    df = df.sort_values(["condition", "seed", "transition"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} runs from {results_dir}")
    print(f"  Conditions × transitions: "
          f"{df['condition'].nunique()} × {df['transition'].nunique()}")
    return df, raw_recs


# ════════════════════════════════════════════════════════════════════════════
# Computed columns
# ════════════════════════════════════════════════════════════════════════════

def add_derived(df: pd.DataFrame, t0: int, t_eval: int) -> pd.DataFrame:
    df = df.copy()

    # Re-derive T_comp_obs respecting hard_failure_counts flag.
    # Old results may have been computed with T_hard always active; correct here.
    hard_fail_col = "hard_failure_counts"
    if hard_fail_col in df.columns:
        def _recompute_comp(row):
            hfc   = bool(row.get(hard_fail_col, True))
            t_h   = row.get("T_hard")
            t_s   = row.get("T_soft")
            cands = []
            if hfc and pd.notna(t_h):  cands.append(int(t_h))
            if pd.notna(t_s):           cands.append(int(t_s))
            if cands:
                return int(min(cands)), 1
            return t_eval, 0
        df[["T_comp_obs", "delta"]] = df.apply(
            lambda r: pd.Series(_recompute_comp(r)), axis=1
        )

    # Timely: T_warn < T_comp_obs AND delta=1
    df["timely_warn"] = (
        (df["delta"] == 1)
        & df["T_warn"].notna()
        & (df["T_warn"] < df["T_comp_obs"])
    )

    # FAR flag: warned but no real event (delta=0)
    df["false_alarm"] = df["T_warn"].notna() & (df["delta"] == 0)

    # Post-pilot exceed rate — computed from series if available
    def _exceed_rate(row):
        dk   = row.get("D_k_list", [])
        sw   = row.get("window_starts_list", [])
        tw   = row.get("tau_warn", None)
        k0   = int(row.get("K0", 4))
        if not dk or tw is None:
            return np.nan
        post = [(d, s) for d, s in zip(dk[k0:], sw[k0:])]
        if not post:
            return np.nan
        n_exceed = sum(1 for d, s in post if d > tw)
        return n_exceed / len(post)

    df["exceed_rate"] = df.apply(_exceed_rate, axis=1)

    # Delta delay in windows (T_warn steps relative to T0)
    df["delta_delay_steps"] = df.apply(
        lambda r: (r["T_warn"] - t0)
        if pd.notna(r["T_warn"]) and r.get("transition") == "abrupt"
        else np.nan,
        axis=1,
    )

    return df


# ════════════════════════════════════════════════════════════════════════════
# Tables
# ════════════════════════════════════════════════════════════════════════════

def eval_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main thesis evaluation table.
    Rows: condition × transition
    Cols: Pr(warn), Pr(timely|warned), Pr(hard_fail), T_warn_med, n_runs
    """
    rows = []
    for cond in sorted(df["condition"].unique()):
        for drift in ["baseline", "abrupt", "gradual"]:
            sub = df[(df["condition"] == cond) & (df["transition"] == drift)]
            if sub.empty:
                continue
            n         = len(sub)
            n_warned  = sub["T_warn"].notna().sum()
            n_timely  = sub["timely_warn"].sum()
            n_fail    = (sub["delta"] == 1).sum()
            t_warn_m  = sub["T_warn"].median()

            rows.append({
                "Condition":         f"{cond} × {drift}",
                "n_runs":            n,
                "Pr(warn)":          round(n_warned / n, 2),
                "Pr(timely|warned)": round(n_timely / n_warned, 2) if n_warned > 0 else np.nan,
                "Pr(hard_fail)":     round(n_fail / n, 2),
                "T_warn_med":        int(t_warn_m) if pd.notna(t_warn_m) else None,
            })

    return pd.DataFrame(rows)


def per_seed_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-seed diagnostic: tau_warn, floor, pilot_D_max, exceed_rate, T_warn, FA?
    Useful for T-01 analysis and calibration reporting.
    """
    rows = []
    for _, row in df.iterrows():
        pilot_D = row.get("pilot_D_list", [])
        rows.append({
            "condition":     row.get("condition"),
            "seed":          int(row.get("seed", -1)),
            "transition":    row.get("transition"),
            "tau_warn":      row.get("tau_warn"),
            "tau_warn_raw":  row.get("tau_warn_raw", np.nan),
            "tau_warn_floor":row.get("tau_warn_floor", 0.0),
            "pilot_D_max":   max(pilot_D) if pilot_D else np.nan,
            "exceed_rate":   row.get("exceed_rate", np.nan),
            "T_warn":        row.get("T_warn"),
            "false_alarm":   row.get("false_alarm", False),
        })
    return pd.DataFrame(rows)


def far_summary(df: pd.DataFrame) -> pd.DataFrame:
    """FAR summary: per condition × drift, how many seeds false-alarm."""
    rows = []
    for cond in sorted(df["condition"].unique()):
        for drift in sorted(df["transition"].unique()):
            sub = df[(df["condition"] == cond) & (df["transition"] == drift)]
            if sub.empty:
                continue
            n_fa  = sub["false_alarm"].sum()
            n_tot = len(sub)
            rows.append({
                "condition":  cond,
                "transition": drift,
                "n_false_alarm": int(n_fa),
                "n_runs":        n_tot,
                "FAR":           round(n_fa / n_tot, 2) if n_tot > 0 else np.nan,
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# LaTeX helper
# ════════════════════════════════════════════════════════════════════════════

def to_latex_rows(tbl: pd.DataFrame):
    lines = []
    for _, r in tbl.iterrows():
        tw = str(r["T_warn_med"]) if r["T_warn_med"] is not None else "—"
        pt = f"{r['Pr(timely|warned)']:.2f}" if pd.notna(r["Pr(timely|warned)"]) else "—"
        line = (
            f"  {r['Condition']:<25} & {r['Pr(warn)']:.2f} & {pt}"
            f" & {r['Pr(hard_fail)']:.2f} & {tw} \\\\"
        )
        lines.append(line)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def _hr(title=""):
    width = 72
    if title:
        pad = (width - len(title) - 2) // 2
        print("═" * pad + f" {title} " + "═" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def main():
    print()
    _hr(f"DEPLOY RESULTS  ·  {args.domain}")
    print(f"  results_dir : {RESULTS_DIR}")
    print(f"  T_eval={T_EVAL:,}  T0={T0:,}")
    print()

    # ── ETL ──────────────────────────────────────────────────────────────
    df_raw, raw_recs = load_results(RESULTS_DIR)
    df               = add_derived(df_raw, t0=T0, t_eval=T_EVAL)

    # Patch raw_recs with corrected T_comp_obs / delta from add_derived().
    # add_derived() may recompute T_comp_obs based on hard_failure_counts flag,
    # which could differ from what was stored in the JSON for older runs.
    # Both lists are sorted identically so index alignment is safe.
    for rec, (_, row) in zip(raw_recs, df.iterrows()):
        rec["T_comp_obs"] = float(row["T_comp_obs"])
        rec["delta"]      = int(row["delta"])
        rec["timely_warn"] = bool(row["timely_warn"])
        rec["T_eval"]     = T_EVAL
    print(f"  Columns in raw df: {len(df_raw.columns)} "
          f"({len([c for c in df_raw.columns if c.endswith('_list')])} series)")

    # ── Evaluation table ─────────────────────────────────────────────────
    _hr("Evaluation Table")
    ev = eval_table(df)
    print(ev.to_string(index=False))

    # ── Per-seed calibration table ────────────────────────────────────────
    _hr("Per-seed Calibration / FAR Diagnostics")
    ps = per_seed_table(df)
    # Show only baseline (FAR diagnostic) if it exists, else all
    ps_show = ps[ps["transition"] == "baseline"] if "baseline" in ps["transition"].values else ps
    print(ps_show.to_string(index=False))

    # ── FAR summary ──────────────────────────────────────────────────────
    _hr("False Alarm Rate Summary")
    fa = far_summary(df)
    print(fa.to_string(index=False))

    # ── tau_warn floor effect ─────────────────────────────────────────────
    if "tau_warn_floor" in df.columns and df["tau_warn_floor"].notna().any():
        _hr("tau_warn Floor Effect")
        floor_val = df["tau_warn_floor"].dropna().iloc[0]
        raw_col   = "tau_warn_raw" if "tau_warn_raw" in df.columns else "tau_warn"
        sub = df[df["transition"] == "baseline"].copy() if "baseline" in df["transition"].values else df.copy()
        sub = sub.sort_values(["condition", "seed"])
        floored = (sub["tau_warn"] > sub[raw_col] + 1e-12).sum() if raw_col in sub.columns else 0
        print(f"  floor = {floor_val:.2e}")
        if raw_col != "tau_warn":
            print(f"  Seeds where floor raised tau_warn: {floored}/{len(sub)}")
            print()
            tbl = sub[["condition", "seed", raw_col, "tau_warn", "exceed_rate", "T_warn", "false_alarm"]]
            print(tbl.rename(columns={raw_col: "tau_warn_raw"}).to_string(index=False))

    # ── T_warn distribution (abrupt drift) ───────────────────────────────
    abrupt = df[df["transition"] == "abrupt"]
    if not abrupt.empty:
        _hr("T_warn Distribution (abrupt drift)")
        tw_stats = (abrupt
                    .groupby("condition")["T_warn"]
                    .agg(["count", "min", "median", "max",
                          lambda x: x.notna().sum()])
                    .rename(columns={"<lambda_0>": "n_warned"}))
        tw_stats["Pr(warn)"] = (tw_stats["n_warned"] / tw_stats["count"]).round(2)
        print(tw_stats.to_string())

    # ── LaTeX ─────────────────────────────────────────────────────────────
    if args.latex:
        _hr("LaTeX rows")
        print(to_latex_rows(ev))

    # ── KM survival + Theil-Sen RUL forecast ─────────────────────────────
    # plot_km_rul lives in eval_plots.py (imports matplotlib).
    # Skipped if --no-figures or matplotlib unavailable.
    km_figure = None
    fc_records = []
    if not args.no_figures:
        try:
            from eval_plots import plot_km_rul
            _hr("KM Survival + RUL Forecast")
            km_path = RESULTS_DIR / "km_rul_forecast.png"
            fc_path = RESULTS_DIR / "rul_forecast.json" if not args.no_save else None
            fc_records = plot_km_rul(raw_recs, str(km_path), save_forecast=str(fc_path) if fc_path else None)
            km_figure = km_path
            if fc_records:
                _hr("RUL Forecast (Theil–Sen)")
                for r in fc_records:
                    rul_pct = f"{r['RUL_T_eval_frac']:.2f} T_eval"
                    ci = ""
                    if r.get("RUL_CI_lo_frac") is not None:
                        ci = f"  [CI: {r['RUL_CI_lo_frac']:.2f}–{r['RUL_CI_hi_frac']:.2f}]"
                    print(f"  {r['condition']} × {r['transition']:<10}  RUL={rul_pct}{ci}  β={r['beta_TS']:.3e}")
        except ImportError:
            print("  [skip] eval_plots not available (matplotlib missing?) — use --no-figures to suppress")

    # ── Save outputs ──────────────────────────────────────────────────────
    if not args.no_save:
        ev_path = RESULTS_DIR / "eval_table.csv"
        ps_path = RESULTS_DIR / "per_seed_table.csv"
        fa_path = RESULTS_DIR / "far_summary.csv"
        ev.to_csv(ev_path, index=False)
        ps.to_csv(ps_path, index=False)
        fa.to_csv(fa_path, index=False)
        # also write full flat df (without series cols) for further pandas work
        scalar_cols = [c for c in df.columns if not c.endswith("_list")]
        df[scalar_cols].to_csv(RESULTS_DIR / "all_runs_flat.csv", index=False)
        print()
        _hr("Saved")
        print(f"  {ev_path}")
        print(f"  {ps_path}")
        print(f"  {fa_path}")
        print(f"  {RESULTS_DIR / 'all_runs_flat.csv'}")
        if km_figure:
            print(f"  {km_figure}")
            print(f"  {RESULTS_DIR / 'rul_forecast.json'}")

    print()
    _hr()
    print()
    return df   # return for interactive use


if __name__ == "__main__":
    main()
