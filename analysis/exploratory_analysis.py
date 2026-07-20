"""
exploratory_analysis.py — Auxiliary EDA for deployment audit data
==================================================================
SCOPE: EXPLORATORY / INTERPRETIVE only.
This script does NOT compute any confirmatory metric and does NOT feed back
into the primary audit framework. Its findings belong in the thesis EDA section,
clearly separated from the confirmatory evaluation in deploy_results.py.

See tuning_journal.md §Confirmatory vs Exploratory boundary for the rationale.

Usage (from audit-framework/):
    python exploratory_analysis.py --domain hopper
    python exploratory_analysis.py --domain hopper --transition baseline abrupt
    python exploratory_analysis.py --domain hopper --no-plots   # tables only

Outputs saved to {results_dir}/eda/:
    01_pilot_calibration.png    Pilot D distribution + tau_warn spread
    02_null_behavior.png        D_k / y_k under baseline
    03_drift_shape.png          D_k timeseries by transition type
    04_action_response.png      Action features co-plotted with D_k
    05_reward_vs_drift.png      y_k timeseries with threshold bands
    eda_summary.csv             Tabular summary of all EDA statistics
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)

parser = argparse.ArgumentParser(description="EDA for deployment audit results")
parser.add_argument("--domain",   required=True,
                    help="Domain name (must match a config in --configs dir)")
parser.add_argument("--configs",  default="./configs")
parser.add_argument("--transition", "--drift", dest="transition", nargs="+",
                    default=["baseline", "abrupt", "gradual"],
                    help="Transition types to include (--drift accepted as a legacy alias)")
parser.add_argument("--no-plots", action="store_true",
                    help="Skip plot generation (tables only)")
args = parser.parse_args()

def _load_json(path):
    with open(path) as f:
        return json.load(f)

cfg_path = Path(args.configs) / f"{args.domain}.json"
if not cfg_path.exists():
    sys.exit(f"Config not found: {cfg_path}")

cfg         = _load_json(cfg_path)
instr_cfg   = _load_json(Path(args.configs) / "instrument.json")
RESULTS_DIR = Path(cfg["results_dir"])
T_EVAL      = int(cfg.get("T_eval", 1_000_000))
T0          = T_EVAL // 2
K0          = int(instr_cfg["K0"])
C_FAIL      = float(instr_cfg["c_fail"])
W           = int(cfg.get("w", 50000))
S           = int(cfg.get("s", 25000))
CONDITIONS  = cfg.get("conditions", ["M1", "M2", "M3"])
EDA_DIR     = RESULTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

TRANSITION_TYPES = args.transition
COLORS = {"baseline": "#2196F3", "abrupt": "#F44336", "gradual": "#4CAF50"}
COND_MARKERS = {"M1": "o", "M2": "s", "M3": "^"}


# ════════════════════════════════════════════════════════════════════════════
# ETL — identical shape to deploy_results.py but keeps list fields for EDA
# ════════════════════════════════════════════════════════════════════════════

SERIES_KEYS = ["pilot_D", "D_k", "y_k", "A_deg", "A_drift", "A_watch",
               "window_starts"]
AUX_SERIES  = ["act_mean_norm", "act_std_norm", "act_smoothness"]

def load_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*_metrics_summary.json"))
    if not files:
        sys.exit(f"No *_metrics_summary.json found in {results_dir}")
    records = []
    for f in files:
        d = json.loads(f.read_text())
        row = {}
        for k, v in d.items():
            if isinstance(v, list) and k in SERIES_KEYS:
                row[f"{k}_list"] = v
            elif isinstance(v, dict) and k == "aux_action_windows":
                # Flatten auxiliary action windows
                for ak, av in v.items():
                    if ak in AUX_SERIES:
                        row[f"{ak}_list"] = av
            elif not isinstance(v, (list, dict)):
                row[k] = v
        records.append(row)
    df = pd.DataFrame(records)
    # Normalize the null-condition label: "baseline" is canonical going forward;
    # "none"/"no_drift" are pre-rename aliases still present in older result JSONs.
    if "transition" in df.columns:
        df["transition"] = df["transition"].replace({"none": "baseline", "no_drift": "baseline"})
    for col in ["condition", "transition"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df = df.sort_values(["condition", "transition", "seed"]).reset_index(drop=True)
    return df

df = load_results(RESULTS_DIR)
print(f"\nLoaded {len(df)} runs from {RESULTS_DIR}")
print(f"Conditions: {sorted(df['condition'].unique())}  |  "
      f"Transition types: {sorted(df['transition'].unique())}\n")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Pilot calibration stability
# ════════════════════════════════════════════════════════════════════════════

def section1_pilot_calibration(df: pd.DataFrame):
    """
    Inspect pilot_D spread (K0 LOO values) and resulting tau_warn per seed.
    Purpose: characterise calibration stability — are some seeds dramatically
    different from others? Are pilot_D values clustering near zero?
    """
    print("═" * 72)
    print("§1  PILOT CALIBRATION STABILITY  [EDA — exploratory]")
    print("═" * 72)

    # Flatten pilot_D into per-observation rows
    rows = []
    for _, r in df.iterrows():
        pdl = r.get("pilot_D_list", [])
        for k, d in enumerate(pdl):
            rows.append({
                "condition":  r["condition"],
                "seed":       r["seed"],
                "transition": r["transition"],
                "k":          k,
                "pilot_D":    d,
                "tau_warn":   r.get("tau_warn", np.nan),
                "tau_warn_floor": r.get("tau_warn_floor", 0.0),
            })
    pilot_df = pd.DataFrame(rows)

    # Summary table: tau_warn stats per condition (baseline only)
    nd = df[df["transition"] == "baseline"] if "baseline" in df["transition"].values else df
    print("\ntau_warn statistics by condition (baseline seeds):")
    for cond in sorted(nd["condition"].unique()):
        sub = nd[nd["condition"] == cond]["tau_warn"].dropna()
        floor_col = nd[nd["condition"] == cond].get("tau_warn_floor", pd.Series(dtype=float))
        floor_val = floor_col.dropna() if hasattr(floor_col, "dropna") else pd.Series(dtype=float)
        floor = float(floor_val.iloc[0]) if len(floor_val) else 0.0
        above = (sub > floor + 1e-12).sum()
        print(f"  {cond}: n={len(sub)}  "
              f"min={sub.min():.3e}  median={sub.median():.3e}  max={sub.max():.3e}  "
              f"above_floor={above}/{len(sub)}")

    print("\npilot_D spread per condition (K0 LOO values, all transition types pooled):")
    for cond in sorted(pilot_df["condition"].unique()):
        sub = pilot_df[pilot_df["condition"] == cond]["pilot_D"].dropna()
        print(f"  {cond}: n={len(sub)}  "
              f"min={sub.min():.3e}  Q25={sub.quantile(0.25):.3e}  "
              f"median={sub.median():.3e}  Q75={sub.quantile(0.75):.3e}  "
              f"max={sub.max():.3e}")

    if args.no_plots or pilot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("§1 Pilot Calibration Stability  [EDA — exploratory]",
                 fontsize=11, style="italic")

    # Left: pilot_D distribution per condition (violin)
    ax = axes[0]
    conds = sorted(pilot_df["condition"].unique())
    data  = [pilot_df[pilot_df["condition"] == c]["pilot_D"].dropna().values
             for c in conds]
    vp = ax.violinplot(data, positions=range(len(conds)), showmedians=True)
    ax.set_xticks(range(len(conds)));  ax.set_xticklabels(conds)
    ax.set_ylabel("pilot_D  (LOO JSD)");  ax.set_xlabel("Condition")
    ax.set_title("Pilot D distribution\n(K0 LOO values, all transition types)")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Right: tau_warn per seed (baseline) with floor line
    ax2 = axes[1]
    for cond in sorted(nd["condition"].unique()):
        sub = nd[nd["condition"] == cond]
        ax2.scatter(sub["seed"], sub["tau_warn"],
                    label=cond, marker=COND_MARKERS.get(cond, "o"), s=60)
    floor_col  = nd.get("tau_warn_floor", pd.Series(dtype=float))
    floor_vals = floor_col.dropna() if hasattr(floor_col, "dropna") else pd.Series(dtype=float)
    if len(floor_vals):
        ax2.axhline(float(floor_vals.iloc[0]), ls="--", color="gray",
                    lw=1.2, label=f"floor={float(floor_vals.iloc[0]):.2e}")
    ax2.set_xlabel("Seed");  ax2.set_ylabel("tau_warn")
    ax2.set_title("tau_warn per seed (baseline)\nvs floor")
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = EDA_DIR / "01_pilot_calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Null-behavior under baseline
# ════════════════════════════════════════════════════════════════════════════

def section2_null_behavior(df: pd.DataFrame):
    """
    Inspect D_k and y_k distributions under baseline to characterise the
    'nominal noise floor' — the natural variability the framework must sit above.
    Helps explain FAR and calibration artifact patterns.
    """
    print("\n" + "═" * 72)
    print("§2  NULL-BEHAVIOR UNDER BASELINE  [EDA — exploratory]")
    print("═" * 72)

    nd = df[df["transition"] == "baseline"] if "baseline" in df["transition"].values else df.iloc[0:0]

    # Flatten post-pilot D_k (exclude pilot windows K0)
    dk_rows = []
    for _, r in nd.iterrows():
        dk  = r.get("D_k_list", [])
        sw  = r.get("window_starts_list", [])
        tw  = r.get("tau_warn", None)
        if not dk:
            continue
        post_dk = [(d, s) for d, s in zip(dk[K0:], sw[K0:])] if len(sw) > K0 else []
        for d, s in post_dk:
            dk_rows.append({
                "condition": r["condition"], "seed": r["seed"],
                "step": s, "D_k": d, "tau_warn": tw,
            })
    dk_df = pd.DataFrame(dk_rows)

    if not dk_df.empty:
        print("\nPost-pilot D_k statistics per condition (baseline):")
        for cond in sorted(dk_df["condition"].unique()):
            sub = dk_df[dk_df["condition"] == cond]["D_k"]
            print(f"  {cond}: n={len(sub)}  "
                  f"min={sub.min():.3e}  Q50={sub.median():.3e}  "
                  f"Q95={sub.quantile(0.95):.3e}  max={sub.max():.3e}")

    # FAR breakdown per seed
    if "T_warn" in nd.columns and "delta" in nd.columns:
        print("\nBaseline-condition false-alarm breakdown per condition × seed:")
        for cond in sorted(nd["condition"].unique()):
            sub = nd[nd["condition"] == cond].sort_values("seed")
            for _, r in sub.iterrows():
                fa = pd.notna(r.get("T_warn")) and r.get("delta", 1) == 0
                print(f"  {cond} seed={int(r['seed'])}:  "
                      f"tau_warn_raw={r.get('tau_warn_raw', float('nan')):.3e}  "
                      f"tau_warn={r.get('tau_warn', float('nan')):.3e}  "
                      f"T_warn={int(r['T_warn']) if pd.notna(r.get('T_warn')) else 'None'}  "
                      f"FA={'YES' if fa else 'no'}")

    if args.no_plots or dk_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("§2 Null-behavior (baseline)  [EDA — exploratory]",
                 fontsize=11, style="italic")

    # Left: D_k violin per condition with tau_warn overlay
    ax = axes[0]
    conds = sorted(dk_df["condition"].unique())
    data  = [dk_df[dk_df["condition"] == c]["D_k"].values for c in conds]
    ax.violinplot(data, positions=range(len(conds)), showmedians=True)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
    ax.set_ylabel("D_k  (post-pilot, baseline)")
    ax.set_title("D_k distribution under baseline\n(nominal noise floor)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Overlay per-seed tau_warn
    if "tau_warn" in nd.columns:
        for i, cond in enumerate(conds):
            tws = nd[nd["condition"] == cond]["tau_warn"].dropna()
            for tw in tws:
                ax.plot([i - 0.35, i + 0.35], [tw, tw],
                        color="red", lw=0.8, alpha=0.6)

    # Right: y_k timeseries per seed (baseline, M1 example)
    ax2 = axes[1]
    primary_cond = conds[0]
    sub_nd = nd[nd["condition"] == primary_cond]
    for _, r in sub_nd.iterrows():
        yk  = r.get("y_k_list", [])
        sw  = r.get("window_starts_list", [])
        mu  = r.get("mu_ref", None)
        sig = r.get("sigma_ref", None)
        if not yk:
            continue
        ax2.plot(sw, yk, alpha=0.5, lw=0.9, label=f"s={int(r['seed'])}")
        if mu is not None and sig is not None:
            ax2.axhline(mu - C_FAIL * sig, ls=":", color="gray", lw=0.7)
    ax2.set_xlabel("Window start (steps)")
    ax2.set_ylabel("y_k  (mean reward / window)")
    ax2.set_title(f"y_k timeseries under baseline\n({primary_cond})")
    ax2.legend(fontsize=7, ncol=3)

    plt.tight_layout()
    out = EDA_DIR / "02_null_behavior.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Drift-shape inspection
# ════════════════════════════════════════════════════════════════════════════

def section3_drift_shape(df: pd.DataFrame):
    """
    Compare D_k timeseries across transition schedules to characterise:
    - whether an abrupt transition produces a step-change or gradual rise in D_k
    - whether a gradual transition gives a slower accumulation
    - magnitude of post-transition D_k relative to tau_warn
    """
    print("\n" + "═" * 72)
    print("§3  DRIFT-SHAPE INSPECTION  [EDA — exploratory]")
    print("═" * 72)

    # Post-transition D_k stats for abrupt / gradual
    for transition in ["abrupt", "gradual"]:
        sub = df[df["transition"] == transition] if transition in df["transition"].values else df.iloc[0:0]
        if sub.empty:
            continue
        print(f"\nPost-transition D_k statistics ({transition}):")
        for cond in sorted(sub["condition"].unique()):
            cs = sub[sub["condition"] == cond]
            post_dk = []
            for _, r in cs.iterrows():
                dk = r.get("D_k_list", [])
                sw = r.get("window_starts_list", [])
                post_dk.extend([d for d, s in zip(dk, sw) if s >= T0])
            if post_dk:
                arr = np.array(post_dk)
                print(f"  {cond}: n_windows={len(arr)}  "
                      f"Q50={np.median(arr):.3e}  Q95={np.quantile(arr,0.95):.3e}  "
                      f"max={arr.max():.3e}")

    if args.no_plots:
        return

    conds_to_plot = [c for c in CONDITIONS if c in df["condition"].values]
    n_rows = len(conds_to_plot)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.5 * n_rows), squeeze=False)
    fig.suptitle("§3 Drift-shape inspection  [EDA — exploratory]",
                 fontsize=11, style="italic")

    for i, cond in enumerate(conds_to_plot):
        for j, dtype in enumerate(["abrupt", "gradual"]):
            ax = axes[i][j]
            sub = df[(df["condition"] == cond) & (df["transition"] == dtype)]
            for _, r in sub.iterrows():
                dk = r.get("D_k_list", [])
                sw = r.get("window_starts_list", [])
                tw = r.get("tau_warn", None)
                if not dk:
                    continue
                ax.plot(sw, dk, alpha=0.6, lw=1.0,
                        label=f"s={int(r['seed'])}")
                if tw is not None:
                    ax.axhline(tw, ls="--", lw=0.8, color="red", alpha=0.5)
            ax.axvline(T0, ls=":", color="gray", lw=1.0, label="T0")
            ax.set_title(f"{cond} × {dtype}")
            ax.set_xlabel("Window start (steps)")
            if j == 0:
                ax.set_ylabel("D_k")
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax.legend(fontsize=7, ncol=3)

    plt.tight_layout()
    out = EDA_DIR / "03_drift_shape.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Action-response inspection
# ════════════════════════════════════════════════════════════════════════════

def section4_action_response(df: pd.DataFrame):
    """
    Co-plot D_k and action features (mean norm, smoothness) per window to inspect
    whether the policy's action distribution visibly changes when D_k rises.
    Interpretive only — action features do NOT define T_warn in the primary framework.
    """
    print("\n" + "═" * 72)
    print("§4  ACTION-RESPONSE INSPECTION  [EDA — exploratory]")
    print("═" * 72)

    has_aux = "act_mean_norm_list" in df.columns
    if not has_aux:
        print("  NOTE: aux_action_windows not found in metrics JSONs.")
        print("  Re-run audit_deploy.py to populate these fields.")
        return

    # Summary: action smoothness change across transition types
    print("\nAction smoothness stats per condition × transition type:")
    for cond in sorted(df["condition"].unique()):
        for dtype in TRANSITION_TYPES:
            sub = df[(df["condition"] == cond) & (df["transition"] == dtype)]
            if sub.empty:
                continue
            pre_vals, post_vals = [], []
            for _, r in sub.iterrows():
                sm  = r.get("act_smoothness_list", [])
                sw  = r.get("window_starts_list", [])
                for s_val, sm_val in zip(sw, sm):
                    if s_val < T0:
                        pre_vals.append(sm_val)
                    else:
                        post_vals.append(sm_val)
            if pre_vals and post_vals:
                pre  = np.mean(pre_vals)
                post = np.mean(post_vals)
                print(f"  {cond} × {dtype}: "
                      f"pre-transition smoothness={pre:.4f}  "
                      f"post-transition={post:.4f}  "
                      f"ratio={post/pre:.2f}x")

    if args.no_plots:
        return

    # Plot: D_k and act_smoothness together for the abrupt transition
    dtype = "abrupt"
    sub_all = df[df["transition"] == dtype]
    conds_to_plot = [c for c in CONDITIONS if c in sub_all["condition"].values]
    if sub_all.empty:
        return

    fig, axes = plt.subplots(len(conds_to_plot), 1,
                             figsize=(12, 3.5 * len(conds_to_plot)), squeeze=False)
    fig.suptitle("§4 Action-response inspection (abrupt transition)  [EDA — exploratory]",
                 fontsize=11, style="italic")

    for i, cond in enumerate(conds_to_plot):
        ax = axes[i][0]
        ax2 = ax.twinx()
        sub = sub_all[sub_all["condition"] == cond]
        for _, r in sub.iterrows():
            dk  = r.get("D_k_list", [])
            sw  = r.get("window_starts_list", [])
            sm  = r.get("act_smoothness_list", [])
            tw  = r.get("tau_warn", None)
            sid = int(r["seed"])
            if dk and sw:
                ax.plot(sw, dk, color="steelblue", alpha=0.5, lw=1.0)
                if tw is not None:
                    ax.axhline(tw, ls="--", color="steelblue", lw=0.7, alpha=0.4)
            if sm and sw:
                ax2.plot(sw, sm, color="darkorange", alpha=0.5, lw=1.0,
                         linestyle="-.")
        ax.axvline(T0, ls=":", color="gray", lw=1.0)
        ax.set_ylabel("D_k", color="steelblue")
        ax2.set_ylabel("act_smoothness", color="darkorange")
        ax.set_title(f"{cond} × {dtype}: D_k (blue) vs action smoothness (orange)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    axes[-1][0].set_xlabel("Window start (steps)")
    plt.tight_layout()
    out = EDA_DIR / "04_action_response.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Reward vs drift descriptive patterns
# ════════════════════════════════════════════════════════════════════════════

def section5_reward_vs_drift(df: pd.DataFrame):
    """
    Plot y_k timeseries with mu_ref ± c_fail*sigma_ref degradation bands and
    T_warn marker. Inspect whether reward stays stable despite D_k shift (JSD
    can activate before reward degrades under gradual or mild transitions).
    """
    print("\n" + "═" * 72)
    print("§5  REWARD VS DRIFT PATTERNS  [EDA — exploratory]")
    print("═" * 72)

    # Descriptive: fraction of windows below degradation threshold per transition type
    print("\nFraction of windows below mu_ref − c_fail*sigma_ref per transition:")
    for dtype in TRANSITION_TYPES:
        sub = df[df["transition"] == dtype] if dtype in df["transition"].values else df.iloc[0:0]
        total, below = 0, 0
        for _, r in sub.iterrows():
            yk  = r.get("y_k_list", [])
            mu  = r.get("mu_ref", None)
            sig = r.get("sigma_ref", None)
            sw  = r.get("window_starts_list", [])
            if not yk or mu is None or sig is None:
                continue
            thr = mu - C_FAIL * sig
            post = [y for y, s in zip(yk[K0:], sw[K0:])]
            total += len(post)
            below += sum(1 for y in post if y < thr)
        if total > 0:
            print(f"  {dtype}: {below}/{total} post-pilot windows = {below/total:.2%}")

    if args.no_plots:
        return

    conds_to_plot = [c for c in CONDITIONS if c in df["condition"].values]
    n_rows = len(conds_to_plot)
    fig, axes = plt.subplots(n_rows, len(TRANSITION_TYPES),
                             figsize=(5 * len(TRANSITION_TYPES), 3.5 * n_rows),
                             squeeze=False)
    fig.suptitle("§5 Reward vs drift patterns  [EDA — exploratory]",
                 fontsize=11, style="italic")

    for i, cond in enumerate(conds_to_plot):
        for j, dtype in enumerate(TRANSITION_TYPES):
            ax = axes[i][j]
            sub = df[(df["condition"] == cond) & (df["transition"] == dtype)]
            for _, r in sub.iterrows():
                yk  = r.get("y_k_list", [])
                sw  = r.get("window_starts_list", [])
                mu  = r.get("mu_ref", None)
                sig = r.get("sigma_ref", None)
                tw  = r.get("T_warn", None)
                delta = r.get("delta", 0)
                if not yk:
                    continue
                color = COLORS.get(dtype, "black")
                ax.plot(sw, yk, alpha=0.55, lw=1.0, color=color)
                if mu is not None and sig is not None:
                    thr = mu - C_FAIL * sig
                    ax.axhline(mu,  ls="--", color="gray",  lw=0.7, alpha=0.5)
                    ax.axhline(thr, ls=":",  color="brown", lw=0.8, alpha=0.7)
                if tw is not None:
                    ax.axvline(tw, ls="-", color="purple", lw=1.0, alpha=0.8)
            if dtype != "baseline":
                ax.axvline(T0, ls=":", color="gray", lw=1.0)
            ax.set_title(f"{cond} × {dtype}")
            if j == 0:
                ax.set_ylabel("y_k")
            if i == n_rows - 1:
                ax.set_xlabel("Window start (steps)")

    plt.tight_layout()
    out = EDA_DIR / "05_reward_vs_drift.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → saved {out}")


# ════════════════════════════════════════════════════════════════════════════
# EDA Summary table
# ════════════════════════════════════════════════════════════════════════════

def build_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact tabular summary of EDA-relevant statistics per condition × transition.
    Saved as eda_summary.csv for thesis appendix reference.
    """
    rows = []
    for cond in sorted(df["condition"].unique()):
        for dtype in TRANSITION_TYPES:
            sub = df[(df["condition"] == cond) & (df["transition"] == dtype)]
            if sub.empty:
                continue
            # pilot_D stats (first row of each cond, same across transition types for same seed)
            pilot_Ds = [d for r in sub.itertuples()
                        for d in (getattr(r, "pilot_D_list", None) or [])]
            tw_vals  = sub["tau_warn"].dropna()
            dk_post  = []
            sm_post  = []
            for _, r in sub.iterrows():
                dk = r.get("D_k_list", [])
                sw = r.get("window_starts_list", [])
                sm = r.get("act_smoothness_list", [])
                if dk:
                    dk_post.extend([d for d, s in zip(dk[K0:], sw[K0:])])
                if sm:
                    sm_post.extend([s for s, step in zip(sm[K0:], sw[K0:])])

            row = {
                "condition":           cond,
                "transition":          dtype,
                "n_runs":              len(sub),
                "pilot_D_median":      np.median(pilot_Ds) if pilot_Ds else np.nan,
                "pilot_D_max":         max(pilot_Ds) if pilot_Ds else np.nan,
                "tau_warn_median":     tw_vals.median() if len(tw_vals) else np.nan,
                "tau_warn_max":        tw_vals.max() if len(tw_vals) else np.nan,
                "Dk_post_median":      np.median(dk_post) if dk_post else np.nan,
                "Dk_post_q95":         np.quantile(dk_post, 0.95) if dk_post else np.nan,
                "act_smoothness_mean": np.mean(sm_post) if sm_post else np.nan,
            }
            if "T_warn" in sub.columns:
                n_warned = sub["T_warn"].notna().sum()
                row["Pr_warn"] = round(n_warned / len(sub), 2)
                fa = (sub["T_warn"].notna() & (sub.get("delta", pd.Series(dtype=int)) == 0)).sum()
                row["n_false_alarms"] = int(fa)
            rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def _hr(title=""):
    w = 72
    if title:
        pad = (w - len(title) - 2) // 2
        print("═" * pad + f" {title} " + "═" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


print()
_hr(f"EDA  ·  {args.domain}")
print(f"  results_dir : {RESULTS_DIR}")
print(f"  eda_dir     : {EDA_DIR}")
print(f"  T_eval={T_EVAL:,}  T0={T0:,}  K0={K0}")
print(f"  [exploratory only — not confirmatory]")
print()

section1_pilot_calibration(df)
section2_null_behavior(df)
section3_drift_shape(df)
section4_action_response(df)
section5_reward_vs_drift(df)

print("\n" + "═" * 72)
print("EDA SUMMARY TABLE")
print("═" * 72)
summary = build_eda_summary(df)
print(summary.to_string(index=False))

summary_path = EDA_DIR / "eda_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"\n  → saved {summary_path}")

print()
_hr()
print()
