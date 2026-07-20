#!/usr/bin/env python3
"""
Hopper deployment audit — 6-subsection pandas analysis
=======================================================
Loads all 45 *_metrics_summary.json files from results/mujoco/hopper/
into a master pandas DataFrame and produces per-subsection figures + tables.

Usage (from audit-framework - Cópia/):
    python analysis.py [--results results/mujoco/hopper] [--out hopper_analysis]
"""

import argparse
import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/mujoco/hopper",
                    help="Folder containing *_metrics_summary.json files")
parser.add_argument("--out",     default="hopper_analysis",
                    help="Output folder for figures and CSV tables")
args = parser.parse_args()

RESULTS_DIR = Path(args.results)
OUT_DIR     = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS  = ["M1", "M2", "M3"]
TRANSITION_TYPES = ["baseline", "abrupt", "gradual"]  # "baseline" = mechanism inactive (null condition)
SEEDS       = [0, 1, 2, 3, 4]
T0_GLOBAL   = 500_000          # same for every hopper run
PILOT_END   = 125_000          # K0=4, w=50k, s=25k  →  4×s + w = 225k… but pilot_end=125k in file

COND_COLORS  = {"M1": "#1f77b4", "M2": "#ff7f0e", "M3": "#2ca02c"}
TRANSITION_LS     = {"abrupt": "-", "gradual": "--", "baseline": ":"}
TRANSITION_COLORS = {"abrupt": "#e41a1c", "gradual": "#984ea3", "baseline": "#999999"}

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def window_mids(row):
    """Return array of window midpoint timesteps."""
    return np.array(row["window_starts_list"]) + int(row["w"]) // 2


def load_all(results_dir: Path) -> pd.DataFrame:
    """Load all *_metrics_summary.json into a flat DataFrame.
    List-valued fields (D_k, y_k, …) are kept as '_list' columns.
    """
    records = []
    list_keys = ["pilot_D", "D_k", "y_k", "A_deg", "A_drift", "A_watch", "window_starts"]
    for f in sorted(results_dir.glob("*_metrics_summary.json")):
        d = json.loads(f.read_text())
        row = {}
        # scalars
        for k, v in d.items():
            if not isinstance(v, (list, dict)):
                row[k] = v
        # safety margins → flat cols
        if isinstance(d.get("safety_margins"), dict):
            for sm, vals in d["safety_margins"].items():
                if isinstance(vals, dict):
                    for metric, val in vals.items():
                        row[f"sm_{sm}_{metric}"] = val
        # series
        for k in list_keys:
            row[f"{k}_list"] = d.get(k, [])
        records.append(row)

    df = pd.DataFrame(records)
    # Normalize the null-condition label: "baseline" is canonical going forward;
    # "none"/"no_drift" are pre-rename aliases still present in older result JSONs.
    if "transition" in df.columns:
        df["transition"] = df["transition"].replace({"none": "baseline", "no_drift": "baseline"})
    df = df.sort_values(["condition", "seed", "transition"]).reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# S6 — Pilot-threshold calibration
# ════════════════════════════════════════════════════════════════════════════
def subsection6(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SUBSECTION 6 — Pilot-threshold calibration")
    print("=" * 72)

    # one row per (condition, seed) — tau_warn is calibrated once, same for all drift_types
    calib = df[df["transition"] == "abrupt"].copy()
    calib = calib.sort_values(["condition", "seed"]).reset_index(drop=True)

    # derived
    calib["sigma_floor_ratio"] = calib["sigma_ref"] / calib["mu_ref"].abs()
    calib["tau_warn_log10"]    = np.log10(calib["tau_warn"])

    # ── 6a: scalar table ─────────────────────────────────────────────────
    print("\n[6a] Calibration scalars per (condition, seed)")
    cols = ["condition", "seed", "K0", "B", "mu_ref", "sigma_ref",
            "sigma_floor_ratio", "tau_warn", "tau_watch"]
    print(calib[cols].to_string(index=False))
    calib[cols].to_csv(OUT_DIR / "s6_calib_table.csv", index=False)

    # ── 6b: sigma_ref floor check ────────────────────────────────────────
    print("\n[6b] sigma_ref / |mu_ref|  (floor = 0.01 means floor was hit every run)")
    floor_check = calib.groupby("condition")["sigma_floor_ratio"].describe()
    print(floor_check.to_string())
    all_hit = np.allclose(calib["sigma_floor_ratio"], 0.01, atol=1e-4)
    print(f"\n  ✓ All 15 runs hit sigma floor: {all_hit}")

    # ── 6c: tau_warn instability per condition ───────────────────────────
    print("\n[6c] τ_warn statistics per condition")
    gstats = calib.groupby("condition")["tau_warn"].agg(["min", "max", "mean", "std"])
    gstats["ratio_max_min"] = gstats["max"] / gstats["min"]
    print(gstats.to_string())
    gstats.to_csv(OUT_DIR / "s6_tau_warn_stats.csv")
    overall_ratio = calib["tau_warn"].max() / calib["tau_warn"].min()
    print(f"\n  Overall τ_warn range: {calib['tau_warn'].min():.3e} – "
          f"{calib['tau_warn'].max():.3e}  ratio = {overall_ratio:.1f}×")

    # ── 6d: per-seed pilot_D values ──────────────────────────────────────
    print("\n[6d] Per-seed pilot_D[0..K0-1] and resulting τ_warn")
    all_pilot_rows = []
    for _, row in calib.iterrows():
        pDs = row["pilot_D_list"]
        print(f"  {row['condition']}_seed{int(row['seed'])}: "
              f"pilot_D = {[f'{v:.3e}' for v in pDs]}  "
              f"→ τ_warn = {row['tau_warn']:.4e}")
        for i, val in enumerate(pDs):
            all_pilot_rows.append({"condition": row["condition"],
                                   "seed": int(row["seed"]),
                                   "k": i, "pilot_D": val,
                                   "tau_warn": row["tau_warn"]})
    pilot_df = pd.DataFrame(all_pilot_rows)
    pilot_df.to_csv(OUT_DIR / "s6_pilot_D_values.csv", index=False)

    # ── Figure 6-1: τ_warn by seed & condition ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    for cond in CONDITIONS:
        sub = calib[calib["condition"] == cond]
        ax1.scatter(sub["seed"], sub["tau_warn"],
                    color=COND_COLORS[cond], s=90, zorder=3, label=cond)
        ax1.plot(sub["seed"], sub["tau_warn"],
                 color=COND_COLORS[cond], lw=1, alpha=0.5)

    ax1.set_yscale("log")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("τ_warn (log scale)")
    ax1.set_title("τ_warn per condition × seed  (K₀=4 LOO)")
    ax1.legend(title="Condition")
    ax1.grid(True, which="both", alpha=0.3)
    # annotate ratio
    ratio_txt = f"{overall_ratio:.0f}× spread\n({calib['tau_warn'].min():.1e}–{calib['tau_warn'].max():.1e})"
    ax1.text(0.02, 0.97, ratio_txt, transform=ax1.transAxes,
             va="top", fontsize=9, bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

    # ── Figure 6-2: pilot_D strip plot + τ_warn range per condition ──────
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(pilot_df))
    for i, cond in enumerate(CONDITIONS):
        sub_p = pilot_df[pilot_df["condition"] == cond]
        ax2.scatter(i + jitter[sub_p.index], sub_p["pilot_D"],
                    color=COND_COLORS[cond], alpha=0.65, s=50, label=f"{cond} pilot_D")
        # τ_warn range bar
        sub_c = calib[calib["condition"] == cond]
        tw_mean = sub_c["tau_warn"].mean()
        tw_lo   = tw_mean - sub_c["tau_warn"].min()
        tw_hi   = sub_c["tau_warn"].max() - tw_mean
        ax2.errorbar(i, tw_mean, yerr=[[tw_lo], [tw_hi]],
                     fmt="D", color="black", ms=7, capsize=6, lw=2,
                     label="τ_warn range" if i == 0 else "")

    ax2.set_yscale("log")
    ax2.set_xticks(range(len(CONDITIONS)))
    ax2.set_xticklabels(CONDITIONS)
    ax2.set_ylabel("D value (log scale)")
    ax2.set_title("Pilot D values + τ_warn range per condition")
    ax2.grid(True, which="both", alpha=0.3)
    handles, labels = ax2.get_legend_handles_labels()
    # keep one pilot_D entry per condition + the τ_warn range
    ax2.legend(handles[:4], labels[:4], fontsize=8)

    fig.suptitle("S6 — Pilot-threshold calibration  (K₀=4, hopper, 15 seeds×conditions)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s6_calibration.png")


# ════════════════════════════════════════════════════════════════════════════
# S1 — Drift channel alone
# ════════════════════════════════════════════════════════════════════════════
def subsection1(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SUBSECTION 1 — Drift channel alone")
    print("=" * 72)

    # ── 1 table ───────────────────────────────────────────────────────────
    t_tbl = df[["condition","seed","transition","T_drift","T_watch","T_warn",
                "t0","timely_warn","delta_delay","tau_warn","tau_watch"]].copy()
    t_tbl["alarm_before_t0"] = t_tbl["T_drift"] < t_tbl["t0"]
    t_tbl["alarm_fires"]     = t_tbl["T_drift"].notna()
    print("\n[1a] T_drift / T_warn full table")
    print(t_tbl.to_string(index=False))
    t_tbl.to_csv(OUT_DIR / "s1_T_drift_table.csv", index=False)

    print("\n[1b] Alarms before t0 per condition × transition")
    grp = t_tbl.groupby(["condition", "transition"]).apply(
        lambda g: pd.Series({
            "n_fires":        g["alarm_fires"].sum(),
            "n_before_t0":    g["alarm_before_t0"].sum(),
            "n_after_or_none": (~g["alarm_before_t0"]).sum(),
            "T_drift_median": g["T_drift"].median(),
        })
    ).reset_index()
    print(grp.to_string(index=False))

    # ── View A: abrupt, D_k aligned at t0 ────────────────────────────────
    abrupt = df[df["transition"] == "abrupt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, cond in zip(axes, CONDITIONS):
        sub = abrupt[abrupt["condition"] == cond]
        for _, row in sub.iterrows():
            t   = window_mids(row)
            dk  = np.array(row["D_k_list"])
            rel = (t - T0_GLOBAL) / 1000
            ax.plot(rel, dk, lw=1.3, alpha=0.55, label=f"seed{int(row['seed'])}")
            ax.axhline(row["tau_warn"],  ls="--", lw=0.7, alpha=0.45,
                       color=COND_COLORS[cond])
            ax.axhline(row["tau_watch"], ls=":", lw=0.7, alpha=0.35,
                       color=COND_COLORS[cond])
        ax.axvline(0, color="black", lw=1.8, ls="-", label="t₀ (drift onset)")
        ax.set_xlabel("Steps relative to t₀ (×10³)")
        ax.set_ylabel("D_k (JSD)")
        ax.set_title(f"{cond} — Abrupt")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("S1A — D_k abrupt runs (aligned at t₀=500k, dashed=τ_warn, dotted=τ_watch)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s1a_drift_abrupt.png")

    # ── View B: gradual, aligned at pilot_end ────────────────────────────
    gradual = df[df["transition"] == "gradual"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, cond in zip(axes, CONDITIONS):
        sub = gradual[gradual["condition"] == cond]
        for _, row in sub.iterrows():
            t   = window_mids(row)
            dk  = np.array(row["D_k_list"])
            pe  = row.get("pilot_end", PILOT_END)
            rel = (t - pe) / 1000
            ax.plot(rel, dk, lw=1.3, alpha=0.55, label=f"seed{int(row['seed'])}")
            ax.axhline(row["tau_warn"], ls="--", lw=0.7, alpha=0.45,
                       color=COND_COLORS[cond])
        t0_rel = (T0_GLOBAL - (row.get("pilot_end", PILOT_END))) / 1000
        ax.axvline(0,     color="gray",  lw=1.2, ls=":", label="pilot end")
        ax.axvline(t0_rel, color="black", lw=1.8, ls="-", label="t₀ (ramp start)")
        ax.set_xlabel("Steps relative to pilot end (×10³)")
        ax.set_ylabel("D_k (JSD)")
        ax.set_title(f"{cond} — Gradual")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("S1B — D_k gradual runs (aligned at pilot_end=125k)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s1b_drift_gradual.png")

    # ── View C: baseline (null condition) ───────────────────────────────────
    none = df[df["transition"] == "baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, cond in zip(axes, CONDITIONS):
        sub = none[none["condition"] == cond]
        for _, row in sub.iterrows():
            t   = window_mids(row)
            dk  = np.array(row["D_k_list"])
            rel = (t - T0_GLOBAL) / 1000
            fires = pd.notna(row["T_drift"]) and row["T_drift"] < T0_GLOBAL
            color = "#d62728" if fires else "#7f7f7f"
            ax.plot(rel, dk, lw=1.3, alpha=0.65, color=color,
                    label=f"seed{int(row['seed'])} {'⚠FA' if fires else 'ok'}")
            ax.axhline(row["tau_warn"], ls="--", lw=0.7, alpha=0.45,
                       color=COND_COLORS[cond])
        ax.axvline(0, color="black", lw=1.8, ls="-", label="t₀ (no actual drift)")
        ax.set_xlabel("Steps relative to t₀ (×10³)")
        ax.set_ylabel("D_k (JSD)")
        ax.set_title(f"{cond} — Baseline (null)\n(red = false alarm)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("S1C — D_k baseline runs as null reference",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s1c_drift_nodrift.png")

    # ── View D: median + IQR overlay, all transition types per condition ───────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cond in zip(axes, CONDITIONS):
        for dtype in TRANSITION_TYPES:
            sub = df[(df["condition"] == cond) & (df["transition"] == dtype)]
            traces = []
            for _, row in sub.iterrows():
                t   = window_mids(row)
                dk  = np.array(row["D_k_list"])
                rel = (t - T0_GLOBAL) / 1000
                traces.append((rel, dk))
            if not traces:
                continue
            min_len = min(len(t[1]) for t in traces)
            xs = traces[0][0][:min_len]
            ys = np.array([t[1][:min_len] for t in traces])
            ax.plot(xs, np.median(ys, axis=0),
                    ls=TRANSITION_LS[dtype], lw=2,
                    color=TRANSITION_COLORS[dtype], label=dtype)
            ax.fill_between(xs, np.percentile(ys, 25, axis=0),
                            np.percentile(ys, 75, axis=0),
                            color=TRANSITION_COLORS[dtype], alpha=0.12)
        ax.axvline(0, color="black", lw=1.5, ls="-")
        ax.set_xlabel("Steps relative to t₀ (×10³)")
        ax.set_ylabel("D_k — median ± IQR")
        ax.set_title(cond)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("S1D — D_k median traces by transition type (all 5 seeds per condition)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s1d_drift_overlay.png")

    # ── View E: A_drift heatmap (condition × seed × time) ────────────────
    for dtype in ["abrupt", "baseline"]:
        sub = df[df["transition"] == dtype]
        # build matrix: rows = runs, cols = windows (up to shortest)
        mats = {}
        for cond in CONDITIONS:
            rows = sub[sub["condition"] == cond]
            arrays = [np.array(r["A_drift_list"]) for _, r in rows.iterrows()]
            min_len = min(len(a) for a in arrays)
            mats[cond] = np.array([a[:min_len] for a in arrays])  # shape (5, T)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for ax, cond in zip(axes, CONDITIONS):
            mat = mats[cond]
            im = ax.imshow(mat, aspect="auto", cmap="Reds",
                           vmin=0, vmax=1, interpolation="nearest")
            # mark t0 window index
            # window_starts for first run
            sample_row = sub[sub["condition"] == cond].iloc[0]
            ws  = np.array(sample_row["window_starts_list"])[:mat.shape[1]]
            t0_idx = np.searchsorted(ws, T0_GLOBAL) - 1
            ax.axvline(t0_idx, color="black", lw=2, ls="--")
            ax.set_xlabel("Window index")
            ax.set_ylabel("Seed")
            ax.set_yticks(range(5))
            ax.set_yticklabels([f"s{s}" for s in SEEDS])
            ax.set_title(f"{cond}")
            plt.colorbar(im, ax=ax, fraction=0.03)
        fig.suptitle(f"S1E — A_drift alarm heatmap ({dtype}, dashed = t₀ window)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        save(fig, f"s1e_adrift_heatmap_{dtype}.png")


# ════════════════════════════════════════════════════════════════════════════
# S2 — Baseline false-alarm analysis
# ════════════════════════════════════════════════════════════════════════════
def subsection2(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SUBSECTION 2 — Baseline false-alarm analysis")
    print("=" * 72)

    nd = df[df["transition"] == "baseline"].copy()
    nd["alarm_fires"]  = nd["T_drift"].notna()
    nd["false_alarm"]  = nd["T_drift"] < nd["t0"]
    nd["T_drift_rel"]  = nd["T_drift"] - nd["t0"]   # negative = fires before drift

    print("\n[2a] Baseline run table")
    cols = ["condition","seed","T_drift","T_drift_rel","alarm_fires","false_alarm",
            "tau_warn","tau_watch","T_hard"]
    print(nd[cols].to_string(index=False))
    nd[cols].to_csv(OUT_DIR / "s2_baseline_table.csv", index=False)

    print("\n[2b] Per-condition false-alarm specificity")
    spec = nd.groupby("condition").apply(
        lambda g: pd.Series({
            "n_seeds":           len(g),
            "n_alarms":          int(g["alarm_fires"].sum()),
            "n_false_alarms":    int(g["false_alarm"].sum()),
            "FAR":               g["false_alarm"].mean(),
            "T_drift_min":       g["T_drift"].min(),
            "T_drift_median":    g["T_drift"].median(),
            "T_drift_mean":      g["T_drift"].mean(),
        })
    ).reset_index()
    print(spec.to_string(index=False))
    spec.to_csv(OUT_DIR / "s2_specificity.csv", index=False)

    print(f"\n  Overall FAR = {nd['false_alarm'].mean():.2f}  "
          f"({int(nd['false_alarm'].sum())}/{len(nd)} baseline runs)")

    # ── Figure 2-1: D_k baseline, false alarms highlighted ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, cond in zip(axes, CONDITIONS):
        sub = nd[nd["condition"] == cond]
        for _, row in sub.iterrows():
            t   = window_mids(row)
            dk  = np.array(row["D_k_list"])
            rel = (t - T0_GLOBAL) / 1000
            c   = "#d62728" if row["false_alarm"] else "#7f7f7f"
            lbl = f"seed{int(row['seed'])} ({'FA!' if row['false_alarm'] else 'ok'})"
            ax.plot(rel, dk, lw=1.5, alpha=0.75, color=c, label=lbl)
            ax.axhline(row["tau_warn"], ls="--", lw=0.8, alpha=0.5,
                       color=COND_COLORS[cond])
        ax.axvline(0, color="black", lw=1.8, ls="-", label="t₀ (no drift)")
        ax.set_xlabel("Steps rel. to t₀ (×10³)")
        ax.set_ylabel("D_k")
        ax.set_title(f"{cond} — Baseline\n(red = false alarm before t₀)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("S2 — False-alarm audit: D_k in baseline runs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s2_false_alarms.png")

    # ── Figure 2-2: τ_warn vs T_drift scatter (baseline) ─────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for cond in CONDITIONS:
        sub = nd[nd["condition"] == cond]
        ax.scatter(sub["tau_warn"], sub["T_drift"] / 1000,
                   color=COND_COLORS[cond], s=90, label=cond, zorder=3)
        for _, r in sub.iterrows():
            if pd.notna(r["T_drift"]):
                ax.annotate(f"s{int(r['seed'])}",
                            (r["tau_warn"], r["T_drift"] / 1000),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(T0_GLOBAL / 1000, color="black", ls="--", lw=1.5, label="t₀=500k")
    ax.set_xscale("log")
    ax.set_xlabel("τ_warn (log scale)")
    ax.set_ylabel("T_drift (×10³)")
    ax.set_title("S2 — τ_warn vs T_drift (baseline runs)\nPoints below dashed line = false alarm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "s2_tau_warn_vs_Tdrift.png")


# ════════════════════════════════════════════════════════════════════════════
# S3 — Hard-failure log audit
# ════════════════════════════════════════════════════════════════════════════
def subsection3(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SUBSECTION 3 — Hard-failure log audit")
    print("=" * 72)

    hf = df[["condition","seed","transition","T_hard","T_comp","T_comp_obs",
              "delta","hard_failure_counts","t0"]].copy()
    hf["T_hard_rel"]      = hf["T_hard"] - hf["t0"]
    hf["hard_fires"]      = hf["T_hard"].notna()
    hf["hard_before_t0"]  = hf["T_hard"] < hf["t0"]
    hf["comp_fires"]      = hf["T_comp"].notna()

    print("\n[3a] Full T_hard table")
    print(hf.to_string(index=False))
    hf.to_csv(OUT_DIR / "s3_hard_failure_table.csv", index=False)

    # consistency: T_hard should be identical across drift_types for same (condition, seed)
    print("\n[3b] T_hard consistency across transition types per (condition, seed)")
    pivot = df.pivot_table(index=["condition","seed"],
                           columns="transition", values="T_hard", aggfunc="first")
    print(pivot.to_string())
    pivot.to_csv(OUT_DIR / "s3_Thard_pivot.csv")

    print("\n[3c] Consistency check")
    for cond in CONDITIONS:
        for seed in SEEDS:
            vals = df[(df["condition"]==cond) & (df["seed"]==seed)]["T_hard"].values
            unique_str = {str(v) for v in vals}
            ok = len(unique_str) == 1
            mark = "✓" if ok else "⚠ VARIES"
            print(f"  {cond}_seed{seed}: {mark}  {sorted(unique_str)}")

    print("\n[3d] T_hard stats per condition (abrupt runs only)")
    ab = hf[hf["transition"] == "abrupt"]
    g3 = ab.groupby("condition").agg(
        n_fires    =("hard_fires",     "sum"),
        n_before_t0=("hard_before_t0", "sum"),
        T_hard_min =("T_hard",         "min"),
        T_hard_max =("T_hard",         "max"),
        T_hard_mean=("T_hard",         "mean"),
    ).reset_index()
    print(g3.to_string(index=False))

    # ── Figure 3: T_hard scatter ──────────────────────────────────────────
    ab = hf[hf["transition"] == "abrupt"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITIONS:
        sub = ab[ab["condition"] == cond]
        ax.scatter(sub["seed"], sub["T_hard"] / 1000,
                   color=COND_COLORS[cond], s=110, zorder=3, label=cond)
        for _, row in sub.iterrows():
            if pd.notna(row["T_hard"]):
                ax.annotate(f"{row['T_hard']/1000:.0f}k",
                            (row["seed"], row["T_hard"] / 1000),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(T0_GLOBAL / 1000, color="black", ls="--", lw=1.8, label="t₀ = 500k")
    ax.set_xlabel("Seed")
    ax.set_ylabel("T_hard (×10³ steps)")
    ax.set_title("S3 — T_hard by condition × seed  (abrupt runs)\n"
                 "hard_failure_counts=False → T_hard is diagnostic only")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "s3_Thard_scatter.png")

    # semantic audit: why delta=0 for most runs?
    print("\n[3e] Semantic audit: delta=0 breakdown")
    for _, row in hf.iterrows():
        if row["delta"] == 1:
            reason = "✓ COMPOSITE DETECTED (delta=1)"
        elif pd.notna(row["T_comp"]):
            reason = "T_comp fires but delta unclear"
        elif pd.isna(row["T_hard"]) and pd.isna(row["T_comp"]):
            reason = "no failure (T_hard=null)"
        elif pd.notna(row["T_hard"]) and pd.isna(row["T_comp"]):
            reason = "T_hard only → excluded (hard_failure_counts=False)"
        else:
            reason = "other"
        print(f"  {row['condition']}_seed{int(row['seed'])}_{row['transition']}: {reason}")


# ════════════════════════════════════════════════════════════════════════════
# S4 — Reward / degradation channel
# ════════════════════════════════════════════════════════════════════════════
def subsection4(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SUBSECTION 4 — Reward / degradation channel")
    print("=" * 72)

    rew = df[["condition","seed","transition","mu_ref","sigma_ref","beta_TS",
               "T_deg","T_soft","delta"]].copy()
    rew["sigma_floor_ratio"] = rew["sigma_ref"] / rew["mu_ref"].abs()

    print("\n[4a] Reward/degradation scalars")
    print(rew.to_string(index=False))
    rew.to_csv(OUT_DIR / "s4_reward_table.csv", index=False)

    print("\n[4b] beta_TS by condition × transition")
    g4b = df.groupby(["condition","transition"])["beta_TS"].agg(["min","max","mean","std"])
    print(g4b.to_string())
    g4b.to_csv(OUT_DIR / "s4_beta_TS_stats.csv")

    print("\n[4c] T_deg fires per condition × transition")
    g4c = df.groupby(["condition","transition"]).apply(
        lambda g: pd.Series({
            "n_T_deg_fires": g["T_deg"].notna().sum(),
            "n_T_soft_fires": g["T_soft"].notna().sum(),
        })
    ).reset_index()
    print(g4c.to_string(index=False))

    # ── Figure 4A: y_k abrupt, aligned at t0 ─────────────────────────────
    abrupt = df[df["transition"] == "abrupt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, cond in zip(axes, CONDITIONS):
        sub = abrupt[abrupt["condition"] == cond]
        for _, row in sub.iterrows():
            t   = window_mids(row)
            yk  = np.array(row["y_k_list"])
            rel = (t - T0_GLOBAL) / 1000
            ax.plot(rel, yk, lw=1.3, alpha=0.55, label=f"seed{int(row['seed'])}")
        # mu_ref band (mean ± sigma_ref) for last row
        ax.axhline(row["mu_ref"], ls="--", lw=1, color="gray", alpha=0.7)
        ax.axhspan(row["mu_ref"] - row["sigma_ref"],
                   row["mu_ref"] + row["sigma_ref"],
                   color="gray", alpha=0.08, label="±σ_ref")
        ax.axvline(0, color="black", lw=1.8, ls="-")
        ax.set_xlabel("Steps rel. to t₀ (×10³)")
        ax.set_ylabel("y_k (mean window reward)")
        ax.set_title(f"{cond} — y_k abrupt")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("S4A — y_k reward traces (abrupt, aligned at t₀)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s4a_reward_abrupt.png")

    # ── Figure 4B: y_k gradual ────────────────────────────────────────────
    gradual = df[df["transition"] == "gradual"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, cond in zip(axes, CONDITIONS):
        sub = gradual[gradual["condition"] == cond]
        for _, row in sub.iterrows():
            t   = window_mids(row)
            yk  = np.array(row["y_k_list"])
            rel = (t - T0_GLOBAL) / 1000
            ax.plot(rel, yk, lw=1.3, alpha=0.55, label=f"seed{int(row['seed'])}")
        ax.axhline(row["mu_ref"], ls="--", lw=1, color="gray", alpha=0.7)
        ax.axvline(0, color="black", lw=1.8, ls="-")
        ax.set_xlabel("Steps rel. to t₀ (×10³)")
        ax.set_ylabel("y_k (mean window reward)")
        ax.set_title(f"{cond} — y_k gradual")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("S4B — y_k reward traces (gradual, aligned at t₀)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s4b_reward_gradual.png")

    # ── Figure 4C: A_deg heatmap (abrupt) ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    abrupt2 = df[df["transition"] == "abrupt"]
    for ax, cond in zip(axes, CONDITIONS):
        sub = abrupt2[abrupt2["condition"] == cond]
        arrays  = [np.array(r["A_deg_list"]) for _, r in sub.iterrows()]
        min_len = min(len(a) for a in arrays)
        mat = np.array([a[:min_len] for a in arrays])
        im = ax.imshow(mat, aspect="auto", cmap="Purples",
                       vmin=0, vmax=1, interpolation="nearest")
        # mark t0
        sample_row = sub.iloc[0]
        ws  = np.array(sample_row["window_starts_list"])[:min_len]
        t0_idx = int(np.searchsorted(ws, T0_GLOBAL)) - 1
        ax.axvline(t0_idx, color="black", lw=2, ls="--")
        ax.set_xlabel("Window index")
        ax.set_ylabel("Seed")
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"s{s}" for s in SEEDS])
        ax.set_title(f"{cond}")
        plt.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("S4C — A_deg alarm heatmap (abrupt, dashed = t₀ window)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "s4c_adeg_heatmap.png")

    # ── Figure 4D: beta_TS distribution per condition ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(CONDITIONS):
        for dtype in TRANSITION_TYPES:
            sub = df[(df["condition"] == cond) & (df["transition"] == dtype)]
            betas = sub["beta_TS"].values
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(betas))
            x_pos = i * 3 + TRANSITION_TYPES.index(dtype)
            ax.scatter(x_pos + jitter, betas,
                       color=TRANSITION_COLORS[dtype], alpha=0.7, s=60)
    xticks = [i * 3 + 1 for i in range(3)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(CONDITIONS)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_ylabel("β_TS (reward slope)")
    ax.set_title("S4D — β_TS by condition × transition type")
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=TRANSITION_COLORS[dt], markersize=9, label=dt)
               for dt in TRANSITION_TYPES]
    ax.legend(handles=handles)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "s4d_beta_TS.png")


# ════════════════════════════════════════════════════════════════════════════
# S5 — Composite failure analysis
# ════════════════════════════════════════════════════════════════════════════
def subsection5(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SUBSECTION 5 — Composite failure analysis")
    print("=" * 72)

    comp = df[["condition","seed","transition","T_hard","T_soft","T_comp",
               "T_comp_obs","delta","hard_failure_counts",
               "T_drift","T_deg","T_warn","timely_warn","delta_delay"]].copy()
    comp["comp_fires"]  = comp["T_comp"].notna()
    comp["T_comp_rel"]  = comp["T_comp"] - T0_GLOBAL

    print("\n[5a] Full composite table")
    print(comp.to_string(index=False))
    comp.to_csv(OUT_DIR / "s5_composite_table.csv", index=False)

    print("\n[5b] delta=1 runs only")
    print(comp[comp["delta"] == 1][
        ["condition","seed","transition","T_hard","T_soft","T_comp",
         "T_warn","timely_warn","delta_delay"]].to_string(index=False))

    print("\n[5c] Detection summary per condition × transition")
    g5 = comp.groupby(["condition","transition"]).apply(
        lambda g: pd.Series({
            "n_delta1":       int((g["delta"] == 1).sum()),
            "n_comp_fires":   int(g["comp_fires"].sum()),
            "n_timely":       int(g["timely_warn"].sum()),
            "delta_delay_mean": g["delta_delay"].mean(),
            "T_comp_mean":    g["T_comp"].mean(),
        })
    ).reset_index()
    print(g5.to_string(index=False))
    g5.to_csv(OUT_DIR / "s5_detection_summary.csv", index=False)

    # ── Figure 5: T_comp vs T_drift scatter ───────────────────────────────
    delta1 = comp[comp["delta"] == 1]
    fig, ax = plt.subplots(figsize=(8, 7))
    for cond in CONDITIONS:
        sub = delta1[delta1["condition"] == cond]
        ax.scatter(sub["T_drift"] / 1000, sub["T_comp"] / 1000,
                   color=COND_COLORS[cond], s=120, label=cond, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['condition']}s{int(r['seed'])}\n{r['transition']}",
                        (r["T_drift"] / 1000, r["T_comp"] / 1000),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    # diagonal
    all_finite = pd.concat([delta1["T_drift"], delta1["T_comp"]]).dropna()
    if len(all_finite):
        lo = all_finite.min() / 1000 - 10
        hi = all_finite.max() / 1000 + 10
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="T_drift = T_comp")
    ax.axvline(T0_GLOBAL / 1000, color="black", lw=1.5, ls=":", label="t₀=500k")
    ax.set_xlabel("T_drift (×10³)")
    ax.set_ylabel("T_comp (×10³)")
    ax.set_title("S5 — T_comp vs T_drift (delta=1 runs only)\n"
                 "Only 4 runs reach composite failure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "s5_composite.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Loading metrics from {RESULTS_DIR} …")
    df = load_all(RESULTS_DIR)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns\n")

    subsection6(df)
    subsection1(df)
    subsection2(df)
    subsection3(df)
    subsection4(df)
    subsection5(df)

    print(f"\n{'='*72}")
    print(f"All outputs in: {OUT_DIR}/")
    print("  Figures: s1a–s1e, s2, s3, s4a–s4d, s5, s6")
    print("  Tables:  s1_T_drift_table.csv, s2_baseline_table.csv,")
    print("           s2_specificity.csv, s3_hard_failure_table.csv,")
    print("           s3_Thard_pivot.csv, s4_reward_table.csv,")
    print("           s4_beta_TS_stats.csv, s5_composite_table.csv,")
    print("           s5_detection_summary.csv, s6_calib_table.csv,")
    print("           s6_tau_warn_stats.csv, s6_pilot_D_values.csv")
