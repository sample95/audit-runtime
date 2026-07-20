#!/usr/bin/env python3
"""
Supplemental analysis — fields not covered in analysis.py
  S7: Episode length fragmentation (behavioral severity proxy)
  S8: Per-dim obs statistics (mu/sigma/clip — which dims drive D_k?)
"""

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS  = Path("/sessions/determined-eloquent-curie/mnt/Accountability and Traceability in RL/audit-framework - Cópia/results/mujoco/hopper")
OUT_DIR  = Path("/sessions/determined-eloquent-curie/mnt/Accountability and Traceability in RL/audit-framework - Cópia/hopper_analysis")
OUT_DIR.mkdir(exist_ok=True)

OBS_NAMES = ['torso_z','torso_angle','thigh_angle','leg_angle','foot_angle',
             'torso_vx','torso_vz','thigh_vel','leg_vel','foot_vel','foot_ang_vel']
CONDITIONS  = ["M1","M2","M3"]
SEEDS       = [0,1,2,3,4]
TRANSITION_TYPES = ["abrupt","gradual","baseline"]
COND_COLORS = {"M1":"#1f77b4","M2":"#ff7f0e","M3":"#2ca02c"}
TRANSITION_LS    = {"abrupt":"-","gradual":"--","baseline":":"}

# ── load all 45 runs ──────────────────────────────────────────────────────────
rows = []
for f in sorted(RESULTS.glob("*_metrics_summary.json")):
    d = json.loads(f.read_text())
    ep_starts = d.get("episode_starts", [])
    T_eval    = d["T_eval"]
    if len(ep_starts) > 1:
        lens = list(np.diff(ep_starts + [T_eval]))
        mean_len   = float(np.mean(lens))
        median_len = float(np.median(lens))
        p10_len    = float(np.percentile(lens, 10))
        frac_short = float(np.mean(np.array(lens) < 500))
    else:
        mean_len = median_len = p10_len = float(T_eval)
        frac_short = 0.0

    rows.append({
        "condition"  : d["condition"],
        "seed"       : d["seed"],
        "transition" : d["transition"],
        "n_episodes" : d.get("n_episodes", 1),
        "mean_ep_len": mean_len,
        "median_ep_len": median_len,
        "p10_ep_len" : p10_len,
        "frac_short" : frac_short,
        "delta"      : d.get("delta", 0),
        "T_comp_obs" : d.get("T_comp_obs", T_eval),
        "mu"         : d["mu"],
        "sigma"      : d["sigma"],
        "clip_lo"    : d["clip_lo"],
        "clip_hi"    : d["clip_hi"],
        "mu_ref"     : d["mu_ref"],
        "sigma_ref"  : d["sigma_ref"],
        "tau_watch_eq_warn": d.get("tau_watch_eq_warn", False),
    })

df = pd.DataFrame(rows)
# "baseline" is canonical going forward; "none"/"no_drift" are pre-rename aliases
# still present in older result JSONs.
df["transition"] = df["transition"].replace({"none": "baseline", "no_drift": "baseline"})
df = df.sort_values(["condition","seed","transition"]).reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────
# S7: Episode length fragmentation
# ────────────────────────────────────────────────────────────────────────────
print("=== S7: Episode Length Fragmentation ===\n")
transition_df = df[df["transition"] != "baseline"].copy()
nd_df    = df[df["transition"] == "baseline"].copy()

# S7A: Mean episode length: abrupt/gradual transition vs baseline, per condition
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, cond in zip(axes, CONDITIONS):
    nd_sub   = nd_df[nd_df["condition"] == cond]
    ax.axhline(nd_sub["mean_ep_len"].mean(), color="black", ls=":", lw=1.5,
               label="baseline mean")

    for dt, marker in [("abrupt","o"), ("gradual","s")]:
        sub = transition_df[(transition_df["condition"]==cond) & (transition_df["transition"]==dt)]
        for _, row in sub.iterrows():
            ax.scatter(row["seed"], row["mean_ep_len"],
                       marker=marker, s=80,
                       color=("crimson" if row["delta"] else "steelblue"),
                       zorder=3,
                       label=f'{dt} ({"FAIL" if row["delta"] else "pass"})' if row["seed"]==0 else "")

    ax.set_xlabel("seed"); ax.set_ylabel("mean episode length (steps)")
    ax.set_title(f"{cond}"); ax.set_xticks(SEEDS)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

fig.suptitle("S7A — Mean episode length under transition (red=T_comp fired, blue=passed)\n"
             "Dotted = baseline mean; short episodes = policy collapse signature",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR/"s7a_episode_length_fragmentation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → s7a_episode_length_fragmentation.png")

# S7B: fraction of episodes < 500 steps  vs delta (failed vs passed)
fig, ax = plt.subplots(figsize=(8, 5))
for cond in CONDITIONS:
    sub = transition_df[transition_df["condition"]==cond]
    fail = sub[sub["delta"]==1]
    pass_ = sub[sub["delta"]==0]
    ax.scatter(fail["frac_short"], fail["mean_ep_len"],
               marker="X", s=100, color=COND_COLORS[cond], label=f"{cond} FAIL", zorder=3)
    ax.scatter(pass_["frac_short"], pass_["mean_ep_len"],
               marker="o", s=60, color=COND_COLORS[cond], alpha=0.4, label=f"{cond} pass")
ax.set_xlabel("fraction of episodes < 500 steps")
ax.set_ylabel("mean episode length")
ax.set_title("S7B — Episode collapse vs detection outcome\n"
             "(X=T_comp fired; o=passed; color=condition)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR/"s7b_episode_collapse_vs_detection.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → s7b_episode_collapse_vs_detection.png")

# S7C: CSV summary
ep_summary = transition_df.groupby(["condition","transition"]).agg(
    mean_ep_len_mean=("mean_ep_len","mean"),
    mean_ep_len_std =("mean_ep_len","std"),
    frac_short_mean =("frac_short","mean"),
    n_delta1        =("delta","sum"),
    n_total         =("delta","count"),
).reset_index()
ep_summary.to_csv(OUT_DIR/"s7_episode_length_summary.csv", index=False)
print("  → s7_episode_length_summary.csv")
print(ep_summary.to_string(index=False))

# ────────────────────────────────────────────────────────────────────────────
# S8: Per-dim obs statistics — which dims drive JSD?
# ────────────────────────────────────────────────────────────────────────────
print("\n=== S8: Per-dim obs statistics ===\n")

# Expand mu/sigma into per-dim columns
for i, name in enumerate(OBS_NAMES):
    df[f"mu_d{i}"]    = df["mu"].apply(lambda x: x[i])
    df[f"sigma_d{i}"] = df["sigma"].apply(lambda x: x[i])

nd_only = df[df["transition"]=="baseline"].copy()

# S8A: sigma distribution across dims (box per dim, all 15 condition×seed combos)
sigma_matrix = np.array(nd_only["sigma"].tolist())   # shape (15, 11)
fig, ax = plt.subplots(figsize=(13, 5))
bp = ax.boxplot(sigma_matrix, positions=range(11), widths=0.6, patch_artist=True,
                medianprops=dict(color="black", lw=2),
                boxprops=dict(facecolor="#aec7e8", alpha=0.7))
ax.set_xticks(range(11))
ax.set_xticklabels([f"d{i}\n{n}" for i,n in enumerate(OBS_NAMES)], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("pilot-phase σ (obs std)")
ax.set_yscale("log")
ax.set_title("S8A — Per-dimension σ (pilot phase, log scale)\n"
             "dim10 foot_ang_vel dominates → drives sigma_ref floor hit in all 45 runs")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR/"s8a_perdim_sigma_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → s8a_perdim_sigma_boxplot.png")

# S8B: mu across conditions per dim (bar chart — shows policy behavioral difference)
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(11)
w = 0.25
for j, cond in enumerate(CONDITIONS):
    sub = nd_only[nd_only["condition"]==cond]
    mu_mean = np.array(sub["mu"].tolist()).mean(axis=0)
    mu_std  = np.array(sub["mu"].tolist()).std(axis=0)
    ax.bar(x + j*w, mu_mean, w, yerr=mu_std, label=cond,
           color=COND_COLORS[cond], alpha=0.8, capsize=3)
ax.set_xticks(x + w)
ax.set_xticklabels([f"d{i}\n{n}" for i,n in enumerate(OBS_NAMES)], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("pilot-phase μ (obs mean)")
ax.set_title("S8B — Per-dimension μ by condition (±1 std across 5 seeds)\n"
             "dim5 torso_vx and dim2/3/4 joint angles distinguish M1/M2/M3 behaviour")
ax.legend()
ax.axhline(0, color="black", lw=0.8)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR/"s8b_perdim_mu_by_condition.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → s8b_perdim_mu_by_condition.png")

# S8C: sigma_ref vs actual sigma_d_max — confirm floor hit
print("\n  sigma_ref floor check:")
for _, row in nd_only.iterrows():
    sigma_ref = row["sigma_ref"]
    mu_ref    = row["mu_ref"]
    floor     = 0.01 * abs(mu_ref)
    floor_hit = abs(sigma_ref - floor) < 1e-10
    print(f"    {row['condition']} s{row['seed']}: sigma_ref={sigma_ref:.6f}  floor={floor:.6f}  hit={floor_hit}")

# S8D: clip_lo/clip_hi width (2*3σ) per dim
sigma_matrix_all = np.array(df["sigma"].tolist())
clip_widths = sigma_matrix_all * 6.0   # clip = mu ± 3σ → width = 6σ
fig, ax = plt.subplots(figsize=(13,5))
bp = ax.boxplot(clip_widths, positions=range(11), widths=0.6, patch_artist=True,
                medianprops=dict(color="black",lw=2),
                boxprops=dict(facecolor="#ffbb78", alpha=0.7))
ax.set_xticks(range(11))
ax.set_xticklabels([f"d{i}\n{n}" for i,n in enumerate(OBS_NAMES)], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("clip window width (6σ)")
ax.set_yscale("log")
ax.set_title("S8C — Winsorization clip width per dim (log scale)\n"
             "Wider clip = less outlier suppression; dim10 least protected")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR/"s8c_clip_width_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → s8c_clip_width_boxplot.png")

print(f"\n  tau_watch_eq_warn=True: {df['tau_watch_eq_warn'].sum()}/{len(df)} runs (always False by design)")

# ── final CSV ──────────────────────────────────────────────────────────────
dim_cols = [f"mu_d{i}" for i in range(11)] + [f"sigma_d{i}" for i in range(11)]
out = df[["condition","seed","transition","n_episodes","mean_ep_len","median_ep_len",
          "frac_short","delta","sigma_ref","mu_ref","tau_watch_eq_warn"] + dim_cols]
out.to_csv(OUT_DIR/"s8_perdim_stats.csv", index=False)
print("  → s8_perdim_stats.csv")
print("\nDone.")
