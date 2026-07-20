"""
generate_510k_figures.py
Figure for the corrected abrupt-onset rerun (t0=510k): the failure-latency
bound. NEW 2026-07-18.

fig1  Per-run T_hard vs the 525k earliest-alarm floor, PPO (2x g) and
      VecNormalize (1.5x g) side by side (thesis fig:latency-510k):
      shows why moving the onset off the window grid recovers 13/15 for
      VecNorm but only 2/15 for raw PPO — post-onset failure latency vs
      the remainder of the straddling window.

Usage:
    python figures/generate_510k_figures.py \
        --ppo results/mujoco/hopper/t10_v2_510k_v2 \
        --vc  results/mujoco/hopper_vc_510k_v2 \
        --out results/mujoco/figures_510k
Dependencies: matplotlib.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = 510_000
EARLIEST = 525_000
DPI = 300


def load(dirpath):
    runs = []
    for f in sorted(glob.glob(os.path.join(dirpath, "M*_metrics_summary.json"))):
        d = json.load(open(f))
        name = os.path.basename(f).split("_abrupt")[0]
        runs.append((name, d.get("T_warn"), d.get("T_hard")))
    return runs


def classify(tw, th):
    if tw is not None and th is not None and tw < th and tw >= T0:
        return "genuine"
    if tw is not None and tw < T0:
        return "pre-onset FA"
    return "miss (dies before 525k)"


COLORS = {"genuine": "#2ca02c", "miss (dies before 525k)": "#d62728",
          "pre-onset FA": "#ff9f43"}


def fig1_latency(ppo_dir, vc_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, runs) in zip(axes, [
            (r"raw PPO ($2\times$ gravity): 2/15", load(ppo_dir)),
            (r"PPO+VecNormalize ($1.5\times$): 13/15", load(vc_dir))]):
        seen = set()
        for i, (name, tw, th) in enumerate(runs):
            cls = classify(tw, th)
            ax.scatter(i, th, color=COLORS[cls], s=45, zorder=5,
                       label=cls if cls not in seen else None)
            seen.add(cls)
        ax.axhline(T0, color="0.4", ls=":", lw=1.2)
        ax.text(0.2, T0 + 800, r"onset $t_0$=510k", fontsize=8, color="0.3")
        ax.axhline(EARLIEST, color="k", ls="--", lw=1.2)
        ax.text(0.2, EARLIEST + 800, "earliest alarm (straddling window end, 525k)",
                fontsize=8)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([r[0] for r in runs], rotation=60, fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(r"$T_{\mathrm{hard}}$ (first failure, steps)")
    handles, labels = axes[1].get_legend_handles_labels()
    h0, l0 = axes[0].get_legend_handles_labels()
    merged = dict(zip(l0 + labels, h0 + handles))
    axes[1].legend(merged.values(), merged.keys(), fontsize=8, loc="lower right")
    fig.suptitle("Corrected abrupt rerun: failure latency vs the one-window response floor",
                 y=1.02)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_latency_510k.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", default="results/mujoco/hopper/t10_v2_510k_v2")
    ap.add_argument("--vc", default="results/mujoco/hopper_vc_510k_v2")
    ap.add_argument("--out", default="results/mujoco/figures_510k")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1_latency(a.ppo, a.vc, a.out)
