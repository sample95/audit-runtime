"""
generate_walker2d_figures.py
Publication figures for the Walker2d cross-morphology replication.
NEW 2026-07-18. Companion to generate_hopper_figures.py (same conventions;
Walker2d has one deployment campaign: M1+M3, baseline/gradual/abrupt(t0=510k)).

fig1  D_k trajectories per schedule (thesis fig:walker-dk)
fig2  Weibull survival overlay, Walker2d vs Hopper PPO gradual (thesis D10):
      the kappa=7.81/7.91 near-coincidence, plotted rather than asserted.

Usage:
    python figures/generate_walker2d_figures.py \
        --data results/mujoco/walker2d_deploy_2026-07-12 \
        --hopper-data results/mujoco/hopper/t10_v2 \
        --out results/mujoco/walker2d_deploy_2026-07-12/figures
Dependencies: matplotlib, numpy.
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 50_000          # window width (Table 2.1)
TAU_FLOOR = 2e-4
FIRST_ELIGIBLE = 175_000
DPI = 300

STYLE = {"baseline": ("0.55", "baseline (null)"),
         "gradual":  ("#1f77b4", "gradual"),
         "abrupt":   ("#d62728", "abrupt ($t_0$=510k)")}


def weibull_mle(ts, iters=400):
    k = 2.0
    for _ in range(iters):
        lt = [math.log(t) for t in ts]
        tk = [t ** k for t in ts]
        k = 1.0 / (sum(t ** k * math.log(t) for t in ts) / sum(tk) - sum(lt) / len(lt))
    lam = (sum(t ** k for t in ts) / len(ts)) ** (1 / k)
    return k, lam


def fig1_dk_trajectories(data_dir, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    seen = set()
    for f in sorted(glob.glob(os.path.join(data_dir, "*metrics_summary*.json"))):
        d = json.load(open(f))
        sched = os.path.basename(f).replace("_metrics_summary.json", "").split("_")[-1]
        c, lab = STYLE[sched]
        x = [ws + W for ws in d["window_starts"]]
        y = [v if v and v > 0 else 1e-7 for v in d["D_k"]]
        ax.plot(x, y, color=c, lw=1.3, alpha=0.85, label=lab if sched not in seen else None)
        seen.add(sched)
    ax.axhline(TAU_FLOOR, color="k", ls="--", lw=1,
               label=r"$\tau_{\mathrm{warn}}$ floor ($2{\times}10^{-4}$)")
    ax.axvline(FIRST_ELIGIBLE, color="0.3", ls=":", lw=1,
               label="first eligible window end (175k)")
    ax.set_yscale("log")
    ax.set_xlabel("deployment step")
    ax.set_ylabel(r"$D_k$ (feature-averaged JSD)")
    ax.set_title("Walker2d M1+M3 auditable deployments: $D_k$ trajectories per schedule")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_dk_trajectories.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _gradual_t_hards(data_dir):
    ths = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*gradual*metrics_summary*.json"))):
        th = json.load(open(f)).get("T_hard")
        if th:
            ths.append(th)
    return ths


def fig2_weibull_overlay(data_dir, hopper_dir, out_dir):
    """D10: Walker2d vs Hopper PPO gradual survival — the wear-out coincidence."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, ths, color in [
        ("Walker2d M1+M3 ($1.5\\times$ g)", _gradual_t_hards(data_dir), "#1f77b4"),
        ("Hopper PPO ($2\\times$ g)",        _gradual_t_hards(hopper_dir), "#d62728"),
    ]:
        if not ths:
            print(f"  [skip] no gradual T_hard found for {name}")
            continue
        k, lam = weibull_mle(ths)
        xs = range(1000, int(max(ths) * 1.15), 500)
        ax.plot(list(xs), [math.exp(-((x / lam) ** k)) for x in xs], color=color, lw=1.8,
                label=f"{name}: $\\kappa$={k:.2f}, $\\lambda$={lam:,.0f} (n={len(ths)})")
        ths_sorted = sorted(ths)
        n = len(ths_sorted)
        km_x, km_y = [0], [1.0]
        for i, t in enumerate(ths_sorted):
            km_x += [t, t]
            km_y += [km_y[-1], 1 - (i + 1) / n]
        ax.plot(km_x, km_y, color=color, lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel("deployment step")
    ax.set_ylabel(r"$\hat S(t)$")
    ax.set_title("Gradual-drift survival: Walker2d vs Hopper (Weibull fits; KM dotted)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_weibull_overlay.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/mujoco/walker2d_deploy_2026-07-12")
    ap.add_argument("--hopper-data", default="results/mujoco/hopper/t10_v2")
    ap.add_argument("--out", default=None, help="default: <data>/figures")
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, "figures")
    os.makedirs(out, exist_ok=True)
    fig1_dk_trajectories(a.data, out)
    fig2_weibull_overlay(a.data, a.hopper_data, out)
