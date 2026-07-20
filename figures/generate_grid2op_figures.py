"""
generate_grid2op_figures.py
Publication figures for the Grid2op natural-lifetime prognostics demo.
NEW 2026-07-18. Data comes from core/grid2op_lifetime_demo.py output
(<subject>_lifetimes.json + <subject>_live_run.json).

fig1  Two-panel lifetime demo (thesis fig:g2op-live):
      left  — Kaplan-Meier of the natural blackout lifetimes + Weibull fit
      right — the live held-out run: rolling conditional RUL bounds vs time,
              actual blackout marked.

Usage:
    python figures/generate_grid2op_figures.py \
        --data results/grid2op_lifetimes --subject G2_seed3 \
        --out results/grid2op_lifetimes/figures
Dependencies: matplotlib.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300


def weibull_mle(ts, iters=400):
    k = 2.0
    for _ in range(iters):
        lt = [math.log(t) for t in ts]
        tk = [t ** k for t in ts]
        k = 1.0 / (sum(t ** k * math.log(t) for t in ts) / sum(tk) - sum(lt) / len(lt))
    lam = (sum(t ** k for t in ts) / len(ts)) ** (1 / k)
    return k, lam


def rul_q(te, k, lam, q):
    """Conditional RUL quantile — identical to the deployed monitor's formula."""
    return lam * ((te / lam) ** k + math.log(q)) ** (1 / k) - te


def fig1_lifetime_demo(data_dir, subject, out_dir):
    L = json.load(open(os.path.join(data_dir, f"{subject}_lifetimes.json")))["lifetimes"]
    live = json.load(open(os.path.join(data_dir, f"{subject}_live_run.json")))
    ts = sorted(L.values())
    n = len(ts)
    k_full, lam_full = weibull_mle(ts)
    k_l, lam_l, T, chronic = live["k"], live["lam"], live["T_hard"], live["chronic"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    km_x, km_y = [0], [1.0]
    for i, t in enumerate(ts):
        km_x += [t, t]
        km_y += [km_y[-1], 1 - (i + 1) / n]
    a.plot(km_x, km_y, color="#1f77b4", lw=1.5, label=f"Kaplan–Meier ({n} chronics)")
    xs = range(1, int(max(ts) * 1.05), 10)
    a.plot(list(xs), [math.exp(-((x / lam_full) ** k_full)) for x in xs],
           color="#d62728", ls="--", lw=1.5,
           label=fr"Weibull fit ($\kappa$={k_full:.2f}, $\lambda$={lam_full:,.0f})")
    a.set_xlabel("deployment step")
    a.set_ylabel(r"$\hat S(t)$")
    a.set_title(f"{subject} natural blackout lifetimes (nominal chronics)")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    tt = list(range(1, T + 1))
    up90 = [t + rul_q(t, k_l, lam_l, 10) for t in tt]
    b.plot(tt, up90, color="#d62728", lw=1.6, label=r"rolling 90% bound: $t+\mathrm{RUL}_{p90}(t)$")
    b.plot(tt, [t + rul_q(t, k_l, lam_l, 2) for t in tt], color="#ff9f43", lw=1.3, ls="--",
           label=r"rolling median: $t+\mathrm{RUL}_{p50}(t)$")
    b.plot(tt, tt, color="0.4", lw=1, label="now ($y=t$)")
    b.scatter([T], [T], marker="x", s=90, color="k", zorder=5,
              label=f"blackout at $t$={T} (inside bound)")
    b.fill_between(tt, tt, up90, color="#d62728", alpha=0.08)
    b.set_xlabel("deployment step $t$")
    b.set_ylabel("predicted blackout step")
    b.set_title(f"Live run, held-out chronic {chronic} (LOO fit)")
    b.legend(fontsize=9, loc="upper left")
    b.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_lifetime_demo.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/grid2op_lifetimes")
    ap.add_argument("--subject", default="G2_seed3")
    ap.add_argument("--out", default=None, help="default: <data>/figures")
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, "figures")
    os.makedirs(out, exist_ok=True)
    fig1_lifetime_demo(a.data, a.subject, out)
