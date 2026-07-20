#!/usr/bin/env python3
"""
Retroactive K0 sensitivity analysis — hopper
=============================================
Simulates tau_warn for K0 = 2, 3, 4, 6, 8, 12, 16, 20
using data already on disk:

  K0 ≤ 4  → first K0 values from stored pilot_D    (LOO-correct)
  K0 > 4  → pilot_D[:4] + D_k[4:K0] from none  (proxy: pre-drift D_k
             computed against the 4-window reference)

tau_warn formula (EVT prediction interval, deployment.py _evt_threshold):
  tau_warn = mean(D) + t_{n-1, 1-λ_FA} · std(D, ddof=1) · sqrt(1 + 1/n)

Output:
  hopper_analysis/s6b_k0_sensitivity.png
  hopper_analysis/s6b_k0_sensitivity.csv
"""

import json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t as t_dist

RESULTS_DIR = Path("results/mujoco/hopper")
OUT_DIR     = Path("hopper_analysis")
OUT_DIR.mkdir(exist_ok=True)

LAMBDA_FA   = 0.05          # from instrument.json
K0_NOMINAL  = 4
K0_GRID     = [2, 3, 4, 6, 8, 12, 16, 20]
CONDITIONS  = ["M1", "M2", "M3"]
SEEDS       = [0, 1, 2, 3, 4]
T0          = 500_000
COND_COLORS = {"M1": "#1f77b4", "M2": "#ff7f0e", "M3": "#2ca02c"}


def evt_tau(D: np.ndarray, lambda_fa: float = LAMBDA_FA) -> float:
    n = len(D)
    if n < 2:
        return float(np.max(D))
    mu     = float(np.mean(D))
    s      = float(np.std(D, ddof=1))
    t_crit = float(t_dist.ppf(1.0 - lambda_fa, df=n - 1))
    return mu + t_crit * s * float(np.sqrt(1.0 + 1.0 / n))


# ── load data ───────────────────────────────────────────────────────────────
rows = []
for cond in CONDITIONS:
    for seed in SEEDS:
        # pilot_D from any transition type (identical across transition types for same seed)
        f_any = RESULTS_DIR / f"{cond}_seed{seed}_abrupt_metrics_summary.json"
        d_any = json.loads(f_any.read_text())
        pilot_D = np.array(d_any["pilot_D"])          # shape (4,)
        assert len(pilot_D) == K0_NOMINAL

        # pre-drift post-pilot D_k from none run (proxy for K0 extension)
        f_nd  = RESULTS_DIR / f"{cond}_seed{seed}_none_metrics_summary.json"
        d_nd  = json.loads(f_nd.read_text())
        ws    = np.array(d_nd["window_starts"])
        dk_nd = np.array(d_nd["D_k"])
        # windows K0..end that start before t0 → proxy stable D_k pool
        ext_mask  = (ws[K0_NOMINAL:] < T0)
        ext_dk    = dk_nd[K0_NOMINAL:][ext_mask]       # up to 16 extra values

        for k0 in K0_GRID:
            if k0 <= K0_NOMINAL:
                D_pool = pilot_D[:k0]
                source = "pilot_D (LOO)"
            else:
                extra_needed = k0 - K0_NOMINAL
                if extra_needed > len(ext_dk):
                    continue            # not enough pre-drift windows
                D_pool = np.concatenate([pilot_D, ext_dk[:extra_needed]])
                source = f"pilot_D + {extra_needed} pre-drift D_k"

            tau = evt_tau(D_pool, LAMBDA_FA)
            rows.append({
                "condition":  cond,
                "seed":       seed,
                "K0":         k0,
                "tau_warn":   tau,
                "D_pool_mean": float(np.mean(D_pool)),
                "D_pool_std":  float(np.std(D_pool, ddof=1)),
                "source":     source,
            })

df = pd.DataFrame(rows)

# ── print summary ────────────────────────────────────────────────────────────
print("K0 sensitivity — tau_warn statistics across 15 (condition × seed) pairs\n")
stats = df.groupby(["condition", "K0"])["tau_warn"].agg(
    ["min", "max", "mean", "std"]
).reset_index()
stats["ratio"] = stats["max"] / stats["min"]
print(stats.to_string(index=False))
df.to_csv(OUT_DIR / "s6b_k0_sensitivity.csv", index=False)

print()
agg_all = df.groupby("K0")["tau_warn"].agg(["min","max","mean","std"])
agg_all["ratio"] = agg_all["max"] / agg_all["min"]
print("Overall (all 15 pairs):")
print(agg_all.to_string())

# ── Figure A: tau_warn vs K0, one line per seed per condition ─────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, cond in zip(axes, CONDITIONS):
    sub = df[df["condition"] == cond]
    for seed in SEEDS:
        s_sub = sub[sub["seed"] == seed]
        ax.plot(s_sub["K0"], s_sub["tau_warn"],
                marker="o", ms=5, lw=1.4, alpha=0.8,
                label=f"s{seed}")
    ax.axvline(K0_NOMINAL, color="black", ls="--", lw=1.2, label=f"K₀=4 (nominal)")
    ax.set_yscale("log")
    ax.set_xticks(K0_GRID)
    ax.set_xlabel("K₀")
    ax.set_ylabel("τ_warn (log scale)")
    ax.set_title(cond)
    ax.legend(fontsize=8, title="seed")
    ax.grid(True, which="both", alpha=0.3)
fig.suptitle("S6b — τ_warn vs K₀  (K₀≤4: LOO pilot_D; K₀>4: pilot_D + pre-drift D_k)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
path = OUT_DIR / "s6b_k0_sensitivity_lines.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  → {path}")

# ── Figure B: spread (ratio max/min) vs K0 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for cond in CONDITIONS:
    sub = df[df["condition"] == cond]
    r = sub.groupby("K0")["tau_warn"].agg(lambda x: x.max() / x.min()).reset_index()
    ax.plot(r["K0"], r["tau_warn"], marker="o", ms=7, lw=2,
            color=COND_COLORS[cond], label=cond)
ax.axvline(K0_NOMINAL, color="black", ls="--", lw=1.2, label="K₀=4 (nominal)")
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_xticks(K0_GRID)
ax.set_xlabel("K₀")
ax.set_ylabel("τ_warn spread  (max / min across seeds)")
ax.set_title("S6b — Calibration instability vs K₀\n"
             "Lower ratio = more consistent threshold")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = OUT_DIR / "s6b_k0_spread.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {path}")

# ── Figure C: tau_warn distribution at each K0 (box per K0) ─────────────────
fig, ax = plt.subplots(figsize=(11, 5))
data_by_k0 = [df[df["K0"] == k]["tau_warn"].values for k in K0_GRID
              if not df[df["K0"] == k].empty]
k0_labels  = [k for k in K0_GRID if not df[df["K0"] == k].empty]
bp = ax.boxplot(data_by_k0, positions=range(len(k0_labels)),
                widths=0.5, patch_artist=True,
                medianprops=dict(color="black", lw=2),
                boxprops=dict(facecolor="#aec7e8", alpha=0.7))
nominal_pos = k0_labels.index(K0_NOMINAL)
ax.axvline(nominal_pos, color="black", ls="--", lw=1.5, label=f"K₀=4 (nominal)")
ax.set_xticks(range(len(k0_labels)))
ax.set_xticklabels([f"K₀={k}" for k in k0_labels], rotation=30)
ax.set_yscale("log")
ax.set_ylabel("τ_warn (log scale)")
ax.set_title("S6b — τ_warn distribution across all 15 (condition × seed) pairs, by K₀")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
path = OUT_DIR / "s6b_k0_boxplot.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {path}")

print("\nDone.")
