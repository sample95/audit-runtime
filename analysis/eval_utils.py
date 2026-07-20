#!/usr/bin/env python3
"""
eval_utils.py  ·  v1.0
Shared helpers for the RL audit evaluation pipeline.

Includes:
  - Colour / marker palettes
  - Matplotlib global style (Agg backend set here)
  - Plot primitives: style_ax, raincloud, lollipop, kde_smooth
  - Statistical functions: kaplan_meier, project_rul, project_rul_boot
  - Data loading: load_metrics, group_by, all_conditions, all_transitions
  - Small utilities: cond_colour, transition_colour, transition_marker, safe_float
"""

import json
import os
import warnings
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")          # must be called before any pyplot import
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Colour palette (Tableau-10) ──────────────────────────────────────────────
COND_PALETTE = {
    "M1": "#4E79A7", "M2": "#59A14F", "M3": "#F28E2B",
    "M4": "#E15759", "M5": "#76B7B2", "M6": "#EDC948",
    "default": "#9C755F",
}
# "baseline" is canonical; "none"/"no_drift" kept as aliases (same colour/marker)
# for any caller that passes a pre-rename transition value without normalizing.
DRIFT_PALETTE = {
    "baseline": "#AECDE8", "none": "#AECDE8", "no_drift": "#AECDE8",
    "gradual":  "#FFBE7D",
    "abrupt":   "#E15759", "sudden": "#E15759",
    "periodic": "#B07AA1",
    "default":  "#BAB0AC",
}
DRIFT_MARKERS = {
    "baseline": "o", "gradual": "s", "abrupt": "^",
    "none":     "o", "no_drift": "o", "sudden":  "^", "periodic": "D",
    "default":  "P",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "grid.color":       "#E8E8E8",
    "grid.linewidth":   0.7,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "legend.frameon":   False,
    "legend.fontsize":  9,
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT PRIMITIVES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def style_ax(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:  ax.set_xlabel(xlabel)
    if ylabel:  ax.set_ylabel(ylabel)
    if title:   ax.set_title(title, pad=6)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E8E8E8", linewidth=0.7, zorder=0)


def raincloud(ax, data_arrays, labels, colours, ylabel="", title="", orient="v"):
    """
    Half-violin (right) + IQR bar (centre) + jitter strip (left).
    data_arrays : list of 1-D np arrays (one per group)
    """
    rng = np.random.default_rng(0)
    positions = np.arange(len(labels), dtype=float)

    for i, (data, col) in enumerate(zip(data_arrays, colours)):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue
        x = positions[i]

        # ── half-violin (right side) ──────────────────────────────────────
        bw = max(1.06 * data.std() * len(data)**(-0.2), 1e-9)
        grid = np.linspace(data.min() - 2*bw, data.max() + 2*bw, 120)
        kde  = np.sum(np.exp(-0.5*((grid[:,None]-data)/bw)**2), axis=1)
        kde /= kde.max() + 1e-30
        kde *= 0.35                         # half-width scale

        if orient == "v":
            ax.fill_betweenx(grid, x, x + kde, alpha=0.55, color=col, zorder=2)
        else:
            ax.fill_between(grid, x, x + kde, alpha=0.55, color=col, zorder=2)

        # ── IQR bar (centre) ─────────────────────────────────────────────
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        if orient == "v":
            ax.plot([x, x], [q1, q3], lw=4, color=col, solid_capstyle="round",
                    zorder=3, alpha=0.85)
            ax.scatter([x], [med], zorder=4, color="white", edgecolors=col,
                       s=40, linewidths=1.5)
        else:
            ax.plot([q1, q3], [x, x], lw=4, color=col, solid_capstyle="round",
                    zorder=3, alpha=0.85)
            ax.scatter([med], [x], zorder=4, color="white", edgecolors=col,
                       s=40, linewidths=1.5)

        # ── jitter strip (left side) ─────────────────────────────────────
        jitter = rng.uniform(-0.08, 0.08, size=len(data))
        if orient == "v":
            ax.scatter(x - 0.25 + jitter, data, alpha=0.5, s=14,
                       color=col, zorder=2, linewidths=0)
        else:
            ax.scatter(data, x - 0.25 + jitter, alpha=0.5, s=14,
                       color=col, zorder=2, linewidths=0)

        # ── n= annotation ────────────────────────────────────────────────
        if orient == "v":
            ax.text(x, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else data.min() * 0.95,
                    f"n={len(data)}", ha="center", va="top", fontsize=7, color="#666666")

    if orient == "v":
        ax.set_xticks(positions); ax.set_xticklabels(labels)
        style_ax(ax, ylabel=ylabel, title=title, grid_axis="y")
    else:
        ax.set_yticks(positions); ax.set_yticklabels(labels)
        style_ax(ax, xlabel=ylabel, title=title, grid_axis="x")


def lollipop(ax, labels, values, colours, ylabel="", title="",
             ref_line=None, orient="v"):
    """Lollipop chart: dot at end of stem, value annotation."""
    positions = np.arange(len(labels), dtype=float)
    for i, (v, col) in enumerate(zip(values, colours)):
        if not np.isfinite(v):
            continue
        x = positions[i]
        if orient == "v":
            ax.plot([x, x], [0, v], color=col, lw=2, zorder=2)
            ax.scatter([x], [v], color=col, s=70, zorder=3)
            ax.text(x, v, f"{v:.3g}", ha="center", va="bottom",
                    fontsize=8, color=col)
        else:
            ax.plot([0, v], [x, x], color=col, lw=2, zorder=2)
            ax.scatter([v], [x], color=col, s=70, zorder=3)
            ax.text(v, x, f" {v:.3g}", ha="left", va="center",
                    fontsize=8, color=col)

    if ref_line is not None:
        kw = dict(color="#555555", lw=1, ls="--", zorder=1)
        if orient == "v": ax.axhline(ref_line, **kw)
        else:             ax.axvline(ref_line, **kw)

    if orient == "v":
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=15, ha="right")
        style_ax(ax, ylabel=ylabel, title=title, grid_axis="y")
    else:
        ax.set_yticks(positions); ax.set_yticklabels(labels)
        style_ax(ax, xlabel=ylabel, title=title, grid_axis="x")


def kde_smooth(data, n=200):
    """Kernel density estimate (Silverman bandwidth, no scipy)."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return np.array([data[0]] * 2), np.array([1.0, 1.0])
    bw   = max(1.06 * data.std() * len(data)**(-0.2), 1e-9)
    grid = np.linspace(data.min() - 3*bw, data.max() + 3*bw, n)
    kde  = np.sum(np.exp(-0.5*((grid[:,None]-data)/bw)**2), axis=1)
    kde /= kde.sum() * (grid[1] - grid[0]) + 1e-30
    return grid, kde


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATISTICAL FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def kaplan_meier(T_obs, delta):
    """
    Kaplan–Meier estimator with Greenwood 95% CI.
    T_obs  : array of observed times (T_comp or T_hard; NaN = censored at T_eval)
    delta  : binary event indicator (1 = failed, 0 = censored)
    Returns (times, S, ci_lo, ci_hi)
    """
    T_obs  = np.asarray(T_obs,  dtype=float)
    delta  = np.asarray(delta,  dtype=int)
    order  = np.argsort(T_obs)
    T_obs  = T_obs[order]; delta = delta[order]
    n      = len(T_obs)
    times  = [0]; S = [1.0]; var_sum = 0.0
    ci_lo  = [1.0]; ci_hi = [1.0]
    s_cur  = 1.0
    for i, (t, d) in enumerate(zip(T_obs, delta)):
        n_at_risk = n - i
        if d == 1:
            s_cur   *= 1 - 1/n_at_risk
            var_sum += 1 / (n_at_risk * max(n_at_risk - 1, 1))
            se       = s_cur * np.sqrt(var_sum)
            times.append(t)
            S.append(s_cur)
            ci_lo.append(max(0, s_cur - 1.96*se))
            ci_hi.append(min(1, s_cur + 1.96*se))
    return np.array(times), np.array(S), np.array(ci_lo), np.array(ci_hi)


def _theilsen(x, y):
    """Theil–Sen slope estimator (O(n²) — use for n < 5000)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    pairs = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dx = x[j] - x[i]
            if abs(dx) > 1e-10:
                pairs.append((y[j]-y[i])/dx)
    if not pairs:
        return float(np.nan)
    return float(np.median(pairs))


def project_rul(y_k, starts, mu_ref, sig_ref, T_ev, c_fail=2.0):
    """
    Theil–Sen RUL projection.
    Threshold = mu_ref − c_fail × sig_ref  (matches the soft-failure criterion).
    Returns T_fail (absolute), RUL, beta_TS, or (None, None, beta_TS).
    """
    y_k    = np.asarray(y_k, dtype=float)
    starts = np.asarray(starts, dtype=float)
    valid  = np.isfinite(y_k) & np.isfinite(starts)
    if valid.sum() < 3:
        return None, None, float("nan")
    beta      = _theilsen(starts[valid], y_k[valid])
    y_last    = y_k[valid][-1]
    T_last    = starts[valid][-1]
    threshold = float(mu_ref) - float(c_fail) * float(sig_ref)
    if not np.isfinite(beta) or beta >= 0:
        return None, None, beta
    T_fail = T_last + (threshold - y_last) / beta
    rul    = max(0.0, T_fail - T_ev)
    return float(T_fail), float(rul), float(beta)


def project_rul_boot(y_k, starts, mu_ref, sig_ref, T_ev,
                     c_fail=2.0, n=500, rng_seed=42):
    """Bootstrap 90% CI for RUL (5th–95th percentile).
    Threshold = mu_ref − c_fail × sig_ref, matching soft-failure criterion."""
    y_k       = np.asarray(y_k, dtype=float)
    starts    = np.asarray(starts, dtype=float)
    threshold = float(mu_ref) - float(c_fail) * float(sig_ref)
    valid     = np.where(np.isfinite(y_k) & np.isfinite(starts))[0]
    if len(valid) < 4:
        return float("nan"), float("nan")
    rng  = np.random.default_rng(rng_seed)
    ruls = []
    for _ in range(n):
        idx  = rng.choice(valid, size=len(valid), replace=True)
        idx  = np.sort(idx)
        beta = _theilsen(starts[idx], y_k[idx])
        if not np.isfinite(beta) or beta >= 0:
            continue
        y_l = y_k[idx][-1]; T_l = starts[idx][-1]
        T_f = T_l + (threshold - y_l) / beta
        ruls.append(max(0.0, T_f - T_ev))
    if len(ruls) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(ruls, 5)), float(np.percentile(ruls, 95))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_metrics(results_dir, domain):
    """
    Load all per-run *_metrics.json files from results_dir.
    Falls back to the aggregate all_rollouts_{domain}_summary.json if none found.
    Returns list of dicts (one per run).
    """
    recs = []
    for fname in sorted(os.listdir(results_dir)):
        if not (fname.endswith("_metrics_summary.json") or fname.endswith("_metrics.json")):
            continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path) as f:
                rec = json.load(f)
            recs.append(rec)
        except Exception as e:
            print(f"  [warn] could not load {fname}: {e}")
    if not recs:
        agg = os.path.join(results_dir, f"all_rollouts_{domain}_summary.json")
        if os.path.exists(agg):
            with open(agg) as f:
                recs = json.load(f)
            print(f"  [info] loaded {len(recs)} records from aggregate summary")
        else:
            print(f"  [error] no metrics files found in {results_dir}")
    # Normalize the null-condition label: "baseline" is canonical going forward;
    # "none"/"no_drift" are pre-rename aliases still present in older result JSONs.
    for r in recs:
        if r.get("transition") in ("none", "no_drift"):
            r["transition"] = "baseline"
    return recs


def group_by(recs, keys):
    """Group records by a tuple of keys."""
    groups = defaultdict(list)
    for r in recs:
        k = tuple(r.get(key) for key in keys)
        groups[k].append(r)
    return groups


def all_conditions(recs):
    return sorted(set(r.get("condition", "?") for r in recs))

def all_transitions(recs):
    return sorted(set(r.get("transition", "?") for r in recs))

def cond_colour(c):
    return COND_PALETTE.get(c, COND_PALETTE["default"])

def transition_colour(d):
    return DRIFT_PALETTE.get(d, DRIFT_PALETTE["default"])

def transition_marker(d):
    return DRIFT_MARKERS.get(d, DRIFT_MARKERS["default"])

def safe_float(v, fallback=float("nan")):
    try:    return float(v) if v is not None else fallback
    except: return fallback
