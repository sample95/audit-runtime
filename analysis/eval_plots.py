#!/usr/bin/env python3
"""
eval_plots.py  ·  v1.0
Nine plot functions for the RL audit evaluation pipeline.

Sections:
  1  plot_overview_heatmap     — condition × drift overview heatmap
  2  plot_signal_trajectories  — D_k / y_k mean ± SD across seeds
  3  plot_warning_quality      — Pr(warn), Pr(timely), alarm burden lollipops
  4  plot_policy_comparison    — rainclouds: δ, T_comp, β_TS per condition
  5  plot_failure_anatomy      — stacked bars: no-fail / soft / hard / both
  6  plot_seed_consistency     — T_warn connected dots + CV lollipop
  7  plot_similarities         — D_k correlation heatmap + per-drift raincloud
  8  plot_km_rul               — Kaplan–Meier survival + Theil–Sen RUL
  9  plot_seed_video           — per-seed temporal filmstrip (4 rows × N seeds)
"""

import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from eval_utils import (
    style_ax, raincloud, lollipop,
    kaplan_meier, project_rul, project_rul_boot,
    all_conditions, all_transitions,
    cond_colour, transition_colour, transition_marker,
    safe_float,
)


def plot_overview_heatmap(recs, out_path):
    """
    Three-panel heatmap: condition × transition.
    Cells: median T_warn/T_eval | median T_hard/T_eval | warn-rate.
    """
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)
    NC, ND = len(conds), len(transitions)

    def cell_matrix(fn):
        M = np.full((NC, ND), np.nan)
        for (ci, c) in enumerate(conds):
            for (di, d) in enumerate(transitions):
                sub = [r for r in recs if r.get("condition")==c and r.get("transition")==d]
                if sub: M[ci, di] = fn(sub)
        return M

    T_eval = max(safe_float(r.get("T_eval", 1e6)) for r in recs)

    M_warn = cell_matrix(lambda s: np.nanmedian([safe_float(r.get("T_warn")) for r in s]) / T_eval)
    M_hard = cell_matrix(lambda s: np.nanmedian([safe_float(r.get("T_hard")) for r in s]) / T_eval)
    M_rate = cell_matrix(lambda s: np.mean([1 if r.get("T_warn") is not None else 0 for r in s]))

    fig, axes = plt.subplots(1, 3, figsize=(13, max(3, 0.6*NC + 1.5)))
    titles   = ["Median T_warn / T_eval", "Median T_hard / T_eval", "Warn rate"]
    matrices = [M_warn, M_hard, M_rate]
    cmaps    = ["Blues", "Reds", "Greens"]

    for ax, M, title, cmap in zip(axes, matrices, titles, cmaps):
        masked = np.ma.masked_invalid(M)
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(ND)); ax.set_xticklabels(transitions, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(NC)); ax.set_yticklabels(conds, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
        for ci in range(NC):
            for di in range(ND):
                v = M[ci, di]
                if np.isfinite(v):
                    ax.text(di, ci, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.55 else "black")

    fig.suptitle("Deployment Overview  ·  condition × drift", fontsize=12,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [01] {out_path}")


def plot_signal_trajectories(recs, out_path):
    """
    D_k and y_k trajectories per (condition, transition), all seeds overlaid
    with mean ± 1SD band. Tau_warn / tau_watch reference lines added.
    """
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)
    NC, ND = len(conds), len(transitions)

    fig, axes = plt.subplots(2, NC, figsize=(4*NC, 3.5*2), squeeze=False)
    axes[0, 0].set_ylabel("D_k  (JS divergence)")
    axes[1, 0].set_ylabel("y_k  (standardised reward)")

    for ci, cond in enumerate(conds):
        for row, key, ylabel in [(0, "D_k", "D_k"), (1, "y_k", "y_k")]:
            ax = axes[row, ci]
            if ci == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(cond if row == 0 else "", fontweight="bold")

            tau_warn_vals = []
            for drift in transitions:
                col = transition_colour(drift)
                sub = [r for r in recs if r.get("condition")==cond
                                       and r.get("transition")==drift
                                       and key in r]
                if not sub:
                    continue

                # Interpolate all runs to a common grid
                max_b  = max(len(r[key]) for r in sub)
                arr    = np.full((len(sub), max_b), np.nan)
                for si, r in enumerate(sub):
                    v = np.asarray(r[key], dtype=float)
                    arr[si, :len(v)] = v

                xs = np.arange(max_b)
                mu = np.nanmean(arr, axis=0)
                sd = np.nanstd(arr,  axis=0)

                ax.plot(xs, mu, color=col, lw=1.5, label=drift,
                        marker=transition_marker(drift), markevery=max(1, max_b//10),
                        markersize=4, zorder=3)
                ax.fill_between(xs, mu-sd, mu+sd, color=col, alpha=0.18, zorder=2)

                for r in sub:
                    tw = safe_float(r.get("tau_warn"))
                    if np.isfinite(tw):
                        tau_warn_vals.append(tw)

            if row == 0 and tau_warn_vals:
                ax.axhline(np.mean(tau_warn_vals), color="#C0392B", lw=1,
                           ls="--", label="τ_warn", zorder=4)
            if row == 1:
                ax.axhline(0, color="#888888", lw=0.8, ls=":", zorder=1)

            style_ax(ax, xlabel="Window index", grid_axis="y")
            if ci == NC-1:
                ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("Signal Trajectories  ·  mean ± 1 SD across seeds", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [02] {out_path}")


def plot_warning_quality(recs, out_path):
    """
    Lollipop: Pr(warn), Pr(timely | warned), mean alarm windows — per condition/drift cell.
    """
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)

    cells  = [(c, d) for c in conds for d in transitions]
    labels = [f"{c}\n{d}" for c, d in cells]
    colours = [cond_colour(c) for c, _ in cells]

    pr_warn   = []
    pr_timely = []
    mean_alarms = []

    for c, d in cells:
        sub = [r for r in recs if r.get("condition")==c and r.get("transition")==d]
        if not sub:
            pr_warn.append(np.nan); pr_timely.append(np.nan); mean_alarms.append(np.nan)
            continue
        warned  = [1 if r.get("T_warn") is not None else 0 for r in sub]
        timely  = [1 if r.get("timely_warn") else 0
                   for r in sub if r.get("T_warn") is not None]
        nalarms = [safe_float(r.get("n_alarm_windows_deg", r.get("n_alarm_windows_perturbed", 0)))
                   for r in sub]
        pr_warn.append(np.mean(warned))
        pr_timely.append(np.mean(timely) if timely else np.nan)
        mean_alarms.append(np.nanmean(nalarms))

    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, 0.4*len(cells)+2)))

    lollipop(axes[0], labels, pr_warn,   colours,
             ylabel="Pr(warn)", title="Warning rate",   ref_line=0.5)
    lollipop(axes[1], labels, pr_timely, colours,
             ylabel="Pr(timely | warned)", title="Timeliness", ref_line=0.5)
    lollipop(axes[2], labels, mean_alarms, colours,
             ylabel="Mean alarm windows", title="Alarm burden")

    # Condition legend
    handles = [mpatches.Patch(color=cond_colour(c), label=c) for c in conds]
    fig.legend(handles=handles, loc="lower center", ncol=len(conds),
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("Warning Quality  ·  condition × drift", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [03] {out_path}")


def plot_policy_comparison(recs, out_path):
    """
    Raincloud per condition: delta (performance gap), T_comp/T_eval, beta_TS slope.
    Transition types shown as separate scatter layers within each condition.
    """
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)
    T_eval = max(safe_float(r.get("T_eval", 1e6)) for r in recs)

    metrics = [
        ("delta",      "Δ performance",       "delta"),
        ("T_comp_norm","T_comp / T_eval",      "T_comp"),
        ("beta_TS",    "β_TS  (drift slope)", "beta_TS"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, 0.7*len(conds)+2)))

    for ax, (field, ylabel, src_key) in zip(axes, metrics):
        data_arrays = []
        for c in conds:
            sub  = [r for r in recs if r.get("condition")==c]
            if src_key == "T_comp":
                vals = [safe_float(r.get("T_comp", r.get("T_comp_obs"))) / T_eval for r in sub]
            else:
                vals = [safe_float(r.get(src_key)) for r in sub]
            data_arrays.append(np.array([v for v in vals if np.isfinite(v)]))

        colours_r = [cond_colour(c) for c in conds]
        raincloud(ax, data_arrays, conds, colours_r, ylabel=ylabel,
                  title=ylabel, orient="v")

        # Overlay drift markers
        for ci, c in enumerate(conds):
            for d in transitions:
                sub  = [r for r in recs if r.get("condition")==c
                                        and r.get("transition")==d]
                if src_key == "T_comp":
                    vals = [safe_float(r.get("T_comp", r.get("T_comp_obs"))) / T_eval for r in sub]
                else:
                    vals = [safe_float(r.get(src_key)) for r in sub]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    jitter = np.random.default_rng(hash(d)%2**32).uniform(-0.06,0.06,len(vals))
                    ax.scatter(ci - 0.25 + jitter, vals,
                               marker=transition_marker(d), s=20,
                               color=transition_colour(d), zorder=5, alpha=0.7,
                               edgecolors="none")

    # Drift legend
    handles = [Line2D([0],[0], marker=transition_marker(d), color=transition_colour(d),
                       linestyle="none", markersize=7, label=d) for d in transitions]
    fig.legend(handles=handles, loc="lower center", ncol=len(transitions),
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle("Policy Comparison  ·  distribution across seeds", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [04] {out_path}")


def plot_failure_anatomy(recs, out_path):
    """
    Stacked bar per (condition, transition):
    categories — no failure | soft failure only | hard failure | both.
    Proportions across seeds.
    """
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)
    cells  = [(c, d) for c in conds for d in transitions]
    labels = [f"{c}\n{d}" for c, d in cells]

    CAT_COLS  = ["#5AAE61", "#FDAE61", "#D73027", "#762A83"]
    CAT_LABS  = ["No failure", "Soft only", "Hard failure", "Both"]

    counts = np.zeros((len(cells), 4))
    for ki, (c, d) in enumerate(cells):
        sub = [r for r in recs if r.get("condition")==c and r.get("transition")==d]
        if not sub: continue
        for r in sub:
            has_soft = r.get("T_soft") is not None
            has_hard = r.get("T_hard") is not None
            if has_hard and has_soft:   counts[ki, 3] += 1
            elif has_hard:              counts[ki, 2] += 1
            elif has_soft:              counts[ki, 1] += 1
            else:                       counts[ki, 0] += 1

    totals = counts.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1, totals)
    props  = counts / totals

    x   = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(max(8, 0.8*len(cells)), 5))
    bottoms = np.zeros(len(cells))
    for cat_i, (col, lab) in enumerate(zip(CAT_COLS, CAT_LABS)):
        ax.bar(x, props[:, cat_i], bottom=bottoms, color=col,
               label=lab, width=0.65, edgecolor="white", linewidth=0.5)
        for xi, (p, b) in enumerate(zip(props[:, cat_i], bottoms)):
            if p > 0.07:
                ax.text(xi, b + p/2, f"{p:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottoms = bottoms + props[:, cat_i]

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Proportion of seeds")
    ax.legend(loc="upper right", fontsize=9)
    style_ax(ax, title="Failure Anatomy  ·  condition × drift", grid_axis=None)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [05] {out_path}")


def plot_seed_consistency(recs, out_path):
    """
    Left: connected dot plot of T_warn per seed, grouped by condition/drift.
    Right: CV lollipop (σ/μ of T_warn) per cell.
    """
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)
    cells  = [(c, d) for c in conds for d in transitions]
    T_eval = max(safe_float(r.get("T_eval", 1e6)) for r in recs)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.55*len(cells)+2)))

    # — Left: connected dot plot ──────────────────────────────────────────
    ax = axes[0]
    seeds = sorted(set(r.get("seed", 0) for r in recs))
    xtick_pos, xtick_lab = [], []

    for ki, (c, d) in enumerate(cells):
        offset  = ki * (len(seeds) + 1)
        sub     = {r.get("seed"): safe_float(r.get("T_warn")) / T_eval
                   for r in recs if r.get("condition")==c and r.get("transition")==d}
        col = cond_colour(c)
        vals = [sub.get(s, np.nan) for s in seeds]
        xs   = [offset + si for si in range(len(seeds))]

        fin_xs  = [x for x,v in zip(xs, vals) if np.isfinite(v)]
        fin_vs  = [v for v in vals if np.isfinite(v)]
        if len(fin_xs) > 1:
            ax.plot(fin_xs, fin_vs, color=col, lw=1.2, alpha=0.6, zorder=2)
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.scatter(x, v, color=col, s=40, zorder=3,
                           marker=transition_marker(d), edgecolors="none")

        cx = offset + (len(seeds)-1) / 2
        xtick_pos.append(cx); xtick_lab.append(f"{c}\n{d}")

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("T_warn / T_eval")
    style_ax(ax, title="T_warn per Seed  ·  connected", grid_axis="y")

    # — Right: CV lollipop ────────────────────────────────────────────────
    labels  = [f"{c}\n{d}" for c, d in cells]
    colours = [cond_colour(c) for c, _ in cells]
    cvs = []
    for c, d in cells:
        sub = [r for r in recs if r.get("condition")==c and r.get("transition")==d]
        vals= [safe_float(r.get("T_warn")) for r in sub if r.get("T_warn") is not None]
        if len(vals) >= 2:
            mu  = np.mean(vals)
            cvs.append(np.std(vals) / max(abs(mu), 1e-9))
        else:
            cvs.append(np.nan)

    lollipop(axes[1], labels, cvs, colours,
             ylabel="CV  (σ/μ)  of T_warn",
             title="Seed Variability  ·  CV lollipop",
             ref_line=0.2)

    fig.suptitle("Seed Consistency", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [06] {out_path}")


def plot_similarities(recs, out_path):
    """
    Left: eigenvector-ordered Pearson correlation heatmap of D_k trajectories.
    Right: raincloud of D_k distributions per transition type.
    """
    run_ids, Dk_vecs = [], []
    max_b = max((len(r.get("D_k", [])) for r in recs), default=0)
    if max_b == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No D_k data", ha="center", va="center")
        fig.savefig(out_path); plt.close(fig); return

    for r in recs:
        dk = r.get("D_k", [])
        if not dk: continue
        v = np.full(max_b, np.nan)
        v[:len(dk)] = dk
        run_ids.append(f"{r.get('condition')}/{r.get('transition')}/s{r.get('seed','?')}")
        Dk_vecs.append(v)

    if not Dk_vecs:
        return

    X = np.array(Dk_vecs)   # (N, B)
    valid_cols = np.where(np.all(np.isfinite(X), axis=0))[0]
    if len(valid_cols) < 2:
        valid_cols = np.arange(X.shape[1])
    Xv  = X[:, valid_cols]
    Xv  = np.nan_to_num(Xv, nan=0.0)
    Xv -= Xv.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xv, axis=1, keepdims=True) + 1e-10
    Xv /= norms
    C = Xv @ Xv.T  # (N, N) Pearson correlation

    # Eigenvector ordering
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvecs[:, -1])
    C_ord = C[np.ix_(order, order)]
    run_ids_ord = [run_ids[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.25*len(run_ids)+3)))

    ax = axes[0]
    im = ax.imshow(C_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = len(run_ids_ord)
    tick_step = max(1, n // 15)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels(run_ids_ord[::tick_step], rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(0, n, tick_step))
    ax.set_yticklabels(run_ids_ord[::tick_step], fontsize=6)
    ax.set_title("D_k Trajectory Correlation\n(eigenvector-ordered)", fontsize=10,
                 fontweight="bold")

    transitions = all_transitions(recs)
    dk_by_transition = []
    for d in transitions:
        sub = [r for r in recs if r.get("transition")==d and r.get("D_k")]
        vals = np.concatenate([np.asarray(r["D_k"], dtype=float) for r in sub]) if sub else np.array([])
        dk_by_transition.append(vals[np.isfinite(vals)])

    cols = [transition_colour(d) for d in transitions]
    raincloud(axes[1], dk_by_transition, transitions, cols,
              ylabel="D_k value",
              title="D_k Distribution  ·  per transition type",
              orient="v")

    fig.suptitle("Trajectory Similarities", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [07] {out_path}")


def plot_km_rul(recs, out_path, save_forecast=None):
    """
    Left:  KM survival curves per condition (pooled over transition types).
    Right: Theil–Sen RUL per (condition, transition) with 90% bootstrap CI.

    Event source: T_comp_obs + delta (canonical, avoids T_hard/T_soft ambiguity).
    RUL threshold: mu_ref − c_fail × sigma_ref (read from each record).
    """
    conds  = all_conditions(recs)
    transitions = all_transitions(recs)
    T_eval = max(safe_float(r.get("T_eval", 1e6)) for r in recs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── KM survival ─────────────────────────────────────────────────────────
    ax = axes[0]
    for cond in conds:
        sub   = [r for r in recs if r.get("condition")==cond]
        T_obs = []
        delta = []
        for r in sub:
            T_obs.append(safe_float(r.get("T_comp_obs", T_eval)))
            delta.append(int(r.get("delta", 0)))
        if not T_obs: continue
        times, S, ci_lo, ci_hi = kaplan_meier(np.array(T_obs), np.array(delta))
        col = cond_colour(cond)
        ax.step(times, S,      where="post", color=col, lw=2, label=cond)
        ax.fill_between(times, ci_lo, ci_hi, step="post", alpha=0.18, color=col)

    ax.set_xlim(left=0, right=T_eval)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="#888", lw=0.9, ls="--")
    ax.text(T_eval * 0.01, 0.52, "S=0.5", fontsize=8, color="#888")
    style_ax(ax, xlabel="Step", ylabel="Survival S(t)",
             title="Kaplan–Meier  ·  per condition", grid_axis="y")
    ax.legend(fontsize=9)

    # ── RUL projection ───────────────────────────────────────────────────────
    ax = axes[1]
    cells   = [(c, d) for c in conds for d in transitions]
    rul_vals, rul_lo, rul_hi = [], [], []
    rul_labels, rul_colours  = [], []
    forecast_records = []

    for c, d in cells:
        sub = [r for r in recs if r.get("condition")==c and r.get("transition")==d
                               and r.get("D_k") and r.get("window_starts")]
        if not sub: continue

        all_yk, all_starts = [], []
        mu_ref = sig_ref = c_fail_val = None
        for r in sub:
            yk = np.asarray(r.get("y_k", []), dtype=float)
            st = np.asarray(r.get("window_starts", []), dtype=float)
            n  = min(len(yk), len(st))
            if n == 0: continue
            all_yk.append(yk[:n]); all_starts.append(st[:n])
            if mu_ref is None:
                mu_ref     = safe_float(r.get("mu_ref",    0.0))
                sig_ref    = safe_float(r.get("sigma_ref", 1.0))
                c_fail_val = safe_float(r.get("c_fail",    2.0))

        if not all_yk: continue
        yk_all = np.concatenate(all_yk)
        st_all = np.concatenate(all_starts)
        ord_   = np.argsort(st_all)
        yk_all = yk_all[ord_]; st_all = st_all[ord_]

        T_fail, rul, beta = project_rul(yk_all, st_all, mu_ref, sig_ref, T_eval,
                                        c_fail=c_fail_val)
        lo, hi            = project_rul_boot(yk_all, st_all, mu_ref, sig_ref, T_eval,
                                             c_fail=c_fail_val)

        if rul is not None:
            rul_vals.append(rul / T_eval)
            rul_lo.append(lo  / T_eval if np.isfinite(lo)  else np.nan)
            rul_hi.append(hi  / T_eval if np.isfinite(hi)  else np.nan)
            rul_labels.append(f"{c}/{d}")
            rul_colours.append(cond_colour(c))
            forecast_records.append({
                "condition": c, "transition": d,
                "T_fail_steps": T_fail, "RUL_steps": rul,
                "RUL_T_eval_frac": rul / T_eval,
                "RUL_CI_lo_frac": lo / T_eval if np.isfinite(lo) else None,
                "RUL_CI_hi_frac": hi / T_eval if np.isfinite(hi) else None,
                "beta_TS": beta,
            })

    if rul_vals:
        positions = np.arange(len(rul_vals))
        ax.barh(positions, rul_vals, color=rul_colours, alpha=0.7, height=0.55, zorder=2)
        for xi, (v, lo, hi, lab) in enumerate(zip(rul_vals, rul_lo, rul_hi, rul_labels)):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(v, xi, xerr=[[v-lo], [hi-v]],
                            fmt="none", color="#333333", capsize=4, lw=1.5, zorder=3)
            ax.text(max(v, 0) + 0.005, xi, f"{v:.2f}", va="center", fontsize=8)
        ax.set_yticks(positions); ax.set_yticklabels(rul_labels, fontsize=8)
        ax.set_xlabel("Projected RUL / T_eval")
        ax.axvline(0, color="#C0392B", lw=1, ls="--")
        style_ax(ax, title="Theil–Sen RUL  ·  90% bootstrap CI", grid_axis="x")
    else:
        ax.text(0.5, 0.5, "Insufficient data for RUL projection\n(need β_TS < 0)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888")
        style_ax(ax, title="Theil–Sen RUL Projection")

    fig.suptitle("Survival & Remaining Useful Life  ·  Bonus", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [08] {out_path}")

    if save_forecast and forecast_records:
        with open(save_forecast, "w") as f:
            json.dump(forecast_records, f, indent=2)
        print(f"  [forecast] {save_forecast}")

    return forecast_records


def plot_seed_video(recs, out_path, max_panels=4):
    """
    Temporal filmstrip for up to *max_panels* representative records
    (lowest-seed first, one per condition × drift cell).

    Each selected run is drawn as a vertical column of 4 linked subplots:

      Row 0 — D_k (JSD divergence) with τ_warn / τ_watch threshold lines
      Row 1 — y_k (mean window reward) with μ_ref and soft-failure threshold
      Row 2 — Alarm channels: A_watch (yellow), A_deg (red), A_drift (orange)
      Row 3 — Episode rug: tick per episode start, red = terminated (T),
               grey = truncated (C), no colour = termination unknown

    Vertical event lines on all rows:
      T_drift  — drift onset                  (blue dotted)
      T_deg    — degradation zone start       (orange dotted)
      T_warn   — composite alarm fired        (green dashed)
      T_comp   — composite failure (δ=1 only) (red solid)
    """
    seen_cells: dict = {}
    for r in sorted(recs, key=lambda x: x.get("seed", 0)):
        cell = (r.get("condition", "?"), r.get("transition", "?"))
        if cell not in seen_cells:
            seen_cells[cell] = r
        if len(seen_cells) >= max_panels:
            break
    selected = list(seen_cells.values())
    if not selected:
        print("  [09] plot_seed_video: no records to plot"); return

    n_cols = len(selected)
    fig, axes = plt.subplots(4, n_cols, figsize=(5 * n_cols, 11),
                             sharex="col")
    if n_cols == 1:
        axes = axes.reshape(4, 1)

    for col, r in enumerate(selected):
        T_eval = safe_float(r.get("T_eval", 1e6))
        ws      = np.asarray(r.get("window_starts", []), dtype=float)
        Dk      = np.asarray(r.get("D_k",   []), dtype=float)
        yk      = np.asarray(r.get("y_k",   []), dtype=float)
        A_deg   = np.asarray(r.get("A_deg",   []), dtype=int)
        A_drift = np.asarray(r.get("A_drift", []), dtype=int)
        A_watch = np.asarray(r.get("A_watch", []), dtype=int)

        # Trim to shortest common length
        n = min(len(ws), len(Dk), len(yk), len(A_deg), len(A_drift), len(A_watch))
        ws      = ws[:n];  Dk = Dk[:n];  yk = yk[:n]
        A_deg   = A_deg[:n]; A_drift = A_drift[:n]; A_watch = A_watch[:n]

        # Thresholds / reference levels
        tau_warn  = safe_float(r.get("tau_warn"))
        tau_watch = safe_float(r.get("tau_watch"))
        mu_ref    = safe_float(r.get("mu_ref",    0.0))
        sig_ref   = safe_float(r.get("sigma_ref", 1.0))
        c_fail    = safe_float(r.get("c_fail",    2.0))
        soft_thr  = mu_ref - c_fail * sig_ref

        # Key events
        T_warn_v  = safe_float(r.get("T_warn"))
        T_deg_v   = safe_float(r.get("T_deg"))
        T_drift_v = safe_float(r.get("T_drift"))
        T_comp_v  = safe_float(r.get("T_comp_obs", T_eval))
        delta     = int(r.get("delta", 0))

        cond      = r.get("condition", "?")
        drift     = r.get("transition", "?")
        seed      = r.get("seed", 0)
        col_title = f"{cond} · {drift}\nseed={seed}"

        def _vlines(ax):
            if np.isfinite(T_drift_v):
                ax.axvline(T_drift_v, color="#4477CC", lw=1.0, ls=":",  alpha=0.75)
            if np.isfinite(T_deg_v):
                ax.axvline(T_deg_v,   color="#EE8833", lw=1.0, ls=":",  alpha=0.75)
            if np.isfinite(T_warn_v):
                ax.axvline(T_warn_v,  color="#22AA44", lw=1.2, ls="--", alpha=0.90)
            if delta == 1 and np.isfinite(T_comp_v):
                ax.axvline(T_comp_v,  color="#CC3322", lw=1.2, ls="-",  alpha=0.90)

        # ── Row 0: D_k ────────────────────────────────────────────────────
        ax0 = axes[0, col]
        if n > 0:
            ax0.step(ws, Dk, where="post", color="#3A86FF", lw=1.4)
        if np.isfinite(tau_warn):
            ax0.axhline(tau_warn,  color="#CC3322", lw=1.0, ls="--", alpha=0.8)
            ax0.text(T_eval * 0.02, tau_warn,  "τ_warn",  fontsize=7,
                     color="#CC3322", va="bottom")
        if np.isfinite(tau_watch):
            ax0.axhline(tau_watch, color="#FF9500", lw=1.0, ls=":",  alpha=0.75)
            ax0.text(T_eval * 0.02, tau_watch, "τ_watch", fontsize=7,
                     color="#FF9500", va="bottom")
        _vlines(ax0)
        style_ax(ax0, ylabel="D_k  (JSD)", title=col_title, grid_axis="y")

        # ── Row 1: y_k ────────────────────────────────────────────────────
        ax1 = axes[1, col]
        if n > 0:
            ax1.step(ws, yk, where="post", color="#2EC4B6", lw=1.4)
        if np.isfinite(mu_ref):
            ax1.axhline(mu_ref,   color="#22AA44", lw=1.0, ls="--", alpha=0.8)
            ax1.text(T_eval * 0.02, mu_ref,   "μ_ref",   fontsize=7,
                     color="#22AA44", va="bottom")
        if np.isfinite(soft_thr):
            ax1.axhline(soft_thr, color="#CC3322", lw=1.0, ls="--", alpha=0.8)
            ax1.text(T_eval * 0.02, soft_thr, "θ_fail",  fontsize=7,
                     color="#CC3322", va="top")
        _vlines(ax1)
        style_ax(ax1, ylabel="y_k  (reward)", grid_axis="y")

        # ── Row 2: Alarm channels ─────────────────────────────────────────
        ax2 = axes[2, col]
        if n > 0:
            ax2.fill_between(ws, A_watch, step="post",
                             color="#FFD166", alpha=0.45, label="A_watch")
            ax2.fill_between(ws, A_deg,   step="post",
                             color="#EF476F", alpha=0.55, label="A_deg")
            ax2.fill_between(ws, A_drift, step="post",
                             color="#FF6B35", alpha=0.55, label="A_drift")
            ax2.step(ws, A_watch, where="post", color="#CCA700", lw=0.7, alpha=0.8)
            ax2.step(ws, A_deg,   where="post", color="#CC0033", lw=0.7, alpha=0.8)
            ax2.step(ws, A_drift, where="post", color="#CC5500", lw=0.7, alpha=0.8)
        _vlines(ax2)
        ax2.set_ylim(-0.1, 1.5)
        ax2.set_yticks([0, 1])
        style_ax(ax2, ylabel="Alarm", grid_axis="y")
        if col == n_cols - 1:
            ax2.legend(fontsize=7, loc="upper right")

        # ── Row 3: Episode rug ────────────────────────────────────────────
        ax3 = axes[3, col]
        ep_starts     = r.get("episode_starts")    or []
        ep_term_types = r.get("episode_term_types") or []
        if ep_starts:
            has_types = len(ep_term_types) == len(ep_starts)
            colors = []
            for i in range(len(ep_starts)):
                tt = ep_term_types[i] if has_types else "?"
                colors.append("#CC3322" if tt == "T" else "#888888")
            ax3.eventplot(ep_starts, lineoffsets=0.5, linelengths=0.6,
                          linewidths=0.5, colors=colors)
        _vlines(ax3)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([])
        style_ax(ax3, xlabel="Step", ylabel="Episodes", grid_axis="x")

        # Shared x-axis
        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_xlim(0, T_eval)

    # Global legend for vertical event lines
    vline_handles = [
        Line2D([0], [0], color="#4477CC", lw=1.0, ls=":",  label="T_drift"),
        Line2D([0], [0], color="#EE8833", lw=1.0, ls=":",  label="T_deg"),
        Line2D([0], [0], color="#22AA44", lw=1.2, ls="--", label="T_warn"),
        Line2D([0], [0], color="#CC3322", lw=1.2, ls="-",  label="T_comp (δ=1)"),
    ]
    fig.legend(handles=vline_handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Seed Deployment Filmstrip", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [09] {out_path}")
