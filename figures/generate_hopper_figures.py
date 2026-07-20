"""
generate_hopper_figures.py
Generate publication-quality figures for Hopper-v5 transition-detection results.
Covers TWO domains in one run:
  1. hopper_vc    — GravityPerturbation ×1.5,  baseline/abrupt/gradual, M1/M2/M3
  2. hopper_t10v2 — GravityPerturbation ×2.0,  baseline/abrupt/gradual, M1/M2/M3

Condition terminology:
  "baseline"           — no environmental transition (the null/reference condition; used for FAR)
  "Abrupt transition" — gravity step-change at T_drift_abrupt_ref
  "Gradual transition"— gravity ramp starting at T_drift_gradual_ref

"none"/"no_drift" are accepted as legacy aliases for "baseline" in result
filenames (see DOMAINS["baseline_key"] / load_domain()).

Detection logic (see detection_outcome()):
  'detect'      — T_warn fired AND T_warn > T_drift  (genuine post-onset detection)
  'false_alarm' — T_warn fired AND T_warn ≤ T_drift  (pre-onset; same regime as baseline)
  'miss'        — no T_warn before T_comp

NOTE on T_drift: the JSON field "T_drift" (the configured onset step) is a technical
identifier and is NEVER renamed. Condition labels use "transition", not "drift".

Run on NAS (place in ~/msc-thesis/audit-framework/):
    ~/msc-thesis/audit_env/bin/python3 generate_hopper_figures.py

Figures saved to:
    results/mujoco/hopper_vc/figures/fig{1-6}_*.png
    results/mujoco/hopper/t10_v2/figures/fig{1-6}_*.png

Dependencies: matplotlib, numpy, scipy  (install: ~/msc-thesis/audit_env/bin/pip install matplotlib)
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import weibull_min

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", default=None,
                    help="Root of audit-framework (auto-detected if omitted)")
args, _ = parser.parse_known_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = args.base_dir or SCRIPT_DIR

# Fallback to ThesisStorage mount on Steam Deck
if not os.path.isdir(os.path.join(BASE_DIR, "results")):
    BASE_DIR = os.path.expanduser("~/ThesisStorage")

# ── Domain registry ─────────────────────────────────────────────────────────────
DOMAINS = {
    "hopper_vc": {
        "label"      : "hopper_vc  (gravity ×1.5)",
        "results_dir": os.path.join(BASE_DIR, "results", "mujoco", "hopper_vc"),
        "conditions" : ["M1", "M2", "M3"],
        "baseline_key": "no_drift",       # filename term for the baseline condition
        "W"          : 50_000,
        "T_drift_abrupt_ref": 525_000,    # configured onset step for abrupt transition
        "T_drift_gradual_ref": 175_000,   # onset step for gradual transition
        "gravity_label": "1.5×g  (−14.7 m/s²)",
        "color_map"  : {"M1": "#2166AC", "M2": "#D6604D", "M3": "#4DAC26"},
    },
    "hopper_t10v2": {
        "label"      : "hopper t10_v2  (gravity ×2.0)",
        "results_dir": os.path.join(BASE_DIR, "results", "mujoco", "hopper", "t10_v2"),
        "conditions" : ["M1", "M2", "M3"],
        "baseline_key": "none",
        "W"          : 50_000,
        "T_drift_abrupt_ref": 250_000,    # configured onset step (pilot 200k + ramp 50k)
        "T_drift_gradual_ref": 175_000,
        "gravity_label": "2.0×g  (−19.6 m/s²)",
        "color_map"  : {"M1": "#2166AC", "M2": "#D6604D", "M3": "#4DAC26"},
    },
}

SEEDS = list(range(5))

# ── Style ───────────────────────────────────────────────────────────────────────
C_DETECT  = "#4DAC26"    # genuine post-onset detection
C_FA      = "#C0392B"    # pre-onset false alarm (T_warn ≤ T_drift; same regime as baseline)
C_MISS    = "#F1A340"    # missed (no T_warn before T_comp)
C_PREFAIL = "#E66101"    # not used for hopper (no hard_failure_counts)
C_TAU     = "#666666"
C_TDRIFT  = "#B2182B"
C_CUSUM   = "#542788"

STYLE = {
    "font.family"    : "DejaVu Sans",
    "font.size"      : 10,
    "axes.titlesize" : 11,
    "axes.labelsize" : 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi"     : 150,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
}


# ── Data helpers ─────────────────────────────────────────────────────────────────
def load_domain(domain_cfg):
    """Return nested dict: data[condition][seed] = metrics dict."""
    rdir         = domain_cfg["results_dir"]
    baseline_key = domain_cfg["baseline_key"]
    conds_map = {"baseline": baseline_key, "abrupt": "abrupt", "gradual": "gradual"}
    data = {}
    for cond_alias, cond_file in conds_map.items():
        data[cond_alias] = {}
        for m in domain_cfg["conditions"]:
            for s in SEEDS:
                fn = os.path.join(rdir, f"{m}_seed{s}_{cond_file}_metrics_summary.json")
                with open(fn) as fh:
                    data[cond_alias][(m, s)] = json.load(fh)
    return data


def row_labels(conditions):
    """Ordered seed labels: M1_s0..M1_s4, M2_s0..M2_s4, M3_s0..M3_s4."""
    return [f"{m} s{s}" for m in conditions for s in SEEDS]


def detection_outcome(d, t_drift=None):
    """Return ('detect', lead) | ('false_alarm', lead) | ('miss', None).

    Classification:
      'detect'      — T_warn fired AND (t_drift is None OR T_warn > T_drift)
                      Genuine post-onset detection.
      'false_alarm' — T_warn fired AND t_drift is not None AND T_warn <= T_drift
                      Pre-onset alarm. The policy is still in the pre-transition
                      regime, so this is statistically identical to a baseline-
                      condition false alarm. NOT a valid detection.
      'miss'        — no T_warn before T_comp.

    Pass t_drift=dom_cfg["T_drift_abrupt_ref"] for the abrupt condition only.
    For baseline and gradual, pass t_drift=None (no onset guard needed):
      - baseline: no transition exists; any alarm is a FAR by definition.
      - Gradual: T_drift_gradual_ref (175k) < minimum possible T_warn (≥250k),
                 so all timely alarms are genuinely post-onset.
    """
    if d.get("timely_warn") and d.get("T_warn") is not None:
        t_comp = d.get("T_comp") or d.get("T_hard")
        lead   = (t_comp - d["T_warn"]) if t_comp else None
        # Pre-onset guard: if T_warn fired before or at the transition onset,
        # the environment was in its pre-transition state — same as baseline condition.
        if t_drift is not None and d["T_warn"] <= t_drift:
            return "false_alarm", lead
        return "detect", lead
    return "miss", None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIG 1 – FAR Comparison                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fig1_far_comparison(dom_id, dom_cfg, data, figures_dir):
    n = len(dom_cfg["conditions"]) * len(SEEDS)   # 15

    dk_fa    = sum(1 for key in data["baseline"] if data["baseline"][key].get("T_warn") is not None)
    cusum_fa = sum(1 for key in data["baseline"] if data["baseline"][key].get("T_cusum") is not None)

    channels = ["D_k  (JSD)", "CUSUM"]
    counts   = [dk_fa, cusum_fa]
    pcts     = [100 * c / n for c in counts]
    colors   = [C_DETECT if p < 5 else C_PREFAIL if p < 60 else "#CA0020"
                for p in pcts]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        bars = ax.bar(channels, pcts, color=colors, width=0.45,
                      edgecolor="white", linewidth=1.2)
        for bar, p, c in zip(bars, pcts, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{c}/{n}  ({p:.0f}%)",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylim(0, 118)
        ax.set_ylabel("False Alarm Rate  (%)")
        ax.set_title(f"Fig 1  –  Channel FAR  |  {dom_cfg['label']}\n"
                     f"Baseline condition  |  n={n} seeds  |  1 M-step deployment")
        ax.axhline(5, color=C_TAU, linestyle="--", linewidth=1,
                   label="Target FAR ≤ 5%")
        ax.legend(loc="upper left")

        fig.tight_layout()
        out = os.path.join(figures_dir, "fig1_far_comparison.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIG 2 – D_k Trajectories                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fig2_dk_trajectories(dom_id, dom_cfg, data, figures_dir):
    W           = dom_cfg["W"]
    cmap        = dom_cfg["color_map"]
    t_drift_ab  = dom_cfg["T_drift_abrupt_ref"]
    t_drift_gr  = dom_cfg["T_drift_gradual_ref"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)

        for ax, (cond_alias, cond_label) in zip(
                axes,
                [("baseline", "Baseline"),
                 ("abrupt",  f"Abrupt transition  (T_onset={t_drift_ab//1e3:.0f} k)"),
                 ("gradual", f"Gradual transition  (T_onset={t_drift_gr//1e3:.0f} k)")]):

            for m in dom_cfg["conditions"]:
                col = cmap[m]
                for s in SEEDS:
                    d  = data[cond_alias][(m, s)]
                    ws = np.array(d["window_starts"])
                    dk = np.array(d["D_k"])
                    we = ws + W
                    K0 = d.get("K0", 4)
                    ax.plot(we[K0:] / 1e3, dk[K0:],
                            color=col, alpha=0.5, linewidth=1.0)

            # Per-seed tau_warn varies; plot the minimum (floor) as reference
            taus = [data[cond_alias][(m, s)].get("tau_warn", 0)
                    for m in dom_cfg["conditions"] for s in SEEDS]
            tau_min = min(taus)
            ax.axhline(tau_min, color=C_TAU, linestyle="--", linewidth=1.0,
                       label=f"τ_warn floor ≈ {tau_min:.4f}")

            if cond_alias == "abrupt":
                ax.axvline(t_drift_ab / 1e3, color=C_TDRIFT,
                           linestyle=":", linewidth=1.3, label="Transition onset  (T_drift)")
            elif cond_alias == "gradual":
                ax.axvline(t_drift_gr / 1e3, color=C_TDRIFT,
                           linestyle=":", linewidth=1.3, label="Transition onset  (T_drift)")

            ax.set_title(cond_label, pad=6)
            ax.set_xlabel("Deployment step  (× 10³)")
            ax.legend(loc="upper left", handlelength=1.5, fontsize=8)

        axes[0].set_ylabel("D_k  (Jensen-Shannon divergence)")
        axes[0].set_ylim(-0.00003, 0.001)

        # Shared y for transition panels
        d_max = 0
        for cond_alias in ("abrupt", "gradual"):
            for m in dom_cfg["conditions"]:
                for s in SEEDS:
                    d_max = max(d_max, max(data[cond_alias][(m, s)]["D_k"]))
        for ax in axes[1:]:
            ax.set_ylim(-0.0005, d_max * 1.1)

        legend_lines = [Line2D([0], [0], color=cmap[m], lw=1.5, label=m)
                        for m in dom_cfg["conditions"]]
        fig.legend(handles=legend_lines, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.04), frameon=False)

        fig.suptitle(f"Fig 2  –  D_k Trajectory by Condition  |  {dom_cfg['label']}",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        out = os.path.join(figures_dir, "fig2_dk_trajectories.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIG 3 – Detection Outcome Matrix                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fig3_detection_matrix(dom_id, dom_cfg, data, figures_dir):
    n_total = len(dom_cfg["conditions"]) * len(SEEDS)   # 15
    labels  = row_labels(dom_cfg["conditions"])

    # Three outcomes: genuine detection, pre-onset false alarm, missed
    COLOR_MAP = {
        "detect"     : C_DETECT,   # green  — genuine post-onset detection
        "false_alarm": C_FA,        # dark red — pre-onset FA (same regime as baseline)
        "miss"       : C_MISS,     # amber  — missed
    }
    CELL_TEXT = {
        "detect"     : lambda lead: f"Δ {lead/1e3:.0f}k" if lead is not None else "✓",
        "false_alarm": lambda lead: "FA",    # pre-onset false alarm
        "miss"       : lambda lead: "Missed",
    }

    t_drift_ab = dom_cfg["T_drift_abrupt_ref"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7, 7))

        for ax, cond_alias in zip(axes, ["abrupt", "gradual"]):
            # Pass t_drift only for abrupt; gradual alarms are all post-onset
            t_drift_arg = t_drift_ab if cond_alias == "abrupt" else None

            colors_col = []
            texts_col  = []
            n_genuine  = 0   # count genuine post-onset detections only

            for m in dom_cfg["conditions"]:
                for s in SEEDS:
                    d = data[cond_alias][(m, s)]
                    status, lead = detection_outcome(d, t_drift=t_drift_arg)
                    colors_col.append(COLOR_MAP[status])
                    texts_col.append(CELL_TEXT[status](lead))
                    if status == "detect":
                        n_genuine += 1

            for idx, (color, txt) in enumerate(zip(colors_col, texts_col)):
                rect = plt.Rectangle((0, n_total - 1 - idx), 1, 1,
                                      color=color, ec="white", lw=1.5)
                ax.add_patch(rect)
                ax.text(0.5, n_total - 1 - idx + 0.5, txt,
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, n_total)
            ax.set_xticks([])
            ax.set_yticks([i + 0.5 for i in range(n_total)])
            ax.set_yticklabels(labels[::-1], fontsize=8.5)

            # Dividers between M1/M2/M3 groups
            for div in [5, 10]:
                ax.axhline(div, color="white", linewidth=2.5)

            t_ref = (dom_cfg["T_drift_abrupt_ref"] if cond_alias == "abrupt"
                     else dom_cfg["T_drift_gradual_ref"])
            cond_title = "Abrupt transition" if cond_alias == "abrupt" else "Gradual transition"
            ax.set_title(
                f"{cond_title}  (T_drift={t_ref//1e3:.0f} k)\n"
                f"{n_genuine}/{n_total} genuine detections",
                pad=6)

        patches = [
            mpatches.Patch(color=C_DETECT, label="Genuine detection  (post-onset, Δ = lead)"),
            mpatches.Patch(color=C_FA,     label="Pre-onset false alarm  (T_warn ≤ T_drift)"),
            mpatches.Patch(color=C_MISS,   label="Missed"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=1,
                   bbox_to_anchor=(0.5, -0.07), frameon=True, edgecolor="#AAAAAA")

        fig.suptitle(f"Fig 3  –  Detection Outcome Matrix  |  D_k  |  {dom_cfg['label']}",
                     y=1.01, fontsize=11)
        fig.tight_layout()
        out = os.path.join(figures_dir, "fig3_detection_matrix.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIG 4 – Lead Time Distribution (genuine post-onset detections only)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fig4_lead_times(dom_id, dom_cfg, data, figures_dir):
    cmap       = dom_cfg["color_map"]
    t_drift_ab = dom_cfg["T_drift_abrupt_ref"]
    conditions = dom_cfg["conditions"]
    n_groups   = len(conditions)

    def get_genuine_leads(cond_alias, t_drift=None):
        """Collect lead times for GENUINE detections only (T_warn > T_drift)."""
        leads = {m: [] for m in conditions}
        for m in conditions:
            for s in SEEDS:
                d = data[cond_alias][(m, s)]
                status, lead = detection_outcome(d, t_drift=t_drift)
                if status == "detect" and lead is not None:
                    leads[m].append(lead / 1e3)
        return leads

    # Abrupt: filter by T_drift_abrupt_ref; gradual: no filter needed
    abrupt_leads  = get_genuine_leads("abrupt",  t_drift=t_drift_ab)
    gradual_leads = get_genuine_leads("gradual", t_drift=None)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

        for ax, (cond_alias, leads_dict, title) in zip(
                axes,
                [("abrupt",  abrupt_leads,  "Abrupt transition"),
                 ("gradual", gradual_leads, "Gradual transition")]):

            total_genuine = sum(len(v) for v in leads_dict.values())

            if total_genuine == 0:
                # No genuine post-onset detections — explain why
                t_ref = (t_drift_ab if cond_alias == "abrupt"
                         else dom_cfg["T_drift_gradual_ref"])
                ax.text(0.5, 0.5,
                        f"No genuine post-onset detections\n"
                        f"(all D_k alarms fired at T_warn ≤ T_drift = {t_ref//1e3:.0f} k\n"
                        f" — pre-onset false alarms, same regime as baseline condition)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.6",
                                  facecolor="#F5F5F5", edgecolor="#AAAAAA"))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title)
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
            else:
                positions = list(range(n_groups))
                for pos, m in zip(positions, conditions):
                    vals = leads_dict[m]
                    if not vals:
                        continue
                    jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
                    ax.scatter(np.full(len(vals), pos) + jitter, vals,
                               color=cmap[m], s=60, zorder=3,
                               edgecolors="white", linewidths=0.5, label=m)
                    ax.hlines(np.mean(vals), pos - 0.15, pos + 0.15,
                              color=cmap[m], linewidth=2.5, zorder=4)
                    ax.text(pos, max(vals) + 3,
                            f"μ={np.mean(vals):.0f}k",
                            ha="center", fontsize=8, color="#444")

                ax.set_xticks(positions)
                ax.set_xticklabels(conditions)
                ax.set_ylabel("Lead time  Δ = T_comp – T_warn  (× 10³ steps)")
                ax.set_title(title)
                ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(f"Fig 4  –  Detection Lead Time  |  {dom_cfg['label']}\n"
                     "Genuine post-onset detections only  (T_warn > T_drift);  bar = mean",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        out = os.path.join(figures_dir, "fig4_lead_times.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIG 5 – Weibull Survival Curves                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fig5_weibull_survival(dom_id, dom_cfg, data, figures_dir):
    cmap = dom_cfg["color_map"]

    def t_comp_values(cond_alias):
        """Return dict: model → list of T_comp (time-to-failure)."""
        out = {m: [] for m in dom_cfg["conditions"]}
        for m in dom_cfg["conditions"]:
            for s in SEEDS:
                d = data[cond_alias][(m, s)]
                tc = d.get("T_comp") or d.get("T_hard")
                if tc:
                    out[m].append(tc)
        return out

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for ax, cond_alias in zip(axes, ["abrupt", "gradual"]):
            comps = t_comp_values(cond_alias)
            t_max = max(max(v) for v in comps.values() if v) * 1.15

            for m in dom_cfg["conditions"]:
                vals = np.array(comps[m])
                if len(vals) < 2:
                    continue
                col = cmap[m]
                try:
                    k, _, lam = weibull_min.fit(vals, floc=0)
                    t_arr = np.linspace(0, t_max, 2000)
                    S = np.exp(-(t_arr / lam) ** k)
                    ax.plot(t_arr / 1e3, S, color=col, linewidth=1.8,
                            label=f"{m}  k={k:.1f}, λ={lam/1e3:.0f}k")
                except Exception:
                    pass

                # Rug marks at empirical T_comp
                for tc in vals:
                    ax.plot(tc / 1e3, 0.02, "|",
                            color=col, markersize=10, markeredgewidth=1.5, alpha=0.8)

            ax.axhline(0.10, color=C_TAU, linestyle=":", linewidth=1,
                       label="S(t)=0.10  (RUL_90)")
            ax.set_xlim(0, t_max / 1e3)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("Deployment step  (× 10³)")
            ax.set_ylabel("Survival  S(t)")
            cond_title = "Abrupt transition" if cond_alias == "abrupt" else "Gradual transition"
            ax.set_title(cond_title)
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(f"Fig 5  –  Weibull Survival Curves  |  {dom_cfg['label']}\n"
                     "S(t)=exp(−(t/λ)^k);  rug marks = empirical T_comp",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        out = os.path.join(figures_dir, "fig5_weibull_survival.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIG 6 – Alarm Timeline (abrupt condition)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fig6_alarm_timelines(dom_id, dom_cfg, data, figures_dir):
    cmap       = dom_cfg["color_map"]
    t_drift_ab = dom_cfg["T_drift_abrupt_ref"]
    labels     = row_labels(dom_cfg["conditions"])
    n_total    = len(labels)   # 15

    ev_cusum = []
    ev_warn  = []
    ev_comp  = []
    ev_cond  = []

    for m in dom_cfg["conditions"]:
        for s in SEEDS:
            d = data["abrupt"][(m, s)]
            ev_cusum.append(d.get("T_cusum"))
            ev_warn.append(d.get("T_warn"))
            ev_comp.append(d.get("T_comp") or d.get("T_hard"))
            ev_cond.append(m)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 7))

        # Shade pre-onset zone (before T_drift): same regime as baseline condition
        ax.axvspan(0, t_drift_ab / 1e3, alpha=0.08, color=C_TDRIFT, zorder=0,
                   label=f"Pre-onset zone  (T < T_drift = {t_drift_ab//1e3:.0f} k)")

        for idx, (lbl, t_cu, t_wn, t_cp, m) in enumerate(
                zip(labels, ev_cusum, ev_warn, ev_comp, ev_cond)):

            y       = n_total - 1 - idx
            col     = cmap[m]
            bar_end = t_cp if t_cp else 600_000

            ax.barh(y, bar_end / 1e3, height=0.5,
                    color="#EEEEEE", edgecolor="#CCCCCC", linewidth=0.4)

            if t_cu:
                ax.plot(t_cu / 1e3, y, "s", color=C_CUSUM, markersize=7, zorder=5)

            if t_wn:
                # Distinguish pre-onset false alarms from genuine post-onset alarms
                is_pre_onset = (t_wn <= t_drift_ab)
                warn_color  = C_FA if is_pre_onset else C_DETECT
                ax.plot(t_wn / 1e3, y, "o",
                        color=warn_color, markersize=8, zorder=6)

            if t_cp:
                ax.plot(t_cp / 1e3, y, "X", color="#1A1A1A", markersize=9, zorder=5)

        # Transition onset line
        ax.axvline(t_drift_ab / 1e3, color=C_TDRIFT, linestyle="--",
                   linewidth=1.5, zorder=7)

        ax.set_yticks(list(range(n_total)))
        ax.set_yticklabels(labels[::-1], fontsize=8.5)
        ax.set_xlabel("Deployment step  (× 10³)")
        ax.set_title(
            f"Fig 6  –  Alarm Timeline — Abrupt Transition  |  {dom_cfg['label']}\n"
            "Alarms in shaded zone are pre-onset (same regime as baseline condition)"
        )

        # Group dividers
        for div_y in [5, 10]:
            ax.axhline(div_y - 0.5, color="#CCCCCC", linewidth=1.5)
        for grp_y, grp_lbl in zip([12, 7, 2], dom_cfg["conditions"]):
            ax.text(2, grp_y, grp_lbl, fontsize=9, color="#555")

        legend_el = [
            mpatches.Patch(color=C_TDRIFT, alpha=0.15,
                           label=f"Pre-onset zone  (T < T_drift = {t_drift_ab//1e3:.0f} k)"),
            Line2D([0], [0], color=C_TDRIFT, linestyle="--", lw=1.5,
                   label="Transition onset  (T_drift)"),
            Line2D([0], [0], marker="s", color=C_CUSUM, lw=0,
                   markersize=7, label="T_cusum  (CUSUM alarm)"),
            Line2D([0], [0], marker="o", color=C_FA, lw=0,
                   markersize=8, label="T_warn  (pre-onset false alarm, ≤ T_drift)"),
            Line2D([0], [0], marker="o", color=C_DETECT, lw=0,
                   markersize=8, label="T_warn  (genuine detection, > T_drift)"),
            Line2D([0], [0], marker="X", color="#1A1A1A", lw=0,
                   markersize=9, label="T_comp  (composite failure)"),
        ]
        ax.legend(handles=legend_el, loc="lower right", fontsize=8.5,
                  frameon=True, edgecolor="#AAAAAA")

        t_max_comp = max(e for e in ev_comp if e) / 1e3
        ax.set_xlim(0, max(t_max_comp * 1.05, t_drift_ab / 1e3 * 1.3))
        fig.tight_layout()
        out = os.path.join(figures_dir, "fig6_alarm_timelines.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    for dom_id, dom_cfg in DOMAINS.items():
        figures_dir = os.path.join(dom_cfg["results_dir"], "figures")
        os.makedirs(figures_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Domain: {dom_cfg['label']}")
        print(f"  Results: {dom_cfg['results_dir']}")
        print(f"  Figures: {figures_dir}")
        print(f"{'='*60}")

        try:
            data = load_domain(dom_cfg)
        except FileNotFoundError as exc:
            print(f"  ERROR loading data: {exc}")
            print("  Skipping this domain.")
            continue

        fig1_far_comparison   (dom_id, dom_cfg, data, figures_dir)
        fig2_dk_trajectories  (dom_id, dom_cfg, data, figures_dir)
        fig3_detection_matrix (dom_id, dom_cfg, data, figures_dir)
        fig4_lead_times       (dom_id, dom_cfg, data, figures_dir)
        fig5_weibull_survival (dom_id, dom_cfg, data, figures_dir)
        fig6_alarm_timelines  (dom_id, dom_cfg, data, figures_dir)

    print("\nAll hopper figures generated.")
