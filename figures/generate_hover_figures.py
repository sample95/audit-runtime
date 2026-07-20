"""
generate_hover_figures.py
Generate publication-quality figures for PyFlyt/QuadX-Hover-v4 transition-detection results.

Condition terminology:
  "baseline"            — no environmental transition (the null/reference condition; used for FAR)
  "Abrupt transition"  — wind perturbation step-change at T_DRIFT_ABRUPT
  "Gradual transition" — wind perturbation ramp starting at T_DRIFT_GRADUAL

"none"/"no_drift" are accepted as legacy aliases for "baseline" when loading
pre-rename result filenames (see load()).

Detection logic (see outcome()):
  'detect'      — T_warn fired AND T_warn > T_drift  (genuine post-onset detection)
  'false_alarm' — T_warn fired AND T_warn ≤ T_drift  (pre-onset; same regime as baseline)
  'prefail'     — policy collapsed before transition injection (T_hard present, no T_drift)
  'miss'        — no T_warn before T_hard

NOTE on D_k geometry (abrupt): W=30k, K0=4 → earliest possible T_warn = 150k.
Window boundaries: 150k, 180k, …, 300k, 330k, …
T_DRIFT_ABRUPT = 315k; T_hard ≈ 325k.
First window boundary AFTER T_drift (330k) exceeds T_hard (325k) → D_k cannot
produce a genuine post-onset detection in the abrupt condition.
CUSUM and OCSVM (continuous monitors) are not window-bounded and may detect
genuinely post-onset; these are plotted separately in fig6.

Run on NAS (place script in ~/msc-thesis/audit-framework/):
    ~/msc-thesis/audit_env/bin/python3 generate_hover_figures.py

Override results dir if running from a different location:
    ~/msc-thesis/audit_env/bin/python3 generate_hover_figures.py \
        --results-dir /path/to/results/pyflyt/hover

Figures saved to:  <results_dir>/figures/
    fig1_far_comparison.png
    fig2_dk_trajectories.png
    fig3_detection_matrix.png
    fig4_lead_times.png
    fig5_weibull_survival.png
    fig6_alarm_timelines.png

Dependencies: matplotlib, numpy, scipy  (all in audit_env)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.special import gamma as gamma_fn

import argparse
parser = argparse.ArgumentParser(description="Generate hover transition-detection figures")
parser.add_argument("--results-dir", default=None,
                    help="Path to results/pyflyt/hover/  (default: auto-detect)")
args, _ = parser.parse_known_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if args.results_dir:
    RESULTS_DIR = args.results_dir
else:
    # Try alongside the script first, then fall back to ThesisStorage mount
    candidate = os.path.join(SCRIPT_DIR, "results", "pyflyt", "hover")
    fallback   = os.path.expanduser("~/ThesisStorage/results/pyflyt/hover")
    RESULTS_DIR = candidate if os.path.isdir(candidate) else fallback

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CONDITIONS        = ["baseline", "abrupt", "gradual"]
# "baseline" is canonical going forward; "none"/"no_drift" are pre-rename
# aliases still present in older result filenames.
_BASELINE_ALIASES = ("baseline", "none", "no_drift")
QUALITIES         = ["Q1", "Q2"]
SEEDS             = list(range(5))
W                 = 30_000    # window width (steps)
T_DRIFT_ABRUPT    = 315_000   # configured abrupt transition onset step
T_DRIFT_GRADUAL   = 105_000   # configured gradual transition onset step
# NOTE: min possible T_warn = (K0+1)*W = 5*30k = 150k
#   For abrupt:  150k < 315k — all D_k alarms are pre-onset false alarms
#   For gradual: 150k > 105k — all D_k alarms are genuine post-onset detections

# ── Colour / style constants ────────────────────────────────────────────────────
C_Q1       = "#2166AC"   # blue
C_Q2       = "#D6604D"   # red-orange
C_DETECT   = "#4DAC26"   # green  — genuine post-onset detection
C_FA       = "#C0392B"   # dark red — pre-onset false alarm (T_warn ≤ T_drift)
C_MISS     = "#F1A340"   # amber
C_PREFAIL  = "#E66101"   # dark orange — policy collapsed before transition
C_TAU      = "#666666"   # grey dashed threshold line
C_TDRIFT   = "#B2182B"   # red vertical transition line

STYLE = {
    "font.family"   : "DejaVu Sans",
    "font.size"     : 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi"    : 150,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
}


# ── Data loader ─────────────────────────────────────────────────────────────────
def load(quality, seed, condition):
    keys = _BASELINE_ALIASES if condition == "baseline" else (condition,)
    for k in keys:
        path = os.path.join(RESULTS_DIR,
                            f"{quality}_seed{seed}_{k}_metrics_summary.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    raise FileNotFoundError(
        f"No metrics file for {quality} seed{seed} condition={condition!r} "
        f"(tried filename suffixes: {keys})"
    )

def load_all():
    data = {}
    for cond in CONDITIONS:
        for q in QUALITIES:
            for s in SEEDS:
                data[(q, s, cond)] = load(q, s, cond)
    return data

ALL = load_all()


def outcome(d, t_drift_ref=None):
    """Return ('detect', lead) | ('false_alarm', lead) | ('prefail', None) | ('miss', None).

    Classification:
      'detect'      — T_warn fired AND (no t_drift_ref OR T_warn > t_drift_ref)
                      Genuine post-onset detection.
      'false_alarm' — T_warn fired AND T_warn ≤ t_drift_ref
                      Pre-onset alarm. Policy is still in pre-transition state —
                      statistically identical to a baseline-condition false alarm.
      'prefail'     — T_hard present but no T_drift in JSON
                      Policy collapsed before transition was injected.
      'miss'        — no T_warn before T_hard.

    Pass t_drift_ref=T_DRIFT_ABRUPT for abrupt condition only.
    For gradual, pass t_drift_ref=None: T_DRIFT_GRADUAL (105k) < min possible
    T_warn (150k), so all gradual alarms are already genuinely post-onset.
    """
    t_hard  = d.get("T_hard")
    t_warn  = d.get("T_warn")
    t_drift = d.get("T_drift")   # JSON field (may be None if not injected yet)
    timely  = d.get("timely_warn", False)

    if timely and t_warn is not None:
        lead = (t_hard - t_warn) if t_hard else None
        # Pre-onset guard
        if t_drift_ref is not None and t_warn <= t_drift_ref:
            return "false_alarm", lead
        return "detect", lead
    elif t_hard is not None and t_drift is None:
        # Policy collapsed before transition injection
        return "prefail", None
    else:
        return "miss", None


def fig1_far_comparison():
    """Grouped bar chart: channel FAR in the baseline condition."""

    dk_fa    = sum(1 for q in QUALITIES for s in SEEDS
                   if ALL[(q, s, "baseline")].get("T_warn") is not None)
    cusum_fa = sum(1 for q in QUALITIES for s in SEEDS
                   if ALL[(q, s, "baseline")].get("T_cusum") is not None)
    ocsvm_fa = sum(1 for q in QUALITIES for s in SEEDS
                   if ALL[(q, s, "baseline")].get("T_ocsvm") is not None)

    n = len(QUALITIES) * len(SEEDS)   # 10
    channels = ["D_k (JSD)", "CUSUM", "OCSVM"]
    counts   = [dk_fa, cusum_fa, ocsvm_fa]
    pcts     = [100 * c / n for c in counts]
    colors   = [C_DETECT if p == 0 else C_PREFAIL if p < 100 else "#CA0020"
                for p in pcts]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        bars = ax.bar(channels, pcts, color=colors, width=0.5,
                      edgecolor="white", linewidth=1.2)

        for bar, p, c in zip(bars, pcts, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{c}/{n}  ({p:.0f}%)",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylim(0, 118)
        ax.set_ylabel("False Alarm Rate  (%)")
        ax.set_title("Fig 1  –  Channel False Alarm Rate\n"
                     "Baseline condition  |  n = 10 seeds  |  600 k-step deployment")
        ax.axhline(5, color=C_TAU, linestyle="--", linewidth=1,
                   label="Target FAR ≤ 5%")
        ax.legend(loc="upper left")

        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, "fig1_far_comparison.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


def fig2_dk_trajectories():
    """Time-series of D_k per condition; all 10 seeds overlaid."""

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)

        for ax, cond in zip(axes, CONDITIONS):
            for q, col in zip(QUALITIES, [C_Q1, C_Q2]):
                for s in SEEDS:
                    d = ALL[(q, s, cond)]
                    ws = np.array(d["window_starts"])
                    dk = np.array(d["D_k"])
                    we = ws + W

                    K0 = d.get("K0", 4)
                    ax.plot(we[K0:] / 1e3, dk[K0:],
                            color=col, alpha=0.55, linewidth=1.0)

            # Threshold line
            tau = ALL[("Q1", 0, cond)].get("tau_warn", 0.008)
            ax.axhline(tau, color=C_TAU, linestyle="--", linewidth=1.0,
                       label=f"τ_warn = {tau:.3f}")

            # Transition onset line (abrupt only — known injection step)
            if cond == "abrupt":
                ax.axvline(T_DRIFT_ABRUPT / 1e3, color=C_TDRIFT,
                           linestyle=":", linewidth=1.3,
                           label="Transition onset  (T_drift)")

            cond_label = {
                "baseline": "Baseline",
                "abrupt" : f"Abrupt transition  (σ=0.2, T_onset={T_DRIFT_ABRUPT//1e3:.0f} k)",
                "gradual": "Gradual transition  (σ ramp, post-pilot)",
            }[cond]
            ax.set_title(cond_label, pad=6)
            ax.set_xlabel("Deployment step  (× 10³)")
            ax.legend(loc="upper left", handlelength=1.5)

        axes[0].set_ylabel("D_k  (Jensen-Shannon divergence)")

        for ax, cond in zip(axes, CONDITIONS):
            if cond == "baseline":
                ax.set_ylim(-0.0005, 0.012)
            elif cond in ("abrupt", "gradual"):
                ax.set_ylim(-0.005, 0.50)

        legend_lines = [
            Line2D([0], [0], color=C_Q1, linewidth=1.5, label="Q1 (10 s episodes)"),
            Line2D([0], [0], color=C_Q2, linewidth=1.5, label="Q2 (2.5 s episodes)"),
        ]
        fig.legend(handles=legend_lines, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.04), frameon=False)

        fig.suptitle("Fig 2  –  D_k (JSD) Trajectory by Condition  |  PyFlyt QuadX-Hover",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, "fig2_dk_trajectories.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


def fig3_detection_matrix():
    """
    Heatmap: rows = 10 seeds, columns = abrupt / gradual.
    Green = genuine post-onset detect, dark-red = pre-onset FA,
    amber = missed, dark-orange = pre-transition failure.
    """
    row_labels_list = [f"Q1 s{s}" for s in SEEDS] + [f"Q2 s{s}" for s in SEEDS]

    COLOR_MAP = {
        "detect"     : C_DETECT,   # green  — genuine post-onset detection
        "false_alarm": C_FA,        # dark red — pre-onset FA (same regime as baseline)
        "prefail"    : C_PREFAIL,  # dark orange — collapsed before transition
        "miss"       : C_MISS,     # amber  — missed
    }
    CELL_TEXT = {
        "detect"     : lambda lead: f"Δ {lead/1e3:.1f}k" if lead is not None else "✓",
        "false_alarm": lambda lead: "FA",
        "prefail"    : lambda lead: "Pre-onset\nfailure",
        "miss"       : lambda lead: "Missed",
    }

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8, 5.5))

        for ax, cond in zip(axes, ["abrupt", "gradual"]):
            # Pre-onset guard only needed for abrupt
            t_drift_ref = T_DRIFT_ABRUPT if cond == "abrupt" else None

            matrix_color = []
            matrix_text  = []
            n_genuine    = 0

            for q in QUALITIES:
                for s in SEEDS:
                    d = ALL[(q, s, cond)]
                    status, lead = outcome(d, t_drift_ref=t_drift_ref)
                    matrix_color.append(COLOR_MAP[status])
                    matrix_text.append(CELL_TEXT[status](lead))
                    if status == "detect":
                        n_genuine += 1

            n_total = len(QUALITIES) * len(SEEDS)

            for idx, (color, txt) in enumerate(zip(matrix_color, matrix_text)):
                row = idx
                rect = plt.Rectangle((0, 9 - row), 1, 1,
                                      color=color, ec="white", lw=1.5)
                ax.add_patch(rect)
                ax.text(0.5, 9 - row + 0.5, txt,
                        ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 10)
            ax.set_xticks([])
            ax.set_yticks([i + 0.5 for i in range(10)])
            ax.set_yticklabels(row_labels_list[::-1], fontsize=9)

            # Divider between Q1 / Q2
            ax.axhline(5, color="white", linewidth=2.5)

            t_ref   = T_DRIFT_ABRUPT if cond == "abrupt" else T_DRIFT_GRADUAL
            title   = "Abrupt transition" if cond == "abrupt" else "Gradual transition"
            ax.set_title(f"{title}  (T_drift = {t_ref//1e3:.0f} k)\n"
                         f"{n_genuine}/{n_total} genuine detections",
                         pad=8)

        patches = [
            mpatches.Patch(color=C_DETECT,  label="Genuine detection  (post-onset, Δ = lead)"),
            mpatches.Patch(color=C_FA,      label="Pre-onset false alarm  (T_warn ≤ T_drift)"),
            mpatches.Patch(color=C_PREFAIL, label="Pre-transition failure  (T_hard < T_drift)"),
            mpatches.Patch(color=C_MISS,    label="Missed"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=1,
                   bbox_to_anchor=(0.5, -0.09), frameon=True, edgecolor="#AAAAAA")

        fig.suptitle("Fig 3  –  Detection Outcome Matrix  |  D_k Channel  |  PyFlyt QuadX-Hover",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, "fig3_detection_matrix.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


def fig4_lead_times():
    """Strip-plot of lead times for GENUINE post-onset detected seeds only."""

    abrupt_leads  = {q: [] for q in QUALITIES}
    gradual_leads = {q: [] for q in QUALITIES}

    for q in QUALITIES:
        for s in SEEDS:
            for cond, bucket, t_drift_ref in [
                    ("abrupt",  abrupt_leads,  T_DRIFT_ABRUPT),
                    ("gradual", gradual_leads, None)]:
                d = ALL[(q, s, cond)]
                status, _ = outcome(d, t_drift_ref=t_drift_ref)
                if status == "detect" and d.get("T_hard") and d.get("T_warn"):
                    lead = (d["T_hard"] - d["T_warn"]) / 1e3
                    bucket[q].append(lead)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

        for ax, (cond, leads_dict, title, t_ref) in zip(
                axes,
                [("abrupt",  abrupt_leads,  "Abrupt transition",  T_DRIFT_ABRUPT),
                 ("gradual", gradual_leads, "Gradual transition", T_DRIFT_GRADUAL)]):

            total_genuine = sum(len(v) for v in leads_dict.values())

            if total_genuine == 0:
                ax.text(0.5, 0.5,
                        f"No genuine post-onset detections\n"
                        f"(all D_k alarms fired at T_warn ≤ T_drift = {t_ref//1e3:.0f} k\n"
                        f" — pre-onset false alarms, same regime as baseline condition)\n\n"
                        f"Note: CUSUM/OCSVM may detect post-onset\n"
                        f"(see fig6 alarm timeline)",
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
                positions = [1, 1.5]
                colors    = [C_Q1, C_Q2]

                for pos, col, q in zip(positions, colors, QUALITIES):
                    vals = leads_dict[q]
                    if not vals:
                        continue
                    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(vals))
                    ax.scatter(np.full(len(vals), pos) + jitter, vals,
                               color=col, s=60, zorder=3,
                               edgecolors="white", linewidths=0.5, label=q)
                    ax.hlines(np.mean(vals), pos - 0.12, pos + 0.12,
                              color=col, linewidth=2.5, zorder=4)
                    ax.text(pos, max(vals) + 3, f"μ={np.mean(vals):.1f}k",
                            ha="center", fontsize=8, color="#444444")

                ax.set_xticks(positions)
                ax.set_xticklabels(["Q1", "Q2"])
                ax.set_ylabel("Lead time  Δ = T_hard – T_warn  (× 10³ steps)")
                ax.set_title(title)

                legend_el = [Line2D([0], [0], color=C_Q1, lw=2, label="Q1 (10 s episodes)"),
                             Line2D([0], [0], color=C_Q2, lw=2, label="Q2 (2.5 s episodes)")]
                ax.legend(handles=legend_el, loc="upper left")

        fig.suptitle("Fig 4  –  Detection Lead Time  |  PyFlyt QuadX-Hover\n"
                     "Genuine post-onset detections only  (T_warn > T_drift);  bar = mean",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, "fig4_lead_times.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


def fig5_weibull_survival():
    """
    Weibull survival S(t) = exp(-(t/λ)^k) for each condition × quality.
    Empirical T_hard values shown as rug marks.
    """
    # Pre-fitted Weibull MLE parameters from audit results
    weibull = {
        ("abrupt",  "Q1"): dict(k=31.2,  lam=324_816),
        ("abrupt",  "Q2"): dict(k=110.2, lam=327_007),
        ("gradual", "Q1"): dict(k=2.93,  lam=125_089),
        ("gradual", "Q2"): dict(k=8.83,  lam=174_529),
    }

    empirical_T_hard = {}
    for cond in ["abrupt", "gradual"]:
        for q in QUALITIES:
            empirical_T_hard[(cond, q)] = [
                ALL[(q, s, cond)]["T_hard"]
                for s in SEEDS
                if ALL[(q, s, cond)].get("T_hard") is not None
            ]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        for ax, cond in zip(axes, ["abrupt", "gradual"]):
            t_max = 420_000 if cond == "abrupt" else 250_000
            t_arr = np.linspace(0, t_max, 2000)

            for q, col, ls in zip(QUALITIES, [C_Q1, C_Q2], ["-", "--"]):
                p = weibull[(cond, q)]
                S = np.exp(-(t_arr / p["lam"]) ** p["k"])
                lbl = (f"{q}  (k={p['k']:.1f}, "
                       f"λ={p['lam']/1e3:.0f}k)")
                ax.plot(t_arr / 1e3, S, color=col, linestyle=ls,
                        linewidth=1.8, label=lbl)

                for th in empirical_T_hard[(cond, q)]:
                    ax.plot(th / 1e3, 0.02, "|",
                            color=col, markersize=10, markeredgewidth=1.5, alpha=0.8)

            ax.axhline(0.10, color=C_TAU, linestyle=":", linewidth=1,
                       label="S(t) = 0.10  (RUL_90)")
            ax.set_xlim(0, t_max / 1e3)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("Deployment step  (× 10³)")
            ax.set_ylabel("Survival probability  S(t)")
            cond_title = "Abrupt transition" if cond == "abrupt" else "Gradual transition"
            ax.set_title(cond_title)
            ax.legend(loc="upper right", fontsize=8)

        fig.suptitle("Fig 5  –  Weibull Survival Curves  |  PyFlyt QuadX-Hover\n"
                     "Rug marks = empirical T_hard;  S(t) = exp(−(t/λ)^k)",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, "fig5_weibull_survival.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


def fig6_alarm_timelines():
    """
    Horizontal timeline per seed for the abrupt condition.
    Shows T_ocsvm, T_cusum, T_warn (D_k), T_hard, and T_drift reference line.
    D_k alarms in the pre-onset zone (T_warn ≤ T_drift) are shown in dark red.
    """
    seed_labels = [f"Q1 s{s}" for s in SEEDS] + [f"Q2 s{s}" for s in SEEDS]

    ev_ocsvm = []
    ev_cusum = []
    ev_warn  = []
    ev_hard  = []

    for q in QUALITIES:
        for s in SEEDS:
            d = ALL[(q, s, "abrupt")]
            ev_ocsvm.append(d.get("T_ocsvm"))
            ev_cusum.append(d.get("T_cusum"))
            ev_warn.append(d.get("T_warn"))
            ev_hard.append(d.get("T_hard"))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))

        # Shade pre-onset zone
        ax.axvspan(0, T_DRIFT_ABRUPT / 1e3, alpha=0.08, color=C_TDRIFT, zorder=0,
                   label=f"Pre-onset zone  (T < T_drift = {T_DRIFT_ABRUPT//1e3:.0f} k)")

        for idx, (lbl, t_oc, t_cu, t_wn, t_hd) in enumerate(
                zip(seed_labels, ev_ocsvm, ev_cusum, ev_warn, ev_hard)):

            y = 9 - idx   # top-down order

            bar_end = t_hd if t_hd else 360_000
            ax.barh(y, bar_end / 1e3, height=0.45, color="#EEEEEE",
                    edgecolor="#CCCCCC", linewidth=0.5)

            if t_oc:
                ax.plot(t_oc / 1e3, y, "^", color="#E08214",
                        markersize=8, zorder=5)
            if t_cu:
                ax.plot(t_cu / 1e3, y, "s", color="#542788",
                        markersize=7, zorder=5)
            if t_wn:
                # Pre-onset false alarm vs genuine post-onset
                is_pre_onset = (t_wn <= T_DRIFT_ABRUPT)
                warn_color   = C_FA if is_pre_onset else C_DETECT
                ax.plot(t_wn / 1e3, y, "o", color=warn_color,
                        markersize=8, zorder=6)
            if t_hd:
                ax.plot(t_hd / 1e3, y, "X", color="#1A1A1A",
                        markersize=9, zorder=5)

        # Transition onset line
        ax.axvline(T_DRIFT_ABRUPT / 1e3, color=C_TDRIFT, linestyle="--",
                   linewidth=1.5, zorder=7)

        ax.set_yticks(list(range(10)))
        ax.set_yticklabels(seed_labels[::-1])
        ax.set_xlabel("Deployment step  (× 10³)")
        ax.set_title(
            "Fig 6  –  Alarm Timeline — Abrupt Transition  |  PyFlyt QuadX-Hover\n"
            "D_k alarms in shaded zone are pre-onset (same regime as baseline condition)"
        )

        legend_el = [
            mpatches.Patch(color=C_TDRIFT, alpha=0.15,
                           label=f"Pre-onset zone  (T < T_drift = {T_DRIFT_ABRUPT//1e3:.0f} k)"),
            Line2D([0], [0], color=C_TDRIFT, linestyle="--", lw=1.5,
                   label="Transition onset  (T_drift)"),
            Line2D([0], [0], marker="^", color="#E08214", lw=0,
                   markersize=8, label="T_ocsvm  (OCSVM alarm)"),
            Line2D([0], [0], marker="s", color="#542788", lw=0,
                   markersize=7, label="T_cusum  (CUSUM alarm)"),
            Line2D([0], [0], marker="o", color=C_FA, lw=0,
                   markersize=8, label="T_warn  (pre-onset false alarm, ≤ T_drift)"),
            Line2D([0], [0], marker="o", color=C_DETECT, lw=0,
                   markersize=8, label="T_warn  (genuine detection, > T_drift)"),
            Line2D([0], [0], marker="X", color="#1A1A1A", lw=0,
                   markersize=9, label="T_hard  (hard failure)"),
        ]
        ax.legend(handles=legend_el, loc="lower right", fontsize=8.5,
                  frameon=True, edgecolor="#AAAAAA")

        # Q1/Q2 divider
        ax.axhline(4.5, color="#CCCCCC", linewidth=1.5, linestyle="-")
        ax.text(2, 5.1, "Q1", fontsize=9, color="#555555")
        ax.text(2, 3.9, "Q2", fontsize=9, color="#555555")

        ax.set_xlim(0, 370)
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, "fig6_alarm_timelines.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    print(f"Writing figures to: {FIGURES_DIR}\n")
    fig1_far_comparison()
    fig2_dk_trajectories()
    fig3_detection_matrix()
    fig4_lead_times()
    fig5_weibull_survival()
    fig6_alarm_timelines()
    print("\nAll hover figures generated.")
