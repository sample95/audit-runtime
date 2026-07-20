"""
plot_framework_monitor.py — Framework monitoring visualisation
Methodology: v12.6

Produces three publication figures from T-05 metrics files:

  framework_dashboard.png
      3 rows (testing regime) × 3 columns (training condition)
      Each cell has TWO stacked panels:
        TOP  — State Distribution Shift Score [D_k]
               Jensen-Shannon divergence across observation dimensions
               Source: observation vector fed into the policy (black-box input)
        BOTTOM — Mean Reward per Window [y_k]
               Environment reward signal received per deployment window
               Source: scalar reward signal from environment
      The framework observes state + reward. It does NOT observe actions or
      policy weights.

  framework_summary.png
      Left:  False Alarm Rate [FAR]       — Baseline test
      Right: Transition Detection Rate    — Reaction & Sensitivity tests

  framework_detail_table.png
      Per-condition table: τ_warn, mean D_k calibration vs post-transition,
      D_k ratio, detection rate, and mean reaction time.

Usage:
    python plot_framework_monitor.py --data ./results/mujoco/hopper_t05 --out ./figures

Arguments:
    --data   Path to folder containing *_metrics_summary.json files
    --out    Output folder (created if absent). Defaults to --data folder.
    --prefix Optional filename prefix (e.g. "hopper_t05")
    --dpi    Output DPI (default 150)
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--data",   required=True, help="Folder with *_metrics_summary.json")
parser.add_argument("--out",    default=None,  help="Output folder (default: same as --data)")
parser.add_argument("--prefix", default="",    help="Filename prefix")
parser.add_argument("--dpi",    default=150,   type=int)
args = parser.parse_args()

DATA_DIR = args.data
OUT_DIR  = args.out or DATA_DIR
PREFIX   = (args.prefix + "_") if args.prefix else ""
DPI      = args.dpi
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
all_files = glob.glob(os.path.join(DATA_DIR, "*_metrics_summary.json"))
if not all_files:
    raise FileNotFoundError(f"No *_metrics_summary.json files found in {DATA_DIR}")

# "baseline" is canonical going forward; "none"/"no_drift" are pre-rename
# aliases still present in older result filenames — match any of them when
# the caller asks for "baseline".
_BASELINE_ALIASES = ("baseline", "none", "no_drift")

def load(cond, transition):
    keys = _BASELINE_ALIASES if transition == "baseline" else (transition,)
    return [
        json.load(open(f)) for f in all_files
        if os.path.basename(f).startswith(f"{cond}_")
        and any(f"_{k}_" in os.path.basename(f) for k in keys)
    ]

# Detect conditions and transition types from filenames
CONDITIONS  = sorted(set(os.path.basename(f).split("_")[0] for f in all_files))
TRANSITION_TYPES = ["baseline", "abrupt", "gradual"]

# Infer window parameters from first file
_sample      = json.load(open(all_files[0]))
T_EVAL       = int(_sample.get("T_eval", 1_000_000))
K0           = int(_sample.get("K0", 4))
_ws          = _sample["window_starts"]
_WIN_SIZE    = (_ws[1] - _ws[0]) if len(_ws) > 1 else 25_000
PILOT_END    = K0 * _WIN_SIZE       # end of calibration phase in steps

# ── Descriptive label maps ────────────────────────────────────────────────────
COND_LABELS = {
    "M1": "M1 — Standard\n(nominal training)",
    "M2": "M2 — Short Episodes\n(time-limited training)",
    "M3": "M3 — Noisy Reset\n(high-variance training)",
}

TRANSITION_ROW_LABELS = {
    "baseline": "Framework Baseline\n[Normal Operation]",
    "abrupt":   "Reaction Test\n[Abrupt Transition]",
    "gradual":  "Sensitivity Test\n[Gradual Transition]",
}

TRANSITION_ROW_SUBTITLE = {
    "baseline": "Policy deployed in its training environment — framework should stay silent",
    "abrupt":   "Environment transitions abruptly at T₀ — tests framework reaction speed",
    "gradual":  "Environment transitions gradually from T₀ — tests framework sensitivity to smooth increments",
}

# ── Colours ───────────────────────────────────────────────────────────────────
C_BASELINE  = "#4878CF"   # blue
C_ABRUPT   = "#D65F5F"   # red
C_GRAD     = "#B47CC7"   # purple
C_WARN     = "#E8943A"   # orange  — framework alert line [T_warn]
C_CUSUM    = "#7B3FA0"   # violet  — CUSUM change-point line [T_cusum]
C_T0       = "#555555"   # dark grey — transition onset
C_CALIB    = "#EFEFEF"   # light grey — calibration zone
C_THRESH   = "#999999"   # dotted threshold line
C_YK       = "#5aa05a"   # green — reward panel
TRANSITION_COLORS = {"baseline": C_BASELINE, "abrupt": C_ABRUPT, "gradual": C_GRAD}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Framework Monitoring Dashboard
# Layout: (2 subplots × 3 transition types) rows  ×  3 condition columns
# Top panel (D_k): State Distribution Shift Score  — source: observation vector
# Bottom panel (y_k): Mean Reward per Window       — source: environment reward
# ═══════════════════════════════════════════════════════════════════════════════

nrows_drift = len(TRANSITION_TYPES)
ncols       = len(CONDITIONS)
# Two sub-rows per transition type: [D_k, y_k] with height ratio 3:1
height_ratios = [3, 1] * nrows_drift

fig = plt.figure(figsize=(5.4 * ncols, 4.5 * nrows_drift), dpi=DPI)
fig.patch.set_facecolor("white")

# Figure-level annotation box — what the framework observes
fig.text(
    0.5, 0.995,
    "Framework input channels:  "
    "① Observation vector (state space) → State Distribution Shift Score [D_k]    "
    "② Scalar reward signal → Mean Reward per Window [y_k]\n"
    "The framework does NOT observe policy weights, gradients, or action outputs. "
    "The policy is treated as a black box.",
    ha="center", va="top", fontsize=9, color="#333333",
    fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#ccaa00", alpha=0.92, lw=1.2),
    transform=fig.transFigure,
)

gs = gridspec.GridSpec(
    nrows_drift * 2, ncols,
    height_ratios=height_ratios,
    hspace=0.08, wspace=0.30,
    left=0.10, right=0.98,
    top=0.93, bottom=0.07,
)

# Build axis array: axes_dk[transition_row, col] and axes_yk[transition_row, col]
axes_dk = np.empty((nrows_drift, ncols), dtype=object)
axes_yk = np.empty((nrows_drift, ncols), dtype=object)

for row in range(nrows_drift):
    for col in range(ncols):
        axes_dk[row, col] = fig.add_subplot(gs[row * 2,     col])
        axes_yk[row, col] = fig.add_subplot(gs[row * 2 + 1, col],
                                             sharex=axes_dk[row, col])

for row, transition in enumerate(TRANSITION_TYPES):
    color = TRANSITION_COLORS[transition]

    for col, cond in enumerate(CONDITIONS):
        ax_dk = axes_dk[row, col]
        ax_yk = axes_yk[row, col]
        data  = load(cond, transition)

        for ax in (ax_dk, ax_yk):
            ax.set_facecolor("white")

        # ── Calibration shading & label (both panels) ─────────────────────────
        for ax in (ax_dk, ax_yk):
            ax.axvspan(0, PILOT_END / 1e3, color=C_CALIB, zorder=0)

        ax_dk.text(
            (PILOT_END / 1e3) / 2, 0.96,
            f"Framework calibrates baseline\nstate distributions here [K₀={K0} windows]",
            transform=ax_dk.get_xaxis_transform(),
            ha="center", va="top", fontsize=6, color="#888888",
            fontstyle="italic", linespacing=1.3,
        )

        # ── Transition onset marker ───────────────────────────────────────────
        if transition != "baseline":
            t_drifts = [d["T_drift"] for d in data if d.get("T_drift") is not None]
            T0_step  = int(np.median(t_drifts)) if t_drifts else T_EVAL // 2
            for ax in (ax_dk, ax_yk):
                ax.axvline(T0_step / 1e3, color=C_T0, lw=1.3, ls="--", zorder=3)

        # ── D_k traces (state distribution shift — top panel) ─────────────────
        all_dk = []
        for d in data:
            ws = np.array(d["window_starts"]) / 1e3
            dk = np.array(d["D_k"])
            all_dk.append(dk)
            ax_dk.plot(ws, dk, color=color, alpha=0.18, lw=0.9, zorder=2)

        if all_dk:
            min_len  = min(len(x) for x in all_dk)
            mean_dk  = np.mean([x[:min_len] for x in all_dk], axis=0)
            ws_mean  = np.array(data[0]["window_starts"][:min_len]) / 1e3
            ax_dk.plot(ws_mean, mean_dk, color=color, lw=2.2, zorder=4)

        # ── Shift alert threshold τ_warn ──────────────────────────────────────
        if data:
            tau = data[0]["tau_warn"]
            ax_dk.axhline(tau, color=C_THRESH, lw=1.1, ls=":", zorder=3)
            ax_dk.text(T_EVAL / 1e3 * 0.99, tau, f"τ_warn={tau:.4f}",
                       ha="right", va="bottom", fontsize=6.5, color=C_THRESH)

        # ── Framework Alert lines [T_warn] ────────────────────────────────────
        t_warns = [d["T_warn"] for d in data if d["T_warn"] is not None]
        unique_tw = sorted(set(t_warns))
        for tw in unique_tw:
            n_tw = t_warns.count(tw)
            for ax in (ax_dk, ax_yk):
                ax.axvline(tw / 1e3, color=C_WARN, lw=1.6, ls="-",
                           alpha=min(0.3 + 0.14 * n_tw, 0.95), zorder=5)

        # ── CUSUM change-point lines [T_cusum] ───────────────────────────────
        t_cusums = [d["T_cusum"] for d in data if d.get("T_cusum") is not None]
        unique_tc = sorted(set(t_cusums))
        for tc in unique_tc:
            n_tc = t_cusums.count(tc)
            ax_dk.axvline(tc / 1e3, color=C_CUSUM, lw=1.4, ls=(0, (4, 2)),
                          alpha=min(0.3 + 0.14 * n_tc, 0.90), zorder=5,
                          label="CUSUM Alert [T_cusum]" if tc == unique_tc[0] else "")

        # ── y_k traces (mean reward — bottom panel) ───────────────────────────
        all_yk = []
        for d in data:
            ws = np.array(d["window_starts"]) / 1e3
            yk = np.array(d["y_k"])
            all_yk.append(yk)
            ax_yk.plot(ws, yk, color=C_YK, alpha=0.18, lw=0.9, zorder=2)

        if all_yk:
            min_len  = min(len(x) for x in all_yk)
            mean_yk  = np.mean([x[:min_len] for x in all_yk], axis=0)
            ws_mean  = np.array(data[0]["window_starts"][:min_len]) / 1e3
            ax_yk.plot(ws_mean, mean_yk, color=C_YK, lw=1.8, zorder=4)

        # ── Axis limits and formatting ────────────────────────────────────────
        ax_dk.set_xlim(0, T_EVAL / 1e3)
        ax_yk.set_xlim(0, T_EVAL / 1e3)
        for ax in (ax_dk, ax_yk):
            ax.tick_params(labelsize=7.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.setp(ax_dk.get_xticklabels(), visible=False)

        # ── Axis labels — explicitly naming the source ────────────────────────
        if col == 0:
            ax_dk.set_ylabel(
                "D_k\n[State JSD]",
                fontsize=8, color=color, labelpad=4,
            )
            ax_yk.set_ylabel(
                "y_k\n[Reward]",
                fontsize=8, color=C_YK, labelpad=4,
            )

        if row == nrows_drift - 1:
            ax_yk.set_xlabel("Deployment Step [thousands]", fontsize=8.5)

        # ── Column header (condition) ─────────────────────────────────────────
        if row == 0:
            ax_dk.set_title(COND_LABELS.get(cond, cond), fontsize=10,
                            fontweight="bold", pad=6)

        # ── Result annotation box ─────────────────────────────────────────────
        n_alert = len(t_warns)
        n_seeds = len(data)
        far_n   = sum(1 for d in data if d["T_warn"] is not None and d["delta"] == 0)

        if transition == "baseline":
            if far_n == 0:
                ann_txt = f"✓  No False Alarms\n[FAR = 0 / {n_seeds}]"
                ann_col = "#2a8a2a"
            else:
                ann_txt = f"⚠  False Alarms: {far_n}/{n_seeds}\n[FAR = {far_n/n_seeds:.0%}]"
                ann_col = "#cc4444"
        else:
            n_detected = n_alert
            if n_detected == n_seeds:
                ann_txt = f"✓  Transition Detected\n[{n_detected}/{n_seeds} seeds alerted]"
                ann_col = "#2a8a2a"
            elif n_detected > 0:
                ann_txt = f"◑  Partial Detection\n[{n_detected}/{n_seeds} seeds alerted]"
                ann_col = "#d08020"
            else:
                ann_txt = f"✗  Not Detected\n[0/{n_seeds} seeds alerted]"
                ann_col = "#cc4444"

        ax_dk.text(0.97, 0.97, ann_txt, transform=ax_dk.transAxes,
                   ha="right", va="top", fontsize=8, color=ann_col,
                   fontweight="bold", linespacing=1.4,
                   bbox=dict(boxstyle="round,pad=0.30", fc="white",
                             ec=ann_col, alpha=0.9, lw=1.2))

    # ── Row label on far left (spans both D_k and y_k sub-rows) ──────────────
    row_label = TRANSITION_ROW_LABELS[transition]
    # Place between the two sub-rows
    mid_y_dk = axes_dk[row, 0].get_position().y0
    mid_y_yk = axes_yk[row, 0].get_position().y1
    mid_y    = (mid_y_dk + mid_y_yk) / 2
    x_left   = axes_dk[row, 0].get_position().x0 - 0.07
    fig.text(
        x_left, mid_y, row_label,
        ha="center", va="center", fontsize=9.5,
        fontweight="bold", color=TRANSITION_COLORS[transition], rotation=90,
    )

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_CALIB,    label="Calibration Phase [K₀ windows — baseline state distributions computed here]"),
    Line2D([0],[0], color=C_T0,      lw=1.5, ls="--", label="Environmental Transition Onset [T₀]"),
    Line2D([0],[0], color=C_THRESH,  lw=1.2, ls=":",  label="Shift Alert Threshold [τ_warn]"),
    Line2D([0],[0], color=C_WARN,    lw=2.0,           label="Framework Alert Raised [T_warn]"),
    Line2D([0],[0], color=C_CUSUM,   lw=2.0, ls=(0,(4,2)), label="CUSUM Change-Point [T_cusum]"),
    Line2D([0],[0], color=C_BASELINE, lw=2.0,           label="D_k — Framework Baseline [Normal Operation]"),
    Line2D([0],[0], color=C_ABRUPT,  lw=2.0,           label="D_k — Reaction Test [Abrupt Transition]"),
    Line2D([0],[0], color=C_GRAD,    lw=2.0,           label="D_k — Sensitivity Test [Gradual Transition]"),
    Line2D([0],[0], color=C_YK,      lw=2.0,           label="y_k — Mean Reward per Window [environment reward signal]"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8,
           frameon=True, bbox_to_anchor=(0.5, 0.0), edgecolor="#cccccc",
           fancybox=True)

fig.suptitle(
    "Framework Monitoring Dashboard — Frozen Policy under Environmental Transition\n"
    "D_k is computed from the observation vector (state space). "
    "y_k is the reward signal. Neither actions nor policy weights are observed.",
    fontsize=11, fontweight="bold", y=0.995,
)

out1 = os.path.join(OUT_DIR, f"{PREFIX}framework_dashboard.png")
plt.savefig(out1, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {out1}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Framework Performance Summary
# Left:  False Alarm Rate [FAR]  — does framework stay silent when it should?
# Right: Transition Detection Rate  — does framework alert when env transitions?
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5), dpi=DPI)
fig2.patch.set_facecolor("white")

x  = np.arange(len(CONDITIONS))
bw = 0.30

# ── Panel 1: False Alarm Rate ─────────────────────────────────────────────────
ax = axes2[0]
ax.set_facecolor("white")
for i, cond in enumerate(CONDITIONS):
    nd  = load(cond, "baseline")
    far = sum(1 for d in nd if d["T_warn"] is not None and d["delta"] == 0) / max(len(nd), 1)
    col = "#cc4444" if far > 0 else "#2a8a2a"
    ax.bar(i, far, color=C_BASELINE, alpha=0.80, width=0.5, zorder=3,
           edgecolor="#dddddd", linewidth=1.0)
    ax.text(i, far + 0.015, f"{far:.0%}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=col)

ax.axhline(0.10, color="#cc4444", lw=1.5, ls="--",
           label="Max acceptable FAR [10%]")
ax.set_xticks(x)
ax.set_xticklabels([COND_LABELS.get(c, c).split("\n")[0] for c in CONDITIONS], fontsize=11)
ax.set_ylabel("False Alarm Rate [FAR]\n[fraction of seeds that raised a spurious alert]", fontsize=10)
ax.set_ylim(0, 0.45)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%"])
ax.set_title(
    "Framework Baseline [Normal Operation]\n"
    "Frozen policy deployed in its training environment\n"
    "→ D_k computed from observation vector should stay below τ_warn",
    fontsize=10, fontweight="bold", color=C_BASELINE,
)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel 2: Transition Detection Rate ───────────────────────────────────────
ax = axes2[1]
ax.set_facecolor("white")
for i, cond in enumerate(CONDITIONS):
    ab_data   = load(cond, "abrupt")
    grad_data = load(cond, "gradual")

    # Detection = T_warn fired (D_k from observation vector crossed τ_warn)
    det_ab   = sum(1 for d in ab_data   if d["T_warn"] is not None) / max(len(ab_data),   1)
    det_grad = sum(1 for d in grad_data if d["T_warn"] is not None) / max(len(grad_data), 1)

    ax.bar(i - bw / 2, det_ab,   width=bw, color=C_ABRUPT, alpha=0.82,
           edgecolor="#dddddd", linewidth=1.0, zorder=3,
           label="Reaction Test [Abrupt]" if i == 0 else "_")
    ax.bar(i + bw / 2, det_grad, width=bw, color=C_GRAD,   alpha=0.82,
           edgecolor="#dddddd", linewidth=1.0, zorder=3,
           label="Sensitivity Test [Gradual]" if i == 0 else "_")
    ax.text(i - bw / 2, det_ab   + 0.015, f"{det_ab:.0%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_ABRUPT)
    ax.text(i + bw / 2, det_grad + 0.015, f"{det_grad:.0%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_GRAD)

ax.axhline(1.0, color="#2a8a2a", lw=1.0, ls=":", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([COND_LABELS.get(c, c).split("\n")[0] for c in CONDITIONS], fontsize=11)
ax.set_ylabel("Transition Detection Rate [T_warn raised]\n[fraction of seeds where D_k exceeded τ_warn]", fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_title(
    "Reaction Test & Sensitivity Test\n"
    "Did D_k (from observation vector) exceed τ_warn after transition?\n"
    "→ Policy outcome is separate — framework detection is measured here",
    fontsize=10, fontweight="bold", color=C_ABRUPT,
)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig2.suptitle(
    "Framework Performance Summary — Frozen Policy under Environmental Transition\n"
    "Alert signal D_k is derived from the observation vector. "
    "These metrics evaluate framework detection behaviour, not policy quality.",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, f"{PREFIX}framework_summary.png")
plt.savefig(out2, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {out2}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Enriched Detail Table
# Shows the raw values behind the detection results:
#   - τ_warn (alert threshold derived from calibration D_k)
#   - Mean D_k during calibration vs post-transition
#   - D_k ratio (post / calib) — how much state distribution shifted
#   - Detection rate (% seeds T_warn fired)
#   - Mean reaction time (windows from T₀ to T_warn, abrupt/gradual only)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_row(data_list, transition):
    """Returns dict of per-condition aggregate metrics."""
    if not data_list:
        return None

    tau_vals, dk_calib, dk_post, det_flags, react_steps = [], [], [], [], []

    for d in data_list:
        ws  = np.array(d["window_starts"])
        dk  = np.array(d["D_k"])
        tau = d["tau_warn"]
        tau_vals.append(tau)

        # Calibration windows (index < K0)
        k0 = d.get("K0", K0)
        dk_calib.append(np.mean(dk[:k0]))

        # Post-transition windows (after T_drift or T_eval//2 for baseline)
        t_drift = d.get("T_drift") or (T_EVAL // 2)
        post_mask = ws > t_drift
        dk_post.append(np.mean(dk[post_mask]) if post_mask.any() else np.nan)

        # Detection
        det_flags.append(1 if d["T_warn"] is not None else 0)

        # Reaction steps (T_warn - T_drift, in windows)
        if d["T_warn"] is not None and d.get("T_drift") is not None:
            react_steps.append((d["T_warn"] - d["T_drift"]) / _WIN_SIZE)

    mean_calib = np.nanmean(dk_calib)
    mean_post  = np.nanmean(dk_post)
    ratio      = mean_post / mean_calib if mean_calib > 0 else float("nan")

    return {
        "tau_warn":    np.mean(tau_vals),
        "dk_calib":    mean_calib,
        "dk_post":     mean_post,
        "dk_ratio":    ratio,
        "det_rate":    np.mean(det_flags),
        "react_mean":  np.mean(react_steps) if react_steps else None,
        "n_seeds":     len(data_list),
    }


# Build table data
table_rows = []
col_headers = ["Regime", "Cond", "τ_warn", "D_k Calib\n(mean)", "D_k Post-T₀\n(mean)",
               "D_k Ratio\n(Post/Calib)", "Detection\nRate", "Mean Reaction\n[windows]",
               "n seeds"]

REGIME_LABELS = {
    "baseline": "Framework Baseline",
    "abrupt":   "Reaction Test",
    "gradual":  "Sensitivity Test",
}

for transition in TRANSITION_TYPES:
    for cond in CONDITIONS:
        data = load(cond, transition)
        r = compute_row(data, transition)
        if r is None:
            continue
        react_str = f"{r['react_mean']:.1f}" if r["react_mean"] is not None else "—"
        table_rows.append([
            REGIME_LABELS[transition],
            COND_LABELS.get(cond, cond).replace("\n", " "),
            f"{r['tau_warn']:.4f}",
            f"{r['dk_calib']:.4f}",
            f"{r['dk_post']:.4f}" if not np.isnan(r["dk_post"]) else "—",
            f"×{r['dk_ratio']:.1f}" if not np.isnan(r["dk_ratio"]) else "—",
            f"{r['det_rate']:.0%}",
            react_str,
            str(r["n_seeds"]),
        ])

# Draw figure
n_table_rows = len(table_rows)
fig3_h = max(3.5, 0.45 * n_table_rows + 2.5)
fig3, ax3 = plt.subplots(figsize=(15, fig3_h), dpi=DPI)
fig3.patch.set_facecolor("white")
ax3.axis("off")

tbl = ax3.table(
    cellText=table_rows,
    colLabels=col_headers,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.65)

# Style header row
for j in range(len(col_headers)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Style data rows: alternate shading + colour-code by regime
row_regime_colors = {
    "Framework Baseline": "#dde8f8",
    "Reaction Test":      "#fce8e8",
    "Sensitivity Test":   "#f3e8fa",
}
alt_offset = {"Framework Baseline": "#cddff5", "Reaction Test": "#f5d8d8", "Sensitivity Test": "#ebe0f5"}

for i, row_data in enumerate(table_rows):
    regime = row_data[0]
    base   = row_regime_colors.get(regime, "#f5f5f5")
    alt    = alt_offset.get(regime, "#ebebeb")
    bg     = base if (i % 2 == 0) else alt
    for j in range(len(col_headers)):
        tbl[i + 1, j].set_facecolor(bg)

fig3.suptitle(
    "Framework Detail Table — Values Behind the Detection Results\n"
    "D_k Calib: mean JSD from observation vector during calibration phase (K₀ windows) | "
    "D_k Post-T₀: mean JSD from observation vector after environmental transition | "
    "Reaction [windows]: windows from T₀ to first T_warn alert",
    fontsize=9.5, fontweight="bold", y=0.98, va="top",
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out3 = os.path.join(OUT_DIR, f"{PREFIX}framework_detail_table.png")
plt.savefig(out3, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"[saved] {out3}")
