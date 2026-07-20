"""
render_deployment.py — 2×3 composite deployment video
Methodology: v12.6

Layout:
  R1C1  Agent (rendered env) + framework overlay
          — background reddens as D_k approaches / exceeds τ_warn
          — FALSE ALARM banner if T_warn fired in a baseline run
          — SILENT FAILURE banner if policy fails with no framework alert
  R1C2  State Distribution Shift Score [D_k]  — source: observation vector (black-box input)
  R1C3  Mean Reward per Window [y_k]          — source: environment reward signal
  R2C1  Framework Alert Status                — current channel readings and alert flags
  R2C2  Action Statistics [act_mean_norm]     — auxiliary; policy output magnitude
  R2C3  Episode Reward History                — cumulative per-episode returns

Metrics are read from a pre-computed *_metrics_summary.json (consistent with T-05).
The environment is re-run with the same seed to produce rendered frames.
GravityPerturbation wrapper is applied identically to the original audit run.

Requirements: imageio[ffmpeg], matplotlib, gymnasium[mujoco], stable-baselines3
    pip install imageio[ffmpeg] --break-system-packages

Usage:
    python render_deployment.py \\
        --metrics ./hopper_t05/M1_seed0_abrupt_metrics_summary.json \\
        --model   ./results/mujoco/hopper/M1_seed0.zip \\
        --out     ./videos/M1_seed0_abrupt.mp4 \\
        --config  ./configs/hopper_t05.json \\
        --render-every 500 \\
        --fps 30

    # Quick preview (1-in-2000 sampling → ~30s at 30fps)
    python render_deployment.py ... --render-every 2000 --fps 20
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Headless MuJoCo
os.environ.setdefault("MUJOCO_GL", "osmesa")

import gymnasium as gym
from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.dirname(__file__))
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "core"))  # [src_clean] import shim
from drift_wrappers import PERTURBATION_REGISTRY

parser = argparse.ArgumentParser()
parser.add_argument("--metrics",      required=True, help="*_metrics_summary.json from T-05")
parser.add_argument("--model",        required=True, help="Model checkpoint .zip")
parser.add_argument("--out",          required=True, help="Output .mp4 path")
parser.add_argument("--config",       required=True, help="Domain config JSON (e.g. hopper_t05.json)")
parser.add_argument("--instrument",   default=None,  help="instrument.json (default: auto-detect)")
parser.add_argument("--vecnorm",      default=None,  help="VecNormalize .pkl (auto-detected if omitted)")
parser.add_argument("--render-every", default=500,   type=int, help="Steps between video frames (default 500)")
parser.add_argument("--fps",          default=30,    type=int, help="Output frame rate (default 30)")
parser.add_argument("--panel-size",   default=400,   type=int, help="Panel height/width in pixels (default 400)")
args = parser.parse_args()

with open(args.config) as f:
    domain_cfg = json.load(f)

configs_dir  = os.path.dirname(args.config)
instr_path   = args.instrument or os.path.join(configs_dir, "instrument.json")
with open(instr_path) as f:
    instr_cfg = json.load(f)

# ── Load metrics (T-05 precomputed — ground truth for this video) ─────────────
with open(args.metrics) as f:
    metrics = json.load(f)

DRIFT_TYPE  = metrics["transition"]
# "baseline" is canonical going forward; "none"/"no_drift" are pre-rename
# aliases still present in older result JSONs.
DRIFT_TYPE  = {"none": "baseline", "no_drift": "baseline"}.get(DRIFT_TYPE, DRIFT_TYPE)
SEED        = int(metrics["seed"])
CONDITION   = metrics["condition"]
ENV_ID      = metrics["env_id"]

WINDOW_STARTS = np.array(metrics["window_starts"], dtype=np.int64)
D_K           = np.array(metrics["D_k"])
Y_K           = np.array(metrics["y_k"])
A_DRIFT       = np.array(metrics["A_drift"])
A_WATCH       = np.array(metrics["A_watch"])
A_DEG         = np.array(metrics["A_deg"])
TAU_WARN      = float(metrics["tau_warn"])
TAU_WATCH     = float(metrics.get("tau_watch", TAU_WARN * 0.95))
T_WARN        = metrics.get("T_warn")       # int or None
T_COMP        = metrics.get("T_comp")       # int or None
T_DRIFT       = metrics.get("T_drift")      # transition onset (None for baseline)
K0            = int(metrics.get("K0", int(instr_cfg["K0"])))
MU_REF        = float(metrics.get("mu_ref", 0.0))
SIGMA_REF     = float(metrics.get("sigma_ref", 1.0))
T_EVAL        = int(metrics["T_eval"])
CUSUM_S       = np.array(metrics.get("cusum_S", [0.0] * len(metrics["D_k"])))
T_CUSUM       = metrics.get("T_cusum")      # int or None
H_CUSUM       = float(metrics.get("h_cusum", 0.0))
K_CUSUM       = float(metrics.get("k_cusum", 0.0))

WIN_SIZE = int(WINDOW_STARTS[1] - WINDOW_STARTS[0]) if len(WINDOW_STARTS) > 1 else 25000
PILOT_END = K0 * WIN_SIZE

# Auxiliary (action stats, may not exist in old files)
_aux = metrics.get("aux_action_windows", {})
ACT_MEAN_NORM = np.array(_aux.get("act_mean_norm", []))

# ── Perturbation parameters ─────────────────────────────────────────────────────
PERTURBATION_WRAPPER_NAME = domain_cfg.get("perturbation_wrapper", domain_cfg.get("drift_wrapper", "GravityPerturbation"))
DRIFT_PARAMS       = domain_cfg.get("drift_params", {})
PerturbationWrapper = PERTURBATION_REGISTRY[PERTURBATION_WRAPPER_NAME]
T0                 = T_EVAL // 2    # abrupt switch point

drift_kwargs = {}
if PERTURBATION_WRAPPER_NAME in ("GravityPerturbation", "GravityDrift"):
    drift_kwargs = {
        "g_nominal": float(DRIFT_PARAMS.get("g_nominal", -9.81)),
        "g_drifted": float(DRIFT_PARAMS.get("g_drifted", -19.62)),
    }

ALGO_NAME = domain_cfg.get("algo", "PPO")
AlgoCls   = {"PPO": PPO, "SAC": SAC}.get(ALGO_NAME, PPO)
model     = AlgoCls.load(args.model, device="cpu")
print(f"[model] loaded {args.model} ({ALGO_NAME})")

# Auto-detect vecnorm if not provided
vecnorm = None
if args.vecnorm:
    vn_path = args.vecnorm
elif args.model.endswith(".zip"):
    for suffix in ("_vecnormalize.pkl", "_vecnorm.pkl"):
        candidate = args.model.replace(".zip", suffix)
        if os.path.exists(candidate):
            vn_path = candidate
            break
    else:
        vn_path = None
else:
    vn_path = None

if vn_path and os.path.exists(vn_path):
    with open(vn_path, "rb") as f:
        vecnorm = pickle.load(f)
    vecnorm.training    = False
    vecnorm.norm_reward = False
    print(f"[vecnorm] loaded {vn_path}")
else:
    print("[vecnorm] none — raw obs")


def _normalize_obs(obs, vn):
    if vn is None:
        return obs
    rms = vn.obs_rms
    return np.clip(
        (obs - rms.mean) / np.sqrt(rms.var + vn.epsilon),
        -vn.clip_obs, vn.clip_obs,
    ).astype(np.float32)


os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
try:
    import imageio
except ImportError:
    sys.exit("imageio not found. Run: pip install imageio[ffmpeg] --break-system-packages")

PANEL  = args.panel_size      # px — each panel is PANEL × PANEL
RENDER = args.render_every
FPS    = args.fps
DPI    = 96
PANEL_IN = PANEL / DPI        # matplotlib figure size in inches

# ── Colour palette ────────────────────────────────────────────────────────────
C_DK    = "#4878CF"
C_YK    = "#5aa05a"
C_WARN  = "#E8943A"
C_ALERT = "#cc3333"
C_WATCH = "#e8a020"
C_CALIB = "#efefef"
C_THRESH= "#999999"
C_ACT   = "#9467bd"

REGIME_LABELS = {
    "baseline": "Framework Baseline [Normal Operation]",
    "abrupt":   "Reaction Test [Abrupt Environmental Transition]",
    "gradual":  "Sensitivity Test [Gradual Environmental Transition]",
}
regime_label = REGIME_LABELS.get(DRIFT_TYPE, DRIFT_TYPE)


# ── Helper: render a matplotlib figure to a PANEL×PANEL RGB numpy array ───────
def fig_to_rgb(fig, panel_px):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgb = buf[:, :, :3]
    # Resize to panel_px × panel_px using simple nearest-neighbour crop/pad
    h, w = rgb.shape[:2]
    side = min(h, w)
    rgb  = rgb[(h - side) // 2:(h - side) // 2 + side,
               (w - side) // 2:(w - side) // 2 + side]
    from PIL import Image
    img = Image.fromarray(rgb).resize((panel_px, panel_px), Image.BILINEAR)
    return np.array(img)


# ── Build per-episode reward list from y_k (approximation from window rewards)
# We'll collect actual episode rewards during the rollout
episode_rewards = []  # filled during rollout

# ── Matplotlib panel figures — created once, artists updated in-place ─────────
# Panel figures: one fig per metric panel (easier artist management)
figs   = {}
axes   = {}
artists= {}  # holds Line2D / Text objects we update

def _make_panel(key, title, yl, color):
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=4)
    ax.set_ylabel(yl, fontsize=7, color=color)
    ax.set_xlabel("Step [×10³]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(0, T_EVAL / 1e3)
    figs[key] = fig
    axes[key] = ax
    plt.tight_layout(pad=0.6)
    return fig, ax

# D_k panel
_f, _a = _make_panel("dk",
    "State Distribution Shift Score [D_k]\nSource: observation vector (black-box input to policy)",
    "D_k [JSD]", C_DK)
# Calibration zone
_a.axvspan(0, PILOT_END / 1e3, color=C_CALIB, zorder=0, label="Calibration")
_a.text((PILOT_END / 1e3) / 2, 0.97, f"K₀={K0}\ncalibration",
        transform=_a.get_xaxis_transform(), ha="center", va="top",
        fontsize=5.5, color="#aaaaaa", fontstyle="italic")
# τ_warn line
_a.axhline(TAU_WARN, color=C_THRESH, lw=1.0, ls=":", zorder=3)
_a.text(T_EVAL / 1e3 * 0.99, TAU_WARN, f"τ_warn={TAU_WARN:.4f}",
        ha="right", va="bottom", fontsize=6, color=C_THRESH)
# Transition onset
if T_DRIFT:
    _a.axvline(T_DRIFT / 1e3, color="#555555", lw=1.2, ls="--", zorder=3,
               label=f"Transition onset T₀={T_DRIFT//1000}k")
# D_k line (will be updated)
_dk_line, = _a.plot([], [], color=C_DK, lw=1.8, zorder=4)
_dk_dot,  = _a.plot([], [], "o", color=C_DK, ms=5, zorder=5)
_twarn_line = _a.axvline(np.nan, color=C_WARN, lw=2.0, ls="-", zorder=6, alpha=0.85)
artists["dk_line"]  = _dk_line
artists["dk_dot"]   = _dk_dot
artists["dk_twarn"] = _twarn_line

# y_k panel
_f, _a = _make_panel("yk",
    "Mean Reward per Window [y_k]\nSource: environment reward signal",
    "y_k [mean reward]", C_YK)
# Reference band ±2σ
_a.axhspan(MU_REF - 2 * SIGMA_REF, MU_REF + 2 * SIGMA_REF,
           color="#d8f0d8", alpha=0.5, zorder=0)
_a.axhline(MU_REF, color=C_YK, lw=1.0, ls="--", alpha=0.6, zorder=2)
_a.text(T_EVAL / 1e3 * 0.01, MU_REF, f"μ_ref={MU_REF:.1f}", ha="left",
        va="bottom", fontsize=6, color=C_YK, alpha=0.7)
if T_DRIFT:
    _a.axvline(T_DRIFT / 1e3, color="#555555", lw=1.2, ls="--", zorder=3)
_yk_line, = _a.plot([], [], color=C_YK, lw=1.8, zorder=4)
_yk_dot,  = _a.plot([], [], "o", color=C_YK, ms=5, zorder=5)
_yk_twarn = _a.axvline(np.nan, color=C_WARN, lw=2.0, ls="-", zorder=6, alpha=0.85)
artists["yk_line"]  = _yk_line
artists["yk_dot"]   = _yk_dot
artists["yk_twarn"] = _yk_twarn

# Action norm panel
_f, _a = _make_panel("act",
    "Action Statistics [act_mean_norm]\n(auxiliary — policy output magnitude)",
    "‖a‖ mean per window", C_ACT)
if T_DRIFT:
    _a.axvline(T_DRIFT / 1e3, color="#555555", lw=1.2, ls="--", zorder=3)
_act_line, = _a.plot([], [], color=C_ACT, lw=1.8, zorder=4)
_act_dot,  = _a.plot([], [], "o", color=C_ACT, ms=5, zorder=5)
artists["act_line"] = _act_line
artists["act_dot"]  = _act_dot

# Episode reward panel
_f, _a = _make_panel("ep",
    "Episode Reward History\n(cumulative returns — policy performance signal)",
    "Episode return", "#2a8a2a")
if T_DRIFT:
    _a.axvline(T_DRIFT / 1e3, color="#555555", lw=1.2, ls="--", zorder=3)
_ep_bar_container = None   # will be rebuilt on first episode
artists["ep_ax"] = _a

# CUSUM panel — S_k cumulative sum over time (replaces text status box)
_f_cusum, _a_cusum = plt.subplots(figsize=(PANEL_IN, PANEL_IN), dpi=DPI)
_f_cusum.patch.set_facecolor("white")
_a_cusum.set_facecolor("white")
_a_cusum.spines["top"].set_visible(False)
_a_cusum.spines["right"].set_visible(False)
_a_cusum.tick_params(labelsize=6.5)
_a_cusum.set_xlim(0, T_EVAL / 1e3)
_a_cusum.set_xlabel("Deployment Step [×10³]", fontsize=7)
_a_cusum.set_ylabel("S_k  [CUSUM statistic]", fontsize=7, color="#7B3FA0")
_a_cusum.set_title(
    "CUSUM Change-Point Detector\n"
    "(cumulative evidence of sustained D_k shift — source: state space)",
    fontsize=7.5, fontweight="bold", pad=4)
if H_CUSUM > 0:
    _a_cusum.axhline(H_CUSUM, color="#7B3FA0", lw=1.1, ls=":", zorder=3)
    _a_cusum.text(T_EVAL / 1e3 * 0.99, H_CUSUM, f"h={H_CUSUM:.4f}",
                  ha="right", va="bottom", fontsize=6.5, color="#7B3FA0")
if T_DRIFT:
    _a_cusum.axvline(T_DRIFT / 1e3, color="#555555", lw=1.2, ls="--", zorder=3)
if T_CUSUM is not None:
    _a_cusum.axvline(T_CUSUM / 1e3, color="#7B3FA0", lw=1.6, ls=(0, (4, 2)), zorder=5,
                     label=f"T_cusum={T_CUSUM//1000}k")
_cusum_line, = _a_cusum.plot([], [], color="#7B3FA0", lw=1.6, zorder=4)
_cusum_dot,  = _a_cusum.plot([], [], "o", color="#7B3FA0", ms=4, zorder=5)
plt.tight_layout(pad=0.4)
figs["cusum"] = _f_cusum
axes["cusum"] = _a_cusum
artists["cusum_line"] = _cusum_line
artists["cusum_dot"]  = _cusum_dot


def _render_status_panel(step, win_idx, d_k_cur, y_k_cur, act_cur,
                         a_watch, a_drift, a_deg, t_warn_fired):
    ax = axes["status"]
    ax.clear()
    ax.axis("off")
    ax.set_facecolor("white")
    figs["status"].set_facecolor("white")

    lines = [
        ("Framework Alert Status", None, 13, "bold", "#222222"),
        (regime_label, None, 8, "normal", "#555555"),
        ("", None, 8, "normal", "#ffffff"),
        (f"Step:  {step:,} / {T_EVAL:,}", None, 9, "normal", "#333333"),
        (f"Window: {win_idx+1} / {len(WINDOW_STARTS)}", None, 9, "normal", "#333333"),
        ("", None, 6, "normal", "#ffffff"),
        ("─── Detection Signal (state) ───", None, 8, "normal", "#666666"),
        (f"D_k  =  {d_k_cur:.5f}", None, 10, "bold",
         C_ALERT if a_drift else (C_WATCH if a_watch else C_DK)),
        (f"τ_warn =  {TAU_WARN:.5f}", None, 9, "normal", C_THRESH),
        (f"D_k / τ =  {d_k_cur/TAU_WARN:.2f}×", None, 9, "normal",
         C_ALERT if d_k_cur >= TAU_WARN else C_WATCH if d_k_cur >= TAU_WATCH else "#555555"),
        ("", None, 6, "normal", "#ffffff"),
        ("─── Reward Signal ───", None, 8, "normal", "#666666"),
        (f"y_k  =  {y_k_cur:.2f}", None, 10, "bold",
         C_ALERT if a_deg else C_YK),
        (f"μ_ref =  {MU_REF:.2f}", None, 9, "normal", "#555555"),
        ("", None, 6, "normal", "#ffffff"),
        ("─── Alert Flags ───", None, 8, "normal", "#666666"),
    ]

    alert_rows = [
        ("A_watch", a_watch, "D_k > τ_watch — watching"),
        ("A_drift",  a_drift,  "D_k > τ_warn  — state shift"),
        ("A_deg",    a_deg,    "y_k below ref — reward drop"),
        ("T_warn",   t_warn_fired, f"alert raised at step {T_WARN}" if T_WARN else "not fired"),
    ]

    y0 = 0.97
    dy = 0.061
    for i, (txt, _, fs, fw, fc) in enumerate(lines):
        _a_status_fig = axes["status"]
        ax.text(0.08, y0 - i * dy, txt,
                transform=ax.transAxes, fontsize=fs,
                fontweight=fw, color=fc, va="top")

    y_base = y0 - len(lines) * dy
    for j, (lbl, fired, desc) in enumerate(alert_rows):
        col  = C_ALERT if fired else "#aaaaaa"
        icon = "●" if fired else "○"
        ax.text(0.08, y_base - j * 0.07,
                f"{icon}  {lbl}:  {desc}",
                transform=ax.transAxes, fontsize=9,
                fontweight="bold" if fired else "normal",
                color=col, va="top")


def _update_panels(w_idx):
    """Redraw D_k, y_k, action panels up to window w_idx."""
    ws_k  = WINDOW_STARTS[:w_idx + 1] / 1e3
    dk_k  = D_K[:w_idx + 1]
    yk_k  = Y_K[:w_idx + 1]

    # D_k
    artists["dk_line"].set_data(ws_k, dk_k)
    if len(ws_k):
        artists["dk_dot"].set_data([ws_k[-1]], [dk_k[-1]])
    if T_WARN is not None:
        artists["dk_twarn"].set_xdata([T_WARN / 1e3, T_WARN / 1e3])
    _ax = axes["dk"]
    _ax.relim()
    _ax.autoscale_view(scalex=False, scaley=True)
    figs["dk"].canvas.draw()

    # y_k
    artists["yk_line"].set_data(ws_k, yk_k)
    if len(ws_k):
        artists["yk_dot"].set_data([ws_k[-1]], [yk_k[-1]])
    if T_WARN is not None:
        artists["yk_twarn"].set_xdata([T_WARN / 1e3, T_WARN / 1e3])
    axes["yk"].relim()
    axes["yk"].autoscale_view(scalex=False, scaley=True)
    figs["yk"].canvas.draw()

    # Action norm
    if len(ACT_MEAN_NORM) > w_idx:
        act_k = ACT_MEAN_NORM[:w_idx + 1]
        artists["act_line"].set_data(ws_k[:len(act_k)], act_k)
        if len(act_k):
            artists["act_dot"].set_data([ws_k[len(act_k) - 1]], [act_k[-1]])
        axes["act"].relim()
        axes["act"].autoscale_view(scalex=False, scaley=True)
    figs["act"].canvas.draw()

    # CUSUM statistic
    if len(CUSUM_S) > w_idx:
        cs_k = CUSUM_S[:w_idx + 1]
        ws_cs = WINDOW_STARTS[:len(cs_k)] / 1e3
        artists["cusum_line"].set_data(ws_cs, cs_k)
        if len(cs_k):
            artists["cusum_dot"].set_data([ws_cs[-1]], [cs_k[-1]])
        axes["cusum"].relim()
        axes["cusum"].autoscale_view(scalex=False, scaley=True)
    figs["cusum"].canvas.draw()


def _update_ep_panel(ep_steps, ep_rets):
    """Redraw episode reward bar chart."""
    ax = artists["ep_ax"]
    ax.clear()
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.5)
    ax.set_ylabel("Episode return", fontsize=7, color="#2a8a2a")
    ax.set_xlabel("Step [×10³]", fontsize=7)
    ax.set_title("Episode Reward History\n(cumulative returns — policy performance signal)",
                 fontsize=7.5, fontweight="bold", pad=4)
    if T_DRIFT:
        ax.axvline(T_DRIFT / 1e3, color="#555555", lw=1.2, ls="--", zorder=3)
    ax.set_xlim(0, T_EVAL / 1e3)
    if ep_steps:
        xs  = np.array(ep_steps) / 1e3
        ys  = np.array(ep_rets)
        cols = [C_ALERT if T_WARN and ep_steps[i] >= T_WARN else "#2a8a2a"
                for i in range(len(ep_steps))]
        ax.bar(xs, ys, width=max(1.0, (T_EVAL / 1e3) / len(ep_steps) * 0.8),
               color=cols, alpha=0.65, zorder=2)
    figs["ep"].canvas.draw()


def _alarm_overlay(frame_rgb, alarm_level):
    """
    Blend a red overlay onto the agent frame proportional to alarm_level [0..1].
    alarm_level = 0: no overlay.  alarm_level = 1: strong red tint.
    """
    red = np.array([220, 30, 30], dtype=np.float32)
    alpha = float(np.clip(alarm_level * 0.55, 0.0, 0.55))
    out = frame_rgb.astype(np.float32) * (1 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_overlay_text(frame_rgb, step, win_idx, d_k_cur, tau, t_warn_fired,
                       t_comp_fired, transition):
    """Draw framework values and alert banners directly on the agent frame."""
    from PIL import Image, ImageDraw, ImageFont
    img  = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    def _shadow(pos, txt, fill, sz=12):
        x, y = pos
        draw.text((x + 1, y + 1), txt, fill=(0, 0, 0, 180), font=None)
        draw.text((x, y), txt, fill=fill)

    # Step counter
    _shadow((6, 4), f"Step {step:,} / {T_EVAL:,}", (255, 255, 255))
    # Window
    _shadow((6, 18), f"Window {win_idx+1}/{len(WINDOW_STARTS)}", (220, 220, 220))
    # D_k value
    ratio = d_k_cur / tau if tau > 0 else 0
    dk_col = (255, 80, 80) if ratio >= 1.0 else (255, 180, 80) if ratio >= 0.8 else (180, 230, 180)
    _shadow((6, 36), f"D_k = {d_k_cur:.4f}  (τ={tau:.4f}  {ratio:.2f}×)", dk_col)

    # Alert banners
    if transition == "baseline" and t_warn_fired:
        bx, by, bw, bh = W // 4, H // 2 - 30, W // 2, 55
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(200, 30, 30, 210))
        draw.text((bx + 6, by + 4),  "⚠ FALSE ALARM", fill=(255, 255, 100))
        draw.text((bx + 6, by + 22), "Framework fired but environment", fill=(255, 220, 220))
        draw.text((bx + 6, by + 36), "has not changed (no transition)", fill=(255, 220, 220))

    elif transition != "baseline" and t_warn_fired:
        bx, by, bw, bh = W // 4, H // 2 - 18, W // 2, 34
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(180, 80, 0, 200))
        draw.text((bx + 6, by + 6), f"✓ FRAMEWORK ALERT — T_warn = {T_WARN:,}", fill=(255, 240, 100))
        draw.text((bx + 6, by + 20), "Transition detected from state distribution", fill=(255, 220, 180))

    elif transition != "baseline" and t_comp_fired and not t_warn_fired:
        bx, by, bw, bh = W // 4, H // 2 - 18, W // 2, 34
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(80, 0, 120, 200))
        draw.text((bx + 6, by + 6), "✗ SILENT FAILURE", fill=(255, 160, 255))
        draw.text((bx + 6, by + 20), "Policy failed — framework gave no alert", fill=(220, 180, 255))

    # Regime label bottom strip
    strip_h = 20
    draw.rectangle([0, H - strip_h, W, H], fill=(30, 30, 30, 180))
    draw.text((4, H - strip_h + 3), regime_label, fill=(200, 200, 200))

    return np.array(img)


# ── Environment setup ─────────────────────────────────────────────────────────
print(f"[env] creating {ENV_ID} with render_mode=rgb_array")
base_env = gym.make(ENV_ID, render_mode="rgb_array")
env      = PerturbationWrapper(base_env, transition=DRIFT_TYPE,
                        t0=T0, pilot_end=PILOT_END, **drift_kwargs)

obs, _   = env.reset(seed=SEED)

print(f"[video] {args.out}  |  render_every={RENDER}  fps={FPS}")
print(f"[run]  {CONDITION} seed={SEED}  drift={DRIFT_TYPE}  T_eval={T_EVAL:,}")
writer   = imageio.get_writer(args.out, fps=FPS, macro_block_size=None)

# ── State ─────────────────────────────────────────────────────────────────────
cur_win_idx   = 0
prev_win_idx  = -1
ep_step_list  = []   # step index of each episode end
ep_ret_list   = []   # cumulative return for that episode
_ep_cur_ret   = 0.0
_ep_start     = 0

print("[running]", flush=True)

for t in range(T_EVAL):
    # ── Determine window index ────────────────────────────────────────────────
    cur_win_idx = int(np.searchsorted(WINDOW_STARTS, t, side="right")) - 1
    cur_win_idx = max(0, min(cur_win_idx, len(WINDOW_STARTS) - 1))

    # ── Policy step ───────────────────────────────────────────────────────────
    action, _ = model.predict(_normalize_obs(obs, vecnorm), deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    _ep_cur_ret += float(reward)

    if terminated or truncated:
        ep_step_list.append(t)
        ep_ret_list.append(_ep_cur_ret)
        _ep_cur_ret = 0.0
        _ep_start   = t + 1
        obs, _ = env.reset()

    # ── Render frame ─────────────────────────────────────────────────────────
    if t % RENDER == 0:
        # Current metric values (look up from precomputed JSON)
        w     = cur_win_idx
        d_k_c = float(D_K[w]) if w < len(D_K) else 0.0
        y_k_c = float(Y_K[w]) if w < len(Y_K) else 0.0
        a_wat = bool(A_WATCH[w]) if w < len(A_WATCH) else False
        a_dft = bool(A_DRIFT[w]) if w < len(A_DRIFT) else False
        a_deg = bool(A_DEG[w])   if w < len(A_DEG)   else False
        act_c = float(ACT_MEAN_NORM[w]) if w < len(ACT_MEAN_NORM) else 0.0
        t_warn_fired = (T_WARN is not None) and (t >= T_WARN)
        t_comp_fired = (T_COMP is not None) and (t >= T_COMP)

        # Alarm level drives red background tint
        alarm_level = min(1.0, d_k_c / TAU_WARN) if TAU_WARN > 0 else 0.0
        if t_warn_fired:
            alarm_level = min(1.0, alarm_level + 0.4)

        # Update matplotlib panels when window changes
        if cur_win_idx != prev_win_idx:
            _update_panels(w)
            _update_ep_panel(ep_step_list, ep_ret_list)
            _render_status_panel(t, w, d_k_c, y_k_c, act_c,
                                 a_wat, a_dft, a_deg, t_warn_fired)
            prev_win_idx = cur_win_idx

        # Agent frame
        env_frame = env.render()
        if env_frame is None:
            env_frame = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
        # Resize to PANEL×PANEL
        from PIL import Image
        env_frame = np.array(
            Image.fromarray(env_frame.astype(np.uint8)).resize((PANEL, PANEL), Image.BILINEAR)
        )
        env_frame = _alarm_overlay(env_frame, alarm_level)
        env_frame = _draw_overlay_text(env_frame, t, w, d_k_c, TAU_WARN,
                                       t_warn_fired, t_comp_fired, DRIFT_TYPE)

        # Render metric panels to numpy
        p_dk     = fig_to_rgb(figs["dk"],     PANEL)
        p_yk     = fig_to_rgb(figs["yk"],     PANEL)
        p_status = fig_to_rgb(figs["status"], PANEL)
        p_act    = fig_to_rgb(figs["act"],    PANEL)
        p_ep     = fig_to_rgb(figs["ep"],     PANEL)

        # Compose 2×3 grid: [agent, dk, yk ; status, act, ep]
        row1 = np.concatenate([env_frame, p_dk,     p_yk],     axis=1)
        row2 = np.concatenate([p_status,  p_act,    p_ep],     axis=1)
        frame = np.concatenate([row1, row2], axis=0)

        writer.append_data(frame)

        # Progress
        if t % (RENDER * 100) == 0:
            pct = 100 * t / T_EVAL
            print(f"  {pct:5.1f}%  step={t:,}  win={w}  D_k={d_k_c:.4f}  "
                  f"alarm={alarm_level:.2f}  {'[T_WARN FIRED]' if t_warn_fired else ''}",
                  flush=True)

env.close()
writer.close()
print(f"\n[saved] {args.out}")
print(f"  frames written: {T_EVAL // RENDER}  |  duration: {T_EVAL // RENDER / FPS:.0f}s at {FPS}fps")
