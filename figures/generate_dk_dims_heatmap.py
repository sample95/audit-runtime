"""
generate_dk_dims_heatmap.py
Reconstruction of viz6_dk_dims_heatmap.png (2026-07-19).

PROVENANCE NOTE: the original script that produced this figure was never
found. Searched exhaustively: full git history (all 12 commits, all 4
local/remote branches, pickaxe search on "dk_dims_heatmap") and the entire
ThesisStorage filesystem (src_clean, archive/{src,outdated,notebooks,
experiments,walker2d,hover_pilot}, analysis/, scripts/, t13_calib/, docs/,
configs/, deployment/, environment/, training/, .Trash-1000) -- no file
anywhere contains "viz1_"..."viz8_" or "dk_dims_heatmap". It was most likely
run once in an untracked notebook environment (Kaggle/Colab) and never
saved back to this repo. This script reconstructs the figure from the raw
per-window per-dimension JSD series (D_k_dims) that IS still on disk, using
the methodology implied by the surviving PNG's own axes/title/caption:

  - Title/caption on the original PNG read "Per-Dimension D_k Heatmap --
    Gradual Drift (mean across seeds, conditions, policies)".
  - Pooled scope inferred as: both Hopper policy variants ("policies") --
    raw PPO (results/mujoco/hopper/t10_v2) and VecNormalize
    (results/mujoco/hopper_vc) -- all three drift-schedule conditions
    M1/M2/M3 ("conditions"), all seeds 0-4 ("seeds"), gradual transition
    only. Simple mean across whatever runs have data at a given window,
    with NO rescaling to correct for VecNormalize's different D_k scale --
    this is a straightforward pooled mean, matching what the caption says
    literally, not a normalised comparison.
  - x-axis restricted to window_start >= 100,000: with K0=4 pilot windows
    and s=25,000, the first post-pilot window starts at s*K0 = 100,000 --
    the original figure is post-pilot windows only, matching every other
    D_k figure in this thesis.
  - Dimension label ORDER on the y-axis (z_pos, angle, vel_x, vel_z,
    ang_vel, thigh_ang, leg_joint, foot_joint, thigh_angvel, leg_angvel,
    foot_angvel) is copied verbatim from the legacy PNG's own axis labels.
    The JSON metrics files store D_k_dims as a positional 11-vector with
    no embedded label field, so this script ASSUMES index i of D_k_dims
    corresponds to label i of this list, matching the legacy image. This
    assumption could not be independently verified against source and
    should be treated as inherited, not re-derived.

Run:
    python3 generate_dk_dims_heatmap.py --base-dir /path/to/src_clean \
        --out /path/to/src_clean/figures
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIM_LABELS = [
    "z_pos", "angle", "vel_x", "vel_z", "ang_vel",
    "thigh_ang", "leg_joint", "foot_joint",
    "thigh_angvel", "leg_angvel", "foot_angvel",
]
S = 25_000       # window stride (Table 2.1)
K0 = 4           # pilot windows
POST_PILOT_START = S * K0   # 100,000
DPI = 300

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", default=None, help="src_clean root (auto-detected if omitted)")
parser.add_argument("--out", default=None, help="output dir (default: <base-dir>/figures)")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = args.base_dir or os.path.dirname(SCRIPT_DIR)
OUT_DIR = args.out or os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

GLOB_PATTERNS = [
    os.path.join(BASE_DIR, "results", "mujoco", "hopper", "t10_v2", "*_gradual_metrics_summary.json"),
    os.path.join(BASE_DIR, "results", "mujoco", "hopper_vc", "*_gradual_metrics_summary.json"),
]


def load_runs():
    files = []
    for pat in GLOB_PATTERNS:
        files += sorted(glob.glob(pat))
    runs = []
    for f in files:
        d = json.load(open(f))
        ws = d.get("window_starts") or []
        dk = d.get("D_k_dims") or []
        if not ws or not dk or len(ws) != len(dk):
            continue
        runs.append((os.path.basename(f), ws, dk))
    return runs


def build_matrix(runs):
    # bucket[window_start] -> list of 11-dim vectors from every run that has
    # a window starting there
    buckets = {}
    for _name, ws, dk in runs:
        for w, v in zip(ws, dk):
            if w < POST_PILOT_START:
                continue
            if v is None or len(v) != len(DIM_LABELS):
                continue
            buckets.setdefault(w, []).append(v)

    if not buckets:
        raise RuntimeError("No post-pilot gradual windows found — check --base-dir")

    starts = sorted(buckets)
    mat = np.zeros((len(DIM_LABELS), len(starts)))
    n_runs_col = []
    for j, w in enumerate(starts):
        vecs = np.array(buckets[w])          # (n_runs_at_this_window, 11)
        mat[:, j] = vecs.mean(axis=0)
        n_runs_col.append(len(vecs))
    return starts, mat, n_runs_col


def plot(starts, mat, n_runs_col, out_path):
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(DIM_LABELS)))
    ax.set_yticklabels(DIM_LABELS)
    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels([f"{w // 1000}k" for w in starts], rotation=45, ha="right")
    ax.set_xlabel("Window start (×10³ steps)")
    ax.set_ylabel("Observation dimension")
    ax.set_title("Per-Dimension D_k Heatmap — Gradual Drift "
                  "(mean across seeds, conditions, policies)\n"
                  "[reconstructed 2026-07-19; original generator not recoverable, see script header]",
                  fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean D_k_dim (JSD per dim)")
    ax.set_xticks(np.arange(-.5, len(starts), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(DIM_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({len(starts)} window columns, "
          f"{starts[0]}..{starts[-1]}, n_runs per column: "
          f"{min(n_runs_col)}-{max(n_runs_col)})")


if __name__ == "__main__":
    runs = load_runs()
    print(f"Loaded {len(runs)} gradual-condition runs "
          f"(raw PPO t10_v2 + VecNormalize hopper_vc, M1/M2/M3, seeds 0-4).")
    starts, mat, n_runs_col = build_matrix(runs)
    plot(starts, mat, n_runs_col, os.path.join(OUT_DIR, "viz6_dk_dims_heatmap.png"))
