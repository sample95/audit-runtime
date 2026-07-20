#!/usr/bin/env python3
"""
fetch_walker2d_zoo.py - downloads the public SB3-Zoo Walker2d checkpoints from
HuggingFace and rename them into the audit framework's checkpoint convention.

Usage: python training/fetch_walker2d_zoo.py [--out results/mujoco/walker2d_zoo]
"""

import os
import argparse
import datetime as dt
import shutil

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

JOBS = [
    # (repo_id, candidate model filenames, target zip, vecnorm candidates, target pkl)
    ("sb3/ppo-Walker2d-v3", ["ppo-Walker2d-v3.zip"], "ZP_seed0.zip",
     ["vec_normalize.pkl", "vecnormalize.pkl"], "ZP_seed0_vecnormalize.pkl"),
    ("sb3/sac-Walker2d-v3", ["sac-Walker2d-v3.zip"], "ZS_seed0.zip",
     [], None),
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/mujoco/walker2d_zoo")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

prov = [f"# Provenance — external Walker2d checkpoints\n",
        f"Downloaded {dt.date.today().isoformat()} via huggingface_hub.\n"]

for repo, model_names, target, vec_names, vec_target in JOBS:
    got = None
    for name in model_names:
        try:
            got = hf_hub_download(repo_id=repo, filename=name)
            break
        except EntryNotFoundError:
            continue
    if got is None:
        raise SystemExit(f"model file not found in {repo}: tried {model_names}")
    dst = os.path.join(args.out, target)
    shutil.copyfile(got, dst)
    print(f"{repo} -> {dst}")
    prov.append(f"- `{target}` <- https://huggingface.co/{repo} (file `{os.path.basename(got)}`)")
    for vname in vec_names:
        try:
            vgot = hf_hub_download(repo_id=repo, filename=vname)
            vdst = os.path.join(args.out, vec_target)
            shutil.copyfile(vgot, vdst)
            print(f"{repo} -> {vdst}")
            prov.append(f"- `{vec_target}` <- https://huggingface.co/{repo} (file `{vname}`)")
            break
        except EntryNotFoundError:
            continue

prov.append("\nTrained on Walker2d-v3 by DLR-RM/rl-baselines3-zoo (zoo hyperparameters, "
            "single seed). Deployed here on Walker2d-v5 — version gap declared in "
            "configs/walker2d_zoo_*.json; run baseline + pre-deployment gate first.\n")
with open(os.path.join(args.out, "PROVENANCE.md"), "w") as f:
    f.write("\n".join(prov))
print("PROVENANCE.md written.")
