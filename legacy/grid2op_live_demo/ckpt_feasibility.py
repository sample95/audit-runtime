#!/usr/bin/env python3
"""
ckpt_feasibility.py — uncapped survival test for a trained Grid2op checkpoint.
NEW 2026-07-12. Companion to dn_feasibility.py, for SB3 checkpoints.

Pre-deployment eval episodes are truncated at the training cap (1,000 steps),
so mean_ep_len is right-censored and says nothing about surviving the demo
pilot (5,040 steps) or reaching chronic-split drift (>=16,128 steps). This
script deploys the policy reset-free (no cap) over nominal chronics and
reports the uncensored survival distribution.

Run locally:
    ~/.venvs/audit-framework/bin/python core/ckpt_feasibility.py \\
        --ckpt ~/.venvs/audit-framework/results/grid2op/G1_seed3.zip --n 20
"""

import argparse
import os

import numpy as np
import grid2op
from grid2op.Action import PlayableAction
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace
from stable_baselines3 import PPO

try:
    from lightsim2grid import LightSimBackend
    backend_kw = {"backend": LightSimBackend()}
except ImportError:
    backend_kw = {}

# keep in sync with train_grid2op.py / grid2op configs
OBS_ATTRS = ["rho", "p_or", "p_ex", "load_p", "gen_p", "actual_dispatch",
             "target_dispatch", "curtailment", "curtailment_limit",
             "gen_p_before_curtail"]
ACT_ATTRS = ["redispatch", "curtail"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="path to G*_seed*.zip")
ap.add_argument("--env", default="l2rpn_case14_sandbox")
ap.add_argument("--n", type=int, default=20, help="nominal chronics to test")
ap.add_argument("--pilot", type=int, default=5040)
ap.add_argument("--drift_at", type=int, default=16128,
                help="steps needed before chronic-split drift arrives (2 chronics)")
args = ap.parse_args()

env = grid2op.make(args.env, action_class=PlayableAction, **backend_kw)
gym_env = GymEnv(env)
gym_env.observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=OBS_ATTRS)
gym_env.action_space = BoxGymActSpace(env.action_space, attr_to_keep=ACT_ATTRS)

model = PPO.load(args.ckpt, device="cpu")
vec_path = args.ckpt.replace(".zip", "_vecnormalize.pkl")
vecnorm = None
if os.path.exists(vec_path):
    from stable_baselines3.common.vec_env import VecNormalize
    import pickle
    with open(vec_path, "rb") as f:
        vecnorm = pickle.load(f)
    vecnorm.training = False
    print(f"[vecnorm] frozen stats loaded from {os.path.basename(vec_path)}")

def norm(obs):
    return vecnorm.normalize_obs(obs) if vecnorm is not None else obs

n_total = len(env.chronics_handler.real_data.subpaths)
train_ids = list(range(n_total // 2))[: args.n]

lengths = []
for cid in train_ids:
    env.set_id(cid)
    out = gym_env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    t, done = 0, False
    while not done:
        action, _ = model.predict(norm(np.asarray(obs, dtype=np.float32)),
                                  deterministic=True)
        out = gym_env.step(action)
        obs, done = out[0], (out[2] or out[3]) if len(out) == 5 else out[2]
        t += 1
    max_t = env.chronics_handler.max_timestep()
    lengths.append((t, t >= max_t))
    print(f"chronic {cid:4d}: survived {t:5d}/{max_t} {'COMPLETE' if t >= max_t else 'BLACKOUT'}")

surv = [t for t, _ in lengths]
p_pilot = sum(t >= args.pilot for t in surv) / len(surv)
p_full  = sum(full for _, full in lengths) / len(surv)
print("\n================ summary ================")
print(f"ckpt                     : {args.ckpt}")
print(f"chronics tested          : {len(surv)} (nominal pool)")
print(f"survival mean/median/max : {np.mean(surv):,.0f} / {np.median(surv):,.0f} / {max(surv):,}")
print(f"P(survive pilot {args.pilot})   : {p_pilot:.2f}")
print(f"P(complete chronic 8064) : {p_full:.2f}   -> P(reach drift ~{args.drift_at}) ~ {p_full**2:.2f}")
print(f"P(demo works, 3 seeds)   : {1-(1-p_full**2)**3:.2f}  (needs pilot AND drift arrival)")
print("=========================================")
print("Verdict:", "GO" if 1-(1-p_full**2)**3 > 0.5 else "NO-GO — live demo stays infeasible with this subject")
