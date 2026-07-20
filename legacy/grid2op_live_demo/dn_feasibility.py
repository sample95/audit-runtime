#!/usr/bin/env python3
"""
dn_feasibility.py — local pre-deployment feasibility check for the DoNothing
live-monitor demo (configs/grid2op_dn.json). NEW 2026-07-12.

pre_deployment.py gates trained checkpoints; the demo subject is the scripted
do-nothing baseline, so this script answers the question that actually decides
demo feasibility: how often does DoNothing complete a full chronic on the
NOMINAL (train-half) pool, and what does that imply for surviving the
5,040-step pilot (~2.5 consecutive chronic-weeks)?

Run locally (fast, no SB3/torch needed):
    ~/.venvs/audit-framework/bin/python core/dn_feasibility.py --n 30
"""

import argparse

import grid2op
from grid2op.Action import PlayableAction

try:
    from lightsim2grid import LightSimBackend
    backend_kw = {"backend": LightSimBackend()}
except ImportError:
    backend_kw = {}

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="l2rpn_case14_sandbox")
ap.add_argument("--n", type=int, default=30, help="nominal chronics to test")
ap.add_argument("--pilot", type=int, default=5040, help="pilot length (steps)")
args = ap.parse_args()

env = grid2op.make(args.env, action_class=PlayableAction, **backend_kw)
n_total = len(env.chronics_handler.real_data.subpaths)
train_ids = list(range(n_total // 2))[: args.n]          # nominal pool = first half
do_nothing = env.action_space({})

lengths, completed = [], 0
for cid in train_ids:
    env.set_id(cid)
    env.reset()
    t, done = 0, False
    while not done:
        _, _, done, _ = env.step(do_nothing)
        t += 1
    max_t = env.chronics_handler.max_timestep()
    full = t >= max_t
    completed += full
    lengths.append(t)
    print(f"chronic {cid:4d}: survived {t:5d}/{max_t} steps {'COMPLETE' if full else 'BLACKOUT'}")

p = completed / len(train_ids)
need = max(1, round(args.pilot / (sum(lengths) / len(lengths) if lengths else 1)))
print("\n================ summary ================")
print(f"chronics tested          : {len(train_ids)} (nominal pool)")
print(f"completion rate p        : {completed}/{len(train_ids)} = {p:.2f}")
print(f"mean survival            : {sum(lengths)/len(lengths):,.0f} steps")
print(f"pilot needs ~{args.pilot} consecutive steps (~{args.pilot/2016:.1f} chronics)")
print(f"rough P(survive pilot)   : p^3 = {p**3:.2f}  (per seed; demo runs 3 seeds)")
print(f"P(>=1 of 3 seeds passes) : {1-(1-p**3)**3:.2f}")
print("=========================================")
print("Verdict:", "GO — run the Kaggle demo" if 1-(1-p**3)**3 > 0.5 else
      "MARGINAL/NO-GO — expect pre-pilot failures; consider more seeds or branch B")
