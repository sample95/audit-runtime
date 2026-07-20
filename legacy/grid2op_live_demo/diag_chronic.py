#!/usr/bin/env python3
"""Diagnostic: is _chronic_exhausted() actually detecting end-of-chronic, or
is it silently failing (via its bare except) and misclassifying every normal
chronic end as a hard failure? Run from src_clean/ on dserver:

    python3 legacy/grid2op_live_demo/diag_chronic.py
"""
import os, sys
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import gymnasium as gym
import grid2op_gym_env
grid2op_gym_env.register_env()

env = gym.make("Grid2opChronics-v0")
g2 = env.unwrapped._g2op

print("=== API check ===")
print("has chronics_handler:", hasattr(g2, "chronics_handler"))
try:
    mt = g2.chronics_handler.max_timestep()
    print("chronics_handler.max_timestep() =", mt)
except Exception as e:
    print("chronics_handler.max_timestep() RAISED:", repr(e))

obs, info = env.reset(seed=0)
print("nb_time_step after reset:", g2.nb_time_step)

print("\n=== Do-nothing rollout (no policy at all) ===")
steps = 0
do_nothing = g2.action_space({})
while True:
    g2obs, reward, done, info = g2.step(do_nothing)
    steps += 1
    if done:
        print(f"done at step {steps}")
        print("info:", info)
        try:
            print("chronics_handler.max_timestep() at done:", g2.chronics_handler.max_timestep())
        except Exception as e:
            print("chronics_handler.max_timestep() RAISED at done:", repr(e))
        break
    if steps > 20000:
        print("no 'done' after 20000 do-nothing steps")
        break

env.close()
