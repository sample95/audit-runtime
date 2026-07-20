#!/usr/bin/env python3
"""
run_grid2op_demo.py — reset-free Grid2op live-monitor demo driver.
[src_clean] NEW FILE 2026-07-07 — DRAFT, UNTESTED ON A LIVE GRID2OP INSTALL.

Runs the UNMODIFIED core/audit_deploy.py (v12.9) against Grid2op by:
  1. registering the "Grid2opChronics-v0" gymnasium env (grid2op_gym_env.py),
  2. swapping PERTURBATION_REGISTRY["ChronicPerturbation"] for the pass-through
     shim (chronic selection happens inside the env),
  3. executing audit_deploy.py via runpy with the demo config.

The instrument code path (windowing, D_k, LOO threshold + floor, channels,
outputs) is therefore identical to the Hopper/hover runs — nothing is
re-implemented (R24/R30).

Usage (from the ThesisStorage root or src_clean/):
    python src_clean/core/run_grid2op_demo.py                       # grid2op_demo config
    python src_clean/core/run_grid2op_demo.py --domain grid2op      # full 3x5 protocol
Smoke test first:  edit configs/grid2op_demo.json T_eval to 60000 and run once;
check the printed window/threshold values before any 600k run.
"""

import os
import sys
import runpy
import argparse
import warnings

# 2026-07-15: pandaPowerBackend emits a SettingWithCopyWarning on effectively
# EVERY env.step() call (internal to grid2op, not something we control). Left
# unsuppressed, a single 600k-step/10-retry deployment run produces so much
# repeated warning noise that the actual [run]/[retry]/[attempts]/[discard]
# diagnostic prints from audit_deploy.py become impossible to find in the
# scrollback.
# FIRST ATTEMPT (didn't work): warnings.filterwarnings("ignore", message=...).
# Still fired on every step -- something in the grid2op/SB3/gym stack resets
# the warnings filter registry between steps (likely a warnings.simplefilter
# call somewhere in that chain), which wipes out any filter we install via
# the standard `warnings` module. pandas has its OWN dedicated switch for
# this exact warning, independent of the general warnings filter registry --
# use that instead, which nothing downstream has a reason to reset.
import pandas as pd
pd.options.mode.chained_assignment = None   # disables SettingWithCopyWarning at the pandas level
warnings.filterwarnings("ignore")  # belt-and-suspenders for anything else noisy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import grid2op_gym_env
grid2op_gym_env.register_env()

# 2026-07-12 FIX (found via live Kaggle run — G1 training succeeded, but both
# the smoke test and the real 600k demo then failed identically):
#   Exception: Can't get attribute 'PlayableAction_l2rpn_case14_sandboxLightSimBackend'
#     on <module 'grid2op.Space.GridObjects'>
#   KeyError: "The observation_space and action_space were not given, can't
#     verify new environments"
# Root cause: grid2op.make() (called inside Grid2opChronicsEnv.__init__)
# dynamically generates a per-environment Action/Observation class at
# CONSTRUCTION time, not at gym.register() time above. audit_deploy.py's
# AlgoCls.load(ckpt_path) runs before any env is constructed in this process,
# so the checkpoint's pickled action_space references a class that doesn't
# exist yet here -> unpickling fails. Works fine during training only because
# train_grid2op.py already constructs the env before saving, in the same
# process. One throwaway construct+close is enough to register the class for
# the rest of this process's lifetime, before audit_deploy.py ever loads a
# checkpoint.
import gymnasium as gym
_warmup_env = gym.make("Grid2opChronics-v0")
_warmup_env.close()
del _warmup_env

import drift_wrappers
# Mutate (not rebind) the dict: audit_deploy imports this same object.
# Both the canonical key and the legacy alias are swapped so the shim takes
# effect regardless of which spelling the domain config uses.
drift_wrappers.PERTURBATION_REGISTRY["ChronicPerturbation"] = grid2op_gym_env.ChronicPassThrough
drift_wrappers.PERTURBATION_REGISTRY["ChronicDrift"] = grid2op_gym_env.ChronicPassThrough

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--domain", default="grid2op_demo")
# 2026-07-15: was HERE/../configs, correct only when this file lived at
# core/run_grid2op_demo.py. Since the 2026-07-12 archival move to
# legacy/grid2op_live_demo/, grid2op.json/grid2op_demo.json/instrument.json
# all live in HERE itself (colocated on purpose) -- default to HERE.
ap.add_argument("--configs", default=HERE)
args, passthrough = ap.parse_known_args()

sys.argv = ["audit_deploy.py", "--domain", args.domain,
            "--configs", args.configs] + passthrough
runpy.run_path(os.path.join(HERE, "audit_deploy.py"), run_name="__main__")
