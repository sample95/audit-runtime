"""
grid2op_gym_env.py — gymnasium adapter for the Grid2op live-monitor demo.
[src_clean] NEW FILE 2026-07-07 — DRAFT, UNTESTED ON A LIVE GRID2OP INSTALL.
Smoke-test on a short horizon (T_eval=60k) before any thesis-quoted run (R24).

Purpose: make the Grid2op deployment runnable through the UNMODIFIED
core/audit_deploy.py (v12.9). audit_deploy builds environments with
gym.make(env_id) and wraps them with PERTURBATION_REGISTRY[perturbation_wrapper].
This module provides:

  1. Grid2opChronicsEnv — a gymnasium.Env that wraps grid2op.make() with the
     SAME BoxGymObsSpace/BoxGymActSpace construction as
     training/train_grid2op.py (keep OBS_ATTRS/ACT_ATTRS in sync with it),
     handles chronic selection through drift_wrappers.ChronicPerturbationWrapper,
     and maps Grid2op episode ends onto reset-free deployment semantics:
        blackout / game-over  -> terminated=True  (real hard failure;
                                  audit_deploy stops: T_hard)
        end-of-chronic        -> truncated=True   (data exhausted, NOT a
                                  failure; audit_deploy resets and the next
                                  chronic is drawn per the transition schedule)
  2. ChronicPassThrough — a thin gym.Wrapper with the calling signature
     audit_deploy uses for perturbation wrappers; it just forwards the
     schedule parameters into the env. Install it with:
        drift_wrappers.PERTURBATION_REGISTRY["ChronicPerturbation"] = ChronicPassThrough
  3. register_env() — registers "Grid2opChronics-v0" with gymnasium.

VERIFY ON FIRST RUN (marked in code):
  - the 4- vs 5-tuple return of grid2op.gym_compat.GymEnv.step for the
    installed grid2op version;
  - the end-of-chronic detection (_chronic_exhausted) against a manual check.
"""

import os
import sys
import random

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # [src_clean] shim
from drift_wrappers import ChronicPerturbationWrapper

import grid2op
from grid2op.Action import PlayableAction
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace, MultiDiscreteActSpace

try:  # LightSim backend: 10-100x faster; silent fallback like train_grid2op.py
    from lightsim2grid import LightSimBackend as _BACKEND_CLS
except ImportError:
    _BACKEND_CLS = None

# Keep in sync with training/train_grid2op.py (and configs/grid2op*.json).
OBS_ATTRS = [
    "rho", "p_or", "p_ex", "load_p", "gen_p",
    "actual_dispatch", "target_dispatch",
    "curtailment", "curtailment_limit", "gen_p_before_curtail",
]
ACT_ATTRS = ["redispatch", "curtail"]


class Grid2opChronicsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_name="l2rpn_case14_sandbox",
                 obs_attrs=None, act_attrs=None, act_space="box", rng_seed=42,
                 scripted_agent=None, **_ignored):
        super().__init__()
        # act_space: "box" (BoxGymActSpace, continuous — original redispatch+curtail
        # setup, still what G2's checkpoint expects) or "multidiscrete"
        # (MultiDiscreteActSpace — G1/G3's retrained one_sub_set topology-action
        # setup, added 2026-07-15). Must match whichever space the loaded
        # checkpoint's policy network was actually trained against, per
        # domain config's action_space_by_condition map — a mismatch here will
        # raise a clear SB3 space-mismatch error rather than fail silently.
        # UNTESTED path (multidiscrete): no grid2op install available to verify
        # locally; smoke-test on a short T_eval before any thesis-quoted run.
        backend_kw = {"backend": _BACKEND_CLS()} if _BACKEND_CLS is not None else {}
        self._g2op = grid2op.make(env_name, action_class=PlayableAction, **backend_kw)
        self._gym = GymEnv(self._g2op)
        self._gym.observation_space = BoxGymObsSpace(
            self._g2op.observation_space, attr_to_keep=obs_attrs or OBS_ATTRS)
        _act_attrs = act_attrs or ACT_ATTRS
        if act_space == "multidiscrete":
            self._gym.action_space = MultiDiscreteActSpace(
                self._g2op.action_space, attr_to_keep=_act_attrs)
        elif act_space == "box":
            self._gym.action_space = BoxGymActSpace(
                self._g2op.action_space, attr_to_keep=_act_attrs)
        else:
            raise ValueError(f"unknown act_space: {act_space!r} (expected 'box' or 'multidiscrete')")
        self.observation_space = self._gym.observation_space
        self.action_space = self._gym.action_space
        self._rng_seed = int(rng_seed)
        self._drift = None          # set via configure_drift (ChronicPassThrough)
        self._global_step = 0       # deployment-global step (never resets)
        # scripted_agent="do_nothing": incoming actions are IGNORED and the grid
        # is stepped with the empty PlayableAction. The audited frozen policy is
        # then the do-nothing baseline; the SB3 checkpoint audit_deploy loads is
        # an inert driver. Declare this in any run that uses it.
        self._scripted = scripted_agent

    # Called by ChronicPassThrough with audit_deploy's schedule parameters.
    def configure_drift(self, transition, t0, pilot_end):
        self._drift = ChronicPerturbationWrapper(
            self._g2op, transition=transition, t0=t0,
            pilot_end=pilot_end, rng_seed=self._rng_seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._g2op.seed(int(seed))
            if self._drift is not None:
                # 2026-07-16: previously self._drift's chronic-selection RNG
                # was seeded ONLY from the domain config's fixed env_kwargs
                # "rng_seed" (constant == 42 for every checkpoint, every
                # attempt, every retry). select_chronic() is called once per
                # reset with random.Random(rng_seed).choice(...) — from an
                # identical seed, that ALWAYS returns the same first chronic.
                # audit_deploy.py's deploy_retry_attempts mechanism therefore
                # was not diversifying which scenario got played across
                # retries at all -- only self._g2op.seed(deploy_seed) (grid
                # solver/load noise) varied, while every attempt replayed the
                # exact same opening chronic. Reseeding the SAME drift
                # instance's RNG here, from the per-attempt deploy seed
                # (already passed into this reset() call), makes retries draw
                # genuinely independent chronic sequences, as
                # deploy_retry_attempts was intended to.
                self._drift._rng = random.Random(int(seed))
        if self._drift is not None:
            self._g2op.set_id(self._drift.select_chronic(self._global_step))
        out = self._gym.reset()
        obs, info = out if isinstance(out, tuple) and len(out) == 2 else (out, {})
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        if self._scripted == "do_nothing":
            g2obs, reward, done, info = self._g2op.step(self._g2op.action_space({}))
            obs = self._gym.observation_space.to_gym(g2obs)
            terminated, truncated = bool(done), False
        else:
            out = self._gym.step(action)
            if len(out) == 5:                  # gymnasium-mode gym_compat
                obs, reward, terminated, truncated, info = out
            else:                              # VERIFY: legacy 4-tuple mode
                obs, reward, done, info = out
                terminated, truncated = bool(done), False
        self._global_step += 1
        if terminated or truncated:
            # Reclassify per reset-free deployment semantics (VERIFY on first run):
            # chronic exhausted -> truncated (not a failure); else blackout -> terminated.
            if self._chronic_exhausted():
                terminated, truncated = False, True
            else:
                terminated, truncated = True, False
        return (np.asarray(obs, dtype=np.float32), float(reward),
                terminated, truncated, info)

    def _chronic_exhausted(self):
        """True if the episode ended because the chronic ran out of data."""
        try:  # VERIFY against the installed grid2op version's API
            return self._g2op.nb_time_step >= self._g2op.chronics_handler.max_timestep()
        except Exception:
            return False

    def close(self):
        self._gym.close()


class ChronicPassThrough(gym.Wrapper):
    """Perturbation-wrapper shim matching audit_deploy's call signature:
    PerturbationWrapper(base_env, transition=..., t0=..., pilot_end=..., **kwargs).
    Chronic selection itself lives inside Grid2opChronicsEnv."""
    def __init__(self, env, transition, t0, pilot_end, **_ignored):
        super().__init__(env)
        env.unwrapped.configure_drift(transition, t0, pilot_end)


def register_env():
    if "Grid2opChronics-v0" not in gym.registry:
        gym.register(id="Grid2opChronics-v0",
                     entry_point=Grid2opChronicsEnv,
                     disable_env_checker=True)
