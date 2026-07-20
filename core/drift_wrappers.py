"""
drift_wrappers.py — Domain-specific perturbation wrappers for deployment audit
Methodology: v12.3

Three wrappers, selected by domain_config.json "perturbation_wrapper" field:
  GravityPerturbation  : MuJoCo gravity modification
  ObsNoisePerturbation : PyFlyt additive Gaussian observation noise
  ChronicPerturbation  : Grid2Op chronic pool switching

Terminology note: "perturbation" names the MECHANISM (what physically changes —
gravity, obs noise, chronic pool). "transition" names the SCHEDULE (when/how that
mechanism activates over the deployment horizon): "baseline" (mechanism present
but inactive — the null/reference condition), "abrupt", "gradual" (or Grid2Op's
"D0"/"D1"/"D2"). Keeping these as two separate words (rather than both being
"drift") avoids the earlier confusion of e.g. "drift=GravityDrift" printed next
to a run whose schedule is the null condition. "none"/"no_drift" are accepted as
legacy aliases for "baseline" so pre-existing configs and historical result data
keep working.

All wrappers share the same interface:
  - global step counter persists across episode resets
  - transition in {"baseline", "abrupt", "gradual"} (or {"D0","D1","D2"} for Grid2Op);
    "none"/"no_drift" accepted as legacy aliases for "baseline"
  - t0 = 0.5 * T_eval (abrupt onset point)
  - pilot_end = K0 * w (gradual ramp start, applied when gradual_ramp_after_pilot=True)

Abrupt semantics (v12.3):
  A short linear ramp of duration abrupt_ramp_duration (default 50 000 steps) is used
  instead of an instantaneous step.  Rationale: an instantaneous gravity step kills the
  Hopper agent within ~200 steps — before any 50 000-step monitoring window can close —
  making detection structurally impossible.  A 50 000-step ramp (1/10 of the gradual
  ramp length of 500 000 steps) is still "abrupt" in practice while producing a
  detectable signal within 2–3 windows.
"""

import random
import numpy as np
import gymnasium as gym

# Schedule values meaning "the perturbation mechanism is present but inactive this
# run" (the null/reference condition). "baseline" is canonical going forward;
# "none"/"no_drift" are accepted so older configs and historical result JSONs
# (which have these values baked in as data) keep working unchanged.
_BASELINE_ALIASES = ("baseline", "none", "no_drift")


class GravityPerturbationWrapper(gym.Wrapper):
    """
    MuJoCo gravity perturbation: modifies model.opt.gravity[2] in-place.

    Baseline : g = g_nominal throughout
    Abrupt   : g = g_nominal until t0, then linear ramp over abrupt_ramp_duration
               steps to g_drifted, then g_drifted (rapid-gradual semantics)
    Gradual  : linear ramp from g_nominal to g_drifted over [pilot_end, t0]
               pilot period [0, pilot_end) always sees g_nominal

    Documented perturbation: g_nominal=-9.81, g_drifted=-14.715 (1.5x nominal).
    abrupt_ramp_duration default: 50 000 steps (1/10 of gradual ramp length).
    """
    def __init__(self, env, transition, t0, pilot_end,
                 g_nominal=-9.81, g_drifted=-14.715,
                 abrupt_ramp_duration=50_000):
        super().__init__(env)
        self.transition           = transition
        self.t0                   = t0
        self.pilot_end            = pilot_end
        self.g_nominal            = g_nominal
        self.g_drifted            = g_drifted
        self.abrupt_ramp_duration = int(abrupt_ramp_duration)
        self._global_t            = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_gravity()
        return obs, info

    def step(self, action):
        self._global_t += 1
        self._apply_gravity()
        return self.env.step(action)

    def _apply_gravity(self):
        t = self._global_t
        if self.transition in _BASELINE_ALIASES:
            g = self.g_nominal
        elif self.transition == "abrupt":
            if t < self.t0:
                g = self.g_nominal
            elif t < self.t0 + self.abrupt_ramp_duration:
                frac = (t - self.t0) / self.abrupt_ramp_duration
                g = self.g_nominal + frac * (self.g_drifted - self.g_nominal)
            else:
                g = self.g_drifted
        else:  # gradual
            if t < self.pilot_end:
                g = self.g_nominal
            elif self.t0 > self.pilot_end:
                frac = min((t - self.pilot_end) / (self.t0 - self.pilot_end), 1.0)
                g = self.g_nominal + frac * (self.g_drifted - self.g_nominal)
            else:
                g = self.g_drifted
        self.env.unwrapped.model.opt.gravity[2] = g


class ObsNoisePerturbationWrapper(gym.Wrapper):
    """
    PyFlyt observation noise perturbation: additive Gaussian noise on observations.

    Baseline : sigma = 0
    Abrupt   : sigma = 0 until t0, then linear ramp over abrupt_ramp_duration
               steps to noise_std_max, then noise_std_max
    Gradual  : linear ramp from 0 to noise_std_max over [pilot_end, t0]
               pilot period [0, pilot_end) always sees sigma = 0

    Documented perturbation: noise_std_max=0.2 (sensor degradation proxy).
    Noise RNG seeded deterministically per run for reproducibility.
    abrupt_ramp_duration default: 50 000 steps (mirrors GravityPerturbationWrapper).
    """
    def __init__(self, env, transition, t0, pilot_end,
                 noise_std_max=0.2, rng_seed=999,
                 abrupt_ramp_duration=50_000):
        super().__init__(env)
        self.transition           = transition
        self.t0                   = t0
        self.pilot_end            = pilot_end
        self.noise_std_max        = noise_std_max
        self._global_t            = 0
        self._rng                 = np.random.default_rng(rng_seed)
        self.abrupt_ramp_duration = int(abrupt_ramp_duration)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        self._global_t += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        noise = self._sample_noise(obs.shape)
        return obs + noise, reward, terminated, truncated, info

    def _sample_noise(self, shape):
        t = self._global_t
        if self.transition in _BASELINE_ALIASES:
            return np.zeros(shape, dtype=np.float32)
        elif self.transition == "abrupt":
            if t < self.t0:
                std = 0.0
            elif t < self.t0 + self.abrupt_ramp_duration:
                frac = (t - self.t0) / self.abrupt_ramp_duration
                std  = frac * self.noise_std_max
            else:
                std = self.noise_std_max
        else:  # gradual
            if t < self.pilot_end:
                std = 0.0
            elif self.t0 > self.pilot_end:
                frac = min((t - self.pilot_end) / (self.t0 - self.pilot_end), 1.0)
                std  = frac * self.noise_std_max
            else:
                std = self.noise_std_max
        if std == 0.0:
            return np.zeros(shape, dtype=np.float32)
        return self._rng.normal(0.0, std, shape).astype(np.float32)


class ChronicPerturbationWrapper:
    """
    Grid2Op chronic perturbation: switches between train-set and test-set chronics.

    Not a gym.Wrapper subclass — manages chronic selection only.
    Intended to be called at each episode reset via select_chronic(global_step).

    Chronic split: available chronics sorted by index, split 50/50.
      Train-set: first half (nominal deployment)
      Test-set : second half (drifted deployment)

    D0 (alias: baseline): train-set only throughout (mechanism inactive)
    D1 (alias: abrupt)  : train-set until t0, test-set after
    D2 (alias: gradual) : increasing probability of test-set over [pilot_end, t0]
        pilot period [0, pilot_end) always draws from train-set
    """
    def __init__(self, g2op_env, transition, t0, pilot_end, rng_seed):
        n_chronics = g2op_env.chronics_handler.real_data.subpaths
        all_ids    = list(range(len(n_chronics)))
        mid        = len(all_ids) // 2
        self.train_ids  = all_ids[:mid]
        self.test_ids   = all_ids[mid:]
        self.transition = transition
        self.t0         = t0
        self.pilot_end  = pilot_end
        self._rng       = random.Random(rng_seed)

    def select_chronic(self, global_step):
        t = global_step
        if self.transition == "D0" or self.transition in _BASELINE_ALIASES:
            return self._rng.choice(self.train_ids)
        elif self.transition in ("D1", "abrupt"):
            pool = self.test_ids if t >= self.t0 else self.train_ids
            return self._rng.choice(pool)
        else:  # D2 / gradual
            if t < self.pilot_end:
                return self._rng.choice(self.train_ids)
            elif self.t0 > self.pilot_end:
                frac = min((t - self.pilot_end) / (self.t0 - self.pilot_end), 1.0)
            else:
                frac = 1.0
            if self._rng.random() < frac:
                return self._rng.choice(self.test_ids)
            return self._rng.choice(self.train_ids)


PERTURBATION_REGISTRY = {
    "GravityPerturbation":  GravityPerturbationWrapper,
    "ObsNoisePerturbation": ObsNoisePerturbationWrapper,
    "ChronicPerturbation":  ChronicPerturbationWrapper,
    # Legacy aliases — accepted so pre-rename configs still resolve.
    "GravityDrift":  GravityPerturbationWrapper,
    "ObsNoiseDrift": ObsNoisePerturbationWrapper,
    "ChronicDrift":  ChronicPerturbationWrapper,
}

# Backward-compat alias — some scripts may still import WRAPPER_REGISTRY by name.
WRAPPER_REGISTRY = PERTURBATION_REGISTRY
