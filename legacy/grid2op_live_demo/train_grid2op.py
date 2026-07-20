"""
train_grid2op.py — l2rpn_case14_sandbox training with PPO + VecNormalize
Platform: NAS / Steam Deck / PC
Conditions : G1 (1 000-step full operation), G2 (250-step contingency window),
             G3 (1 000-step, independent seed run — policy diversity)
Budget     : 1 000 000 steps / seed / condition

Saves <run>_vecnormalize.pkl alongside each .zip — required by audit_deploy.py.
Domain config: configs/grid2op.json
"""

import os
import gc
import json
import time
import datetime as dt
import random
import argparse
import concurrent.futures
from collections import defaultdict
from functools import partial

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import grid2op
from grid2op.Action import PlayableAction
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

try:
    from lightsim2grid import LightSimBackend as _LSB
    _BACKEND_CLS = _LSB
    _BACKEND_NAME = "LightSimBackend"
except ImportError:
    _BACKEND_CLS = None
    _BACKEND_NAME = "PandaPower (default)"

DEVICE         = "cpu"
G2OP_ENV_NAME  = "l2rpn_case14_sandbox"

parser = argparse.ArgumentParser()
parser.add_argument("--t_steps",    type=int,   default=1_000_000)
parser.add_argument("--seeds",      type=int,   default=5,
                    help="Number of seeds 0..N-1  (default: 5)")
parser.add_argument("--workers",    type=int,   default=None)
parser.add_argument("--conditions", type=str,   nargs="+", default=None,
                    help="Subset of conditions, e.g. --conditions G1")
parser.add_argument("--initlr",     type=float, default=None)
parser.add_argument("--stps",       type=int,   default=2048,
                    help="PPO n_steps (rollout length)")
parser.add_argument("--eps",        type=int,   default=10,
                    help="PPO n_epochs")
parser.add_argument("--batch",      type=int,   default=64)
parser.add_argument("--ent",        type=float, default=0.0,
                    help="Entropy coefficient")
parser.add_argument("--force",      action="store_true",
                    help="Retrain even if checkpoint + VecNormalize already exist")
args, _ = parser.parse_known_args()

def _auto_output() -> str:
    minute = dt.datetime.now().strftime("%y%m%d%H%M")
    return f"./results/grid2op/grid2op_{minute}"

OUTPUT_DIR = _auto_output()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Observation attributes
OBS_ATTRS = [
    "rho",                  # line loading ratio — key safety metric (20)
    "p_or", "p_ex",         # active power at line origin / extremity (20+20)
    "load_p",               # active power demand at each load (11)
    "gen_p",                # generator active power output (6)
    "actual_dispatch",      # current redispatch delta vs nominal (6)
    "target_dispatch",      # target redispatch setpoint (6)
    "curtailment",          # renewable curtailment fraction (2)
    "curtailment_limit",    # curtailment cap (2)
    "gen_p_before_curtail", # pre-curtailment generation (6)
]

# Action attributes: continuous redispatch (thermal generators) + curtailment
# (renewable generators).  BoxGymActSpace maps these to a flat Box ∈ [-1,1].
ACT_ATTRS = ["redispatch", "curtail"]

# ── Conditions ──
# G1: full-horizon standard operation (analogous to Q1 in hover)
# G2: short-horizon contingency window (analogous to Q2 in hover)
# G3: same max_steps as G1 — independently seeded run for policy diversity.

ALL_CONDITIONS = {
    "G1": {"max_steps": 1000},
    "G2": {"max_steps": 250},
    "G3": {"max_steps": 1000},
}

_cond_filter = set(args.conditions) if args.conditions else None
CONDITIONS   = {k: v for k, v in ALL_CONDITIONS.items()
                if _cond_filter is None or k in _cond_filter}

TOTAL_STEPS = args.t_steps
SEEDS       = list(range(args.seeds))

_cpus       = os.cpu_count() or 2
_n_seeds    = len(SEEDS)
_n_conds    = len(CONDITIONS)

_half_cpus  = max(1, _cpus // 2)
MAX_WORKERS = args.workers if args.workers is not None else min(_n_seeds, _half_cpus)

def LinearSchedule(start: float, end: float):
    def _fn(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return _fn

init_lr = args.initlr
lr      = LinearSchedule(init_lr, 1e-5) if init_lr else 3e-4

PPO_KWARGS = dict(
    learning_rate       = lr,
    n_steps             = args.stps,
    batch_size          = args.batch,
    n_epochs            = args.eps,
    gamma               = 0.99,
    gae_lambda          = 0.95,
    clip_range          = 0.2,
    clip_range_vf       = None,
    ent_coef            = args.ent,
    vf_coef             = 0.5,
    max_grad_norm       = 0.5,
    normalize_advantage = True,
    use_sde             = False,
    target_kl           = None,
    policy_kwargs       = dict(
        net_arch        = dict(pi=[256, 256], vf=[256, 256]),
        activation_fn   = torch.nn.ReLU,   # consistent with other training scripts
        optimizer_class = torch.optim.Adam,
    ),
    verbose = 0,
    device  = DEVICE,
)

class DiagCallback(BaseCallback):
    """Log episode metrics every log_every steps."""

    def __init__(self, run_name: str, total_steps: int, log_every: int = 50_000):
        super().__init__(verbose=0)
        self.run_name  = run_name
        self.total     = total_steps
        self.log_every = log_every
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self.log_every:
            return True
        self._last_log = self.num_timesteps
        buf    = self.model.ep_info_buffer
        mean_r = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
        mean_l = float(np.mean([e["l"] for e in buf])) if buf else float("nan")
        lg     = self.model.logger.name_to_value
        kl     = lg.get("train/approx_kl",    float("nan"))
        vloss  = lg.get("train/value_loss",    float("nan"))
        ent    = lg.get("train/entropy_loss",  float("nan"))
        clip   = lg.get("train/clip_fraction", float("nan"))
        print(
            f"  {self.run_name}  "
            f"steps={self.num_timesteps:>7,}/{self.total:,}  "
            f"ret={mean_r:8.1f}  len={mean_l:6.0f}  "
            f"kl={kl:.4f}  v_loss={vloss:.3f}  ent={ent:.3f}  clip={clip:.3f}",
            flush=True,
        )
        return True


def _make_env(max_steps: int, seed: int, monitor_path: str):
    backend_kw = {"backend": _BACKEND_CLS()} if _BACKEND_CLS is not None else {}
    g2op_env   = grid2op.make(G2OP_ENV_NAME, action_class=PlayableAction, **backend_kw)
    g2op_env.seed(seed)

    gym_env = GymEnv(g2op_env)
    gym_env.observation_space = BoxGymObsSpace(
        g2op_env.observation_space,
        attr_to_keep=OBS_ATTRS,
    )
    gym_env.action_space = BoxGymActSpace(
        g2op_env.action_space,
        attr_to_keep=ACT_ATTRS,
    )

    limited = TimeLimit(gym_env, max_episode_steps=max_steps)
    return Monitor(limited, monitor_path)


# ── Atomic worker: one (condition, seed) pair ───
def train_single(cond_name: str, seed: int, n_workers: int) -> tuple:
    cond_cfg = CONDITIONS[cond_name]
    run_name = f"{cond_name}_seed{seed}"

    threads = max(1, _cpus // n_workers)
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    print(
        f"\n[START] {run_name} | max_steps={cond_cfg['max_steps']} "
        f"threads={threads} pid={os.getpid()}",
        flush=True,
    )

    save_path    = os.path.join(OUTPUT_DIR, f"{run_name}.zip")
    vec_path     = os.path.join(OUTPUT_DIR, f"{run_name}_vecnormalize.pkl")
    monitor_path = os.path.join(OUTPUT_DIR, run_name)

    if os.path.exists(save_path) and os.path.exists(vec_path) and not args.force:
        print(f"  [skip] {run_name} — exists (--force to retrain)", flush=True)
        return cond_name, seed, 0.0, {}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()

    env_factory = partial(_make_env, cond_cfg["max_steps"], seed, monitor_path)
    vec_env     = DummyVecEnv([env_factory])
    vec_env     = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, gamma=PPO_KWARGS["gamma"])

    model = PPO("MlpPolicy", vec_env, seed=seed, **PPO_KWARGS)
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=DiagCallback(run_name, TOTAL_STEPS),
    )

    model.save(save_path)
    vec_env.save(vec_path)
    vec_env.close()

    elapsed   = time.time() - t0
    fps       = int(TOTAL_STEPS / max(elapsed, 1))
    buf       = model.ep_info_buffer
    final_ret = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
    final_len = float(np.mean([e["l"] for e in buf])) if buf else float("nan")

    metrics = {
        run_name: {
            "final_mean_ret": round(final_ret, 2),
            "final_mean_len": round(final_len, 1),
            "train_min":      round(elapsed / 60, 2),
            "avg_fps":        fps,
        }
    }

    print(
        f"  [done] {run_name}  {elapsed/60:.1f} min  ~{fps} fps  "
        f"ret={final_ret:.1f}  len={final_len:.0f}",
        flush=True,
    )

    del model, vec_env
    gc.collect()

    return cond_name, seed, elapsed, metrics


def _run_batch(pairs: list, n_workers: int) -> list:
    if n_workers <= 1 or len(pairs) == 1:
        return [train_single(c, s, 1) for c, s in pairs]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(train_single, c, s, n_workers): (c, s) for c, s in pairs
        }
        results = []
        for fut in concurrent.futures.as_completed(futures):
            res             = fut.result()
            cn, sd, elapsed, _ = res
            print(f"\n[COMPLETE] {cn}_seed{sd} in {elapsed/60:.1f} min", flush=True)
            results.append(res)
    return results


def _write_pre_deployment(all_results: list) -> None:
    ppo_serial = {k: v for k, v in PPO_KWARGS.items() if k != "policy_kwargs"}
    ppo_serial["policy_kwargs"] = {
        "net_arch":      PPO_KWARGS["policy_kwargs"]["net_arch"],
        "activation_fn": "ReLU",
        "optimizer":     "Adam",
    }
    runs = {}
    for _, _, _, metrics in all_results:
        runs.update(metrics)
    payload = {
        "methodology":   "v12.9",
        "env":           G2OP_ENV_NAME,
        "total_steps":   TOTAL_STEPS,
        "n_envs":        1,
        "seeds":         SEEDS,
        "conditions":    ALL_CONDITIONS,
        "obs_attrs":     OBS_ATTRS,
        "act_attrs":     ACT_ATTRS,
        "ppo_kwargs":    ppo_serial,
        "output_dir":    OUTPUT_DIR,
        "runs":          runs,
        "vec_normalize": True,
        "backend":       _BACKEND_NAME,
    }
    out_path = os.path.join(OUTPUT_DIR, "pre_deployment.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\npre_deployment.json → {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"{G2OP_ENV_NAME} TRAINING  [CPU]")
    print(f"Backend    : {_BACKEND_NAME}")
    print(f"Conditions : {list(CONDITIONS.keys())}")
    print(f"Seeds      : {SEEDS}")
    print(f"Workers    : {MAX_WORKERS}  (cpu_count={_cpus})")
    print(f"Steps/run  : {TOTAL_STEPS:,}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60)

    total_start = time.time()
    all_results: list = []

    for cond_name in CONDITIONS:
        print(f"\n{'─' * 40}")
        n_w = min(MAX_WORKERS, _n_seeds)
        print(f"[BATCH] {cond_name} — {_n_seeds} seeds × {n_w} workers")
        pairs = [(cond_name, s) for s in SEEDS]
        all_results.extend(_run_batch(pairs, n_w))

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done in {total_elapsed / 60:.1f} min")

    cond_times: dict = defaultdict(float)
    for cn, _, elapsed, _ in all_results:
        cond_times[cn] += elapsed
    for cn in sorted(cond_times):
        print(f"  {cn}: {cond_times[cn] / 60:.1f} min (cumulative seed time)")
    print("=" * 60)

    _write_pre_deployment(all_results)

    print("\nInventory:")
    for cond_name in ALL_CONDITIONS:
        for seed in SEEDS:
            base    = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}")
            ckpt    = "ok" if os.path.exists(f"{base}.zip")              else "--"
            vec_pkl = "ok" if os.path.exists(f"{base}_vecnormalize.pkl") else "--"
            monitor = "ok" if os.path.exists(f"{base}.monitor.csv")      else "--"
            print(f"  ckpt={ckpt}  vec={vec_pkl}  monitor={monitor}  {cond_name}_seed{seed}")
