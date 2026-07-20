"""
train_hover.py — PyFlyt/QuadX-Hover-v4 training with VecNormalize
Platform: Kaggle / Colab / NAS / PC
Conditions: Q1 (long-horizon), Q2 (short-horizon) | Seeds: configurable
Budget: 1,000,000 steps/seed/condition
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

os.environ["CUDA_VISIBLE_DEVICES"]  = ""
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("PYFLYT_HEADLESS", "1")

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

DEVICE = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--t_steps", type=int, default=1_000_000)
parser.add_argument("--seeds",   type=int, default=5,
                    help="Number of seeds 0..N-1 (default: 5)")
parser.add_argument("--n_envs",   type=int, default=None)
parser.add_argument("--workers",   type=int, default=None)
parser.add_argument("--initlr",  type=float, default=None,
                    help="Initial learning rate")
parser.add_argument("--stps",    type=int, default=2048)
parser.add_argument("--eps",     type=int, default=10)
parser.add_argument("--endclp",  type=float, default=0.7)
parser.add_argument("--batch",   type=int, default=64)
parser.add_argument("--ent",     type=float, default=0.0,
                    help="Entropy coefficient")
parser.add_argument("--contrun", type=int, default=None)
parser.add_argument("--force",   action="store_true",
                    help="Retrain even if checkpoint + VecNormalize already exist")
args, _ = parser.parse_known_args()

HOVER_ENV_ID = "PyFlyt/QuadX-Hover-v4"

def _auto_output() -> str:
    minute    = args.contrun or dt.datetime.now().strftime("%y%m%d%H%M")
    base_name = f"hover_{minute}"
    if os.path.isdir("/content/drive/MyDrive"):
        return f"/content/drive/MyDrive/{base_name}"
    if os.path.isdir("/kaggle/working"):
        return f"/kaggle/working/{base_name}"
    return f"./results/pyflyt/hover/{base_name}"
OUTPUT_DIR = _auto_output()
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_STEPS = args.t_steps
SEEDS       = list(range(args.seeds))

ALL_CONDITIONS = {
    "Q1": {"max_duration_seconds": 10.0},   # long-horizon hover  (~1200 steps at 120 Hz)
    "Q2": {"max_duration_seconds": 2.5},    # short-horizon hover (~300 steps at 120 Hz)
}
CONDITIONS = {k: v for k, v in ALL_CONDITIONS.items()}

_cpus    = os.cpu_count() or 2
_n_seeds = len(SEEDS)
_n_conds = len(CONDITIONS)

_target_workers = min(_n_conds, _n_seeds)          # how many jobs we want in parallel
_default_n_envs = max(1, _cpus // _target_workers // 2)
N_ENVS = args.n_envs if args.n_envs is not None else _default_n_envs

_effective_cpus = max(1, _cpus // max(1, N_ENVS))

# ── Parallelisation mode (secondary axis) ─
if _effective_cpus >= _n_seeds:
    _MODE       = "by_condition"
    MAX_WORKERS = min(_n_seeds, _effective_cpus)
else:
    _MODE       = "by_seed"
    MAX_WORKERS = min(_n_conds, _effective_cpus)

if args.workers is not None:
    MAX_WORKERS = max(1, min(args.workers, _effective_cpus))

def LinearSchedule(start: float, end: float, total_frac: float = 1.0):
    def _schedule(progress_remaining: float) -> float:
        return end + (start - end) * min(progress_remaining, total_frac) / total_frac
    return _schedule

# ── Hyperparameters ─────
init_lr = args.initlr
lr      = LinearSchedule(init_lr, 1e-5, 1.0) if init_lr else args.initlr or 1e-4
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
        activation_fn   = torch.nn.ReLU,
        optimizer_class = torch.optim.Adam,
    ),
    verbose = 0,
    device  = DEVICE,
)

class DiagCallback(BaseCallback):
    def __init__(self, run_name: str, total_steps: int, log_every: int = 100_000):
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


def _merge_monitor_files(base_path: str, n_envs: int) -> None:
    import re as _re

    all_records = []
    for i in range(n_envs):
        src = f"{base_path}_env{i}.monitor.csv"
        if not os.path.exists(src):
            continue
        with open(src, "r") as f:
            raw = f.read()
        for m in _re.finditer(
            r"(-?\d+\.?\d+(?:[eE][+-]?\d+)?),(\d+),(-?\d+\.?\d+(?:[eE][+-]?\d+)?)",
            raw,
        ):
            all_records.append((float(m.group(3)), float(m.group(1)), int(m.group(2))))

    if not all_records:
        return

    all_records.sort()          # sort by wall-clock time (field 0)
    t_start = all_records[0][0]

    out_path = f"{base_path}.monitor.csv"
    with open(out_path, "w") as f:
        f.write(f'#{{"t_start": {t_start:.7f}, "env_id": "{HOVER_ENV_ID}"}}\n')
        f.write("r,l,t\n")
        for t, r, l in all_records:
            f.write(f"{r:.6f},{l},{t:.6f}\n")

    # Remove per-env files once merged
    for i in range(n_envs):
        src = f"{base_path}_env{i}.monitor.csv"
        if os.path.exists(src):
            os.remove(src)

    print(f"  [monitor] merged {len(all_records)} episodes from {n_envs} env(s) → {out_path}",
          flush=True)


def make_env_fn(max_duration_seconds: float, seed: int, monitor_path: str):
    def _init():
        env = gym.make(HOVER_ENV_ID, max_duration_seconds=max_duration_seconds)
        env.reset(seed=seed)
        return Monitor(env, monitor_path)
    return _init


# ── Atomic worker: one (condition, seed) pair ───
def train_single(cond_name: str, seed: int, n_workers: int) -> tuple:
    cond_cfg = CONDITIONS[cond_name]
    run_name = f"{cond_name}_seed{seed}"

    threads = max(1, _cpus // (n_workers * N_ENVS))
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    print(
        f"\n[START] {run_name} | max_duration={cond_cfg['max_duration_seconds']}s "
        f"N_ENVS={N_ENVS} threads={threads} pid={os.getpid()}",
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

    import PyFlyt.gym_envs  # noqa: F401

    env_fns = [
        make_env_fn(
            cond_cfg["max_duration_seconds"],
            seed + i * 1000,
            f"{monitor_path}_env{i}",
        )
        for i in range(N_ENVS)
    ]

    if N_ENVS > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    model   = PPO("MlpPolicy", vec_env, seed=seed, **PPO_KWARGS)
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=DiagCallback(run_name, TOTAL_STEPS),
    )

    model.save(save_path)
    vec_env.save(vec_path)
    vec_env.close()

    _merge_monitor_files(monitor_path, N_ENVS)

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
        batch_results = []
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            cond_name, seed, elapsed, _ = result
            print(
                f"\n[COMPLETE] {cond_name}_seed{seed} in {elapsed/60:.1f} min",
                flush=True,
            )
            batch_results.append(result)
    return batch_results


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
        "env":           HOVER_ENV_ID,
        "total_steps":   TOTAL_STEPS,
        "n_envs":        N_ENVS,
        "seeds":         SEEDS,
        "conditions":    ALL_CONDITIONS,
        "ppo_kwargs":    ppo_serial,
        "output_dir":    OUTPUT_DIR,
        "runs":          runs,
        "vec_normalize": True,
    }
    out_path = os.path.join(OUTPUT_DIR, "pre_deployment.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\npre_deployment.json → {out_path}")


if __name__ == "__main__":
    vec_type = f"SubprocVecEnv×{N_ENVS}" if N_ENVS > 1 else "DummyVecEnv"
    print("=" * 60)
    print(f"{HOVER_ENV_ID} TRAINING  [CPU]")
    print(f"Mode       : {_MODE}  (cpu_count={_cpus}, effective_cpus={_effective_cpus})")
    print(f"VecEnv     : {vec_type}  →  rollout={PPO_KWARGS['n_steps'] * N_ENVS:,} steps/update")
    print(f"Conditions : {list(CONDITIONS.keys())}")
    print(f"Seeds      : {SEEDS}")
    print(f"Workers    : {MAX_WORKERS}")
    print(f"Steps/run  : {TOTAL_STEPS:,}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60)

    total_start = time.time()
    all_results: list = []

    if _MODE == "by_condition":
        for cond_name in CONDITIONS:
            print(f"\n{'─' * 40}")
            print(f"[BATCH] {cond_name} — {_n_seeds} seeds × {MAX_WORKERS} workers")
            pairs = [(cond_name, s) for s in SEEDS]
            all_results.extend(_run_batch(pairs, MAX_WORKERS))
    else:
        for seed in SEEDS:
            print(f"\n{'─' * 40}")
            print(f"[BATCH] seed {seed} — {_n_conds} conditions × {MAX_WORKERS} workers")
            pairs = [(c, seed) for c in CONDITIONS]
            all_results.extend(_run_batch(pairs, MAX_WORKERS))

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done in {total_elapsed/60:.1f} min")

    cond_times: dict = defaultdict(float)
    for cond_name, seed, elapsed, _ in all_results:
        cond_times[cond_name] += elapsed
    for cond_name in sorted(cond_times):
        print(f"  {cond_name}: {cond_times[cond_name]/60:.1f} min (cumulative seed time)")
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
