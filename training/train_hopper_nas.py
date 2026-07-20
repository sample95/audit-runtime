import os
import gc
import json
import time
import datetime as dt
import random
import argparse
import concurrent.futures

# ── Force CPU ─────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"]  = ""
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import LinearSchedule


DEVICE = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--conditions", nargs="+", default=None)
parser.add_argument("--seeds",   type=int, default=5,
                    help="Number of seeds 0..N-1 (default: 5)")
parser.add_argument("--workers", type=int, default=None,
                    help="Parallel condition workers (default: cpu_count-1, min 1)")
parser.add_argument("--output",  type=str, default=None,
                    help="Output directory (default: auto-detected)")
parser.add_argument("--force",   action="store_true",
                    help="Retrain even if checkpoint exists")
args, _ = parser.parse_known_args()  # ignore Jupyter kernel args

def _auto_output() -> str:
    minute = dt.datetime.now().strftime("%y%m%d%H%M")
    base_name = f"hopper_{minute}"
    if os.path.isdir("/content/drive/MyDrive"):
        return f"/content/drive/MyDrive/{base_name}"
    if os.path.isdir("/kaggle/working"):
        return f"/kaggle/working/{base_name}"
    return f"./results/mujoco/hopper/{base_name}"

OUTPUT_DIR = _auto_output()
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_STEPS = 1_500_000
SEEDS       = list(range(args.seeds))

ALL_CONDITIONS = {
    "M1": {"time_limit": 1000, "reset_noise_scale": 0.005},
    "M2": {"time_limit": 250,  "reset_noise_scale": 0.005},
    "M3": {"time_limit": 1000, "reset_noise_scale": 0.02},
}
CONDITIONS = {k: v for k, v in ALL_CONDITIONS.items()
              if args.conditions is None or k in args.conditions}

_cpus       = os.cpu_count() or 2
MAX_WORKERS = min(
    args.workers if args.workers is not None else max(_cpus - 1, 1),
    len(CONDITIONS),
)

PPO_KWARGS = dict(
    learning_rate   = LinearSchedule(2e-4, 1e-5, 1.0),
    n_steps         = 2048,
    batch_size      = 128,
    n_epochs        = 10,
    gamma           = 0.995,
    gae_lambda      = 0.95,
    clip_range      = LinearSchedule(0.2, 0.1, 0.5),
    clip_range_vf   = None,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
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


# ── Single-condition worker ───────────────────────────────────────────────────
def train_condition(cond_name: str) -> tuple[str, float, dict]:
    cond_cfg   = CONDITIONS[cond_name]
    cond_start = time.time()

    threads = max(1, (_cpus) // MAX_WORKERS)
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    print(f"\n[START] {cond_name} | limit={cond_cfg['time_limit']} "
          f"noise={cond_cfg['reset_noise_scale']} threads={threads} pid={os.getpid()}",
          flush=True)

    summary = {}

    for seed in SEEDS:
        run_name     = f"{cond_name}_seed{seed}"
        save_path    = os.path.join(OUTPUT_DIR, f"{run_name}.zip")
        monitor_path = os.path.join(OUTPUT_DIR, run_name)

        if os.path.exists(save_path) and not args.force:
            print(f"  [skip] {run_name} — exists", flush=True)
            continue

        # Isolate RNG state between seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        t0 = time.time()

        def _make_env():
            env = gym.make(
                "Hopper-v5",
                max_episode_steps = cond_cfg["time_limit"],
                reset_noise_scale = cond_cfg["reset_noise_scale"],
            )
            env.reset(seed=seed)
            return Monitor(env, monitor_path)

        vec_env = DummyVecEnv([_make_env])

        model = PPO("MlpPolicy", vec_env, seed=seed, **PPO_KWARGS)
        model.learn(
            total_timesteps = TOTAL_STEPS,
            callback        = DiagCallback(run_name, TOTAL_STEPS),
        )
        model.save(save_path)
        vec_env.close()

        elapsed = time.time() - t0
        fps     = int(TOTAL_STEPS / max(elapsed, 1))

        buf       = model.ep_info_buffer
        final_ret = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
        final_len = float(np.mean([e["l"] for e in buf])) if buf else float("nan")

        summary[run_name] = {
            "final_mean_ret": round(final_ret, 2),
            "final_mean_len": round(final_len, 1),
            "train_min":      round(elapsed / 60, 2),
            "avg_fps":        fps,
        }

        print(f"  [done] {run_name}  {elapsed/60:.1f} min  ~{fps} fps  "
              f"ret={final_ret:.1f}  len={final_len:.0f}", flush=True)

        del model, vec_env
        gc.collect()

    return cond_name, time.time() - cond_start, summary


def _write_pre_deployment(results: list) -> None:
    l_r = PPO_KWARGS.pop("learning_rate")
    c_r = PPO_KWARGS.pop("clip_range")
    ppo_serial = {k: v for k, v in PPO_KWARGS.items() if k != "policy_kwargs"}
    ppo_serial["learning_rate"] = str(l_r)
    ppo_serial["clip_range"] = str(c_r)
    ppo_serial["policy_kwargs"] = {
        "net_arch":     PPO_KWARGS["policy_kwargs"]["net_arch"],
        "activation_fn": "ReLU",
        "optimizer":     "Adam",
    }

    runs = {}
    for _, _, summary in results:
        runs.update(summary)

    payload = {
        "methodology":  "v12.6",
        "env":          "Hopper-v5",
        "total_steps":  TOTAL_STEPS,
        "seeds":        SEEDS,
        "conditions":   ALL_CONDITIONS,
        "ppo_kwargs":   ppo_serial,
        "output_dir":   OUTPUT_DIR,
        "runs":         runs,
    }

    out_path = os.path.join(OUTPUT_DIR, "pre_deployment.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\npre_deployment.json → {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("HOPPER-v5 TRAINING  [CPU]")
    print(f"Conditions : {list(CONDITIONS.keys())}")
    print(f"Seeds      : {SEEDS}")
    print(f"Workers    : {MAX_WORKERS}  (os.cpu_count={_cpus})")
    print(f"Steps/run  : {TOTAL_STEPS:,}")
    print(f"Output     : {OUTPUT_DIR}")
    print("=" * 60)

    total_start = time.time()

    if MAX_WORKERS == 1:
        results = [train_condition(c) for c in CONDITIONS]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(train_condition, c): c for c in CONDITIONS}
            results = []
            for fut in concurrent.futures.as_completed(futures):
                cond_name, elapsed, summary = fut.result()
                results.append((cond_name, elapsed, summary))
                print(f"\n[COMPLETE] {cond_name} in {elapsed/60:.1f} min", flush=True)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done in {total_elapsed/60:.1f} min")
    for cond_name, elapsed, _ in sorted(results):
        print(f"  {cond_name}: {elapsed/60:.1f} min")
    print("=" * 60)

    _write_pre_deployment(results)

    print("\nInventory:")
    for cond_name in ALL_CONDITIONS:
        for seed in SEEDS:
            base    = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}")
            ckpt    = "ok" if os.path.exists(f"{base}.zip")                else "--"
            monitor = "ok" if os.path.exists(f"{base}.monitor.csv")        else "--"
            print(f"  ckpt={ckpt}  monitor={monitor}  {cond_name}_seed{seed}")
