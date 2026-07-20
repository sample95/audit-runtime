"""
train_hopper_vecnorm.py — Hopper-v5 PPO + VecNormalize
Conditions: M1, M2, M3 | Seeds: 0-2 | Steps: 1,000,000
Saves: <cond>_seed<n>.zip + <cond>_seed<n>_vecnormalize.pkl

Runs M1/M2/M3 in parallel (ProcessPoolExecutor, max_workers=2).
Thread limiting prevents CPU thrashing across processes.
"""

import os
import time
import concurrent.futures

# Limit threads per process — prevents CPU lockup when running parallel conditions
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

OUTPUT_DIR  = "./results/mujoco/hopper_vecnorm"
TOTAL_STEPS = 1_000_000
SEEDS       = [0, 1, 2]

CONDITIONS = {
    "M1": {"time_limit": 1000, "reset_noise_scale": 0.005},
    "M2": {"time_limit": 250,  "reset_noise_scale": 0.005},
    "M3": {"time_limit": 1000, "reset_noise_scale": 0.02},
}

PPO_KWARGS = dict(
    learning_rate       = 1e-4,
    n_steps             = 2048,
    batch_size          = 128,
    n_epochs            = 10,
    gamma               = 0.99,
    gae_lambda          = 0.95,
    clip_range          = 0.1,
    ent_coef            = 0.01,
    vf_coef             = 0.5,
    max_grad_norm       = 0.5,
    normalize_advantage = True,
    policy_kwargs       = dict(
        net_arch        = dict(pi=[256, 256], vf=[256, 256]),
        activation_fn   = torch.nn.ReLU,
        optimizer_class = torch.optim.Adam,
    ),
    verbose = 0,   # silent — parallel processes would garble the terminal
    device  = "cpu",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_condition(cond_name):
    cond_cfg   = CONDITIONS[cond_name]
    cond_start = time.time()
    print(f"[START] {cond_name} | TimeLimit={cond_cfg['time_limit']} | "
          f"noise={cond_cfg['reset_noise_scale']}", flush=True)

    for seed in SEEDS:
        save_path    = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}.zip")
        vecnorm_path = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}_vecnormalize.pkl")

        if os.path.exists(save_path) and os.path.exists(vecnorm_path):
            print(f"  [skip] {cond_name} seed={seed}", flush=True)
            continue

        t0 = time.time()

        def _make_env(s=seed, cfg=cond_cfg):
            env = gym.make("Hopper-v5",
                           max_episode_steps=cfg["time_limit"],
                           reset_noise_scale=cfg["reset_noise_scale"])
            env.reset(seed=s)
            return env

        vec_env = DummyVecEnv([_make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = PPO("MlpPolicy", vec_env, seed=seed, **PPO_KWARGS)
        model.learn(total_timesteps=TOTAL_STEPS)

        model.save(save_path)
        vec_env.save(vecnorm_path)
        vec_env.close()

        print(f"  [done] {cond_name} seed={seed} ({(time.time()-t0)/60:.1f} min)", flush=True)

    return cond_name, time.time() - cond_start


if __name__ == "__main__":
    print("=" * 60)
    print("HOPPER-v5 VecNorm TRAINING (Parallelized by Condition)")
    print(f"Conditions: {list(CONDITIONS.keys())} | Seeds: {SEEDS}")
    print(f"Steps/run: {TOTAL_STEPS:,} | max_workers=2 | VecNorm=True | Device: cpu")
    print("=" * 60)

    total_start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(train_condition, cond): cond for cond in CONDITIONS}
        for future in concurrent.futures.as_completed(futures):
            cond_name, elapsed = future.result()
            print(f"[COMPLETE] {cond_name} in {elapsed/60:.1f} min", flush=True)

    print(f"\nAll done in {(time.time()-total_start)/60:.1f} min")

    print("\nInventory:")
    for cond_name in CONDITIONS:
        for seed in SEEDS:
            p1 = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}.zip")
            p2 = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}_vecnormalize.pkl")
            print(f"  {'✓' if os.path.exists(p1) else '✗'}  {cond_name}_seed{seed}.zip"
                  f"    {'✓' if os.path.exists(p2) else '✗'}  _vecnormalize.pkl")
