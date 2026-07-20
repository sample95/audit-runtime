"""
train_hopper_sac.py — Hopper-v5 SAC + VecNormalize
Conditions: M1, M2, M3 | Seeds: 0-4 | Steps: 1,000,000
Saves: <cond>_seed<n>.zip + <cond>_seed<n>_vecnormalize.pkl
"""

import os
import time
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

torch.set_num_threads(os.cpu_count())

OUTPUT_DIR  = "./results/mujoco/hopper_sac"
TOTAL_STEPS = 1_000_000
SEEDS       = [0, 1, 2, 3, 4]

CONDITIONS = {
    "M1": {"time_limit": 1000, "reset_noise_scale": 0.005},
    "M2": {"time_limit": 250,  "reset_noise_scale": 0.005},
    "M3": {"time_limit": 1000, "reset_noise_scale": 0.02},
}

SAC_KWARGS = dict(
    learning_rate   = 3e-4,
    buffer_size     = 1_000_000,
    learning_starts = 10_000,
    batch_size      = 256,
    tau             = 0.005,
    gamma           = 0.99,
    train_freq      = 1,
    gradient_steps  = 1,
    ent_coef        = "auto",
    policy_kwargs   = dict(
        net_arch        = [256, 256],
        activation_fn   = torch.nn.ReLU,
        optimizer_class = torch.optim.Adam,
    ),
    verbose = 1,
    device  = "cpu",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for cond_name, cond_cfg in CONDITIONS.items():
    for seed in SEEDS:
        save_path    = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}.zip")
        vecnorm_path = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}_vecnormalize.pkl")

        if os.path.exists(save_path) and os.path.exists(vecnorm_path):
            print(f"[skip] {cond_name} seed={seed}")
            continue

        print(f"[train] {cond_name} seed={seed} ...", flush=True)
        t0 = time.time()

        def _make_env(s=seed, cfg=cond_cfg):
            env = gym.make("Hopper-v5",
                           max_episode_steps=cfg["time_limit"],
                           reset_noise_scale=cfg["reset_noise_scale"])
            env.reset(seed=s)
            return env

        vec_env = DummyVecEnv([_make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = SAC("MlpPolicy", vec_env, seed=seed, **SAC_KWARGS)
        model.learn(total_timesteps=TOTAL_STEPS)

        model.save(save_path)
        vec_env.save(vecnorm_path)
        vec_env.close()

        print(f"[done]  {cond_name} seed={seed} ({(time.time()-t0)/60:.1f} min)")

print("\nInventory:")
for cond_name in CONDITIONS:
    for seed in SEEDS:
        p1 = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}.zip")
        p2 = os.path.join(OUTPUT_DIR, f"{cond_name}_seed{seed}_vecnormalize.pkl")
        print(f"  {'✓' if os.path.exists(p1) else '✗'}  {cond_name}_seed{seed}.zip"
              f"    {'✓' if os.path.exists(p2) else '✗'}  _vecnormalize.pkl")
