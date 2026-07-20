"""
pre_deployment.py — Pre-deployment training quality diagnostic (NAS)
Methodology: v12.8 | VecNormalize-safe | Strict train-curve maturity | Run before deployment.py to verify training quality

Usage:
  python pre_deployment.py --domain hopper
  python pre_deployment.py --domain hover
  python pre_deployment.py --domain grid2op
  python pre_deployment.py --domain hopper --save_actions

Outputs:
  pre_deployment_{domain}/
    {cond}_seed{seed}_observations.csv
    {cond}_seed{seed}_rewards.csv
    {cond}_seed{seed}_performance.csv
    {cond}_seed{seed}_events.csv
    {cond}_seed{seed}_actions.csv          (if --save_actions)
    {cond}_seed{seed}_safety_margins.csv
    {cond}_seed{seed}_summary.json
    pre_deployment_{domain}_report.json    (domain-level pass/fail)
  pre_deployment_{domain}.png              (diagnostic figure)
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--domain",       required=True)
parser.add_argument("--configs",      default="configs")
parser.add_argument("--n_seeds",      type=int, default=5)
parser.add_argument("--n_episodes",   type=int, default=20)
parser.add_argument("--save_actions", action="store_true")
args = parser.parse_args()

domain_cfg_path = os.path.join(args.configs, f"{args.domain}.json")
if not os.path.exists(domain_cfg_path):
    print(f"ERROR: config not found: {domain_cfg_path}")
    sys.exit(1)

with open(domain_cfg_path) as f:
    dcfg = json.load(f)

ENV_ID      = dcfg["env_id"]
RESULTS_DIR = dcfg["results_dir"]
CONDITIONS  = dcfg["conditions"]
ALL_SEEDS   = dcfg["seeds"]
ALGO        = dcfg.get("algo", "PPO")
CKPT_SUFFIX = dcfg.get("ckpt_suffix", "")
CHECK_SEEDS = ALL_SEEDS[:args.n_seeds]
N_EPISODES  = args.n_episodes
WRAPPER     = dcfg.get("perturbation_wrapper", dcfg.get("drift_wrapper"))
# "baseline"-era configs may still use the pre-rename wrapper names; normalize
# to canonical so the comparisons below only need to check one spelling.
WRAPPER = {"GravityDrift": "GravityPerturbation",
           "ObsNoiseDrift": "ObsNoisePerturbation",
           "ChronicDrift": "ChronicPerturbation"}.get(WRAPPER, WRAPPER)

# ── Policy preprocessing / VecNormalize contract ──────────────────────────────
# If a policy was trained with SB3 VecNormalize, evaluation must load the saved
# normalization statistics and freeze them. Diagnostics still export raw env
# observations and raw rewards so safety margins/drift metrics keep physical units.
VECNORMALIZE_REQUIRED    = bool(dcfg.get("vecnormalize_required", dcfg.get("vec_normalize", False)))
VECNORMALIZE_SUFFIX      = dcfg.get("vecnormalize_suffix", "_vecnormalize.pkl")
VECNORMALIZE_TRAINING    = bool(dcfg.get("vecnormalize_training", False))
VECNORMALIZE_NORM_REWARD = bool(dcfg.get("vecnormalize_norm_reward", False))
AUDIT_OBS_SPACE          = dcfg.get("audit_observation_space", "raw_env_obs")
AUDIT_REWARD_SPACE       = dcfg.get("audit_reward_space", "raw_env_reward")

EXPORT_DIR = os.path.join(RESULTS_DIR, f"pre_deployment_{args.domain}")
os.makedirs(EXPORT_DIR, exist_ok=True)

# ── Pass/fail thresholds (domain-aware) ───────────────────────────────────────
PASS_THRESHOLDS = {
    "GravityPerturbation": {
        "min_mean_return":  800.0,
        "max_fail_rate":    0.3,
        "min_mean_ep_len":  200.0,
        "max_return_cv":    1.5,
        "min_return_trend": -0.5,
    },
    "ObsNoisePerturbation": {
        "min_mean_return":  0.5,
        "max_fail_rate":    0.7,
        "min_mean_ep_len":  50.0,
        "max_return_cv":    2.0,
        "min_return_trend": -1.0,
    },
    "ChronicPerturbation": {
        "min_mean_return":  -500.0,
        "max_fail_rate":    0.95,
        "min_mean_ep_len":  50.0,
        "max_return_cv":    5.0,
        "min_return_trend": -2.0,
    },
}
THRESH = PASS_THRESHOLDS.get(WRAPPER, PASS_THRESHOLDS["GravityPerturbation"])

# Training-curve and reload-consistency thresholds. These are diagnostic only:
# they do not change the thesis audit instrument. They prevent a stable but poor
# training plateau, or a broken reload/evaluation wrapper, from being reported as
# "converged" without qualification.
TRAIN_CURVE_THRESH = {
    # Minimum improvement from early training. This guards against a flat bad curve.
    "min_improvement_ratio": float(dcfg.get("min_train_improvement_ratio", 0.25)),

    # Strict plateau/maturity checks. The older script used plateau_cv=0.50 and
    # slope=0.01, which only detected exploding curves. For a pre-deployment
    # quality screen, a curve that is still visibly rising should not be labelled
    # as converged/stable.
    "max_plateau_cv":        float(dcfg.get("max_train_plateau_cv", 0.25)),
    "max_abs_last20p_slope": float(dcfg.get("max_abs_last20p_slope", 1e-4)),
    "max_abs_tail_gain":     float(dcfg.get("max_abs_tail_gain", 0.05)),

    # Reload consistency is not convergence; it detects wrapper/checkpoint/stats
    # mismatches, especially VecNormalize mistakes.
    "min_deploy_train_ratio": float(dcfg.get("min_deploy_train_ratio", 0.25)),
}

# Convergence curves must be comparable across training horizons.  In MuJoCo,
# M2 is trained with a short TimeLimit (250) but the pre-deployment smoke test
# runs on the default/evaluation horizon (1000).  Raw Monitor episode returns
# therefore make M2 look artificially low.  For the convergence plot and
# training-vs-reload consistency check, convert Monitor returns to a per-step
# return rate and then to an evaluation-horizon-equivalent return.
EVAL_EQUIV_HORIZON = int(dcfg.get(
    "pre_deployment_eval_horizon",
    1000 if WRAPPER == "GravityPerturbation" else 0,
))


def training_score_curve(returns, lengths=None):
    """Return the curve used for convergence diagnostics and plotting.

    If episode lengths are available and EVAL_EQUIV_HORIZON > 0, use
    return_per_step * EVAL_EQUIV_HORIZON.  Otherwise fall back to raw episode
    returns.  This keeps M1/M2/M3 comparable when training truncation differs.
    """
    returns = np.asarray(returns, dtype=float)
    if lengths is None or EVAL_EQUIV_HORIZON <= 0:
        return returns, "raw_episode_return"
    lengths = np.asarray(lengths, dtype=float)
    n = min(len(returns), len(lengths))
    if n == 0:
        return returns, "raw_episode_return"
    score = (returns[:n] / np.maximum(lengths[:n], 1.0)) * float(EVAL_EQUIV_HORIZON)
    return score, f"eval{EVAL_EQUIV_HORIZON}_equiv_return"

COND_COLORS = ["#2563EB", "#F59E0B", "#16A34A", "#9333EA", "#0891B2", "#A855F7"]
CMAP_COND   = {c: COND_COLORS[i % len(COND_COLORS)] for i, c in enumerate(CONDITIONS)}
PASS_COLOR  = "#22C55E"
FAIL_COLOR  = "#EF4444"
WARN_COLOR  = "#FBBF24"
STYLE = {
    "bg": "#0F0F0F", "panel": "#1A1A1A", "grid": "#2A2A2A",
    "text": "#E8E8E8", "subtext": "#888888", "accent": "#3B82F6",
}
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],  "axes.facecolor":  STYLE["panel"],
    "axes.edgecolor":   STYLE["grid"], "axes.labelcolor": STYLE["text"],
    "xtick.color":   STYLE["subtext"], "ytick.color":   STYLE["subtext"],
    "text.color":    STYLE["text"],    "grid.color":    STYLE["grid"],
    "grid.linewidth": 0.5, "font.family": "monospace", "font.size": 9,
})

def load_model(ckpt_path, env=None):
    if ALGO == "SAC":
        from stable_baselines3 import SAC
        return SAC.load(ckpt_path, env=env, device="cpu")
    from stable_baselines3 import PPO
    return PPO.load(
        ckpt_path,
        env=env,
        device="cpu",
        custom_objects={"lr_schedule": lambda _: 0.0,
                        "clip_range":  lambda _: 0.0},
    )


def checkpoint_base(condition, seed):
    return os.path.join(RESULTS_DIR, f"{condition}{CKPT_SUFFIX}_seed{seed}")


def checkpoint_path(condition, seed):
    return f"{checkpoint_base(condition, seed)}.zip"


def vecnormalize_path(condition, seed):
    return f"{checkpoint_base(condition, seed)}{VECNORMALIZE_SUFFIX}"

def make_env(condition):
    if WRAPPER == "GravityPerturbation":
        import gymnasium as gym
        return gym.make(ENV_ID), None
    elif WRAPPER == "ObsNoisePerturbation":
        import gymnasium as gym
        import PyFlyt.gym_envs  # noqa: F401
        env_kwargs     = dcfg.get("env_kwargs", {})
        cond_durations = dcfg.get("condition_durations", {})
        return gym.make(ENV_ID,
                        max_duration_seconds=cond_durations.get(condition, 10.0),
                        **env_kwargs), None
    elif WRAPPER == "ChronicPerturbation":
        import grid2op
        from grid2op.Action import PlayableAction
        from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace
        from lightsim2grid import LightSimBackend
        from gymnasium.wrappers import TimeLimit
        action_attrs = dcfg.get("action_attrs", ["redispatch", "curtail"])
        # Must match the attr_to_keep used during training (train_grid2op.py OBS_ATTRS).
        # Hardcoded fallback covers the case14-sandbox policy; always set obs_attrs in
        # the domain config to make this explicit and avoid silent obs-dim mismatches.
        obs_attrs = dcfg.get("obs_attrs", [
            "rho", "p_or", "p_ex", "load_p", "gen_p",
            "actual_dispatch", "target_dispatch",
            "curtailment", "curtailment_limit", "gen_p_before_curtail",
        ])
        h_train      = dcfg.get("condition_h", {}).get(condition, 1000)
        # action_class=PlayableAction is required for curtailment + redispatch to be
        # legal actions; without it grid2op raises IllegalAction on every curtail step.
        g2op_env = grid2op.make(ENV_ID, backend=LightSimBackend(),
                                action_class=PlayableAction)
        gym_env  = GymEnv(g2op_env)
        gym_env.observation_space = BoxGymObsSpace(
            g2op_env.observation_space, attr_to_keep=obs_attrs)
        gym_env.action_space      = BoxGymActSpace(
            gym_env.init_env.action_space, attr_to_keep=action_attrs)
        return TimeLimit(gym_env, max_episode_steps=h_train), g2op_env
    raise ValueError(f"Unknown perturbation_wrapper: {WRAPPER}")

def extract_safety_margins(obs_arr, condition):
    margins = {}
    if WRAPPER == "GravityPerturbation":
        if obs_arr.shape[1] >= 2:
            margins["Torso height (z)"] = obs_arr[:, 0]
            margins["Torso angle"]       = obs_arr[:, 1]
        if obs_arr.shape[1] >= 5:
            margins["Joint angles |mean|"] = np.abs(obs_arr[:, 2:5]).mean(axis=1)
        if obs_arr.shape[1] >= 8:
            margins["Joint vels |mean|"]   = np.abs(obs_arr[:, 5:8]).mean(axis=1)
    elif WRAPPER == "ObsNoisePerturbation":
        if obs_arr.shape[1] >= 16:
            margins["Quaternion w"]  = obs_arr[:, 6]
            margins["|Roll|"]        = np.abs(obs_arr[:, 13])
            margins["|Pitch|"]       = np.abs(obs_arr[:, 14])
        if obs_arr.shape[1] >= 17:
            margins["Height proxy"]  = obs_arr[:, 12]
    elif WRAPPER == "ChronicPerturbation":
        n = min(20, obs_arr.shape[1])
        margins["Feature |mean| (0-19)"] = np.abs(obs_arr[:, :n]).mean(axis=1)
        margins["High-load proxy >0.9"]  = (
            np.abs(obs_arr[:, :n]) > 0.9).any(axis=1).astype(float)
    return margins

def load_training_curves(cond, seeds):
    """Load training episode-return curves from SB3 Monitor CSV files only.

    Monitor CSV is the reliable artifact for training convergence because it
    stores episode return/length over training. A saved checkpoint's
    ep_info_buffer is not used as a fallback: after loading, it may be empty or
    only contain a recent/incomplete buffer, so using it can create misleading
    convergence claims.
    """
    curves = {}
    for seed in seeds:
        monitor_path = os.path.join(RESULTS_DIR, f"{cond}_seed{seed}.monitor.csv")
        if os.path.exists(monitor_path):
            try:
                import pandas as pd
                df = pd.read_csv(monitor_path, skiprows=1)
                if {"r", "l"}.issubset(df.columns) and len(df) > 0:
                    df["cumsteps"] = df["l"].cumsum()
                    curves[seed] = {
                        "steps":   df["cumsteps"].values,
                        "returns": df["r"].values,
                        "lengths":  df["l"].values,
                        "source":  "monitor",
                        "monitor_path": monitor_path,
                    }
                    continue
                curves[seed] = {
                    "steps": np.array([]), "returns": np.array([]),
                    "source": "invalid_monitor", "monitor_path": monitor_path,
                    "error": "Monitor CSV missing required r/l columns or is empty.",
                }
            except Exception as e:
                curves[seed] = {
                    "steps": np.array([]), "returns": np.array([]),
                    "source": "monitor_read_error", "monitor_path": monitor_path,
                    "error": str(e),
                }
        else:
            curves[seed] = {
                "steps": np.array([]), "returns": np.array([]),
                "source": "missing_monitor", "monitor_path": monitor_path,
                "error": "No Monitor CSV found; convergence is unavailable.",
            }
    return curves

def analyze_convergence(returns, lengths=None):
    """Assess training-curve maturity, not just improvement.

    The assessed curve is horizon-normalized when episode lengths are available:
    return_per_step × EVAL_EQUIV_HORIZON.  This prevents the short-horizon M2
    condition from looking artificially worse than M1/M3 only because its
    training episodes are capped at 250 steps.

    A curve is labelled OK only when it:
      1) improved from the beginning,
      2) has acceptable dispersion in the tail,
      3) has near-zero normalized tail slope,
      4) shows little change between the penultimate and final tail blocks.

    This intentionally separates "policy is usable when reloaded" from
    "training curve appears mature/plateaued".
    """
    raw_returns = np.asarray(returns, dtype=float)
    score, score_name = training_score_curve(raw_returns, lengths)
    score = np.asarray(score, dtype=float)
    n = len(score)
    if n < 10:
        return {
            "available": False,
            "score_name": score_name,
            "training_curve_stable": None,
            "training_improved": None,
            "training_curve_ok": None,
            "converged": None,  # backward-compatible alias; do not interpret alone
            "improvement_ratio": None,
            "plateau_cv": None,
            "final_mean": None,
            "initial_mean": None,
            "last20p_slope": None,
            "tail_gain_ratio": None,
            "final_mean_raw_episode_return": None,
            "initial_mean_raw_episode_return": None,
            "final_mean_return_per_step": None,
            "initial_mean_return_per_step": None,
            "reason": "Need at least 10 completed training episodes in Monitor CSV.",
        }

    p20 = max(2, n // 5)
    initial_mean = float(score[:p20].mean())
    final_mean   = float(score[-p20:].mean())
    impr_ratio   = float((final_mean - initial_mean) / (abs(initial_mean) + 1e-8))

    # Tail stability over the final 20%.
    last_chunk   = score[-p20:]
    plateau_cv   = float(last_chunk.std() / (abs(last_chunk.mean()) + 1e-8))
    x_fit        = np.arange(len(last_chunk), dtype=float)
    raw_slope    = float(np.polyfit(x_fit, last_chunk, 1)[0])
    norm_slope   = float(raw_slope / (abs(final_mean) + 1e-8))

    # Tail maturity: compare the final 10% with the preceding 10%.
    p10 = max(2, n // 10)
    prev_tail = score[-2*p10:-p10]
    final_tail = score[-p10:]
    if len(prev_tail) and len(final_tail):
        prev_tail_mean = float(prev_tail.mean())
        final_tail_mean = float(final_tail.mean())
        tail_gain_ratio = float((final_tail_mean - prev_tail_mean) / (abs(prev_tail_mean) + 1e-8))
    else:
        prev_tail_mean = None
        final_tail_mean = None
        tail_gain_ratio = None

    raw_p20 = max(2, len(raw_returns) // 5)
    raw_initial = float(raw_returns[:raw_p20].mean()) if len(raw_returns) else None
    raw_final   = float(raw_returns[-raw_p20:].mean()) if len(raw_returns) else None
    if lengths is not None and len(lengths) >= len(raw_returns) and len(raw_returns):
        lengths_arr = np.asarray(lengths, dtype=float)
        rate = raw_returns / np.maximum(lengths_arr[:len(raw_returns)], 1.0)
        rate_initial = float(rate[:raw_p20].mean())
        rate_final   = float(rate[-raw_p20:].mean())
    else:
        rate_initial = None
        rate_final = None

    training_curve_stable = bool(
        plateau_cv <= TRAIN_CURVE_THRESH["max_plateau_cv"]
        and abs(norm_slope) <= TRAIN_CURVE_THRESH["max_abs_last20p_slope"]
        and tail_gain_ratio is not None
        and abs(tail_gain_ratio) <= TRAIN_CURVE_THRESH["max_abs_tail_gain"]
    )
    training_improved = bool(
        impr_ratio >= TRAIN_CURVE_THRESH["min_improvement_ratio"]
    )
    training_curve_ok = bool(training_curve_stable and training_improved)

    return {
        "available": True,
        "score_name": score_name,
        "eval_equiv_horizon": EVAL_EQUIV_HORIZON if EVAL_EQUIV_HORIZON > 0 else None,
        "training_curve_stable": training_curve_stable,
        "training_improved": training_improved,
        "training_curve_ok": training_curve_ok,
        "converged": training_curve_ok,  # backward-compatible alias
        "improvement_ratio": impr_ratio,
        "plateau_cv": plateau_cv,
        "final_mean": final_mean,
        "initial_mean": initial_mean,
        "last20p_slope": norm_slope,
        "tail_gain_ratio": tail_gain_ratio,
        "prev_tail_mean": prev_tail_mean,
        "final_tail_mean": final_tail_mean,
        "final_mean_raw_episode_return": raw_final,
        "initial_mean_raw_episode_return": raw_initial,
        "final_mean_return_per_step": rate_final,
        "initial_mean_return_per_step": rate_initial,
        "thresholds": TRAIN_CURVE_THRESH,
    }

def evaluate_pass_fail(d, training_curves):
    returns    = d["ep_returns"]
    lengths    = d["ep_lengths"]
    terminated = d["terminated"]
    seed       = d["seed"]

    mean_ret  = float(returns.mean())
    std_ret   = float(returns.std())
    cv_ret    = std_ret / (abs(mean_ret) + 1e-8)
    fail_rate = float(terminated.mean())
    mean_len  = float(lengths.mean())

    # Normalise to eval-equiv horizon when conditions have different truncations
    # (e.g. hover Q1=400 steps vs Q2=100 steps). CV is scale-invariant so unchanged.
    if EVAL_EQUIV_HORIZON > 0 and mean_len > 0:
        mean_ret_norm = (mean_ret / mean_len) * EVAL_EQUIV_HORIZON
    else:
        mean_ret_norm = mean_ret

    curve_record = training_curves.get(seed)
    conv_data = curve_record.get("convergence") if curve_record else None
    conv_source = curve_record.get("source") if curve_record else "missing_monitor"

    # Training trend: second half vs first half of Monitor curve.
    trend_ok  = True
    trend_val = None
    if curve_record and len(curve_record.get("returns", [])) >= 10:
        tr, _score_name = training_score_curve(
            curve_record["returns"], curve_record.get("lengths")
        )
        mid  = len(tr) // 2
        fh   = tr[:mid].mean()
        sh   = tr[mid:].mean()
        trend_val = float((sh - fh) / (abs(fh) + 1e-8))
        trend_ok  = trend_val >= THRESH["min_return_trend"]

    # Training-curve quality is separate from deployment reload quality.
    training_curve_ok = True
    training_curve_available = bool(conv_data and conv_data.get("available"))
    if conv_data:
        training_curve_ok = bool(conv_data.get("training_curve_ok")) if training_curve_available else False
    else:
        training_curve_ok = False

    # Detect wrapper/reload mismatch: a policy that trained to high final return but
    # collapses immediately when reloaded is usually being evaluated with the wrong
    # preprocessing/wrapper/order, especially with VecNormalize policies.
    deploy_train_ratio = None
    reload_consistency_ok = True
    if training_curve_available and conv_data.get("final_mean") is not None:
        # Use normalised return so Q2 (100-step) is not compared raw against
        # a training score that was already normalised to EVAL_EQUIV_HORIZON scale.
        deploy_train_ratio = float(mean_ret_norm / (abs(conv_data["final_mean"]) + 1e-8))
        reload_consistency_ok = bool(
            deploy_train_ratio >= TRAIN_CURVE_THRESH["min_deploy_train_ratio"]
        )

    checks = {
        "mean_return_ok": mean_ret_norm >= THRESH["min_mean_return"],
        "fail_rate_ok":   fail_rate <= THRESH["max_fail_rate"],
        "ep_length_ok":   mean_len  >= THRESH["min_mean_ep_len"],
        "stability_ok":   cv_ret    <= THRESH["max_return_cv"],
        "trend_ok":       trend_ok,
        "training_curve_ok": training_curve_ok,
        "reload_consistency_ok": reload_consistency_ok,
    }
    # Backward-compatible alias for older plotting/JSON readers.
    checks["convergence_ok"] = training_curve_ok

    passed = all(checks[k] for k in [
        "mean_return_ok", "fail_rate_ok", "ep_length_ok", "stability_ok",
        "trend_ok", "training_curve_ok", "reload_consistency_ok",
    ])

    # Diagnostics: plain-language failure explanations
    diags = []
    if not checks["mean_return_ok"]:
        diags.append(
            f"LOW EVAL RETURN {mean_ret_norm:.1f} < {THRESH['min_mean_return']:.1f}: "
            "the reloaded frozen policy performs poorly in the smoke test."
        )
    if not checks["fail_rate_ok"]:
        diags.append(
            f"HIGH EVAL FAILURE RATE {fail_rate:.2f} > {THRESH['max_fail_rate']:.2f}: "
            "policy terminates too often in smoke-test rollouts."
        )
    if not checks["ep_length_ok"]:
        diags.append(
            f"SHORT EVAL EPISODES mean_len={mean_len:.1f} < {THRESH['min_mean_ep_len']:.1f}: "
            "policy fails fast, or the reload/evaluation wrapper is wrong."
        )
    if not checks["stability_ok"]:
        diags.append(
            f"HIGH EVAL RETURN VARIANCE CV={cv_ret:.2f} > {THRESH['max_return_cv']:.2f}: "
            "unstable smoke-test performance."
        )
    if not checks["trend_ok"]:
        diags.append(
            f"DECLINING TRAINING TREND {trend_val:.3f} < {THRESH['min_return_trend']:.2f}: "
            "training returns decreased from first half to second half."
        )
    if not training_curve_available:
        reason = conv_data.get("reason") if conv_data else "No convergence record."
        diags.append(
            f"NO VALID TRAINING CURVE: source={conv_source}. {reason} "
            "Use Monitor CSV for convergence; checkpoint ep_info_buffer is not trusted."
        )
    elif not checks["training_curve_ok"]:
        diags.append(
            f"TRAINING CURVE NOT MATURE: stable={conv_data['training_curve_stable']} "
            f"improved={conv_data['training_improved']} "
            f"impr={conv_data['improvement_ratio']:.2f}, "
            f"plateau_cv={conv_data['plateau_cv']:.3f}, "
            f"last20p_slope={conv_data['last20p_slope']:.5f}, "
            f"tail_gain={conv_data.get('tail_gain_ratio'):.3f}. "
            "Policy may be usable, but the training curve is still rising/noisy rather than clearly plateaued."
        )
    if not checks["reload_consistency_ok"]:
        diags.append(
            f"TRAIN/EVAL MISMATCH: smoke-test return is only "
            f"{deploy_train_ratio:.3f} × final training return. "
            "Likely wrong VecNormalize stats, checkpoint/stats pairing, or wrapper order."
        )
    if not diags and not passed:
        diags.append("Unknown failure — inspect raw data.")

    return {
        "passed":      passed,
        "checks":      checks,
        "diagnostics": diags,
        "metrics": {
            "mean_return": mean_ret,
            "mean_return_norm": mean_ret_norm,
            "std_return":  std_ret,
            "cv_return":   cv_ret,
            "fail_rate":   fail_rate,
            "mean_ep_len": mean_len,
            "trend_val":   trend_val,
            "convergence": conv_data,
            "convergence_source": conv_source,
            "deploy_train_ratio": deploy_train_ratio,
            "reload_consistency_ok": reload_consistency_ok,
        },
    }

def _collect_rollouts_raw_env(condition, seed, ckpt_path):
    model  = load_model(ckpt_path)
    env, _ = make_env(condition)

    obs_list, act_list, rew_list = [], [], []
    ep_returns, ep_lengths, terminated_flags, reset_steps = [], [], [], []
    global_t = 0

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        ep_ret, ep_len, ep_terminated = 0.0, 0, False
        reset_steps.append(global_t)
        done = False
        while not done:
            # Raw-env branch: obs is already physical/raw observation.
            obs_list.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            if args.save_actions:
                act_list.append(np.atleast_1d(action).copy())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            rew_list.append(float(reward))
            ep_ret  += reward
            ep_len  += 1
            global_t += 1
            done = terminated or truncated
            if terminated:
                ep_terminated = True
            obs = next_obs
        ep_returns.append(float(ep_ret))
        ep_lengths.append(int(ep_len))
        terminated_flags.append(ep_terminated)

    env.close()
    return {
        "obs":         np.array(obs_list,  dtype=np.float32),
        "rewards":     np.array(rew_list,  dtype=np.float32),
        "ep_returns":  np.array(ep_returns),
        "ep_lengths":  np.array(ep_lengths),
        "terminated":  np.array(terminated_flags, dtype=bool),
        "reset_steps": reset_steps,
        "condition":   condition,
        "seed":        seed,
        "vecnormalize_used": False,
        "obs_export_space": "raw_env_obs",
        "reward_export_space": "raw_env_reward",
    }, act_list


def _collect_rollouts_vecnormalize(condition, seed, ckpt_path, vec_path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _make_vec_env():
        env, _ = make_env(condition)
        return env

    vec_env = DummyVecEnv([_make_vec_env])
    vec_env = VecNormalize.load(vec_path, vec_env)

    # Freeze training statistics for evaluation. Keep rewards raw for diagnostics.
    vec_env.training    = VECNORMALIZE_TRAINING
    vec_env.norm_reward = VECNORMALIZE_NORM_REWARD

    model = load_model(ckpt_path, env=vec_env)

    obs_list, act_list, rew_list = [], [], []
    ep_returns, ep_lengths, terminated_flags, reset_steps = [], [], [], []
    global_t = 0

    for ep in range(N_EPISODES):
        # SB3 VecEnv reset does not return an info dict. Seed before reset for reproducibility.
        try:
            vec_env.seed(seed * 1000 + ep)
        except Exception:
            pass
        obs_norm = vec_env.reset()
        ep_ret, ep_len, ep_terminated = 0.0, 0, False
        reset_steps.append(global_t)
        done = False

        while not done:
            # The policy sees normalized observations, but exports use raw env observations.
            raw_obs = vec_env.get_original_obs()[0].copy()
            obs_list.append(raw_obs)

            action, _ = model.predict(obs_norm, deterministic=True)
            if args.save_actions:
                act_arr = np.asarray(action)
                act_list.append(act_arr[0].copy() if act_arr.ndim > 1 else act_arr.copy())

            obs_norm, reward_norm, dones, infos = vec_env.step(action)
            info = infos[0] if infos else {}
            done = bool(dones[0])

            # With norm_reward=False this equals the env reward; get_original_reward is kept
            # for explicitness and for future configs that may train with reward norm enabled.
            try:
                raw_reward = float(vec_env.get_original_reward()[0])
            except Exception:
                raw_reward = float(np.asarray(reward_norm).reshape(-1)[0])
            rew_list.append(raw_reward)

            ep_ret  += raw_reward
            ep_len  += 1
            global_t += 1

            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = done and not truncated
            if terminated:
                ep_terminated = True

        ep_returns.append(float(ep_ret))
        ep_lengths.append(int(ep_len))
        terminated_flags.append(ep_terminated)

    vec_env.close()
    return {
        "obs":         np.array(obs_list,  dtype=np.float32),
        "rewards":     np.array(rew_list,  dtype=np.float32),
        "ep_returns":  np.array(ep_returns),
        "ep_lengths":  np.array(ep_lengths),
        "terminated":  np.array(terminated_flags, dtype=bool),
        "reset_steps": reset_steps,
        "condition":   condition,
        "seed":        seed,
        "vecnormalize_used": True,
        "vecnormalize_path": vec_path,
        "obs_export_space": AUDIT_OBS_SPACE,
        "reward_export_space": AUDIT_REWARD_SPACE,
    }, act_list


def collect_rollouts(condition, seed):
    ckpt_path = checkpoint_path(condition, seed)
    if not os.path.exists(ckpt_path):
        return None

    if VECNORMALIZE_REQUIRED:
        vec_path = vecnormalize_path(condition, seed)
        if not os.path.exists(vec_path):
            print(f"  [skip] VecNormalize stats not found: {vec_path}")
            return None
        result, act_list = _collect_rollouts_vecnormalize(condition, seed, ckpt_path, vec_path)
    else:
        result, act_list = _collect_rollouts_raw_env(condition, seed, ckpt_path)

    if args.save_actions and act_list:
        result["actions"] = np.array(act_list, dtype=np.float32)
    return result

def export_data(d, pf_result):
    cond   = d["condition"]
    seed   = d["seed"]
    prefix = os.path.join(EXPORT_DIR, f"{cond}_seed{seed}")

    hdr = ",".join([f"obs_{j}" for j in range(d["obs"].shape[1])])
    np.savetxt(f"{prefix}_observations.csv", d["obs"],
               delimiter=",", header=hdr, comments="")

    np.savetxt(f"{prefix}_rewards.csv", d["rewards"].reshape(-1, 1),
               delimiter=",", header="reward", comments="")

    with open(f"{prefix}_performance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "length"])
        for i, (r, l) in enumerate(zip(d["ep_returns"], d["ep_lengths"])):
            w.writerow([i, round(float(r), 4), int(l)])

    with open(f"{prefix}_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "terminated", "length", "reset_step"])
        for i, (term, length, rst) in enumerate(
                zip(d["terminated"], d["ep_lengths"], d["reset_steps"])):
            w.writerow([i, int(term), int(length), int(rst)])

    if "actions" in d:
        hdr_a = ",".join([f"act_{j}" for j in range(d["actions"].shape[1])])
        np.savetxt(f"{prefix}_actions.csv", d["actions"],
                   delimiter=",", header=hdr_a, comments="")

    margins = extract_safety_margins(d["obs"], cond)
    if margins:
        rows = list(zip(*margins.values()))
        with open(f"{prefix}_safety_margins.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(margins.keys()))
            w.writerows([[round(float(v), 6) for v in row] for row in rows])

    summary = {
        "domain": args.domain, "condition": cond, "seed": seed, "algo": ALGO,
        "n_episodes": N_EPISODES, "pass_fail": pf_result,
        "obs_shape": list(d["obs"].shape), "has_actions": "actions" in d,
        "safety_margin_keys": list(margins.keys()),
        "vecnormalize_used": bool(d.get("vecnormalize_used", False)),
        "vecnormalize_path": d.get("vecnormalize_path"),
        "obs_export_space": d.get("obs_export_space", "raw_env_obs"),
        "reward_export_space": d.get("reward_export_space", "raw_env_reward"),
    }
    with open(f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COLLECTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"PRE-DEPLOYMENT DIAGNOSTIC — {args.domain.upper()} ({ALGO})")
print(f"Conditions: {CONDITIONS} | Seeds: {CHECK_SEEDS} | Episodes: {N_EPISODES}")
print(f"VecNormalize: {'ON' if VECNORMALIZE_REQUIRED else 'OFF'} | obs export={AUDIT_OBS_SPACE} | reward export={AUDIT_REWARD_SPACE}")
print("=" * 60)

all_data   = {c: [] for c in CONDITIONS}
all_pf     = {c: [] for c in CONDITIONS}
all_curves = {}

for cond in CONDITIONS:
    all_curves[cond] = load_training_curves(cond, CHECK_SEEDS)
    for seed_cv, cv_data in all_curves[cond].items():
        cv_data["convergence"] = analyze_convergence(
            cv_data.get("returns", []), cv_data.get("lengths")
        )
        cv_data["convergence"]["source"] = cv_data.get("source")
        cv_data["convergence"]["monitor_path"] = cv_data.get("monitor_path")
        if cv_data.get("error"):
            cv_data["convergence"]["reason"] = cv_data.get("error")
    for seed in CHECK_SEEDS:
        print(f"  collecting {cond} seed={seed} ...", flush=True)
        d = collect_rollouts(cond, seed)
        if d is None:
            print(f"  [skip] checkpoint not found")
            continue
        pf = evaluate_pass_fail(d, all_curves[cond])
        export_data(d, pf)
        all_data[cond].append(d)
        all_pf[cond].append(pf)
        verdict = "PASS" if pf["passed"] else "FAIL"
        print(f"  [{verdict}] return={pf['metrics']['mean_return']:.1f}  "
              f"fail_rate={pf['metrics']['fail_rate']:.2f}  "
              f"len={pf['metrics']['mean_ep_len']:.1f}")
        for diag in pf["diagnostics"]:
            print(f"    ⚠  {diag}")

has_data = any(len(v) > 0 for v in all_data.values())
if not has_data:
    print("ERROR: No checkpoints found. Aborting.")
    sys.exit(1)

domain_report = {
    "domain": args.domain, "algo": ALGO,
    "n_episodes": N_EPISODES, "seeds_checked": CHECK_SEEDS,
    "thresholds": THRESH,
    "training_curve_thresholds": TRAIN_CURVE_THRESH,
    "vecnormalize": {
        "required": VECNORMALIZE_REQUIRED,
        "suffix": VECNORMALIZE_SUFFIX,
        "training": VECNORMALIZE_TRAINING,
        "norm_reward": VECNORMALIZE_NORM_REWARD,
        "audit_observation_space": AUDIT_OBS_SPACE,
        "audit_reward_space": AUDIT_REWARD_SPACE,
    },
    "conditions": {
        cond: {
            "overall_pass": all(p["passed"] for p in all_pf[cond]) if all_pf[cond] else None,
            "seeds": {all_data[cond][i]["seed"]: all_pf[cond][i]
                      for i in range(len(all_pf[cond]))}
        }
        for cond in CONDITIONS if all_pf[cond]
    }
}
report_path = os.path.join(EXPORT_DIR, f"pre_deployment_{args.domain}_report.json")
with open(report_path, "w") as f:
    json.dump(domain_report, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════
N_CONDS    = len(CONDITIONS)
N_ROWS     = 6 if args.save_actions else 5
row_ratios = ([1.4, 2.5, 2.0, 2.0, 1.5, 2.0] if args.save_actions
              else [1.4, 2.5, 2.0, 2.0, 2.0])

fig = plt.figure(figsize=(22, 30 if args.save_actions else 26),
                 facecolor=STYLE["bg"])
fig.suptitle(
    f"Pre-Deployment Diagnostic  ·  {args.domain.upper()}  ({ALGO})  "
    f"·  {N_EPISODES} ep × {len(CHECK_SEEDS)} seeds / condition",
    fontsize=13, color=STYLE["text"], y=0.998, fontweight="bold"
)

gs = gridspec.GridSpec(N_ROWS, 1, figure=fig, hspace=0.62,
                       height_ratios=row_ratios,
                       top=0.984, bottom=0.035, left=0.06, right=0.97)

def make_row(parent_gs, n_cols, wspace=0.35):
    return gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=parent_gs, wspace=wspace)

def style_ax(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_facecolor(STYLE["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(STYLE["grid"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=STYLE["grid"], linewidth=0.5, alpha=0.7)
    ax.set_title(title, fontsize=8.5, color=STYLE["text"], pad=4)
    ax.set_xlabel(xlabel, fontsize=7.5, color=STYLE["subtext"])
    ax.set_ylabel(ylabel, fontsize=7.5, color=STYLE["subtext"])

def jitter(n, spread=0.07):
    return np.random.uniform(-spread, spread, n)

CHECK_LABELS = ["return", "fail_rate", "ep_len", "stability", "trend", "train_curve", "reload"]
CHECK_KEYS   = ["mean_return_ok", "fail_rate_ok", "ep_length_ok",
                "stability_ok",   "trend_ok", "training_curve_ok",
                "reload_consistency_ok"]

# ── Panel 0: Pass/Fail dashboard ──────────────────────────────────────────────
gs0 = make_row(gs[0], N_CONDS, wspace=0.25)
for ci, cond in enumerate(CONDITIONS):
    ax = fig.add_subplot(gs0[0, ci])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(STYLE["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(CMAP_COND[cond]); sp.set_linewidth(1.5)
    ax.set_title(cond, fontsize=9, color=CMAP_COND[cond], pad=3, fontweight="bold")
    if not all_pf[cond]:
        ax.text(0.5, 0.5, "NO DATA", ha="center", va="center",
                color=STYLE["subtext"], fontsize=10, fontweight="bold",
                transform=ax.transAxes)
        continue
    cond_pass   = all(p["passed"] for p in all_pf[cond])
    ax.set_facecolor("#0F2A1A" if cond_pass else "#2A0F0F")
    verdict_col = PASS_COLOR if cond_pass else FAIL_COLOR
    ax.text(0.5, 0.90, "✓  PASS" if cond_pass else "✗  FAIL",
            ha="center", va="center", color=verdict_col, fontsize=13,
            fontweight="bold", transform=ax.transAxes)
    n_seeds_shown = len(all_pf[cond])
    cell_h = 0.58 / len(CHECK_KEYS)
    label_x = 0.08
    seed_xs = np.linspace(0.50, 0.90, max(n_seeds_shown, 1))

    # Seed headers stay inside the panel regardless of how many seeds are shown.
    for si, (pf, d) in enumerate(zip(all_pf[cond], all_data[cond])):
        ax.text(seed_xs[si], 0.79, f"s{d['seed']}", ha="center", va="center",
                color=STYLE["subtext"], fontsize=5.8, transform=ax.transAxes)

    for ki, (key, label) in enumerate(zip(CHECK_KEYS, CHECK_LABELS)):
        y = 0.72 - ki * cell_h
        ax.text(label_x, y, label, ha="left", va="center",
                color=STYLE["subtext"], fontsize=6.0, transform=ax.transAxes)
        for si, pf in enumerate(all_pf[cond]):
            ok = pf["checks"][key]
            ax.text(seed_xs[si], y, "●" if ok else "✕", ha="center", va="center",
                    color=PASS_COLOR if ok else FAIL_COLOR,
                    fontsize=9, transform=ax.transAxes)
    mean_ret = np.mean([p["metrics"]["mean_return"] for p in all_pf[cond]])
    ax.text(0.5, 0.05, f"μ return: {mean_ret:.1f}", ha="center", va="center",
            color=STYLE["subtext"], fontsize=6.5, transform=ax.transAxes)
fig.text(0.005, 0.965, "① PASS/FAIL", fontsize=8, color=STYLE["accent"],
         fontweight="bold", rotation=90, va="center")

# ── Panel 1: State/Obs ────────────────────────────────────────────────────────
gs1 = make_row(gs[1], min(N_CONDS, 4))
N_OBS_FEAT = 8
for ci, cond in enumerate(CONDITIONS[:4]):
    if not all_data[cond]: continue
    ax       = fig.add_subplot(gs1[0, ci])
    obs_pool = np.concatenate([d["obs"] for d in all_data[cond]], axis=0)
    n_feat   = min(N_OBS_FEAT, obs_pool.shape[1])
    obs_std  = (obs_pool - obs_pool.mean(axis=0)) / (obs_pool.std(axis=0) + 1e-8)
    color    = CMAP_COND[cond]
    parts = ax.violinplot([obs_std[:, j] for j in range(n_feat)],
                          positions=np.arange(n_feat), widths=0.55,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(0.3); pc.set_edgecolor(color)
    parts["cmedians"].set_color(color); parts["cmedians"].set_linewidth(1.5)
    idx = np.random.choice(len(obs_std), min(300, len(obs_std)), replace=False)
    for j in range(n_feat):
        ax.scatter(np.full(len(idx), j) + jitter(len(idx)),
                   obs_std[idx, j], s=1.5, alpha=0.2, color=color, linewidths=0)
    ax.axhline(0, color=STYLE["grid"], linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(n_feat))
    ax.set_xticklabels([f"f{j}" for j in range(n_feat)], fontsize=7)
    ax.set_ylim(-4.5, 4.5)
    feat_stds  = obs_std.std(axis=0)[:n_feat]
    degenerate = np.where(feat_stds < 0.1)[0]
    for j in degenerate:
        ax.axvspan(j - 0.3, j + 0.3, alpha=0.15, color=FAIL_COLOR, zorder=0)
    if len(degenerate):
        ax.text(0.98, 0.97, f"⚠ {len(degenerate)} const feat",
                ha="right", va="top", color=FAIL_COLOR,
                fontsize=6, transform=ax.transAxes)
    style_ax(ax, title=f"{cond} — Obs (std)", ylabel="σ units" if ci == 0 else "")
fig.text(0.005, 0.80, "② STATE/OBS", fontsize=8, color=STYLE["accent"],
         fontweight="bold", rotation=90, va="center")

# ── Panel 2: Performance + convergence ───────────────────────────────────────
gs2 = make_row(gs[2], 4, wspace=0.4)

ax2a = fig.add_subplot(gs2[0, 0])
for ci, cond in enumerate(CONDITIONS):
    if not all_data[cond]: continue
    raw_returns = np.concatenate([d["ep_returns"] for d in all_data[cond]])
    raw_lengths = np.concatenate([d["ep_lengths"] for d in all_data[cond]]).astype(float)
    # Normalise when conditions have different truncations (e.g. hover Q1 vs Q2).
    if EVAL_EQUIV_HORIZON > 0:
        returns = (raw_returns / np.maximum(raw_lengths, 1.0)) * EVAL_EQUIV_HORIZON
    else:
        returns = raw_returns
    color   = CMAP_COND[cond]
    parts   = ax2a.violinplot([returns], positions=[ci], widths=0.55,
                               showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_alpha(0.4); pc.set_edgecolor(color)
    parts["cmedians"].set_color(color); parts["cmedians"].set_linewidth(2)
    for k in ["cmins", "cmaxes", "cbars"]:
        parts[k].set_color(color); parts[k].set_linewidth(1)
    ax2a.scatter(np.full(len(returns), ci) + jitter(len(returns)),
                 returns, s=18, alpha=0.5, color=color, linewidths=0, zorder=3)
if EVAL_EQUIV_HORIZON <= 0:
    ax2a.axhline(THRESH["min_mean_return"], color=WARN_COLOR, linewidth=1.2,
                 linestyle=":", alpha=0.8, label=f"threshold={THRESH['min_mean_return']:.0f}")
    ax2a.legend(fontsize=6.5, framealpha=0.3, facecolor=STYLE["panel"], labelcolor=STYLE["text"])
ax2a.set_xticks(np.arange(len(CONDITIONS)))
ax2a.set_xticklabels(CONDITIONS, fontsize=8)
ret_ylabel = (f"eval{EVAL_EQUIV_HORIZON}-equiv return" if EVAL_EQUIV_HORIZON > 0 else "Total return")
style_ax(ax2a, title="Episode Return by Condition", ylabel=ret_ylabel)

ax2b = fig.add_subplot(gs2[0, 1])
any_curve = False
for cond in CONDITIONS:
    color = CMAP_COND[cond]
    for seed, data in all_curves[cond].items():
        tr_raw, st = data.get("returns", np.array([])), data.get("steps", np.array([]))
        tr, curve_name = training_score_curve(tr_raw, data.get("lengths"))
        w      = min(20, len(tr))
        sm     = np.convolve(tr, np.ones(w)/w, mode="valid") if w > 1 else tr
        st_sm  = st[w-1:] if w > 1 else st
        ax2b.plot(st[:len(tr)], tr, alpha=0.12, color=color, linewidth=0.7)
        ax2b.plot(st_sm, sm, alpha=0.85, color=color, linewidth=1.5,
                  label=f"{cond}/s{seed}")
        if len(tr) >= w:
            roll_std = np.array([tr[max(0, i-w):i].std() for i in range(w, len(tr)+1)])
            ax2b.fill_between(st_sm, sm - roll_std, sm + roll_std,
                              alpha=0.10, color=color, linewidth=0)
        p20_idx = max(1, len(st) * 4 // 5)
        ax2b.axvspan(st[p20_idx], st[-1], alpha=0.05, color=color, linewidth=0, zorder=0)
        any_curve = True
ax2b.axhline(THRESH["min_mean_return"], color=WARN_COLOR, linewidth=1.0,
             linestyle=":", alpha=0.7, label="eval return threshold")
if not any_curve:
    ax2b.text(0.5, 0.5,
              "No curve data.\nAdd Monitor wrapper\nto training script\nfor full curves.",
              ha="center", va="center", color=STYLE["subtext"],
              transform=ax2b.transAxes, fontsize=8, linespacing=1.7)
else:
    ax2b.legend(fontsize=6.5, framealpha=0.3, facecolor=STYLE["panel"],
                labelcolor=STYLE["text"], ncol=2)
curve_scale_label = f"{EVAL_EQUIV_HORIZON}-step equivalent" if EVAL_EQUIV_HORIZON > 0 else "raw episode return"
style_ax(ax2b, title=f"Training Curve Stability  ({curve_scale_label}; band=±1σ rolling)",
         xlabel="Training steps", ylabel="Eval-horizon-equivalent return")

ax2c = fig.add_subplot(gs2[0, 2])
y_pos, yticks_c, ylabels_c = 0, [], []
for cond in CONDITIONS:
    for i, (pf, d) in enumerate(zip(all_pf[cond], all_data[cond])):
        cv    = pf["metrics"]["cv_return"]
        ok    = cv <= THRESH["max_return_cv"]
        color = PASS_COLOR if ok else FAIL_COLOR
        ax2c.hlines(y_pos, 0, cv, color=color, linewidth=1.8, alpha=0.8)
        ax2c.scatter(cv, y_pos, s=60, color=color,
                     zorder=5, edgecolors=STYLE["bg"], linewidth=0.8)
        ax2c.scatter(0, y_pos, s=12, color=color, alpha=0.4, zorder=4)
        yticks_c.append(y_pos); ylabels_c.append(f"{cond}/s{d['seed']}")
        y_pos += 1
    y_pos += 0.4
ax2c.axvline(THRESH["max_return_cv"], color=WARN_COLOR, linewidth=1.2,
             linestyle="--", alpha=0.8, label=f"threshold={THRESH['max_return_cv']:.1f}")
ax2c.set_yticks(yticks_c); ax2c.set_yticklabels(ylabels_c, fontsize=7.5)
ax2c.legend(fontsize=7, framealpha=0.3, facecolor=STYLE["panel"], labelcolor=STYLE["text"])
style_ax(ax2c, title="Return Stability (CV = std/|mean|)",
         xlabel="Coefficient of variation", grid_axis="x")

ax2d = fig.add_subplot(gs2[0, 3])
y_cv, yticks_cv, ylabels_cv = 0, [], []
for cond in CONDITIONS:
    for seed, cv_data in all_curves[cond].items():
        conv = cv_data.get("convergence") or {}
        impr = conv.get("improvement_ratio")
        plat = conv.get("plateau_cv")
        cvgd = conv.get("converged")
        if impr is None: continue
        ok    = cvgd is not False
        color = PASS_COLOR if ok else FAIL_COLOR
        x0, x1 = min(impr, 0.0), max(impr, 0.0)
        ax2d.hlines(y_cv, x0, x1, color=color, linewidth=1.8, alpha=0.8)
        ax2d.scatter(impr, y_cv, s=55, color=color,
                     zorder=5, edgecolors=STYLE["bg"], linewidth=0.8)
        if plat is not None:
            ax2d.text(impr + 0.03, y_cv, f"σ/μ={plat:.2f}",
                      fontsize=5.5, color=STYLE["subtext"], va="center")
        mk     = "✓" if cvgd else ("○" if cvgd is None else "✕")
        mk_col = PASS_COLOR if cvgd else (STYLE["subtext"] if cvgd is None else FAIL_COLOR)
        ax2d.text(-0.02, y_cv, mk, ha="right", va="center", color=mk_col, fontsize=9)
        yticks_cv.append(y_cv); ylabels_cv.append(f"{cond}/s{seed}")
        y_cv += 1
    y_cv += 0.4
ax2d.axvline(0, color=STYLE["subtext"], linewidth=0.8, linestyle="--", alpha=0.5)
if yticks_cv:
    ax2d.set_yticks(yticks_cv); ax2d.set_yticklabels(ylabels_cv, fontsize=7.5)
else:
    ax2d.text(0.5, 0.5, "No training\ncurve data", ha="center", va="center",
              color=STYLE["subtext"], transform=ax2d.transAxes, fontsize=8)
style_ax(ax2d, title="Training-Curve Quality\n(bar=impr. ratio; ✓/✕=curve ok; σ/μ=plateau CV)",
         xlabel="(final − initial) / |initial|", grid_axis="x")
fig.text(0.005, 0.635, "③ PERFORMANCE", fontsize=8, color=STYLE["accent"],
         fontweight="bold", rotation=90, va="center")

# ── Panel 3: Event-time / failures ────────────────────────────────────────────
gs3 = make_row(gs[3], 3, wspace=0.4)

ax3a = fig.add_subplot(gs3[0, 0])
for cond in CONDITIONS:
    if not all_data[cond]: continue
    lengths  = np.concatenate([d["ep_lengths"] for d in all_data[cond]])
    sorted_l = np.sort(lengths)
    ecdf     = np.arange(1, len(sorted_l)+1) / len(sorted_l)
    ax3a.step(sorted_l, ecdf, where="post",
              color=CMAP_COND[cond], linewidth=1.8, label=cond, alpha=0.85)
ax3a.axhline(0.5, color=STYLE["subtext"], linewidth=0.8, linestyle="--", alpha=0.5)
ax3a.axvline(THRESH["min_mean_ep_len"], color=WARN_COLOR, linewidth=1,
             linestyle=":", alpha=0.8, label=f"threshold={THRESH['min_mean_ep_len']:.0f}")
ax3a.set_ylim(0, 1.05)
ax3a.legend(fontsize=7.5, framealpha=0.3, facecolor=STYLE["panel"], labelcolor=STYLE["text"])
style_ax(ax3a, title="Episode Length ECDF",
         xlabel="Steps per episode", ylabel="Cumulative prob.")

ax3b = fig.add_subplot(gs3[0, 1])
y_pos, yticks_f, ylabels_f = 0, [], []
for cond in CONDITIONS:
    for d in all_data[cond]:
        fail_rate = float(d["terminated"].mean())
        ok        = fail_rate <= THRESH["max_fail_rate"]
        color     = PASS_COLOR if ok else FAIL_COLOR
        ax3b.hlines(y_pos, 0, fail_rate, color=color, linewidth=1.8, alpha=0.8)
        ax3b.scatter(fail_rate, y_pos, s=65, color=color,
                     zorder=5, edgecolors=STYLE["bg"], linewidth=0.8)
        ax3b.scatter(0, y_pos, s=12, color=color, alpha=0.35, zorder=4)
        yticks_f.append(y_pos); ylabels_f.append(f"{cond}/s{d['seed']}")
        y_pos += 1
    y_pos += 0.5
ax3b.axvline(THRESH["max_fail_rate"], color=WARN_COLOR, linewidth=1.2,
             linestyle="--", alpha=0.8, label=f"threshold={THRESH['max_fail_rate']:.2f}")
ax3b.set_xlim(-0.02, 1.05)
ax3b.set_yticks(yticks_f); ax3b.set_yticklabels(ylabels_f, fontsize=7.5)
ax3b.legend(fontsize=7, framealpha=0.3, facecolor=STYLE["panel"], labelcolor=STYLE["text"])
style_ax(ax3b, title="Failure Rate (terminated/total)",
         xlabel="Failure rate", grid_axis="x")

ax3c = fig.add_subplot(gs3[0, 2])
if any(all_data[c] for c in CONDITIONS):
    all_rews = np.concatenate([d["rewards"] for c in CONDITIONS for d in all_data[c]])
    ax3c.hexbin(np.arange(len(all_rews)), all_rews, gridsize=40,
                cmap="Blues", mincnt=1, linewidths=0.2)
    ax3c.axhline(0, color=STYLE["grid"], linewidth=0.8, linestyle="--", alpha=0.7)
    for cond in CONDITIONS:
        if not all_data[cond]: continue
        flat_r = np.concatenate([d["rewards"] for d in all_data[cond]])
        ax3c.axhline(flat_r.mean(), color=CMAP_COND[cond], linewidth=1.3,
                     linestyle="-", alpha=0.8, label=f"{cond} μ")
    ax3c.legend(fontsize=6.5, framealpha=0.3, facecolor=STYLE["panel"],
                labelcolor=STYLE["text"])
style_ax(ax3c, title="Per-step Reward Density",
         xlabel="Step index", ylabel="Reward", grid_axis="")
fig.text(0.005, 0.47, "④ EVENT-TIME", fontsize=8, color=STYLE["accent"],
         fontweight="bold", rotation=90, va="center")

# ── Panel 4: Actions (optional) ───────────────────────────────────────────────
safety_row = 5 if args.save_actions else 4
if args.save_actions:
    gs4 = make_row(gs[4], min(N_CONDS, 4))
    for ci, cond in enumerate(CONDITIONS[:4]):
        rollouts_w = [d for d in all_data[cond] if "actions" in d]
        if not rollouts_w: continue
        ax    = fig.add_subplot(gs4[0, ci])
        acts  = np.concatenate([d["actions"] for d in rollouts_w], axis=0)
        n_act = acts.shape[1]
        color = CMAP_COND[cond]
        parts = ax.violinplot([acts[:, j] for j in range(n_act)],
                              positions=np.arange(n_act), widths=0.55,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.4); pc.set_edgecolor(color)
        parts["cmedians"].set_color(color); parts["cmedians"].set_linewidth(1.5)
        act_stds = acts.std(axis=0)
        for j in np.where(act_stds < 0.05)[0]:
            ax.axvspan(j - 0.3, j + 0.3, alpha=0.15, color=FAIL_COLOR, zorder=0)
        if (act_stds < 0.05).any():
            ax.text(0.98, 0.97, "⚠ const actions", ha="right", va="top",
                    color=FAIL_COLOR, fontsize=6, transform=ax.transAxes)
        ax.set_xticks(np.arange(n_act))
        ax.set_xticklabels([f"a{j}" for j in range(n_act)], fontsize=7)
        style_ax(ax, title=f"{cond} — Actions", ylabel="Value" if ci == 0 else "")
    fig.text(0.005, 0.31, "⑤ ACTIONS", fontsize=8, color=STYLE["accent"],
             fontweight="bold", rotation=90, va="center")

# ── Panel 5: Safety margins ───────────────────────────────────────────────────
gs5 = make_row(gs[safety_row], min(N_CONDS, 4))
for ci, cond in enumerate(CONDITIONS[:4]):
    if not all_data[cond]: continue
    ax      = fig.add_subplot(gs5[0, ci])
    obs_all = np.concatenate([d["obs"] for d in all_data[cond]], axis=0)
    margins = extract_safety_margins(obs_all, cond)
    color   = CMAP_COND[cond]
    if not margins:
        ax.text(0.5, 0.5, "No domain margins\navailable", ha="center", va="center",
                color=STYLE["subtext"], transform=ax.transAxes, fontsize=8)
        style_ax(ax, title=f"{cond} — Safety Margins")
        continue
    labels = list(margins.keys())
    for li, label in enumerate(labels):
        vals = margins[label]
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        iqr  = q75 - q25
        w_lo = max(float(vals.min()), q25 - 1.5 * iqr)
        w_hi = min(float(vals.max()), q75 + 1.5 * iqr)
        ax.barh(li, q75 - q25, left=q25, height=0.4,
                color=color, alpha=0.4, edgecolor=color, linewidth=1)
        ax.vlines(q50, li - 0.2, li + 0.2, color=color, linewidth=2)
        ax.hlines(li, w_lo, w_hi, color=color, linewidth=1, alpha=0.6)
        warn = False
        if "angle"  in label.lower() and abs(q50) > 1.0:  warn = True
        if "height" in label.lower() and q50 < 0.5:        warn = True
        if "quat"   in label.lower() and abs(q50) < 0.85:  warn = True
        if warn:
            ax.text(w_hi + abs(w_hi) * 0.05 + 0.01, li, "⚠",
                    color=FAIL_COLOR, fontsize=9, va="center")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([l[:22] for l in labels], fontsize=7)
    style_ax(ax, title=f"{cond} — Safety Margins", xlabel="Value", grid_axis="x")
label_num = "⑥" if args.save_actions else "⑤"
fig.text(0.005, 0.14, f"{label_num} SAFETY", fontsize=8, color=STYLE["accent"],
         fontweight="bold", rotation=90, va="center")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = (
    [Line2D([0], [0], color=CMAP_COND[c], linewidth=3, label=c)
     for c in CONDITIONS if all_data.get(c)]
    + [Line2D([0], [0], color=PASS_COLOR, linewidth=3, label="PASS"),
       Line2D([0], [0], color=FAIL_COLOR, linewidth=3, label="FAIL"),
       Line2D([0], [0], color=WARN_COLOR, linewidth=2, linestyle="--", label="threshold")]
)
fig.legend(handles=handles, loc="lower center", ncol=len(handles),
           fontsize=8, framealpha=0.25, facecolor=STYLE["panel"],
           edgecolor=STYLE["grid"], labelcolor=STYLE["text"],
           bbox_to_anchor=(0.5, 0.003))

out_path = os.path.join(RESULTS_DIR, f"pre_deployment_{args.domain}.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=STYLE["bg"], edgecolor="none")
plt.close(fig)

print(f"\nDiagnostic figure  → {out_path}")
print(f"Data export        → {EXPORT_DIR}/")
print(f"Domain report      → {report_path}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING QUALITY SUMMARY")
print("=" * 60)
for cond in CONDITIONS:
    if not all_pf[cond]:
        print(f"  {cond:<6} : NO DATA"); continue
    for i, (pf, d) in enumerate(zip(all_pf[cond], all_data[cond])):
        m       = pf["metrics"]
        verdict = "PASS" if pf["passed"] else "FAIL"
        print(f"  {cond}/seed{d['seed']} [{verdict}]  "
              f"ret={m['mean_return']:8.1f}±{m['std_return']:.1f}  "
              f"cv={m['cv_return']:.2f}  "
              f"fail={m['fail_rate']:.2f}  "
              f"len={m['mean_ep_len']:.1f}")
        for diag in pf["diagnostics"]:
            print(f"    ⚠  {diag}")
print("=" * 60)

# ── Convergence summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONVERGENCE SUMMARY  (training curves)")
print("=" * 60)
any_conv = False
for cond in CONDITIONS:
    for seed, cv_data in all_curves[cond].items():
        conv = cv_data.get("convergence")
        if not conv: continue
        any_conv = True
        cvgd_str = ("CURVE OK"      if conv["converged"] is True
                    else "CURVE NOT OK" if conv["converged"] is False
                    else "N/A")
        print(f"  {cond}/seed{seed}: {cvgd_str:<14}  "
              f"impr={conv['improvement_ratio']:+.2f}  "
              f"plateau_cv={conv['plateau_cv']:.3f}  "
              f"slope={conv['last20p_slope']:+.5f}  "
              f"init={conv['initial_mean']:.1f} → final={conv['final_mean']:.1f}")
if not any_conv:
    print("  No training curve data found "
          "(add Monitor wrapper to training script for full convergence diagnostics).")
print("=" * 60)
