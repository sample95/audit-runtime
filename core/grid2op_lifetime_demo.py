#!/usr/bin/env python3
"""
grid2op_lifetime_demo.py — natural-lifetime prognostics demo on Grid2op.
NEW 2026-07-12. Replaces the infeasible drift-injection live demo.

Rationale (design decision 2026-07-12): l2rpn_case14_sandbox deployments
fail intrinsically — every attainable subject blacks out on NOMINAL
chronics (measured: 0/90 chronic completions across do-nothing, three
gate-passing PPO policies, and a demo-grade PPO). Drift induction is
therefore unnecessary on this domain; the live-monitor question becomes
pure prognostics: predict WHEN the blackout happens. This exercises the
framework's survival/RUL layer (thesis eq. RUL_q, implemented here with
the identical formula) reset-free and in real time on an operationally
irreversible domain. It does NOT exercise D_k drift detection — no
detection/FAR claim may be quoted from these runs.

Determinism note: frozen deterministic policy x deterministic chronic
replay => one lifetime per (policy, chronic). Sample size = number of
DISTINCT nominal chronics; LOO coverage is therefore out-of-sample
prediction over unseen weeks of grid data.

Usage (local venv):
  # 1. collect lifetimes over N nominal chronics (seconds per run)
  python core/grid2op_lifetime_demo.py --ckpt results/grid2op/G1_seed0.zip --n 60
  python core/grid2op_lifetime_demo.py --do-nothing --n 60
  # 2. LOO coverage analysis (reads the lifetimes file it wrote)
  python core/grid2op_lifetime_demo.py --ckpt ... --analyze
  # 3. live demo: one held-out deployment with rolling conditional RUL
  python core/grid2op_lifetime_demo.py --ckpt ... --live 57
"""

import argparse
import json
import math
import os

import numpy as np
import grid2op
from grid2op.Action import PlayableAction
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace, MultiDiscreteActSpace

try:
    from lightsim2grid import LightSimBackend
    _BK = {"backend": LightSimBackend()}
except ImportError:
    _BK = {}

OBS_ATTRS = ["rho", "p_or", "p_ex", "load_p", "gen_p", "actual_dispatch",
             "target_dispatch", "curtailment", "curtailment_limit",
             "gen_p_before_curtail"]
# Action space differs per condition (matches legacy grid2op.json
# action_space_by_condition, campaign of 2026-07-15): G1/G3 were retrained as
# topology-capable MultiDiscrete agents; G2 kept the continuous Box setup.
ACT_SPACE_BY_COND = {
    "G1": ("multidiscrete", ["redispatch", "curtail", "one_sub_set"]),
    "G2": ("box",           ["redispatch", "curtail"]),
    "G3": ("multidiscrete", ["redispatch", "curtail", "one_sub_set"]),
}
ACT_ATTRS = ["redispatch", "curtail"]   # default / do-nothing (unused for stepping)

# ── Weibull machinery — formulas identical to the deployed monitor ────────────
def weibull_mle(ts, iters=300):
    k = 1.0
    for _ in range(iters):
        lt = [math.log(t) for t in ts]
        tk = [t ** k for t in ts]
        k = 1.0 / (sum(t ** k * math.log(t) for t in ts) / sum(tk) - sum(lt) / len(lt))
    lam = (sum(t ** k for t in ts) / len(ts)) ** (1 / k)
    return k, lam

def rul_q(t_e, k, lam, q):
    """Thesis eq. (RUL): conditional quantile given survival to t_e.
    q=2 -> conditional median; q=10 -> conditional 90th percentile."""
    return lam * ((t_e / lam) ** k + math.log(q)) ** (1 / k) - t_e

def clopper_pearson(x, n, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0
    return lo, hi

# ── Environment / subject ─────────────────────────────────────────────────────
def build(env_name, act_space="box", act_attrs=None):
    g2 = grid2op.make(env_name, action_class=PlayableAction, **_BK)
    gymenv = GymEnv(g2)
    gymenv.observation_space = BoxGymObsSpace(g2.observation_space, attr_to_keep=OBS_ATTRS)
    attrs = act_attrs or ACT_ATTRS
    if act_space == "multidiscrete":
        gymenv.action_space = MultiDiscreteActSpace(g2.action_space, attr_to_keep=attrs)
    else:
        gymenv.action_space = BoxGymActSpace(g2.action_space, attr_to_keep=attrs)
    return g2, gymenv

class Subject:
    def __init__(self, ckpt, gymenv, g2):
        self.g2 = g2
        if ckpt is None:
            self.name, self._dn = "do_nothing", True
        else:
            from stable_baselines3 import PPO
            self._dn = False
            # custom_objects: the pickled spaces reference grid2op's dynamically
            # generated per-(env,backend) classes (e.g. PlayableAction_..._LightSimBackend),
            # which don't exist unless the SAME backend built an env first in this
            # process. Supplying the freshly built spaces sidesteps unpickling
            # entirely — works under LightSim AND PandaPower fallback.
            self.model = PPO.load(ckpt, device="cpu", custom_objects={
                "observation_space": gymenv.observation_space,
                "action_space": gymenv.action_space})
            self.name = os.path.basename(ckpt).replace(".zip", "")
            vec = ckpt.replace(".zip", "_vecnormalize.pkl")
            self.vecnorm = None
            if os.path.exists(vec):
                import pickle
                def _load_vec():
                    with open(vec, "rb") as f:
                        return pickle.load(f)
                try:
                    self.vecnorm = _load_vec()
                except AttributeError:
                    # The pkl pickles gym spaces whose grid2op class name embeds
                    # the TRAINING backend (e.g. ..._LightSimBackend). Under a
                    # different active backend that class is never generated.
                    # Register it with a throwaway env of the training backend,
                    # then retry — the ACTIVE deployment backend is unaffected.
                    from lightsim2grid import LightSimBackend
                    # NB: use the plain dataset name (args.env) — the constructed
                    # env's .env_name is decorated with the backend suffix and
                    # is not a valid dataset id.
                    _tmp = grid2op.make(args.env, action_class=PlayableAction,
                                        backend=LightSimBackend())
                    _tmp.close()
                    self.vecnorm = _load_vec()
                self.vecnorm.training = False
        self.gymenv = gymenv

    def act(self, obs):
        if self._dn:
            return None  # signals empty PlayableAction
        o = np.asarray(obs, dtype=np.float32)
        if self.vecnorm is not None:
            o = self.vecnorm.normalize_obs(o)
        a, _ = self.model.predict(o, deterministic=True)
        return a

def deploy_once(subject, chronic_id, rul_ctx=None, print_every=200):
    """One reset-free deployment on one chronic; returns lifetime.
    rul_ctx=(k,lam) -> live mode: print rolling conditional RUL.
    NOTE: do-nothing and policy modes each use ONE grid2op API consistently
    (raw env vs gym wrapper) — mixing reset/step across the two breaks
    grid2op's backend state after a game over."""
    g2, gymenv = subject.g2, subject.gymenv
    g2.set_id(chronic_id)                      # takes effect at next reset
    t, done = 0, False
    if subject._dn:
        g2.reset()
        empty = g2.action_space({})
        while not done:
            _, _, done, _ = g2.step(empty)
            t += 1
            if rul_ctx and t % print_every == 0:
                _rul_print(t, rul_ctx)
    else:
        # DETERMINISM FIX 2026-07-18: never call GymEnv.reset() — it reseeds the
        # underlying env each reset, reshuffling chronic order and overriding
        # set_id (verified: same "chronic" gave 652/4,438/1,379 across runs,
        # while the raw-env do-nothing path is byte-identical across machines
        # AND backends). Drive the raw env; use the gym spaces as converters
        # only — exactly what GymEnv.step does internally, minus the reseeding.
        g2obs = g2.reset()
        obs = gymenv.observation_space.to_gym(g2obs)
        while not done:
            g2act = gymenv.action_space.from_gym(subject.act(obs))
            g2obs, _, done, _ = g2.step(g2act)
            obs = gymenv.observation_space.to_gym(g2obs)
            t += 1
            if rul_ctx and t % print_every == 0:
                _rul_print(t, rul_ctx)
    return t

def _rul_print(t, rul_ctx):
    k, lam = rul_ctx
    p50, p90 = rul_q(t, k, lam, 2), rul_q(t, k, lam, 10)
    print(f"    t={t:5d}  alive | RUL_p50={p50:7,.0f}  ->  blackout expected "
          f"by t={t + p90:7,.0f} (90% bound)")

# ── Main ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None, help="SB3 zip; omit with --do-nothing")
ap.add_argument("--do-nothing", action="store_true")
ap.add_argument("--env", default="l2rpn_case14_sandbox")
ap.add_argument("--n", type=int, default=60, help="distinct nominal chronics")
ap.add_argument("--analyze", action="store_true")
ap.add_argument("--live", type=int, default=None, metavar="CHRONIC_ID",
                help="live demo on this chronic (held out of the fit)")
ap.add_argument("--out", default="results/grid2op_lifetimes")
ap.add_argument("--backend", choices=["auto", "lightsim", "pandapower"], default="auto",
                help="force the power-flow solver (solver-independence tests)")
args = ap.parse_args()

if args.backend == "pandapower":
    _BK = {}                                   # grid2op default backend
elif args.backend == "lightsim":
    from lightsim2grid import LightSimBackend  # hard requirement when forced
    _BK = {"backend": LightSimBackend()}
assert args.ckpt or args.do_nothing, "--ckpt or --do-nothing required"

# infer the checkpoint's action space from its condition prefix (G1/G2/G3)
if args.ckpt:
    cond = os.path.basename(args.ckpt)[:2]
    act_space, act_attrs = ACT_SPACE_BY_COND.get(cond, ("box", ACT_ATTRS))
    print(f"[space] {cond}: {act_space} over {act_attrs}")
else:
    act_space, act_attrs = "box", ACT_ATTRS
g2, gymenv = build(args.env, act_space, act_attrs)
print(f"[backend] {type(g2.backend).__name__}")   # provenance: active solver
subject = Subject(args.ckpt, gymenv, g2)
os.makedirs(args.out, exist_ok=True)
lf_path = os.path.join(args.out, f"{subject.name}_lifetimes.json")

n_pool = len(g2.chronics_handler.real_data.subpaths) // 2  # nominal = first half

if args.live is not None:
    data = json.load(open(lf_path))
    fit = {int(k): v for k, v in data["lifetimes"].items() if int(k) != args.live}
    k, lam = weibull_mle(list(fit.values()))
    print(f"[live] subject={subject.name}  Weibull from {len(fit)} held-out chronics: "
          f"k={k:.2f} lam={lam:,.0f}")
    print(f"[live] deploying reset-free on UNSEEN chronic {args.live} — rolling RUL:")
    T = deploy_once(subject, args.live, rul_ctx=(k, lam))
    checks = list(range(200, T, 200))
    held = sum(T <= t + rul_q(t, k, lam, 10) for t in checks)
    bound0 = lam * (math.log(10)) ** (1 / k)  # unconditional p90 (10% misses expected)
    print(f"[live] BLACKOUT at t={T:,}")
    print(f"[live] rolling conditional 90% bound held at {held}/{len(checks)} checkpoints"
          f" | unconditional p90={bound0:,.0f} -> {'inside' if T <= bound0 else 'outside (allowed ~10% tail)'}")
    json.dump({"chronic": args.live, "T_hard": T, "k": k, "lam": lam},
              open(os.path.join(args.out, f"{subject.name}_live_run.json"), "w"), indent=2)

elif args.analyze:
    data = json.load(open(lf_path))
    ts = list(data["lifetimes"].values())
    k, lam = weibull_mle(ts)
    h90 = h50 = 0
    for i in range(len(ts)):
        rest = ts[:i] + ts[i + 1:]
        kk, ll = weibull_mle(rest)
        h90 += ts[i] <= ll * (math.log(10)) ** (1 / kk)
        h50 += ts[i] <= ll * (math.log(2)) ** (1 / kk)
    n = len(ts)
    lo90, hi90 = clopper_pearson(h90, n)
    lo50, hi50 = clopper_pearson(h50, n)
    print(f"subject          : {subject.name}   (n={n} distinct nominal chronics)")
    print(f"lifetimes        : mean={np.mean(ts):,.0f} median={np.median(ts):,.0f} "
          f"min={min(ts)} max={max(ts):,}")
    print(f"Weibull          : k={k:.2f}  lambda={lam:,.0f}")
    print(f"LOO p90 coverage : {h90}/{n} = {h90/n:.1%}  CP95=[{lo90:.3f}, {hi90:.3f}]  (nominal 0.90)")
    print(f"LOO p50 coverage : {h50}/{n} = {h50/n:.1%}  CP95=[{lo50:.3f}, {hi50:.3f}]  (nominal 0.50)")

else:
    n = min(args.n, n_pool)
    print(f"[collect] {subject.name}: {n} distinct nominal chronics (pool={n_pool})")
    lifetimes = {}
    for cid in range(n):
        T = deploy_once(subject, cid)
        lifetimes[cid] = T
        print(f"  chronic {cid:4d}: T_hard={T:6,}")
    json.dump({"subject": subject.name, "env": args.env, "pool": "nominal-first-half",
               "lifetimes": lifetimes}, open(lf_path, "w"), indent=2)
    print(f"[collect] wrote {lf_path}")
