"""
LOO conditional-Weibull RUL coverage validation (T-40).

Validates the RUL prediction range emitted by audit_deploy.compute_conditional_rul
against observed T_hard, on every dataset with actual failures. Addresses the
T-39 gap: RUL series operational but unverifiable in the null condition.

Method (leave-one-out, per user decision 2026-07-04):
  For each detected+failed run i in a dataset:
    1. Fit Weibull (MLE, right-censoring aware) to T_hard of ALL OTHER runs
       in the same dataset/condition (censored runs enter as censored).
    2. Compute conditional RUL at t_e = T_warn_i using the exact deployed
       formula: delta_q = lam*((t_e/lam)^k + ln q)^(1/k) - t_e, q in {2,10}.
    3. Coverage: T_hard_i <= T_warn_i + delta_p90 (nominal 90%),
                 T_hard_i <= T_warn_i + delta_p50 (nominal 50%).
    4. Per-window series: recompute RUL at each post-alarm window end while
       the run is alive; check narrowing + rolling p90 coverage.

Abrupt (T0=510k) runs with T_warn < 510,000 are pre-onset false alarms and
are EXCLUDED from coverage (prediction not conditioned on a genuine detection);
they are listed separately.

Run:  /home/deck/.venvs/audit-framework/bin/python3 rul_coverage_validation.py
"""

import json
import math
import glob
import os

import numpy as np
from scipy.stats import beta as beta_dist
from lifelines import WeibullFitter

ROOT = "/home/deck/ThesisStorage/src_clean/results/mujoco"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "ppo_gradual":     dict(glob_pat=f"{ROOT}/hopper/t10_v2/*_gradual_metrics_summary.json",     t0=None),
    "vecnorm_gradual": dict(glob_pat=f"{ROOT}/hopper_vc/*_gradual_metrics_summary.json",         t0=None),
}

# EXCLUDED (data-quality finding, 2026-07-04):
# t10_v2_510k and hopper_vc_510k abrupt JSONs store T_hard ≈ 999.7k–1.0M for
# ALL 30 runs with delta=1 and deployment_len=1,000,000. Original t10_v2 abrupt
# runs stop at first termination (T_hard = deployment_len, max 520,050). The
# 510k reruns therefore ran episodically to the full horizon and their T_hard
# is a LAST-termination timestamp, not first hard failure. Trajectory npz files
# contain no dones array, so first post-onset failure is not recoverable
# without a proxy (e.g. z_pos healthy-range reconstruction — PPO only; VecNorm
# obs are normalised). RUL coverage validation on these datasets is invalid
# until T_hard semantics are resolved. T-36 detection-rate conclusions are
# unaffected (they use T_warn only). See test_journal.md T-40 entry.


def load_runs(pat):
    runs = []
    for f in sorted(glob.glob(pat)):
        if "_t09" in f:
            continue
        j = json.load(open(f))
        runs.append(dict(
            run=os.path.basename(f).replace("_metrics_summary.json", ""),
            T_hard=j.get("T_hard"),
            delta=j.get("delta"),
            T_warn=j.get("T_warn"),
            T_eval=j.get("T_eval"),
            W=j.get("W") or 50_000,   # older JSONs store W as null; mujoco design W=50k (S=25k)
            starts=j.get("window_starts"),
        ))
    return runs


def fit_weibull_loo(runs, leave_idx):
    """Censoring-aware Weibull MLE on all runs except leave_idx.
    Censored runs enter with duration=T_eval, event=0."""
    dur, ev = [], []
    for i, r in enumerate(runs):
        if i == leave_idx:
            continue
        if r["delta"] == 1 and r["T_hard"] is not None:
            dur.append(r["T_hard"]); ev.append(1)
        else:
            dur.append(r["T_eval"]); ev.append(0)
    wf = WeibullFitter().fit(np.asarray(dur, float), np.asarray(ev, int))
    return float(wf.rho_), float(wf.lambda_)   # (kappa/shape, lambda/scale)


def cond_rul(t_e, kappa, lam, q):
    """Exact deployed formula (audit_deploy.compute_conditional_rul)."""
    base = (t_e / lam) ** kappa
    return max(lam * (base + math.log(q)) ** (1.0 / kappa) - t_e, 0.0)


def clopper_pearson(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def analyse(name, cfg):
    runs = load_runs(cfg["glob_pat"])
    n_total = len(runs)
    # data quality: censored flagged, not dropped
    n_cens = sum(1 for r in runs if r["delta"] != 1 or r["T_hard"] is None)

    rows, preonset = [], []
    for i, r in enumerate(runs):
        if r["T_warn"] is None or r["delta"] != 1 or r["T_hard"] is None:
            continue
        if cfg["t0"] is not None and r["T_warn"] < cfg["t0"]:
            preonset.append(r["run"])
            continue
        if r["T_hard"] < r["T_warn"]:          # alarm after failure — invalid
            continue
        kappa, lam = fit_weibull_loo(runs, i)
        t_e = float(r["T_warn"])
        d50, d90 = cond_rul(t_e, kappa, lam, 2), cond_rul(t_e, kappa, lam, 10)
        lead = r["T_hard"] - r["T_warn"]

        # per-window series after alarm, while alive
        w = r["W"]
        narrowing, roll_cov, n_alive = None, None, 0
        if r["starts"]:
            p90s = []
            for ps in r["starts"]:
                we = int(ps) + w
                if we < r["T_warn"] or we >= r["T_hard"]:
                    continue
                p90s.append(cond_rul(float(we), kappa, lam, 10))
            n_alive = len(p90s)
            if n_alive >= 2:
                narrowing = bool(all(b < a for a, b in zip(p90s, p90s[1:])))
            if n_alive >= 1:
                # rolling coverage: at each alive post-alarm window, does
                # win_end + rul_p90 still bracket the eventual failure?
                covs = []
                for ps in r["starts"]:
                    we = int(ps) + w
                    if we < r["T_warn"] or we >= r["T_hard"]:
                        continue
                    covs.append(r["T_hard"] <= we + cond_rul(float(we), kappa, lam, 10))
                roll_cov = float(np.mean(covs))

        rows.append(dict(
            run=r["run"], T_warn=r["T_warn"], T_hard=r["T_hard"], lead=lead,
            loo_kappa=round(kappa, 3), loo_lambda=round(lam, 1),
            rul_p50=round(d50), rul_p90=round(d90),
            T_pred_p50=round(t_e + d50), T_pred_p90=round(t_e + d90),
            cov_p50=bool(lead <= d50), cov_p90=bool(lead <= d90),
            n_postalarm_windows=n_alive,
            p90_monotone_narrowing=narrowing,
            rolling_p90_coverage=roll_cov,
        ))

    n = len(rows)
    k90 = sum(r["cov_p90"] for r in rows)
    k50 = sum(r["cov_p50"] for r in rows)
    summary = dict(
        dataset=name, n_runs_total=n_total, n_censored=n_cens,
        n_validated=n, preonset_false_alarms=preonset,
        p90_coverage=f"{k90}/{n}" if n else "0/0",
        p90_coverage_rate=round(k90 / n, 3) if n else None,
        p90_CI95=[round(x, 3) for x in clopper_pearson(k90, n)] if n else None,
        p50_coverage=f"{k50}/{n}" if n else "0/0",
        p50_coverage_rate=round(k50 / n, 3) if n else None,
        p50_CI95=[round(x, 3) for x in clopper_pearson(k50, n)] if n else None,
        narrowing_confirmed=sum(1 for r in rows if r["p90_monotone_narrowing"]),
        narrowing_evaluable=sum(1 for r in rows if r["p90_monotone_narrowing"] is not None),
    )
    return summary, rows


def main():
    all_summ, all_rows = [], {}
    for name, cfg in DATASETS.items():
        s, rows = analyse(name, cfg)
        all_summ.append(s)
        all_rows[name] = rows
        print(f"\n=== {name} ===")
        for k, v in s.items():
            print(f"  {k}: {v}")

    out = dict(test="T-40 LOO conditional RUL coverage validation",
               date="2026-07-04", summaries=all_summ, per_run=all_rows)
    path = os.path.join(OUT_DIR, "t40_rul_coverage_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    main()
