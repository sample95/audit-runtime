"""
diag_alive_bonus.py
===================
Retroactive alive-bonus masking diagnostic for A_deg.

PURPOSE
-------
Hopper-v5 rewards include a constant alive bonus of +1.0 per surviving step.
This is always collected regardless of locomotion quality.  Because sigma_ref
is set to 1% of mu_ref (which itself includes the alive bonus), a concern is
that the alive component holds y_k artificially close to mu_ref and masks
degradation in the velocity/control signal.

This script quantifies that concern formally by decomposing each window's
mean reward into its alive-bonus and velocity-only components, then
re-running the A_deg threshold check on each component separately.

KEY RESULT (derived in docstring below)
---------------------------------------
When sigma_ref = c * mu_ref  (our design choice: c = 0.01), the A_deg
threshold condition is *invariant* to subtracting a constant alive bonus:

    y_k < mu_ref - 2*sigma_ref
    ⟺  (y_k - b) < (mu_ref - b) - 2*(sigma_ref)     [threshold shifts by b]

If sigma is re-prescribed as c*(mu_ref - b) the sensitivity is also
unchanged in relative terms.  The diagnostic therefore reports:

  (1) alive-bonus fraction of mu_ref per seed (alive/mu_ref %)
  (2) velocity-only range under each drift condition
  (3) required absolute drop (2*sigma_ref) vs observed drop
  (4) detection-efficiency ratio = observed_drop / (2*sigma_ref)
  (5) conclusion: STRONG (≥2σ), MARGINAL (1–2σ), WEAK (<1σ)

If A_deg fires in the original run, detection ratio > 1 is confirmed.
If A_deg does NOT fire, ratio < 1 means insufficient signal, not masking.

USAGE
-----
    python diag_alive_bonus.py [results_dir] [--alive-bonus FLOAT]

    results_dir  Path to folder containing *_metrics_summary.json files.
                 Default: ./results/mujoco/hopper/idk
    --alive-bonus  Per-step alive bonus value (default 1.0 for Hopper-v5)

OUTPUT
------
Console table + CSV file: alive_bonus_diagnostic.csv
"""

import argparse
import json
import os
import sys
import glob
import csv
from collections import defaultdict

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "mujoco", "hopper", "idk"
)
DEFAULT_ALIVE_BONUS = 1.0   # Hopper-v5 is_healthy reward per surviving step
C_DEG               = 2.0   # σ-multiplier used in A_deg threshold
M_FAIL              = 2     # consecutive windows required to confirm A_deg alarm


def apply_m_fail(alarm_raw: list, m_fail: int = 2) -> list:
    """
    Returns confirmed alarm array: alarm fires only when m_fail consecutive
    windows are in alarm state (same logic as the framework's A_deg).
    """
    confirmed = [0] * len(alarm_raw)
    streak = 0
    for i, a in enumerate(alarm_raw):
        streak = streak + 1 if a else 0
        if streak >= m_fail:
            confirmed[i] = 1
    return confirmed


def diagnose_file(path: str, alive_bonus: float) -> dict:
    with open(path) as f:
        d = json.load(f)

    mu_ref    = d["mu_ref"]
    sigma_ref = d["sigma_ref"]
    y_k       = d["y_k"]
    A_deg_orig = d["A_deg"]
    K0        = d["K0"]

    # ── Decompose into velocity-only signal ───────────────────────────────────
    # alive_bonus is constant per step → subtracts cleanly from per-step mean
    y_k_vel = [y - alive_bonus for y in y_k]

    # ── Reference stats for velocity-only component ───────────────────────────
    # Alive bonus contributes ZERO variance (it is constant per step), so the
    # absolute sigma of the velocity component equals sigma_ref.
    mu_vel    = mu_ref - alive_bonus
    sigma_vel = sigma_ref       # unchanged — constant shift does not add variance

    # ── Thresholds ────────────────────────────────────────────────────────────
    thresh_total = mu_ref - C_DEG * sigma_ref     # original framework threshold
    thresh_vel   = mu_vel - C_DEG * sigma_vel     # velocity-only (= thresh_total - alive_bonus)

    # ── Raw (pre-persistence) alarm arrays ───────────────────────────────────
    raw_orig = [1 if y < thresh_total else 0 for y in y_k]
    raw_vel  = [1 if y < thresh_vel   else 0 for y in y_k_vel]

    # ── Confirmed alarms (m_fail persistence) ────────────────────────────────
    # The two conditions are mathematically equivalent:
    #   y_k < thresh_total  ⟺  y_k_vel < thresh_vel
    # This is confirmed/refuted here numerically.
    conf_orig = apply_m_fail(raw_orig, M_FAIL)
    conf_vel  = apply_m_fail(raw_vel,  M_FAIL)

    equiv_check = all(a == b for a, b in zip(raw_orig, raw_vel))

    # ── Sensitivity characterisation ─────────────────────────────────────────
    # Pilot windows (first K0 windows) are excluded from drift analysis
    drift_y_k     = y_k[K0:]
    drift_y_k_vel = y_k_vel[K0:]

    if drift_y_k:
        min_total   = min(drift_y_k)
        min_vel     = min(drift_y_k_vel)
        drop_total  = mu_ref - min_total           # max observed total drop
        drop_vel    = mu_vel - min_vel             # same value (constant shift)
        required    = C_DEG * sigma_ref            # drop needed to trigger
        det_ratio   = drop_total / required if required > 0 else float("inf")
        # Pilot windows: check variance of y_k as cross-check for sigma_ref
        pilot_y_k   = y_k[:K0]
        emp_sigma   = (max(pilot_y_k) - min(pilot_y_k)) / 2 if len(pilot_y_k) > 1 else 0.0
    else:
        min_total = min_vel = drop_total = drop_vel = required = det_ratio = float("nan")
        emp_sigma = 0.0

    # ── Detection conclusion ──────────────────────────────────────────────────
    n_alarms_orig = sum(conf_orig)
    n_alarms_vel  = sum(conf_vel)

    if det_ratio >= C_DEG:
        conclusion = "STRONG  — signal exceeds threshold (A_deg fires)"
    elif det_ratio >= 1.0:
        conclusion = "MARGINAL — 1–2σ drop; A_deg cannot confirm with m_fail persistence"
    else:
        conclusion = "WEAK    — <1σ drop; insufficient degradation signal"

    # ── Alive-bonus fraction ──────────────────────────────────────────────────
    alive_frac = alive_bonus / mu_ref if mu_ref > 0 else float("nan")

    return {
        "file":         os.path.basename(path),
        "condition":    d.get("condition", "?"),
        "seed":         d.get("seed", "?"),
        # "baseline" is canonical going forward; "none"/"no_drift" are pre-rename
        # aliases still present in older result JSONs.
        "transition":   {"none": "baseline", "no_drift": "baseline"}.get(
                             d.get("transition", "?"), d.get("transition", "?")),
        "mu_ref":       mu_ref,
        "sigma_ref":    sigma_ref,
        "alive_frac":   alive_frac,
        "mu_vel":       mu_vel,
        "sigma_vel":    sigma_vel,
        "thresh_total": thresh_total,
        "thresh_vel":   thresh_vel,
        "thresh_equiv": equiv_check,     # True → subtraction is a no-op (expected)
        "min_y_total":  min_total,
        "min_y_vel":    min_vel,
        "drop_total":   drop_total,
        "required_drop":required,
        "det_ratio":    det_ratio,
        "emp_sigma_pilot": emp_sigma,
        "n_alarms_orig":   n_alarms_orig,
        "n_alarms_vel":    n_alarms_vel,
        "conclusion":  conclusion,
    }


def print_table(rows: list):
    header = (
        f"{'FILE':<40} {'DRIFT':<10} {'mu_ref':>7} {'alive%':>7} "
        f"{'drop/2σ':>8} {'n_alarm':>7}  CONCLUSION"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        alive_pct = r["alive_frac"] * 100
        det       = r["det_ratio"]
        det_str   = f"{det:.2f}" if det == det else "  n/a"   # nan check
        print(
            f"{r['file']:<40} {r['transition']:<10} {r['mu_ref']:>7.3f} "
            f"{alive_pct:>6.1f}% {det_str:>8} {r['n_alarms_orig']:>7}  "
            f"{r['conclusion']}"
        )
    print(sep)


def write_csv(rows: list, out_path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Folder containing *_metrics_summary.json files")
    parser.add_argument("--alive-bonus", type=float, default=DEFAULT_ALIVE_BONUS,
                        help="Per-step alive bonus value (default: 1.0 for Hopper-v5)")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "*_metrics_summary.json")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: no *_metrics_summary.json files found under:\n  {args.results_dir}")
        sys.exit(1)

    print(f"\nAlive-bonus diagnostic  (bonus = {args.alive_bonus:.2f}/step, "
          f"C_deg = {C_DEG}σ, m_fail = {M_FAIL})\n")
    print(f"Found {len(files)} metrics files in {args.results_dir}\n")

    rows = []
    for path in files:
        try:
            r = diagnose_file(path, args.alive_bonus)
            rows.append(r)
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}")

    print_table(rows)

    # ── Summary insight ───────────────────────────────────────────────────────
    print("\nINSIGHT")
    print("-------")
    print("The threshold condition  y_k < mu_ref - 2σ  is INVARIANT to subtracting")
    print("a constant alive bonus b, because:                                       ")
    print("   y_k - b  <  (mu_ref - b) - 2*sigma_vel   ⟺   y_k  <  mu_ref - 2*sigma_ref")
    print("(holds exactly when sigma_vel = sigma_ref, i.e. bonus contributes 0 variance)\n")

    no_fire = [r for r in rows if r["n_alarms_orig"] == 0
               and r["transition"] != "baseline"]
    if no_fire:
        print("Runs where A_deg DID NOT fire under drift:")
        for r in no_fire:
            print(f"  {r['file']}  →  det_ratio = {r['det_ratio']:.2f}σ  "
                  f"({r['conclusion'].split('—')[0].strip()})")
        print("\nFor these runs: alive-bonus masking is NOT the cause.")
        print("The velocity component itself shows < 2σ degradation.")
        print("Recommendation: check if policy is genuinely more robust at this seed,")
        print("or whether drift magnitude is insufficient to overcome sigma sensitivity.\n")
    else:
        print("A_deg fires correctly under all drift conditions in this result set.")
        print("Alive-bonus masking (H1) is not observed.\n")

    # ── Check mathematical equivalence ───────────────────────────────────────
    all_equiv = all(r["thresh_equiv"] for r in rows)
    print(f"Threshold equivalence check (total ⟺ vel): {'PASS ✓' if all_equiv else 'FAIL ✗'}")
    if not all_equiv:
        bad = [r["file"] for r in rows if not r["thresh_equiv"]]
        print(f"  Discrepant files: {bad}")

    # ── CSV output ────────────────────────────────────────────────────────────
    out_csv = os.path.join(args.results_dir, "alive_bonus_diagnostic.csv")
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
