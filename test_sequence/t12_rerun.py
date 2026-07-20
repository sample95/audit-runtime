"""
T-12  Calibrate → Re-run Deployment
=====================================
1. Runs t11_calibrate to compute empirical thresholds from baseline runs.
2. Injects new tau_* values into the domain config (in-memory, no file mutation).
3. Spawns one worker per model condition (M1, M2, M3) in parallel, each
   running abrupt and gradual deployments with the calibrated thresholds.
   Workers are staggered 30s apart to avoid filesystem collisions at startup.

Usage
-----
  python t12_rerun.py \\
      --domain hopper_t10 \\
      --configs ./configs \\
      --results-dir ./results/mujoco/hopper/t10_rf \\
      --out-dir    ./results/mujoco/hopper/t12_calib_rf

  # Run only one transition schedule (both model conditions still parallelised):
      [--schedule abrupt]

Internal worker mode (called automatically by the launcher — do not invoke directly):
      --worker M1   (or M2, M3)
      --calib-thresholds ./t11_calib/calibration_thresholds.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = Path(sys.executable)


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain",      default="hopper_t10")
    ap.add_argument("--configs",     default="./configs")
    ap.add_argument("--results-dir", required=True,
                    help="Directory with existing baseline-condition metrics files "
                         "(*_baseline_*_metrics*.json, or legacy *_none_*/*_no_drift_* "
                         "naming) — source for T-11 calibration")
    ap.add_argument("--out-dir",     default="./t12_calib_rf",
                    help="Output directory for re-run results")
    ap.add_argument("--schedule",    choices=["abrupt", "gradual"], default=None,
                    help="Run only one transition schedule (default: both)")
    ap.add_argument("--calib-dir",   default="./t11_calib",
                    help="Where to write T-11 calibration output")
    # ── Internal worker args (set by the launcher, not by the user) ──────────
    ap.add_argument("--worker",      choices=["M1", "M2", "M3"], default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--calib-thresholds", default=None,
                    help=argparse.SUPPRESS)
    return ap.parse_args()


def _run_worker(args, model_cond):
    """
    Worker entry point: load calibrated thresholds, inject overrides,
    run part2_reset_free_deploy for the given model condition.
    """
    new_taus = json.load(open(args.calib_thresholds))

    domain_cfg = json.load(open(os.path.join(args.configs, f"{args.domain}.json")))
    instr_cfg  = json.load(open(os.path.join(args.configs, "instrument.json")))

    TAU_MAP = {
        "tau_warn_override":  "tau_warn",
        "tau_watch_override": "tau_watch",
        "tau_R_override":     "tau_R",
        "tau_E_override":     "tau_E",
        "tau_M_override":     "tau_M",
        "tau_cusum_override": "tau_cusum",
        "tau_TS_override":    "tau_TS",
    }
    for cfg_key, calib_key in TAU_MAP.items():
        if calib_key in new_taus:
            instr_cfg[cfg_key] = new_taus[calib_key]

    domain_cfg["results_dir"] = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    schedules = ["abrupt", "gradual"]
    if args.schedule:
        schedules = [args.schedule]
    domain_cfg["transition_schedules"] = schedules

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from t10_survival_deploy import part2_reset_free_deploy
    part2_reset_free_deploy(domain_cfg, instr_cfg, cond_filter=model_cond)


def main():
    args = _parse_args()

    # ── Worker mode (spawned internally) ──────────────────────────────────────
    if args.worker is not None:
        _run_worker(args, args.worker)
        return

    # ── Coordinator mode ──────────────────────────────────────────────────────

    # Step 1: calibrate
    print("=" * 60)
    print("Step 1 — running t11_calibrate …")
    os.makedirs(args.calib_dir, exist_ok=True)
    calib_script = Path(__file__).parent / "t11_calibrate.py"
    ret = os.system(
        f"{PY} {calib_script} "
        f"--results-dir {args.results_dir} "
        f"--out-dir {args.calib_dir}"
    )
    if ret != 0:
        sys.exit(f"t11_calibrate failed (exit {ret})")

    thresholds_path = os.path.join(args.calib_dir, "calibration_thresholds.json")
    new_taus = json.load(open(thresholds_path))
    print(f"\nCalibrated thresholds: {new_taus}\n")

    # Step 2: resolve model conditions from domain config
    domain_cfg  = json.load(open(os.path.join(args.configs, f"{args.domain}.json")))
    model_conds = domain_cfg.get("conditions", ["M1", "M2", "M3"])
    schedules   = ["abrupt", "gradual"] if not args.schedule else [args.schedule]
    print(f"Step 2 — launching {len(model_conds)} parallel workers "
          f"(conditions: {model_conds}, schedules: {schedules}) …\n")

    os.makedirs("logs", exist_ok=True)

    # Step 3: spawn one worker per model condition
    procs = {}
    base_cmd = [
        str(PY), str(Path(__file__).resolve()),
        "--worker", "PLACEHOLDER",
        "--domain",            args.domain,
        "--configs",           args.configs,
        "--results-dir",       args.results_dir,
        "--out-dir",           args.out_dir,
        "--calib-dir",         args.calib_dir,
        "--calib-thresholds",  thresholds_path,
    ]
    if args.schedule:
        base_cmd += ["--schedule", args.schedule]

    for i, cond in enumerate(model_conds):
        if i > 0:
            print(f"  waiting 30s before starting worker {cond} …")
            time.sleep(30)

        log_path = f"logs/t12_{cond}.log"
        cmd = base_cmd[:]
        cmd[cmd.index("PLACEHOLDER")] = cond

        log_f = open(log_path, "w")
        proc  = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
        procs[cond] = (proc, log_f, log_path)
        print(f"  worker {cond} → PID {proc.pid}  (log: {log_path})")

    print()
    print("All workers launched — waiting for completion …")
    print(f"(tail -f logs/t12_M1.log   to monitor)\n")

    failed = []
    for cond, (proc, log_f, log_path) in procs.items():
        ret = proc.wait()
        log_f.close()
        if ret == 0:
            print(f"  ✓ {cond} done")
        else:
            print(f"  ✗ {cond} FAILED (exit {ret}) — check {log_path}")
            failed.append(cond)

    print(f"\nResults → {args.out_dir}")
    if failed:
        print(f"Failed conditions: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
