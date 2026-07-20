"""
launch_audit.py — parallel 3-worker audit launcher
Run from: ~/msc-thesis/audit-framework/

Usage:
    python launch_audit.py configs/hopper_vc.json
    python launch_audit.py configs/hopper_t10.json
    python launch_audit.py                          # defaults to configs/hopper_vc.json

Each worker handles one condition (M1, M2, M3) across all seeds and transition
schedules (baseline/abrupt/gradual). Workers are staggered 30s apart to avoid
filesystem write collisions at startup.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = Path(sys.executable)   # same interpreter that launched this script

parser = argparse.ArgumentParser()
parser.add_argument("config", nargs="?", default="configs/hopper_vc.json")
args = parser.parse_args()

config_path = Path(args.config)
domain      = config_path.stem
configs_dir = str(config_path.parent)

with open(config_path) as f:
    cfg = json.load(f)

conditions  = cfg.get("conditions", ["M1", "M2", "M3"])
transitions = cfg.get("transition_schedules", cfg.get("drift_schedules", ["baseline", "abrupt", "gradual"]))
results_dir = cfg.get("results_dir", "results/")

# Pick the axis that fills the most workers (max 3).
# Conditions take priority when both axes are equal length.
if len(transitions) > len(conditions):
    axis        = "transition"
    worker_keys = transitions
else:
    axis        = "condition"
    worker_keys = conditions

os.makedirs("logs", exist_ok=True)

print(f"[launch_audit] config  : {config_path}")
print(f"[launch_audit] domain  : {domain}")
print(f"[launch_audit] axis    : {axis}  ({len(worker_keys)} workers: {worker_keys})")
print()

procs = {}
for i, key in enumerate(worker_keys):
    if i > 0:
        print(f"  waiting 30s before starting {key}...")
        time.sleep(30)

    log_path = f"logs/audit_{domain}_{key}.log"
    # [src_clean] resolve audit_deploy.py next to this script (was CWD-relative)
    audit_script = str(Path(__file__).resolve().parent / "audit_deploy.py")
    cmd = [str(PY), audit_script, "--domain", domain, "--configs", configs_dir,
           f"--{axis}", key]

    log_f = open(log_path, "w")
    proc  = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
    procs[key] = (proc, log_f, log_path)
    print(f"  worker {key} → PID {proc.pid:6d}  (log: {log_path})")

print()
print("All workers launched — waiting for completion...")
print(f"(tail -f logs/audit_{domain}_M1.log   to monitor)")
print()

failed = []
for label, (proc, log_f, log_path) in procs.items():
    ret = proc.wait()
    log_f.close()
    if ret == 0:
        print(f"  ✓ {label} done")
    else:
        print(f"  ✗ {label} FAILED (exit {ret}) — check {log_path}")
        failed.append(label)

print()
print(f"Results: {results_dir}")

if failed:
    print(f"\nFailed conditions: {failed}")
    sys.exit(1)
