"""
t01_report.py — T-01 diagnostic result reader
Usage (from audit-framework/):
    python t01_report.py
Reads results/mujoco/hopper_t01/ and prints a clear FAR + tau_warn table.
"""
import json, glob, os, sys
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "results", "mujoco", "hopper_t01")
files = sorted(glob.glob(os.path.join(RESULTS, "*_metrics_summary.json")))

if not files:
    print(f"No results found in {RESULTS}")
    print("Run the diagnostic first:  python ../audit_deploy.py --domain hopper_t01 --configs configs")
    sys.exit(0)

print(f"\n{'='*75}")
print(f"  T-01 DIAGNOSTIC RESULTS — tau_warn_floor check")
print(f"  Results dir: {RESULTS}")
print(f"{'='*75}\n")

rows = []
for fp in files:
    with open(fp) as f:
        m = json.load(f)
    K0     = m["K0"]
    Dk     = m["D_k"]
    starts = m["window_starts"]
    tw     = m["tau_warn"]
    floor  = m.get("tau_warn_floor", "n/a")
    T_warn = m.get("T_warn")
    pilot_D = m["pilot_D"]

    post   = [(d, s) for d, s in zip(Dk[K0:], starts[K0:])]
    exceed = [(d, s) for d, s in post if d > tw]
    rate   = len(exceed) / len(post) if post else 0.0

    rows.append({
        "seed":        m["seed"],
        "tau_warn":    tw,
        "floor":       floor,
        "pilot_D_max": max(pilot_D) if pilot_D else 0,
        "n_post":      len(post),
        "n_exceed":    len(exceed),
        "exceed_rate": rate,
        "T_warn":      T_warn,
        "false_alarm": T_warn is not None,
    })

rows.sort(key=lambda r: r["seed"])

hdr = f"{'Seed':>5}  {'tau_warn':>11}  {'floor':>11}  {'pilot_D_max':>12}  " \
      f"{'n_post':>7}  {'exceed_rate':>12}  {'T_warn':>9}  FA?"
print(hdr)
print("-" * 90)
for r in rows:
    tw_str  = f"{r['tau_warn']:.3e}"
    fl_str  = f"{r['floor']:.3e}" if isinstance(r['floor'], float) else str(r['floor'])
    pd_str  = f"{r['pilot_D_max']:.3e}"
    tw_out  = str(r['T_warn']) if r['T_warn'] else "None"
    fa_flag = "YES ◄" if r['false_alarm'] else "no"
    print(f"{r['seed']:>5}  {tw_str:>11}  {fl_str:>11}  {pd_str:>12}  "
          f"{r['n_post']:>7}  {r['exceed_rate']:>11.2f}  {tw_out:>9}  {fa_flag}")

n_fa  = sum(1 for r in rows if r['false_alarm'])
total = len(rows)
print(f"\n  False alarms: {n_fa}/{total} seeds")
print(f"  Target: ≤1/5 seeds (Pr(warn) ≤ 0.20 under M1 × none)\n")

if n_fa == 0:
    print("  ✅ PASS — floor eliminates all false alarms. Ready for T-03.")
elif n_fa == 1:
    print("  ✅ PASS (marginal) — one residual false alarm. Check exceed_rate; consider m_fail=3.")
else:
    print("  ✗  FAIL — floor insufficient. Need higher tau_warn_floor (try 5e-4) or m_fail increase.")
print()
