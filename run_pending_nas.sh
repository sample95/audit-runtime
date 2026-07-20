#!/bin/bash
# ============================================================
# run_pending_nas.sh — All pending NAS runs (2026-07-05)
#
# Run on NAS from:  ~/msc-thesis/audit-framework/
#
# Usage:
#   ./run_pending_nas.sh               # all runs sequentially
#   ./run_pending_nas.sh 510k          # T0=510k abrupt rerun only
#   ./run_pending_nas.sh none_ext      # Extended baseline-condition only
#   ./run_pending_nas.sh favorable     # Favorable-drift E_k retry only
#   ./run_pending_nas.sh hover_5M_q1   # Hover Q1 ~5M run (with RUL)
#   ./run_pending_nas.sh hover_5M_q2   # Hover Q2 ~5M run (with RUL)
#   ./run_pending_nas.sh hover_5M      # Both Q1 and Q2 hover 5M runs
#   ./run_pending_nas.sh sac           # Hopper SAC cross-algorithm comparison
#   ./run_pending_nas.sh sac_5M        # Hopper SAC baseline-condition 5M stability run
#
# After 'favorable' run completes, copy results to SSD then run locally:
#   python test_sequence/t09_compute_ek.py /path/to/t10_favorable --q-warn 0.99
# ============================================================

set -euo pipefail

PYTHON=~/msc-thesis/audit_env/bin/python3
LAUNCH=core/launch_audit.py
RUN_TARGET="${1:-all}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_if() {
    local tag=$1; local cfg=$2
    if [[ "$RUN_TARGET" == "all" || "$RUN_TARGET" == "$tag" ]]; then
        log "=== START: $tag ($cfg) ==="
        $PYTHON $LAUNCH $cfg
        log "=== DONE:  $tag ==="
    else
        log "SKIP: $tag (not selected)"
    fi
}

# hover_5M: Q1 and Q2 run in parallel.
# Each config has 1 condition × 1 schedule → 1 worker per instance, running
# seeds 0–4 sequentially. Two instances in parallel = 2 processes total.
run_hover_5M() {
    log "=== START: hover_5M_q1 + hover_5M_q2 (parallel) ==="
    $PYTHON $LAUNCH configs/hover_q1_5M.json &
    PID_Q1=$!
    $PYTHON $LAUNCH configs/hover_q2_5M.json &
    PID_Q2=$!
    wait $PID_Q1; log "=== DONE: hover_5M_q1 ==="
    wait $PID_Q2; log "=== DONE: hover_5M_q2 ==="
}

# ── 1. T0=510k abrupt rerun ──────────────────────────────────────────────────
# 15 PPO abrupt runs + 15 VecNorm abrupt runs (M1/M2/M3 × seed 0-4)
# Prerequisite: audit_deploy.py patch T0 = domain_cfg.get("t0_abrupt", T_EVAL//2)
#               ALREADY APPLIED in src_clean (2026-06-30).
run_if "510k_ppo" "configs/hopper_t10_510k.json"
run_if "510k_vc"  "configs/hopper_vc_510k.json"

# ── 2. Extended baseline-condition survival (2M steps) ───────────────────────
# 15 PPO baseline runs + 15 VecNorm baseline runs (M1/M2/M3 × seed 0-4)
# B=79 windows → covers full 2M-step evaluation.
run_if "none_ext_ppo" "configs/hopper_t10_none_extended.json"
run_if "none_ext_vc"  "configs/hopper_vc_none_extended.json"

# ── 3. Favorable-drift E_k retry (T-09 Q0.99) ───────────────────────────────
# g_drifted=-4.905 (0.5× nominal). Abrupt + gradual.
# save_trajectories=true — required for t09_compute_ek.py after run.
run_if "favorable" "configs/hopper_t10_favorable.json"

# ── 4. Hover ~5M extended runs (with conditional Weibull RUL) ────────────────
# Q1: Weibull k=2.93, λ=125k (gradual fit).  Q2: k=8.83, λ=174k (gradual fit).
# Both write to results/pyflyt/hover_5M — output files named by condition so
# no collision. Baseline condition only — checks long-term stability beyond 600k
# right-censoring horizon. RUL fires if any agent fails and T_warn triggers.
# B=332, T_eval=4,995,000, w=30k, s=15k.
# Q1 and Q2 run in parallel (1 worker each — 1 condition × 1 schedule × 5 seeds).
if [[ "$RUN_TARGET" == "all" || "$RUN_TARGET" == "hover_5M" ]]; then
    run_hover_5M
elif [[ "$RUN_TARGET" == "hover_5M_q1" ]]; then
    run_if "hover_5M_q1" "configs/hover_q1_5M.json"
elif [[ "$RUN_TARGET" == "hover_5M_q2" ]]; then
    run_if "hover_5M_q2" "configs/hover_q2_5M.json"
fi

# ── 5. Hopper SAC — cross-algorithm comparison (v2 rerun) ────────────────────
# 45 runs (M1/M2/M3 × seed 0-4 × baseline/abrupt/gradual). Same drift params as
# PPO hopper_t10 for direct comparison. 3 workers split by condition.
# v1 (2026-07-06, in t10_sac/) predates the 2026-07-07 audit_deploy.py fix
# (hard_failure_counts was unset -> defaulted False -> episodic reset on
# terminated instead of stopping). Confirmed corrupted: deployment_len==T_eval
# for every run, T_hard clustered 999,898-999,989 — same artifact as the
# invalid Hopper 510k v1 rerun. thesis.tex Sec 3.1.4 "SAC continues operating
# ... without catastrophic failure" was drawn from this artifact and needs
# re-derivation from this v2 rerun. results_dir now t10_sac_v2/ (no collision
# with the invalid v1 output).
# Results → results/mujoco/hopper_sac/t10_sac_v2/
run_if "sac" "configs/hopper_sac.json"

# ── 6. Hopper SAC — baseline-condition 5M stability run (v2 rerun) ───────────
# 15 runs (M1/M2/M3 × seed 0-4 × baseline only). Mirrors hopper_t10_none_5M.json
# and hopper_vc_none_5M.json — all other subjects have 5M baseline-condition runs.
# B=199, T_eval=5M, w=50k, s=25k. 3 workers split by condition.
# v1 (2026-07-06, in t10_sac_none_5M/) predates the same 2026-07-07 fix as
# item 5 above; any run that actually failed mid-deployment has an unreliable
# T_hard. results_dir now t10_sac_none_5M_v2/.
# Results → results/mujoco/hopper_sac/t10_sac_none_5M_v2/
run_if "sac_5M" "configs/hopper_sac_none_5M.json"

if [[ "$RUN_TARGET" == "sac_5M_m2" ]]; then
    log "=== START: sac_5M_m2 (M2 only, OOM rerun) ==="
    $PYTHON $LAUNCH configs/hopper_sac_none_5M_m2.json
    log "=== DONE: sac_5M_m2 ==="
fi

log "All selected runs complete."
log ""
log "Next steps:"
log "  1. Copy new results directories to SSD (ThesisStorage)"
log "  2. For favorable run: python test_sequence/t09_compute_ek.py"
log "       <results>/t10_favorable --q-warn 0.99"
