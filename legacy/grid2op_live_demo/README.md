# Grid2op Live-Monitor Demo — Archived Workstream (2026-07-12)

Grid2op was removed from the thesis (author decision, 2026-07-12). This folder preserves the complete workstream as evidence for the Learning Agreement discussion — the authoritative analysis is `grid2op_feasibility_2026-07-12.md` in the thesis project folder.

**Why archived, in one line:** reset-free blackout-terminal monitoring requires surviving ≥16,128 steps before chronic-split drift arrives; the best gate-passing agents (means 1,380–1,642 steps), the do-nothing baseline (1,430), and a demo-grade PPO (~800) complete 0/90 nominal chronics. Long survival on this benchmark requires discrete topology control (competition-grade engineering) — outside the thesis's continuous-action subject class.

Contents: training (`train_grid2op.py`, `grid2op.json`), gymnasium adapter + demo driver (`grid2op_gym_env.py` — includes the `scripted_agent="do_nothing"` mode, `run_grid2op_demo.py`), feasibility tests (`dn_feasibility.py`, `ckpt_feasibility.py`), demo configs (`grid2op_demo.json`, `grid2op_dn.json`, `grid2op_smoke.json`), the Kaggle notebook, and the trained G1_seed0 checkpoint (return 37,862, gate-passing under the Chronic thresholds).

If resurrected: scripts assume they live in `core/` (sys.path shims) — run with `PYTHONPATH=core:` or move them back. The redesign that would make the demo viable is recorded in the feasibility document: continuous mid-chronic parameter perturbation instead of chronic switching, plus a topology-capable subject.
