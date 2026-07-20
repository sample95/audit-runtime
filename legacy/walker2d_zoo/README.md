# Walker2d External-Checkpoint Audit — Archived Workstream (2026-07-12)

Superseded by the author's own Walker2d campaign (deployed 2026-07-12, in the thesis). Preserved as evidence of the external-subjects experiment.

**Outcome:** both public SB3-Zoo checkpoints (HuggingFace `sb3/ppo-Walker2d-v3`, `sb3/sac-Walker2d-v3`; provenance in `checkpoints/PROVENANCE.md`) failed the pre-deployment admission in reset-free deployment on Walker2d-v4: ZP fell at step 175, ZS at 27,955 — both before the 125,000-step pilot (outcome 0). Cause: the v3→v4 transfer gap. The framework's gate correctly rejected unfit third-party policies — the accountability mechanism working as designed. Logs in `logs/`.

Not thesis content (author decision); available if the external-subjects angle is ever revisited.
