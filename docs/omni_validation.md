# Omni migration validation — 2026-11-04

Validation ran inside `dbn_wcr` on `172.16.9.20`, using the target repository's
`.venv-omni`. The original DobotNav working tree was left unchanged.

The completed delivery covers model structure and training-method migration.
Dataset path investigation and production training are deferred. The real-data
checks below record earlier completed checks; no further actual-data inspection
is required for the current scope.

## Component and framework equivalence

- All 43 copied source dependencies match DobotNav `c4a7f59` after normalizing
  the `dbn` → `vla.omni.ported` import namespace. Source hashes are recorded in
  `vla/omni/ported/SOURCE.json`.
- The resolved model, dataset, optimizer and scheduler settings match the
  original recipe. The tokenizer path was updated to the local model mount;
  its four image/video marker IDs match the recipe.
- The production eight-GPU data setup indexed 4,698,084 records and forms
  4,078 complete optimizer steps per epoch with the original batch size,
  workers and accumulation settings.
- 23 tests passed: 14 original component tests and 9 Lightning integration
  cases. These cover effective-batch gradients, data and optimizer checkpoint
  replay with zero/two workers, resume at an epoch boundary, validation metrics,
  and checkpoint enable/disable behavior including the `last.ckpt` symlink.
- A two-process CPU/Gloo check matched the globally combined reference batch
  update on both ranks, including per-task denominator reduction and DDP
  gradient averaging. The probe is `tests/omni_ddp_probe.py`.
- 80 real samples matched the original loader's images, tokens, goals,
  trajectories, masks and metadata: tracking 32, roaming 24, intent 4,
  point_nav 12 and image_nav 8. Random RGB noise was disabled for this equality
  comparison. This subset used the available V2 and TrackVLA recordings.

## Full pretrained model execution

These checks used the complete Qwen3.5 model with real data and small batches
(2 samples per GPU, 2 microbatches per optimizer step).

| Check | Result |
|---|---|
| Pretrained weights | Loaded with zero missing, unexpected or unrouted keys |
| One B300 | One optimizer step, two validation batches and a full checkpoint |
| Checkpoint resume | Restored and continued to optimizer step 2 |
| Two B300 GPUs | One optimizer step and validation through the new launcher and NCCL 2.28.9 |
| Loss/metrics | Finite in all runs; training total loss 0.284114, resumed 0.453221, two-GPU 0.530617 |

Smoke configs, sample manifests and CSV metrics are retained under
`/tmp/omni-migration-real-smoke`. Large temporary smoke checkpoints were removed
after the successful restore check. The smoke validation subset overlaps the
training subset and is only an execution check, not a model-quality evaluation.

The full eight-GPU, batch-72, 9000-step production run has not been launched.
Its memory capacity and long-run convergence are not established by the small
batch checks above.

## Deferred data integration (outside the current migration scope)

The JSONL manifests exist, but sampled V3 rows point to missing recording files
under `/mnt/juicefs/mh4datasets/pipeline_v3_20261013`. All 12 sampled V3
`training.json` references were missing. In contrast, 12 sampled references
from each checked V2 and TrackVLA source were available.

The original V2+V3 production recipe has not been filtered or rewritten. The
existing dataset's `resample_on_error` behavior can replace failed draws, so
starting training before restoring V3 paths could silently change the effective
data mix. This observation is retained for a future data integration phase;
no path remapping or further data investigation is part of this delivery.
