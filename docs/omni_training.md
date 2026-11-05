# Qwen3.5 Omni V2 training with Lightning

This recipe migrates DobotNav revision `c4a7f59`'s
`dbn/configs/omni_v2/qwen35_08B_omni_v2_base_v2v3data.yaml` into this project's
`scripts/train.py` entry point. It retains Qwen3.5-0.8B, one ACT token, the shared
8×3 trajectory head, all five tasks, raw data processing and trajectory clocks.
It does not enable DobotTrack's InternVL3, Q-identity, Box or Mask heads.

The migration scope is the model architecture and training method. These are
implemented and verified. Actual dataset availability, path remapping and the
full production run are deferred separately; they are not prerequisites for
accepting the framework migration.

## Model structure and framework boundary

The forward path is preserved:

1. The Qwen3.5 vision encoder embeds the observation clip. For image navigation,
   the same encoder embeds the goal image separately from the history clip.
2. Visual embeddings replace the corresponding video/image placeholders in the
   tokenized instruction. The original visual grids, timestamps, attention masks
   and MRoPE positions are retained.
3. A single learned ACT embedding is appended. The Qwen3.5 language model's ACT
   hidden state feeds the shared trajectory head for all five tasks.
4. The head maps 1024 features through two 2048-wide hidden layers to 8×3 local
   x/y/yaw predictions. The original output scale is `[2, 2, 1]`.

The masked trajectory MSE keeps the original coordinate scaling, five task
weights of 1 and global optimization-loss scale of 10. Task means are computed
over valid supervision across the complete accumulation window and all ranks.

| Original component/config | Target integration |
|---|---|
| `MODEL` / `OmniVLNPlannerWithActToken` | `model.planner`, built by `vla/omni/build.py` |
| Qwen3.5 adapter, ACT, shared head, task loss | Preserved implementations under `vla/omni/ported/` |
| `FLOW.optimizer` / `FLOW.lr_scheduler` | `training.optimizer` / `training.lr_scheduler`, same factories and parameter names |
| `SupervisedTrainer` optimization loop | `OmniLightningModule` with Lightning manual optimization |
| Original microbatch fetcher | `OmniDataModule` groups consecutive microbatches into one optimizer step |
| Distributed execution, logging, checkpoints | Lightning `Trainer` and callbacks via `scripts/train.py` |

The target does not import the source checkout or execute its training loop.
The imported source dependencies are namespaced inside this repository, and
the existing training entry point dispatches `model_type: qwen35_omni` to the
Lightning integration. Other model types keep their existing entry path.

## Environment and launch

Use a Python 3.10+ environment with the existing CUDA-compatible torch build.
The original repository-wide `requirements.txt` pins Transformers below 5;
Omni has a separate `requirements-omni.txt` for the official Qwen3.5 backend.
Do not install both requirement sets into the same environment.

In the current `dbn_wcr` container, `.venv-omni` is already prepared. It inherits
the existing torch 2.7.0+cu128 / Transformers 5.14.1 environment and isolates
Lightning 2.5.6, torchmetrics 1.8.2 and lightning-utilities 0.15.2.

The recipe retains the original V2+V3 source definitions. Commands below are
provided for a later run when the configured weights and data are available.
See [validation results and deferred data integration](omni_validation.md).

From the repository root:

```bash
# Recommended in the current B300 container: preserves its validated NCCL
# library and IPv6-loopback rendezvous settings.
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_omni_b300.sh

# Inspect the command without allocating GPUs or starting training.
DRY_RUN=1 bash scripts/train_omni_b300.sh

# Single GPU (batch size is a hardware choice; the recipe defaults to 72/GPU).
CUDA_VISIBLE_DEVICES=0 .venv-omni/bin/python scripts/train.py \
  --config configs/omni_v2/qwen35_08B_omni_v2_base_v2v3data.yaml \
  trainer.devices=1 trainer.strategy=auto

# Resume with the same number of GPUs as the checkpoint.
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_omni_b300.sh \
  --resume outputs/qwen35_08B_omni_v2_base_v2v3data/checkpoints/last.ckpt
```

On another machine, create an environment using the appropriate torch build,
then install `requirements-omni.txt`. The tokenizer path in the migrated YAML
uses the same local Qwen3.5 weight directory; the original tokenizer path was
on an unavailable `/mnt/xsky-mh4/shared` mount. The four image/video special
token IDs were verified against the local tokenizer.

The configuration still contains the original absolute data paths. Data mount
verification and path adaptation are outside the current migration scope.

## Preserved training semantics

| Setting | Value |
|---|---|
| Tasks | tracking / roaming / intent / point_nav / image_nav |
| Sampling shares | 40% / 30% / 5% / 15% / 10% |
| Tracking sources | Dobot : AT : DT : STT = 3 : 1 : 1 : 1 |
| Observation | 5 RGB frames, 256×256, 2 Hz, native video tokens |
| Prediction | 8 local x/y/yaw points, 0.3–2.4 s |
| Goal sampling | 0.3–2.4 s at 80%; 2.4–4.8 s at 20% |
| Optimization | Full parameters; AdamW; original parameter groups and scheduler |
| Peak learning rates | backbone 1e-5; ACT/head 5e-5 |
| Microbatch / accumulation | 72 per GPU / 2 |
| Effective batch | 144 × world size; 1152 on 8 GPUs |
| Training budget | 9000 optimizer steps |
| Checkpoints | Every 3000 optimizer steps, retain 4; last.ckpt links to latest |
| Validation | Every 50000 optimizer steps; absent in a fresh 9000-step run |

`training.gradient_accum_steps` controls accumulation. One Lightning batch is a
list of that many consecutive original microbatches, and one Lightning global
step is one optimizer update. Keep Lightning `accumulate_grad_batches=1`.
The wrapper computes each task's denominator across the whole accumulation
window and all ranks, then backpropagates each microbatch separately. It does
not divide the objective by the accumulation count again. DDP averages the
gradients; the preserved loss compensates by world size.

RGB tensors stay on CPU until their individual microbatch is processed. The
original dataset partitions task plans across ranks/workers; Lightning's
automatic distributed sampler is disabled. Incomplete accumulation windows
raise an error instead of changing the loss normalization.

To validate more frequently, add `trainer.val_check_interval=500`. The value is
in optimizer steps, including across epochs. Validation reports task trajectory
losses and masked XY ADE/FDE; it uses the configured five-task validation loader.
This migration covers training and validation, not the legacy Habitat CLI.

## Checkpoints

```bash
.venv-omni/bin/python scripts/train.py \
  --config configs/omni_v2/qwen35_08B_omni_v2_base_v2v3data.yaml \
  --resume outputs/qwen35_08B_omni_v2_base_v2v3data/checkpoints/last.ckpt
```

Use the same distributed launch/topology as the original run. Lightning restores
model, optimizer, scheduler and loop state. The data module restores the committed
optimizer-batch cursor and replays the consumed portion of the deterministic
epoch plan. Keep data manifests, seed, batch size, workers and world size unchanged.
Worker augmentation / CUDA execution are not promised to resume bit-for-bit.

Original DBN `.pt` optimizer checkpoints cannot be passed to `--resume`. To load
their **model weights only**, set `training.init_checkpoint=/absolute/model.pt`;
strict state-dict loading is used and optimizer/scheduler/data progress start fresh.
The model's internal parameter names remain compatible with the original planner.
Lightning checkpoint state keys add the enclosing `model.` prefix.

## Implementation and checks

- `vla/omni/ported/`: required original component dependency closure, with only
  absolute import namespaces changed. No original trainer is included.
  `SOURCE.json` records source hashes and revision.
- `vla/omni/build.py`: narrow factories for Omni datasets and models.
- `vla/omni/data.py`: Lightning data module, microbatch grouping and replay.
- `vla/omni/module.py`: Lightning optimization, task metrics and validation.
- `vla/omni/train.py`: callbacks, logging, initialization and Trainer integration.

```bash
OMP_NUM_THREADS=1 .venv-omni/bin/python -m pytest -q \
  tests/test_omni_components.py tests/test_omni_lightning.py

# Optional later data integration check; outside the current migration scope.
# Writes subset manifests and a separate resolved config.
CUDA_VISIBLE_DEVICES=0 .venv-omni/bin/python scripts/smoke_omni.py \
  --output /path/to/new/smoke-directory --steps 2
```

The smoke tool uses real recording references and full pretrained weights. It
does not rewrite original manifests or reduce the model. Its validation subset
is for execution checks, not for measuring model quality.
