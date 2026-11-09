<div align="center">

# DobotRoam

### Roaming and Motion-Intent Navigation with a Shared Vision-Language-Action Policy

Chengrui Wei<sup>1</sup>, Songyuan Liu<sup>1</sup><br>, Liu Liang<sup>1</sup>, Pengkun Zheng<sup>1,†</sup>

<sup>1</sup>Dobot Robotics &nbsp; <sup>†</sup>Project lead
<!-- <sup>&#42;</sup>Corresponding authors &nbsp; <sup>†</sup>Project lead -->

**Qwen3.5-0.8B · One ACT Token · Two Navigation Tasks · PyTorch Lightning**

[Overview](#overview) · [Tasks](#navigation-tasks) · [Examples](#qualitative-results) · [Model](#model)

</div>

## Overview

**DobotRoam** learns **Roam** and **Motion Intent** with one shared vision-language-action policy. Given a short history of RGB observations and a natural-language instruction, **Qwen3.5-0.8B** produces a single **ACT hidden state**. A shared trajectory head maps this state to **8 future egocentric waypoints**, each containing **x, y and yaw**.

<img src="assets/omni/roam_intent_architecture.png" alt="Qwen3.5-0.8B policy: RGB history and task instruction feed one ACT token and a shared trajectory head; training uses Roam and Motion Intent trajectory supervision" width="100%">

*Shared model and training supervision. Motion-intent labels are derived from ground-truth trajectories to construct instructions during training; inference consumes an instruction and predicts a trajectory.*

## News

- **[2026-11-05]** Roam + Motion Intent documentation, architecture figures and a dedicated two-task training configuration are available. Six open-loop evaluation visualizations are included below.
- **[2026-11-04]** The original Qwen3.5 Omni model and optimization method were integrated with PyTorch Lightning.

## Qualitative results

**Green: ground truth. Orange: prediction.** Each visualization contains the current RGB input, a metric bird's-eye trajectory plot, and the five observation frames. Waypoints are spaced by **0.3 s**, spanning **2.4 s**. The camera overlay uses an **approximate, uncalibrated flat-ground projection**; the bird's-eye view uses local metric coordinates. No robot actuation is shown.

### Roam

**Instruction:** “Roam slowly while prioritizing safety.”

<a href="assets/omni/eval_20261014/roaming_001.svg"><img src="assets/omni/eval_20261014/roaming_001.png" alt="Roam validation sample 001: indoor corridor, ground-truth and predicted local trajectories, and five observation frames" width="100%"></a>

*Roam / 001 — indoor corridor. Click the image to open the original SVG.*

<details>
<summary><strong>View two more Roam examples</strong></summary>

**Roam / 002 — outdoor walkway.**

<a href="assets/omni/eval_20261014/roaming_002.svg"><img src="assets/omni/eval_20261014/roaming_002.png" alt="Roam validation sample 002: outdoor walkway with ground-truth and predicted trajectories" width="100%"></a>

**Roam / 003 — covered passage.**

<a href="assets/omni/eval_20261014/roaming_003.svg"><img src="assets/omni/eval_20261014/roaming_003.png" alt="Roam validation sample 003: covered passage with ground-truth and predicted trajectories" width="100%"></a>

</details>

### Motion Intent

**Instruction:** “Turn right.”

<a href="assets/omni/eval_20261014/intent_002.svg"><img src="assets/omni/eval_20261014/intent_002.png" alt="Motion Intent validation sample 002: Turn right instruction with ground-truth and predicted trajectories and observation history" width="100%"></a>

*Motion Intent / 002 — right-turn instruction. Click the image to open the original SVG.*

<details>
<summary><strong>View “Go straight.” and “Move forward-left.” examples</strong></summary>

**Motion Intent / 001 — “Go straight.”**

<a href="assets/omni/eval_20261014/intent_001.svg"><img src="assets/omni/eval_20261014/intent_001.png" alt="Motion Intent validation sample 001: Go straight instruction with ground-truth and predicted trajectories" width="100%"></a>

**Motion Intent / 003 — “Move forward-left.”**

<a href="assets/omni/eval_20261014/intent_003.svg"><img src="assets/omni/eval_20261014/intent_003.png" alt="Motion Intent validation sample 003: Move forward-left instruction with ground-truth and predicted trajectories" width="100%"></a>

</details>

### Displayed-sample metrics

These values are copied from the six saved sample records. **They are per-example metrics, not an aggregate benchmark result.** All six examples have 8 valid ground-truth waypoints. ADE and FDE measure XY displacement; yaw MAE is reported separately.

| Sample | Instruction | ADE ↓ (m) | FDE ↓ (m) | Yaw MAE ↓ (°) |
|---|---|---:|---:|---:|
| [Roam / 001](assets/omni/eval_20261014/roaming_001.svg) | Roam slowly while prioritizing safety. | 0.200 | 0.390 | 7.2 |
| [Roam / 002](assets/omni/eval_20261014/roaming_002.svg) | Roam slowly while prioritizing safety. | 0.077 | 0.131 | 2.0 |
| [Roam / 003](assets/omni/eval_20261014/roaming_003.svg) | Roam slowly while prioritizing safety. | 0.169 | 0.400 | 14.5 |
| [Motion Intent / 001](assets/omni/eval_20261014/intent_001.svg) | Go straight. | 0.029 | 0.055 | 2.5 |
| [Motion Intent / 002](assets/omni/eval_20261014/intent_002.svg) | Turn right. | 0.124 | 0.278 | 7.9 |
| [Motion Intent / 003](assets/omni/eval_20261014/intent_003.svg) | Move forward-left. | 0.108 | 0.303 | 10.5 |

## Navigation tasks

Both tasks use recorded roaming trajectories. They share the observation history, model parameters, action head and trajectory targets; the language instruction determines the task.

<img src="assets/omni/roam_intent_tasks.png" alt="Roam uses a general navigation instruction; Motion Intent supports left turn, forward-left, straight, forward-right, right turn and stop" width="100%">

*Task illustrations are schematic examples, not measured robot rollouts.*

| Task | Configuration key | Instruction | Supervision |
|---|---|---|---|
| **Roam** | `roaming` | “Roam slowly while prioritizing safety.” | Recorded local future trajectory |
| **Motion Intent** | `intent` | One of the six commands below | The same trajectory, with an instruction derived from its motion |

### Motion Intent commands

| Motion class | Instruction |
|---|---|
| `left_turn` | Turn left. |
| `forward_left` | Move forward-left. |
| `straight` | Go straight. |
| `forward_right` | Move forward-right. |
| `right_turn` | Turn right. |
| `stop` | Stop. |

Intent labeling examines the **complete valid future trajectory**, using translation, yaw and path bearing. The preserved thresholds are **0.2 m** for stop translation, **0.15 rad** for stop/straight heading bounds and straight path bearing, and **0.45 rad** for left/right turn deviation. Stop is checked first. The implementation is in [the intent processor](vla/omni/ported/projects/vln/data/dataset/data_processor/omni_tasks.py) and [motion classification](vla/omni/ported/projects/vln/data/dataset/dobot_omtrack_dataset.py).

Motion Intent is an instruction-conditioned trajectory task. The model has a shared regression head; it does not use a separate six-class prediction head.