"""Field-key constants for the VLN project (single source of truth).

Keeping data/model dict keys here (instead of magic strings) is what lets the
data, model, loss and evaluation modules stay decoupled yet aligned - the same
principle used across ai_foundation ``constant/``.
"""


class DataKeys:
    """Keys present in a collated training batch dict."""
    OBSERVATIONS = "observations"
    PREV_ACTIONS = "prev_actions"
    MASKS = "masks"
    CORRECTED_ACTIONS = "corrected_actions"  # [T, N] expert actions
    WEIGHTS = "weights"                      # [T, N] inflection weights
    RNN_STATES = "rnn_states"

    # keys inside the observations sub-dict
    RGB = "rgb"
    DEPTH = "depth"
    INSTRUCTION = "instruction"
    PROGRESS = "progress"


class ModelKeys:
    """Keys produced by model forward / consumed by losses & evaluators."""
    MODE = "mode"                    # 'train' | 'inference' | other
    ACTION_LOGITS = "action_logits"  # [T*N, num_actions]
    ACTIONS = "actions"              # [T*N, 1] argmax actions (inference)
    PROGRESS_HAT = "progress_hat"    # [T*N, 1] predicted progress
    RNN_STATES_OUT = "rnn_states_out"

    TOTAL_LOSS = "total_loss"


class DualSystemDataKeys:
    """Batch keys for the InternVLA-N1 dual-system (Qwen-VL style) model."""
    INPUT_IDS = "input_ids"            # [B, L] token ids
    LABELS = "labels"                  # [B, L] causal-LM labels (-100 = ignore)
    ATTENTION_MASK = "attention_mask"  # [B, L]
    IMAGE_EMBEDS = "image_embeds"      # [B, n_img, vis_dim] synthetic visual features
    TRAJ_TOKEN_POS = "traj_token_pos"  # [B] start index of the traj-token block (t_s_pos)
    TRAJ_POSES = "traj_poses"          # [B, num_traj, predict_step, 3] GT future waypoints
    TRAJ_VALID = "traj_valid"          # [B, num_traj] loss mask over trajectories


class DualSystemModelKeys:
    """Model outputs for the dual-system model."""
    LM_LOGITS = "lm_logits"            # [B, L, vocab] System-2 causal-LM logits
    TRAJ_NOISE_PRED = "traj_noise_pred"  # [B*num_traj, predict_step, 3] System-1 DiT output
    TRAJ_TARGET = "traj_target"        # [B*num_traj, predict_step, 3] flow-matching target
    TRAJ_LOSS_MASK = "traj_loss_mask"  # [B*num_traj] valid-trajectory mask
    PRED_TRAJ = "pred_traj"            # [B*num_traj, predict_step, 3] sampled traj (inference)


class InternNavDataKeys:
    """Batch keys for the real InternNav pixel-goal training pipeline."""

    INPUT_IDS = "input_ids"                  # [B, L]
    LABELS = "labels"                        # [B, L], -100 outside assistant replies
    ATTENTION_MASK = "attention_mask"        # [B, L]
    PIXEL_VALUES = "pixel_values"            # concatenated Qwen image patches
    IMAGE_GRID_THW = "image_grid_thw"        # [num_images, 3]
    TRAJ_TOKEN_POS = "traj_token_pos"        # [B], first appended trajectory token
    TRAJ_IMAGES = "traj_images"              # [B, F, H, W, 3], RGB in [0, 1]
    TRAJ_DEPTHS = "traj_depths"              # [B, F, H, W], meters
    TRAJ_POSES = "traj_poses"                # [B, F, P, 3]
    VIDEO_FRAME_NUM = "video_frame_num"      # [B], valid F before batch padding


class PlannerDataKeys:
    """Batch keys for the e2e-style trajectory-token planner (VLN task).

    Mirrors the role of ai_foundation's E2EDatasetKeys for the planner path.
    """
    RGB = "rgb"                          # [B, n_img, H, W, 3] raw images
    PROMPT_IDS = "prompt_ids"            # [B, L] instruction token ids (with IMAGE placeholders)
    ATTENTION_MASK = "attention_mask"    # [B, L]
    FUTURE_TRAJECTORY = "future_trajectory"  # [B, num_waypoints, coord_dim] GT waypoints
    FUTURE_TRAJECTORY_MASK = "future_trajectory_mask"  # [B, num_waypoints] valid mask


class PlannerModelKeys:
    """Model output keys for the planner (mirror ai_foundation E2EModelKeys)."""
    LLM_HIDDEN_STATES = "llm_hidden_states"          # [B, L', token_dim] last-layer hidden states
    LLM_LAST_HIDDEN_STATE = "llm_last_hidden_state"  # [B, num_waypoints, token_dim] traj-token slice
    MM_MERGED_PROMPT_IDS = "mm_merged_prompt_ids"    # [B, L'] merged prompt ids (debug)
    TRAJECTORY_PROMPT_MATCH_MASK = "trajectory_prompt_match_mask"  # [B, L'] traj-token positions
    PRED_FUTURE_TRAJECTORY = "pred_future_trajectory"  # [B, num_waypoints, coord_dim]
    FLOW_VELOCITY = "flow_velocity"  # [B, W, C], normalized noise-to-data velocity
    FLOW_TARGET = "flow_target"      # [B, W, C], normalized GT minus Gaussian noise
    FLOW_MASK = "flow_mask"          # [B, W], valid waypoint weights
    TEMPORAL_GRID_LOGITS = "temporal_grid_logits"      # [B, num_grid_classes]
    TEMPORAL_GRID_PRED_ID = "temporal_grid_pred_id"    # [B]
    TEMPORAL_GRID_TARGET = "temporal_grid_target"      # [B]
    TEMPORAL_GRID_TARGET_MASK = "temporal_grid_target_mask"  # [B]
    TEMPORAL_GRID_CONDITION_ID = "temporal_grid_condition_id"  # [B]
    TEMPORAL_PREV_GRID_INPUT_ID = "temporal_prev_grid_input_id"  # [B]
    TEMPORAL_PREV_GRID_USE_START = "temporal_prev_grid_use_start"  # [B]


class OmTrackDataKeys:
    """Batch keys for the cached-feature OmTrackVLA training pipeline."""

    COARSE_TOKENS = "coarse_tokens"        # [B, history * coarse_tokens_per_frame, C]
    COARSE_TIME_INDEX = "coarse_tidx"      # [B, history * coarse_tokens_per_frame]
    FINE_TOKENS = "fine_tokens"            # [B, fine_tokens_per_frame, C]
    FINE_TIME_INDEX = "fine_tidx"          # [B, fine_tokens_per_frame]
    YAW_HISTORY = "yaw_hist"               # [B, history]
    YAW_CURRENT = "yaw_curr"               # [B, 1]
    INSTRUCTION = "instruction"             # list[str]
    CURRENT_PATH = "current_path"           # list[str], diagnostics only
    TEMPORAL_PREV_GRID_ID = "temporal_prev_grid_id"       # [B]
    TEMPORAL_PREV_GRID_VALID = "temporal_prev_grid_valid" # [B]
