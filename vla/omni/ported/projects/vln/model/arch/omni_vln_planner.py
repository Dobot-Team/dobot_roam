"""One shared ACT/action head with separate observation and goal visual blocks."""
import torch
from vla.omni.ported.projects.vln.model.arch.vln_planner_with_act_token import VLNPlannerWithActToken


class OmniVLNPlannerWithActToken(VLNPlannerWithActToken):
    def __init__(self, task_loss=None, **kwargs):
        super().__init__(**kwargs)
        self.task_loss = task_loss
        if self.video_token_id is None:
            raise ValueError("Omni planner needs a video_token_id for observations")

    def get_loss_dict(self, gt_dict, pred_dict):
        if self.task_loss is None:
            return super().get_loss_dict(gt_dict, pred_dict)
        return self.task_loss(gt_dict, pred_dict)

    def _prepare_embeds(self, input_dict, prompt_ids, attn_mask):
        embeds = self._forward_llm_embedding(prompt_ids)
        observations = self.vision_encoder({"rgb": input_dict["rgb"]})
        if self.token_compressor is not None:
            raise ValueError("Omni visual grids currently require uncompressed ViT tokens")
        for b in range(embeds.shape[0]):
            locations = prompt_ids[b] == self.video_token_id
            if int(locations.sum()) != observations.shape[1]:
                raise ValueError("observation video placeholder/token count mismatch")
            embeds[b, locations] = observations[b].to(embeds.dtype)
        goal_indices = input_dict.get("goal_batch_indices")
        if goal_indices is None:
            goal_indices = torch.empty(0, dtype=torch.long, device=embeds.device)
        if goal_indices.unique().numel() != goal_indices.numel():
            raise ValueError("duplicate goal batch index")
        if ((goal_indices < 0) | (goal_indices >= embeds.shape[0])).any():
            raise ValueError("goal batch index out of range")
        goal_set = set(goal_indices.tolist())
        if goal_set:
            goal_rgb = input_dict["goal_rgb"]
            if goal_rgb.shape[0] != len(goal_set) or goal_rgb.shape[1] != 1:
                raise ValueError("goal images must be [N_goal,1,H,W,3]")
            # T=1 is padded by repeating the image itself by the existing Qwen
            # adapter. This is the image patch contract, with shared ViT weights;
            # never append a goal to the history clip's temporal patches.
            goal_tokens = self.vision_encoder({"rgb": goal_rgb})
            for i, b in enumerate(goal_indices.tolist()):
                positions = prompt_ids[b] == self.image_token_id
                if int(positions.sum()) != goal_tokens.shape[1]:
                    raise ValueError("goal image placeholder/token count mismatch")
                embeds[b, positions] = goal_tokens[i].to(embeds.dtype)
        for b in range(embeds.shape[0]):
            if b not in goal_set and (prompt_ids[b] == self.image_token_id).any():
                raise ValueError("image placeholder has no goal image")
        act = self.act_token.expand(embeds.shape[0], -1, -1).to(embeds.dtype)
        embeds = torch.cat((embeds, act), dim=1)
        prompt_ids = torch.cat((prompt_ids, torch.full(
            act.shape[:2], self.act_token_marker_id, device=prompt_ids.device, dtype=prompt_ids.dtype)), dim=1)
        attn_mask = torch.cat((attn_mask, torch.ones(
            act.shape[:2], device=attn_mask.device, dtype=attn_mask.dtype)), dim=1)
        return embeds, attn_mask, prompt_ids

    def _build_position_ids(self, attn_mask, prompt_ids, input_dict):
        positions = torch.zeros((3, *prompt_ids.shape), device=prompt_ids.device, dtype=torch.long)
        for b in range(prompt_ids.shape[0]):
            real = attn_mask[b].nonzero(as_tuple=True)[0].tolist()
            cursor, i = 0, 0
            while i < len(real):
                token = int(prompt_ids[b, real[i]])
                if token not in (self.video_token_id, self.image_token_id):
                    positions[:, b, real[i]] = cursor
                    cursor += 1
                    i += 1
                    continue
                start = i
                while i < len(real) and int(prompt_ids[b, real[i]]) == token:
                    i += 1
                span = real[start:i]
                if token == self.video_token_id:
                    full_t, h, w = self._video_llm_grid(input_dict, b)
                    t = 1 if len(span) == h*w else full_t
                else:
                    adapter = self.vision_encoder.vision_model
                    rgb = input_dict["goal_rgb"]
                    stride = adapter.patch_size * adapter.merge_size
                    t, h, w = 1, rgb.shape[2] // stride, rgb.shape[3] // stride
                if len(span) != t*h*w:
                    raise ValueError("visual span / MRoPE grid mismatch")
                axes = torch.meshgrid(
                    torch.arange(t, device=positions.device),
                    torch.arange(h, device=positions.device),
                    torch.arange(w, device=positions.device), indexing="ij")
                positions[:, b, span] = torch.stack([a.reshape(-1) for a in axes]) + cursor
                cursor += max(t, h, w)
        return positions
