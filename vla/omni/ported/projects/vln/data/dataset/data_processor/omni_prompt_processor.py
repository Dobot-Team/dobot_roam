"""System -> observation video -> task text/goal image; ACT is appended by model."""
import torch
from vla.omni.ported.projects.vln.data.dataset.data_processor.prompt_processor import VLNPromptProcessor
from vla.omni.ported.projects.vln.data.dataset.data_processor.processors import PlannerRGBNoiseProcessor


class OmniVLNPromptProcessor(VLNPromptProcessor):
    def __init__(self, **kwargs):
        # Task processors have already supplied the final instruction.
        kwargs["use_origin_prompt"] = True
        super().__init__(**kwargs)

    def __call__(self, sample, mode="train"):
        # Reuse the established video timestamp/token construction without
        # changing its temporal patch layout or tokenizer.
        observation = dict(sample)
        observation["instruction_text"] = ""
        super().__call__(observation, mode)
        ids = observation["prompt_ids"].tolist()
        types = observation["mm_token_type_ids"].tolist()
        text = observation["prompt_text"]
        for part in sample.get("instruction_parts", [{"text": sample.get("instruction_text", "")}]):
            if "text" in part:
                encoded = self._encode(part["text"])
                ids.extend(encoded)
                types.extend([0] * len(encoded))
                text += part["text"]
            elif part.get("image") == "goal_rgb":
                goal = sample["goal_rgb"]
                h, w = goal.shape[:2]
                n = (h // 16 // 2) * (w // 16 // 2)
                if h % 32 or w % 32 or n <= 0:
                    raise ValueError("goal image dimensions must be positive multiples of 32")
                ids.extend([self.vision_start_id] + [self.image_token_id] * n + [self.vision_end_id])
                types.extend([0] + [1] * n + [0])
                text += "<vision_start><goal_image><vision_end>"
            else:
                raise ValueError(f"unsupported instruction block {part}")
        sample["prompt_ids"] = torch.tensor(ids, dtype=torch.long)
        sample["attention_mask"] = torch.ones(len(ids), dtype=torch.long)
        sample["mm_token_type_ids"] = torch.tensor(types, dtype=torch.long)
        sample["prompt_text"] = text


class OmniRGBNoiseProcessor(PlannerRGBNoiseProcessor):
    def __call__(self, sample, mode="train"):
        super().__call__(sample, mode)
        if mode == "train" and "goal_rgb" in sample:
            rgb = sample["goal_rgb"]
            sample["goal_rgb"] = (rgb + torch.randn_like(rgb) * self.std).clamp_(0, 255)
