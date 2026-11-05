"""Pack optional goal images without dummy visuals for other tasks."""
import torch
from vla.omni.ported.projects.vln.data.collate_fn.planner_collate import planner_collate_fn


def omni_planner_collate_fn(batch):
    result = planner_collate_fn(batch)
    for key in ("task_id", "source_id"):
        result[key] = torch.tensor([sample[key] for sample in batch], dtype=torch.long)
    indices = [i for i, sample in enumerate(batch) if "goal_rgb" in sample]
    result["goal_batch_indices"] = torch.tensor(indices, dtype=torch.long)
    if indices:
        result["goal_rgb"] = torch.stack([batch[i]["goal_rgb"] for i in indices]).unsqueeze(1)
    for key in ("task_name", "source_name", "sample_id", "prompt_text", "goal_metadata", "intent_class"):
        result[key] = [sample.get(key) for sample in batch]
    return result
