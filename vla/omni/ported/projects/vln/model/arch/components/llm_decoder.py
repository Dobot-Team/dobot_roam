"""LLMDecoder - wraps the language model and normalizes its I/O.

Faithful (dependency-light) analogue of ai_foundation's
``model/arch/components/llm_decoder.py``: consumes a dict with
``inputs_embeds`` / ``attention_mask`` / ``position_ids`` and returns a dict with
last-layer hidden states, so the arch stays agnostic to the concrete LLM backend.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vla.omni.ported.projects.vln.constant.keys import PlannerModelKeys

# input keys (mirror ai_foundation LLMDecoderInputKeys)
INPUTS_EMBEDS = "inputs_embeds"
ATTN_MASK = "attn_mask"
POSITION_IDS = "position_ids"


class LLMDecoder(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        llm_peft_config: Optional[Any] = None,
        retrieve_last_hidden_from_states: bool = True,
        output_llm_hidden_states: bool = True,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.llm_peft_config = llm_peft_config
        self.retrieve_last_hidden_from_states = retrieve_last_hidden_from_states
        self.output_llm_hidden_states = output_llm_hidden_states

    def forward(self, llm_input_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        inputs_embeds = llm_input_dict[INPUTS_EMBEDS]
        attn_mask = llm_input_dict.get(ATTN_MASK)
        position_ids = llm_input_dict.get(POSITION_IDS)
        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )  # [B, L, token_dim]
        return {PlannerModelKeys.LLM_HIDDEN_STATES: hidden_states}
