"""TokenCompressor - optional vision-token compression (ai_foundation parity).

Thin wrapper over a compressor module, mirroring
``model/arch/components/token_compressor.py``. In the dependency-light dbn build
the default compressor is a simple mean-pool-by-factor; a registry-built module
can be injected instead.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class MeanPoolCompressor(nn.Module):
    """Compress ``[B, N, C]`` visual tokens to ``[B, N//factor, C]`` by mean pooling."""

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = tokens.shape
        keep = (N // self.factor) * self.factor
        if keep == 0:
            return tokens
        pooled = tokens[:, :keep].view(B, keep // self.factor, self.factor, C).mean(dim=2)
        return pooled


class TokenCompressor(nn.Module):
    def __init__(
        self,
        vision_token_compressor: Optional[nn.Module] = None,
        percam_conf: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vision_token_compressor = vision_token_compressor or MeanPoolCompressor()
        self.percam_conf = percam_conf

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.vision_token_compressor(tokens)
