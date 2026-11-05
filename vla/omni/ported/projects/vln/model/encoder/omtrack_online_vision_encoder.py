"""Configurable DINOv3/SigLIP online token encoder for OmTrack."""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _square_size(value: int) -> int:
    size = int(round(math.sqrt(value)))
    if size * size != value:
        raise ValueError(f"{value} is not a perfect square")
    return size


def grid_pool_tokens(
    patch_tokens: torch.Tensor,
    grid_height: int,
    grid_width: int,
    output_tokens: int,
) -> torch.Tensor:
    """Adaptive-average-pool a flattened patch grid to a square token grid."""

    batch_size, patch_count, channels = patch_tokens.shape
    if patch_count != grid_height * grid_width:
        raise ValueError(
            f"patch count {patch_count} != grid {grid_height}x{grid_width}"
        )
    output_size = _square_size(output_tokens)
    features = patch_tokens.transpose(1, 2).contiguous().view(
        batch_size, channels, grid_height, grid_width
    )
    features = F.adaptive_avg_pool2d(features, (output_size, output_size))
    return features.flatten(2).transpose(1, 2).contiguous()


def adapt_siglip_grid(
    tokens: torch.Tensor,
    grid_size: Tuple[int, int],
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Pool the SigLIP token grid onto the DINO patch grid."""

    batch_size, patch_count, channels = tokens.shape
    height, width = grid_size
    if height * width != patch_count:
        raise ValueError(f"SigLIP token count {patch_count} != grid {grid_size}")
    features = tokens.transpose(1, 2).contiguous().view(
        batch_size, channels, height, width
    )
    features = F.adaptive_avg_pool2d(features, output_size)
    return features.flatten(2).transpose(1, 2).contiguous()


class OmTrackOnlineVisionEncoder(nn.Module):
    """Encode one RGB frame with DINO, SigLIP, or both vision towers."""

    def __init__(
        self,
        dino_model_path: Optional[str] = None,
        siglip_model_path: Optional[str] = None,
        image_size: int = 384,
        coarse_tokens: int = 4,
        fine_tokens: int = 64,
        device: str = "cuda",
        local_files_only: bool = True,
        vision_backbone: Optional[str] = None,
        siglip_reference_grid_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if vision_backbone is None:
            vision_backbone = (
                "dino+siglip" if siglip_model_path is not None else "dino"
            )
        if vision_backbone not in {"dino", "siglip", "dino+siglip"}:
            raise ValueError(
                "vision_backbone must be 'dino', 'siglip', or 'dino+siglip'"
            )
        needs_dino = vision_backbone in {"dino", "dino+siglip"}
        needs_siglip = vision_backbone in {"siglip", "dino+siglip"}
        if needs_dino and dino_model_path is None:
            raise ValueError(f"dino_model_path is required for {vision_backbone}")
        if needs_siglip and siglip_model_path is None:
            raise ValueError(
                f"siglip_model_path is required for {vision_backbone}"
            )

        self.vision_backbone = vision_backbone
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.coarse_tokens = int(coarse_tokens)
        self.fine_tokens = int(fine_tokens)
        self.siglip_reference_grid_size = (
            None
            if siglip_reference_grid_size is None
            else int(siglip_reference_grid_size)
        )
        if (
            self.siglip_reference_grid_size is not None
            and self.siglip_reference_grid_size <= 0
        ):
            raise ValueError("siglip_reference_grid_size must be positive")
        self.dino_processor: Optional[Any] = None
        self.dino: Optional[nn.Module] = None
        self.dino_register_tokens = 0
        self.siglip_processor: Optional[Any] = None
        self.siglip: Optional[nn.Module] = None

        if needs_dino:
            try:
                from transformers import AutoImageProcessor, AutoModel
            except ImportError as exc:
                raise ImportError("DINO encoding requires transformers") from exc
            self.dino_processor = AutoImageProcessor.from_pretrained(
                dino_model_path, local_files_only=local_files_only
            )
            self.dino = AutoModel.from_pretrained(
                dino_model_path, local_files_only=local_files_only
            ).eval().to(self.device)
            self.dino_register_tokens = int(
                getattr(self.dino.config, "num_register_tokens", 0) or 0
            )

        if needs_siglip:
            try:
                from transformers import SiglipImageProcessor, SiglipVisionModel
            except ImportError as exc:
                raise ImportError(
                    "SigLIP encoding requires transformers with SigLIP support"
                ) from exc
            self.siglip_processor = SiglipImageProcessor.from_pretrained(
                siglip_model_path, local_files_only=local_files_only
            )
            self.siglip = SiglipVisionModel.from_pretrained(
                siglip_model_path, local_files_only=local_files_only
            ).eval().to(self.device)

    def _preprocess(self, image: Any, processor: Any) -> Dict[str, torch.Tensor]:
        from PIL import Image

        resized = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        inputs = processor(
            images=[resized],
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        )
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
        }

    def _encode_dino_tokens(self, image: Any) -> Tuple[torch.Tensor, int]:
        assert self.dino is not None and self.dino_processor is not None
        output = self.dino(**self._preprocess(image, self.dino_processor))
        tokens = output.last_hidden_state[
            :, 1 + self.dino_register_tokens:, :
        ]
        return tokens, _square_size(tokens.size(1))

    def _encode_siglip_tokens(self, image: Any) -> Tuple[torch.Tensor, int]:
        assert self.siglip is not None and self.siglip_processor is not None
        output = self.siglip(**self._preprocess(image, self.siglip_processor))
        tokens = output.last_hidden_state
        grid_size = int(round(math.sqrt(tokens.size(1))))
        if grid_size * grid_size == tokens.size(1) - 1:
            tokens = tokens[:, 1:, :]
            grid_size = _square_size(tokens.size(1))
        elif grid_size * grid_size != tokens.size(1):
            raise ValueError(
                f"SigLIP token count {tokens.size(1)} is not square"
            )
        return tokens, grid_size

    @torch.inference_mode()
    def forward(self, rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(coarse[4,C], fine[64,C])`` float32 tensors."""

        from PIL import Image

        image = Image.fromarray(rgb.astype(np.uint8))
        if self.vision_backbone == "dino":
            vision_tokens, vision_grid = self._encode_dino_tokens(image)
        elif self.vision_backbone == "siglip":
            vision_tokens, vision_grid = self._encode_siglip_tokens(image)
            if self.siglip_reference_grid_size is not None:
                reference_grid = self.siglip_reference_grid_size
                vision_tokens = adapt_siglip_grid(
                    vision_tokens,
                    (vision_grid, vision_grid),
                    (reference_grid, reference_grid),
                )
                vision_grid = reference_grid
        else:
            dino_tokens, dino_grid = self._encode_dino_tokens(image)
            siglip_tokens, siglip_grid = self._encode_siglip_tokens(image)
            siglip_tokens = adapt_siglip_grid(
                siglip_tokens,
                (siglip_grid, siglip_grid),
                (dino_grid, dino_grid),
            )
            vision_tokens = torch.cat((dino_tokens, siglip_tokens), dim=-1)
            vision_grid = dino_grid
        fine = grid_pool_tokens(
            vision_tokens, vision_grid, vision_grid, self.fine_tokens
        )[0].float()
        coarse = grid_pool_tokens(
            vision_tokens, vision_grid, vision_grid, self.coarse_tokens
        )[0].float()
        return coarse, fine
