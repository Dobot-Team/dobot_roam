"""ImageNormalizeAndCast - normalize uint8 images and cast dtype.

Faithful (dependency-light) analogue of ai_foundation's
``model/arch/components/image_normalize_and_cast.py``: when the dataloader hands
raw uint8 [0,255] images, normalize to [0,1] (ImageNet mean/std) and cast to the
model compute dtype. A no-op when ``use_uint8_input_images=False`` and inputs are
already normalized.
"""

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
class ImageNormalizeAndCast(nn.Module):
    def __init__(
        self,
        use_uint8_input_images: bool = False,
        normalized_data_type: str = "fp32",
        normalize: bool = True,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.enabled = use_uint8_input_images
        self.normalize = normalize
        self.output_dtype = _DTYPE_MAP.get(normalized_data_type, torch.float32)
        mean = torch.as_tensor(image_mean, dtype=torch.float32)
        std = torch.as_tensor(image_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("image_mean and image_std must contain three values")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("image_mean/std must be finite and image_std must be positive")
        self.register_buffer("_mean", mean.view(1, 1, 1, 1, 3), persistent=False)
        self.register_buffer("_std", std.view(1, 1, 1, 1, 3), persistent=False)

    def forward(self, image_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = dict(image_dict)
        for key, img in image_dict.items():
            if not torch.is_tensor(img):
                continue
            x = img.float()
            if self.enabled or x.max() > 1.5:
                x = x / 255.0
            if self.normalize and x.shape[-1] == 3:
                x = (x - self._mean) / self._std
            out[key] = x.to(self.output_dtype)
        return out
