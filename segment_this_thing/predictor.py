# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import torch

from .foveation import Foveator
from .model import SegmentThisThing

from .utils import (
    get_centered_crop,
    get_crop_bounds,
    get_imagenet_mean,
    get_imagenet_std,
)


class SegmentThisThingPredictor:
    def __init__(self, model: SegmentThisThing, foveator: Foveator):
        self.model = model
        self.foveator = foveator

    def get_prediction(
        self,
        image: torch.Tensor,
        foveation_center: torch.Tensor,
        return_foveation: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        assert image.dtype == torch.uint8, "Image should be a byte Tensor"
        assert image.shape[-1] == 3, "Expected 3-Channel RGB image"
        assert image.ndim == 3, "Image should be a 3D Tensor (H, W, C)"

        assert foveation_center.ndim == 1, "Foveation center should be a 1D Tensor"
        assert len(foveation_center) == 2, "Foveation center should be a 2D point"

        device = image.device

        crop_bounds = get_crop_bounds(
            foveation_center, self.foveator.get_pattern_bounds_size()
        ).to(device)
        crop = get_centered_crop(image, crop_bounds)

        valid_token_mask = self.foveator.get_in_bounds_tokens(
            torch.tensor(image.shape[1::-1], device=device), crop_bounds.to(device)
        ).unsqueeze(0)

        foveation = self.foveator.extract_foveated_image(
            crop.permute(2, 0, 1).to(device)
        )

        masks, ious = self.model(
            (foveation.unsqueeze(0) - 255.0 * get_imagenet_mean(device).view(3, 1, 1))
            / (255.0 * get_imagenet_std(device).view(3, 1, 1)),
            valid_token_mask,
        )

        if return_foveation:
            return masks.squeeze(0), ious.squeeze(0), foveation
        else:
            return masks.squeeze(0), ious.squeeze(0)
