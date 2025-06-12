# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch


def get_imagenet_mean(device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Returns the mean values for the ImageNet dataset for each channel (RGB).

    These mean values are commonly used for normalizing images in deep learning
    models trained on the ImageNet dataset.

    Args:
        device (Optional[torch.device]): The device on which the tensor should be
        allocated. If None, the tensor will be allocated on the default device.

    Returns:
        torch.Tensor: A tensor containing the mean values for the ImageNet dataset
        for each channel (RGB).
    """
    return torch.tensor([0.485, 0.456, 0.406], device=device)


def get_imagenet_std(device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Returns the standard deviation values for the ImageNet dataset for each channel (RGB).

    These standard deviation values are commonly used for normalizing images in deep learning
    models trained on the ImageNet dataset.

    Args:
        device (Optional[torch.device]): The device on which the tensor should be
        allocated. If None, the tensor will be allocated on the default device.

    Returns:
        torch.Tensor: A tensor containing the standard deviation values for the ImageNet dataset
        for each channel (RGB).
    """
    return torch.tensor([0.229, 0.224, 0.225], device=device)


def get_crop_bounds(center: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Calculates the lower and upper bounds for cropping an image centered around a given point.

    Args:
        center (torch.Tensor): A tensor representing the center point of the crop.
        crop_size (int): The size of the crop.

    Returns:
        torch.Tensor: The bounds for the crop, stored as a 2x2 tensor with the format
                      [lower_bound, upper_bound], where each bound is a tensor of the
                      the form [x, y].
    """
    lower_corner = (center - crop_size / 2).int()
    upper_corner = lower_corner + crop_size
    return torch.stack([lower_corner, upper_corner])


def get_centered_crop(image: torch.Tensor, crop_bounds: torch.Tensor) -> torch.Tensor:
    """
    Extracts a centered crop from an image based on specified crop bounds, and pads
    the crop with a filler if necessary to maintain the specified crop size. The filler
    used is the ImageNet mean pixel value.

    Args:
        image (torch.Tensor): A 3D tensor representing the image to be cropped.
                              The image must have 3 channels (RGB).
        crop_bounds (torch.Tensor): A tensor containing the bounds of the crop, stored
                                    as a 2x2 tensor with the format [lower_bound,
                                    upper_bound], where each bound is a tensor of the
                                    form [x, y].

    Returns:
        torch.Tensor: A tensor representing the cropped image, potentially padded
                      with the ImageNet mean pixel value to maintain the crop size.
    """
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D, got {image.shape}")
    if image.shape[-1] != 3:
        raise ValueError(f"Image must have 3 channels, got {image.shape[-1]}")

    device = image.device
    if device != crop_bounds.device:
        raise RuntimeError("Expected image and crop_bounds to be on the same device.")

    lower_corner, upper_corner = crop_bounds
    lower_pad = (-lower_corner).clamp(min=0)
    upper_pad = (upper_corner - torch.tensor(image.shape[1::-1], device=device)).clamp(
        min=0
    )

    lower_corner = lower_corner + lower_pad
    upper_corner = upper_corner - upper_pad
    crop = image[
        lower_corner[1] : upper_corner[1], lower_corner[0] : upper_corner[0], :
    ]

    filler = (255.0 * get_imagenet_mean()).round().byte().view(1, 1, 3).to(device)

    if lower_pad[0] > 0:
        crop = torch.cat(
            [
                filler.view(1, 1, 3).expand(crop.shape[0], lower_pad[0], -1),
                crop,
            ],
            dim=1,
        )
    if upper_pad[0] > 0:
        crop = torch.cat(
            [
                crop,
                filler.view(1, 1, 3).expand(crop.shape[0], upper_pad[0], -1),
            ],
            dim=1,
        )
    if lower_pad[1] > 0:
        crop = torch.cat(
            [filler.view(1, 1, 3).expand(lower_pad[1], crop.shape[1], -1), crop]
        )
    if upper_pad[1] > 0:
        crop = torch.cat(
            [crop, filler.view(1, 1, 3).expand(upper_pad[1], crop.shape[1], -1)]
        )

    return crop
