#!/usr/bin/env python3
"""
Utility functions for MUG-DiT-10B inference pipeline.

This module provides essential utility functions for video processing,
including random seed management, image processing, and video saving.

Author: MUG-DiT Team
License: Apache 2.0
"""

import random
from pathlib import Path
from typing import Optional, Union, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torchvision.io import write_video


def set_random_seed(seed: Optional[int] = None) -> int:
    """Set random seed for reproducible results across all random number generators.

    Args:
        seed: Random seed value. If None, uses a random seed.

    Returns:
        The seed value that was set.

    Raises:
        ValueError: If seed is not a valid integer.
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


# Supported aspect ratios for 720p generation
# Format: "height:width" -> (height_pixels, width_pixels)
ASPECT_RATIO_720P: Dict[str, Tuple[int, int]] = {
    "9:16": (720, 1280),  # Portrait (mobile)
    "3:4": (832, 1110),  # Portrait
    "1:1": (960, 960),  # Square
    "4:3": (1108, 832),  # Standard
    "16:9": (1280, 720),  # Widescreen
}


def get_frame_size(ar_ratio: str, resolution: str = "720p") -> Tuple[int, int]:
    """Get frame size based on aspect ratio and resolution.

    Args:
        ar_ratio: Aspect ratio in "height:width" format (e.g., "16:9")
        resolution: Video resolution (currently only "720p" is supported)

    Returns:
        Tuple of (height, width) in pixels

    Raises:
        ValueError: If the aspect ratio is not supported
        NotImplementedError: If resolution other than "720p" is requested
    """
    if resolution != "720p":
        raise NotImplementedError(
            f"Only 720p resolution is currently supported, got {resolution}"
        )

    if ar_ratio not in ASPECT_RATIO_720P:
        supported_ratios = ", ".join(ASPECT_RATIO_720P.keys())
        raise ValueError(
            f"Unsupported aspect ratio '{ar_ratio}'. "
            f"Supported ratios: {supported_ratios}"
        )

    return ASPECT_RATIO_720P[ar_ratio]


# Supported video lengths and their corresponding frame counts
NUM_FRAMES_MAP: Dict[str, int] = {
    "5s": 105,  # 5 seconds at 21 FPS effective rate
    "3s": 73,  # 3 seconds at ~24 FPS effective rate
}


def get_num_frames(video_len: str) -> int:
    """Get the number of frames for a given video length.

    Args:
        video_len: Video length string (e.g., "3s", "5s")

    Returns:
        Number of frames for the specified video length

    Raises:
        ValueError: If the video length is not supported
    """
    if video_len not in NUM_FRAMES_MAP:
        supported_lengths = ", ".join(NUM_FRAMES_MAP.keys())
        raise ValueError(
            f"Unsupported video length '{video_len}'. "
            f"Supported lengths: {supported_lengths}"
        )

    return NUM_FRAMES_MAP[video_len]


def read_image_from_path(
    path: Union[str, Path], image_size: Tuple[int, int]
) -> torch.Tensor:
    """Read and preprocess an image from file path.

    Args:
        path: Path to the image file
        image_size: Target size as (height, width)

    Returns:
        Preprocessed image tensor in CTHW format (C=3, T=1, H, W)

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be processed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    def resize_crop_to_fill(
        pil_image: Image.Image, image_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize and crop image to fill the target size while maintaining aspect ratio.

        Args:
            pil_image: PIL Image object
            image_size: Target size as (height, width)

        Returns:
            Processed PIL Image
        """
        w, h = pil_image.size  # PIL uses (W, H) format
        th, tw = image_size

        # Calculate scaling ratios
        rh, rw = th / h, tw / w

        if rh > rw:
            # Scale based on height, crop width
            sh, sw = th, round(w * rh)
            image = pil_image.resize((sw, sh), Image.BICUBIC)
            i = 0
            j = int(round((sw - tw) / 2.0))
        else:
            # Scale based on width, crop height
            sh, sw = round(h * rw), tw
            image = pil_image.resize((sw, sh), Image.BICUBIC)
            i = int(round((sh - th) / 2.0))
            j = 0

        # Convert to numpy for cropping
        arr = np.array(image)

        # Validate crop boundaries
        if i + th > arr.shape[0] or j + tw > arr.shape[1]:
            raise ValueError(
                f"Crop boundaries exceed image dimensions: "
                f"crop=({i}, {j}, {i+th}, {j+tw}), image_shape={arr.shape}"
            )

        return Image.fromarray(arr[i : i + th, j : j + tw])

    try:
        image = pil_loader(str(path))
    except Exception as e:
        raise ValueError(f"Failed to load image from {path}: {e}")

    # Define preprocessing transforms
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: resize_crop_to_fill(pil_image, image_size)
            ),
            transforms.ToTensor(),  # Converts to [0, 1] and changes to CHW
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),  # Normalize to [-1, 1]
        ]
    )

    # Apply transforms
    image = transform(image)

    # Add temporal dimension: C H W -> C T H W (T=1 for single image)
    video = image.unsqueeze(0)

    # Rearrange to C T H W format
    video = video.permute(1, 0, 2, 3)

    return video


def get_reference_vae_latent(
    path: Union[str, Path], vae: torch.nn.Module, image_size: Tuple[int, int]
) -> torch.Tensor:
    """Get VAE latent representation of reference image for conditional generation.

    Args:
        path: Path to the reference image
        vae: VAE model for encoding
        image_size: Target image size as (height, width)

    Returns:
        VAE latent tensor for the reference image

    Raises:
        FileNotFoundError: If the image file doesn't exist
        RuntimeError: If VAE encoding fails
    """
    if not isinstance(vae, torch.nn.Module):
        raise TypeError("VAE must be a PyTorch module")

    # Read and preprocess the image
    ori_img = read_image_from_path(path, image_size)

    # Repeat the image 8 times along temporal dimension
    # (8 is the VAE temporal compression rate)
    img = ori_img.repeat(1, 8, 1, 1)  # C T H W -> C (T*8) H W

    # Add batch dimension and move to VAE device
    img_batch = img.unsqueeze(0).to(vae.device, vae.dtype)

    try:
        # Encode using VAE
        with torch.no_grad():
            latent = vae.encode(img_batch)
    except Exception as e:
        raise RuntimeError(f"VAE encoding failed: {e}")

    # Remove batch dimension
    return latent.squeeze(0), ori_img


def save_video(
    vid: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    fps: int = 20,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1, 1),
) -> str:
    """Save a video tensor to file.

    Args:
        vid: Video tensor of shape [C, T, H, W]
        save_path: Output path for the video file. If None, uses "output.mp4"
        fps: Frames per second for the output video
        normalize: Whether to normalize the video tensor
        value_range: Input value range for normalization (min, max)

    Returns:
        Path to the saved video file

    Raises:
        ValueError: If video tensor has invalid shape or parameters
        RuntimeError: If video saving fails
    """
    if not isinstance(vid, torch.Tensor):
        raise TypeError("Video must be a torch.Tensor")

    if vid.ndim != 4:
        raise ValueError(
            f"Video tensor must be 4D [C, T, H, W], got shape {vid.shape}"
        )

    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")

    # Set default save path
    if save_path is None:
        save_path = "output.mp4"

    save_path = Path(save_path)

    # Ensure .mp4 extension
    if save_path.suffix.lower() != ".mp4":
        save_path = save_path.with_suffix(".mp4")

    # Create output directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Clone to avoid modifying original tensor
    vid = vid.clone()

    # Normalize if requested
    if normalize:
        low, high = value_range
        if low >= high:
            raise ValueError(
                f"Invalid value range: {value_range}. Low must be < high."
            )

        vid.clamp_(min=low, max=high)
        vid.sub_(low).div_(max(high - low, 1e-5))

    # Convert to uint8 format for video saving
    vid = (
        vid.mul(255)
        .add_(0.5)  # Round by adding 0.5 before truncation
        .clamp_(0, 255)
        .permute(1, 2, 3, 0)  # [C, T, H, W] -> [T, H, W, C]
        .to("cpu", torch.uint8)
    )

    try:
        write_video(
            str(save_path),
            vid,
            fps=fps,
            video_codec="h264",
            options={"crf": "17"},  # High quality encoding
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save video to {save_path}: {e}")

    return str(save_path)
