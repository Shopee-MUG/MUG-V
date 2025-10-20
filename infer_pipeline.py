#!/usr/bin/env python3
"""
MUG-DiT-10B Inference Pipeline.

This module provides the main inference pipeline for generating videos from text prompts
using MUG-DiT-10B (Multi-scale Unified Generation Diffusion Transformer) with VAE encoding/decoding.

Author: MUG-DiT Team
License: Apache 2.0
"""

import os
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

import torch
from tqdm import tqdm

from mug_v.encoder import MUGVAE, T5
from mug_v.dit import MUGDiT_10B
from mug_v.scheduler import RFlow
from mug_v.utils import (
    get_frame_size,
    get_num_frames,
    save_video,
    set_random_seed,
    get_reference_vae_latent,
)


# Default configuration
class MUGDiTConfig:
    """Configuration class for MUG-DiT-10B inference parameters."""

    # Device and precision settings
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Model paths (these should be updated to your model paths)
    # TODO change to hf path
    vae_pretrained_path: str = (
        "./pretrained_ckpt/MUG-V-inference/vae.pt"
    )
    dit_pretrained_path: str = (
        "./pretrained_ckpt/MUG-V-inference/dit.pt"
    )

    # Video generation settings
    resolution: str = "720p"  # Only 720p is currently supported
    video_length: str = "5s"  # Recommended: "3s" or "5s"
    video_ar_ratio: str = "16:9"  # Height:Width ratio

    # Generation parameters
    cfg_scale: float = 4.0
    num_sampling_steps: int = 25
    fps: int = 30
    aes_score: float = 6.0

    # Random seed for reproducibility
    seed: int = 42

    # Default prompt and paths
    prompt: str = (
        "This video describes a young woman standing in a minimal studio with a warm beige backdrop, "
        "wearing a white cropped top with thin straps and a matching long tiered skirt. "
        "She faces the camera directly with a relaxed posture, "
        "and the lighting is bright and even, giving the scene a soft, neutral appearance. "
        "The background features a seamless beige wall and a smooth floor with no additional props, "
        "creating a simple setting that keeps attention on the outfit. "
        "The main subject is a woman with long curly hair, "
        "dressed in a white spaghetti-strap crop top and a flowing ankle-length skirt with gathered tiers. "
        "She wears black strappy sandals and is positioned centrally in the frame, "
        "standing upright with her arms resting naturally at her sides. "
        "The camera is stationary and straight-on, capturing a full-body shot that keeps her entire figure visible from head to toe. "
        "She appears to hold a calm expression while breathing steadily, occasionally shifting her weight slightly from one foot to the other. "
        "There may be a subtle tilt of the head or a gentle adjustment of her hands, but movements remain small and unhurried throughout the video. "
        "The background remains static with no visible changes, and the framing stays consistent for a clear view of the outfit details."
    )
    reference_image_path: str = "./assets/sample.png"
    output_path: str = "./outputs/sample.mp4"


class MUGDiTPipeline:
    """MUG-DiT-10B inference pipeline.

    This class encapsulates the entire video generation pipeline including
    VAE encoding/decoding, text encoding, and diffusion sampling using
    MUG-DiT-10B (Multi-scale Unified Generation Diffusion Transformer).
    """

    def __init__(self, config: MUGDiTConfig):
        """Initialize the MUG-DiT-10B pipeline.

        Args:
            config: Configuration object containing all necessary parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Initialize models (will be loaded lazily)
        self.vae = None
        self.text_encoder = None
        self.model = None
        self.scheduler = None

    def _load_models(self) -> None:
        """Load all required models."""
        print("Loading models...")

        # Load VAE
        print("Loading VAE...")
        self.vae = (
            MUGVAE(from_pretrained=self.config.vae_pretrained_path)
            .to(self.device, self.dtype)
            .eval()
        )

        # Load text encoder
        print("Loading T5 text encoder...")
        self.text_encoder = T5(
            from_pretrained="DeepFloyd/t5-v1_1-xxl",
            max_model_len=300,
            device=self.device,
        )
        self.text_encoder.t5.model = self.text_encoder.t5.model.eval()

        # Get video dimensions
        frame_size = get_frame_size(self.config.video_ar_ratio)
        num_frames = get_num_frames(self.config.video_length)
        input_size = (num_frames, *frame_size)
        latent_size = self.vae.get_latent_size(input_size)

        # Load DiT model
        print("Loading DiT model...")
        self.model = (
            MUGDiT_10B(
                input_size=latent_size,
                input_channels=self.vae.out_channels,
                text_input_channels=self.text_encoder.output_dim,
                from_pretrained=self.config.dit_pretrained_path,
            )
            .to(self.device, self.dtype)
            .eval()
        )

        # Initialize scheduler
        self.scheduler = RFlow(
            num_sampling_steps=self.config.num_sampling_steps,
            cfg_scale=self.config.cfg_scale,
        )

        print("All models loaded successfully!")

    def generate(
        self,
        prompt: Optional[str] = None,
        reference_image_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a video from text prompt and reference image.

        Args:
            prompt: Text description of the desired video
            reference_image_path: Path to reference image for image-to-video generation
            output_path: Path where the generated video will be saved
            seed: Random seed for reproducible generation
            **kwargs: Additional generation parameters

        Returns:
            Path to the generated video file
        """
        # Use default values if not provided
        prompt = prompt or self.config.prompt
        reference_image_path = (
            reference_image_path or self.config.reference_image_path
        )
        output_path = output_path or self.config.output_path
        seed = seed or self.config.seed

        # Validate inputs
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(
                f"Reference image not found: {reference_image_path}"
            )

        # Load models if not already loaded
        if self.vae is None:
            self._load_models()

        return self._run_inference(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
            seed=seed,
            **kwargs,
        )

    def _run_inference(
        self,
        prompt: str,
        reference_image_path: Union[str, Path],
        output_path: Union[str, Path],
        seed: int,
        **kwargs,
    ) -> str:
        """Run the actual inference process."""
        # Optimize CUDA performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Disable gradient calculation for inference
        torch.set_grad_enabled(False)

        # Set random seed for reproducibility
        set_random_seed(seed)

        # Prepare output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Enhance prompt with aesthetic score if specified
        enhanced_prompt = (
            prompt
            if self.config.aes_score is None
            else f"{prompt} aesthetic score: {self.config.aes_score:.1f}."
        )

        # Get video dimensions
        frame_size = get_frame_size(self.config.video_ar_ratio)
        num_frames = get_num_frames(self.config.video_length)
        input_size = (num_frames, *frame_size)
        latent_size = self.vae.get_latent_size(input_size)

        print(f"Generating video with prompt: {prompt[:100]}...")
        print(f"Reference image: {reference_image_path}")
        print(f"Output path: {output_path}")

        with torch.inference_mode():
            # Calculate VAE temporal compression rate
            vae_temporal_compress_rate = input_size[0] // latent_size[0]

            # Prepare multi-resolution information for the model
            model_args = {
                "fps": torch.tensor(
                    [self.config.fps], device=self.device, dtype=self.dtype
                ),
                "height": torch.tensor(
                    [frame_size[0]], device=self.device, dtype=self.dtype
                ),
                "width": torch.tensor(
                    [frame_size[1]], device=self.device, dtype=self.dtype
                ),
                "num_frames": torch.tensor(
                    [latent_size[0]], device=self.device, dtype=self.dtype
                ),
                "frame_indice": torch.tensor(
                    [range(latent_size[0] + 2)],
                    device=self.device,
                    dtype=torch.long,
                ),
            }

            # Get reference image latent representation
            print("Encoding reference image...")
            ref_latent, ref_img = get_reference_vae_latent(
                reference_image_path, self.vae, frame_size
            )

            # Initialize random noise tensor
            z = torch.randn(
                1,  # batch size is set to 1
                self.vae.out_channels,
                *latent_size,
                device=self.device,
                dtype=self.dtype,
            )

            # Create masks for conditional generation
            # In image-to-video generation, the first frame is conditioned on the reference image
            masks = torch.ones(
                (1, latent_size[0]),
                dtype=torch.float,
                device=z.device,
            )
            # Set mask to 0 for conditional latents (first frame)
            masks[:, :1] = 0
            # Set the first frame latent to the reference image latent
            z[0, :, :1] = ref_latent

            # Run diffusion denoising process
            print("Running diffusion sampling...")
            samples = self.scheduler.sample(
                self.model,
                self.text_encoder,
                z=z,
                prompts=[enhanced_prompt],
                device=self.device,
                additional_args=model_args,
                mask=masks,
            )
            # Decode latent samples to video frames
            print("Decoding video frames...")
            # Post-process video clips
            gen_video_clips = self.vae.decode(
                samples[:, :, 1:].to(self.dtype),
                num_frames=num_frames-1,
            )
            video_clips = torch.cat(
                [
                    ref_img.unsqueeze(0).to(dtype=gen_video_clips.dtype, device=gen_video_clips.device),
                    gen_video_clips,
                ],
                dim=2,
            )[0]
            # Save the generated video
            print(f"Saving video to {output_path}...")
            final_path = save_video(
                video_clips,
                save_path=output_path,
            )

            print(f"Video generation completed! Saved to: {final_path}")
            return final_path


def main() -> None:
    """Main function for command-line usage."""
    config = MUGDiTConfig()
    pipeline = MUGDiTPipeline(config)

    try:
        output_path = pipeline.generate()
        print(f"\nGeneration successful! Video saved to: {output_path}")
    except Exception as e:
        print(f"\nError during generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
