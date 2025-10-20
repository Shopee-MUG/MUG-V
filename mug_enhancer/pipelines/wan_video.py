from diffsynth.models import ModelManager
from diffsynth.models.model_manager import load_model_from_single_file
from diffsynth.models.utils import load_state_dict
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.wan_video_text_encoder import WanTextEncoder
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.schedulers.flow_match import FlowMatchScheduler
from diffsynth.pipelines.base import BasePipeline
from diffsynth.prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
import math

from diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from diffsynth.models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from diffsynth.models.wan_video_dit import WanLayerNorm, WanRMSNorm
from diffsynth.models.wan_video_vae import RMS_norm, CausalConv3d, Upsample


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None, sigma_max=1.0, sigma_min=0.0):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_max=sigma_max, sigma_min=sigma_min, extra_one_step=True)
        self.scheduler_for_predict = FlowMatchScheduler(shift=1.0, sigma_max=0.3, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, sigma_max=1.0, sigma_min=0.0):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype, sigma_max=sigma_max, sigma_min=sigma_min)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            image = self.preprocess_image(image.resize((width, height))).to(self.device)
            clip_context = self.image_encoder.encode_image([image])
            msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = self.vae.encode([torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)], device=self.device)[0]
            y = torch.concat([msk, y])
        return {"clip_fea": clip_context, "y": [y]}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {"seq_len": latents.shape[2] * latents.shape[3] * latents.shape[4] // 4}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32).to(self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=noise.dtype, device=noise.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise
        
        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit"])
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            for progress_id, timestep in enumerate(
                progress_bar_cmd(self.scheduler.timesteps)
            ):
                timestep = timestep.unsqueeze(0).to(
                    dtype=torch.float32, device=self.device
                )

                # Inference
                noise_pred_posi = self.dit(
                    latents,
                    timestep=timestep,
                    **prompt_emb_posi,
                    **image_emb,
                    **extra_input,
                )
                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit(
                        latents,
                        timestep=timestep,
                        **prompt_emb_nega,
                        **image_emb,
                        **extra_input,
                    )
                    noise_pred = noise_pred_nega + cfg_scale * (
                        noise_pred_posi - noise_pred_nega
                    )
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(
                    noise_pred, self.scheduler.timesteps[progress_id], latents
                )

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames

    @torch.no_grad()
    def enhance_video(
            self,
            prompt,
            negative_prompt="",
            input_video=None,
            denoising_strength=1.0,
            seed=None,
            rand_device="cpu",
            height=480,
            width=832,
            num_frames=81,
            cfg_scale=5.0,
            num_inference_steps=50,
            sigma_shift=5.0,
            progress_bar_cmd=tqdm,
    ):
        batch_size = len(prompt)
        # Parameter check
        height, width = self.check_resize_height_width(height, width)

        # Scheduler
        self.scheduler_for_predict.set_timesteps(num_inference_steps, denoising_strength, shift=sigma_shift)

        print(f'{self.scheduler_for_predict.timesteps=}')

        # Initialize noise
        noise = self.generate_noise((1, 16, math.ceil(num_frames / 4) , height // 8, width // 8), seed=seed,
                                    device=rand_device, dtype=torch.float32).to(self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            latents_origin = self.encode_video(input_video, tiled=False).to(dtype=noise.dtype, device=noise.device)
            latents = self.scheduler_for_predict.add_noise(latents_origin, noise, timestep=self.scheduler_for_predict.timesteps[0])
        else:
            latents_origin = None # cause error, might be zero tensor
            latents = noise

        first_frame_latent = latents_origin[:, :, :1].clone()

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit"])
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler_for_predict.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)

                latents[:, :, :1] = first_frame_latent

                # Inference
                noise_pred_posi = self.dit(latents, y=latents_origin, timestep=timestep, **prompt_emb_posi, **extra_input,
                                            opensora_i2v_enable=True,
                                            timestep_t0=torch.zeros_like(
                                                timestep,
                                                dtype=timestep.dtype,
                                                device=timestep.device,
                                            ))
                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit(latents, y=latents_origin, timestep=timestep, **prompt_emb_nega, **extra_input,
                                            opensora_i2v_enable=True,
                                            timestep_t0=torch.zeros_like(
                                                timestep,
                                                dtype=timestep.dtype,
                                                device=timestep.device,
                                            ))
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler_for_predict.step(noise_pred, self.scheduler_for_predict.timesteps[progress_id], latents)

        latents[:, :, :1] = first_frame_latent

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, tiled=False)
        self.load_models_to_device([])
        frames_list = [self.tensor2video(frames[i]) for i in range(batch_size)]
        return frames_list, frames
