import os

import torch
import torch.nn as nn
from .vae2d import AutoencoderKL
from .vae_temporal import MUGVAETemporal
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel


class MUGAutoencoderKL(nn.Module):
    def __init__(
        self,
        from_pretrained=None,
        micro_batch_size=None,
        cache_dir=None,
        local_files_only=False,
        subfolder=None,
        scaling_factor=0.18215,
    ):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            subfolder=subfolder,
            use_quant_conv=False,
            use_post_quant_conv=False,
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size
        self.scaling_factor = scaling_factor

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).mul(self.scaling_factor)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).mul(self.scaling_factor)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x, **kwargs):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / self.scaling_factor).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / self.scaling_factor).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            latent_size.append(
                input_size[i] // self.patch_size[i]
                if input_size[i] is not None
                else None
            )
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class MUGAutoencoderPipelineConfig(PretrainedConfig):
    model_type = "VideoAutoencoderPipeline"

    def __init__(
        self,
        from_pretrained=None,
        freeze_vae_2d=False,
        micro_frame_size=None,
        micro_batch_size=None,
        shift=0.0,
        scale=1.0,
        **kwargs,
    ):
        self.from_pretrained = from_pretrained
        self.freeze_vae_2d = freeze_vae_2d
        self.micro_frame_size = micro_frame_size
        self.micro_batch_size = micro_batch_size
        self.shift = shift
        self.scale = scale
        super().__init__(**kwargs)


class MUGAutoencoderPipeline(PreTrainedModel):
    config_class = MUGAutoencoderPipelineConfig

    def __init__(self, config):
        super().__init__(config=config)
        self.spatial_vae = MUGAutoencoderKL(
            from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            subfolder="vae",
            micro_batch_size=config.micro_batch_size,
        )
        self.temporal_vae = MUGVAETemporal(
            in_out_channels=512,
            latent_embed_dim=24,
            embed_dim=24,
            filters=128,
            num_res_blocks=4,
            channel_multipliers=(1, 2, 2, 4),
            temporal_downsample=(True, True, True),
        )
        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size(
            [config.micro_frame_size, None, None]
        )[0]
        if config.freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels
        # normalization parameters
        scale = torch.tensor(config.scale)
        shift = torch.tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

        self.logvar = nn.Parameter(torch.ones(size=()) * 4.0)
        # self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        if self.micro_frame_size is None:
            posterior = self.temporal_vae.encode(x_z)
            z = posterior.sample()
        else:
            if x_z.shape[2] % self.temporal_vae.time_downsample_factor == 1:
                x_z = torch.cat(
                    [
                        x_z[:, :, 0, :, :]
                        .unsqueeze(2)
                        .repeat(
                            1,
                            1,
                            self.temporal_vae.time_downsample_factor - 1,
                            1,
                            1,
                        ),
                        x_z,
                    ],
                    dim=2,
                )
                assert (
                    x_z.shape[2] % self.temporal_vae.time_downsample_factor
                    == 0
                ), f"{x_z.shape} {self.temporal_vae.time_downsample_factor}"
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i : i + self.micro_frame_size]
                posterior = self.temporal_vae.encode(x_z_bs)
                z_list.append(posterior.sample())
            z = torch.cat(z_list, dim=2)

        return (z - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i : i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(
                    z_bs, num_frames=min(self.micro_frame_size, num_frames)
                )
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z)

        return x

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(
                self.spatial_vae.get_latent_size(input_size)
            )
        else:
            sub_input_size = [
                self.micro_frame_size,
                input_size[1],
                input_size[2],
            ]
            sub_latent_size = self.temporal_vae.get_latent_size(
                self.spatial_vae.get_latent_size(sub_input_size)
            )
            sub_latent_size[0] = sub_latent_size[0] * (
                input_size[0] // self.micro_frame_size
            )
            remain_temporal_size = [
                input_size[0] % self.micro_frame_size,
                None,
                None,
            ]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(
                    remain_temporal_size
                )
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def MUGVAE(
    micro_batch_size=4,
    micro_frame_size=48,
    from_pretrained=None,
    local_files_only=False,
    freeze_vae_2d=False,
    mean=(
        -0.0194091796875,
        0.09619140625,
        -0.796875,
        0.1591796875,
        -0.2578125,
        0.359375,
        -0.3203125,
        -0.287109375,
        -0.0069580078125,
        0.3046875,
        0.310546875,
        -0.451171875,
        -0.1728515625,
        0.369140625,
        0.2177734375,
        -3.075599670410156e-05,
        0.1630859375,
        -0.267578125,
        -0.1962890625,
        -0.1298828125,
        -0.28515625,
        -0.515625,
        0.5859375,
        -0.34375,
    ),
    std=(
        0.8817,
        0.6523,
        0.9152,
        1.2117,
        3.3516,
        0.7528,
        0.8177,
        0.8637,
        0.9075,
        2.8875,
        0.8980,
        1.1202,
        1.0003,
        2.5163,
        0.6652,
        1.2573,
        0.7279,
        1.0777,
        1.5159,
        0.8680,
        1.1859,
        1.0484,
        2.4750,
        1.5881,
    ),
):

    kwargs = dict(
        freeze_vae_2d=freeze_vae_2d,
        micro_frame_size=micro_frame_size,
        micro_batch_size=micro_batch_size,
        shift=mean,
        scale=std,
    )

    config = MUGAutoencoderPipelineConfig(**kwargs)
    model = MUGAutoencoderPipeline(config)

    if from_pretrained:
        state_dict = torch.load(from_pretrained, weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=True
        )
        print("MUGVAE Missing keys: %s", missing_keys)
        print("MUGVAE Unexpected keys: %s", unexpected_keys)
    return model
