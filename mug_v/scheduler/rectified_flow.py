import torch
from tqdm import tqdm
import torch
import torch.nn.functional as F


def timestep_transform(
    timepoint,
    height,
    width,
    num_frames,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    timepoint = timepoint / num_timesteps
    ratio_space = (height * width / base_resolution).sqrt()
    ratio_time = (num_frames / base_num_frames).sqrt()
    ratio = ratio_space * ratio_time * scale
    new_timepoint = ratio * timepoint / (1 + (ratio - 1) * timepoint)
    return new_timepoint * num_timesteps


class RFlow:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_timestep_transform=True,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_timestep_transform = use_timestep_transform

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = (
            timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )
        timepoints = timepoints.repeat(
            1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]
        )

        return timepoints * original_samples + (1 - timepoints) * noise

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        if guidance_scale is None:
            guidance_scale = self.cfg_scale
        negative = "ugly, blurry, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

        n = len(prompts)
        prompts = prompts + [negative] * n
        model_args = text_encoder.encode(prompts)
        if additional_args is not None:
            model_args.update(additional_args)

        timesteps = [
            (1.0 - i / self.num_sampling_steps) * self.num_timesteps
            for i in range(self.num_sampling_steps)
        ]
        timesteps = [
            torch.tensor([t] * z.shape[0], device=device) for t in timesteps
        ]
        timesteps = [
            timestep_transform(
                t,
                height=additional_args["height"],
                width=additional_args["width"],
                num_frames=additional_args["num_frames"],
                num_timesteps=self.num_timesteps,
            )
            for t in timesteps
        ]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)

        initial_noise_t = timesteps[0]
        x0 = z.clone()
        x_noise = self.add_noise(x0, torch.randn_like(x0), initial_noise_t)
        z = torch.where(
            mask[:, None, :, None, None].to(torch.bool), x_noise, x0
        )

        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(
                    mask_add_noise[:, None, :, None, None], x_noise, x0
                )
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)

            # get v_pred
            delta = pred_cond - pred_uncond
            v_pred = pred_uncond + guidance_scale * delta

            # update z
            dt = (
                timesteps[i] - timesteps[i + 1]
                if i < len(timesteps) - 1
                else timesteps[i]
            )
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]
            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
