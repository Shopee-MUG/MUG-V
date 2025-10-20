# modified based on Open-Sora

import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .module import (
    MLP,
    Attention,
    CrossAttention,
    precompute_freqs_cis_for_3drope,
    ada_modulate,
    FinalLayer,
    TimestepEmbedder,
    FpsEmbedder,
    PatchEmbed3D,
    TextEmbedder,
)


class MUGDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qk_norm=False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
        )
        self.cross_attn_norm = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=True
        )
        self.cross_attn = CrossAttention(
            hidden_size, num_heads, qk_norm=qk_norm
        )
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

    def t_mask_select(self, x_mask, expand_x, masked_x, T, S):
        assert expand_x.shape[1] == T * S + 2, expand_x.shape
        expand_x = torch.where(x_mask[:, :, None], expand_x, masked_x)
        return expand_x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
        freq_cis=None,
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            (
                shift_msa_zero,
                scale_msa_zero,
                gate_msa_zero,
                shift_mlp_zero,
                scale_mlp_zero,
                gate_mlp_zero,
            ) = (self.scale_shift_table[None] + t0.reshape(B, 6, -1)).chunk(
                6, dim=1
            )
        # modulate (attention)
        x_m = ada_modulate(self.norm(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = ada_modulate(
                self.norm(x), shift_msa_zero, scale_msa_zero
            )
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # attention
        f_b, f_t, f_c = freq_cis.shape

        x_m = self.attn(x_m, freq_cis=freq_cis)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + x_m_s

        # cross attention
        x = x + self.cross_attn(self.cross_attn_norm(x), y, mask)
        # modulate (MLP)
        x_m = ada_modulate(self.norm(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = ada_modulate(
                self.norm(x), shift_mlp_zero, scale_mlp_zero
            )
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + x_m_s

        return x


class MUGDiT(nn.Module):
    def __init__(
        self,
        input_size=(None, None, None),
        input_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        text_input_channels=4096,
        qk_norm=True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels * 2
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.qk_norm = qk_norm

        self.freqs_cis = precompute_freqs_cis_for_3drope(
            dim=self.hidden_size // self.num_heads,
            space_end=384,
            time_end=480,  # support videos' length up to 1min
        )

        self.x_embedder = PatchEmbed3D(
            self.patch_size, self.input_channels, self.hidden_size
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.fps_embedder = FpsEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True),
        )
        self.y_embedder = TextEmbedder(
            in_features=text_input_channels,
            hidden_features=self.hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                MUGDiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_norm=self.qk_norm,
                )
                for i in range(self.depth)
            ]
        )

        self.start_token = nn.Parameter(torch.empty(self.hidden_size))
        self.end_token = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.start_token, std=0.02)
        nn.init.normal_(self.end_token, std=0.02)

        # final layer
        self.final_layer = FinalLayer(
            self.hidden_size,
            np.prod(self.patch_size),
            self.output_channels,
        )

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = (
                y.squeeze(1)
                .masked_select(mask.unsqueeze(-1) != 0)
                .view(1, -1, self.hidden_size)
            )
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def get_freq_cis(self, H, W, expand_frame_indice):
        B, T = expand_frame_indice.shape
        dim = self.freqs_cis[0].shape[-1]
        assert (
            self.freqs_cis[0].shape[-1]
            == self.freqs_cis[1].shape[-1]
            == self.hidden_size // self.num_heads // 6
        )
        h_freq_cis = (
            self.freqs_cis[0][:H]
            .reshape(1, 1, H, 1, dim)
            .repeat(B, T, 1, W, 1)
        )
        w_freq_cis = (
            self.freqs_cis[0][:W]
            .reshape(1, 1, 1, W, dim)
            .repeat(B, T, H, 1, 1)
        )
        expand_frame_indice = expand_frame_indice.clamp(0, 480 - 1)
        t_freq_cis = (
            self.freqs_cis[1][expand_frame_indice]
            .reshape(B, T, 1, 1, dim)
            .repeat(1, 1, H, W, 1)
        )
        freq_cis = torch.cat([h_freq_cis, w_freq_cis, t_freq_cis], dim=-1)
        ori_freq_cis = freq_cis[:, 1:-1, :, :, :].flatten(1, 3)
        start_token_cis = freq_cis[:, 0, 0, 0, :].unsqueeze(1)
        end_token_cis = freq_cis[:, -1, -1, -1, :].unsqueeze(1)
        return torch.cat([start_token_cis, ori_freq_cis, end_token_cis], dim=1)

    def forward(
        self,
        x,
        timestep,
        y,
        mask=None,
        x_mask=None,
        fps=None,
        frame_indice=None,
        **kwargs,
    ):
        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]

        # pad start and end token
        expand_x = torch.cat(
            [
                self.start_token.view(1, 1, -1).repeat(B, 1, 1),
                x,
                self.end_token.view(1, 1, -1).repeat(B, 1, 1),
            ],
            dim=1,
        )

        if x_mask is not None:
            expand_x_mask = x_mask.unsqueeze(2).repeat(1, 1, S).reshape(B, -1)
            expand_x_mask = torch.cat(
                [
                    torch.zeros(
                        B,
                        dtype=expand_x_mask.dtype,
                        device=expand_x_mask.device,
                    ).reshape(-1, 1),
                    expand_x_mask,
                    torch.zeros(
                        B,
                        dtype=expand_x_mask.dtype,
                        device=expand_x_mask.device,
                    ).reshape(-1, 1),
                ],
                dim=1,
            )

        # === freq_cis for 3D Rope ===
        frame_indice = frame_indice.detach().cpu()
        freq_cis = self.get_freq_cis(H, W, frame_indice)
        if freq_cis.shape[0] != x.shape[0]:
            assert freq_cis.shape[0] == x.shape[0] // 2
            freq_cis = freq_cis.repeat(2, 1, 1)
        freq_cis = freq_cis.to(x.device)

        # === blocks ===
        for block in self.blocks:
            expand_x = block(
                expand_x,
                y,
                t_mlp,
                y_lens,
                expand_x_mask,
                t0_mlp,
                T,
                S,
                freq_cis,
            )

        # === final layer ===
        x = expand_x[:, 1:-1, :]
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        return x.to(torch.float32)

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.output_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


def MUGDiT_10B(from_pretrained=None, **kwargs):
    model = MUGDiT(
        depth=56,
        hidden_size=3456,
        patch_size=(1, 2, 2),
        num_heads=48,
        **kwargs,
    )
    if from_pretrained is not None:
        state_dict = torch.load(from_pretrained, weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=True
        )
        print("MUGDiT_10B Missing keys: %s", missing_keys)
        print("MUGDiT_10B Unexpected keys: %s", unexpected_keys)
    return model
