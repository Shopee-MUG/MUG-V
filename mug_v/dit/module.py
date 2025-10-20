import torch
import torch.nn as nn
import xformers.ops
from einops import rearrange
import torch.nn.functional as F
import math
from flash_attn import flash_attn_func


def precompute_freqs_cis_for_3drope(
    dim: int,
    space_end: int,
    time_end: int,
    theta: float = 10000.0,
    scale_factor: float = 1.0,
    scale_watershed: float = 1.0,
    timestep: float = 1.0,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with
    given dimensions.

    This function calculates a frequency tensor with complex exponentials
    using the given dimension 'dim' and the end index 'end'. The 'theta'
    parameter scales the frequencies. The returned tensor contains complex
    values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        space_end (int): End index for precomputing frequencies.
        time_end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation.
            Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex
            exponentials.
    """
    if timestep < scale_watershed:
        linear_factor = scale_factor
        ntk_factor = 1.0
    else:
        linear_factor = 1.0
        ntk_factor = scale_factor

    theta = theta * ntk_factor

    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)].float() / dim))
        / linear_factor
    )
    space_t = torch.arange(space_end, device=freqs.device, dtype=torch.float)
    space_freqs = torch.outer(space_t, freqs).float()
    space_freqs_cis = torch.polar(
        torch.ones_like(space_freqs), space_freqs
    )  # complex64

    time_t = torch.arange(time_end, device=freqs.device, dtype=torch.float)
    time_freqs = torch.outer(time_t, freqs).float()
    time_freqs_cis = torch.polar(
        torch.ones_like(time_freqs), time_freqs
    )  # complex64
    return (space_freqs_cis, time_freqs_cis)


def ada_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def modulate(norm_layer, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_layer(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
        enable_flash_attn: bool = False,
        qk_norm_legacy: bool = False,
        temporal=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_norm_legacy = qk_norm_legacy

    def apply_rotary_emb(
        self,
        x_in: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(
                x_in.float().reshape(*x_in.shape[:-1], -1, 2)
            )
            # print(x.shape, freqs_cis.shape)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    def forward(
        self, x, num_frames=None, frame_indices=None, freq_cis=None
    ) -> torch.Tensor:
        B, C = x.shape[0], x.shape[-1]
        qkv = self.qkv(x)
        qkv_shape = (B, -1, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)  # B, seq, num_heads, head_dim

        q, k = self.q_norm(q), self.k_norm(k)
        if freq_cis.ndim == 3:
            freq_cis = freq_cis.unsqueeze(2)
        else:
            freq_cis = freq_cis.flatten(1, 3).unsqueeze(2)
        q = self.apply_rotary_emb(q, freqs_cis=freq_cis)
        k = self.apply_rotary_emb(k, freqs_cis=freq_cis)

        x = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.scale,
        )
        x_output_shape = (B, -1, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=False,
        norm_layer: nn.Module = RMSNorm,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        q, k = self.q_norm(q), self.k_norm(k)

        attn_bias = None
        if mask is not None:
            attn_bias = (
                xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    [N] * B, mask
                )
            )
        x = xformers.ops.memory_efficient_attention(
            q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
        )

        x = x.view(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Multilayer Perceptron (MLP)"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = (
            act_layer if act_layer is not None else nn.GELU(approximate="tanh")
        )
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TextEmbedder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
    ):
        super().__init__()
        self.y_proj = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=hidden_features,
        )
        self.pre_norm = nn.LayerNorm(
            in_features, eps=1e-6, elementwise_affine=False
        )
        self.post_norm = nn.LayerNorm(
            hidden_features, eps=1e-6, elementwise_affine=False
        )

    def forward(self, text):
        caption = self.pre_norm(text)
        caption = self.y_proj(caption)
        caption = self.post_norm(caption)
        return caption


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_patch,
        output_channels,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size, num_patch * output_channels, bias=True
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        assert x.shape[1] == T * S, x.shape
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(
            2, dim=1
        )
        x_noise = ada_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (
                self.scale_shift_table[None] + t0[:, None]
            ).chunk(2, dim=1)
            x_zero = ada_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x_noise, x_zero, T, S)
        else:
            x = x_noise
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FpsEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(
            hidden_size=hidden_size,
            frequency_embedding_size=frequency_embedding_size,
        )
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(
            self.dtype
        )
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(
            s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim
        )
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        input_channels (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        input_channels=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(
                x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1])
            )
        if D % self.patch_size[0] != 0:
            x = F.pad(
                x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0])
            )

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x
