"""Cosmos-Predict2-2B DiT in MLX — transpose of diffusers CosmosTransformer3DModel.

Structure is kept isomorphic to transformer_cosmos.py (same class/method names) for
1:1 parity diffing. Anima config: extra_pos_embed_type=None (RoPE only), no img_context,
no crossattn_projection. Self-attn (attn1) gets RoPE; cross-attn (attn2) does not.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn


@dataclass
class CosmosDiTConfig:
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 28
    mlp_ratio: float = 4.0
    text_embed_dim: int = 1024
    adaln_lora_dim: int = 256
    max_size: tuple = (128, 240, 240)
    patch_size: tuple = (1, 2, 2)
    rope_scale: tuple = (2.0, 1.0, 1.0)
    base_fps: int = 24
    concat_padding_mask: bool = True

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


# ---------------------------------------------------------------- primitives

def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def layer_norm_no_affine(x: mx.array, eps: float = 1e-6) -> mx.array:
    # diffusers nn.LayerNorm(elementwise_affine=False): biased variance over last dim.
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps)


def get_timestep_embedding(timesteps: mx.array, dim: int,
                           flip_sin_to_cos: bool = True,
                           downscale_freq_shift: float = 0.0,
                           max_period: int = 10000) -> mx.array:
    # diffusers.models.embeddings.get_timestep_embedding (scale=1).
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32)
    exponent = exponent / (half - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps.astype(mx.float32)[:, None] * emb[None, :]
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # diffusers apply_rotary_emb(use_real=True, use_real_unbind_dim=-2).
    # x: [B, H, S, D]; cos/sin: [S, D] -> broadcast over B,H.
    cos = cos[None, None]
    sin = sin[None, None]
    d = x.shape[-1]
    x_real = x[..., : d // 2]
    x_imag = x[..., d // 2:]
    x_rot = mx.concatenate([-x_imag, x_real], axis=-1)
    return x.astype(mx.float32) * cos + x_rot.astype(mx.float32) * sin


# ---------------------------------------------------------------- modules

class CosmosPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: tuple):
        super().__init__()
        self.patch_size = patch_size
        p_t, p_h, p_w = patch_size
        self.proj = nn.Linear(in_channels * p_t * p_h * p_w, out_channels, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        p_t, p_h, p_w = self.patch_size
        x = x.reshape(B, C, T // p_t, p_t, H // p_h, p_h, W // p_w, p_w)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)  # B, T', H', W', C, pt, ph, pw
        x = x.reshape(B, T // p_t, H // p_h, W // p_w, C * p_t * p_h * p_w)
        return self.proj(x)


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def __call__(self, t: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(t)))


class CosmosEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = nn.RMSNorm(embedding_dim, eps=1e-6)

    def __call__(self, timestep: mx.array):
        proj = get_timestep_embedding(timestep, self.embedding_dim,
                                      flip_sin_to_cos=True, downscale_freq_shift=0.0)
        temb = self.t_embedder(proj)
        embedded_timestep = self.norm(proj)
        return temb, embedded_timestep


class _AdaLNZero(nn.Module):
    """norm1/2/3: LayerNorm(no affine) then shift/scale/gate from AdaLN-LoRA + temb."""
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def __call__(self, x, embedded_timestep, temb):
        e = self.linear_2(self.linear_1(nn.silu(embedded_timestep)))
        e = e + temb
        shift, scale, gate = mx.split(e, 3, axis=-1)
        x = layer_norm_no_affine(x)
        if e.ndim == 2:
            shift = shift[:, None]; scale = scale[:, None]; gate = gate[:, None]
        return x * (1 + scale) + shift, gate


class _AdaLN(nn.Module):
    """norm_out: shift/scale only; adds temb[..., :2*dim]."""
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def __call__(self, x, embedded_timestep, temb):
        e = self.linear_2(self.linear_1(nn.silu(embedded_timestep)))
        e = e + temb[..., : 2 * self.in_features]
        shift, scale = mx.split(e, 2, axis=-1)
        x = layer_norm_no_affine(x)
        if e.ndim == 2:
            shift = shift[:, None]; scale = scale[:, None]
        return x * (1 + scale) + shift


class CosmosAttention(nn.Module):
    def __init__(self, cfg: CosmosDiTConfig, cross_dim: int | None):
        super().__init__()
        self.heads = cfg.num_attention_heads
        self.dim_head = cfg.attention_head_dim
        dim = cfg.hidden_size
        kv_in = cross_dim if cross_dim is not None else dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_in, dim, bias=False)
        self.to_v = nn.Linear(kv_in, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.scale = self.dim_head ** -0.5

    def __call__(self, x, context=None, rope=None):
        ctx = x if context is None else context
        B, S, _ = x.shape
        Sk = ctx.shape[1]
        q = self.to_q(x).reshape(B, S, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = self.to_k(ctx).reshape(B, Sk, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = self.to_v(ctx).reshape(B, Sk, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        q = self.norm_q(q)
        k = self.norm_k(k)
        if rope is not None:
            cos, sin = rope
            q = apply_rotary_emb(q, cos, sin).astype(v.dtype)
            k = apply_rotary_emb(k, cos, sin).astype(v.dtype)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.heads * self.dim_head)
        return self.to_out[0](out)


class CosmosFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float):
        super().__init__()
        inner = int(dim * mult)
        # ff.net.0.proj (Linear) -> GELU -> ff.net.2 (Linear)
        self.net = [_FFProj(dim, inner), None, nn.Linear(inner, dim, bias=False)]

    def __call__(self, x):
        x = nn.gelu(self.net[0](x))
        return self.net[2](x)


class _FFProj(nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.proj = nn.Linear(dim, inner, bias=False)

    def __call__(self, x):
        return self.proj(x)


class CosmosTransformerBlock(nn.Module):
    def __init__(self, cfg: CosmosDiTConfig):
        super().__init__()
        h = cfg.hidden_size
        self.norm1 = _AdaLNZero(h, cfg.adaln_lora_dim)
        self.attn1 = CosmosAttention(cfg, cross_dim=None)
        self.norm2 = _AdaLNZero(h, cfg.adaln_lora_dim)
        self.attn2 = CosmosAttention(cfg, cross_dim=cfg.text_embed_dim)
        self.norm3 = _AdaLNZero(h, cfg.adaln_lora_dim)
        self.ff = CosmosFeedForward(h, cfg.mlp_ratio)

    def __call__(self, x, context, embedded_timestep, temb, rope):
        n, g = self.norm1(x, embedded_timestep, temb)
        x = x + g * self.attn1(n, rope=rope)
        n, g = self.norm2(x, embedded_timestep, temb)
        x = x + g * self.attn2(n, context=context)
        n, g = self.norm3(x, embedded_timestep, temb)
        x = x + g * self.ff(n)
        return x


class CosmosRotaryPosEmbed:
    def __init__(self, cfg: CosmosDiTConfig):
        head_dim = cfg.attention_head_dim
        self.patch_size = cfg.patch_size
        self.base_fps = cfg.base_fps
        self.max_size = [s // p for s, p in zip(cfg.max_size, cfg.patch_size)]
        self.dim_h = head_dim // 6 * 2
        self.dim_w = head_dim // 6 * 2
        self.dim_t = head_dim - self.dim_h - self.dim_w
        rs = cfg.rope_scale
        self.h_ntk = rs[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk = rs[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk = rs[0] ** (self.dim_t / (self.dim_t - 2))

    def __call__(self, T, H, W, fps=None):
        pe = [T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]]
        h_theta = 10000.0 * self.h_ntk
        w_theta = 10000.0 * self.w_ntk
        t_theta = 10000.0 * self.t_ntk
        seq = mx.arange(max(self.max_size), dtype=mx.float32)
        dim_h_range = mx.arange(0, self.dim_h, 2, dtype=mx.float32)[: self.dim_h // 2] / self.dim_h
        dim_w_range = mx.arange(0, self.dim_w, 2, dtype=mx.float32)[: self.dim_w // 2] / self.dim_w
        dim_t_range = mx.arange(0, self.dim_t, 2, dtype=mx.float32)[: self.dim_t // 2] / self.dim_t
        h_freqs = 1.0 / (h_theta ** dim_h_range)
        w_freqs = 1.0 / (w_theta ** dim_w_range)
        t_freqs = 1.0 / (t_theta ** dim_t_range)
        emb_h = mx.outer(seq[: pe[1]], h_freqs)[None, :, None, :]
        emb_h = mx.broadcast_to(emb_h, (pe[0], pe[1], pe[2], emb_h.shape[-1]))
        emb_w = mx.outer(seq[: pe[2]], w_freqs)[None, None, :, :]
        emb_w = mx.broadcast_to(emb_w, (pe[0], pe[1], pe[2], emb_w.shape[-1]))
        if fps is None:
            emb_t = mx.outer(seq[: pe[0]], t_freqs)
        else:
            emb_t = mx.outer(seq[: pe[0]] / fps * self.base_fps, t_freqs)
        emb_t = emb_t[:, None, None, :]
        emb_t = mx.broadcast_to(emb_t, (pe[0], pe[1], pe[2], emb_t.shape[-1]))
        freqs = mx.concatenate([emb_t, emb_h, emb_w, emb_t, emb_h, emb_w], axis=-1)
        freqs = freqs.reshape(-1, freqs.shape[-1])
        return mx.cos(freqs), mx.sin(freqs)


class CosmosTransformer3DModel(nn.Module):
    def __init__(self, cfg: CosmosDiTConfig):
        super().__init__()
        self.config = cfg
        in_ch = cfg.in_channels + (1 if cfg.concat_padding_mask else 0)
        self.patch_embed = CosmosPatchEmbed(in_ch, cfg.hidden_size, cfg.patch_size)
        self.rope = CosmosRotaryPosEmbed(cfg)
        self.time_embed = CosmosEmbedding(cfg.hidden_size, cfg.hidden_size)
        self.transformer_blocks = [CosmosTransformerBlock(cfg) for _ in range(cfg.num_layers)]
        self.norm_out = _AdaLN(cfg.hidden_size, cfg.adaln_lora_dim)
        p = cfg.patch_size
        self.proj_out = nn.Linear(cfg.hidden_size, p[0] * p[1] * p[2] * cfg.out_channels, bias=False)

    def __call__(self, hidden_states, timestep, encoder_hidden_states,
                 padding_mask=None, fps=None):
        cfg = self.config
        B, C, T, H, W = hidden_states.shape
        if cfg.concat_padding_mask:
            if padding_mask is None:
                padding_mask = mx.zeros((B, 1, H, W), dtype=hidden_states.dtype)
            pm = padding_mask[:, :, None, :, :]  # [B,1,1,H,W]
            pm = mx.broadcast_to(pm, (B, 1, T, H, W))
            hidden_states = mx.concatenate([hidden_states, pm], axis=1)

        cos, sin = self.rope(T, H, W, fps=fps)

        x = self.patch_embed(hidden_states)            # [B,T',H',W',hidden]
        Tp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape(B, Tp * Hp * Wp, x.shape[-1])    # [B, THW, hidden]

        temb, embedded_timestep = self.time_embed(timestep)

        for blk in self.transformer_blocks:
            x = blk(x, encoder_hidden_states, embedded_timestep, temb, (cos, sin))

        x = self.norm_out(x, embedded_timestep, temb)
        x = self.proj_out(x)                           # [B, THW, p*p*out_ch]
        return self._unpatchify(x, Tp, Hp, Wp)

    def _unpatchify(self, x, Tp, Hp, Wp):
        p_t, p_h, p_w = self.config.patch_size
        oc = self.config.out_channels
        B = x.shape[0]
        # diffusers: unflatten(2,(p_h,p_w,p_t,-1)); unflatten(1,(Tp,Hp,Wp));
        # permute(0,7,1,6,2,4,3,5); flatten(6,7).flatten(4,5).flatten(2,3)
        x = x.reshape(B, x.shape[1], p_h, p_w, p_t, oc)
        x = x.reshape(B, Tp, Hp, Wp, p_h, p_w, p_t, oc)
        x = x.transpose(0, 7, 1, 6, 2, 4, 3, 5)        # B, oc, Tp, p_t, Hp, p_h, Wp, p_w
        x = x.reshape(B, oc, Tp * p_t, Hp * p_h, Wp * p_w)
        return x
