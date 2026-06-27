"""Qwen3-0.6B as Anima's text encoder, in MLX. Causal; returns the last decoder
layer output BEFORE the final norm (comfy layer_norm_hidden_state=False)."""
from __future__ import annotations
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6


def _rope(x, cos, sin):
    d = x.shape[-1] // 2
    rot = mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)
    return x * cos + rot * sin


def _rope_tables(seq, head_dim, theta):
    inv = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    pos = mx.arange(seq, dtype=mx.float32)
    f = pos[:, None] * inv[None, :]
    emb = mx.concatenate([f, f], axis=-1)
    return mx.cos(emb)[None, None], mx.sin(emb)[None, None]


class Qwen3Attention(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.nh, self.nkv, self.hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, c.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.hd, eps=c.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.hd, eps=c.rms_norm_eps)
        self.scale = self.hd ** -0.5

    def __call__(self, x, cos, sin, mask):
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, S, self.nh, self.hd)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.nkv, self.hd)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.nkv, self.hd).transpose(0, 2, 1, 3)
        q = _rope(q, cos, sin)
        k = _rope(k, cos, sin)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, self.nh * self.hd)
        return self.o_proj(o)


class Qwen3MLP(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Layer(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.self_attn = Qwen3Attention(c)
        self.post_attention_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.mlp = Qwen3MLP(c)

    def __call__(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3TextEncoder(nn.Module):
    """Wraps the `model.*` namespace; returns the pre-final-norm last hidden."""
    def __init__(self, c: Qwen3Config = Qwen3Config()):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = [Qwen3Layer(c) for _ in range(c.num_hidden_layers)]
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)  # held but NOT applied to output

    def __call__(self, input_ids: mx.array) -> mx.array:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = _rope_tables(S, self.c.head_dim, self.c.rope_theta)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(S).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x  # pre-final-norm (layer_norm_hidden_state=False)


def load_qwen3_te(model, ckpt_path, dtype=mx.float32):
    raw = mx.load(ckpt_path)
    flat = {}
    for k, v in raw.items():
        if k.startswith("model."):
            flat[k[len("model."):]] = v.astype(dtype)
    model.load_weights(list(flat.items()))
    return model, len(flat)
