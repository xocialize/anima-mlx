"""Anima llm_adapter in MLX — transpose of comfy/ldm/anima/model.py LLMAdapter.

Bridges Qwen3-0.6B hidden (source/KV) + T5 token ids (target/Q stream) -> 1024-d
DiT cross-attn context. Its own LLaMA-style RoPE (rotate_half), distinct from the DiT.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn


def rotate_half(x):
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rotary(x, cos, sin):
    # x: [B,H,S,D]; cos/sin: [B,S,D] -> unsqueeze head dim (axis=1)
    cos = cos[:, None]; sin = sin[:, None]
    return x * cos + rotate_half(x) * sin


def rope_cos_sin(seq_len: int, head_dim: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    pos = mx.arange(seq_len, dtype=mx.float32)
    freqs = pos[:, None] * inv_freq[None, :]            # [S, D/2]
    emb = mx.concatenate([freqs, freqs], axis=-1)       # [S, D]
    return mx.cos(emb)[None], mx.sin(emb)[None]         # [1, S, D]


class AdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)
        self.scale = head_dim ** -0.5

    def __call__(self, x, context=None, pe=None, pe_ctx=None):
        ctx = x if context is None else context
        B, S, _ = x.shape
        Sk = ctx.shape[1]
        q = self.q_norm(self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(ctx).reshape(B, Sk, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(ctx).reshape(B, Sk, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        if pe is not None:
            q = apply_rotary(q, pe[0], pe[1])
            k = apply_rotary(k, pe_ctx[0], pe_ctx[1])
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(o)


class AdapterBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        hd = model_dim // num_heads
        self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.self_attn = AdapterAttention(model_dim, model_dim, num_heads, hd)
        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = AdapterAttention(model_dim, source_dim, num_heads, hd)
        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        inner = int(model_dim * mlp_ratio)
        self.mlp = [nn.Linear(model_dim, inner), None, nn.Linear(inner, model_dim)]

    def __call__(self, x, context, pe, pe_ctx):
        x = x + self.self_attn(self.norm_self_attn(x), pe=pe, pe_ctx=pe)
        x = x + self.cross_attn(self.norm_cross_attn(x), context=context, pe=pe, pe_ctx=pe_ctx)
        h = self.norm_mlp(x)
        h = self.mlp[2](nn.gelu(self.mlp[0](h)))
        return x + h


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.embed = nn.Embedding(32128, target_dim)
        self.blocks = [AdapterBlock(source_dim, model_dim, num_heads=num_heads) for _ in range(num_layers)]
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)

    def __call__(self, source_hidden_states, target_input_ids, pad_to: int | None = 512,
                 target_weights=None):
        context = source_hidden_states
        x = self.embed(target_input_ids).astype(context.dtype)
        head_dim = self.model_dim // self.num_heads
        pe = rope_cos_sin(x.shape[1], head_dim)
        pe_ctx = rope_cos_sin(context.shape[1], head_dim)
        for b in self.blocks:
            x = b(x, context, pe, pe_ctx)
        out = self.norm(self.out_proj(x))
        if target_weights is not None:
            out = out * target_weights[..., None]
        if pad_to is not None and out.shape[1] < pad_to:
            out = mx.pad(out, [(0, 0), (0, pad_to - out.shape[1]), (0, 0)])
        return out


def load_adapter_weights(model, ckpt_path: str, dtype=mx.float32):
    raw = mx.load(ckpt_path)
    flat = {}
    for k, v in raw.items():
        if not k.startswith("net.llm_adapter."):
            continue
        leaf = k[len("net.llm_adapter."):]
        # mlp.0/.2 already match the list-based module; embed/out_proj/norm direct.
        flat[leaf] = v.astype(dtype)
    model.load_weights(list(flat.items()))
    return model, len(flat)
