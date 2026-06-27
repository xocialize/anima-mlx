"""Load the native Anima ComfyUI checkpoint (net.*) into the MLX Cosmos DiT.

The MLX module paths mirror diffusers, so we reuse the same native->diffusers remap
the oracle uses, then hand the arrays to mlx model.load_weights.
"""
from __future__ import annotations
import re
import mlx.core as mx

_BLOCK = {
    "self_attn.q_proj": "attn1.to_q", "self_attn.k_proj": "attn1.to_k",
    "self_attn.v_proj": "attn1.to_v", "self_attn.output_proj": "attn1.to_out.0",
    "self_attn.q_norm": "attn1.norm_q", "self_attn.k_norm": "attn1.norm_k",
    "cross_attn.q_proj": "attn2.to_q", "cross_attn.k_proj": "attn2.to_k",
    "cross_attn.v_proj": "attn2.to_v", "cross_attn.output_proj": "attn2.to_out.0",
    "cross_attn.q_norm": "attn2.norm_q", "cross_attn.k_norm": "attn2.norm_k",
    "mlp.layer1": "ff.net.0.proj", "mlp.layer2": "ff.net.2",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
}
_TOP = {
    "net.x_embedder.proj.1.weight": "patch_embed.proj.weight",
    "net.t_embedder.1.linear_1.weight": "time_embed.t_embedder.linear_1.weight",
    "net.t_embedder.1.linear_2.weight": "time_embed.t_embedder.linear_2.weight",
    "net.t_embedding_norm.weight": "time_embed.norm.weight",
    "net.final_layer.adaln_modulation.1.weight": "norm_out.linear_1.weight",
    "net.final_layer.adaln_modulation.2.weight": "norm_out.linear_2.weight",
    "net.final_layer.linear.weight": "proj_out.weight",
}


def native_to_mlx_key(k: str) -> str | None:
    """Returns the MLX module path for a native main-DiT key, or None to drop it."""
    if k.startswith("net.llm_adapter."):
        return None
    if k in _TOP:
        return _TOP[k]
    m = re.match(r"net\.blocks\.(\d+)\.(.+)\.weight$", k)
    if m:
        i, leaf = m.group(1), m.group(2)
        assert leaf in _BLOCK, f"unmapped block leaf: {leaf}"
        return f"transformer_blocks.{i}.{_BLOCK[leaf]}.weight"
    raise KeyError(f"unmapped checkpoint key: {k}")


def load_dit_weights(model, ckpt_path: str, dtype=mx.float32):
    raw = mx.load(ckpt_path)
    flat = {}
    for k, v in raw.items():
        mk = native_to_mlx_key(k)
        if mk is None:
            continue
        # RMSNorm scales (q_norm/k_norm/time norm) are 1-D; keep as-is.
        flat[mk] = v.astype(dtype)
    model.load_weights(list(flat.items()))
    return model, len(flat)
