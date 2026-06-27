"""Anima DiT oracle: load the ComfyUI-native Anima checkpoint into diffusers
CosmosTransformer3DModel via a strict key remap, for MLX-port parity goldens.

Main DiT only (the llm_adapter is Anima-custom and handled separately).
"""
import re
import torch
from safetensors.torch import load_file
from diffusers import CosmosTransformer3DModel

CKPT = "weights/split_files/diffusion_models/anima-base-v1.0.safetensors"

# 2B config; Anima omits the learnable extra-pos-embed (RoPE only).
CONFIG = dict(
    in_channels=16, out_channels=16, num_attention_heads=16, attention_head_dim=128,
    num_layers=28, mlp_ratio=4.0, text_embed_dim=1024, adaln_lora_dim=256,
    patch_size=(1, 2, 2), concat_padding_mask=True, extra_pos_embed_type=None,
)

# Per-block native->diffusers leaf remaps.
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


def remap_main_dit(state):
    """native Anima keys -> diffusers keys. Drops llm_adapter.* (handled separately)."""
    out, adapter = {}, {}
    for k, v in state.items():
        if k.startswith("net.llm_adapter."):
            adapter[k] = v
            continue
        if k in _TOP:
            out[_TOP[k]] = v
            continue
        m = re.match(r"net\.blocks\.(\d+)\.(.+)\.weight$", k)
        if m:
            i, leaf = m.group(1), m.group(2)
            assert leaf in _BLOCK, f"unmapped block leaf: {leaf}"
            out[f"transformer_blocks.{i}.{_BLOCK[leaf]}.weight"] = v
            continue
        raise KeyError(f"unmapped checkpoint key: {k}")
    return out, adapter


def load_oracle(dtype=torch.float32):
    state = load_file(CKPT)
    mapped, adapter = remap_main_dit(state)
    model = CosmosTransformer3DModel(**CONFIG)
    expected = set(model.state_dict().keys())
    got = set(mapped.keys())
    missing, unexpected = expected - got, got - expected
    assert not missing, f"MISSING {len(missing)}: {sorted(missing)[:8]}"
    assert not unexpected, f"UNEXPECTED {len(unexpected)}: {sorted(unexpected)[:8]}"
    model.load_state_dict(mapped, strict=True)
    model = model.to(dtype).eval()
    print(f"[oracle] strict load OK — {len(mapped)} main-DiT tensors, "
          f"{len(adapter)} llm_adapter tensors held out")
    return model, adapter


if __name__ == "__main__":
    model, adapter = load_oracle()
    print("[oracle] CosmosTransformer3DModel ready.")
