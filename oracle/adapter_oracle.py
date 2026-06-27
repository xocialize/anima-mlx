"""Anima llm_adapter PyTorch oracle (from comfy/ldm/anima/model.py, operations=nn).

Loads net.llm_adapter.* from the Anima checkpoint and dumps parity goldens.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

CKPT = "weights/split_files/diffusion_models/anima-base-v1.0.safetensors"
OUT = os.path.join(os.path.dirname(__file__), "..", "anima-mlx", "tests", "goldens", "adapter")
os.makedirs(OUT, exist_ok=True)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner = head_dim * n_heads
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, pe=None, pe_ctx=None):
        context = x if context is None else context
        ish = x.shape[:-1]; csh = context.shape[:-1]
        q = self.q_norm(self.q_proj(x).view(*ish, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(*csh, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(context).view(*csh, self.n_heads, self.head_dim).transpose(1, 2)
        if pe is not None:
            cos, sin = pe; q = apply_rotary_pos_emb(q, cos, sin)
            cos, sin = pe_ctx; k = apply_rotary_pos_emb(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(*ish, -1)
        return self.o_proj(o)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn
        hd = model_dim // num_heads
        self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.self_attn = Attention(model_dim, model_dim, num_heads, hd)
        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = Attention(model_dim, source_dim, num_heads, hd)
        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(model_dim, int(model_dim * mlp_ratio)),
                                 nn.GELU(), nn.Linear(int(model_dim * mlp_ratio), model_dim))

    def forward(self, x, context, pe=None, pe_ctx=None):
        if self.use_self_attn:
            x = x + self.self_attn(self.norm_self_attn(x), pe=pe, pe_ctx=pe)
        x = x + self.cross_attn(self.norm_cross_attn(x), context=context, pe=pe, pe_ctx=pe_ctx)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim)
        self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(source_dim, model_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids):
        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(context.dtype))
        pos = torch.arange(x.shape[1])[None]
        pos_ctx = torch.arange(context.shape[1])[None]
        pe = self.rotary_emb(x, pos)
        pe_ctx = self.rotary_emb(x, pos_ctx)
        for b in self.blocks:
            x = b(x, context, pe=pe, pe_ctx=pe_ctx)
        return self.norm(self.out_proj(x))


def load_adapter(dtype=torch.float32):
    state = load_file(CKPT)
    a = {k[len("net.llm_adapter."):]: v for k, v in state.items() if k.startswith("net.llm_adapter.")}
    m = LLMAdapter()
    missing, unexpected = m.load_state_dict(a, strict=False)
    # buffers (inv_freq) are non-persistent -> appear "missing"; tolerate only that
    miss = [k for k in missing if "inv_freq" not in k]
    assert not miss, f"missing {miss[:8]}"
    assert not unexpected, f"unexpected {list(unexpected)[:8]}"
    return m.to(dtype).eval()


def save(name, t):
    np.save(os.path.join(OUT, name + ".npy"), t.detach().to(torch.float32).cpu().numpy())


if __name__ == "__main__":
    torch.manual_seed(0)
    m = load_adapter()
    src = torch.randn(1, 20, 1024)          # Qwen3 hidden (20 tokens)
    ids = torch.randint(0, 32128, (1, 14))  # T5 ids (14 tokens)
    save("in_source", src)
    np.save(os.path.join(OUT, "in_ids.npy"), ids.cpu().numpy().astype(np.int32))
    with torch.no_grad():
        # block-0 internals
        ctx = src
        x = m.embed(ids).to(ctx.dtype)
        save("after_embed", x)
        pos = torch.arange(x.shape[1])[None]; pos_ctx = torch.arange(ctx.shape[1])[None]
        pe = m.rotary_emb(x, pos); pe_ctx = m.rotary_emb(x, pos_ctx)
        save("rope_cos", pe[0]); save("rope_sin", pe[1])
        b0 = m.blocks[0]
        x = x + b0.self_attn(b0.norm_self_attn(x), pe=pe, pe_ctx=pe)
        save("b0_after_selfattn", x)
        x = x + b0.cross_attn(b0.norm_cross_attn(x), context=ctx, pe=pe, pe_ctx=pe_ctx)
        save("b0_after_crossattn", x)
        out = m(src, ids)
        save("out_final", out)
    print("[adapter goldens] done. out shape", tuple(out.shape))
