"""Parity: MLX llm_adapter vs PyTorch-oracle goldens. CPU stream, fp32."""
import os, sys
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from anima_mlx.models.llm_adapter import (  # noqa: E402
    LLMAdapter, load_adapter_weights, rope_cos_sin)

GOLD = os.path.join(ROOT, "tests", "goldens", "adapter")
CKPT = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files",
                    "diffusion_models", "anima-base-v1.0.safetensors")


def g(n): return mx.array(np.load(os.path.join(GOLD, n + ".npy")))


def cmp(name, got, want, tol=1e-3):
    got = np.asarray(got, np.float32); want = np.asarray(want, np.float32)
    if got.shape != want.shape:
        print(f"  [SHAPE!] {name}: {got.shape} vs {want.shape}"); return False
    mad = float(np.max(np.abs(got - want)))
    ok = mad < tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:18s} max_abs={mad:.2e}")
    return ok


def main():
    m = LLMAdapter()
    m, n = load_adapter_weights(m, CKPT)
    m.eval(); mx.eval(m.parameters())
    print(f"[load] {n} adapter tensors")
    src = g("in_source")
    ids = mx.array(np.load(os.path.join(GOLD, "in_ids.npy")))
    res = []

    x = m.embed(ids).astype(src.dtype)
    res.append(cmp("after_embed", x, g("after_embed")))
    cos, sin = rope_cos_sin(x.shape[1], 64)
    res.append(cmp("rope_cos", cos, g("rope_cos")))
    res.append(cmp("rope_sin", sin, g("rope_sin")))

    b0 = m.blocks[0]
    pe = rope_cos_sin(x.shape[1], 64)
    pe_ctx = rope_cos_sin(src.shape[1], 64)
    x = x + b0.self_attn(b0.norm_self_attn(x), pe=pe, pe_ctx=pe)
    res.append(cmp("b0_after_selfattn", x, g("b0_after_selfattn")))
    x = x + b0.cross_attn(b0.norm_cross_attn(x), context=src, pe=pe, pe_ctx=pe_ctx)
    res.append(cmp("b0_after_crossattn", x, g("b0_after_crossattn")))

    out = m(src, ids, pad_to=None)
    res.append(cmp("out_final", out, g("out_final")))

    print(f"\n{'PASS' if all(res) else 'FAIL'}: {sum(res)}/{len(res)}")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
