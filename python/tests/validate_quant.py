"""Quantize the 2B Cosmos DiT and gate per-pass cosine vs the bf16 DiT on an identical
injected input (mlx-porting doctrine: quant perturbs the trajectory → gate per-pass
cosine + image validity, NOT PSNR). Quant scope = transformer-block attn + ff Linears
only; keep precision-sensitive embeds/AdaLN/time/proj at bf16 (keep_hi_precision).

Reports int4(g64)/int8(g128) cosine, quantized DiT resident bytes, peak unified memory,
and saves an int4 e2e sample. Run on GPU (quantized matmuls route to Metal).
"""
import os, sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anima_mlx.models.cosmos_dit import CosmosTransformer3DModel, CosmosDiTConfig
from anima_mlx.weights import load_dit_weights
from anima_mlx.pipeline import AnimaPipeline
from anima_mlx.tokenizer import AnimaTokenizer

ORACLE = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files")
DIT_CKPT = os.path.join(ORACLE, "diffusion_models", "anima-base-v1.0.safetensors")
QWEN_CKPT = os.path.join(ORACLE, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_CKPT = os.path.join(ROOT, "tests", "goldens", "vae", "wan_vae_decoder_mlx.safetensors")
GD = os.path.join(ROOT, "tests", "goldens")


def keep_hi_precision(path, m):
    """Quantize ONLY transformer-block attention + feed-forward Linears."""
    if not isinstance(m, nn.Linear):
        return False
    return "transformer_blocks" in path and (".attn" in path or ".ff." in path)


def dit_bf16():
    d = CosmosTransformer3DModel(CosmosDiTConfig()); load_dit_weights(d, DIT_CKPT, dtype=mx.bfloat16)
    d.eval(); mx.eval(d.parameters()); return d


def quantize_dit(bits, group):
    d = CosmosTransformer3DModel(CosmosDiTConfig()); load_dit_weights(d, DIT_CKPT, dtype=mx.bfloat16)
    nn.quantize(d, group_size=group, bits=bits, class_predicate=keep_hi_precision)
    d.eval(); mx.eval(d.parameters()); return d


def resident_bytes(model):
    tot = 0
    for _, v in nn.utils.tree_flatten(model.parameters()):
        if isinstance(v, mx.array):
            tot += v.size * v.dtype.size
    return tot


def cos(a, b):
    a = np.array(a.astype(mx.float32)).ravel(); b = np.array(b.astype(mx.float32)).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def g(sub, n): return np.load(os.path.join(GD, sub, n + ".npy"))


def main():
    # identical injected input (the dit golden fixture, bf16)
    x = mx.array(g("dit", "in_hidden")).astype(mx.bfloat16)
    t = mx.array(g("dit", "in_timestep")).astype(mx.bfloat16)
    enc = mx.array(g("dit", "in_encoder")).astype(mx.bfloat16)
    pad = mx.array(g("dit", "in_padding")).astype(mx.bfloat16)

    ref = dit_bf16()
    v_ref = ref(x, t, enc, padding_mask=pad); mx.eval(v_ref)
    rb_bf16 = resident_bytes(ref)
    print(f"[bf16] DiT resident {rb_bf16/1e9:.2f} GB")
    del ref; mx.clear_cache()

    for bits, group in [(8, 128), (4, 64)]:
        q = quantize_dit(bits, group)
        v = q(x, t, enc, padding_mask=pad); mx.eval(v)
        c = cos(v, v_ref)
        rb = resident_bytes(q)
        gate = 0.9999 if bits == 8 else 0.99
        ok = c >= gate
        print(f"[int{bits} g{group}] per-pass cos={c:.5f} (gate {gate})  resident {rb/1e9:.2f} GB "
              f"({rb/rb_bf16*100:.0f}% of bf16)  {'ok' if ok else 'FAIL'}")
        del q; mx.clear_cache()

    # int4 e2e sample + peak memory at 512²
    print("\n[e2e int4] 512² / 24 steps …")
    pipe = AnimaPipeline.from_checkpoints(DIT_CKPT, QWEN_CKPT, VAE_CKPT, dtype=mx.bfloat16)
    nn.quantize(pipe.dit, group_size=64, bits=4, class_predicate=keep_hi_precision)
    pipe.dit.eval(); mx.eval(pipe.dit.parameters())
    tok = AnimaTokenizer()
    cond = pipe.encode_context(*tok.encode("1girl, anime, masterpiece, detailed background, soft lighting"))
    unc = pipe.encode_context(*tok.encode(""))
    img = pipe.generate(cond, unc, height=512, width=512, steps=24, cfg=5.0, seed=1234); mx.eval(img)
    arr = np.array(img[0].astype(mx.float32))
    print(f"[e2e int4] nan={bool(np.isnan(arr).any())} mean={arr.mean():.3f} std={arr.std():.3f} "
          f"peak {mx.get_peak_memory()/1e9:.2f} GB")
    Image.fromarray((arr * 255).round().clip(0, 255).astype(np.uint8)).save(
        os.path.join(ROOT, "..", "anima-oracle", "anima_int4_gpu.png"))


if __name__ == "__main__":
    main()
