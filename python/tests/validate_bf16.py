"""bf16 + GPU-stream validation: run each component at bf16 on the GPU stream vs the
fp32 PT goldens (cosine — bf16+GPU adds ~1e-3 noise so max_abs won't hold), then a full
bf16+GPU e2e generation (NaN check + coherent image). Decides the publish dtype.

Run WITHOUT a CPU pin (default = GPU). Usage: .venv/bin/python tests/validate_bf16.py
"""
import os, sys
import numpy as np
import mlx.core as mx
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anima_mlx.models.cosmos_dit import CosmosTransformer3DModel, CosmosDiTConfig
from anima_mlx.models.llm_adapter import LLMAdapter, load_adapter_weights
from anima_mlx.models.qwen3_te import Qwen3TextEncoder, load_qwen3_te
from anima_mlx.models.wan_vae import WanVAE
from anima_mlx.weights import load_dit_weights
from anima_mlx.pipeline import AnimaPipeline, load_vae_weights

ORACLE = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files")
DIT_CKPT = os.path.join(ORACLE, "diffusion_models", "anima-base-v1.0.safetensors")
QWEN_CKPT = os.path.join(ORACLE, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_CKPT = os.path.join(ROOT, "tests", "goldens", "vae", "wan_vae_decoder_mlx.safetensors")
GD = os.path.join(ROOT, "tests", "goldens")
DT = mx.bfloat16


def cos(a, b):
    a = np.asarray(a, np.float32).ravel(); b = np.asarray(b, np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rep(tag, got, want, gate=0.999):
    if isinstance(got, mx.array):
        got = np.array(got.astype(mx.float32))
    got = np.asarray(got, np.float32)
    c = cos(got, want)
    rel = float(np.max(np.abs(got - want)) / (np.abs(want).max() + 1e-9))
    nan = bool(np.isnan(got).any())
    ok = (c >= gate) and not nan
    print(f"  [{'ok ' if ok else 'FAIL'}] {tag:10s} cos={c:.6f} relmax={rel:.2e} nan={nan}")
    return ok


def g(sub, n): return np.load(os.path.join(GD, sub, n + ".npy"))


def main():
    print(f"[bf16/GPU validation] device default = GPU, dtype = {DT}")
    res = []

    # ---- Qwen3 TE ----
    qm = Qwen3TextEncoder(); load_qwen3_te(qm, QWEN_CKPT, dtype=DT); qm.eval(); mx.eval(qm.parameters())
    out = qm(mx.array(g("qwen3", "in_ids")))
    res.append(rep("qwen3", out, g("qwen3", "hidden_prenorm")))

    # ---- llm_adapter ----
    am = LLMAdapter(); load_adapter_weights(am, DIT_CKPT, dtype=DT); am.eval(); mx.eval(am.parameters())
    src = mx.array(g("adapter", "in_source")).astype(DT)
    out = am(src, mx.array(g("adapter", "in_ids")), pad_to=None)
    res.append(rep("adapter", out, g("adapter", "out_final")))

    # ---- Cosmos DiT ----
    dm = CosmosTransformer3DModel(CosmosDiTConfig()); load_dit_weights(dm, DIT_CKPT, dtype=DT)
    dm.eval(); mx.eval(dm.parameters())
    out = dm(mx.array(g("dit", "in_hidden")).astype(DT),
             mx.array(g("dit", "in_timestep")).astype(DT),
             mx.array(g("dit", "in_encoder")).astype(DT),
             padding_mask=mx.array(g("dit", "in_padding")).astype(DT))
    res.append(rep("cosmos_dit", out, g("dit", "out_final")))

    # ---- Wan VAE ----
    vm = WanVAE(); load_vae_weights(vm, VAE_CKPT, dtype=DT); vm.eval(); mx.eval(vm.parameters())
    out = vm.decode(mx.array(g("vae", "in_latent")).astype(DT))
    res.append(rep("wan_vae", out, g("vae", "out_image"), gate=0.998))

    print(f"\n[components] {'PASS' if all(res) else 'FAIL'}: {sum(res)}/{len(res)}  peak {mx.get_peak_memory()/1e9:.2f} GB")

    # ---- full bf16 + GPU e2e generation ----
    print("\n[e2e] bf16 GPU generation 512² / 24 steps …")
    from anima_mlx.tokenizer import AnimaTokenizer
    pipe = AnimaPipeline.from_checkpoints(DIT_CKPT, QWEN_CKPT, VAE_CKPT, dtype=DT)
    tok = AnimaTokenizer()
    cond = pipe.encode_context(*tok.encode("1girl, anime, masterpiece, detailed background, soft lighting"))
    unc = pipe.encode_context(*tok.encode(""))
    img = pipe.generate(cond, unc, height=512, width=512, steps=24, cfg=5.0, seed=1234)
    mx.eval(img)
    arr = np.array(img[0].astype(mx.float32))
    nan = bool(np.isnan(arr).any())
    print(f"[e2e] nan={nan} mean={arr.mean():.3f} std={arr.std():.3f} peak {mx.get_peak_memory()/1e9:.2f} GB")
    Image.fromarray((arr * 255).round().clip(0, 255).astype(np.uint8)).save(
        os.path.join(ROOT, "..", "anima-oracle", "anima_bf16_gpu.png"))
    res.append(not nan)

    print(f"\nOVERALL {'PASS' if all(res) else 'FAIL'}")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
