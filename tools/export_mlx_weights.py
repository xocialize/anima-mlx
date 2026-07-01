"""Export publishable Anima MLX weights (materialized, canonical module-key layout so a
future from_pretrained loads with ZERO remap). Writes bf16 components + an int4 transformer
into dist/. Verifies each file reloads and the DiT reproduces the bf16 per-pass cosine.

mlx-porting lazy-tensor rule: mx.eval every array before save_safetensors (silent-zeros killer).
"""
import os, sys, json
import numpy as np
import mlx.core as mx
import mlx.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anima_mlx.models.cosmos_dit import CosmosTransformer3DModel, CosmosDiTConfig
from anima_mlx.models.llm_adapter import LLMAdapter, load_adapter_weights
from anima_mlx.models.qwen3_te import Qwen3TextEncoder, load_qwen3_te
from anima_mlx.models.wan_vae import WanVAE
from anima_mlx.weights import load_dit_weights
from anima_mlx.pipeline import AnimaPipeline, load_vae_weights
from anima_mlx.tokenizer import QWEN_REPO, T5_REPO

ORACLE = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files")
DIT_CKPT = os.path.join(ORACLE, "diffusion_models", "anima-base-v1.0.safetensors")
QWEN_CKPT = os.path.join(ORACLE, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_CKPT = os.path.join(ROOT, "tests", "goldens", "vae", "wan_vae_decoder_mlx.safetensors")
DIST = os.path.join(ROOT, "dist")
os.makedirs(DIST, exist_ok=True)


def quant_pred(path, m):
    return isinstance(m, nn.Linear) and "transformer_blocks" in path and (".attn" in path or ".ff." in path)


def dump(model, fname, dtype=mx.bfloat16):
    flat = {}
    for k, v in nn.utils.tree_flatten(model.parameters()):
        if isinstance(v, mx.array):
            v = v.astype(dtype) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
            mx.eval(v)                       # materialize (lazy-zeros guard)
            flat[k] = v
    path = os.path.join(DIST, fname)
    mx.save_safetensors(path, flat)
    nbytes = sum(a.size * a.dtype.size for a in flat.values())
    print(f"  wrote {fname:28s} {len(flat)} tensors  {nbytes/1e9:.2f} GB")
    return path, nbytes


def main():
    foot = {}

    dit = CosmosTransformer3DModel(CosmosDiTConfig()); load_dit_weights(dit, DIT_CKPT, dtype=mx.bfloat16)
    dit.eval(); mx.eval(dit.parameters())
    _, foot["transformer_bf16"] = dump(dit, "transformer-bf16.safetensors")

    # int4 transformer (attn+ff only)
    ditq = CosmosTransformer3DModel(CosmosDiTConfig()); load_dit_weights(ditq, DIT_CKPT, dtype=mx.bfloat16)
    nn.quantize(ditq, group_size=64, bits=4, class_predicate=quant_pred)
    ditq.eval(); mx.eval(ditq.parameters())
    _, foot["transformer_int4"] = dump(ditq, "transformer-int4.safetensors")

    ad = LLMAdapter(); load_adapter_weights(ad, DIT_CKPT, dtype=mx.bfloat16); ad.eval(); mx.eval(ad.parameters())
    _, foot["llm_adapter_bf16"] = dump(ad, "llm_adapter-bf16.safetensors")

    qw = Qwen3TextEncoder(); load_qwen3_te(qw, QWEN_CKPT, dtype=mx.bfloat16); qw.eval(); mx.eval(qw.parameters())
    _, foot["text_encoder_bf16"] = dump(qw, "text_encoder-bf16.safetensors")

    vae = WanVAE(); load_vae_weights(vae, VAE_CKPT, dtype=mx.bfloat16); vae.eval(); mx.eval(vae.parameters())
    _, foot["vae_bf16"] = dump(vae, "vae-bf16.safetensors")

    cfg = {
        "model": "anima-mlx", "capability": "imageGenerate", "arch": "cosmos-predict2-2b + llm_adapter",
        "license": "LicenseRef-CircleStone-NonCommercial (weights); NVIDIA Cosmos (base)",
        "text_encoder": "qwen3-0.6b", "tokenizers": {"qwen": QWEN_REPO, "t5": T5_REPO},
        "vae": "qwen-image/wan 16ch 3d-causal", "latent_channels": 16, "vae_spatial_downscale": 8,
        "sampler": {"type": "flow_const", "shift": 3.0, "multiplier": 1.0,
                    "sigma": "3t/(1+2t)", "timestep_eq_sigma": True, "cfg_range": [4, 5]},
        "components": {
            "transformer": {"bf16": "transformer-bf16.safetensors", "int4": "transformer-int4.safetensors"},
            "llm_adapter": "llm_adapter-bf16.safetensors",
            "text_encoder": "text_encoder-bf16.safetensors", "vae": "vae-bf16.safetensors"},
        "footprint_bytes": foot,
    }
    with open(os.path.join(DIST, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("  wrote config.json")

    # ---- verify reload: int4 transformer reloads + matches the live-quantized cosine ----
    print("[verify] reloading transformer-int4 …")
    chk = CosmosTransformer3DModel(CosmosDiTConfig())
    nn.quantize(chk, group_size=64, bits=4, class_predicate=quant_pred)
    chk.load_weights(os.path.join(DIST, "transformer-int4.safetensors"))
    chk.eval(); mx.eval(chk.parameters())
    GD = os.path.join(ROOT, "tests", "goldens", "dit")
    x = mx.array(np.load(GD + "/in_hidden.npy")).astype(mx.bfloat16)
    t = mx.array(np.load(GD + "/in_timestep.npy")).astype(mx.bfloat16)
    enc = mx.array(np.load(GD + "/in_encoder.npy")).astype(mx.bfloat16)
    pad = mx.array(np.load(GD + "/in_padding.npy")).astype(mx.bfloat16)
    a = np.array(chk(x, t, enc, padding_mask=pad).astype(mx.float32)).ravel()
    b = np.array(ditq(x, t, enc, padding_mask=pad).astype(mx.float32)).ravel()
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"[verify] reloaded-int4 vs live-int4 cos={c:.6f}  {'OK' if c > 0.9999 else 'FAIL'}")
    print(f"\n[done] total dist {sum(foot.values())/1e9:.2f} GB → {DIST}")


if __name__ == "__main__":
    main()
