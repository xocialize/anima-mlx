"""Anima T2I pipeline in pure MLX.

Glue over the four parity-locked nets (Qwen3-0.6B TE → llm_adapter → Cosmos DiT → Wan
VAE). Sampler = comfy ModelType.FLOW: CONST prediction + ModelSamplingDiscreteFlow
(shift=3, multiplier=1) → sigma(t)=3t/(1+2t), DiT timestep == sigma ∈ [0,1], deterministic
Euler. CFG: v = v_unc + cfg*(v_cond - v_unc). Wan21 latent denorm before VAE decode.

The four nets carry their own PT-parity gates; the e2e gate (tests/parity/test_pipeline_parity.py)
checks this glue against the torch oracle pipeline with an injected noise + identical token ids.
"""
from __future__ import annotations
import os
import re
import numpy as np
import mlx.core as mx

from .models.cosmos_dit import CosmosTransformer3DModel, CosmosDiTConfig
from .models.llm_adapter import LLMAdapter, load_adapter_weights
from .models.qwen3_te import Qwen3TextEncoder, load_qwen3_te
from .models.wan_vae import WanVAE
from .weights import load_dit_weights

PAD_TO = 512
VAE_SPATIAL = 8                      # Wan VAE spatial downscale
QWEN_PAD = 151643                    # Qwen3 pad token (min_length 1)

_CAUSAL = {"conv_in", "conv_out", "conv1", "conv2", "conv_shortcut", "post_quant_conv", "time_conv"}


def _vae_remap(k: str) -> str:
    parts = k.split(".")
    if parts[-1] in ("weight", "bias") and parts[-2] in _CAUSAL:
        return ".".join(parts[:-1] + ["conv", parts[-1]])
    return k


def load_vae_weights(model: WanVAE, ckpt_path: str, dtype=mx.float32):
    raw = mx.load(ckpt_path)
    flat = {_vae_remap(k): v.astype(dtype) for k, v in raw.items()}
    model.load_weights(list(flat.items()))
    return model, len(flat)


# ----------------------------------------------------------------- sampler math
def time_snr_shift(alpha: float, t):
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def flow_sigmas(steps: int, shift: float = 3.0, mult: float = 1.0) -> np.ndarray:
    """comfy 'normal' scheduler for ModelSamplingDiscreteFlow. sigma_max=1.0."""
    sigma_max = time_snr_shift(shift, 1.0 * mult)        # t=1 -> 1.0
    sigma_min = time_snr_shift(shift, (1.0 / 1000.0) * mult)
    ts = np.linspace(sigma_max * mult, sigma_min * mult, steps)
    sigs = [time_snr_shift(shift, t / mult) for t in ts]
    sigs.append(0.0)
    return np.asarray(sigs, dtype=np.float64)


class AnimaPipeline:
    def __init__(self, dit, adapter, qwen, vae):
        self.dit = dit
        self.adapter = adapter
        self.qwen = qwen
        self.vae = vae

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_checkpoints(cls, dit_ckpt: str, qwen_ckpt: str, vae_ckpt: str, dtype=mx.float32):
        dit = CosmosTransformer3DModel(CosmosDiTConfig())
        load_dit_weights(dit, dit_ckpt, dtype=dtype)
        adapter = LLMAdapter()
        load_adapter_weights(adapter, dit_ckpt, dtype=dtype)   # adapter weights live in the DiT ckpt
        qwen = Qwen3TextEncoder()
        load_qwen3_te(qwen, qwen_ckpt, dtype=dtype)
        vae = WanVAE()
        load_vae_weights(vae, vae_ckpt, dtype=dtype)
        for m in (dit, adapter, qwen, vae):
            m.eval()
            mx.eval(m.parameters())
        return cls(dit, adapter, qwen, vae)

    @classmethod
    def from_pretrained(cls, repo_id: str = "xocialize/anima-mlx", quant: str = "bf16",
                        dtype=mx.bfloat16):
        """Load the published MLX components (canonical module-key layout, zero remap).
        quant: 'bf16' (full) or 'int4' (quantized transformer attn+ff)."""
        import mlx.nn as nn
        from huggingface_hub import hf_hub_download

        def fetch(name):
            return hf_hub_download(repo_id, name)

        dit = CosmosTransformer3DModel(CosmosDiTConfig())
        if quant == "int4":
            nn.quantize(dit, group_size=64, bits=4,
                        class_predicate=lambda p, m: isinstance(m, nn.Linear)
                        and "transformer_blocks" in p and (".attn" in p or ".ff." in p))
            dit.load_weights(fetch("transformer-int4.safetensors"))
        else:
            dit.load_weights(fetch("transformer-bf16.safetensors"))
        adapter = LLMAdapter(); adapter.load_weights(fetch("llm_adapter-bf16.safetensors"))
        qwen = Qwen3TextEncoder(); qwen.load_weights(fetch("text_encoder-bf16.safetensors"))
        vae = WanVAE(); vae.load_weights(fetch("vae-bf16.safetensors"))
        for m in (dit, adapter, qwen, vae):
            m.eval()
            mx.eval(m.parameters())
        return cls(dit, adapter, qwen, vae)

    # ---- text path ----------------------------------------------------------
    def encode_context(self, qwen_ids, t5_ids) -> mx.array:
        qids = mx.array(np.asarray(qwen_ids, np.int32)[None])
        tids = mx.array(np.asarray(t5_ids, np.int32)[None])
        src = self.qwen(qids)                              # [1, Lq, 1024] pre-final-norm
        return self.adapter(src, tids, pad_to=PAD_TO)      # [1, 512, 1024]

    # ---- sampling -----------------------------------------------------------
    def _dit_v(self, x, sigma, ctx):
        B = x.shape[0]
        t = mx.full((B,), float(sigma), dtype=mx.float32)
        return self.dit(x, t, ctx)                          # padding_mask=None -> zeros

    def sample(self, noise: mx.array, cond_ctx: mx.array, uncond_ctx: mx.array,
               sigmas: np.ndarray, cfg: float, verbose: bool = False) -> mx.array:
        x = (float(sigmas[0]) * noise).astype(mx.float32)
        ctx = mx.concatenate([cond_ctx, uncond_ctx], axis=0)
        for i in range(len(sigmas) - 1):
            s = float(sigmas[i])
            xb = mx.concatenate([x, x], axis=0)
            v = self._dit_v(xb, s, ctx)
            v_cond, v_unc = v[:1], v[1:]
            v_cfg = v_unc + cfg * (v_cond - v_unc)
            x = x + v_cfg * (float(sigmas[i + 1]) - s)
            mx.eval(x)
            if verbose:
                print(f"  step {i}: sigma {s:.4f} -> {float(sigmas[i+1]):.4f}  x std {float(x.std()):.4f}")
        return x                                            # model-space x0

    # ---- decode -------------------------------------------------------------
    def decode(self, x0: mx.array) -> mx.array:
        """x0 model-space latent [B,16,1,Hl,Wl] -> image [B,H,W,3] in [0,1]."""
        lat = WanVAE.denormalize(x0)
        img = self.vae.decode(lat)                          # [B,3,1,H,W] in [-1,1]
        img = (img[:, :, 0].transpose(0, 2, 3, 1) + 1.0) * 0.5
        return mx.clip(img, 0.0, 1.0)

    # ---- high level ---------------------------------------------------------
    def generate(self, cond_ctx, uncond_ctx, height=512, width=512, steps=30, cfg=5.0,
                 seed=0, noise=None, verbose=False) -> mx.array:
        Hl, Wl = height // VAE_SPATIAL, width // VAE_SPATIAL
        if noise is None:
            rng = np.random.default_rng(seed)
            noise = rng.standard_normal((1, 16, 1, Hl, Wl)).astype(np.float32)
        noise = mx.array(np.asarray(noise, np.float32))
        sigmas = flow_sigmas(steps)                         # shift=3.0
        x0 = self.sample(noise, cond_ctx, uncond_ctx, sigmas, cfg, verbose=verbose)
        return self.decode(x0)
