"""Localize the e2e sampler divergence: compare per-step latents + the step-0 DiT
output (v0_cfg) between MLX and the torch-oracle golden, with GOLDEN contexts injected.
Tells accumulation (tiny step-0 v, growing x drift) from a regime bug (large step-0 v)."""
import os, sys
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from anima_mlx.pipeline import AnimaPipeline  # noqa: E402

GOLD = os.path.join(ROOT, "tests", "goldens", "pipeline")
ORACLE = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files")
DIT_CKPT = os.path.join(ORACLE, "diffusion_models", "anima-base-v1.0.safetensors")
QWEN_CKPT = os.path.join(ORACLE, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_CKPT = os.path.join(ROOT, "tests", "goldens", "vae", "wan_vae_decoder_mlx.safetensors")


def g(n): return np.load(os.path.join(GOLD, n + ".npy"))


def stats(name, got, want):
    got = np.asarray(got, np.float32).ravel(); want = np.asarray(want, np.float32).ravel()
    mad = float(np.max(np.abs(got - want)))
    cos = float(np.dot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30))
    print(f"  {name:14s} max_abs={mad:.3e}  cos={cos:.7f}  std g/w {got.std():.4f}/{want.std():.4f}")


def main():
    pipe = AnimaPipeline.from_checkpoints(DIT_CKPT, QWEN_CKPT, VAE_CKPT)
    noise = mx.array(g("noise"))
    sigmas = g("sigmas").astype(np.float64)
    gcond = mx.array(g("cond_context")); gunc = mx.array(g("uncond_context"))

    # step 0 DiT output (cfg'd v) in isolation
    x = (float(sigmas[0]) * noise).astype(mx.float32)
    ctx = mx.concatenate([gcond, gunc], axis=0)
    xb = mx.concatenate([x, x], axis=0)
    v = pipe._dit_v(xb, float(sigmas[0]), ctx)
    v0 = v[1:] + 5.0 * (v[:1] - v[1:])  # cfg=5: v_unc + 5*(v_cond - v_unc)
    mx.eval(v0)
    print("[step-0 DiT output v0_cfg]")
    stats("v0_cfg", v0, g("v0_cfg"))

    # per-step latents
    print("[per-step latent x]")
    x = (float(sigmas[0]) * noise).astype(mx.float32)
    for i in range(len(sigmas) - 1):
        s = float(sigmas[i])
        xb = mx.concatenate([x, x], axis=0)
        vv = pipe._dit_v(xb, s, ctx)
        v_cfg = vv[1:] + 5.0 * (vv[:1] - vv[1:])
        x = x + v_cfg * (float(sigmas[i + 1]) - s)
        mx.eval(x)
        stats(f"x_step{i}", x, g(f"x_step{i}"))


if __name__ == "__main__":
    main()
