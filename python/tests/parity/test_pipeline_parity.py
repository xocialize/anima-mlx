"""e2e pipeline parity: MLX AnimaPipeline vs the torch oracle golden.

Injected noise + identical token ids (dumped by anima-oracle/pipeline_oracle.py) remove
RNG incompatibility, so this is a true op-for-op gate over the NEW glue: text path
(qwen3→adapter→pad512), the flow sampler, the DiT-in-loop, and CFG. CPU stream, fp32.

GATING (per mlx-porting "quantized/generative models" doctrine): a denoise loop is a
chaotic ODE — tiny per-op fp differences between torch-CPU and MLX-CPU amplify
exponentially through cfg×feedback (here ~10×/step, worsened by the coarse 6-step
schedule's huge final dt σ0.43→0.009). So we gate on what is actually invariant:
  • deterministic text path: tight max_abs (<5e-3)
  • the per-pass DiT output (step-0 v0_cfg), BEFORE any accumulation: cosine ≈ 1 +
    tight max_abs — this is the real proof the sampler glue, DiT-in-loop and CFG are
    correct.
  • the final latent by COSINE (same image / equally-valid trajectory), NOT max_abs.
Probe `probe_sampler.py` shows the per-step growth with cosine staying ≥0.999.
"""
import os, sys
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from anima_mlx.pipeline import AnimaPipeline, flow_sigmas  # noqa: E402

GOLD = os.path.join(ROOT, "tests", "goldens", "pipeline")
ORACLE = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files")
DIT_CKPT = os.path.join(ORACLE, "diffusion_models", "anima-base-v1.0.safetensors")
QWEN_CKPT = os.path.join(ORACLE, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_CKPT = os.path.join(ROOT, "tests", "goldens", "vae", "wan_vae_decoder_mlx.safetensors")


def g(n):
    return np.load(os.path.join(GOLD, n + ".npy"))


def cmp(name, got, want, tol):
    got = np.asarray(got, np.float32); want = np.asarray(want, np.float32)
    if got.shape != want.shape:
        print(f"  [SHAPE!] {name}: {got.shape} vs {want.shape}"); return False
    mad = float(np.max(np.abs(got - want)))
    ok = mad < tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:28s} max_abs={mad:.2e} (tol {tol:.0e})")
    return ok


def cmp_cos(name, got, want, min_cos):
    a = np.asarray(got, np.float32).ravel(); b = np.asarray(want, np.float32).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    ok = cos >= min_cos
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:28s} cos={cos:.7f} (min {min_cos:.6f})")
    return ok


def main():
    print("[load] building MLX AnimaPipeline …")
    pipe = AnimaPipeline.from_checkpoints(DIT_CKPT, QWEN_CKPT, VAE_CKPT)

    noise = mx.array(g("noise"))
    sigmas = g("sigmas").astype(np.float64)
    cond_ids = (g("cond_qwen_ids"), g("cond_t5_ids"))
    unc_ids = (g("uncond_qwen_ids"), g("uncond_t5_ids"))
    res = []

    # (1) text path
    cond_ctx = pipe.encode_context(*cond_ids); mx.eval(cond_ctx)
    unc_ctx = pipe.encode_context(*unc_ids); mx.eval(unc_ctx)
    res.append(cmp("cond_context", cond_ctx, g("cond_context"), 5e-3))
    res.append(cmp("uncond_context", unc_ctx, g("uncond_context"), 5e-3))

    # sanity: MLX flow schedule matches the dumped oracle schedule
    res.append(cmp("sigma_schedule", flow_sigmas(len(sigmas) - 1), sigmas, 1e-5))

    # (2) step-0 DiT output (DiT-in-loop + CFG), BEFORE chaotic accumulation: the real gate
    gcond = mx.array(g("cond_context")); gunc = mx.array(g("uncond_context"))
    x0i = (float(sigmas[0]) * noise).astype(mx.float32)
    ctx = mx.concatenate([gcond, gunc], axis=0)
    v = pipe._dit_v(mx.concatenate([x0i, x0i], axis=0), float(sigmas[0]), ctx)
    v0 = v[1:] + 5.0 * (v[:1] - v[1:]); mx.eval(v0)
    res.append(cmp("v0_cfg (step-0 DiT)", v0, g("v0_cfg"), 5e-3))
    res.append(cmp_cos("v0_cfg (step-0 DiT)", v0, g("v0_cfg"), 0.999999))

    # (3) full denoise: equally-valid trajectory → gate final latent by COSINE, not max_abs
    x0_inj = pipe.sample(noise, gcond, gunc, sigmas, cfg=5.0, verbose=True); mx.eval(x0_inj)
    res.append(cmp_cos("final_latent (inj ctx)", x0_inj, g("final_latent"), 0.999))
    x0_full = pipe.sample(noise, cond_ctx, unc_ctx, sigmas, cfg=5.0); mx.eval(x0_full)
    res.append(cmp_cos("final_latent (full MLX)", x0_full, g("final_latent"), 0.999))

    print(f"\n{'PASS' if all(res) else 'FAIL'}: {sum(res)}/{len(res)}")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
