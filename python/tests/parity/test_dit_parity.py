"""Component + e2e parity: MLX Cosmos DiT vs PyTorch-oracle goldens. CPU stream, fp32."""
import os
import sys
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from anima_mlx.models.cosmos_dit import (  # noqa: E402
    CosmosDiTConfig, CosmosTransformer3DModel)
from anima_mlx.weights import load_dit_weights  # noqa: E402

GOLD = os.path.join(ROOT, "tests", "goldens", "dit")
CKPT = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files",
                    "diffusion_models", "anima-base-v1.0.safetensors")


def g(name):
    return mx.array(np.load(os.path.join(GOLD, name + ".npy")))


def cmp(name, got, want, tol=1e-3):
    got = np.asarray(got, dtype=np.float32)
    want = np.asarray(want, dtype=np.float32)
    if got.shape != want.shape:
        print(f"  [SHAPE!] {name}: got {got.shape} want {want.shape}")
        return False
    mad = float(np.max(np.abs(got - want)))
    denom = float(np.mean(np.abs(want))) + 1e-9
    rel = float(np.mean(np.abs(got - want)) / denom)
    ok = mad < tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:16s} max_abs={mad:.2e} mean_rel={rel:.2e}")
    return ok


def main():
    cfg = CosmosDiTConfig()
    model = CosmosTransformer3DModel(cfg)
    model, n = load_dit_weights(model, CKPT, dtype=mx.float32)
    model.eval()
    print(f"[load] {n} tensors into MLX DiT")
    mx.eval(model.parameters())

    results = []

    # --- inputs ---
    hidden = g("in_hidden"); timestep = g("in_timestep")
    enc = g("in_encoder"); pad = g("in_padding")

    # --- rope ---
    cos, sin = model.rope(1, 32, 32, fps=None)
    results.append(cmp("rope_cos", cos, g("rope_cos")))
    results.append(cmp("rope_sin", sin, g("rope_sin")))

    # --- time_embed ---
    temb, emb_ts = model.time_embed(timestep)
    results.append(cmp("temb", temb, g("temb")))
    results.append(cmp("embedded_timestep", emb_ts, g("embedded_timestep")))

    # --- patch_embed (concat padding mask first, like the model does) ---
    B, C, T, H, W = hidden.shape
    pm = mx.broadcast_to(pad[:, :, None, :, :], (B, 1, T, H, W))
    hin = mx.concatenate([hidden, pm], axis=1)
    pe = model.patch_embed(hin)
    pe_flat = pe.reshape(B, -1, pe.shape[-1])
    results.append(cmp("patch_embed", pe_flat, g("patch_embed").reshape(B, -1, 2048)))

    # --- block 0 internals ---
    blk = model.transformer_blocks[0]
    x = pe_flat
    n1, gate1 = blk.norm1(x, emb_ts, temb)
    results.append(cmp("b0_norm1", n1, g("b0_norm1")))
    results.append(cmp("b0_gate1", gate1, g("b0_gate1")))
    a1 = blk.attn1(n1, rope=(cos, sin))
    results.append(cmp("b0_attn1", a1, g("b0_attn1")))
    x = x + gate1 * a1
    n2, gate2 = blk.norm2(x, emb_ts, temb)
    a2 = blk.attn2(n2, context=enc)
    results.append(cmp("b0_attn2", a2, g("b0_attn2")))
    x = x + gate2 * a2
    n3, gate3 = blk.norm3(x, emb_ts, temb)
    ff = blk.ff(n3)
    results.append(cmp("b0_ff", ff, g("b0_ff")))
    x = x + gate3 * ff
    results.append(cmp("block0_out", x, g("block0_out")))

    # --- e2e ---
    out = model(hidden, timestep, enc, padding_mask=pad)
    results.append(cmp("out_final", out, g("out_final"), tol=1e-2))

    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
