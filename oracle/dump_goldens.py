"""Dump granular per-sub-op goldens from the Anima DiT oracle for MLX parity.

Fixed seed, fixed small inputs, CPU stream, fp32. One .npy per intermediate.
Targets land in ../anima-mlx/tests/goldens/dit/.
"""
import os
import numpy as np
import torch
from anima_oracle import load_oracle

OUT = os.path.join(os.path.dirname(__file__), "..", "anima-mlx", "tests", "goldens", "dit")
os.makedirs(OUT, exist_ok=True)


def save(name, t):
    arr = t.detach().to(torch.float32).cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    np.save(os.path.join(OUT, name + ".npy"), arr)
    print(f"  saved {name:42s} {tuple(arr.shape)}")


def main():
    torch.manual_seed(0)
    model, _ = load_oracle(dtype=torch.float32)

    # Fixed inputs: 32x32 latent (=> 16x16=256 patches), 16-token text context.
    B, C, T, H, W = 1, 16, 1, 32, 32
    hidden = torch.randn(B, C, T, H, W, dtype=torch.float32)
    timestep = torch.tensor([700.0], dtype=torch.float32)
    enc = torch.randn(B, 16, 1024, dtype=torch.float32)
    pad = torch.zeros(B, 1, H, W, dtype=torch.float32)
    save("in_hidden", hidden); save("in_timestep", timestep)
    save("in_encoder", enc); save("in_padding", pad)

    caps = {}

    def hook(name):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[name] = o
        return fn

    handles = []
    handles.append(model.patch_embed.register_forward_hook(hook("patch_embed")))
    handles.append(model.time_embed.register_forward_hook(
        lambda m, i, o: caps.update(temb=o[0], embedded_timestep=o[1])))
    handles.append(model.rope.register_forward_hook(
        lambda m, i, o: caps.update(rope_cos=o[0], rope_sin=o[1])))
    # block 0 internals + a few block outputs
    b0 = model.transformer_blocks[0]
    handles.append(b0.norm1.register_forward_hook(
        lambda m, i, o: caps.update(b0_norm1=o[0], b0_gate1=o[1])))
    handles.append(b0.attn1.register_forward_hook(hook("b0_attn1")))
    handles.append(b0.norm2.register_forward_hook(
        lambda m, i, o: caps.update(b0_norm2=o[0], b0_gate2=o[1])))
    handles.append(b0.attn2.register_forward_hook(hook("b0_attn2")))
    handles.append(b0.norm3.register_forward_hook(
        lambda m, i, o: caps.update(b0_norm3=o[0], b0_gate3=o[1])))
    handles.append(b0.ff.register_forward_hook(hook("b0_ff")))
    for bi in (0, 1, 13, 27):
        handles.append(model.transformer_blocks[bi].register_forward_hook(hook(f"block{bi}_out")))
    handles.append(model.norm_out.register_forward_hook(hook("norm_out")))
    handles.append(model.proj_out.register_forward_hook(hook("proj_out")))

    with torch.no_grad():
        out = model(hidden_states=hidden, timestep=timestep, encoder_hidden_states=enc,
                    padding_mask=pad, return_dict=False)[0]
    for h in handles:
        h.remove()

    print("[goldens] intermediates:")
    for k in ["patch_embed", "rope_cos", "rope_sin", "temb", "embedded_timestep",
              "b0_norm1", "b0_gate1", "b0_attn1", "b0_norm2", "b0_gate2", "b0_attn2",
              "b0_norm3", "b0_gate3", "b0_ff", "block0_out", "block1_out", "block13_out",
              "block27_out", "norm_out", "proj_out"]:
        save(k, caps[k])
    save("out_final", out)
    print("[goldens] done.")


if __name__ == "__main__":
    main()
