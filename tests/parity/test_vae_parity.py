import os, sys
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from anima_mlx.models.wan_vae import WanVAE
GOLD = os.path.join(ROOT, "tests", "goldens", "vae")

CAUSAL = {"conv_in", "conv_out", "conv1", "conv2", "conv_shortcut", "post_quant_conv", "time_conv"}

def remap(k):
    # insert `.conv` for CausalConv3d leaves (X.weight -> X.conv.weight)
    parts = k.split(".")
    if parts[-1] in ("weight", "bias") and parts[-2] in CAUSAL:
        return ".".join(parts[:-1] + ["conv", parts[-1]])
    return k

def main():
    m = WanVAE()
    raw = mx.load(os.path.join(GOLD, "wan_vae_decoder_mlx.safetensors"))
    flat = {remap(k): v.astype(mx.float32) for k, v in raw.items()}
    m.load_weights(list(flat.items()))
    m.eval(); mx.eval(m.parameters())
    print(f"[load] {len(flat)} VAE tensors")
    z = mx.array(np.load(os.path.join(GOLD, "in_latent.npy")))
    img = np.asarray(m.decode(z), np.float32)
    want = np.load(os.path.join(GOLD, "out_image.npy")).astype(np.float32)
    if img.shape != want.shape:
        print(f"SHAPE {img.shape} vs {want.shape}"); sys.exit(1)
    mad = float(np.max(np.abs(img - want)))
    print(f"decode max_abs={mad:.2e}  got mean {img.mean():.4f}/std {img.std():.4f}  want {want.mean():.4f}/{want.std():.4f}")
    print("PASS" if mad < 5e-3 else "FAIL")
    sys.exit(0 if mad < 5e-3 else 1)

main()
