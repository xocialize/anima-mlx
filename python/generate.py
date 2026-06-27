"""Anima MLX text-to-image — end-to-end generate a PNG from a prompt.

Usage:
  .venv/bin/python generate.py --prompt "1girl, anime, masterpiece" --steps 30 --cfg 5 \
      --height 512 --width 512 --seed 1234 --out out.png [--cpu]
"""
import argparse
import os
import numpy as np
import mlx.core as mx
from PIL import Image

from anima_mlx.pipeline import AnimaPipeline
from anima_mlx.tokenizer import AnimaTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "anima-oracle", "weights", "split_files")
DIT_CKPT = os.path.join(ORACLE, "diffusion_models", "anima-base-v1.0.safetensors")
QWEN_CKPT = os.path.join(ORACLE, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_CKPT = os.path.join(HERE, "tests", "goldens", "vae", "wan_vae_decoder_mlx.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="1girl, anime, masterpiece, detailed background, soft lighting")
    ap.add_argument("--neg", default="")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="anima_out.png")
    ap.add_argument("--cpu", action="store_true", help="force CPU stream (parity); default GPU")
    args = ap.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    print("[load] building pipeline …")
    pipe = AnimaPipeline.from_checkpoints(DIT_CKPT, QWEN_CKPT, VAE_CKPT)
    tok = AnimaTokenizer()

    cond = pipe.encode_context(*tok.encode(args.prompt))
    unc = pipe.encode_context(*tok.encode(args.neg))
    print(f"[gen] {args.width}x{args.height} steps={args.steps} cfg={args.cfg} seed={args.seed}")
    img = pipe.generate(cond, unc, height=args.height, width=args.width, steps=args.steps,
                        cfg=args.cfg, seed=args.seed, verbose=True)
    mx.eval(img)
    arr = (np.asarray(img[0], np.float32) * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(args.out)
    print(f"[done] peak mem {mx.get_peak_memory()/1e9:.2f} GB → {args.out}")


if __name__ == "__main__":
    main()
