# anima-mlx

Python/MLX reference port of **Anima** ([circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)) —
Cosmos-Predict2-2B anime T2I — for Apple Silicon. This is the parity-locked
Python rung behind [anima-mlx-swift](https://github.com/xocialize/anima-mlx-swift);
published MLX weights at [xocialize/anima-mlx](https://huggingface.co/xocialize/anima-mlx) (non-commercial).

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/python generate.py --prompt "masterpiece, best quality, 1girl, ..." \
  --height 1024 --width 1024 --steps 30 --cfg 5 --out out.png
```

Four parity-locked components (Cosmos DiT, llm_adapter, Qwen3-0.6B TE, Wan VAE)
+ e2e gates in `tests/parity/`. **Defaults matter for this model** (learned the
hard way — see PIPELINE.md "STATUS UPDATE 2026-07-22"): sampler **er_sde**
(euler produces structural garbage at every size), quality-tag negative prompt
(empty neg collapses ≥1024²), Qwen3 TE tapped **post**-final-RMSNorm, comfy
`\(`/`(text:w)` prompt parsing. Ground truth is a headless ComfyUI render —
parity to a self-written oracle alone certified the wrong computation for weeks.

License: port code MIT; weights non-commercial (see LICENSE).
