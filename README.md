# anima-mlx

Apple-MLX port of **[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)** — an
anime/illustration text-to-image model (NVIDIA Cosmos-Predict2-2B DiT + Qwen3-0.6B → `llm_adapter`
conditioning + Qwen-Image/Wan 16-channel VAE). This repo holds the **Python-MLX** reference port,
parity-locked to the PyTorch original. The **Swift-MLX** engine package lives in its own repo:
**[`xocialize/anima-mlx-swift`](https://github.com/xocialize/anima-mlx-swift)** (`textToImage`
`AnimaT2IPackage`), parity-locked to this Python rung.

> **Non-Commercial.** The Anima weights are licensed Non-Commercial (CircleStone Labs); the base
> denoiser is "Built on NVIDIA Cosmos" (NVIDIA Cosmos Open Model License). Personal / research use
> only. The port **code** here is MIT. See [LICENSE](LICENSE).

Weights (bf16 + int4, NC-flagged): **https://huggingface.co/xocialize/anima-mlx**

## Layout

| dir | what |
|---|---|
| `python/` | Pure Python-MLX port (`anima_mlx/`): Cosmos DiT, llm_adapter, Qwen3 TE, Wan VAE, pipeline, tokenizer, `generate.py`. e2e parity-locked 7/7 (`tests/parity/`). |
| `oracle/` | PyTorch oracles (diffusers Cosmos + adapter + Qwen3) that generate the parity goldens. |

The Swift-MLX port + engine-conformant `AnimaT2IPackage` now lives in
**[`xocialize/anima-mlx-swift`](https://github.com/xocialize/anima-mlx-swift)**.

## Parity ([anima-mlx-swift](https://github.com/xocialize/anima-mlx-swift) vs this Python-MLX rung / PT goldens)

| component | cosine | max_abs |
|---|---|---|
| Cosmos DiT | 1.000000 | 3.1e-5 |
| llm_adapter | 1.000000 | 2.9e-6 |
| Qwen3-0.6B TE | 1.000000 | 6.1e-4 |
| Wan VAE | 1.000000 | 6.7e-6 |
| **e2e pipeline** | step-0 v0_cfg 0.9999996 · final latent **0.999105** | (== Python bit-for-bit) |
| int4 transformer | per-pass 0.99619 | |

## Sampling

ComfyUI `ModelType.FLOW`: `CONST` prediction + `ModelSamplingDiscreteFlow(shift=3, multiplier=1)` →
`sigma(t) = 3t/(1+2t)`, **DiT timestep == sigma ∈ [0,1]**, Wan21 latent denorm before decode. CFG 4–5.
Tokenizers: Qwen2.5 (raw BPE, pad 151643) + T5-v1.1 SentencePiece (trailing eos).

## Quick start

**Python:** `cd python && pip install -e . && python generate.py --prompt "1girl, anime, masterpiece"`
(loads the published weights via `AnimaPipeline.from_pretrained("xocialize/anima-mlx")`).

**Swift:** see **[`xocialize/anima-mlx-swift`](https://github.com/xocialize/anima-mlx-swift)** —
`swift run anima-cli --generate "1girl, anime, masterpiece" <weights-dir> out.png`; engine
integration `MLXServeEngine.register(.of(AnimaT2IPackage.self), configuration:)`.

## Credits

Anima — CircleStone Labs (NC) · Cosmos-Predict2 — NVIDIA · Qwen3 / Wan VAE — Alibaba · MLX port — xocialize.
