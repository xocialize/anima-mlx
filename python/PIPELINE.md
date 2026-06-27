# Anima MLX pipeline spec

**All FOUR neural nets are MLX-ported + fp32 parity-locked**: Cosmos DiT (3.1e-5), llm_adapter (2.9e-5),
Qwen3-0.6B TE (6.1e-4), Wan/Qwen-Image VAE (7.5e-6) — see `tests/parity/`.

## STATUS 2026-06-26 — PIPELINE BUILT + e2e PARITY-LOCKED (7/7) ✅
`anima_mlx/pipeline.py` (AnimaPipeline) + `anima_mlx/tokenizer.py` are done. e2e gate
`tests/parity/test_pipeline_parity.py` PASS 7/7 vs a torch oracle (`anima-oracle/pipeline_oracle.py`)
with **injected noise + identical token ids** (no RNG dependence):
- text path (qwen3→adapter→pad512): cond/uncond context **max_abs 2.3e-6 / 1.0e-6** (exact).
- flow sigma schedule: **bit-exact** (0.0).
- step-0 DiT-in-loop + CFG (`v0_cfg`, before chaotic accumulation): **max_abs 5.8e-4, cos 1.0000000**.
- final latent: gated by **cosine 0.9991** (NOT max_abs) — a denoise loop is a chaotic ODE; tiny torch-CPU↔MLX-CPU
  fp differences amplify ~10×/step through cfg×feedback (worse at the coarse 6-step gate's σ0.43→0.009 final dt).
  This is the mlx-porting "quantized/generative" doctrine: gate per-pass cosine + image-validity, not PSNR.
- tokenizer ids: MLX AnimaTokenizer reproduces the oracle ids exactly (`test_tokenizer_parity.py`).
- e2e generate (`generate.py`): 512² / 30 steps / cfg5 → **coherent, prompt-correlated** anime image,
  peak **13.98 GB** fp32 GPU. No checkerboard / tint / garbage.

**bf16 + QUANT + PUBLISH DONE (2026-06-26):** bf16/GPU all 4 comps cos≥0.99995 → publish dtype **bf16**;
quant DiT attn+ff (keep_hi_precision) int8 g128 cos 0.99991 / int4 g64 cos 0.99628; resident bf16 3.91/int8 2.21/
int4 1.38 GB; e2e peak ~14GB(bf16)/~6GB(int4)@512². **PUBLISHED PUBLIC NC** → `huggingface.co/xocialize/anima-mlx`
(canary fresh-download VAE cos 0.999683). `AnimaPipeline.from_pretrained(repo, quant='bf16'|'int4')`. Tools:
`tests/validate_bf16.py`, `tests/validate_quant.py`, `tools/export_mlx_weights.py`.

Remaining: the **Swift port** (Path B; mirror lens-mlx-swift; VAE=lift QwenVAE.swift, rest translate from this rung).
**Phase B (image-quality parity, optional):** match ComfyUI `example.png` exactly — needs its sampler
(**er_sde**, default) + scheduler + steps/cfg. Current pipeline uses deterministic **euler + comfy "normal"
schedule** (faithful flow math, but back-loaded toward high σ → busy/under-resolved subjects vs a tuned er_sde
run). CFG stay-in-range 4–5 (cfg≥7.5 over-guides → streak artifacts). Port er_sde from
comfy `k_diffusion/sampling.py` when chasing beauty parity.

### Resolved open question (timestep scale)
comfy Anima = `ModelType.FLOW` → `CONST` pred + `ModelSamplingDiscreteFlow(shift=3, multiplier=1)`.
`timestep(σ)=σ·multiplier=σ`, and comfy `MiniTrainDIT.Timesteps` is byte-identical to the diffusers
`get_timestep_embedding` the DiT port matches (cat[cos,sin], exp/(half_dim), max_period 1e4, NO internal
rescale). ⇒ **DiT timestep == σ ∈ [0,1] fed directly** (the DiT golden's t=700 was just a probe value).

## Text path (per prompt)
1. Tokenize with BOTH (comfy `AnimaTokenizer`):
   - **Qwen3**: uses the **qwen2.5 tokenizer**; pad=151643, no start/end tokens, weights forced 1.0.
   - **T5**: standard `t5_tokenizer` (32128 SP vocab), no start token.
2. Run **Qwen3-0.6B** (causal) → take **pre-final-norm** last hidden (`qwen3_te.py`, already parity-locked).
3. **llm_adapter**(source=Qwen3 hidden, target_ids=T5 ids) → ×t5xxl_weights (usually 1.0) → **pad seq to 512**
   → this is the DiT `encoder_hidden_states` (cross-attn context). (`llm_adapter.py`, parity-locked.)
4. CFG: run cond + uncond (empty prompt) contexts; `pred = uncond + cfg*(cond-uncond)`, cfg 4–5.

## Sampling — rectified flow (comfy `ModelType.FLOW`)
- `sampling_settings = {shift: 3.0, multiplier: 1.0}`, `latent_format = Wan21` (16-ch).
- `sigma(t) = time_snr_shift(3.0, t) = 3t / (1 + 2t)`, t∈[0,1]; `sigma_max=1.0`.
- `timestep(sigma) = sigma * multiplier = sigma`  → **DiT timestep == sigma ∈ [0,1]**.
  ⚠ VERIFY the timestep scale into the DiT sinusoid against a golden — diffusers/comfy may pre-scale
  (our DiT golden used t=700). comfy `Timesteps` = `cat([cos,sin])`, exponent/(half_dim) — matches our
  `get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)`.
- CONST denoise: `x0 = x - sigma * v` (v = DiT output). `noise_scaling: x = sigma*noise + (1-sigma)*x0`.
- Samplers: **er_sde** (default), euler_a, dpmpp_2m_sde. Build the sigma schedule from sigma_max→0
  (scheduler: normal/simple, optional beta57). Port er_sde from ComfyUI `comfy/k_diffusion/sampling.py`
  (`sample_er_sde`); euler_a is the simpler first target for the e2e gate.
- Steps 30–50, res 512²–1536² (latent = img/8, /16 after 2× patch → seqlen (H/16)(W/16)).

## VAE — Wan/Qwen-Image 16-ch 3D-causal (single-frame)
- Decode latent → image. **Wan21 latent format**: apply latent mean/std (scale/shift) before decode.
  See `comfy/latent_formats.py::Wan21` for the per-channel `latents_mean`/`latents_std` (16 values).
- Python-MLX: port the single-frame decode (T=1) of the Wan VAE (`vae.json` keys: decoder.upsamples/middle,
  WanRMS gamma eps1e-12, [O,I,3,3,3] convs, time_conv unused at T=1).
- Swift: reuse donor `mlxengine-image/PROD/qwen-image-edit-swift/Sources/QwenImageEdit/QwenVAE.swift`
  (native Wan keys; near drop-in + the existing `.resample.1→.0` remap).

## E2E gate
Generate with a fixed prompt/seed; compare to a **ComfyUI golden image** (drag `example.png` workflow).
Quantize-validate later (per-pass cosine, not PSNR). Then bf16 + GPU-stream, then Swift port.

## Files
- `anima_mlx/models/cosmos_dit.py`, `llm_adapter.py`, `qwen3_te.py` (+ `weights.py`)
- oracles: `../anima-oracle/{anima_oracle,adapter_oracle,qwen3_oracle,dump_goldens}.py`
- goldens: `tests/goldens/{dit,adapter,qwen3}/`
