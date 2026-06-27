# Anima — architecture map (for the MLX port)

Anima = CircleStone Labs anime/illustration T2I diffusion model. ComfyUI single-file format.
Built on **NVIDIA Cosmos-Predict2-2B-Text2Image**, with the T5 text path replaced by
**Qwen3-0.6B base + a custom `llm_adapter`**.

## Components (all confirmed from safetensors headers)

| Component | File | Params | Reuse status |
|---|---|---|---|
| DiT | `split_files/diffusion_models/anima-base-v1.0.safetensors` | 2.091B (685 bf16 tensors) | **net-new** — oracle = diffusers `CosmosTransformer3DModel` |
| Text encoder | `split_files/text_encoders/qwen_3_06b_base.safetensors` | 0.596B (310 tensors) | **reuse** — stock Qwen3-0.6B base, mlx-swift-lm / mlx-lm loads it |
| VAE | `split_files/vae/qwen_image_vae.safetensors` | 0.127B (194 tensors) | **reuse** — Qwen-Image/Wan 16-ch 3D-causal VAE; Swift donor `qwen-image-edit-swift/QwenVAE.swift` |

There are also `anima-preview*.safetensors` (alt DiT checkpoints, same shape) — ignore for v1.

## DiT = Cosmos-Predict2-2B (diffusers `CosmosTransformer3DModel`)

Confirmed diffusers config (defaults overridden by the 2B checkpoint where noted):
- `num_layers = 28`, hidden = **2048**, `num_attention_heads = 16` (2048/128), `attention_head_dim = 128`
- `mlp_ratio = 4.0` (8192), QK-norm RMSNorm per head (`q_norm`/`k_norm` shape [128])
- `adaln_lora_dim = 256` → per-block `adaln_modulation_{self_attn,cross_attn,mlp}` = `[256,2048]` (AdaLN-**LoRA**)
- `patch_size = (1,2,2)`, `in_channels = 16`, `concat_padding_mask = True`
  → `x_embedder.proj` input **68** = 16ch·(1·2·2)=64 + paddingmask 1ch·(2·2)=4
- `text_embed_dim = 1024` → cross-attn k/v project 1024→2048 (`cross_attn.k_proj/v_proj` = [2048,1024])
- `extra_pos_embed_type = learnable`, `rope_scale = (2.0,1.0,1.0)` (CosmosRotaryPosEmbed, 3D)
- `t_embedder`: sinusoid→2048→2048, `linear_2`→6144 (=3·2048 global AdaLN base); `t_embedding_norm` [2048]
- `final_layer`: adaln_modulation [256,2048]×2, linear [64,2048] → unpatchify 16ch latent (64 = 16·2·2)

ComfyUI-native key namespace is `net.blocks.{i}.{self_attn,cross_attn,mlp,adaln_modulation_*}`,
`net.{x_embedder,t_embedder,t_embedding_norm,final_layer}`. diffusers uses `transformer_blocks.*`
— a key remap is needed between checkpoint ↔ oracle (diffusers ships
`scripts/convert_cosmos_to_diffusers.py`).

## llm_adapter — the ONE net-new piece (no public oracle)

Anima-custom bridge: Qwen3-0.6B hidden (1024) → DiT cross-attn context (1024). Keys under
`net.llm_adapter.*`:
- `embed.weight [32128,1024]` — **T5 vocab size** embedding (32128). Seeds adapter queries?
- 6 blocks, dim 1024, head_dim 64 (q_norm/k_norm [64] → 16 heads):
  - `self_attn` (q/k/v/o_proj [1024,1024]), `cross_attn` (q/k/v/o_proj [1024,1024]) — k/v from Qwen3 hidden
  - `mlp` 2-layer 4x (4096) **with bias**, pre-norm RMSNorm (`norm_self_attn/norm_cross_attn/norm_mlp`)
- `norm [1024]`, `out_proj [1024,1024]+bias`

**RESOLVED forward** (confirmed by maintainer/community — HF discussions + comfyui-qwen35-anima):
The SAME prompt is tokenized by BOTH tokenizers:
- **T5 tokenizer** → token ids → `embed` [32128,1024] → the adapter's **self-attn / query stream**
- **Qwen3 tokenizer** → Qwen3-0.6B → last_hidden_state (1024) → **K/V for the adapter cross-attn**
- 6× [pre-norm self_attn → pre-norm cross_attn(Q=stream, KV=Qwen3 hidden) → pre-norm MLP]
  → final `norm` → `out_proj` (1024→1024 +bias)
- Output: **1024-d sequence, padded to length 512** → fed to the DiT as `encoder_hidden_states`.

Dependencies this implies: a **T5 tokenizer** (google/t5-v1_1, 32128 SP vocab) in addition to the
Qwen3 tokenizer. No RoPE tensors checkpointed → adapter attention is non-rotary (verify vs ref).
Reference forward exists in ComfyUI core (Comfy Org collaboration) — use as the adapter oracle;
final gate = ComfyUI golden image.

## Text encoder = stock Qwen3-0.6B base

28 layers, hidden 1024, GQA 16:8 (q_proj [2048,1024]=16·128, k/v_proj [1024,1024]=8·128), QK-norm [128],
SwiGLU (gate/up [3072,1024], down), vocab 151936, `model.norm`. No lm_head → use last_hidden_state.

## VAE = Qwen-Image / Wan 16-ch 3D-causal

Native Wan key convention (`encoder.downsamples`/`decoder.upsamples`/`residual`/`resample`/`shortcut`/
`time_conv`/`middle.to_qkv`), WanRMS gamma norm (eps 1e-12). Single-frame (T=1) image path.
Donor: `mlxengine-image/PROD/qwen-image-edit-swift/Sources/QwenImageEdit/QwenVAE.swift` (+ Weights.swift
remap). 16 latent channels (conv2→16).

## Pipeline params (from README)

Resolution 512²–1536², steps 30–50, CFG 4–5. Samplers: `er_sde` (default), `euler_a`, `dpmpp_2m_sde_gpu`.
Optional beta57 scheduler. `example.png` = embedded ComfyUI workflow; `anima_comparison.json` = comparison graph.

## License (soft-check result)

- Anima weights: **CircleStone Labs Non-Commercial License** → NC. Outputs (images) commercial-OK; weights not.
- Base: NVIDIA Cosmos Open Model License (permissive, commercial-OK) — requires "Built on NVIDIA Cosmos"
  attribution + safety clause.
- Net: fine for personal/research/community use; **fails a strict permissive engine gate** →
  ModelPackage must declare `licenseClass = .nonCommercial` (C7). Build gated-but-flagged, not gate-stripped.
