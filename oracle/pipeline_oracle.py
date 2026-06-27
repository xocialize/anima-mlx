"""Anima END-TO-END PyTorch oracle pipeline → e2e parity golden.

Runs the full Anima T2I pipeline on the torch oracles (diffusers Cosmos DiT,
adapter, Qwen3-0.6B TE) with a FIXED numpy-injected noise so the MLX port can be
compared op-for-op without RNG incompatibility. Tokenizes with real transformers
tokenizers (Qwen2.5 + T5-v1_1) and dumps the token ids so the MLX side consumes
identical ids (tokenizer parity is checked separately).

Sampler = comfy ModelType.FLOW: CONST prediction + ModelSamplingDiscreteFlow
(shift=3, multiplier=1) → sigma(t)=3t/(1+2t), timestep==sigma∈[0,1]. Deterministic
Euler (no SDE) for a clean gate. CFG: v = v_unc + cfg*(v_cond - v_unc).

The PRIMARY golden is `final_latent` (model-space x0): it validates the whole new
glue (text path + sampler + DiT-in-loop). The VAE is parity-locked separately
(7.5e-6), so the decoded image is produced on the MLX side with the MLX VAE.

Dumps → ../anima-mlx/tests/goldens/pipeline/.
"""
import os
import numpy as np
import torch

from anima_oracle import load_oracle
from adapter_oracle import load_adapter
from qwen3_oracle import load_qwen3

torch.set_grad_enabled(False)
torch.manual_seed(0)

OUT = os.path.join(os.path.dirname(__file__), "..", "anima-mlx", "tests", "goldens", "pipeline")
os.makedirs(OUT, exist_ok=True)

# ---- fixed generation config (small + few steps so CPU fp32 oracle is fast) ----
PROMPT = "1girl, anime, masterpiece, detailed background, soft lighting"
NEG = ""
HEIGHT = 256
WIDTH = 256
STEPS = 6
CFG = 5.0
SEED = 1234
VAE_SPATIAL = 8            # Wan VAE spatial downscale
PAD_TO = 512


def save(name, t):
    arr = t.detach().to(torch.float32).cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    np.save(os.path.join(OUT, name + ".npy"), arr)
    print(f"  saved {name:18s} {tuple(np.asarray(arr).shape)} {np.asarray(arr).dtype}")


# ----------------------------------------------------------------- tokenizers
def build_tokenizers():
    from transformers import AutoTokenizer, T5Tokenizer
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    t5 = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    return qwen, t5


def tokenize(qwen_tok, t5_tok, text):
    """comfy AnimaTokenizer rules: qwen=raw BPE (no specials, min_length 1 → pad 151643);
    t5=BPE + trailing eos(1). Returns (qwen_ids[list], t5_ids[list])."""
    qids = qwen_tok(text, add_special_tokens=True)["input_ids"]
    if len(qids) == 0:
        qids = [151643]                       # pad_token, min_length=1
    tids = t5_tok(text)["input_ids"]          # already ends with eos=1; '' -> [1]
    return qids, tids


# ----------------------------------------------------------------- text path
def qwen3_prenorm(qwen_model, ids):
    """pre-final-norm last hidden (comfy layer_norm_hidden_state=False)."""
    cap = {}
    h = qwen_model.norm.register_forward_hook(lambda m, i, o: cap.update(pre=i[0]))
    qwen_model(input_ids=ids)
    h.remove()
    return cap["pre"]                          # [1, Lq, 1024]


def encode_context(qwen_model, adapter, qwen_ids, t5_ids):
    qids = torch.tensor([qwen_ids], dtype=torch.long)
    tids = torch.tensor([t5_ids], dtype=torch.int)
    src = qwen3_prenorm(qwen_model, qids)               # [1, Lq, 1024]
    out = adapter(src, tids)                            # [1, Lt5, 1024]
    if out.shape[1] < PAD_TO:                           # pad to 512 (preprocess_text_embeds)
        out = torch.nn.functional.pad(out, (0, 0, 0, PAD_TO - out.shape[1]))
    return out                                          # [1, 512, 1024]


# ----------------------------------------------------------------- sampler
def time_snr_shift(alpha, t):
    return alpha * t / (1 + (alpha - 1) * t)


def flow_sigmas(steps, shift=3.0, mult=1.0):
    """comfy 'normal' scheduler for ModelSamplingDiscreteFlow. sigma_max=1.0."""
    sigma_max = time_snr_shift(shift, 1000.0 / 1000.0 * mult)          # t=1 -> 1.0
    sigma_min = time_snr_shift(shift, 1.0 / 1000.0 * mult)            # t=1/1000
    start = sigma_max * mult                                          # timestep(sigma)=sigma*mult
    end = sigma_min * mult
    ts = np.linspace(start, end, steps)
    sigs = [time_snr_shift(shift, t / mult) for t in ts]             # sigma(timestep)
    sigs.append(0.0)
    return np.asarray(sigs, dtype=np.float64)


def dit_v(dit, x, sigma, ctx, pad):
    B = x.shape[0]
    t = torch.full((B,), float(sigma), dtype=torch.float32)
    return dit(hidden_states=x, timestep=t, encoder_hidden_states=ctx,
               padding_mask=pad, return_dict=False)[0]


def sample(dit, noise, cond_ctx, uncond_ctx, sigmas, cfg, probe=None):
    Hl, Wl = noise.shape[-2], noise.shape[-1]
    x = (sigmas[0] * noise).to(torch.float32)            # noise_scaling: sigma0*noise (latent=0)
    ctx = torch.cat([cond_ctx, uncond_ctx], dim=0)       # batch cond+uncond
    pad = torch.zeros(1, 1, Hl, Wl, dtype=torch.float32)  # diffusers repeats by batch_size
    for i in range(len(sigmas) - 1):
        s = sigmas[i]
        xb = torch.cat([x, x], dim=0)
        v = dit_v(dit, xb, s, ctx, pad)
        v_cond, v_unc = v[:1], v[1:]
        v_cfg = v_unc + cfg * (v_cond - v_unc)
        if probe is not None and i == 0:
            probe["v0_cfg"] = v_cfg.clone()
        x = x + v_cfg * (sigmas[i + 1] - s)              # euler flow step
        if probe is not None:
            probe[f"x_step{i}"] = x.clone()
        print(f"  step {i}: sigma {s:.4f} -> {sigmas[i+1]:.4f}  x std {float(x.std()):.4f}")
    return x                                              # model-space x0


def main():
    print("[oracle] loading models …")
    dit, _ = load_oracle(dtype=torch.float32)
    adapter = load_adapter(dtype=torch.float32)
    qwen = load_qwen3(dtype=torch.float32)
    qwen_tok, t5_tok = build_tokenizers()

    qids, tids = tokenize(qwen_tok, t5_tok, PROMPT)
    uqids, utids = tokenize(qwen_tok, t5_tok, NEG)
    print(f"[tok] cond qwen={qids}\n      cond t5={tids}\n      unc qwen={uqids} unc t5={utids}")

    cond_ctx = encode_context(qwen, adapter, qids, tids)
    uncond_ctx = encode_context(qwen, adapter, uqids, utids)
    print(f"[ctx] cond {tuple(cond_ctx.shape)} std {float(cond_ctx.std()):.4f} | "
          f"uncond std {float(uncond_ctx.std()):.4f}")

    Hl, Wl = HEIGHT // VAE_SPATIAL, WIDTH // VAE_SPATIAL
    rng = np.random.default_rng(SEED)
    noise_np = rng.standard_normal((1, 16, 1, Hl, Wl)).astype(np.float32)
    noise = torch.from_numpy(noise_np)

    sigmas = flow_sigmas(STEPS)
    print(f"[sched] sigmas {np.round(sigmas, 4)}")
    probe = {}
    x0 = sample(dit, noise, cond_ctx, uncond_ctx, sigmas, CFG, probe=probe)
    print(f"[done] final_latent std {float(x0.std()):.4f}")
    save("v0_cfg", probe["v0_cfg"])
    for i in range(STEPS):
        save(f"x_step{i}", probe[f"x_step{i}"])

    # dump everything the MLX e2e parity test needs
    save("noise", noise_np)
    np.save(os.path.join(OUT, "cond_qwen_ids.npy"), np.asarray(qids, np.int32))
    np.save(os.path.join(OUT, "cond_t5_ids.npy"), np.asarray(tids, np.int32))
    np.save(os.path.join(OUT, "uncond_qwen_ids.npy"), np.asarray(uqids, np.int32))
    np.save(os.path.join(OUT, "uncond_t5_ids.npy"), np.asarray(utids, np.int32))
    save("cond_context", cond_ctx)
    save("uncond_context", uncond_ctx)
    save("sigmas", sigmas.astype(np.float32))
    save("final_latent", x0)
    with open(os.path.join(OUT, "config.txt"), "w") as f:
        f.write(f"PROMPT={PROMPT!r}\nNEG={NEG!r}\nHEIGHT={HEIGHT}\nWIDTH={WIDTH}\n"
                f"STEPS={STEPS}\nCFG={CFG}\nSEED={SEED}\nPAD_TO={PAD_TO}\n")
    print("[oracle] e2e golden dumped to", OUT)


if __name__ == "__main__":
    main()
