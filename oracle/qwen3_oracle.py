"""Qwen3-0.6B text-encoder oracle. Returns the pre-final-norm last-layer hidden
(comfy layer_norm_hidden_state=False). Causal. Dumps goldens for MLX parity."""
import os
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3Model

CKPT = "weights/split_files/text_encoders/qwen_3_06b_base.safetensors"
OUT = os.path.join(os.path.dirname(__file__), "..", "anima-mlx", "tests", "goldens", "qwen3")
os.makedirs(OUT, exist_ok=True)

CONFIG = Qwen3Config(
    vocab_size=151936, hidden_size=1024, intermediate_size=3072,
    num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
    head_dim=128, rope_theta=1000000.0, rms_norm_eps=1e-6,
    max_position_embeddings=40960, tie_word_embeddings=True, attention_bias=False,
)


def load_qwen3(dtype=torch.float32):
    m = Qwen3Model(CONFIG)
    st = load_file(CKPT)
    st = {k[len("model."):] if k.startswith("model.") else k: v for k, v in st.items()}
    missing, unexpected = m.load_state_dict(st, strict=False)
    miss = [k for k in missing if "rotary" not in k]
    assert not miss, f"missing {miss[:8]}"
    assert not unexpected, f"unexpected {list(unexpected)[:8]}"
    return m.to(dtype).eval()


if __name__ == "__main__":
    torch.manual_seed(0)
    m = load_qwen3()
    ids = torch.randint(0, 151936, (1, 18))
    np.save(os.path.join(OUT, "in_ids.npy"), ids.cpu().numpy().astype(np.int32))
    cap = {}
    m.norm.register_forward_hook(lambda mod, i, o: cap.update(pre=i[0]))
    with torch.no_grad():
        m(input_ids=ids)
    pre_norm_last = cap["pre"]  # input to model.norm = comfy layer_norm_hidden_state=False
    np.save(os.path.join(OUT, "hidden_prenorm.npy"),
            pre_norm_last.to(torch.float32).cpu().numpy())
    print("[qwen3 goldens] hidden_prenorm", tuple(pre_norm_last.shape),
          "std %.3f" % float(pre_norm_last.std()))
