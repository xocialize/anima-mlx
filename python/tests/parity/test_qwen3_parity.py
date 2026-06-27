import os, sys
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from anima_mlx.models.qwen3_te import Qwen3TextEncoder, load_qwen3_te
GOLD = os.path.join(ROOT, "tests", "goldens", "qwen3")
CKPT = os.path.join(ROOT, "..", "anima-oracle", "weights", "split_files",
                    "text_encoders", "qwen_3_06b_base.safetensors")
m = Qwen3TextEncoder()
m, n = load_qwen3_te(m, CKPT); m.eval(); mx.eval(m.parameters())
ids = mx.array(np.load(os.path.join(GOLD, "in_ids.npy")))
out = np.asarray(m(ids), np.float32)
want = np.load(os.path.join(GOLD, "hidden_prenorm.npy")).astype(np.float32)
mad = float(np.max(np.abs(out - want)))
print(f"[load] {n} tensors  | hidden_prenorm max_abs={mad:.2e}  shape {out.shape}")
print("PASS" if mad < 2e-3 else "FAIL")
sys.exit(0 if mad < 2e-3 else 1)
