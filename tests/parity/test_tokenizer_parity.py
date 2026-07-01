"""Tokenizer id parity: MLX AnimaTokenizer must reproduce the ids the torch oracle
pipeline dumped (which the e2e golden was generated from). Reads config.txt for the
exact prompts."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from anima_mlx.tokenizer import AnimaTokenizer  # noqa: E402

GOLD = os.path.join(ROOT, "tests", "goldens", "pipeline")


def cfg_val(key):
    for line in open(os.path.join(GOLD, "config.txt")):
        if line.startswith(key + "="):
            return eval(line.split("=", 1)[1].strip())
    raise KeyError(key)


def chk(name, got, want):
    ok = list(got) == list(want)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:14s} {list(got)}")
    if not ok:
        print(f"        want    {list(want)}")
    return ok


def main():
    tok = AnimaTokenizer()
    res = []
    for tag, prompt, qf, tf in [
        ("cond", cfg_val("PROMPT"), "cond_qwen_ids", "cond_t5_ids"),
        ("uncond", cfg_val("NEG"), "uncond_qwen_ids", "uncond_t5_ids"),
    ]:
        qids, tids = tok.encode(prompt)
        res.append(chk(f"{tag} qwen", qids, np.load(os.path.join(GOLD, qf + ".npy"))))
        res.append(chk(f"{tag} t5", tids, np.load(os.path.join(GOLD, tf + ".npy"))))
    print("PASS" if all(res) else "FAIL")
    sys.exit(0 if all(res) else 1)


if __name__ == "__main__":
    main()
