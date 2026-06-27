"""Anima dual tokenizer in MLX-land — transpose of comfy text_encoders/anima.py.

Two HF tokenizers (transformers, tokenizer-only — no torch needed):
  • Qwen3 path  : Qwen2.5 tokenizer, raw BPE, NO start/end token, pad=151643,
                  min_length 1 (empty prompt → [151643]).
  • T5 path     : T5-v1_1 SentencePiece (32128 rows), BPE + trailing eos(1),
                  empty prompt → [1].
Both weights forced to 1.0 (AnimaTokenizer) → we drop weights entirely; the adapter
multiply by t5xxl_weights is a no-op at 1.0.
"""
from __future__ import annotations

QWEN_PAD = 151643
QWEN_REPO = "Qwen/Qwen2.5-0.5B"
T5_REPO = "t5-base"                # same SentencePiece vocab (32128) as comfy t5_tokenizer / t5-v1_1


class AnimaTokenizer:
    def __init__(self, qwen_repo: str = QWEN_REPO, t5_repo: str = T5_REPO):
        from transformers import AutoTokenizer, T5Tokenizer
        self.qwen = AutoTokenizer.from_pretrained(qwen_repo)
        self.t5 = T5Tokenizer.from_pretrained(t5_repo, legacy=False)

    def encode(self, text: str):
        """Returns (qwen_ids, t5_ids) as python int lists, matching comfy AnimaTokenizer."""
        qids = self.qwen(text, add_special_tokens=True)["input_ids"]
        if len(qids) == 0:                 # min_length 1 → pad token
            qids = [QWEN_PAD]
        tids = self.t5(text)["input_ids"]  # SentencePiece adds trailing eos=1; '' → [1]
        return list(qids), list(tids)
