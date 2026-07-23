"""Anima dual tokenizer in MLX-land — transpose of comfy text_encoders/anima.py.

Two HF tokenizers (transformers, tokenizer-only — no torch needed):
  • Qwen3 path  : Qwen2.5 tokenizer, raw BPE, NO start/end token, pad=151643,
                  min_length 1 (empty prompt → [151643]).
  • T5 path     : T5-v1_1 SentencePiece (32128 rows), BPE + trailing eos(1),
                  empty prompt → [1].
Prompt-weight syntax (comfy sd1_clip): "(text:1.2)" scales weight, bare "(text)"
×1.1, "\\(" / "\\)" are literal parens (danbooru escapes). Qwen weights are forced
to 1.0 (AnimaTokenizer); T5 weights are returned for the adapter's t5xxl_weights
multiply (encode() drops them — use encode_weighted() when any weight ≠ 1.0).
"""
from __future__ import annotations

QWEN_PAD = 151643
QWEN_REPO = "Qwen/Qwen2.5-0.5B"
T5_REPO = "t5-base"                # same SentencePiece vocab (32128) as comfy t5_tokenizer / t5-v1_1


def _parse_parentheses(string: str):
    # comfy sd1_clip.parse_parentheses, verbatim semantics
    result, current, nesting = [], "", 0
    for ch in string:
        if ch == "(":
            if nesting == 0:
                if current:
                    result.append(current)
                current = "("
            else:
                current += ch
            nesting += 1
        elif ch == ")":
            nesting -= 1
            if nesting == 0:
                result.append(current + ")")
                current = ""
            else:
                current += ch
        else:
            current += ch
    if current:
        result.append(current)
    return result


def _token_weights(string: str, current_weight: float):
    # comfy sd1_clip.token_weights: "(x:1.2)" sets weight, bare "(x)" ×1.1
    out = []
    for x in _parse_parentheses(string):
        weight = current_weight
        if len(x) >= 2 and x[0] == "(" and x[-1] == ")":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1:])
                    x = x[:xx]
                except ValueError:
                    pass
            out += _token_weights(x, weight)
        else:
            out.append((x, current_weight))
    return out


def _escape(text: str) -> str:
    return text.replace("\\)", "\0\1").replace("\\(", "\0\2")


def _unescape(text: str) -> str:
    return text.replace("\0\1", ")").replace("\0\2", "(")


class AnimaTokenizer:
    def __init__(self, qwen_repo: str = QWEN_REPO, t5_repo: str = T5_REPO):
        from transformers import AutoTokenizer, T5Tokenizer
        self.qwen = AutoTokenizer.from_pretrained(qwen_repo)
        self.t5 = T5Tokenizer.from_pretrained(t5_repo, legacy=False)

    def encode_weighted(self, text: str):
        """Returns (qwen_ids, t5_ids, t5_weights) matching comfy AnimaTokenizer:
        weighted segments tokenized separately and concatenated; qwen weights
        forced 1.0; t5 gets one trailing eos(1) with weight 1.0."""
        segments = _token_weights(_escape(text), 1.0)
        qids: list[int] = []
        tids: list[int] = []
        tweights: list[float] = []
        for seg, w in segments:
            seg = _unescape(seg)
            if not seg:
                continue
            qids += list(self.qwen(seg, add_special_tokens=True)["input_ids"])
            seg_t = list(self.t5(seg)["input_ids"])
            if seg_t and seg_t[-1] == 1:          # strip per-segment eos; re-added once
                seg_t = seg_t[:-1]
            tids += seg_t
            tweights += [w] * len(seg_t)
        if len(qids) == 0:                        # min_length 1 → pad token
            qids = [QWEN_PAD]
        tids.append(1)                            # single trailing eos
        tweights.append(1.0)
        return qids, tids, tweights

    def encode(self, text: str):
        """Returns (qwen_ids, t5_ids) as python int lists, matching comfy AnimaTokenizer."""
        qids, tids, _ = self.encode_weighted(text)
        return qids, tids
