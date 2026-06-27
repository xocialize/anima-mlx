// Anima dual tokenizer — transpose of anima_mlx/tokenizer.py / comfy text_encoders/anima.py.
//   Qwen3 path: Qwen2.5 tokenizer, raw BPE, NO specials, pad 151643 (empty -> [151643]).
//   T5 path   : T5-v1.1 SentencePiece, BPE + trailing eos(1) (empty -> [1]).
// Both weights forced to 1.0 → dropped; the adapter t5xxl_weights multiply is a no-op.
import Foundation
import Tokenizers
import Hub

public struct AnimaTokenizer {
    public static let qwenPad = 151643
    let qwen: any Tokenizer
    let t5: any Tokenizer

    public static func load(qwenRepo: String = "Qwen/Qwen2.5-0.5B",
                            t5Repo: String = "google-t5/t5-base") async throws -> AnimaTokenizer {
        let qwen = try await AutoTokenizer.from(pretrained: qwenRepo)
        let t5 = try await AutoTokenizer.from(pretrained: t5Repo)
        return AnimaTokenizer(qwen: qwen, t5: t5)
    }

    /// Returns (qwenIds, t5Ids) matching comfy AnimaTokenizer.
    public func encode(_ text: String) -> (qwen: [Int], t5: [Int]) {
        var q = qwen.encode(text: text, addSpecialTokens: false)  // raw BPE, no specials
        if q.isEmpty { q = [Self.qwenPad] }                        // min_length 1 -> pad
        let t = t5.encode(text: text, addSpecialTokens: true)      // SentencePiece adds trailing eos=1
        return (q, t)
    }
}
