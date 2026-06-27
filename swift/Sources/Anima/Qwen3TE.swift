// Qwen3-0.6B as Anima's text encoder — 1:1 transpose of anima_mlx/models/qwen3_te.py.
// Causal; returns the LAST decoder-layer hidden BEFORE the final norm (comfy
// layer_norm_hidden_state=False). GQA 16:8, head_dim 128 (≠ hidden 1024), QK-norm, rope θ1e6.
import Foundation
import MLX
import MLXNN
import MLXFast

public struct Qwen3Config {
    public var vocabSize = 151936
    public var hiddenSize = 1024
    public var intermediateSize = 3072
    public var numHiddenLayers = 28
    public var numAttentionHeads = 16
    public var numKeyValueHeads = 8
    public var headDim = 128
    public var ropeTheta: Float = 1_000_000.0
    public var rmsNormEps: Float = 1e-6
    public init() {}
}

/// cos/sin tables [1,1,S,headDim], built with float32 MLX ops to match the Python-MLX rung.
func ropeTables(_ seq: Int, _ headDim: Int, _ theta: Float) -> (MLXArray, MLXArray) {
    let half = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })  // [hd/2]
    let invFreq = 1.0 / pow(MLXArray(theta), half / Float(headDim))
    let pos = MLXArray(0 ..< seq).asType(.float32)                              // [S]
    let f = pos.reshaped(seq, 1) * invFreq.reshaped(1, headDim / 2)             // [S, hd/2]
    let emb = concatenated([f, f], axis: -1)                                    // [S, hd]
    return (cos(emb).reshaped(1, 1, seq, headDim), sin(emb).reshaped(1, 1, seq, headDim))
}

/// rotate_half rope: x*cos + concat([-x2, x1])*sin.
func applyRope(_ x: MLXArray, _ cosT: MLXArray, _ sinT: MLXArray) -> MLXArray {
    let d = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< d]
    let x2 = x[.ellipsis, d ..< (2 * d)]
    let rot = concatenated([-x2, x1], axis: -1)
    return x * cosT + rot * sinT
}

final class Qwen3Attention: Module {
    let nh: Int, nkv: Int, hd: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: Qwen3Config) {
        nh = c.numAttentionHeads; nkv = c.numKeyValueHeads; hd = c.headDim
        scale = pow(Float(hd), -0.5)
        _qProj.wrappedValue = Linear(c.hiddenSize, nh * hd, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, nkv * hd, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, nkv * hd, bias: false)
        _oProj.wrappedValue = Linear(nh * hd, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: hd, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: hd, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ cosT: MLXArray, _ sinT: MLXArray, _ mask: MLXArray) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, s, nh, hd)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, s, nkv, hd)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, s, nkv, hd).transposed(0, 2, 1, 3)
        q = applyRope(q, cosT, sinT)
        k = applyRope(k, cosT, sinT)
        var o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        o = o.transposed(0, 2, 1, 3).reshaped(b, s, nh * hd)
        return oProj(o)
    }
}

final class Qwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: Qwen3Config) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class Qwen3Layer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLN: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
    @ModuleInfo(key: "post_attention_layernorm") var postLN: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP
    init(_ c: Qwen3Config) {
        _inputLN.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _selfAttn.wrappedValue = Qwen3Attention(c)
        _postLN.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = Qwen3MLP(c)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, _ cosT: MLXArray, _ sinT: MLXArray, _ mask: MLXArray) -> MLXArray {
        var x = x + selfAttn(inputLN(x), cosT, sinT, mask)
        x = x + mlp(postLN(x))
        return x
    }
}

/// Wraps the `model.*` namespace (embed_tokens/layers/norm); returns PRE-final-norm last hidden.
public final class Qwen3TextEncoder: Module {
    let c: Qwen3Config
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm  // held but NOT applied to the output

    public init(_ c: Qwen3Config = Qwen3Config()) {
        self.c = c
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in Qwen3Layer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let s = inputIds.dim(1)
        var x = embedTokens(inputIds)
        let (cosT, sinT) = ropeTables(s, c.headDim, c.ropeTheta)
        let mask = causalAdditiveMask(s, dtype: x.dtype)
        for layer in layers { x = layer(x, cosT, sinT, mask) }
        return x  // pre-final-norm
    }
}

/// Additive causal mask [S,S]: 0 on/below diagonal, -inf above.
func causalAdditiveMask(_ n: Int, dtype: DType) -> MLXArray {
    let idx = MLXArray(0 ..< n)
    let m = idx.reshaped(n, 1) .< idx.reshaped(1, n)        // true above diagonal
    return MLX.where(m, MLXArray(Float(-1e9)), MLXArray(Float(0))).asType(dtype)
}
