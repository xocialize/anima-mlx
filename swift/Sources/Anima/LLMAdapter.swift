// Anima llm_adapter — 1:1 transpose of anima_mlx/models/llm_adapter.py.
// Bridges Qwen3 hidden (source/KV) + T5 ids (target/Q stream) -> 1024-d DiT cross-attn context.
// Its OWN LLaMA-style rope (rotate_half, θ1e4, head_dim 64) — distinct from the DiT's rope.
// Self-attn: q,k both target-pos. Cross-attn: q=target-pos, k(Qwen3)=CONTEXT-pos. MLP has bias + exact GELU.
import Foundation
import MLX
import MLXNN
import MLXFast

final class AdapterAttention: Module {
    let nHeads: Int, headDim: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(queryDim: Int, contextDim: Int, nHeads: Int, headDim: Int) {
        self.nHeads = nHeads; self.headDim = headDim
        let inner = nHeads * headDim
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(queryDim, inner, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _kProj.wrappedValue = Linear(contextDim, inner, bias: false)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _vProj.wrappedValue = Linear(contextDim, inner, bias: false)
        _oProj.wrappedValue = Linear(inner, queryDim, bias: false)
        super.init()
    }

    /// pe = (cos,sin) for q positions; peCtx for k positions.
    func callAsFunction(_ x: MLXArray, context: MLXArray?, pe: (MLXArray, MLXArray), peCtx: (MLXArray, MLXArray)) -> MLXArray {
        let ctx = context ?? x
        let (b, s, sk) = (x.dim(0), x.dim(1), ctx.dim(1))
        var q = qNorm(qProj(x).reshaped(b, s, nHeads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(ctx).reshaped(b, sk, nHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(ctx).reshaped(b, sk, nHeads, headDim).transposed(0, 2, 1, 3)
        q = applyRope(q, pe.0, pe.1)
        k = applyRope(k, peCtx.0, peCtx.1)
        var o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        o = o.transposed(0, 2, 1, 3).reshaped(b, s, nHeads * headDim)
        return oProj(o)
    }
}

/// mlp = [Linear, GELU, Linear]; checkpoint keys mlp.0/mlp.2 are remapped to fc0/fc2 at load
/// (a numeric-leaf list with a param-less hole can't be a clean Swift Module).
final class AdapterMLP: Module {
    @ModuleInfo(key: "fc0") var fc0: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(dim: Int, inner: Int) {
        _fc0.wrappedValue = Linear(dim, inner, bias: true)
        _fc2.wrappedValue = Linear(inner, dim, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc0(x))) }
}

final class AdapterBlock: Module {
    @ModuleInfo(key: "norm_self_attn") var normSelf: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: AdapterAttention
    @ModuleInfo(key: "norm_cross_attn") var normCross: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: AdapterAttention
    @ModuleInfo(key: "norm_mlp") var normMlp: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: AdapterMLP

    init(sourceDim: Int, modelDim: Int, numHeads: Int = 16, mlpRatio: Float = 4.0) {
        let hd = modelDim / numHeads
        _normSelf.wrappedValue = RMSNorm(dimensions: modelDim, eps: 1e-6)
        _selfAttn.wrappedValue = AdapterAttention(queryDim: modelDim, contextDim: modelDim, nHeads: numHeads, headDim: hd)
        _normCross.wrappedValue = RMSNorm(dimensions: modelDim, eps: 1e-6)
        _crossAttn.wrappedValue = AdapterAttention(queryDim: modelDim, contextDim: sourceDim, nHeads: numHeads, headDim: hd)
        _normMlp.wrappedValue = RMSNorm(dimensions: modelDim, eps: 1e-6)
        _mlp.wrappedValue = AdapterMLP(dim: modelDim, inner: Int(Float(modelDim) * mlpRatio))
        super.init()
    }

    func callAsFunction(_ x0: MLXArray, context: MLXArray, pe: (MLXArray, MLXArray), peCtx: (MLXArray, MLXArray)) -> MLXArray {
        var x = x0 + selfAttn(normSelf(x0), context: nil, pe: pe, peCtx: pe)
        x = x + crossAttn(normCross(x), context: context, pe: pe, peCtx: peCtx)
        x = x + mlp(normMlp(x))
        return x
    }
}

public final class LLMAdapter: Module {
    let modelDim: Int, numHeads: Int
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "blocks") var blocks: [AdapterBlock]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(sourceDim: Int = 1024, targetDim: Int = 1024, modelDim: Int = 1024,
                numLayers: Int = 6, numHeads: Int = 16) {
        self.modelDim = modelDim; self.numHeads = numHeads
        _embed.wrappedValue = Embedding(embeddingCount: 32128, dimensions: targetDim)
        _blocks.wrappedValue = (0 ..< numLayers).map { _ in AdapterBlock(sourceDim: sourceDim, modelDim: modelDim, numHeads: numHeads) }
        _outProj.wrappedValue = Linear(modelDim, targetDim, bias: true)
        _norm.wrappedValue = RMSNorm(dimensions: targetDim, eps: 1e-6)
        super.init()
    }

    /// source = Qwen3 hidden [B,Lq,1024]; targetIds = T5 ids [B,Lt5]. padTo: 512 for DiT, nil for parity.
    public func callAsFunction(_ source: MLXArray, _ targetIds: MLXArray, padTo: Int? = 512) -> MLXArray {
        let headDim = modelDim / numHeads
        var x = embed(targetIds).asType(source.dtype)
        let pe = ropeTables(x.dim(1), headDim, 10000.0)
        let peCtx = ropeTables(source.dim(1), headDim, 10000.0)
        for b in blocks { x = b(x, context: source, pe: pe, peCtx: peCtx) }
        var out = norm(outProj(x))
        if let padTo, out.dim(1) < padTo {
            out = padded(out, widths: [IntOrPair([0, 0]), IntOrPair([0, padTo - out.dim(1)]), IntOrPair([0, 0])])
        }
        return out
    }
}
