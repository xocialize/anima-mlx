// Cosmos-Predict2-2B DiT — 1:1 transpose of anima_mlx/models/cosmos_dit.py (itself isomorphic to
// diffusers CosmosTransformer3DModel). Anima config: extra_pos_embed_type=None (RoPE only), no
// img_context. Self-attn(attn1) gets the 3D RoPE; cross-attn(attn2) does NOT. AdaLN-LoRA(256).
import Foundation
import MLX
import MLXNN
import MLXFast

public struct CosmosDiTConfig {
    public var inChannels = 16
    public var outChannels = 16
    public var numAttentionHeads = 16
    public var attentionHeadDim = 128
    public var numLayers = 28
    public var mlpRatio: Float = 4.0
    public var textEmbedDim = 1024
    public var adalnLoraDim = 256
    public var maxSize = (128, 240, 240)
    public var patchSize = (1, 2, 2)
    public var ropeScale = (2.0, 1.0, 1.0)
    public var baseFps = 24
    public var concatPaddingMask = true
    public var hiddenSize: Int { numAttentionHeads * attentionHeadDim }
    public init() {}
}

// ---------------------------------------------------------------- primitives
func layerNormNoAffine(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let mu = mean(x, axis: -1, keepDims: true)
    let v = mean(square(x - mu), axis: -1, keepDims: true)
    return (x - mu) * rsqrt(v + eps)
}

func getTimestepEmbedding(_ timesteps: MLXArray, dim: Int, maxPeriod: Float = 10000) -> MLXArray {
    let half = dim / 2
    let exponent = (-log(maxPeriod) * MLXArray(0 ..< half).asType(.float32)) / Float(half)
    let emb0 = exp(exponent)
    let emb = timesteps.asType(.float32).reshaped(timesteps.dim(0), 1) * emb0.reshaped(1, half)
    return concatenated([cos(emb), sin(emb)], axis: -1)  // flip_sin_to_cos=True
}

/// diffusers apply_rotary_emb(use_real=True, use_real_unbind_dim=-2): x*cos + rotate_half(x)*sin,
/// cos/sin [S,D] -> broadcast over B,H. Computed in fp32 (matches the Python port).
func applyRopeDiT(_ x: MLXArray, _ cosT: MLXArray, _ sinT: MLXArray) -> MLXArray {
    let c = cosT.reshaped(1, 1, cosT.dim(0), cosT.dim(1))
    let s = sinT.reshaped(1, 1, sinT.dim(0), sinT.dim(1))
    let d = x.dim(-1) / 2
    let xr = x[.ellipsis, 0 ..< d]
    let xi = x[.ellipsis, d ..< (2 * d)]
    let xrot = concatenated([-xi, xr], axis: -1)
    return x.asType(.float32) * c + xrot.asType(.float32) * s
}

/// 3D RoPE (per-axis t/h/w split, concat-tiled). Returns (cos,sin) [THW, headDim].
final class CosmosRotaryPosEmbed {
    let patchSize: (Int, Int, Int)
    let maxSizePatched: (Int, Int, Int)
    let dimH: Int, dimW: Int, dimT: Int
    let hTheta: Float, wTheta: Float, tTheta: Float

    init(_ cfg: CosmosDiTConfig) {
        let hd = cfg.attentionHeadDim
        patchSize = cfg.patchSize
        maxSizePatched = (cfg.maxSize.0 / cfg.patchSize.0, cfg.maxSize.1 / cfg.patchSize.1, cfg.maxSize.2 / cfg.patchSize.2)
        dimH = hd / 6 * 2
        dimW = hd / 6 * 2
        dimT = hd - dimH - dimW
        let hNtk = pow(Float(cfg.ropeScale.1), Float(dimH) / Float(dimH - 2))
        let wNtk = pow(Float(cfg.ropeScale.2), Float(dimW) / Float(dimW - 2))
        let tNtk = pow(Float(cfg.ropeScale.0), Float(dimT) / Float(dimT - 2))
        hTheta = 10000.0 * hNtk
        wTheta = 10000.0 * wNtk
        tTheta = 10000.0 * tNtk
    }

    private func axisFreqs(_ dim: Int, _ theta: Float) -> MLXArray {
        // arange(0,dim,2)[:dim/2]/dim ; 1/(theta**range)
        let rng = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })[0 ..< (dim / 2)] / Float(dim)
        return 1.0 / pow(MLXArray(theta), rng)
    }

    func callAsFunction(_ T: Int, _ H: Int, _ W: Int) -> (MLXArray, MLXArray) {
        let pe = (T / patchSize.0, H / patchSize.1, W / patchSize.2)
        let maxS = max(maxSizePatched.0, max(maxSizePatched.1, maxSizePatched.2))
        let seq = MLXArray(0 ..< maxS).asType(.float32)
        let hFreqs = axisFreqs(dimH, hTheta)
        let wFreqs = axisFreqs(dimW, wTheta)
        let tFreqs = axisFreqs(dimT, tTheta)

        var embH = outer(seq[0 ..< pe.1], hFreqs).reshaped(1, pe.1, 1, dimH / 2)
        embH = broadcast(embH, to: [pe.0, pe.1, pe.2, dimH / 2])
        var embW = outer(seq[0 ..< pe.2], wFreqs).reshaped(1, 1, pe.2, dimW / 2)
        embW = broadcast(embW, to: [pe.0, pe.1, pe.2, dimW / 2])
        var embT = outer(seq[0 ..< pe.0], tFreqs).reshaped(pe.0, 1, 1, dimT / 2)
        embT = broadcast(embT, to: [pe.0, pe.1, pe.2, dimT / 2])

        var freqs = concatenated([embT, embH, embW, embT, embH, embW], axis: -1)
        freqs = freqs.reshaped(-1, freqs.dim(-1))
        return (cos(freqs), sin(freqs))
    }
}

// ---------------------------------------------------------------- modules
final class CosmosPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let patchSize: (Int, Int, Int)
    init(inChannels: Int, outChannels: Int, patchSize: (Int, Int, Int)) {
        self.patchSize = patchSize
        let (pt, ph, pw) = patchSize
        _proj.wrappedValue = Linear(inChannels * pt * ph * pw, outChannels, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let (pt, ph, pw) = patchSize
        var y = x.reshaped(b, c, t / pt, pt, h / ph, ph, w / pw, pw)
        y = y.transposed(0, 2, 4, 6, 1, 3, 5, 7)  // B,T',H',W',C,pt,ph,pw
        y = y.reshaped(b, t / pt, h / ph, w / pw, c * pt * ph * pw)
        return proj(y)
    }
}

final class CosmosTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ inF: Int, _ outF: Int) {
        _linear1.wrappedValue = Linear(inF, outF, bias: false)
        _linear2.wrappedValue = Linear(outF, 3 * outF, bias: false)
        super.init()
    }
    func callAsFunction(_ t: MLXArray) -> MLXArray { linear2(silu(linear1(t))) }
}

final class CosmosEmbedding: Module {
    let embeddingDim: Int
    @ModuleInfo(key: "t_embedder") var tEmbedder: CosmosTimestepEmbedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    init(embeddingDim: Int, conditionDim: Int) {
        self.embeddingDim = embeddingDim
        _tEmbedder.wrappedValue = CosmosTimestepEmbedding(embeddingDim, conditionDim)
        _norm.wrappedValue = RMSNorm(dimensions: embeddingDim, eps: 1e-6)
        super.init()
    }
    func callAsFunction(_ timestep: MLXArray) -> (MLXArray, MLXArray) {
        let proj = getTimestepEmbedding(timestep, dim: embeddingDim)
        return (tEmbedder(proj), norm(proj))
    }
}

/// norm1/2/3: LayerNorm(no affine) then shift/scale/gate from AdaLN-LoRA + temb.
final class AdaLNZero: Module {
    let inFeatures: Int
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ inF: Int, _ hiddenF: Int) {
        inFeatures = inF
        _linear1.wrappedValue = Linear(inF, hiddenF, bias: false)
        _linear2.wrappedValue = Linear(hiddenF, 3 * inF, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, _ embeddedTimestep: MLXArray, _ temb: MLXArray) -> (MLXArray, MLXArray) {
        let e = linear2(linear1(silu(embeddedTimestep))) + temb
        let parts = split(e, parts: 3, axis: -1)
        var shift = parts[0], scale = parts[1], gate = parts[2]
        let xn = layerNormNoAffine(x)
        if e.ndim == 2 { shift = shift.expandedDimensions(axis: 1); scale = scale.expandedDimensions(axis: 1); gate = gate.expandedDimensions(axis: 1) }
        return (xn * (1 + scale) + shift, gate)
    }
}

/// norm_out: shift/scale only; adds temb[..., :2*dim].
final class AdaLN: Module {
    let inFeatures: Int
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ inF: Int, _ hiddenF: Int) {
        inFeatures = inF
        _linear1.wrappedValue = Linear(inF, hiddenF, bias: false)
        _linear2.wrappedValue = Linear(hiddenF, 2 * inF, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, _ embeddedTimestep: MLXArray, _ temb: MLXArray) -> MLXArray {
        let e = linear2(linear1(silu(embeddedTimestep))) + temb[.ellipsis, 0 ..< (2 * inFeatures)]
        let parts = split(e, parts: 2, axis: -1)
        var shift = parts[0], scale = parts[1]
        let xn = layerNormNoAffine(x)
        if e.ndim == 2 { shift = shift.expandedDimensions(axis: 1); scale = scale.expandedDimensions(axis: 1) }
        return xn * (1 + scale) + shift
    }
}

final class CosmosAttention: Module {
    let heads: Int, dimHead: Int
    let scale: Float
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(_ cfg: CosmosDiTConfig, crossDim: Int?) {
        heads = cfg.numAttentionHeads; dimHead = cfg.attentionHeadDim
        let dim = cfg.hiddenSize
        let kvIn = crossDim ?? dim
        scale = pow(Float(dimHead), -0.5)
        _toQ.wrappedValue = Linear(dim, dim, bias: false)
        _toK.wrappedValue = Linear(kvIn, dim, bias: false)
        _toV.wrappedValue = Linear(kvIn, dim, bias: false)
        _toOut.wrappedValue = [Linear(dim, dim, bias: false)]
        _normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-6)
        _normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray?, rope: (MLXArray, MLXArray)?) -> MLXArray {
        let ctx = context ?? x
        let (b, s, sk) = (x.dim(0), x.dim(1), ctx.dim(1))
        var q = toQ(x).reshaped(b, s, heads, dimHead).transposed(0, 2, 1, 3)
        var k = toK(ctx).reshaped(b, sk, heads, dimHead).transposed(0, 2, 1, 3)
        let v = toV(ctx).reshaped(b, sk, heads, dimHead).transposed(0, 2, 1, 3)
        q = normQ(q)
        k = normK(k)
        if let rope {
            q = applyRopeDiT(q, rope.0, rope.1).asType(v.dtype)
            k = applyRopeDiT(k, rope.0, rope.1).asType(v.dtype)
        }
        var o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        o = o.transposed(0, 2, 1, 3).reshaped(b, s, heads * dimHead)
        return toOut[0](o)
    }
}

final class FFProj: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ dim: Int, _ inner: Int) { _proj.wrappedValue = Linear(dim, inner, bias: false); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// net = [FFProj(.0), GELU, Linear(.2)] -> keys ff.net.0.proj / ff.net.2 remapped to fc0/fc2 at load.
final class CosmosFFNet: Module {
    @ModuleInfo(key: "fc0") var fc0: FFProj
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(_ dim: Int, _ inner: Int) {
        _fc0.wrappedValue = FFProj(dim, inner)
        _fc2.wrappedValue = Linear(inner, dim, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc0(x))) }
}

final class CosmosFeedForward: Module {
    @ModuleInfo(key: "net") var net: CosmosFFNet
    init(_ dim: Int, _ mult: Float) { _net.wrappedValue = CosmosFFNet(dim, Int(Float(dim) * mult)); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { net(x) }
}

final class CosmosTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: AdaLNZero
    @ModuleInfo(key: "attn1") var attn1: CosmosAttention
    @ModuleInfo(key: "norm2") var norm2: AdaLNZero
    @ModuleInfo(key: "attn2") var attn2: CosmosAttention
    @ModuleInfo(key: "norm3") var norm3: AdaLNZero
    @ModuleInfo(key: "ff") var ff: CosmosFeedForward

    init(_ cfg: CosmosDiTConfig) {
        let h = cfg.hiddenSize
        _norm1.wrappedValue = AdaLNZero(h, cfg.adalnLoraDim)
        _attn1.wrappedValue = CosmosAttention(cfg, crossDim: nil)
        _norm2.wrappedValue = AdaLNZero(h, cfg.adalnLoraDim)
        _attn2.wrappedValue = CosmosAttention(cfg, crossDim: cfg.textEmbedDim)
        _norm3.wrappedValue = AdaLNZero(h, cfg.adalnLoraDim)
        _ff.wrappedValue = CosmosFeedForward(h, cfg.mlpRatio)
        super.init()
    }

    func callAsFunction(_ x0: MLXArray, _ context: MLXArray, _ embeddedTimestep: MLXArray,
                        _ temb: MLXArray, _ rope: (MLXArray, MLXArray)) -> MLXArray {
        var (n, g) = norm1(x0, embeddedTimestep, temb)
        var x = x0 + g * attn1(n, context: nil, rope: rope)
        (n, g) = norm2(x, embeddedTimestep, temb)
        x = x + g * attn2(n, context: context, rope: nil)
        (n, g) = norm3(x, embeddedTimestep, temb)
        x = x + g * ff(n)
        return x
    }
}

public final class CosmosTransformer3DModel: Module {
    let cfg: CosmosDiTConfig
    let rope: CosmosRotaryPosEmbed
    @ModuleInfo(key: "patch_embed") var patchEmbed: CosmosPatchEmbed
    @ModuleInfo(key: "time_embed") var timeEmbed: CosmosEmbedding
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [CosmosTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLN
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(_ cfg: CosmosDiTConfig = CosmosDiTConfig()) {
        self.cfg = cfg
        rope = CosmosRotaryPosEmbed(cfg)
        let inCh = cfg.inChannels + (cfg.concatPaddingMask ? 1 : 0)
        _patchEmbed.wrappedValue = CosmosPatchEmbed(inChannels: inCh, outChannels: cfg.hiddenSize, patchSize: cfg.patchSize)
        _timeEmbed.wrappedValue = CosmosEmbedding(embeddingDim: cfg.hiddenSize, conditionDim: cfg.hiddenSize)
        _transformerBlocks.wrappedValue = (0 ..< cfg.numLayers).map { _ in CosmosTransformerBlock(cfg) }
        _normOut.wrappedValue = AdaLN(cfg.hiddenSize, cfg.adalnLoraDim)
        let p = cfg.patchSize
        _projOut.wrappedValue = Linear(cfg.hiddenSize, p.0 * p.1 * p.2 * cfg.outChannels, bias: false)
        super.init()
    }

    /// hiddenStates [B,16,T,H,W]; timestep [B]; encoderHiddenStates [B,Lctx,1024]; paddingMask [B,1,H,W] (or nil).
    public func callAsFunction(_ hiddenStates: MLXArray, _ timestep: MLXArray, _ encoderHiddenStates: MLXArray,
                               paddingMask: MLXArray? = nil) -> MLXArray {
        var hs = hiddenStates
        let (b, _, t, h, w) = (hs.dim(0), hs.dim(1), hs.dim(2), hs.dim(3), hs.dim(4))
        if cfg.concatPaddingMask {
            let pm0 = paddingMask ?? MLXArray.zeros([b, 1, h, w], dtype: hs.dtype)
            var pm = pm0.reshaped(b, 1, 1, h, w)
            pm = broadcast(pm, to: [b, 1, t, h, w])
            hs = concatenated([hs, pm], axis: 1)
        }
        let (cosT, sinT) = rope(t, h, w)
        var x = patchEmbed(hs)  // [B,Tp,Hp,Wp,hidden]
        let (tp, hp, wp) = (x.dim(1), x.dim(2), x.dim(3))
        x = x.reshaped(b, tp * hp * wp, x.dim(-1))
        let (temb, embeddedTimestep) = timeEmbed(timestep)
        for blk in transformerBlocks { x = blk(x, encoderHiddenStates, embeddedTimestep, temb, (cosT, sinT)) }
        x = normOut(x, embeddedTimestep, temb)
        x = projOut(x)
        return unpatchify(x, tp, hp, wp)
    }

    private func unpatchify(_ x0: MLXArray, _ tp: Int, _ hp: Int, _ wp: Int) -> MLXArray {
        let (pt, ph, pw) = cfg.patchSize
        let oc = cfg.outChannels
        let b = x0.dim(0)
        var x = x0.reshaped(b, x0.dim(1), ph, pw, pt, oc)
        x = x.reshaped(b, tp, hp, wp, ph, pw, pt, oc)
        x = x.transposed(0, 7, 1, 6, 2, 4, 3, 5)  // B,oc,Tp,pt,Hp,ph,Wp,pw
        return x.reshaped(b, oc, tp * pt, hp * ph, wp * pw)
    }
}
