// Anima T2I pipeline — 1:1 transpose of anima_mlx/pipeline.py.
// Sampler = comfy ModelType.FLOW: CONST prediction + ModelSamplingDiscreteFlow(shift=3, mult=1)
// → sigma(t)=3t/(1+2t), DiT timestep == sigma ∈ [0,1], deterministic Euler. CFG: v_unc + cfg*(v_cond-v_unc).
import Foundation
import MLX
import MLXNN

public let PAD_TO = 512
public let VAE_SPATIAL = 8

public func timeSnrShift(_ a: Double, _ t: Double) -> Double { a * t / (1 + (a - 1) * t) }

/// comfy 'normal' scheduler for ModelSamplingDiscreteFlow (Double = numpy float64). sigma_max=1.0.
public func flowSigmas(_ steps: Int, shift: Double = 3.0, mult: Double = 1.0) -> [Double] {
    let sMax = timeSnrShift(shift, mult)
    let sMin = timeSnrShift(shift, (1.0 / 1000.0) * mult)
    let start = sMax * mult, end = sMin * mult
    var sigs = (0 ..< steps).map { i -> Double in
        let t = start + (end - start) * Double(i) / Double(steps - 1)
        return timeSnrShift(shift, t / mult)
    }
    sigs.append(0.0)
    return sigs
}

public final class AnimaPipeline {
    public let dit: CosmosTransformer3DModel
    public let adapter: LLMAdapter
    public let qwen: Qwen3TextEncoder
    public let vae: AnimaVAE

    public init(dit: CosmosTransformer3DModel, adapter: LLMAdapter, qwen: Qwen3TextEncoder, vae: AnimaVAE) {
        self.dit = dit; self.adapter = adapter; self.qwen = qwen; self.vae = vae
    }

    public static func load(transformer: URL, textEncoder: URL, adapter: URL, vae: URL,
                            dtype: DType = .bfloat16) throws -> AnimaPipeline {
        AnimaPipeline(
            dit: try AnimaWeights.loadDiT(file: transformer, dtype: dtype),
            adapter: try AnimaWeights.loadAdapter(file: adapter, dtype: dtype),
            qwen: try AnimaWeights.loadTextEncoder(file: textEncoder, dtype: dtype),
            vae: { let v = try! AnimaWeights.loadVAE(file: vae, dtype: dtype); v.weightDtype = dtype; return v }())
    }

    /// qwenIds [1,Lq], t5Ids [1,Lt5] -> DiT cross-attn context [1,512,1024].
    public func encodeContext(_ qwenIds: MLXArray, _ t5Ids: MLXArray) -> MLXArray {
        let src = qwen(qwenIds)
        return adapter(src, t5Ids, padTo: PAD_TO)
    }

    /// Deterministic Euler over the flow CONST schedule. noise/contexts on the model stream.
    public func sample(noise: MLXArray, condCtx: MLXArray, uncondCtx: MLXArray, sigmas: [Double],
                       cfg: Float) -> MLXArray {
        var x = Float(sigmas[0]) * noise
        let ctx = concatenated([condCtx, uncondCtx], axis: 0)
        for i in 0 ..< (sigmas.count - 1) {
            let s = Float(sigmas[i])
            let xb = concatenated([x, x], axis: 0)
            let t = MLXArray([s, s])
            let v = dit(xb, t, ctx)
            let vCond = v[0 ..< 1]
            let vUnc = v[1 ..< 2]
            let vCfg = vUnc + cfg * (vCond - vUnc)
            x = x + vCfg * (Float(sigmas[i + 1]) - s)
            eval(x)
        }
        return x
    }

    /// model-space x0 [B,16,1,Hl,Wl] -> image [B,H,W,3] in [0,1].
    public func decode(_ x0: MLXArray) -> MLXArray {
        let lat = AnimaVAE.deNormalize(x0)
        let img = vae.decode(lat)                       // [B,3,1,H,W] in [-1,1]
        let im = (img[0..., 0..., 0].transposed(0, 2, 3, 1) + 1.0) * 0.5
        return clip(im, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
    }
}
