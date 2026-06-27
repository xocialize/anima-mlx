import Foundation
import MLX
import MLXNN

public enum AnimaError: Error, CustomStringConvertible {
    case keyMismatch(component: String, missing: [String], unused: [String])
    public var description: String {
        switch self {
        case let .keyMismatch(c, missing, unused):
            return "[\(c)] key mismatch — missing \(missing.prefix(6)) … unused \(unused.prefix(6))"
        }
    }
}

public enum AnimaWeights {
    /// S0 contract enforcement: published canonical-MLX-layout keys must equal the module's
    /// flattened parameter keys (0 missing / 0 unused) — a refuse-partial-load.
    static func loadStrict<M: Module>(_ model: M, file: URL, dtype: DType, component: String,
                                      remap: (String) -> String = { $0 }) throws {
        let raw = try MLX.loadArrays(url: file)
        let floats: Set<DType> = [.float32, .float16, .bfloat16]
        var weights: [String: MLXArray] = [:]
        // cast only floating tensors; leave uint32-packed quant weights / int scales untouched.
        for (k, v) in raw { weights[remap(k)] = floats.contains(v.dtype) ? v.asType(dtype) : v }
        let moduleKeys = Set(model.parameters().flattened().map { $0.0 })
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys)
        let unused = fileKeys.subtracting(moduleKeys)
        guard missing.isEmpty, unused.isEmpty else {
            throw AnimaError.keyMismatch(component: component, missing: missing.sorted(), unused: unused.sorted())
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }

    public static func loadVAE(file: URL, dtype: DType = .bfloat16) throws -> AnimaVAE {
        let model = AnimaVAE()
        model.weightDtype = dtype
        try loadStrict(model, file: file, dtype: dtype, component: "vae")
        return model
    }

    public static func loadTextEncoder(file: URL, dtype: DType = .bfloat16) throws -> Qwen3TextEncoder {
        let model = Qwen3TextEncoder()
        try loadStrict(model, file: file, dtype: dtype, component: "qwen3")
        return model
    }

    public static func loadAdapter(file: URL, dtype: DType = .bfloat16) throws -> LLMAdapter {
        let model = LLMAdapter()
        // numeric-leaf list `mlp.0`/`mlp.2` (GELU hole at 1) can't be a clean Swift Module → named slots.
        try loadStrict(model, file: file, dtype: dtype, component: "adapter") {
            $0.replacingOccurrences(of: ".mlp.0.", with: ".mlp.fc0.")
              .replacingOccurrences(of: ".mlp.2.", with: ".mlp.fc2.")
        }
        return model
    }

    public static func loadDiT(file: URL, dtype: DType = .bfloat16) throws -> CosmosTransformer3DModel {
        let model = CosmosTransformer3DModel()
        // ff.net.0.proj / ff.net.2 (GELU hole at 1) -> named slots fc0/fc2.
        try loadStrict(model, file: file, dtype: dtype, component: "dit", remap: ffRemap)
        return model
    }

    /// int4 variant: quantize the SAME scope as the Python export (transformer-block attn+ff Linears,
    /// g64) before loading the published int4 transformer.
    public static func loadDiTInt4(file: URL, dtype: DType = .bfloat16) throws -> CosmosTransformer3DModel {
        let model = CosmosTransformer3DModel()
        quantize(model: model, filter: { path, module in
            guard module is Linear, path.contains("transformer_blocks"),
                  path.contains(".attn") || path.contains(".ff.") else { return nil }
            return (groupSize: 64, bits: 4, mode: .affine)
        })
        try loadStrict(model, file: file, dtype: dtype, component: "dit-int4", remap: ffRemap)
        return model
    }

    static func ffRemap(_ k: String) -> String {
        // ff.net.0.proj / ff.net.2 (GELU hole at 1) -> named slots fc0/fc2.
        k.replacingOccurrences(of: ".ff.net.0.", with: ".ff.net.fc0.")
         .replacingOccurrences(of: ".ff.net.2.", with: ".ff.net.fc2.")
    }
}
