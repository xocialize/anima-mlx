// AnimaT2IPackage — engine-conformant ModelPackage wrapping the Anima Core (Cosmos-Predict2-2B
// anime T2I). Capability .textToImage. WEIGHTS are Non-Commercial (CircleStone Labs Anima, built on
// NVIDIA Cosmos) → C7 license gate denies admission under .permissiveOnly (gated-but-flagged, by design).
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Anima
import MLX
import MLXRandom
import MLXToolKit

/// Init-time configuration (C9): where the published components live + generation defaults.
public struct AnimaConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var snapshotPath: String   // dir holding {transformer-bf16|transformer-int4, text_encoder-bf16, llm_adapter-bf16, vae-bf16}.safetensors
    public var quant: Quant           // .bf16 (full) or .int4 (quantized transformer)
    public var defaultSteps: Int
    public var defaultGuidanceScale: Float
    public var modelsRootDirectory: URL?

    public init(snapshotPath: String = "", quant: Quant = .bf16,
                defaultSteps: Int = 30, defaultGuidanceScale: Float = 5.0,
                modelsRootDirectory: URL? = nil) {
        self.snapshotPath = snapshotPath
        self.quant = quant
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case snapshotPath, quant, defaultSteps, defaultGuidanceScale }
}

public enum AnimaPackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode
}

@InferenceActor
public final class AnimaT2IPackage: ModelPackage {
    public typealias Configuration = AnimaConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7 weight license: CircleStone Non-Commercial (built on NVIDIA Cosmos). C8 port code: MIT.
            license: LicenseDeclaration(weightLicense: "LicenseRef-CircleStone-NonCommercial", portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "circlestone-labs/Anima", revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // residentBytes = measured Swift peak unified memory @512² (see PORTING-SPEC).
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 8_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 6_500_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .pro),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "anima-t2i",
                    summary: "Anima anime/illustration text-to-image (Cosmos-Predict2-2B DiT + Qwen3-0.6B → "
                        + "llm_adapter conditioning + Wan 16-ch VAE): flow-matching, 512–1024px, 30-step, CFG 4–5. "
                        + "Non-Commercial weights.",
                    modes: [])
            ])
    }

    private let configuration: Configuration
    private var pipeline: AnimaPipeline?
    private var tokenizer: AnimaTokenizer?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    private func file(_ name: String) -> URL {
        let root = configuration.modelsRootDirectory.map { $0.appendingPathComponent(configuration.snapshotPath) }
            ?? URL(fileURLWithPath: configuration.snapshotPath)
        return root.appendingPathComponent(name)
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        let transformer = configuration.quant == .int4 ? "transformer-int4.safetensors" : "transformer-bf16.safetensors"
        guard FileManager.default.fileExists(atPath: file(transformer).path) else {
            throw AnimaPackageError.unreadableSnapshot(file(transformer).path)
        }
        let dit = configuration.quant == .int4
            ? try AnimaWeights.loadDiTInt4(file: file(transformer), dtype: .bfloat16)
            : try AnimaWeights.loadDiT(file: file(transformer), dtype: .bfloat16)
        let adapter = try AnimaWeights.loadAdapter(file: file("llm_adapter-bf16.safetensors"), dtype: .bfloat16)
        let qwen = try AnimaWeights.loadTextEncoder(file: file("text_encoder-bf16.safetensors"), dtype: .bfloat16)
        let vae = try AnimaWeights.loadVAE(file: file("vae-bf16.safetensors"), dtype: .bfloat16)
        pipeline = AnimaPipeline(dit: dit, adapter: adapter, qwen: qwen, vae: vae)
        tokenizer = try await AnimaTokenizer.load()
    }

    public func unload() async { pipeline = nil; tokenizer = nil }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline, let tokenizer else { throw PackageError.notLoaded }
        guard request.capability == .textToImage, let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()
        let (pixels, w, h) = try generate(
            pipeline: pipeline, tokenizer: tokenizer,
            prompt: t2i.prompt, negative: t2i.negativePrompt ?? "",
            width: t2i.width ?? 512, height: t2i.height ?? 512,
            steps: t2i.steps ?? configuration.defaultSteps,
            cfg: t2i.guidanceScale.map(Float.init) ?? configuration.defaultGuidanceScale,
            seed: t2i.seed ?? 0)
        try Task.checkCancellation()
        let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
        return T2IResponse(image: Image(format: .png, data: png, width: w, height: h))
    }

    private func generate(pipeline: AnimaPipeline, tokenizer: AnimaTokenizer, prompt: String, negative: String,
                          width: Int, height: Int, steps: Int, cfg: Float, seed: UInt64) throws -> ([UInt8], Int, Int) {
        func idArr(_ a: [Int]) -> MLXArray { MLXArray(a.map(Int32.init)).reshaped(1, a.count) }
        let (cq, ct) = tokenizer.encode(prompt)
        let (uq, ut) = tokenizer.encode(negative)
        let cond = pipeline.encodeContext(idArr(cq), idArr(ct))
        let unc = pipeline.encodeContext(idArr(uq), idArr(ut))
        MLXRandom.seed(seed)
        let lat = height / VAE_SPATIAL
        let latW = width / VAE_SPATIAL
        let noise = MLXRandom.normal([1, 16, 1, lat, latW]).asType(.bfloat16)
        let x0 = pipeline.sample(noise: noise, condCtx: cond, uncondCtx: unc, sigmas: flowSigmas(steps), cfg: cfg)
        let img = pipeline.decode(x0)                       // [1,H,W,3] in [0,1]
        eval(img)
        let pixels = (img[0] * 255).asType(.uint8).asArray(UInt8.self)
        return (pixels, width, height)
    }

    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw AnimaPackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0 ..< (width * height) {
            buf[i * 4] = pixels[i * 3]; buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]; buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw AnimaPackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else { throw AnimaPackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw AnimaPackageError.pngEncode }
        return out as Data
    }
}

extension AnimaT2IPackage {
    public nonisolated static var registration: PackageRegistration { .of(AnimaT2IPackage.self) }
}
