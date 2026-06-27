// anima-cli — Stage-1 CLI gates (the metallib resolves for `swift run`; SPM test product is fragile).
//   swift run anima-cli --vae-gate <vae-bf16.safetensors> <vae_golden.safetensors>
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXRandom
import Anima
import MLXAnima
import MLXToolKit

func savePNG(_ img: MLXArray, to path: String) throws {
    // img [1,H,W,3] in [0,1]
    let h = img.dim(1), w = img.dim(2)
    let px = (img[0] * 255).asType(.uint8).asArray(UInt8.self)  // row-major H*W*3
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: w * h * 4)
    for i in 0 ..< (w * h) {
        buf[i * 4] = px[i * 3]; buf[i * 4 + 1] = px[i * 3 + 1]; buf[i * 4 + 2] = px[i * 3 + 2]; buf[i * 4 + 3] = 255
    }
    let image = ctx.makeImage()!
    let outData = NSMutableData()
    let dest = CGImageDestinationCreateWithData(outData, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    _ = CGImageDestinationFinalize(dest)
    try (outData as Data).write(to: URL(fileURLWithPath: path))
}

func cosMaxabs(_ a: MLXArray, _ b: MLXArray) -> (cos: Float, maxabs: Float) {
    let af = a.asType(.float32).flattened()
    let bf = b.asType(.float32).flattened()
    let dot = sum(af * bf).item(Float.self)
    let na = sqrt(sum(af * af)).item(Float.self)
    let nb = sqrt(sum(bf * bf)).item(Float.self)
    let maxabs = MLX.max(MLX.abs(af - bf)).item(Float.self)
    return (dot / (na * nb + 1e-30), maxabs)
}

let args = CommandLine.arguments
do {
    guard args.count >= 2 else { print("usage: anima-cli --vae-gate <vae> <golden>"); exit(2) }
    switch args[1] {
    case "--vae-gate":
        guard args.count >= 4 else { print("usage: --vae-gate <vae> <golden>"); exit(2) }
        Device.setDefault(device: .cpu)  // CPU stream for tight parity
        let vae = try AnimaWeights.loadVAE(file: URL(fileURLWithPath: args[2]), dtype: .float32)
        FileHandle.standardError.write("[vae] loaded (strict key match)\n".data(using: .utf8)!)
        let golden = try MLX.loadArrays(url: URL(fileURLWithPath: args[3]))
        guard let z = golden["in_latent"], let want = golden["out_image"] else {
            print("golden missing in_latent/out_image"); exit(2)
        }
        let img = vae.decode(z)
        eval(img)
        let (c, m) = cosMaxabs(img, want)
        let nan = MLX.any(img .!= img).item(Bool.self)  // NaN != NaN
        let ok = c > 0.999 && !nan
        print(String(format: "[vae-gate] shape %@ cos=%.6f maxabs=%.2e nan=%@ -> %@",
                     "\(img.shape)", c, m, nan ? "true" : "false", ok ? "PASS" : "FAIL"))
        exit(ok ? 0 : 1)
    case "--te-gate":
        guard args.count >= 4 else { print("usage: --te-gate <text_encoder> <golden>"); exit(2) }
        Device.setDefault(device: .cpu)
        let te = try AnimaWeights.loadTextEncoder(file: URL(fileURLWithPath: args[2]), dtype: .float32)
        FileHandle.standardError.write("[te] loaded (strict key match)\n".data(using: .utf8)!)
        let golden = try MLX.loadArrays(url: URL(fileURLWithPath: args[3]))
        let ids = golden["in_ids"]!.asType(.int32)
        let out = te(ids)
        eval(out)
        let (c, m) = cosMaxabs(out, golden["hidden_prenorm"]!)
        let ok = c > 0.999
        print(String(format: "[te-gate] shape %@ cos=%.6f maxabs=%.2e -> %@",
                     "\(out.shape)", c, m, ok ? "PASS" : "FAIL"))
        exit(ok ? 0 : 1)
    case "--adapter-gate":
        guard args.count >= 4 else { print("usage: --adapter-gate <adapter> <golden>"); exit(2) }
        Device.setDefault(device: .cpu)
        let ad = try AnimaWeights.loadAdapter(file: URL(fileURLWithPath: args[2]), dtype: .float32)
        FileHandle.standardError.write("[adapter] loaded (strict key match)\n".data(using: .utf8)!)
        let golden = try MLX.loadArrays(url: URL(fileURLWithPath: args[3]))
        let out = ad(golden["in_source"]!, golden["in_ids"]!.asType(.int32), padTo: nil)
        eval(out)
        let (c, m) = cosMaxabs(out, golden["out_final"]!)
        let ok = c > 0.999
        print(String(format: "[adapter-gate] shape %@ cos=%.6f maxabs=%.2e -> %@",
                     "\(out.shape)", c, m, ok ? "PASS" : "FAIL"))
        exit(ok ? 0 : 1)
    case "--dit-gate":
        guard args.count >= 4 else { print("usage: --dit-gate <transformer> <golden>"); exit(2) }
        Device.setDefault(device: .cpu)
        let dit = try AnimaWeights.loadDiT(file: URL(fileURLWithPath: args[2]), dtype: .float32)
        FileHandle.standardError.write("[dit] loaded (strict key match)\n".data(using: .utf8)!)
        let golden = try MLX.loadArrays(url: URL(fileURLWithPath: args[3]))
        let out = dit(golden["in_hidden"]!, golden["in_timestep"]!, golden["in_encoder"]!,
                      paddingMask: golden["in_padding"]!)
        eval(out)
        let (c, m) = cosMaxabs(out, golden["out_final"]!)
        let ok = c > 0.999
        print(String(format: "[dit-gate] shape %@ cos=%.6f maxabs=%.2e -> %@",
                     "\(out.shape)", c, m, ok ? "PASS" : "FAIL"))
        exit(ok ? 0 : 1)
    case "--dit-int4-gate":
        guard args.count >= 4 else { print("usage: --dit-int4-gate <transformer-int4> <golden>"); exit(2) }
        let dit = try AnimaWeights.loadDiTInt4(file: URL(fileURLWithPath: args[2]), dtype: .bfloat16)
        FileHandle.standardError.write("[dit-int4] loaded (strict key match)\n".data(using: .utf8)!)
        let golden = try MLX.loadArrays(url: URL(fileURLWithPath: args[3]))
        let out = dit(golden["in_hidden"]!.asType(.bfloat16), golden["in_timestep"]!.asType(.bfloat16),
                      golden["in_encoder"]!.asType(.bfloat16), paddingMask: golden["in_padding"]!.asType(.bfloat16))
        eval(out)
        let (c, m) = cosMaxabs(out, golden["out_final"]!)
        let ok = c > 0.99
        print(String(format: "[dit-int4-gate] per-pass cos=%.5f maxabs=%.2e -> %@", c, m, ok ? "PASS" : "FAIL"))
        exit(ok ? 0 : 1)

    case "--e2e-gate":
        guard args.count >= 7 else { print("usage: --e2e-gate <transformer> <text_encoder> <adapter> <vae> <golden>"); exit(2) }
        Device.setDefault(device: .cpu)
        let pipe = try AnimaPipeline.load(
            transformer: URL(fileURLWithPath: args[2]), textEncoder: URL(fileURLWithPath: args[3]),
            adapter: URL(fileURLWithPath: args[4]), vae: URL(fileURLWithPath: args[5]), dtype: .float32)
        FileHandle.standardError.write("[e2e] pipeline loaded\n".data(using: .utf8)!)
        let g = try MLX.loadArrays(url: URL(fileURLWithPath: args[6]))
        func ids(_ n: String) -> MLXArray { let a = g[n]!.asType(.int32); return a.reshaped(1, a.dim(0)) }
        let sigmas = g["sigmas"]!.asArray(Float.self).map(Double.init)
        var pass = true

        // (1) text path
        let condCtx = pipe.encodeContext(ids("cond_qwen_ids"), ids("cond_t5_ids"))
        let uncCtx = pipe.encodeContext(ids("uncond_qwen_ids"), ids("uncond_t5_ids"))
        eval(condCtx, uncCtx)
        for (name, got, want) in [("cond_context", condCtx, g["cond_context"]!),
                                  ("uncond_context", uncCtx, g["uncond_context"]!)] {
            let (c, m) = cosMaxabs(got, want)
            let ok = m < 5e-3; pass = pass && ok
            print(String(format: "  [%@] %@ cos=%.6f maxabs=%.2e", ok ? "ok " : "FAIL", name, c, m))
        }

        // (2) step-0 DiT-in-loop + CFG (inject golden contexts), BEFORE chaotic accumulation
        let gcond = g["cond_context"]!, gunc = g["uncond_context"]!
        let z0 = Float(sigmas[0]) * g["noise"]!
        let ctx0 = concatenated([gcond, gunc], axis: 0)
        let v = pipe.dit(concatenated([z0, z0], axis: 0), MLXArray([Float(sigmas[0]), Float(sigmas[0])]), ctx0)
        let v0 = v[1 ..< 2] + 5.0 * (v[0 ..< 1] - v[1 ..< 2]); eval(v0)
        let (vc, vm) = cosMaxabs(v0, g["v0_cfg"]!)
        let vok = vc > 0.999999 && vm < 5e-3; pass = pass && vok
        print(String(format: "  [%@] v0_cfg(step-0) cos=%.7f maxabs=%.2e", vok ? "ok " : "FAIL", vc, vm))

        // (3) full denoise (computed contexts) -> final latent by COSINE
        let x0 = pipe.sample(noise: g["noise"]!, condCtx: condCtx, uncondCtx: uncCtx, sigmas: sigmas, cfg: 5.0)
        eval(x0)
        let (fc, _) = cosMaxabs(x0, g["final_latent"]!)
        let fok = fc > 0.999; pass = pass && fok
        print(String(format: "  [%@] final_latent cos=%.6f", fok ? "ok " : "FAIL", fc))

        print("[e2e-gate] -> \(pass ? "PASS" : "FAIL")")
        exit(pass ? 0 : 1)
    case "--tok-gate":
        guard args.count >= 3 else { print("usage: --tok-gate <pipeline_golden>"); exit(2) }
        let tok = try await AnimaTokenizer.load()
        let g = try MLX.loadArrays(url: URL(fileURLWithPath: args[2]))
        func gold(_ n: String) -> [Int] { g[n]!.asArray(Int32.self).map(Int.init) }
        let (cq, ct) = tok.encode("1girl, anime, masterpiece, detailed background, soft lighting")
        let (uq, ut) = tok.encode("")
        var ok = true
        for (name, got, want) in [("cond qwen", cq, gold("cond_qwen_ids")), ("cond t5", ct, gold("cond_t5_ids")),
                                  ("uncond qwen", uq, gold("uncond_qwen_ids")), ("uncond t5", ut, gold("uncond_t5_ids"))] {
            let match = got == want; ok = ok && match
            print("  [\(match ? "ok " : "FAIL")] \(name): \(got)\(match ? "" : "  want \(want)")")
        }
        print("[tok-gate] -> \(ok ? "PASS" : "FAIL")")
        exit(ok ? 0 : 1)

    case "--generate":
        // --generate <prompt> <distDir> <out.png> [steps] [cfg] [size] [seed]
        guard args.count >= 5 else { print("usage: --generate <prompt> <distDir> <out.png> [steps] [cfg] [size] [seed]"); exit(2) }
        let prompt = args[2], dir = args[3], outPath = args[4]
        let steps = args.count > 5 ? Int(args[5])! : 24
        let cfg = args.count > 6 ? Float(args[6])! : 5.0
        let size = args.count > 7 ? Int(args[7])! : 512
        let seed = args.count > 8 ? UInt64(args[8])! : 1234
        let d = URL(fileURLWithPath: dir)
        let pipe = try AnimaPipeline.load(
            transformer: d.appendingPathComponent("transformer-bf16.safetensors"),
            textEncoder: d.appendingPathComponent("text_encoder-bf16.safetensors"),
            adapter: d.appendingPathComponent("llm_adapter-bf16.safetensors"),
            vae: d.appendingPathComponent("vae-bf16.safetensors"), dtype: .bfloat16)
        let tok = try await AnimaTokenizer.load()
        func idArr(_ a: [Int]) -> MLXArray { MLXArray(a.map(Int32.init)).reshaped(1, a.count) }
        let (cq, ct) = tok.encode(prompt); let (uq, ut) = tok.encode("")
        let cond = pipe.encodeContext(idArr(cq), idArr(ct))
        let unc = pipe.encodeContext(idArr(uq), idArr(ut))
        MLXRandom.seed(seed)
        let lat = size / VAE_SPATIAL
        let noise = MLXRandom.normal([1, 16, 1, lat, lat]).asType(.bfloat16)
        let x0 = pipe.sample(noise: noise, condCtx: cond, uncondCtx: unc, sigmas: flowSigmas(steps), cfg: cfg)
        let img = pipe.decode(x0); eval(img)
        let nan = MLX.any(img .!= img).item(Bool.self)
        try savePNG(img.asType(.float32), to: outPath)
        print(String(format: "[generate] %dx%d steps=%d cfg=%.1f nan=%@ peak=%.2fGB -> %@",
                     size, size, steps, cfg, nan ? "true" : "false", Double(GPU.peakMemory) / 1e9, outPath))
        exit(0)

    case "--package-gate":
        // Drives the full ModelPackage contract (load→run→unload). --package-gate <snapshotDir> <out.png> [quant]
        guard args.count >= 4 else { print("usage: --package-gate <snapshotDir> <out.png> [bf16|int4]"); exit(2) }
        let quant: Quant = (args.count > 4 && args[4] == "int4") ? .int4 : .bf16
        let pkg = AnimaT2IPackage(configuration: AnimaConfiguration(snapshotPath: args[2], quant: quant, defaultSteps: 24))
        let t0 = Date()
        try await pkg.load()
        FileHandle.standardError.write("[pkg] loaded in \(String(format: "%.1f", -t0.timeIntervalSinceNow))s\n".data(using: .utf8)!)
        let t1 = Date()
        let resp = try await pkg.run(T2IRequest(prompt: "1girl, anime, masterpiece, detailed background, soft lighting",
                                                width: 512, height: 512, steps: 24, guidanceScale: 5.0, seed: 1234))
        guard let t2i = resp as? T2IResponse else { print("wrong response type"); exit(1) }
        try t2i.image.data.write(to: URL(fileURLWithPath: args[3]))
        await pkg.unload()
        let ok = t2i.image.format == .png && t2i.image.data.count > 10_000
        print(String(format: "[package-gate] quant=%@ run=%.1fs format=%@ bytes=%d %dx%d -> %@",
                     "\(quant)", -t1.timeIntervalSinceNow, "\(t2i.image.format)", t2i.image.data.count,
                     t2i.image.width ?? 0, t2i.image.height ?? 0, ok ? "PASS" : "FAIL"))
        exit(ok ? 0 : 1)

    default:
        print("unknown mode \(args[1])"); exit(2)
    }
} catch {
    print("ERROR: \(error)")
    exit(1)
}
