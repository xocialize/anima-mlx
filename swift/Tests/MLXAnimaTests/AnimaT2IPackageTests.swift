import Foundation
import MLXToolKit
import XCTest
@testable import MLXAnima

final class AnimaT2IPackageTests: XCTestCase {
    func testManifest() {
        let m = AnimaT2IPackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .textToImage)
        XCTAssertEqual(m.surfaces[0].name, "anima-t2i")
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.requirements.requiredBackends, [.metalGPU])
        XCTAssertTrue(m.requirements.footprints.contains { $0.quant == .bf16 })
        XCTAssertTrue(m.requirements.footprints.contains { $0.quant == .int4 })
    }

    /// C7: Non-Commercial weights MUST be denied under the default `.permissiveOnly` policy.
    func testNonCommercialWeightGated() {
        let result = LicensePolicy.permissiveOnly.evaluate(AnimaT2IPackage.manifest.license)
        XCTAssertFalse(result.isAdmitted, "NC weights must not be admitted under permissiveOnly")
        if case .rejectedWeight = result {} else { XCTFail("expected rejectedWeight, got \(result)") }
    }

    func testConfigurationDefaults() {
        let c = AnimaConfiguration()
        XCTAssertEqual(c.defaultSteps, 30)
        XCTAssertEqual(c.defaultGuidanceScale, 5.0)
        XCTAssertEqual(c.quant, .bf16)
    }

    /// Full load→run→unload — env-gated (needs the published snapshot on disk + GPU).
    func testPackageRun() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["ANIMA_PKG"] == "1", "set ANIMA_PKG=1")
        let snap = ProcessInfo.processInfo.environment["ANIMA_SNAPSHOT"] ?? ""
        let pkg = AnimaT2IPackage(configuration: AnimaConfiguration(snapshotPath: snap))
        try await pkg.load()
        let resp = try await pkg.run(T2IRequest(prompt: "1girl, anime, masterpiece", width: 512, height: 512, steps: 12, seed: 7))
        guard let t2i = resp as? T2IResponse else { return XCTFail("wrong response type") }
        XCTAssertEqual(t2i.image.format, .png)
        XCTAssertGreaterThan(t2i.image.data.count, 10_000)
        try t2i.image.data.write(to: URL(fileURLWithPath: NSHomeDirectory() + "/Desktop/anima-package.png"))
        await pkg.unload()
    }
}
