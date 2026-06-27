// swift-tools-version: 6.2
// anima-swift — Swift/MLX port of Anima (Cosmos-Predict2-2B anime T2I, Non-Commercial).
// Mirrors lens-mlx-swift / qwen-image-edit-swift. Core (`Anima`) = ported model graph (no
// MLXToolKit); the engine-conformant ModelPackage wrapper (`MLXAnima`) is added once the four
// components are parity-gated. Consumes the published canonical-MLX weights (xocialize/anima-mlx).

import PackageDescription

let package = Package(
    name: "Anima",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "Anima", targets: ["Anima"]),
        .library(name: "MLXAnima", targets: ["MLXAnima"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.3.0"),
    ],
    targets: [
        .target(
            name: "Anima",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/Anima"
        ),
        .target(
            name: "MLXAnima",
            dependencies: [
                "Anima",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/MLXAnima"
        ),
        .executableTarget(
            name: "anima-cli",
            dependencies: [
                "Anima",
                "MLXAnima",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/AnimaCLI"
        ),
        .testTarget(
            name: "MLXAnimaTests",
            dependencies: ["MLXAnima", .product(name: "MLXToolKit", package: "mlx-engine-swift")],
            path: "Tests/MLXAnimaTests"
        ),
    ]
)
