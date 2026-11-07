// swift-tools-version:6.2
import Foundation
import PackageDescription

// voxtnew-llm: the Swift process that runs VoxtNew's language models, apart
// from voxtnew-helper so that a model's memory and a crash stay out of the
// helper. One model at a time: MLX (mlx-swift-lm) or GGUF (llama.cpp,
// through llama.swift). JSON lines over stdin/stdout: llm protocol 1
// (docs/llm-protocol.md); Go (internal/llmrt) supervises it and chooses
// every parameter.
//
// VoxtNewLLMKit holds what needs no model: the wire types, the server
// (routing, jobs, cancellation) over an Engine protocol, and the text
// helpers. `swift test` covers it against llm/Fixtures with a fake engine.
// The engines and main.swift are the executable's.
//
// The pins are Voxt's (Voxt.xcodeproj Package.resolved, 2026-09): the same
// mlx-swift-lm revision its loaders were written for. SwiftPM on the
// command line cannot compile MLX's Metal shaders: scripts/build-llm.sh
// builds with xcodebuild and ships mlx.metallib beside the binary.

/// The process's own Info.plist, linked into the binary (its name and
/// version when it runs outside VoxtNew.app).
let infoPlist = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Support/Info.plist").path

let package = Package(
    name: "VoxtNewLLM",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "voxtnew-llm", targets: ["VoxtNewLLM"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        // FoundationModelsIntegration (on by default) needs macOS 27.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "c6446cf7bfb7cea76408013b614d4b2c530eaa03", traits: []),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.3.4"),
        .package(url: "https://github.com/mattt/llama.swift.git", exact: "2.10549.0"),
    ],
    targets: [
        .target(
            name: "VoxtNewLLMKit",
            path: "Sources/VoxtNewLLMKit"
        ),
        .executableTarget(
            name: "VoxtNewLLM",
            dependencies: [
                "VoxtNewLLMKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "LlamaSwift", package: "llama.swift"),
            ],
            path: "Sources/VoxtNewLLM",
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker", infoPlist]),
                // llama.framework sits beside the binary (scripts/build-llm.sh).
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"]),
            ]
        ),
        .testTarget(
            name: "VoxtNewLLMKitTests",
            dependencies: ["VoxtNewLLMKit"],
            path: "Tests/VoxtNewLLMKitTests"
        ),
    ]
)
