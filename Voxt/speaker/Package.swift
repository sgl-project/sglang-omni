// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "VoxtSpeaker",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "e2ea498dafd6db9b3f18f0f644bc35312b3a7679",
            traits: ["Diarizer"]
        )
    ],
    targets: [
        .executableTarget(name: "voxtnew-speaker", dependencies: [
            .product(name: "FluidAudio", package: "FluidAudio")
        ])
    ]
)
