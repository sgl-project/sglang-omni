// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OmniDictation",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "DictationCore", targets: ["DictationCore"]),
        .executable(name: "OmniDictation", targets: ["OmniDictation"]),
    ],
    targets: [
        .target(name: "DictationCore"),
        .executableTarget(name: "OmniDictation", dependencies: ["DictationCore"], path: "Sources/DictationApp"),
        .testTarget(name: "DictationCoreTests", dependencies: ["DictationCore"]),
    ]
)
