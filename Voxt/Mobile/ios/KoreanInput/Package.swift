// swift-tools-version: 6.0
// Hangul 2-beolsik (두벌식). Syllables are composed with Unicode arithmetic
// written here; no dictionary and no third-party code.
import PackageDescription

let package = Package(
    name: "KoreanInput",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "KoreanInput", targets: ["KoreanInput"])],
    dependencies: [.package(path: "../TypingInput")],
    targets: [
        .target(name: "KoreanInput", dependencies: ["TypingInput"]),
        .testTarget(name: "KoreanInputTests", dependencies: ["KoreanInput"])
    ]
)
