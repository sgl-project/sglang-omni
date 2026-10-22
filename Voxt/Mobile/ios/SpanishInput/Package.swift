// swift-tools-version: 6.0
// Spanish QWERTY: a direct layout (keys insert what they show), no engine code.
import PackageDescription

let package = Package(
    name: "SpanishInput",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "SpanishInput", targets: ["SpanishInput"])],
    dependencies: [.package(path: "../TypingInput")],
    targets: [
        .target(name: "SpanishInput", dependencies: ["TypingInput"]),
        .testTarget(name: "SpanishInputTests", dependencies: ["SpanishInput"])
    ]
)
