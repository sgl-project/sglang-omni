// swift-tools-version: 6.0
// The contract between the Voxt keyboard and its typing layouts. No other
// dependencies: the voice path and the layout packages never see each other.
import PackageDescription

let package = Package(
    name: "TypingInput",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [
        .library(name: "TypingInput", targets: ["TypingInput"]),
        // Keyboard side: writes marked text into the host field and checks each write landed.
        .library(name: "TypingHost", targets: ["TypingHost"]),
    ],
    targets: [
        .target(name: "TypingInput"),
        .target(name: "TypingHost"),
        .testTarget(name: "TypingInputTests", dependencies: ["TypingInput"]),
        .testTarget(name: "TypingHostTests", dependencies: ["TypingHost"]),
    ]
)
