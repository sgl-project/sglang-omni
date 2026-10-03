// swift-tools-version: 6.0
// SPDX-License-Identifier: Apache-2.0
import PackageDescription

let package = Package(
    name: "OmniTyper",
    defaultLocalization: "en",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "OmniTyper", targets: ["OmniTyper"])],
    targets: [
        .executableTarget(name: "OmniTyper", resources: [.process("Resources")]),
        .testTarget(name: "OmniTyperTests", dependencies: ["OmniTyper"])
    ],
    swiftLanguageModes: [.v5]
)
