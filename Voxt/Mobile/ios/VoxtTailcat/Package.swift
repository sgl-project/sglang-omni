// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "VoxtTailcat",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "VoxtTailcat", targets: ["VoxtTailcat"])],
    targets: [
        .binaryTarget(name: "CVoxtTailcat", path: "Artifacts/CVoxtTailcat.xcframework"),
        .target(name: "VoxtTailcat", dependencies: ["CVoxtTailcat"], linkerSettings: [
            .linkedFramework("Security"), .linkedFramework("CoreFoundation"), .linkedLibrary("resolv")
        ]),
        .testTarget(name: "VoxtTailcatTests", dependencies: ["VoxtTailcat"])
    ]
)
