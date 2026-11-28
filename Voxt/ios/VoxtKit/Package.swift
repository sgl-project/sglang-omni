// swift-tools-version: 5.10
// VoxtKit: the iOS app's logic (VoxtKit) and screens (VoxtUI), over the Go
// core shared with the macOS app (CVoxtCore: libvoxtcore.a, built by
// ../scripts/build-core.sh). `swift test` on a Mac links build/macos's
// library; the app's project links the iOS ones (../App/project.yml).
import Foundation
import PackageDescription

let macLibraries = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .appendingPathComponent("../build/macos").standardized.path

let package = Package(
    name: "VoxtKit",
    defaultLocalization: "en",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "VoxtKit", targets: ["VoxtKit"]),
        .library(name: "VoxtUI", targets: ["VoxtUI"]),
    ],
    targets: [
        .target(
            name: "CVoxtCore",
            linkerSettings: [
                .linkedLibrary("voxtcore"),
                .linkedLibrary("resolv"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("Security"),
                .unsafeFlags(["-L", macLibraries], .when(platforms: [.macOS])),
            ]
        ),
        .target(name: "VoxtKit", dependencies: ["CVoxtCore"]),
        .target(name: "VoxtUI", dependencies: ["VoxtKit"], resources: [.process("Resources")]),
        .testTarget(name: "VoxtKitTests", dependencies: ["VoxtKit", "VoxtUI"]),
    ]
)
