// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VoxtConnectionKit",
    platforms: [.macOS(.v14)],
    products: [.library(name: "VoxtConnectionKit", targets: ["VoxtConnectionKit"])],
    targets: [
        .target(name: "VoxtConnectionKit"),
        .testTarget(name: "VoxtConnectionKitTests", dependencies: ["VoxtConnectionKit"]),
    ]
)
