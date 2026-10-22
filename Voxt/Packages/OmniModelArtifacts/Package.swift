// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "OmniModelArtifacts",
    platforms: [.macOS("10.15.4")],
    products: [.library(name: "OmniModelArtifacts", targets: ["OmniModelArtifacts"])],
    targets: [
        .target(name: "OmniModelStoragePOSIX"),
        .target(name: "OmniModelArtifacts", dependencies: ["OmniModelStoragePOSIX"]),
        .executableTarget(name: "StorageLeaseProbe", dependencies: ["OmniModelArtifacts"], path: "Tests/StorageLeaseProbe"),
        .testTarget(name: "OmniModelArtifactsTests", dependencies: ["OmniModelArtifacts", "StorageLeaseProbe"]),
    ]
)
