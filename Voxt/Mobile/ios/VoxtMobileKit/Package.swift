// swift-tools-version: 6.0
import Foundation
import PackageDescription

let package = Package(
    name: "VoxtMobileKit",
    defaultLocalization: "en",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "VoxtMobileKit", targets: ["VoxtMobileKit"]),
    ],
    targets: [
        .target(
            name: "VoxtMobileKit",
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "VoxtMobileKitTests",
            dependencies: ["VoxtMobileKit"],
            // A copy of the Mac's recorded contract, where the repository keeps one; ContractTests
            // read it from source.
            exclude: FileManager.default.fileExists(atPath: Context.packageDirectory + "/Tests/VoxtMobileKitTests/Fixtures")
                ? ["Fixtures"] : []
        ),
    ]
)
