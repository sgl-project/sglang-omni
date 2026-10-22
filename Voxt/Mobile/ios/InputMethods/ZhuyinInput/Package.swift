// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ZhuyinInput",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "ZhuyinInput", targets: ["ZhuyinInput"])],
    dependencies: [.package(path: "../../TypingInput"), .package(path: "../McBopomofoEngine")],
    targets: [
        .target(name: "CZhuyin", dependencies: [.product(name: "CMcBopomofo", package: "McBopomofoEngine")], publicHeadersPath: "include"),
        .target(name: "ZhuyinInput", dependencies: ["CZhuyin", "TypingInput"], resources: [.copy("Resources/data.txt"), .copy("Resources/Notices")]),
        .testTarget(name: "ZhuyinInputTests", dependencies: ["ZhuyinInput"])
    ],
    cxxLanguageStandard: .cxx17
)
