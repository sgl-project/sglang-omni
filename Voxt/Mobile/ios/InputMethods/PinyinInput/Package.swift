// swift-tools-version: 6.0
// Simplified Chinese pinyin: McBopomofo's reading grid (MIT) with a pruned
// rime-wanxiang dictionary (CC-BY-4.0). Everything converts on the phone.
import PackageDescription

let package = Package(
    name: "PinyinInput",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "PinyinInput", targets: ["PinyinInput"])],
    dependencies: [.package(path: "../../TypingInput"), .package(path: "../McBopomofoEngine")],
    targets: [
        .target(name: "CPinyin", dependencies: [.product(name: "CMcBopomofo", package: "McBopomofoEngine")], publicHeadersPath: "include"),
        .target(name: "PinyinInput", dependencies: ["CPinyin", "TypingInput"],
                resources: [.copy("Resources/pinyin-data.txt"), .copy("Resources/Notices")]),
        .testTarget(name: "PinyinInputTests", dependencies: ["PinyinInput"], resources: [.copy("Fixtures")])
    ],
    cxxLanguageStandard: .cxx17
)
