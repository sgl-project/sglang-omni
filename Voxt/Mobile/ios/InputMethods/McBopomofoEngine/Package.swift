// swift-tools-version: 6.0
// McBopomofo 3.1.1's reading grid and phrase database (MIT), unmodified, shared by the
// Zhuyin and pinyin layouts so the keyboard links one copy.
import PackageDescription

let package = Package(
    name: "McBopomofoEngine",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [.library(name: "CMcBopomofo", targets: ["CMcBopomofo"])],
    targets: [
        // Headers are included by path ("Engine/ParselessLM.h"), from C++ wrappers only.
        .target(name: "CMcBopomofo", publicHeadersPath: "."),
    ],
    cxxLanguageStandard: .cxx17
)
