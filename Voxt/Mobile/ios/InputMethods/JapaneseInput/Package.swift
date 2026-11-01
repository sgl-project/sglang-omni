// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "JapaneseInput",
    platforms: [.iOS(.v17), .macOS(.v13)],
    products: [
        .library(name: "JapaneseInput", targets: ["JapaneseInput"])
    ],
    dependencies: [
        // Exact pins for every transitive dependency; licenses in Resources/Notices.
        .package(url: "https://github.com/azooKey/AzooKeyKanaKanjiConverter", exact: "0.11.2"),
        .package(url: "https://github.com/ensan-hcl/swift-tokenizers", exact: "0.0.1"),
        .package(url: "https://github.com/johnmai-dev/Jinja", exact: "1.1.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", exact: "1.4.0"),
        .package(url: "https://github.com/apple/swift-collections", exact: "1.7.3"),
        .package(url: "https://github.com/apple/swift-algorithms", exact: "1.2.1"),
        .package(url: "https://github.com/apple/swift-numerics", exact: "1.1.1"),
        .package(path: "../../TypingInput"),
    ],
    targets: [
        .target(
            name: "JapaneseInput",
            dependencies: [
                .product(name: "KanaKanjiConverterModule", package: "AzooKeyKanaKanjiConverter"),
                .product(name: "TypingInput", package: "TypingInput"),
            ],
            resources: [
                .copy("Resources/Dictionary"),
                .process("Resources/Notices")
            ]
        ),
        .testTarget(
            name: "JapaneseInputTests",
            dependencies: ["JapaneseInput"]
        ),
    ]
)
