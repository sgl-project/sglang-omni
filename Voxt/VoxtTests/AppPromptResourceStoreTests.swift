// AppPromptResourceStoreTests.swift
// Verifies that every bundled prompt file is registered and loadable.

import Foundation
import XCTest
@testable import Voxt

final class AppPromptResourceStoreTests: XCTestCase {
    private let languages: [(directory: String, language: AppInterfaceLanguage)] = [
        ("en", .english),
        ("zh-Hans", .chineseSimplified),
        ("ja", .japanese)
    ]

    func testEveryRegisteredLocalizedPromptIsBundledForEveryLanguage() {
        for resource in LocalizedPromptResource.allCases {
            for entry in languages {
                let text = AppPromptResourceStore.text(
                    for: resource,
                    language: entry.language
                )
                XCTAssertFalse(
                    text?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true,
                    "Missing \(entry.directory)-\(resource.rawValue).txt"
                )
            }
        }

        for kind in [AppPromptKind.enhancement, .translation, .rewrite] {
            for entry in languages {
                XCTAssertFalse(
                    FeaturePromptPresetCatalog.presets(for: kind, language: entry.language).isEmpty,
                    "Missing presets for \(kind) in \(entry.directory)"
                )
            }
        }
    }

    func testEveryRegisteredSharedPromptIsBundled() {
        for resource in SharedPromptResource.allCases {
            let text = AppPromptResourceStore.text(for: resource)
            XCTAssertFalse(
                text?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true,
                "Missing shared/\(resource.rawValue).txt"
            )
        }
    }

    func testPromptDirectoryContainsNoUnregisteredTextFiles() throws {
        let promptsDirectory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Voxt/Resources/Prompts", isDirectory: true)

        for entry in languages {
            let directory = promptsDirectory.appendingPathComponent(entry.directory, isDirectory: true)
            let fileNames = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            let actualNames = Set(
                fileNames.compactMap { fileName -> String? in
                    guard fileName.hasSuffix(".txt") else { return nil }
                    let stem = String(fileName.dropLast(4))
                    let prefix = "\(entry.directory)-"
                    guard stem.hasPrefix(prefix) else { return stem }
                    return String(stem.dropFirst(prefix.count))
                }
            )
            XCTAssertEqual(
                actualNames,
                AppPromptResourceStore.registeredLocalizedResourceNames,
                "Prompt registry mismatch in \(entry.directory)"
            )
        }

        let sharedDirectory = promptsDirectory.appendingPathComponent("shared", isDirectory: true)
        let sharedNames = Set(
            try FileManager.default.contentsOfDirectory(atPath: sharedDirectory.path)
                .compactMap { fileName -> String? in
                    guard fileName.hasSuffix(".txt") else { return nil }
                    return String(fileName.dropLast(4))
                }
        )
        XCTAssertEqual(
            sharedNames,
            AppPromptResourceStore.registeredSharedResourceNames,
            "Shared prompt registry mismatch"
        )
    }
}
