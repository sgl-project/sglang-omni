// SenseVoiceOfficialFixtureIntegrationTests.swift
// Provides Sense Voice Official Fixture Integration Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class SenseVoiceOfficialFixtureIntegrationTests: XCTestCase {
    private let repo = "mlx-community/SenseVoiceSmall"

    private func requireModelTestsEnabled() throws {
        try ModelTestGate.requireEnabled("SenseVoice official fixture integration tests")
    }

    private func fixtureDirectoryURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Audio/qwen-official", isDirectory: true)
    }

    private func fixtureURL(named fileName: String) throws -> URL {
        let url = fixtureDirectoryURL().appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Missing official audio fixture: \(fileName)")
        }
        return url
    }

    private func makeTranscriber() throws -> MLXTranscriber {
        let defaults = UserDefaults.standard
        let previousModelRoot = defaults.object(forKey: AppPreferenceKey.modelStorageRootPath)
        let previousModelRootBookmark = defaults.object(forKey: AppPreferenceKey.modelStorageRootBookmark)
        addTeardownBlock {
            if let previousModelRoot {
                defaults.set(previousModelRoot, forKey: AppPreferenceKey.modelStorageRootPath)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
            }
            if let previousModelRootBookmark {
                defaults.set(previousModelRootBookmark, forKey: AppPreferenceKey.modelStorageRootBookmark)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
            }
            ModelStorageDirectoryManager.resetForTesting()
        }
        if let modelRoot = ProcessInfo.processInfo.environment["VOXT_MODEL_STORAGE_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !modelRoot.isEmpty {
            defaults.set(modelRoot, forKey: AppPreferenceKey.modelStorageRootPath)
            defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
            ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(
                URL(fileURLWithPath: modelRoot, isDirectory: true)
            )
        }
        let hubURL = defaults.bool(forKey: AppPreferenceKey.useHfMirror)
            ? MLXModelManager.mirrorHubBaseURL
            : MLXModelManager.defaultHubBaseURL

        let manager = MLXModelManager(modelRepo: repo, hubBaseURL: hubURL)
        guard manager.modelDirectoryURL(repo: repo) != nil else {
            throw XCTSkip("SenseVoiceSmall is not available locally for regression testing. Set VOXT_MODEL_STORAGE_ROOT if needed.")
        }
        return MLXTranscriber(modelManager: manager)
    }

    private func isNumericTokenSequence(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        return trimmed.range(
            of: #"^\d+(?:\s+\d+)+$"#,
            options: .regularExpression
        ) != nil
    }

    private func normalizedLatinText(_ text: String) -> String {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    func testSenseVoiceEnglishFixtureDecodesReadableText() async throws {
        try requireModelTestsEnabled()
        let transcriber = try makeTranscriber()
        let text = try await transcriber.transcribeAudioFile(
            fixtureURL(named: "qwen_audio_short_en.wav")
        )
        let metadata = transcriber.latestSenseVoiceMetadata

        XCTAssertFalse(
            isNumericTokenSequence(text),
            "SenseVoice regressed to raw token IDs: \(text)"
        )
        XCTAssertTrue(
            normalizedLatinText(text).contains("middle classes"),
            "Expected SenseVoice to decode readable English text. Got: \(text)"
        )
        XCTAssertNotNil(metadata)
        XCTAssertFalse(metadata?.segments.isEmpty ?? true)
        XCTAssertEqual(metadata?.usedVADSegmentation, false)
    }

    func testSenseVoiceLongChineseFixtureDoesNotDegenerateToTokenIDs() async throws {
        try requireModelTestsEnabled()
        let transcriber = try makeTranscriber()
        let text = try await transcriber.transcribeAudioFile(
            fixtureURL(named: "qwen_audio_long_zh_composite.wav")
        )
        let metadata = transcriber.latestSenseVoiceMetadata

        XCTAssertFalse(
            isNumericTokenSequence(text),
            "SenseVoice long-form output regressed to raw token IDs: \(text)"
        )
        XCTAssertGreaterThan(
            text.trimmingCharacters(in: .whitespacesAndNewlines).count,
            40,
            "Expected a non-trivial long-form transcript. Got: \(text)"
        )
        XCTAssertNotNil(metadata)
        XCTAssertTrue(metadata?.usedVADSegmentation ?? false)
        XCTAssertFalse(metadata?.segments.isEmpty ?? true)
    }
}
