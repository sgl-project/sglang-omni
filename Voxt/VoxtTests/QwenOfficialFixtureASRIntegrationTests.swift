// QwenOfficialFixtureASRIntegrationTests.swift
// Provides Qwen Official Fixture ASRIntegration Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class QwenOfficialFixtureASRIntegrationTests: XCTestCase {
    private func requireModelTestsEnabled() throws {
        try ModelTestGate.requireEnabled("Qwen official fixture ASR integration tests")
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

    private func resolvedModelRepoAndHubURL() async throws -> (repo: String, hubURL: URL) {
        let defaults = UserDefaults.standard
        ModelTestGate.configureStorageRoot(for: self)
        let hubURL = defaults.bool(forKey: AppPreferenceKey.useHfMirror)
            ? MLXModelManager.mirrorHubBaseURL
            : MLXModelManager.defaultHubBaseURL

        let preferredRepo = MLXModelManager.canonicalModelRepo(
            defaults.string(forKey: AppPreferenceKey.mlxModelRepo) ?? MLXModelManager.defaultModelRepo
        )
        let probeManager = MLXModelManager(modelRepo: preferredRepo, hubBaseURL: hubURL)
        try await ModelTestGate.waitForASRInstallations(probeManager)
        if probeManager.isModelDownloaded(repo: preferredRepo),
           MLXModelManager.isMultilingualModelRepo(preferredRepo) {
            return (preferredRepo, hubURL)
        }

        guard let fallbackRepo = MLXModelManager.availableModels
            .map(\.id)
            .map(MLXModelManager.canonicalModelRepo(_:))
            .first(where: {
                MLXModelManager.isMultilingualModelRepo($0) &&
                    probeManager.isModelDownloaded(repo: $0)
            }) else {
            throw XCTSkip("No downloaded multilingual MLX ASR model is available for official fixture regression.")
        }
        return (fallbackRepo, hubURL)
    }

    private func makeTranscriber() async throws -> MLXTranscriber {
        let resolved = try await resolvedModelRepoAndHubURL()
        return MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: resolved.repo, hubBaseURL: resolved.hubURL)
        )
    }

    private func normalizedLatinText(_ text: String) -> String {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func normalizedHanText(_ text: String) -> String {
        text.replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
            .replacingOccurrences(of: "[，。！？、；：,.!?;:\"'“”‘’（）()\\-]", with: "", options: .regularExpression)
    }

    func testOfficialEnglishFixtureContainsExpectedTranscriptAnchors() async throws {
        try requireModelTestsEnabled()
        let transcriber = try await makeTranscriber()
        let text = try await transcriber.transcribeAudioFile(
            fixtureURL(named: "qwen_audio_short_en.wav")
        )
        let normalized = normalizedLatinText(text)

        XCTAssertTrue(
            normalized.contains("middle classes"),
            "Expected transcript to preserve the official sample phrase 'middle classes'. Got: \(text)"
        )
        XCTAssertTrue(
            normalized.contains("welcome his gospel"),
            "Expected transcript to preserve the ending phrase from the official sample. Got: \(text)"
        )
    }

    func testOfficialChongqingChineseFixtureContainsExpectedTranscriptAnchors() async throws {
        try requireModelTestsEnabled()
        let transcriber = try await makeTranscriber()
        let text = try await transcriber.transcribeAudioFile(
            fixtureURL(named: "qwen_audio_short_zh_chongqing.wav")
        )
        let normalized = normalizedHanText(text)

        XCTAssertTrue(
            normalized.contains("租一些自行车"),
            "Expected Chongqing dialect sample to mention renting bicycles. Got: \(text)"
        )
        XCTAssertTrue(
            normalized.contains("锻炼身体"),
            "Expected Chongqing dialect sample to preserve the exercise phrase. Got: \(text)"
        )
    }

    func testOfficialShortChineseEmotionFixturesPreserveCoreUtterance() async throws {
        try requireModelTestsEnabled()
        let transcriber = try await makeTranscriber()

        for fileName in ["qwen_audio_short_zh_relaxed.wav", "qwen_audio_short_zh_negative.wav"] {
            let text = try await transcriber.transcribeAudioFile(fixtureURL(named: fileName))
            let normalized = normalizedHanText(text)
            XCTAssertTrue(
                normalized.contains("你没事吧"),
                "Expected \(fileName) to preserve the core utterance '你没事吧'. Got: \(text)"
            )
        }
    }
}
