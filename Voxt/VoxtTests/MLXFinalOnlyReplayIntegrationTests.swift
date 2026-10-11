// MLXFinalOnlyReplayIntegrationTests.swift
// Provides MLXFinal Only Replay Integration Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class MLXFinalOnlyReplayIntegrationTests: XCTestCase {
    private let minimumLongFormDurationSeconds = 20.0

    private func requireModelTestsEnabled() throws {
        try ModelTestGate.requireEnabled("Final-only MLX replay integration tests")
    }

    private func officialFixtureDirectoryURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Audio/qwen-official", isDirectory: true)
    }

    private func fixtureURL(named fileName: String) throws -> URL {
        let url = officialFixtureDirectoryURL().appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Missing official audio fixture: \(fileName)")
        }
        return url
    }

    private func discoveredRecentHistoryPaths(limit: Int = 4) -> [URL] {
        let directory = "/Users/guanwei/Library/Application Support/Voxt/transcription-history-audio/transcription"
        guard let enumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: directory),
            includingPropertiesForKeys: [.isRegularFileKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var candidates: [(url: URL, modifiedAt: Date)] = []
        for case let url as URL in enumerator {
            guard url.pathExtension.lowercased() == "wav" else { continue }
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .contentModificationDateKey]),
                  values.isRegularFile == true else {
                continue
            }
            guard let clip = try? DebugAudioClipIO.clip(for: url),
                  clip.durationSeconds >= minimumLongFormDurationSeconds else {
                continue
            }
            candidates.append((url, values.contentModificationDate ?? .distantPast))
        }

        return candidates
            .sorted { $0.modifiedAt > $1.modifiedAt }
            .prefix(limit)
            .map(\.url)
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
            throw XCTSkip("No downloaded multilingual MLX ASR model is available for final-only replay regression.")
        }
        return (fallbackRepo, hubURL)
    }

    private func makeTranscriber() async throws -> MLXTranscriber {
        let resolved = try await resolvedModelRepoAndHubURL()
        return MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: resolved.repo, hubBaseURL: resolved.hubURL)
        )
    }

    func testFinalOnlyReplayOfficialLongFixturePublishesOnlyFinalTranscript() async throws {
        try requireModelTestsEnabled()
        let transcriber = try await makeTranscriber()
        let diagnostics = try await transcriber.debugReplayFinalOnlyAudioFileWithTrace(
            fixtureURL(named: "qwen_audio_long_zh_composite.wav"),
            stepSeconds: 4.0
        )

        XCTAssertFalse(diagnostics.events.contains(where: { !$0.isFinal && $0.source == "post-stop-quick" }))
        XCTAssertTrue(diagnostics.events.contains(where: \.isFinal))
    }

    func testFinalOnlyReplayOfficialLongFixtureProducesMeaningfulFinalTranscript() async throws {
        try requireModelTestsEnabled()
        let transcriber = try await makeTranscriber()
        let diagnostics = try await transcriber.debugReplayFinalOnlyAudioFileWithTrace(
            fixtureURL(named: "qwen_audio_long_en_composite.wav"),
            stepSeconds: 4.0
        )

        let finalEvent = try XCTUnwrap(diagnostics.events.last(where: { $0.isFinal }))

        XCTAssertGreaterThan(
            finalEvent.text.count,
            40,
            """
            Final-only replay should still produce a meaningfully non-empty final transcript.
            finalChars=\(finalEvent.text.count)
            trace:
            \(diagnostics.trace.joined(separator: "\n"))
            """
        )
    }

    func testFinalOnlyReplayRecentHistoryClipsPublishFinalWithoutQuickPreview() async throws {
        try requireModelTestsEnabled()
        let historyURLs = discoveredRecentHistoryPaths()
        guard !historyURLs.isEmpty else {
            throw XCTSkip("No recent local history clips are available for final-only replay diagnostics.")
        }

        let transcriber = try await makeTranscriber()
        for clipURL in historyURLs {
            let diagnostics = try await transcriber.debugReplayFinalOnlyAudioFileWithTrace(
                clipURL,
                stepSeconds: 4.0
            )
            let finalEvent = try XCTUnwrap(diagnostics.events.last(where: { $0.isFinal }))

            XCTAssertFalse(
                diagnostics.events.contains(where: { !$0.isFinal && $0.source == "post-stop-quick" }),
                """
                Final-only replay should not emit a post-stop quick preview for \(clipURL.lastPathComponent)
                trace:
                \(diagnostics.trace.joined(separator: "\n"))
                """
            )
            XCTAssertGreaterThan(finalEvent.text.count, 10)
        }
    }
}
