// MLXLongFormReplayIntegrationTests.swift
// Provides MLXLong Form Replay Integration Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class MLXLongFormReplayIntegrationTests: XCTestCase {
    private struct FixtureManifest: Decodable {
        let fixtures: [FixtureEntry]
    }

    private struct FixtureEntry: Decodable {
        let filename: String
        let referenceText: String?
        let sourceParts: [String]?

        enum CodingKeys: String, CodingKey {
            case filename
            case referenceText = "reference_text"
            case sourceParts = "source_parts"
        }
    }

    private let minimumLongFormDurationSeconds = 30.0
    private let replayStepSeconds = 4.0

    private func requireModelTestsEnabled() throws {
        try ModelTestGate.requireEnabled("MLX long-form replay integration tests")
    }

    private func officialFixtureDirectoryURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Audio/qwen-official", isDirectory: true)
    }

    private func officialFixtureManifest() throws -> FixtureManifest {
        let manifestURL = officialFixtureDirectoryURL().appendingPathComponent("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        return try JSONDecoder().decode(FixtureManifest.self, from: data)
    }

    private func officialFixturePath(named fileName: String) -> String? {
        let path = officialFixtureDirectoryURL().appendingPathComponent(fileName).path
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    private func resolvedOfficialLongFormClipPaths() -> [String] {
        [
            officialFixturePath(named: "qwen_audio_long_en_composite.wav"),
            officialFixturePath(named: "qwen_audio_long_zh_composite.wav")
        ]
        .compactMap { $0 }
    }

    private func discoveredWAVPaths(in directoryPath: String, limit: Int = 24) -> [String] {
        guard let enumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: directoryPath),
            includingPropertiesForKeys: [.isRegularFileKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var candidates: [(path: String, modifiedAt: Date)] = []
        for case let url as URL in enumerator {
            guard url.pathExtension.lowercased() == "wav" else { continue }
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .contentModificationDateKey]),
                  values.isRegularFile == true
            else {
                continue
            }
            candidates.append((url.path, values.contentModificationDate ?? .distantPast))
        }

        return candidates
            .sorted { $0.modifiedAt > $1.modifiedAt }
            .prefix(limit)
            .map(\.path)
    }

    private func resolvedCandidateClipPaths() -> [String] {
        let candidates = resolvedOfficialLongFormClipPaths() + [
            ProcessInfo.processInfo.environment["VOXT_LONGFORM_REPLAY_CLIP"],
            "/Users/guanwei/Library/Application Support/Voxt/transcription-history-audio/transcription/transcription-6247D986-B2EC-4758-AB40-7C1030296D7A.wav",
            "/Users/guanwei/Library/Application Support/Voxt/transcription-history-audio/transcription/transcription-FD3C99FC-822F-45DB-8734-FFADEF6DC6EE.wav",
            "/Users/guanwei/Downloads/transcription/20260505-104918-transcription-5F2FAD9F-D22E-4D0E-BA36-E1D95A53197D.wav",
            "/Users/guanwei/Downloads/transcription/transcription-0A0E87B1-7C9A-4BB6-8469-E18485A63103.wav",
            "/Users/guanwei/Downloads/transcription/20260507-123725-transcription-CF5D4F69-31F4-4F86-ADCA-18029BDB0EE8.wav"
        ]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && FileManager.default.fileExists(atPath: $0) }

        let discovered = discoveredWAVPaths(
            in: "/Users/guanwei/Library/Application Support/Voxt/transcription-history-audio/transcription"
        ) + discoveredWAVPaths(in: "/Users/guanwei/Downloads/transcription")

        var seen = Set<String>()
        return (candidates + discovered).filter { seen.insert($0).inserted }
    }

    private func resolvedLongFormClipPaths() -> [String] {
        resolvedCandidateClipPaths().filter { path in
            guard let clip = try? DebugAudioClipIO.clip(for: URL(fileURLWithPath: path)) else {
                return false
            }
            return clip.durationSeconds >= minimumLongFormDurationSeconds
        }
    }

    private func sampledLongFormClipPaths(limit: Int = 2) -> [String] {
        Array(resolvedLongFormClipPaths().prefix(limit))
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
            throw XCTSkip("No downloaded multilingual MLX model is available for long-form replay.")
        }
        return (fallbackRepo, hubURL)
    }

    func testReplayLongFormClipTracksOfflineBaselineAndTail() async throws {
        try requireModelTestsEnabled()
        let clipPaths = resolvedLongFormClipPaths()
        guard let clipPath = clipPaths.first else {
            throw XCTSkip("No long-form MLX replay clip is available.")
        }

        let resolved = try await resolvedModelRepoAndHubURL()
        let transcriber = MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: resolved.repo, hubBaseURL: resolved.hubURL)
        )
        let clipURL = URL(fileURLWithPath: clipPath)

        let offlineText = try await transcriber.transcribeAudioFile(clipURL)
        let diagnostics = try await transcriber.debugReplayRealtimeAudioFileWithTrace(
            clipURL,
            stepSeconds: replayStepSeconds
        )
        let finalEvent = try XCTUnwrap(diagnostics.events.last(where: { $0.isFinal }))

        let offlineCount = offlineText.trimmingCharacters(in: .whitespacesAndNewlines).count
        let finalCount = finalEvent.text.trimmingCharacters(in: .whitespacesAndNewlines).count

        XCTAssertFalse(offlineText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertFalse(finalEvent.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertGreaterThan(offlineCount, 20)
        XCTAssertGreaterThanOrEqual(
            finalCount,
            Int(Double(offlineCount) * 0.6),
            "MLX replay final transcript collapsed too far below the offline baseline."
        )

        let offlineTail = normalizedTailText(from: offlineText, length: 24)
        let replayTail = normalizedTailText(from: finalEvent.text, length: 24)
        let sharedSuffixCount = longestCommonSuffixCount(offlineTail, replayTail)
        XCTAssertGreaterThanOrEqual(
            sharedSuffixCount,
            min(8, offlineTail.count),
            "MLX replay final transcript appears to lose too much of the tail compared with the offline baseline."
        )
    }

    func testReplaySampledLongFormClipsProduceNonEmptyFinalTranscript() async throws {
        try requireModelTestsEnabled()
        let clipPaths = sampledLongFormClipPaths()
        guard !clipPaths.isEmpty else {
            throw XCTSkip("No long-form MLX replay clips are available.")
        }

        let resolved = try await resolvedModelRepoAndHubURL()
        let transcriber = MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: resolved.repo, hubBaseURL: resolved.hubURL)
        )

        for clipPath in clipPaths {
            let diagnostics = try await transcriber.debugReplayRealtimeAudioFileWithTrace(
                URL(fileURLWithPath: clipPath),
                stepSeconds: replayStepSeconds
            )
            let finalEvent = try XCTUnwrap(diagnostics.events.last(where: { $0.isFinal }))
            XCTAssertFalse(
                finalEvent.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "Expected non-empty final transcript for \(clipPath)"
            )
        }
    }

    func testReplaySampledLongFormClipsTrackOfflineTailAndLateCoverage() async throws {
        try requireModelTestsEnabled()
        let clipPaths = sampledLongFormClipPaths()
        guard !clipPaths.isEmpty else {
            throw XCTSkip("No long-form MLX replay clips are available.")
        }

        let resolved = try await resolvedModelRepoAndHubURL()
        let transcriber = MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: resolved.repo, hubBaseURL: resolved.hubURL)
        )

        for clipPath in clipPaths {
            let clipURL = URL(fileURLWithPath: clipPath)
            let clip = try DebugAudioClipIO.clip(for: clipURL)
            let offlineText = try await transcriber.transcribeAudioFile(clipURL)
            let diagnostics = try await transcriber.debugReplayRealtimeAudioFileWithTrace(
                clipURL,
                stepSeconds: replayStepSeconds
            )
            let finalEvent = try XCTUnwrap(diagnostics.events.last(where: { $0.isFinal }))

            let offlineTail = normalizedTailText(from: offlineText, length: 24)
            let replayTail = normalizedTailText(from: finalEvent.text, length: 24)
            let sharedSuffixCount = longestCommonSuffixCount(offlineTail, replayTail)
            XCTAssertGreaterThanOrEqual(
                sharedSuffixCount,
                min(8, offlineTail.count),
                "MLX replay final transcript appears to lose too much tail for \(clipPath)"
            )

            if clip.durationSeconds >= minimumLongFormDurationSeconds {
                let latestLiveEventTime = diagnostics.events
                    .filter { !$0.isFinal }
                    .map(\.elapsedSeconds)
                    .max() ?? 0
                XCTAssertGreaterThanOrEqual(
                    latestLiveEventTime,
                    clip.durationSeconds * 0.75,
                    "MLX replay should keep publishing updates into the later portion of the clip for \(clipPath)"
                )
            }
        }
    }

    func testOfficialLongFormFixturesAreAvailableLocally() throws {
        try requireModelTestsEnabled()
        let paths = resolvedOfficialLongFormClipPaths()
        XCTAssertEqual(paths.count, 2, "Expected both official composite long-form fixtures to be present locally.")

        for path in paths {
            let clip = try DebugAudioClipIO.clip(for: URL(fileURLWithPath: path))
            XCTAssertGreaterThanOrEqual(
                clip.durationSeconds,
                minimumLongFormDurationSeconds,
                "Expected official long-form fixture to be at least \(minimumLongFormDurationSeconds)s: \(path)"
            )
        }
    }

    func testOfficialCompositeLongFixturesPreserveTailAnchors() async throws {
        try requireModelTestsEnabled()
        let manifest = try officialFixtureManifest()
        let resolved = try await resolvedModelRepoAndHubURL()
        let transcriber = MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: resolved.repo, hubBaseURL: resolved.hubURL)
        )

        for fileName in ["qwen_audio_long_en_composite.wav", "qwen_audio_long_zh_composite.wav"] {
            let clipURL = try XCTUnwrap(
                officialFixturePath(named: fileName).map(URL.init(fileURLWithPath:)),
                "Missing official long-form fixture \(fileName)"
            )
            let text = try await transcriber.transcribeAudioFile(clipURL)
            try assertTailAnchorPreserved(
                for: fileName,
                transcript: text,
                manifest: manifest
            )
        }
    }

    private func normalizedTailText(from text: String, length: Int) -> String {
        let filtered = text.unicodeScalars
            .filter { CharacterSet.alphanumerics.contains($0) }
            .map(String.init)
            .joined()
            .lowercased()
        guard filtered.count > length else { return filtered }
        return String(filtered.suffix(length))
    }

    private func longestCommonSuffixCount(_ lhs: String, _ rhs: String) -> Int {
        let lhsChars = Array(lhs)
        let rhsChars = Array(rhs)
        var count = 0
        while count < lhsChars.count,
              count < rhsChars.count,
              lhsChars[lhsChars.count - 1 - count] == rhsChars[rhsChars.count - 1 - count] {
            count += 1
        }
        return count
    }

    private func assertTailAnchorPreserved(
        for fileName: String,
        transcript: String,
        manifest: FixtureManifest,
    ) throws {
        guard let compositeEntry = manifest.fixtures.first(where: { $0.filename == fileName }),
              let lastPart = compositeEntry.sourceParts?.last,
              let lastPartReference = manifest.fixtures.first(where: { $0.filename == lastPart })?.referenceText
        else {
            XCTFail("Missing manifest metadata for long-form composite tail anchor: \(fileName)")
            return
        }

        if fileName.hasSuffix("_en_composite.wav") {
            let transcriptTokens = normalizedLatinTokens(transcript)
            let anchorTokens = tailLatinAnchorTokens(lastPartReference)
            XCTAssertGreaterThanOrEqual(
                transcriptTokens.count,
                anchorTokens.count,
                "English long-form transcript is shorter than the expected tail anchor. Got: \(transcript)"
            )
            XCTAssertEqual(
                Array(transcriptTokens.suffix(anchorTokens.count)),
                anchorTokens,
                "Expected the long-form English transcript tail to preserve the final official sample anchor. Got: \(transcript)"
            )
        } else {
            let transcriptTail = normalizedHanText(transcript)
            let anchor = tailHanAnchor(lastPartReference)
            XCTAssertTrue(
                transcriptTail.hasSuffix(anchor),
                "Expected the long-form Chinese transcript tail to preserve the final official sample anchor '\(anchor)'. Got: \(transcript)"
            )
        }
    }

    private func normalizedLatinTokens(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    private func tailLatinAnchorTokens(_ text: String, wordCount: Int = 4) -> [String] {
        let tokens = normalizedLatinTokens(text)
        return Array(tokens.suffix(min(wordCount, tokens.count)))
    }

    private func tailHanAnchor(_ text: String, length: Int = 6) -> String {
        let normalized = normalizedHanText(text)
        guard normalized.count > length else { return normalized }
        return String(normalized.suffix(length))
    }

    private func normalizedHanText(_ text: String) -> String {
        text.replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
            .replacingOccurrences(of: "[，。！？、；：,.!?;:\"'“”‘’（）()\\-]", with: "", options: .regularExpression)
    }
}
