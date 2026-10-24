// MLXPipelineMetricsIntegrationTests.swift
// Provides MLXPipeline Metrics Integration Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class MLXPipelineMetricsIntegrationTests: XCTestCase {
    private struct ReplayMetrics {
        let clipCount: Int
        let previewPublishedRate: Double
        let previewReuseEligibleRate: Double
        let meanPreviewFinalDiffRate: Double
        let maxPreviewFinalDiffRate: Double
        let lateCoverageRate: Double
    }

    private func requireModelTestsEnabled() throws {
        try ModelTestGate.requireEnabled("MLX pipeline metrics regression tests")
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

    private func officialLongFixtureURLs() throws -> [URL] {
        [
            try fixtureURL(named: "qwen_audio_long_en_composite.wav"),
            try fixtureURL(named: "qwen_audio_long_zh_composite.wav")
        ]
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
            throw XCTSkip("No downloaded multilingual MLX ASR model is available for pipeline metrics regression.")
        }
        return (fallbackRepo, hubURL)
    }

    private func makeTranscriber(repo: String, hubURL: URL) -> MLXTranscriber {
        MLXTranscriber(
            modelManager: MLXModelManager(modelRepo: repo, hubBaseURL: hubURL)
        )
    }

    private func normalizedMetricText(_ text: String) -> [Character] {
        Array(
            text.lowercased()
                .replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
                .replacingOccurrences(
                    of: "[，。！？、；：,.!?;:\"'“”‘’（）()\\-]",
                    with: "",
                    options: .regularExpression
                )
        )
    }

    private func levenshteinDistance(_ lhs: [Character], _ rhs: [Character]) -> Int {
        if lhs.isEmpty { return rhs.count }
        if rhs.isEmpty { return lhs.count }

        var previous = Array(0...rhs.count)
        for (leftIndex, leftChar) in lhs.enumerated() {
            var current = [leftIndex + 1]
            current.reserveCapacity(rhs.count + 1)

            for (rightIndex, rightChar) in rhs.enumerated() {
                let substitutionCost = leftChar == rightChar ? 0 : 1
                current.append(
                    min(
                        current[rightIndex] + 1,
                        previous[rightIndex + 1] + 1,
                        previous[rightIndex] + substitutionCost
                    )
                )
            }
            previous = current
        }

        return previous[rhs.count]
    }

    private func normalizedDiffRate(_ lhs: String, _ rhs: String) -> Double {
        let left = normalizedMetricText(lhs)
        let right = normalizedMetricText(rhs)
        let baseline = max(left.count, right.count, 1)
        return Double(levenshteinDistance(left, right)) / Double(baseline)
    }

    private func isPreviewReuseEligible(
        previewText: String,
        finalText: String,
        diffRate: Double
    ) -> Bool {
        let maxTrustedPreviewCount = finalText.count + max(24, finalText.count / 8)
        return previewText.count <= maxTrustedPreviewCount && diffRate <= 0.45
    }

    private func computeMetrics(from diagnosticsList: [MLXRealtimeReplayDiagnostics]) -> ReplayMetrics {
        var previewCount = 0
        var previewReuseEligibleCount = 0
        var diffRates: [Double] = []
        var lateCoverageCount = 0

        for diagnostics in diagnosticsList {
            guard let finalEvent = diagnostics.events.last(where: { $0.isFinal }) else { continue }
            let liveEvents = diagnostics.events.filter { !$0.isFinal }
            if let previewEvent = diagnostics.events.last(where: { !$0.isFinal && $0.source == "post-stop-quick" }) {
                previewCount += 1
                let diffRate = normalizedDiffRate(previewEvent.text, finalEvent.text)
                diffRates.append(diffRate)
                if isPreviewReuseEligible(
                    previewText: previewEvent.text,
                    finalText: finalEvent.text,
                    diffRate: diffRate
                ) {
                    previewReuseEligibleCount += 1
                }
            }
            let latestLiveTime = liveEvents.map(\.elapsedSeconds).max() ?? 0
            if latestLiveTime >= finalEvent.elapsedSeconds * 0.75 {
                lateCoverageCount += 1
            }
        }

        let clipCount = diagnosticsList.count
        let previewPublishedRate = clipCount > 0 ? Double(previewCount) / Double(clipCount) : 0
        let previewReuseEligibleRate = clipCount > 0 ? Double(previewReuseEligibleCount) / Double(clipCount) : 0
        let meanDiffRate = diffRates.isEmpty ? 0.0 : diffRates.reduce(0, +) / Double(diffRates.count)
        let maxDiffRate = diffRates.max() ?? 0.0
        let lateCoverageRate = clipCount > 0 ? Double(lateCoverageCount) / Double(clipCount) : 0

        return ReplayMetrics(
            clipCount: clipCount,
            previewPublishedRate: previewPublishedRate,
            previewReuseEligibleRate: previewReuseEligibleRate,
            meanPreviewFinalDiffRate: meanDiffRate,
            maxPreviewFinalDiffRate: maxDiffRate,
            lateCoverageRate: lateCoverageRate
        )
    }

    func testFinalOnlyOfficialLongFixtureMetricsStayWithinExpectedEnvelope() async throws {
        try requireModelTestsEnabled()
        let resolved = try await resolvedModelRepoAndHubURL()
        let transcriber = makeTranscriber(repo: resolved.repo, hubURL: resolved.hubURL)
        let diagnosticsList = try await officialLongFixtureURLs().asyncMap {
            try await transcriber.debugReplayFinalOnlyAudioFileWithTrace($0, stepSeconds: 4.0)
        }
        let metrics = computeMetrics(from: diagnosticsList)

        print(
            "ASR_METRICS provider=mlx model=\(resolved.repo) pipeline=finalOnly " +
            "clips=\(metrics.clipCount) previewPublishedRate=\(metrics.previewPublishedRate) " +
            "previewReuseEligibleRate=\(metrics.previewReuseEligibleRate) " +
            "meanPreviewFinalDiffRate=\(metrics.meanPreviewFinalDiffRate) " +
            "maxPreviewFinalDiffRate=\(metrics.maxPreviewFinalDiffRate)"
        )

        XCTAssertEqual(metrics.clipCount, 2)
        XCTAssertEqual(
            metrics.previewPublishedRate,
            0.0,
            "Final-only official long fixtures should no longer emit a post-stop quick preview."
        )
        XCTAssertEqual(
            metrics.previewReuseEligibleRate,
            0.0,
            "Final-only official long fixtures should not rely on preview reuse."
        )
        XCTAssertEqual(metrics.meanPreviewFinalDiffRate, 0.0, accuracy: 0.0001)
        XCTAssertEqual(metrics.maxPreviewFinalDiffRate, 0.0, accuracy: 0.0001)
    }

    func testRealtimeOfficialLongFixtureMetricsStayWithinExpectedEnvelope() async throws {
        try requireModelTestsEnabled()
        let resolved = try await resolvedModelRepoAndHubURL()
        let transcriber = makeTranscriber(repo: resolved.repo, hubURL: resolved.hubURL)
        let diagnosticsList = try await officialLongFixtureURLs().asyncMap {
            try await transcriber.debugReplayRealtimeAudioFileWithTrace($0, stepSeconds: 4.0)
        }
        let metrics = computeMetrics(from: diagnosticsList)

        print(
            "ASR_METRICS provider=mlx model=\(resolved.repo) pipeline=liveDisplay " +
            "clips=\(metrics.clipCount) previewPublishedRate=\(metrics.previewPublishedRate) " +
            "previewReuseEligibleRate=\(metrics.previewReuseEligibleRate) " +
            "meanPreviewFinalDiffRate=\(metrics.meanPreviewFinalDiffRate) " +
            "maxPreviewFinalDiffRate=\(metrics.maxPreviewFinalDiffRate) " +
            "lateCoverageRate=\(metrics.lateCoverageRate)"
        )

        XCTAssertEqual(metrics.clipCount, 2)
        XCTAssertGreaterThanOrEqual(
            metrics.previewPublishedRate,
            1.0,
            "Realtime official long fixtures should always publish a stop preview."
        )
        XCTAssertGreaterThanOrEqual(
            metrics.lateCoverageRate,
            1.0,
            "Realtime official long fixtures should keep publishing into the later portion of the clip."
        )
        XCTAssertLessThanOrEqual(
            metrics.meanPreviewFinalDiffRate,
            0.45,
            "Realtime preview vs final divergence is too high on average."
        )
        XCTAssertLessThanOrEqual(
            metrics.maxPreviewFinalDiffRate,
            0.65,
            "At least one realtime preview diverged too far from the final transcript."
        )
    }
}

private extension Sequence {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var results: [T] = []
        for element in self {
            let transformed = try await transform(element)
            results.append(transformed)
        }
        return results
    }
}
