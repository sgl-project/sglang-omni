// ModelDownloadSourceSupportTests.swift
// Tests automatic source selection for model downloads.

import XCTest
@testable import Voxt

final class ModelDownloadSourceSupportTests: XCTestCase {
    func testSelectChoosesFastestReachableSourceAndPersistsIt() async throws {
        let defaults = try makeDefaults()
        let slow = candidate(id: "slow", url: "https://slow.example.com")
        let fast = candidate(id: "fast", url: "https://fast.example.com")

        let selection = try await ModelDownloadSourceSelector.select(
            candidates: [slow, fast],
            targetKey: "test:model",
            reuseSavedSource: false,
            defaults: defaults
        ) { candidate in
            if candidate.id == "fast" {
                return (0.01, 100)
            }
            return (0.5, 100)
        }

        XCTAssertEqual(selection.candidate.id, "fast")
        XCTAssertFalse(selection.reusedSavedSource)
        XCTAssertEqual(
            ModelDownloadSourceSelectionStore.savedSourceID(for: "test:model", defaults: defaults),
            "fast"
        )
    }

    func testSelectReusesSavedSourceWithoutProbingWhenRequested() async throws {
        let defaults = try makeDefaults()
        let saved = candidate(id: "saved", url: "https://saved.example.com")
        let other = candidate(id: "other", url: "https://other.example.com")
        ModelDownloadSourceSelectionStore.saveSourceID("saved", for: "test:model", defaults: defaults)

        let selection = try await ModelDownloadSourceSelector.select(
            candidates: [other, saved],
            targetKey: "test:model",
            reuseSavedSource: true,
            defaults: defaults
        ) { _ in
            return (0.01, 100)
        }

        XCTAssertEqual(selection.candidate.id, "saved")
        XCTAssertTrue(selection.reusedSavedSource)
        XCTAssertTrue(selection.probeResults.isEmpty)
    }

    func testSelectIgnoresSavedSourceForFreshInstall() async throws {
        let defaults = try makeDefaults()
        let saved = candidate(id: "saved", url: "https://saved.example.com")
        let fast = candidate(id: "fast", url: "https://fast.example.com")
        ModelDownloadSourceSelectionStore.saveSourceID("saved", for: "test:model", defaults: defaults)

        let selection = try await ModelDownloadSourceSelector.select(
            candidates: [saved, fast],
            targetKey: "test:model",
            reuseSavedSource: false,
            defaults: defaults
        ) { candidate in
            candidate.id == "fast" ? (0.01, 100) : (0.5, 100)
        }

        XCTAssertEqual(selection.candidate.id, "fast")
        XCTAssertFalse(selection.reusedSavedSource)
        XCTAssertEqual(
            ModelDownloadSourceSelectionStore.savedSourceID(for: "test:model", defaults: defaults),
            "fast"
        )
    }

    @MainActor
    func testResumeAttemptStaysOnSavedSourceEvenWhenAnotherProbeIsFaster() {
        let saved = candidate(id: "saved", url: "https://saved.example.com")
        let fast = candidate(id: "fast", url: "https://fast.example.com")
        let selection = ModelDownloadSourceSelection(
            candidate: saved, reusedSavedSource: true,
            probeResults: [.init(candidate: fast, elapsed: 0.01, bytes: 100, errorDescription: nil)]
        )
        XCTAssertEqual(selection.attemptCandidates, [saved])
    }

    @MainActor
    func testFreshAttemptsExcludeFailuresAndFollowProbeLatency() {
        let slow = candidate(id: "slow", url: "https://slow.example.com")
        let fast = candidate(id: "fast", url: "https://fast.example.com")
        let failed = candidate(id: "failed", url: "https://failed.example.com")
        let selection = ModelDownloadSourceSelection(
            candidate: fast, reusedSavedSource: false,
            probeResults: [
                .init(candidate: slow, elapsed: 1, bytes: 100, errorDescription: nil),
                .init(candidate: failed, elapsed: 0, bytes: 0, errorDescription: "offline"),
                .init(candidate: fast, elapsed: 0.1, bytes: 100, errorDescription: nil)
            ]
        )
        XCTAssertEqual(selection.attemptCandidates, [fast, slow])
    }

    @MainActor
    func testMissingOrUnreachableProbesKeepSelectedSource() {
        let selected = candidate(id: "selected", url: "https://selected.example.com")
        let probeSets: [[ModelDownloadSourceProbeResult]] = [
            [], [.init(candidate: selected, elapsed: 0, bytes: 0, errorDescription: "offline")]
        ]
        for probes in probeSets {
            let selection = ModelDownloadSourceSelection(candidate: selected, reusedSavedSource: false, probeResults: probes)
            XCTAssertEqual(selection.attemptCandidates, [selected])
        }
    }

    private func makeDefaults() throws -> UserDefaults {
        let suiteName = "VoxtTests.ModelDownloadSourceSupportTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        addTeardownBlock {
            UserDefaults(suiteName: suiteName)?.removePersistentDomain(forName: suiteName)
        }
        return defaults
    }

    private func candidate(id: String, url: String) -> ModelDownloadSourceCandidate {
        ModelDownloadSourceCandidate(
            id: id,
            displayName: id,
            url: URL(string: url)!
        )
    }
}
