// RecordingStartPlannerTests.swift
// Provides Recording Start Planner Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RecordingStartPlannerTests: XCTestCase {
    func testMLXAudioNotDownloadedBlocksRecordingStart() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            mlxModelState: .notDownloaded
        )

        XCTAssertEqual(decision, .blocked(.mlxModelNotInstalled))
    }

    func testMLXAudioErrorBlocksRecordingStart() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            mlxModelState: .error("broken")
        )

        XCTAssertEqual(decision, .blocked(.mlxModelUnavailable(detail: "broken")))
    }

    func testMLXAudioDownloadedStartsWithMLXAudio() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            mlxModelState: .downloaded
        )

        XCTAssertEqual(decision, .start(.mlxAudio))
    }

    func testMLXAudioDownloadingBlocksRecordingStart() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            selectedMLXRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            activeMLXDownloadRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            mlxModelState: .downloading(
                progress: 0.5,
                completed: 10,
                total: 20,
                currentFile: "weights.bin",
                completedFiles: 1,
                totalFiles: 2
            )
        )

        XCTAssertEqual(decision, .blocked(.mlxModelDownloading))
    }

    func testMLXAudioDownloadingDifferentRepoDoesNotBlockInstalledSelection() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            selectedMLXRepo: "mlx-community/parakeet-tdt-0.6b-v3",
            activeMLXDownloadRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            isSelectedMLXModelDownloaded: true,
            mlxModelState: .downloading(
                progress: 0.5,
                completed: 10,
                total: 20,
                currentFile: "weights.bin",
                completedFiles: 1,
                totalFiles: 2
            )
        )

        XCTAssertEqual(decision, .start(.mlxAudio))
    }

    func testMLXAudioPausedSelectedModelBlocksRecordingStart() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: .mlxAudio,
            selectedMLXRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            activeMLXDownloadRepo: "mlx-community/Qwen3-ASR-0.6B-4bit",
            mlxModelState: .paused(
                progress: 0.5,
                completed: 10,
                total: 20,
                currentFile: "weights.bin",
                completedFiles: 1,
                totalFiles: 2
            )
        )

        XCTAssertEqual(decision, .blocked(.mlxModelDownloading))
    }

    func testDictationStartIgnoresMLXModelState() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: .dictation,
            mlxModelState: .notDownloaded
        )

        XCTAssertEqual(decision, .start(.dictation))
    }

    func testLegacyWhisperUsesMLXAvailabilityWhenNotDownloaded() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: TranscriptionEngine.resolved(rawValue: "whisperKit"),
            mlxModelState: .notDownloaded
        )

        XCTAssertEqual(decision, .blocked(.mlxModelNotInstalled))
    }

    func testLegacyWhisperUsesMLXAvailabilityWhenUnavailable() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: TranscriptionEngine.resolved(rawValue: "whisperKit"),
            mlxModelState: .error("broken")
        )

        XCTAssertEqual(decision, .blocked(.mlxModelUnavailable(detail: "broken")))
    }

    func testLegacyWhisperDownloadedStartsWithMLXEngine() {
        let decision = RecordingStartPlanner.resolve(
            selectedEngine: TranscriptionEngine.resolved(rawValue: "whisperKit"),
            mlxModelState: .downloaded
        )

        XCTAssertEqual(decision, .start(.mlxAudio))
    }



    func testMLXUnavailableMessageIncludesErrorDetail() {
        let message = RecordingStartBlockReason.mlxModelUnavailable(detail: "Model load failed: missing key").userMessage

        XCTAssertTrue(message.contains("missing key"))
    }

    func testDetailedUnavailableReminderUsesLongerDuration() {
        XCTAssertEqual(
            RecordingStartBlockReason.mlxModelUnavailable(detail: "broken").reminderDuration,
            4.2
        )
        XCTAssertEqual(
            RecordingStartBlockReason.mlxModelUnavailable(detail: nil).reminderDuration,
            2.4
        )
    }
}
