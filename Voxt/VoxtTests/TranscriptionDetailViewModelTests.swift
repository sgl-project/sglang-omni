// TranscriptionDetailViewModelTests.swift
// Provides Transcription Detail View Model Tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class TranscriptionDetailViewModelTests: XCTestCase {
    func testManualCorrectionAvailabilityRequiresTranscriptionAndHandler() {
        let transcriptionViewModel = makeViewModel(
            entry: makeEntry(kind: .normal),
            manualCorrectionHandler: { entry, _ in entry }
        )
        XCTAssertTrue(transcriptionViewModel.canShowManualCorrection)

        let missingHandlerViewModel = makeViewModel(
            entry: makeEntry(kind: .normal),
            manualCorrectionHandler: nil
        )
        XCTAssertFalse(missingHandlerViewModel.canShowManualCorrection)

        let translationViewModel = makeViewModel(
            entry: makeEntry(kind: .translation),
            manualCorrectionHandler: { entry, _ in entry }
        )
        XCTAssertFalse(translationViewModel.canShowManualCorrection)
    }

    func testBeginAndCancelManualCorrectionToggleEditingState() {
        let entry = makeEntry(kind: .normal, text: "Cloud Code")
        let viewModel = makeViewModel(
            entry: entry,
            manualCorrectionHandler: { entry, _ in entry }
        )

        viewModel.beginManualCorrection()
        XCTAssertTrue(viewModel.isEditingCorrection)
        XCTAssertEqual(viewModel.correctionDraft, "Cloud Code")

        viewModel.correctionDraft = "Claude Code"
        viewModel.cancelManualCorrection()
        XCTAssertFalse(viewModel.isEditingCorrection)
        XCTAssertEqual(viewModel.correctionDraft, "Cloud Code")
    }

    func testSubmitManualCorrectionRefreshesEntryAndEndsEditing() async {
        let originalEntry = makeEntry(kind: .normal, text: "Cloud Code")
        let updatedEntry = makeEntry(
            id: originalEntry.id,
            kind: .normal,
            text: "Claude Code",
            correctedTerms: ["Claude Code"],
            correctionSnapshots: [
                DictionaryCorrectionSnapshot(
                    originalText: "Cloud Code",
                    correctedText: "Claude Code",
                    finalLocation: 0,
                    finalLength: 11
                )
            ]
        )
        let viewModel = makeViewModel(
            entry: originalEntry,
            manualCorrectionHandler: { _, _ in updatedEntry }
        )

        viewModel.beginManualCorrection()
        viewModel.correctionDraft = "Claude Code"
        viewModel.submitManualCorrection()

        for _ in 0..<20 where viewModel.isSubmittingCorrection {
            try? await Task.sleep(for: .milliseconds(20))
        }

        XCTAssertFalse(viewModel.isEditingCorrection)
        XCTAssertFalse(viewModel.isSubmittingCorrection)
        XCTAssertEqual(viewModel.entry.text, "Claude Code")
        XCTAssertEqual(viewModel.entry.dictionaryCorrectedTerms, ["Claude Code"])
        XCTAssertEqual(viewModel.visibleCorrectionText, "Claude Code")
    }

    func testSubmitManualCorrectionWithoutChangesShowsToast() {
        let originalEntry = makeEntry(kind: .normal, text: "Cloud Code")
        var didInvokeHandler = false
        let viewModel = makeViewModel(
            entry: originalEntry,
            manualCorrectionHandler: { entry, _ in
                didInvokeHandler = true
                return entry
            }
        )

        viewModel.beginManualCorrection()
        viewModel.submitManualCorrection()

        XCTAssertFalse(didInvokeHandler)
        XCTAssertTrue(viewModel.isEditingCorrection)
        XCTAssertEqual(
            viewModel.toastMessage,
            AppLocalization.localizedString("Please modify the text before correcting.")
        )
    }
}

private extension TranscriptionDetailViewModelTests {
    func makeViewModel(
        entry: TranscriptionHistoryEntry,
        manualCorrectionHandler: TranscriptionDetailViewModel.ManualCorrectionHandler?
    ) -> TranscriptionDetailViewModel {
        TranscriptionDetailViewModel(
            entry: entry,
            audioURL: nil,
            followUpStatusProvider: { _ in
                TranscriptionFollowUpProviderStatus(isAvailable: true, message: "")
            },
            followUpAnswerer: { _, _, _ in "" },
            followUpPersistence: { _, _ in nil },
            manualCorrectionHandler: manualCorrectionHandler
        )
    }

    func makeEntry(
        id: UUID = UUID(),
        kind: TranscriptionHistoryKind,
        text: String = "Transcript",
        correctedTerms: [String] = [],
        correctionSnapshots: [DictionaryCorrectionSnapshot] = []
    ) -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: id,
            text: text,
            createdAt: Date(timeIntervalSinceReferenceDate: 10),
            transcriptionEngine: "MLX Whisper",
            transcriptionModel: "whisper-large-v3-turbo",
            enhancementMode: "Off",
            enhancementModel: "None",
            kind: kind,
            isTranslation: kind == .translation,
            audioDurationSeconds: nil,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: correctedTerms,
            dictionaryCorrectionSnapshots: correctionSnapshots,
            dictionarySuggestedTerms: []
        )
    }
}
