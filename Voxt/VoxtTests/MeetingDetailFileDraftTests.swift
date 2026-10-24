import AppKit
import SwiftUI
import XCTest
@testable import Voxt

@MainActor
final class MeetingDetailFileDraftTests: XCTestCase {
    func testDraftUsesTimelineSearchButCannotEditOrRunInference() async throws {
        let segments = [
            MeetingTranscriptSegment(speaker: .them, startSeconds: 2, endSeconds: 4, text: "first phrase"),
            MeetingTranscriptSegment(speaker: .them, startSeconds: 8, endSeconds: 9, text: "second phrase")
        ]
        let model = MeetingDetailViewModel(fileTaskTitle: "file.wav", segments: segments)
        model.handleViewAppear()
        XCTAssertEqual(model.mode, .fileDraft)
        XCTAssertNil(model.historyEntryID)
        XCTAssertNil(model.audioURL)
        XCTAssertFalse(model.canEditTranscript)
        XCTAssertFalse(model.canEditSpeakers)
        XCTAssertFalse(model.canRegenerateSummary)
        XCTAssertFalse(model.canSendSummaryChat)
        XCTAssertFalse(model.canExport)
        XCTAssertFalse(model.summaryAutoGenerate)
        XCTAssertEqual(model.availableTranscriptPresentationModes, [.timeline])
        model.beginEditingSegment(segments[0])
        model.deleteSegment(segments[0])
        XCTAssertNil(model.editingSegmentID)
        XCTAssertEqual(model.segments, segments)

        model.setTranslationEnabled(true)
        model.confirmTranslationLanguageSelection()
        model.setSummaryAutoGenerate(true)
        model.toggleSummaryCollapsed()
        XCTAssertFalse(model.translationEnabled)
        XCTAssertFalse(model.isTranslationLanguagePickerPresented)
        XCTAssertFalse(model.summaryAutoGenerate)
        XCTAssertTrue(model.isSummaryCollapsed)
        XCTAssertNil(model.summary)
        model.setSearchQuery("second")
        XCTAssertEqual(model.displayedSegments.map(\.id), [segments[1].id])
        XCTAssertEqual(model.segments[1].startSeconds, 8)
        XCTAssertEqual(model.segments[1].endSeconds, 9)
    }

    func testDraftWindowIsReusedAndPromotedInPlaceToHistory() throws {
        _ = NSApplication.shared
        let manager = MeetingDetailWindowManager()
        let taskID = UUID()
        let title = "draft-\(taskID)"
        let segments = [MeetingTranscriptSegment(speaker: .them, startSeconds: 1, endSeconds: 3, text: "draft")]
        manager.presentFileTranscript(taskID: taskID, title: title, segments: segments)
        let window = try XCTUnwrap(NSApp.windows.first {
            ($0.contentViewController as? NSHostingController<MeetingDetailWindowView>)?.rootView.viewModel.title == title
        })
        defer { window.close() }
        let frame = window.frame
        let originalHost = window.contentViewController
        manager.presentFileTranscript(taskID: taskID, title: title, segments: segments)
        XCTAssertTrue(manager.hasFileTranscriptWindow(taskID: taskID))
        XCTAssertTrue(window.contentViewController === originalHost)
        XCTAssertEqual(NSApp.windows.filter {
            ($0.contentViewController as? NSHostingController<MeetingDetailWindowView>)?.rootView.viewModel.title == title
        }.count, 1)
        let entry = makeEntry()
        let settings = MeetingSummarySettingsSnapshot(autoGenerate: false, promptTemplate: "", modelSelectionID: nil)
        manager.presentHistoryMeeting(
            entry: entry, replacingFileTaskID: taskID, activate: false, audioURL: nil,
            initialSummarySettings: settings, summaryModelOptionsProvider: { [] },
            summarySettingsProvider: { settings }, translationHandler: { _, _ in .cancelled() },
            summaryStatusProvider: { _ in MeetingSummaryProviderStatus(isAvailable: false, message: "") },
            summaryGenerator: { _, _ in throw CancellationError() }, summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, _, _, _ in "" }, summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, _ in nil }
        )
        XCTAssertFalse(manager.hasFileTranscriptWindow(taskID: taskID))
        XCTAssertEqual(window.frame, frame)
        let updated = try XCTUnwrap((window.contentViewController as? NSHostingController<MeetingDetailWindowView>)?.rootView.viewModel)
        XCTAssertEqual(updated.mode, .history)
        XCTAssertEqual(updated.historyEntryID, entry.id)
        XCTAssertEqual(updated.segments, entry.transcriptSegments)
        manager.closeFileTranscript(taskID: taskID)
        XCTAssertTrue(window.isVisible, "Removing the task must not close its promoted history window")
    }

    func testClosingDraftDoesNotLeaveAStaleWindowRegistration() {
        _ = NSApplication.shared
        let manager = MeetingDetailWindowManager()
        let id = UUID()
        manager.presentFileTranscript(taskID: id, title: "draft", segments: [])
        manager.closeFileTranscript(taskID: id)
        XCTAssertFalse(manager.hasFileTranscriptWindow(taskID: id))
    }

    private func makeEntry() -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: UUID(), text: "final", createdAt: Date(), transcriptionEngine: "test",
            transcriptionModel: "test", enhancementMode: "off", enhancementModel: "",
            kind: .transcript, isTranslation: false, audioDurationSeconds: 4,
            transcriptionProcessingDurationSeconds: nil, llmDurationSeconds: nil,
            focusedAppName: nil, focusedAppBundleID: nil, matchedGroupID: nil, matchedGroupName: nil,
            matchedAppGroupName: nil, matchedURLGroupName: nil, remoteASRProvider: nil,
            remoteASRModel: nil, remoteASREndpoint: nil, remoteLLMProvider: nil,
            remoteLLMModel: nil, remoteLLMEndpoint: nil, whisperWordTimings: nil,
            transcriptSegments: [MeetingTranscriptSegment(speaker: .them, speakerID: "speaker-1", startSeconds: 1, endSeconds: 3, text: "final")],
            meetingCaptureMode: .meeting,
            dictionaryHitTerms: [], dictionaryCorrectedTerms: [], dictionarySuggestedTerms: []
        )
    }
}
