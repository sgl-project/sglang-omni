import Combine
import XCTest
@testable import Voxt

@MainActor
final class MeetingDetailViewModelTranscriptTests: MeetingDetailViewModelTestCase {
    func testHistoryMeetingModeEnablesSpeakerPresentationEvenWithSingleSystemSpeaker() {
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: nil,
                modelSelectionID: "custom-llm:test"
            ),
            modelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")
            ],
            captureMode: .meeting,
            segments: [
                MeetingTranscriptSegment(
                    speaker: .them,
                    audioSource: .systemAudio,
                    startSeconds: 0,
                    endSeconds: 3,
                    text: "A single realtime system speaker should still be treated as meeting mode."
                )
            ]
        )

        XCTAssertEqual(viewModel.captureMode, .meeting)
        XCTAssertTrue(viewModel.showsSpeakerDisplayModePicker)
        XCTAssertTrue(viewModel.availableTranscriptPresentationModes.contains(.speakerMarks))
    }

    func testRenameSpeakerUpdatesMatchingSegmentsAndPersists() {
        let entryID = UUID()
        var persistedSegments: [MeetingTranscriptSegment]?
        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: entryID,
            initialSummary: nil,
            initialSummaryChatMessages: [],
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Default summary prompt",
                modelSelectionID: "custom-llm:test"
            ),
            summaryModelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")
            ],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(
                    autoGenerate: false,
                    promptTemplate: "Default summary prompt",
                    modelSelectionID: "custom-llm:test"
                )
            },
            summaryModelOptionsProvider: {
                [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")]
            },
            segments: [
                MeetingTranscriptSegment(
                    speaker: .them,
                    speakerID: "S1",
                    speakerDisplayName: "Speaker 1",
                    audioSource: .systemAudio,
                    startSeconds: 0,
                    endSeconds: 1,
                    text: "hello"
                ),
                MeetingTranscriptSegment(
                    speaker: .them,
                    speakerID: "S2",
                    speakerDisplayName: "Speaker 2",
                    audioSource: .systemAudio,
                    startSeconds: 1,
                    endSeconds: 2,
                    text: "world"
                ),
                MeetingTranscriptSegment(
                    speaker: .me,
                    audioSource: .microphone,
                    startSeconds: 2,
                    endSeconds: 3,
                    text: "ack"
                )
            ],
            audioURL: nil,
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: true, message: "Ready")
            },
            summaryGenerator: { _, settings in
                MeetingSummarySnapshot(
                    title: "Generated",
                    body: "Body",
                    todoItems: [],
                    generatedAt: Date(),
                    settingsSnapshot: settings
                )
            },
            summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, _, _, _ in "" },
            summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, segments in
                persistedSegments = segments
                return nil
            }
        )

        viewModel.renameSpeaker(identityKey: "systemAudio:S1", displayName: "Alice")

        XCTAssertEqual(viewModel.segments[0].speakerDisplayName, "Alice")
        XCTAssertEqual(viewModel.segments[1].speakerDisplayName, "Speaker 2")
        XCTAssertEqual(persistedSegments?.first?.speakerDisplayName, "Alice")
        XCTAssertTrue(viewModel.isSummaryStale)
    }

    func testRenameSpeakerAcceptsLegacyDisplayIdentityKey() {
        let entryID = UUID()
        var persistedSegments: [MeetingTranscriptSegment]?
        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: entryID,
            initialSummary: nil,
            initialSummaryChatMessages: [],
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Default summary prompt",
                modelSelectionID: "custom-llm:test"
            ),
            summaryModelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")
            ],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(
                    autoGenerate: false,
                    promptTemplate: "Default summary prompt",
                    modelSelectionID: "custom-llm:test"
                )
            },
            summaryModelOptionsProvider: {
                [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")]
            },
            segments: [
                MeetingTranscriptSegment(
                    speaker: .them,
                    speakerID: "S1",
                    speakerDisplayName: "Speaker 1",
                    audioSource: .systemAudio,
                    startSeconds: 0,
                    endSeconds: 1,
                    text: "hello"
                ),
                MeetingTranscriptSegment(
                    speaker: .them,
                    speakerID: "S2",
                    speakerDisplayName: "Speaker 2",
                    audioSource: .systemAudio,
                    startSeconds: 1,
                    endSeconds: 2,
                    text: "world"
                ),
                MeetingTranscriptSegment(
                    speaker: .me,
                    audioSource: .microphone,
                    startSeconds: 2,
                    endSeconds: 3,
                    text: "ack"
                )
            ],
            audioURL: nil,
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: true, message: "Ready")
            },
            summaryGenerator: { _, settings in
                MeetingSummarySnapshot(
                    title: "Generated",
                    body: "Body",
                    todoItems: [],
                    generatedAt: Date(),
                    settingsSnapshot: settings
                )
            },
            summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, _, _, _ in "" },
            summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, segments in
                persistedSegments = segments
                return nil
            }
        )

        viewModel.renameSpeaker(identityKey: "display:Speaker 1", displayName: "Alice")

        XCTAssertEqual(viewModel.segments[0].speakerDisplayName, "Alice")
        XCTAssertEqual(viewModel.segments[1].speakerDisplayName, "Speaker 2")
        XCTAssertEqual(persistedSegments?.first?.speakerDisplayName, "Alice")
    }

    func testManualTranscriptEditClearsTranslationAndMarksSummaryStale() {
        let segment = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 2,
            text: "old text",
            translatedText: "old translation"
        )
        var persisted: [MeetingTranscriptSegment] = []
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Prompt",
                modelSelectionID: "custom-llm:test"
            ),
            modelOptions: [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test", subtitle: "Local")],
            segments: [segment],
            transcriptSegmentsPersistence: { _, segments in
                persisted = segments
                return nil
            }
        )

        viewModel.beginEditingSegment(segment)
        viewModel.editingText = "  updated text  "
        viewModel.saveEditingSegment()

        XCTAssertEqual(viewModel.segments.first?.text, "updated text")
        XCTAssertNil(viewModel.segments.first?.translatedText)
        XCTAssertTrue(viewModel.isSummaryStale)
        XCTAssertEqual(persisted.first?.text, "updated text")
        XCTAssertNil(persisted.first?.translatedText)
        XCTAssertNil(viewModel.editingSegmentID)
    }

    func testDeleteAndUndoRestoresSegmentAtOriginalPosition() {
        let first = MeetingTranscriptSegment(speaker: .me, startSeconds: 0, endSeconds: 1, text: "first")
        let second = MeetingTranscriptSegment(speaker: .them, startSeconds: 1, endSeconds: 2, text: "second")
        var persisted: [[MeetingTranscriptSegment]] = []
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Prompt",
                modelSelectionID: "custom-llm:test"
            ),
            modelOptions: [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test", subtitle: "Local")],
            segments: [first, second],
            transcriptSegmentsPersistence: { _, segments in
                persisted.append(segments)
                return nil
            }
        )

        viewModel.deleteSegment(first)
        XCTAssertEqual(viewModel.segments.map(\.id), [second.id])
        XCTAssertTrue(viewModel.isUndoDeleteAvailable)

        viewModel.undoDelete()
        XCTAssertEqual(viewModel.segments.map(\.id), [first.id, second.id])
        XCTAssertFalse(viewModel.isUndoDeleteAvailable)
        XCTAssertEqual(persisted.count, 2)
    }

    func testHighlightTogglePersistsOnTranscriptSegment() {
        let segment = MeetingTranscriptSegment(speaker: .them, startSeconds: 2, endSeconds: 4, text: "important")
        var persisted: MeetingTranscriptSegment?
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Prompt",
                modelSelectionID: "custom-llm:test"
            ),
            modelOptions: [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test", subtitle: "Local")],
            segments: [segment],
            transcriptSegmentsPersistence: { _, segments in
                persisted = segments.first
                return nil
            }
        )

        viewModel.toggleHighlight(for: segment)

        XCTAssertTrue(viewModel.segments.first?.isHighlighted == true)
        XCTAssertTrue(persisted?.isHighlighted == true)
    }
}
