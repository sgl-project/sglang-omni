import Combine
import XCTest
@testable import Voxt

@MainActor
final class MeetingDetailViewModelTranslationTests: MeetingDetailViewModelTestCase {
    func testFailedDetailTranslationDoesNotRetryIndefinitely() async {
        var invocationCount = 0
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 1,
            text: "Translate once"
        )
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: nil,
                modelSelectionID: "custom-llm:test"
            ),
            modelOptions: [],
            segments: [segment],
            translationHandler: { _, _ in
                MeetingTranslationOperation(executionScope: .externalRequest) {
                    invocationCount += 1
                    throw MeetingDetailTranslationTestError.failed
                }
            }
        )

        viewModel.translationDraftLanguageRaw = TranslationTargetLanguage.english.rawValue
        viewModel.confirmTranslationLanguageSelection()
        try? await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(invocationCount, 1)
        XCTAssertFalse(viewModel.segments[0].isTranslationPending)
    }

    func testEmptyDetailTranslationDoesNotRetryIndefinitely() async {
        var invocationCount = 0
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: nil,
                modelSelectionID: "custom-llm:test"
            ),
            modelOptions: [],
            segments: [MeetingTranscriptSegment(
                speaker: .them,
                startSeconds: 0,
                endSeconds: 1,
                text: "Empty result"
            )],
            translationHandler: { _, _ in
                MeetingTranslationOperation(executionScope: .externalRequest) {
                    invocationCount += 1
                    return "   "
                }
            }
        )

        viewModel.translationDraftLanguageRaw = TranslationTargetLanguage.english.rawValue
        viewModel.confirmTranslationLanguageSelection()
        try? await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(invocationCount, 1)
        XCTAssertFalse(viewModel.segments[0].isTranslationPending)
    }

    func testUpdatedSegmentTextIsTranslatedAfterActiveRevisionCompletes() async {
        let segmentID = UUID()
        let gate = MeetingDetailTranslationGate()
        var translatedSources: [String] = []
        let liveState = MeetingOverlayState()
        liveState.isPresented = true
        liveState.isRecording = true
        liveState.segments = [MeetingTranscriptSegment(
            id: segmentID,
            speaker: .them,
            startSeconds: 0,
            endSeconds: 1,
            text: "old text"
        )]
        let viewModel = MeetingDetailViewModel(
            liveState: liveState,
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: nil,
                modelSelectionID: "custom-llm:test"
            ),
            summaryModelOptions: [],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(autoGenerate: false, promptTemplate: nil, modelSelectionID: "custom-llm:test")
            },
            summaryModelOptionsProvider: { [] },
            translationHandler: { source, _ in
                MeetingTranslationOperation(executionScope: .externalRequest) {
                    translatedSources.append(source)
                    if source == "old text" { await gate.wait() }
                    return "translated: \(source)"
                }
            }
        )

        viewModel.translationDraftLanguageRaw = TranslationTargetLanguage.english.rawValue
        viewModel.confirmTranslationLanguageSelection()
        await gate.waitUntilStarted()
        viewModel.updateLiveSegments([MeetingTranscriptSegment(
            id: segmentID,
            speaker: .them,
            startSeconds: 0,
            endSeconds: 2,
            text: "new text"
        )])
        await gate.open()
        try? await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(translatedSources, ["old text", "new text"])
        XCTAssertEqual(viewModel.segments.first?.translatedText, "translated: new text")
        XCTAssertFalse(viewModel.segments.first?.isTranslationPending ?? true)
        XCTAssertEqual(viewModel.displayedSegments, viewModel.segments)
        XCTAssertEqual(viewModel.speakerGroups.first?.segments, viewModel.segments)
        XCTAssertEqual(viewModel.speakerGroups.first?.wordCount, 2)
    }

    func testTranslationCompletionRefreshesSearchMembership() async {
        let viewModel = makeHistoryViewModel(
            initialSettings: .init(autoGenerate: false, promptTemplate: nil, modelSelectionID: "custom-llm:test"),
            modelOptions: [],
            segments: [.init(speaker: .them, startSeconds: 0, endSeconds: 2, text: "original words")],
            translationHandler: { _, _ in
                MeetingTranslationOperation(executionScope: .externalRequest) { "translated result" }
            }
        )
        viewModel.setSearchQuery("translated")
        XCTAssertTrue(viewModel.displayedSegments.isEmpty)
        let published = expectation(description: "translated text matches search")
        let subscription = viewModel.$displayedSegments
            .first(where: { $0.first?.translatedText == "translated result" })
            .sink { _ in published.fulfill() }
        defer { subscription.cancel() }
        viewModel.translationDraftLanguageRaw = TranslationTargetLanguage.english.rawValue
        viewModel.confirmTranslationLanguageSelection()
        await fulfillment(of: [published], timeout: 1)
        XCTAssertEqual(viewModel.speakerGroups.first?.segments.first?.translatedText, "translated result")
    }
}

private enum MeetingDetailTranslationTestError: Error {
    case failed
}

private actor MeetingDetailTranslationGate {
    private var started = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        started = true
        await withCheckedContinuation { continuation = $0 }
    }

    func waitUntilStarted() async {
        while !started { await Task.yield() }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}
