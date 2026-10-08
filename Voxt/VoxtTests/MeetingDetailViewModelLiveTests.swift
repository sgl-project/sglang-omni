import Combine
import XCTest
@testable import Voxt

@MainActor
final class MeetingDetailViewModelLiveTests: MeetingDetailViewModelTestCase {
    func testLiveViewModelTracksFinalizingState() async {
        let liveState = MeetingOverlayState()
        liveState.isPresented = true
        liveState.isRecording = true
        liveState.segments = [
            MeetingTranscriptSegment(
                speaker: .them,
                startSeconds: 0,
                endSeconds: 1,
                text: "Live transcript"
            )
        ]

        let viewModel = MeetingDetailViewModel(
            liveState: liveState,
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: true,
                promptTemplate: "Default summary prompt",
                modelSelectionID: "custom-llm:test"
            ),
            summaryModelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")
            ],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(
                    autoGenerate: true,
                    promptTemplate: "Default summary prompt",
                    modelSelectionID: "custom-llm:test"
                )
            },
            summaryModelOptionsProvider: {
                [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")]
            },
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } }
        )
        let localeIdentifiers = ["en", "zh-Hans", "ja"]
        let inProgressSubtitles = localeIdentifiers.map { localeIdentifier in
            String(
                format: "%@ · %@",
                AppLocalization.localizedString("Meeting", localeIdentifier: localeIdentifier),
                AppLocalization.localizedString("Meeting In Progress", localeIdentifier: localeIdentifier)
            )
        }

        XCTAssertFalse(viewModel.isFinalizing)
        XCTAssertTrue(inProgressSubtitles.contains(viewModel.subtitle))

        liveState.isRecording = false
        liveState.isFinalizing = true
        try? await Task.sleep(for: .milliseconds(50))

        XCTAssertTrue(viewModel.isFinalizing)
        let finalizingSubtitles = localeIdentifiers.map {
            AppLocalization.localizedString("Preparing final meeting details", localeIdentifier: $0)
        }
        XCTAssertTrue(finalizingSubtitles.contains(viewModel.subtitle))
    }

    func testLiveViewModelSplitsLongDisplaySegments() {
        let segments = [
            MeetingTranscriptSegment(
                speaker: .them,
                startSeconds: 0,
                endSeconds: 24,
                text: "第一句介绍当前模型选择和采集链路。第二句继续说明发言人识别的误差来源，以及短暂停顿不应该直接生成新的发言人。第三句讨论会议详情应该保持连续语义，同时实时浮层需要更快地给出可读段落和清晰反馈。第四句补充说明如果文本继续变长，实时显示仍然需要在自然句边界拆开，避免一整块内容压在同一个气泡里。"
            )
        ]

        let displaySegments = MeetingDetailViewModel.liveDisplaySegments(from: segments)

        XCTAssertGreaterThan(displaySegments.count, 1)
        XCTAssertTrue(displaySegments.allSatisfy { $0.text.count <= 130 })
        XCTAssertEqual(
            displaySegments.map(\.text).joined(),
            segments[0].text
        )
    }

    func testLiveViewModelPreservesSpeakerMetadataWhenUpdatingSegments() async {
        let segmentID = UUID()
        let liveState = MeetingOverlayState()
        liveState.isPresented = true
        liveState.isRecording = true
        liveState.captureMode = .meeting
        liveState.segments = [
            MeetingTranscriptSegment(
                id: segmentID,
                speaker: .them,
                speakerID: "sortformer-0",
                speakerDisplayName: "Speaker 1",
                audioSource: .systemAudio,
                speakerConfidence: 0.71,
                startSeconds: 0,
                endSeconds: 2,
                text: "Initial text"
            )
        ]

        let viewModel = MeetingDetailViewModel(
            liveState: liveState,
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: true,
                promptTemplate: "Default summary prompt",
                modelSelectionID: "custom-llm:test"
            ),
            summaryModelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")
            ],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(
                    autoGenerate: true,
                    promptTemplate: "Default summary prompt",
                    modelSelectionID: "custom-llm:test"
                )
            },
            summaryModelOptionsProvider: {
                [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")]
            },
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } }
        )

        liveState.segments = [
            MeetingTranscriptSegment(
                id: segmentID,
                speaker: .them,
                speakerID: "sortformer-1",
                speakerDisplayName: "Speaker 2",
                audioSource: .systemAudio,
                speakerConfidence: 0.82,
                startSeconds: 0,
                endSeconds: 2.5,
                text: "Updated text"
            )
        ]
        try? await Task.sleep(for: .milliseconds(50))

        XCTAssertEqual(viewModel.segments.first?.speakerID, "sortformer-1")
        XCTAssertEqual(viewModel.segments.first?.speakerDisplayName, "Speaker 2")
        XCTAssertEqual(viewModel.segments.first?.audioSource, .systemAudio)
        XCTAssertEqual(viewModel.segments.first?.speakerConfidence ?? -1, 0.82, accuracy: 0.001)
        XCTAssertEqual(viewModel.segments.first?.displaySpeakerTitle, "Speaker 2")
    }
}
