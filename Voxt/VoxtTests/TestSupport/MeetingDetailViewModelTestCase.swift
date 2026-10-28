import Combine
import XCTest
@testable import Voxt

@MainActor
class MeetingDetailViewModelTestCase: XCTestCase {
    func makeHistoryViewModel(
        initialSettings: MeetingSummarySettingsSnapshot,
        modelOptions: [MeetingSummaryModelOption],
        captureMode: MeetingCaptureMode? = nil,
        segments: [MeetingTranscriptSegment] = [],
        translationHandler: @escaping MeetingDetailWindowManager.TranslationHandler = { text, _ in
            MeetingTranslationOperation(executionScope: .externalRequest) { text }
        },
        transcriptSegmentsPersistence: @escaping MeetingDetailWindowManager.TranscriptSegmentsPersistence = { _, _ in nil }
    ) -> MeetingDetailViewModel {
        MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: UUID(),
            initialSummary: nil,
            initialSummaryChatMessages: [],
            initialSummarySettings: initialSettings,
            summaryModelOptions: modelOptions,
            summarySettingsProvider: { initialSettings },
            summaryModelOptionsProvider: { modelOptions },
            segments: segments,
            captureMode: captureMode,
            audioURL: nil,
            translationHandler: translationHandler,
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
            transcriptSegmentsPersistence: transcriptSegmentsPersistence
        )
    }

    func withRestoredStandardDefaults(
        _ keys: [String],
        _ body: () throws -> Void
    ) rethrows {
        let defaults = UserDefaults.standard
        let savedValues = keys.map { key in
            (key, defaults.object(forKey: key))
        }
        defer {
            for (key, value) in savedValues {
                if let value {
                    defaults.set(value, forKey: key)
                } else {
                    defaults.removeObject(forKey: key)
                }
            }
        }
        try body()
    }
}
