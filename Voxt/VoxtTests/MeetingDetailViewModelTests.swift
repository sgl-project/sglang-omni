import Combine
import XCTest
@testable import Voxt

@MainActor
final class MeetingDetailViewModelTests: MeetingDetailViewModelTestCase {
    func testHistoryViewModelAutoGeneratesSummaryOnlyOnce() async {
        let persisted = expectation(description: "summary persisted")
        var generateCount = 0

        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: UUID(),
            initialSummary: nil,
            initialSummaryChatMessages: [],
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
            segments: [
                MeetingTranscriptSegment(
                    speaker: .them,
                    startSeconds: 0,
                    endSeconds: 4,
                    text: "Let's finish the release checklist today."
                )
            ],
            audioURL: nil,
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: true, message: "Ready")
            },
            summaryGenerator: { _, settings in
                generateCount += 1
                return MeetingSummarySnapshot(
                    title: "Release Check",
                    body: "The team agreed to finish the release checklist today.",
                    todoItems: ["Finish release checklist"],
                    generatedAt: Date(),
                    settingsSnapshot: settings
                )
            },
            summaryPersistence: { _, _ in
                persisted.fulfill()
                return nil
            },
            summaryChatAnswerer: { _, _, _, _, _ in "" },
            summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, _ in nil }
        )

        viewModel.handleViewAppear()
        await fulfillment(of: [persisted], timeout: 1.0)
        viewModel.handleViewAppear()

        XCTAssertEqual(generateCount, 1)
        XCTAssertEqual(viewModel.summary?.title, "Release Check")
    }

    func testSummaryGenerationCannotReplaceSummaryAfterTranscriptMutation() async {
        let generationStarted = expectation(description: "summary generation started")
        var releaseGeneration: CheckedContinuation<MeetingSummarySnapshot, Never>?
        let existingSummary = MeetingSummarySnapshot(
            title: "Existing",
            body: "Saved summary",
            todoItems: [],
            generatedAt: Date(),
            settingsSnapshot: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Prompt",
                modelSelectionID: "custom-llm:test"
            )
        )
        let segment = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 2,
            text: "Original text"
        )

        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: UUID(),
            initialSummary: existingSummary,
            initialSummaryChatMessages: [],
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Prompt",
                modelSelectionID: "custom-llm:test"
            ),
            summaryModelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test", subtitle: "Local")
            ],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(
                    autoGenerate: false,
                    promptTemplate: "Prompt",
                    modelSelectionID: "custom-llm:test"
                )
            },
            summaryModelOptionsProvider: {
                [MeetingSummaryModelOption(id: "custom-llm:test", title: "Test", subtitle: "Local")]
            },
            segments: [segment],
            audioURL: nil,
            translationHandler: { text, _ in
                MeetingTranslationOperation(executionScope: .externalRequest) { text }
            },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: true, message: "Ready")
            },
            summaryGenerator: { _, settings in
                await withCheckedContinuation { continuation in
                    releaseGeneration = continuation
                    generationStarted.fulfill()
                }
                return MeetingSummarySnapshot(
                    title: "Generated",
                    body: "Generated from the old transcript",
                    todoItems: [],
                    generatedAt: Date(),
                    settingsSnapshot: settings
                )
            },
            summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, _, _, _ in "" },
            summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, _ in nil }
        )

        viewModel.regenerateSummary()
        await fulfillment(of: [generationStarted], timeout: 1.0)

        viewModel.beginEditingSegment(segment)
        viewModel.editingText = "Updated text"
        viewModel.saveEditingSegment()
        releaseGeneration?.resume(returning: existingSummary)
        try? await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(viewModel.summary?.title, "Existing")
        XCTAssertTrue(viewModel.isSummaryStale)
    }

    func testHistoryViewModelDoesNotAutoGenerateWhenSummaryAlreadyExists() async {
        var generateCount = 0

        let existing = MeetingSummarySnapshot(
            title: "Existing",
            body: "Saved summary",
            todoItems: [],
            generatedAt: Date(),
            settingsSnapshot: MeetingSummarySettingsSnapshot(
                autoGenerate: true,
                promptTemplate: "Default summary prompt",
                modelSelectionID: "custom-llm:test"
            )
        )

        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: UUID(),
            initialSummary: existing,
            initialSummaryChatMessages: [],
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
            segments: [
                MeetingTranscriptSegment(
                    speaker: .them,
                    startSeconds: 0,
                    endSeconds: 4,
                    text: "Already summarized."
                )
            ],
            audioURL: nil,
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: true, message: "Ready")
            },
            summaryGenerator: { _, settings in
                generateCount += 1
                return MeetingSummarySnapshot(
                    title: "New",
                    body: "Should not run",
                    todoItems: [],
                    generatedAt: Date(),
                    settingsSnapshot: settings
                )
            },
            summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, _, _, _ in "" },
            summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, _ in nil }
        )

        viewModel.handleViewAppear()
        try? await Task.sleep(for: .milliseconds(100))

        XCTAssertEqual(generateCount, 0)
        XCTAssertEqual(viewModel.summary?.title, "Existing")
    }

    func testHistoryViewModelSendsAndPersistsSummaryChatMessages() async {
        let persisted = expectation(description: "chat persisted")
        persisted.expectedFulfillmentCount = 2
        var answerInvocationCount = 0

        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: UUID(),
            initialSummary: MeetingSummarySnapshot(
                title: "Existing",
                body: "Saved summary",
                todoItems: ["Prepare release notes"],
                generatedAt: Date(),
                settingsSnapshot: MeetingSummarySettingsSnapshot(
                    autoGenerate: true,
                    promptTemplate: "Default summary prompt",
                    modelSelectionID: "custom-llm:test"
                )
            ),
            initialSummaryChatMessages: [],
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
            segments: [
                MeetingTranscriptSegment(
                    speaker: .them,
                    startSeconds: 0,
                    endSeconds: 4,
                    text: "Alex will finish the release notes."
                )
            ],
            audioURL: nil,
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: true, message: "Ready")
            },
            summaryGenerator: { _, settings in
                MeetingSummarySnapshot(
                    title: "Existing",
                    body: "Saved summary",
                    todoItems: ["Prepare release notes"],
                    generatedAt: Date(),
                    settingsSnapshot: settings
                )
            },
            summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, history, question, _ in
                answerInvocationCount += 1
                XCTAssertEqual(history.count, 1)
                XCTAssertEqual(history.first?.role, .user)
                XCTAssertEqual(question, "Who owns the release notes?")
                return "Alex owns the release notes."
            },
            summaryChatPersistence: { _, messages in
                persisted.fulfill()
                XCTAssertLessThanOrEqual(messages.count, 2)
                return nil
            },
            transcriptSegmentsPersistence: { _, _ in nil }
        )

        viewModel.summaryChatDraft = "Who owns the release notes?"
        viewModel.sendSummaryChat()
        await fulfillment(of: [persisted], timeout: 1.0)

        XCTAssertEqual(answerInvocationCount, 1)
        XCTAssertEqual(viewModel.summaryChatMessages.count, 2)
        XCTAssertEqual(viewModel.summaryChatMessages.first?.role, .user)
        XCTAssertEqual(viewModel.summaryChatMessages.last?.role, .assistant)
    }

    func testHistoryViewModelUsesResolvedInitialSummarySettings() {
        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: UUID(),
            initialSummary: nil,
            initialSummaryChatMessages: [],
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Focus on decisions and owners.",
                modelSelectionID: "remote-llm:openAI"
            ),
            summaryModelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local"),
                MeetingSummaryModelOption(id: "remote-llm:openAI", title: "OpenAI · gpt-5.4", subtitle: "Configured Remote LLM")
            ],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(
                    autoGenerate: false,
                    promptTemplate: "Focus on decisions and owners.",
                    modelSelectionID: "remote-llm:openAI"
                )
            },
            summaryModelOptionsProvider: {
                [
                    MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local"),
                    MeetingSummaryModelOption(id: "remote-llm:openAI", title: "OpenAI · gpt-5.4", subtitle: "Configured Remote LLM")
                ]
            },
            segments: [],
            audioURL: nil,
            translationHandler: { text, _ in MeetingTranslationOperation(executionScope: .externalRequest) { text } },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: true, message: "Ready")
            },
            summaryGenerator: { _, settings in
                MeetingSummarySnapshot(
                    title: "Existing",
                    body: "Saved summary",
                    todoItems: [],
                    generatedAt: Date(),
                    settingsSnapshot: settings
                )
            },
            summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, _, _, _ in "" },
            summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, _ in nil }
        )

        XCTAssertFalse(viewModel.summaryAutoGenerate)
        XCTAssertEqual(viewModel.summaryPromptTemplate, "Focus on decisions and owners.")
        XCTAssertEqual(viewModel.resolvedSummaryModelSelectionID, "remote-llm:openAI")
    }

    func testResetSummaryPromptTemplateRestoresDefaultPrompt() {
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: true,
                promptTemplate: "Custom prompt",
                modelSelectionID: "custom-llm:test"
            ),
            modelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")
            ]
        )

        viewModel.resetSummaryPromptTemplate()

        XCTAssertEqual(viewModel.summaryPromptTemplate, AppPromptDefaults.text(for: .transcriptSummary))
    }

    func testRefreshSummaryConfigurationFallsBackToFirstAvailableModel() {
        let viewModel = makeHistoryViewModel(
            initialSettings: MeetingSummarySettingsSnapshot(
                autoGenerate: true,
                promptTemplate: nil,
                modelSelectionID: "remote-llm:missing"
            ),
            modelOptions: [
                MeetingSummaryModelOption(id: "custom-llm:test", title: "Test Model", subtitle: "Local")
            ]
        )

        viewModel.refreshSummaryConfiguration(
            settings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "Refreshed prompt",
                modelSelectionID: "remote-llm:missing"
            ),
            modelOptions: [
                MeetingSummaryModelOption(id: "remote-llm:available", title: "Remote Model", subtitle: "Configured")
            ]
        )

        XCTAssertFalse(viewModel.summaryAutoGenerate)
        XCTAssertEqual(viewModel.summaryPromptTemplate, "Refreshed prompt")
        XCTAssertEqual(viewModel.resolvedSummaryModelSelectionID, "remote-llm:available")
    }

    func testSummarySettingsPersistThroughFeatureSettingsStore() throws {
        try withRestoredStandardDefaults([
            AppPreferenceKey.featureSettings
        ]) {
            var settings = FeatureSettingsStore.deriveFromLegacy(defaults: .standard)
            settings.meeting.summaryAutoGenerate = true
            settings.meeting.summaryPrompt = "Initial prompt"
            settings.meeting.summaryModelSelectionID = .localLLM("initial-model")
            FeatureSettingsStore.save(settings, defaults: .standard)

            let viewModel = makeHistoryViewModel(
                initialSettings: MeetingSummarySettingsSnapshot(
                    autoGenerate: true,
                    promptTemplate: "Initial prompt",
                    modelSelectionID: "custom-llm:initial-model"
                ),
                modelOptions: [
                    MeetingSummaryModelOption(id: "remote-llm:openAI", title: "OpenAI", subtitle: "Remote")
                ]
            )

            viewModel.setSummaryAutoGenerate(false)
            viewModel.setSummaryPromptTemplate("Persist this prompt")
            viewModel.setSummaryModelSelectionID("remote-llm:openAI")

            let reloaded = FeatureSettingsStore.load(defaults: .standard)
            XCTAssertFalse(reloaded.meeting.summaryAutoGenerate)
            XCTAssertEqual(reloaded.meeting.summaryPrompt, "Persist this prompt")
            XCTAssertEqual(reloaded.meeting.summaryModelSelectionID, .remoteLLM(.openAI))
        }
    }
}
