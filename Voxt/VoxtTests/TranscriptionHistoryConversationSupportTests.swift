// TranscriptionHistoryConversationSupportTests.swift
// Provides Transcription History Conversation Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class TranscriptionHistoryConversationSupportTests: XCTestCase {
    func testSupportsDetailOnlyForRewriteHistory() {
        XCTAssertFalse(TranscriptionHistoryConversationSupport.supportsDetail(for: .normal))
        XCTAssertFalse(TranscriptionHistoryConversationSupport.supportsDetail(for: .transcript))
        XCTAssertFalse(TranscriptionHistoryConversationSupport.supportsDetail(for: .translation))
        XCTAssertTrue(TranscriptionHistoryConversationSupport.supportsDetail(for: .rewrite))
    }

    func testShouldContinueConversationExpiresAfterTimeout() {
        let now = Date(timeIntervalSinceReferenceDate: 1_000)
        let recent = now.addingTimeInterval(-30)
        let expired = now.addingTimeInterval(-(TranscriptionHistoryConversationSupport.continuationTimeout + 1))

        XCTAssertTrue(
            TranscriptionHistoryConversationSupport.shouldContinueConversation(
                activeEntryID: UUID(),
                lastUpdatedAt: recent,
                now: now
            )
        )
        XCTAssertFalse(
            TranscriptionHistoryConversationSupport.shouldContinueConversation(
                activeEntryID: UUID(),
                lastUpdatedAt: expired,
                now: now
            )
        )
    }

    func testInitialChatMessagesSeedsAssistantTranscript() {
        let createdAt = Date(timeIntervalSinceReferenceDate: 123)
        let messages = TranscriptionHistoryConversationSupport.initialChatMessages(
            forTranscript: "First turn",
            createdAt: createdAt
        )

        XCTAssertEqual(messages?.count, 1)
        XCTAssertEqual(messages?.first?.role, .assistant)
        XCTAssertEqual(messages?.first?.content, "First turn")
        XCTAssertEqual(messages?.first?.createdAt, createdAt)
    }

    func testMergedChatMessagesBootstrapsSeedBeforeExistingFollowUpHistory() {
        let entry = makeEntry(
            text: "Seed transcript",
            createdAt: Date(timeIntervalSinceReferenceDate: 100),
            transcriptionChatMessages: [
                TranscriptSummaryChatMessage(role: .user, content: "What happened next?")
            ]
        )

        let merged = TranscriptionHistoryConversationSupport.mergedChatMessages(
            for: entry,
            appendingTranscript: "Second transcript turn",
            createdAt: Date(timeIntervalSinceReferenceDate: 200)
        )

        XCTAssertEqual(merged.count, 3)
        XCTAssertEqual(merged[0].role, .assistant)
        XCTAssertEqual(merged[0].content, "Seed transcript")
        XCTAssertEqual(merged[1].role, .user)
        XCTAssertEqual(merged[2].role, .assistant)
        XCTAssertEqual(merged[2].content, "Second transcript turn")
    }

    func testRewriteConversationMessagesFlattensTurnsIntoUserAssistantSequence() {
        let messages = TranscriptionHistoryConversationSupport.rewriteConversationMessages(
            from: [
                RewriteConversationTurn(
                    userPromptText: "Initial request",
                    resultTitle: "A",
                    resultContent: "First answer"
                ),
                RewriteConversationTurn(
                    userPromptText: "Follow up",
                    resultTitle: "B",
                    resultContent: "Second answer"
                )
            ],
            createdAt: Date(timeIntervalSinceReferenceDate: 300)
        )

        XCTAssertEqual(messages.map(\.role), [.user, .assistant, .user, .assistant])
        XCTAssertEqual(messages.map(\.content), ["Initial request", "First answer", "Follow up", "Second answer"])
    }

    @MainActor
    func testDisplayMessagesUsesAssistantFirstConversationWithoutDuplicatingSeed() {
        let entry = makeEntry(
            text: "A",
            transcriptionChatMessages: [
                TranscriptSummaryChatMessage(role: .assistant, content: "A"),
                TranscriptSummaryChatMessage(role: .assistant, content: "B")
            ]
        )
        let viewModel = TranscriptionDetailViewModel(
            entry: entry,
            audioURL: nil,
            followUpStatusProvider: { _ in
                TranscriptionFollowUpProviderStatus(isAvailable: true, message: "")
            },
            followUpAnswerer: { _, _, _ in "" },
            followUpPersistence: { _, _ in nil }
        )

        XCTAssertEqual(viewModel.displayMessages.count, 2)
        XCTAssertEqual(viewModel.displayMessages.map(\.content), ["A", "B"])
    }

    @MainActor
    func testDisplayMessagesPrependsSeedWhenHistoryStartsWithUserFollowUp() {
        let entry = makeEntry(
            text: "Seed transcript",
            transcriptionChatMessages: [
                TranscriptSummaryChatMessage(role: .user, content: "Summarize this")
            ]
        )
        let viewModel = TranscriptionDetailViewModel(
            entry: entry,
            audioURL: nil,
            followUpStatusProvider: { _ in
                TranscriptionFollowUpProviderStatus(isAvailable: true, message: "")
            },
            followUpAnswerer: { _, _, _ in "" },
            followUpPersistence: { _, _ in nil }
        )

        XCTAssertEqual(viewModel.displayMessages.count, 2)
        XCTAssertEqual(viewModel.displayMessages[0].role, .assistant)
        XCTAssertEqual(viewModel.displayMessages[0].content, "Seed transcript")
        XCTAssertEqual(viewModel.displayMessages[1].role, .user)
    }

    @MainActor
    func testDisplayMessagesKeepsCompleteUserFirstRewriteConversation() {
        let entry = makeEntry(
            text: "First answer",
            transcriptionChatMessages: [
                TranscriptSummaryChatMessage(role: .user, content: "Initial request"),
                TranscriptSummaryChatMessage(role: .assistant, content: "First answer")
            ]
        )
        let viewModel = TranscriptionDetailViewModel(
            entry: entry,
            audioURL: nil,
            followUpStatusProvider: { _ in
                TranscriptionFollowUpProviderStatus(isAvailable: true, message: "")
            },
            followUpAnswerer: { _, _, _ in "" },
            followUpPersistence: { _, _ in nil }
        )

        XCTAssertEqual(viewModel.displayMessages.map(\.role), [.user, .assistant])
        XCTAssertEqual(viewModel.displayMessages.map(\.content), ["Initial request", "First answer"])
    }

    private func makeEntry(
        text: String,
        createdAt: Date = Date(timeIntervalSinceReferenceDate: 42),
        transcriptionChatMessages: [TranscriptSummaryChatMessage]? = nil
    ) -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: UUID(),
            text: text,
            createdAt: createdAt,
            transcriptionEngine: "MLX Whisper",
            transcriptionModel: "whisper-large-v3-turbo",
            enhancementMode: "Off",
            enhancementModel: "None",
            kind: .normal,
            isTranslation: false,
            audioDurationSeconds: 3,
            transcriptionProcessingDurationSeconds: 1,
            llmDurationSeconds: nil,
            focusedAppName: "Notes",
            focusedAppBundleID: "com.apple.Notes",
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
            transcriptSegments: nil,
            transcriptAudioRelativePath: nil,
            transcriptSummary: nil,
            transcriptSummaryChatMessages: nil,
            displayTitle: nil,
            transcriptionChatMessages: transcriptionChatMessages,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }
}
