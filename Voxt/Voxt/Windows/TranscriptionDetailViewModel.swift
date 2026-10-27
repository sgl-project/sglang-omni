// TranscriptionDetailViewModel.swift
// Provides Transcription Detail View Model for window and overlay UI.

import Foundation
import Combine

@MainActor
final class TranscriptionDetailViewModel: ObservableObject {
    typealias FollowUpStatusProvider = @MainActor (TranscriptionHistoryEntry) -> TranscriptionFollowUpProviderStatus
    typealias FollowUpAnswerer = @MainActor (TranscriptionHistoryEntry, [TranscriptSummaryChatMessage], String) async throws -> String
    typealias FollowUpPersistence = @MainActor (UUID, [TranscriptSummaryChatMessage]) -> TranscriptionHistoryEntry?
    typealias ManualCorrectionHandler = @MainActor (TranscriptionHistoryEntry, String) async throws -> TranscriptionHistoryEntry?

    @Published private(set) var entry: TranscriptionHistoryEntry
    @Published private(set) var audioURL: URL?
    @Published private(set) var chatMessages: [TranscriptSummaryChatMessage]
    @Published private(set) var providerStatus: TranscriptionFollowUpProviderStatus
    @Published private(set) var isLoading = false
    @Published private(set) var errorMessage: String?
    @Published var draft = ""
    @Published private(set) var isEditingCorrection = false
    @Published var correctionDraft = ""
    @Published private(set) var isSubmittingCorrection = false
    @Published private(set) var correctionErrorMessage: String?
    @Published private(set) var toastMessage = ""

    private let followUpStatusProvider: FollowUpStatusProvider
    private let followUpAnswerer: FollowUpAnswerer
    private let followUpPersistence: FollowUpPersistence
    private let manualCorrectionHandler: ManualCorrectionHandler?

    private var sendTask: Task<Void, Never>?
    private var manualCorrectionTask: Task<Void, Never>?
    private var toastDismissTask: Task<Void, Never>?

    init(
        entry: TranscriptionHistoryEntry,
        audioURL: URL?,
        followUpStatusProvider: @escaping FollowUpStatusProvider,
        followUpAnswerer: @escaping FollowUpAnswerer,
        followUpPersistence: @escaping FollowUpPersistence,
        manualCorrectionHandler: ManualCorrectionHandler? = nil
    ) {
        self.entry = entry
        self.audioURL = audioURL
        self.chatMessages = entry.transcriptionChatMessages ?? []
        self.followUpStatusProvider = followUpStatusProvider
        self.followUpAnswerer = followUpAnswerer
        self.followUpPersistence = followUpPersistence
        self.manualCorrectionHandler = manualCorrectionHandler
        self.providerStatus = followUpStatusProvider(entry)
        self.correctionDraft = HistoryCorrectionPresentation.correctedText(
            for: entry.text,
            snapshots: entry.dictionaryCorrectionSnapshots
        )
    }

    deinit {
        sendTask?.cancel()
        manualCorrectionTask?.cancel()
        toastDismissTask?.cancel()
    }

    var title: String {
        let trimmedTitle = entry.displayTitle?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmedTitle.isEmpty {
            return trimmedTitle
        }
        return TranscriptionDetailSupport.title(for: entry.kind)
    }

    var createdAtText: String {
        entry.createdAt.formatted(date: .abbreviated, time: .shortened)
    }

    var headerMetaText: String {
        "\(createdAtText) · \(modelText)"
    }

    var modelText: String {
        let enhancementModel = entry.enhancementModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if !enhancementModel.isEmpty, enhancementModel != "None" {
            return enhancementModel
        }

        let transcriptionModel = entry.transcriptionModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if !transcriptionModel.isEmpty {
            return transcriptionModel
        }

        return entry.transcriptionEngine
    }

    var canSend: Bool {
        providerStatus.isAvailable &&
        !isLoading &&
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canShowManualCorrection: Bool {
        entry.kind == .normal && manualCorrectionHandler != nil
    }

    var canConfirmManualCorrection: Bool {
        canShowManualCorrection &&
        !isSubmittingCorrection &&
        !trimmedCorrectionDraft.isEmpty
    }

    var visibleCorrectionText: String {
        HistoryCorrectionPresentation.correctedText(
            for: entry.text,
            snapshots: entry.dictionaryCorrectionSnapshots
        )
    }

    var displayMessages: [TranscriptSummaryChatMessage] {
        if TranscriptionHistoryConversationSupport.isCompleteConversation(chatMessages) {
            return chatMessages
        }
        return [seedAssistantMessage] + chatMessages
    }

    func refresh(entry: TranscriptionHistoryEntry) {
        self.entry = entry
        self.chatMessages = entry.transcriptionChatMessages ?? []
        self.providerStatus = followUpStatusProvider(entry)
        if !isEditingCorrection {
            correctionDraft = visibleCorrectionText
        }
    }

    func refresh(entry: TranscriptionHistoryEntry, audioURL: URL?) {
        self.entry = entry
        self.audioURL = audioURL
        self.chatMessages = entry.transcriptionChatMessages ?? []
        self.providerStatus = followUpStatusProvider(entry)
        if !isEditingCorrection {
            correctionDraft = visibleCorrectionText
        }
    }

    func refreshProviderStatus() {
        providerStatus = followUpStatusProvider(entry)
    }

    func send() {
        let question = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }
        guard providerStatus.isAvailable else {
            errorMessage = providerStatus.message
            return
        }

        draft = ""
        errorMessage = nil

        let userMessage = TranscriptSummaryChatMessage(role: .user, content: question)
        chatMessages.append(userMessage)
        _ = followUpPersistence(entry.id, chatMessages)
        isLoading = true

        let entrySnapshot = entry
        let historySnapshot = chatMessages

        sendTask?.cancel()
        sendTask = Task { [weak self] in
            guard let self else { return }
            do {
                let answer = try await followUpAnswerer(
                    entrySnapshot,
                    historySnapshot,
                    question
                )
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    let assistantMessage = TranscriptSummaryChatMessage(role: .assistant, content: answer)
                    self.chatMessages.append(assistantMessage)
                    self.isLoading = false
                    self.errorMessage = nil
                    _ = self.followUpPersistence(entrySnapshot.id, self.chatMessages)
                }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.isLoading = false
                    self.errorMessage = error.localizedDescription
                    _ = self.followUpPersistence(entrySnapshot.id, self.chatMessages)
                }
            }
        }
    }

    func beginManualCorrection() {
        guard canShowManualCorrection else { return }
        correctionErrorMessage = nil
        dismissToast()
        correctionDraft = visibleCorrectionText
        isEditingCorrection = true
    }

    func cancelManualCorrection() {
        manualCorrectionTask?.cancel()
        isSubmittingCorrection = false
        correctionErrorMessage = nil
        dismissToast()
        correctionDraft = visibleCorrectionText
        isEditingCorrection = false
    }

    func submitManualCorrection() {
        guard let manualCorrectionHandler, canConfirmManualCorrection else { return }
        guard trimmedCorrectionDraft != visibleCorrectionText else {
            showToast(AppLocalization.localizedString("Please modify the text before correcting."))
            return
        }

        let entrySnapshot = entry
        let correctedText = trimmedCorrectionDraft
        correctionErrorMessage = nil
        dismissToast()
        isSubmittingCorrection = true

        manualCorrectionTask?.cancel()
        manualCorrectionTask = Task { [weak self] in
            guard let self else { return }
            do {
                let updatedEntry = try await manualCorrectionHandler(entrySnapshot, correctedText)
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    if let updatedEntry {
                        self.refresh(entry: updatedEntry, audioURL: self.audioURL)
                    }
                    self.isSubmittingCorrection = false
                    self.isEditingCorrection = false
                    self.correctionErrorMessage = nil
                    self.correctionDraft = self.visibleCorrectionText
                }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.isSubmittingCorrection = false
                    self.correctionErrorMessage = error.localizedDescription
                }
            }
        }
    }

    private var trimmedCorrectionDraft: String {
        correctionDraft.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func dismissToast() {
        toastDismissTask?.cancel()
        toastMessage = ""
    }

    private func showToast(_ message: String, duration: TimeInterval = 2.2) {
        toastDismissTask?.cancel()
        toastMessage = message
        toastDismissTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
            guard !Task.isCancelled else { return }
            self.toastMessage = ""
        }
    }

    private var seedAssistantMessage: TranscriptSummaryChatMessage {
        TranscriptionHistoryConversationSupport.seedMessage(for: entry)
    }
}
