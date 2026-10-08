// RecordingOverlayState.swift
// Provides Recording Overlay State for window and overlay UI.

import AppKit
import SwiftUI
import Combine

enum OverlayDisplayMode: Equatable {
    case recording
    case processing
    case answer
}

enum OverlaySessionIconMode: Equatable {
    case transcription
    case note
    case translation
    case rewrite
}

enum AnswerInteractionMode: Equatable {
    case singleResult
    case conversation
}

enum AnswerSpaceShortcutAction: Equatable {
    case continueAndRecord
    case toggleConversationRecording
}

/// Observable state that drives the overlay UI. Either transcriber populates this.
@MainActor
class OverlayState: ObservableObject {
    @Published var isRecording = false
    @Published var isModelInitializing = false
    @Published var initializingEngine: TranscriptionEngine?
    @Published var audioLevel: Float = 0.0
    @Published var transcribedText = ""
    @Published var statusMessage = ""
    @Published var isEnhancing = false
    @Published var isRequesting = false
    @Published var isFinalizingTranscription = false
    @Published var isCompleting = false
    @Published var displayMode: OverlayDisplayMode = .recording
    @Published var sessionIconMode: OverlaySessionIconMode = .transcription
    @Published var answerTitle = ""
    @Published var answerContent = ""
    @Published var answerInteractionMode: AnswerInteractionMode = .singleResult
    @Published var rewriteConversationTurns: [RewriteConversationTurn] = []
    @Published var latestRewriteResult: RewriteAnswerPayload?
    @Published var latestHistoryEntryID: UUID?
    @Published var rewriteConversationRemoteResponseID: String?
    @Published var rewriteConversationRemoteContextKey: String?
    @Published var pendingConversationUserPrompt: String?
    @Published var pendingConversationSourceText: String?
    @Published var isRewriteConversationTurnInProgress = false
    @Published var isStreamingAnswer = false
    @Published var canInjectAnswer = false
    @Published var isPresented = false
    @Published var sessionTranslationTargetLanguage: TranslationTargetLanguage?
    @Published var sessionTranslationDraftLanguage: TranslationTargetLanguage?
    @Published var isSessionTranslationTargetPickerPresented = false
    @Published var isSessionTranslationLanguageHovering = false
    @Published var allowsSessionTranslationLanguageSwitching = false
    @Published var compactLeadingIconImage: NSImage?
    var answerTranslationSourceText = ""
    private(set) var rewriteConversationRemoteContextGeneration = UUID()

    private let notificationCenter: NotificationCenter
    private var remoteLLMProviderConfigurationsObserver: NSObjectProtocol?
    private var cancellables = Set<AnyCancellable>()
    private var latestSourceTranscribedText = ""
    private var transcribedTextTransformer: ((String) -> String)?

    init(notificationCenter: NotificationCenter = .default) {
        self.notificationCenter = notificationCenter
        remoteLLMProviderConfigurationsObserver = notificationCenter.addObserver(
            forName: .voxtRemoteLLMProviderConfigurationsDidChange,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.invalidateRewriteConversationRemoteContext()
            }
        }
    }

    deinit {
        if let remoteLLMProviderConfigurationsObserver {
            notificationCenter.removeObserver(remoteLLMProviderConfigurationsObserver)
        }
        // Keep an explicit deinit here. On macOS 26 test hosts, the synthesized
        // teardown path intermittently trips a malloc crash while destroying this
        // ObservableObject's @Published storage. An explicit deinit stabilizes
        // the generated destruction path without changing runtime behavior.
    }

    func invalidateRewriteConversationRemoteContext() {
        rewriteConversationRemoteResponseID = nil
        rewriteConversationRemoteContextKey = nil
        rewriteConversationRemoteContextGeneration = UUID()
    }

    @discardableResult
    func storeRewriteConversationRemoteContext(
        responseID: String,
        contextKey: String?,
        expectedGeneration: UUID
    ) -> Bool {
        guard rewriteConversationRemoteContextGeneration == expectedGeneration else {
            return false
        }
        rewriteConversationRemoteResponseID = responseID
        rewriteConversationRemoteContextKey = contextKey
        return true
    }

    func bind(to transcriber: SpeechTranscriber) {
        bind(
            isRecording: transcriber.$isRecording.eraseToAnyPublisher(),
            isModelInitializing: Just(false).eraseToAnyPublisher(),
            audioLevel: transcriber.$audioLevel.eraseToAnyPublisher(),
            transcribedText: transcriber.$transcribedText.eraseToAnyPublisher(),
            isEnhancing: transcriber.$isEnhancing.eraseToAnyPublisher(),
            isRequesting: Just(false).eraseToAnyPublisher(),
            isFinalizingTranscription: transcriber.$isFinalizingTranscription.eraseToAnyPublisher(),
            initializingEngine: .none
        )
    }

    func bind(to transcriber: MLXTranscriber) {
        bind(
            isRecording: transcriber.$isRecording.eraseToAnyPublisher(),
            isModelInitializing: transcriber.$isModelInitializing.eraseToAnyPublisher(),
            audioLevel: transcriber.$audioLevel.eraseToAnyPublisher(),
            transcribedText: transcriber.$transcribedText.eraseToAnyPublisher(),
            isEnhancing: transcriber.$isEnhancing.eraseToAnyPublisher(),
            isRequesting: Just(false).eraseToAnyPublisher(),
            isFinalizingTranscription: transcriber.$isFinalizingTranscription.eraseToAnyPublisher(),
            initializingEngine: .mlxAudio
        )
    }

    func bind(to transcriber: RemoteASRTranscriber) {
        bind(
            isRecording: transcriber.$isRecording.eraseToAnyPublisher(),
            isModelInitializing: Just(false).eraseToAnyPublisher(),
            audioLevel: transcriber.$audioLevel.eraseToAnyPublisher(),
            transcribedText: transcriber.$transcribedText.eraseToAnyPublisher(),
            isEnhancing: transcriber.$isEnhancing.eraseToAnyPublisher(),
            isRequesting: transcriber.$isRequesting.eraseToAnyPublisher(),
            isFinalizingTranscription: transcriber.$isFinalizingTranscription.eraseToAnyPublisher(),
            initializingEngine: .none
        )
    }

    func reset() {
        isRecording = false
        isModelInitializing = false
        initializingEngine = nil
        audioLevel = 0
        transcribedText = ""
        statusMessage = ""
        isEnhancing = false
        isRequesting = false
        isFinalizingTranscription = false
        isCompleting = false
        displayMode = .recording
        sessionIconMode = .transcription
        answerTitle = ""
        answerContent = ""
        answerInteractionMode = .singleResult
        rewriteConversationTurns = []
        latestRewriteResult = nil
        latestHistoryEntryID = nil
        invalidateRewriteConversationRemoteContext()
        pendingConversationUserPrompt = nil
        pendingConversationSourceText = nil
        isRewriteConversationTurnInProgress = false
        isStreamingAnswer = false
        canInjectAnswer = false
        isPresented = false
        sessionTranslationTargetLanguage = nil
        sessionTranslationDraftLanguage = nil
        isSessionTranslationTargetPickerPresented = false
        isSessionTranslationLanguageHovering = false
        allowsSessionTranslationLanguageSwitching = false
        compactLeadingIconImage = nil
        answerTranslationSourceText = ""
        latestSourceTranscribedText = ""
        transcribedTextTransformer = nil
        cancellables.removeAll()
    }

    func setTranscribedTextTransformer(_ transformer: ((String) -> String)?) {
        transcribedTextTransformer = transformer
        refreshDisplayedTranscribedText()
    }

    func refreshDisplayedTranscribedText() {
        transcribedText = transformedTranscribedText(from: latestSourceTranscribedText)
    }

    func clearDisplayedTranscribedText() {
        transcribedText = ""
    }

    func presentRecording(iconMode: OverlaySessionIconMode? = nil) {
        displayMode = .recording
        if let iconMode {
            sessionIconMode = iconMode
        }
        compactLeadingIconImage = nil
        answerTitle = ""
        answerContent = ""
        isFinalizingTranscription = false
        isStreamingAnswer = false
    }

    func presentProcessing(iconMode: OverlaySessionIconMode? = nil) {
        guard displayMode != .answer else { return }
        displayMode = .processing
        compactLeadingIconImage = nil
        dismissSessionTranslationTargetPicker()
        if let iconMode {
            sessionIconMode = iconMode
        }
    }

    private func bind(
        isRecording recordingPublisher: AnyPublisher<Bool, Never>,
        isModelInitializing modelInitializingPublisher: AnyPublisher<Bool, Never>,
        audioLevel audioLevelPublisher: AnyPublisher<Float, Never>,
        transcribedText transcribedTextPublisher: AnyPublisher<String, Never>,
        isEnhancing isEnhancingPublisher: AnyPublisher<Bool, Never>,
        isRequesting isRequestingPublisher: AnyPublisher<Bool, Never>,
        isFinalizingTranscription isFinalizingPublisher: AnyPublisher<Bool, Never>,
        initializingEngine: TranscriptionEngine?
    ) {
        cancellables.removeAll()
        audioLevel = 0
        isFinalizingTranscription = false
        self.initializingEngine = initializingEngine

        recordingPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] isRecording in
                guard let self else { return }
                self.isRecording = isRecording
                if !isRecording {
                    self.audioLevel = 0
                }
            }
            .store(in: &cancellables)

        modelInitializingPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] isInitializing in
                guard let self else { return }
                self.isModelInitializing = isInitializing
                self.initializingEngine = isInitializing ? initializingEngine : nil
                if isInitializing {
                    self.audioLevel = 0
                }
            }
            .store(in: &cancellables)

        audioLevelPublisher
            .combineLatest(recordingPublisher)
            .map { level, isRecording in
                guard isRecording else { return Float.zero }
                return Self.quantizedAudioLevel(level)
            }
            .removeDuplicates()
            .throttle(for: .milliseconds(50), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] level in
                self?.audioLevel = level
            }
            .store(in: &cancellables)

        transcribedTextPublisher
            .receive(on: RunLoop.main)
            .removeDuplicates()
            .throttle(for: .milliseconds(70), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] text in
                guard let self else { return }
                self.latestSourceTranscribedText = text
                self.transcribedText = self.transformedTranscribedText(from: text)
            }
            .store(in: &cancellables)

        isEnhancingPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] isEnhancing in
                self?.isEnhancing = isEnhancing
            }
            .store(in: &cancellables)

        isRequestingPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] isRequesting in
                self?.isRequesting = isRequesting
            }
            .store(in: &cancellables)

        isFinalizingPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] isFinalizing in
                self?.isFinalizingTranscription = isFinalizing
            }
            .store(in: &cancellables)
    }

    private static func quantizedAudioLevel(_ rawLevel: Float) -> Float {
        let clamped = max(0, min(rawLevel, 1))
        let steps: Float = 20
        return (clamped * steps).rounded() / steps
    }

    private func transformedTranscribedText(from sourceText: String) -> String {
        let transformed = transcribedTextTransformer?(sourceText) ?? sourceText
        return RecordingSessionSupport.textAfterSuppressingPromptEcho(transformed)
    }
}
