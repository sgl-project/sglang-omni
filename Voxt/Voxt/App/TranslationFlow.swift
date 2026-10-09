// TranslationFlow.swift
// Provides Translation Flow for app lifecycle and routing.

import Foundation
import AppKit

extension AppDelegate {
    // MARK: - Translation Flow
    // Keeps translation/enhancement orchestration isolated from recording lifecycle.

    private func textTransformFailureMessage(
        for error: Error,
        fallbackMessage: String
    ) -> String {
        guard let localizedError = error as? LocalizedError else {
            return fallbackMessage
        }

        let description = localizedError.errorDescription?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let suggestion = localizedError.recoverySuggestion?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        switch (description.isEmpty, suggestion.isEmpty) {
        case (false, false):
            return "\(description) \(suggestion)"
        case (false, true):
            return description
        case (true, false):
            return suggestion
        case (true, true):
            return fallbackMessage
        }
    }

    private func failCurrentTextTransformSession(
        _ message: String,
        finishAfter delay: TimeInterval = 2.8
    ) {
        showOverlayStatus(message, clearAfter: delay)
        finishSession(after: delay)
    }

    private func failCurrentRewriteConversationTurn(
        _ message: String,
        clearAfter delay: TimeInterval = 2.8
    ) {
        guard overlayState.restoreLatestCompletedRewriteConversation(status: message) else {
            failCurrentTextTransformSession(message, finishAfter: delay)
            return
        }

        finishSession(after: 0)
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard let self,
                  self.overlayState.isRewriteConversationActive,
                  self.overlayState.statusMessage == message
            else { return }
            self.overlayState.statusMessage = ""
        }
    }

    func runTranslationPreview(_ text: String) async throws -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let previousSelectedTextTranslationFlow = isSelectedTextTranslationFlow
        isSelectedTextTranslationFlow = false
        defer {
            isSelectedTextTranslationFlow = previousSelectedTextTranslationFlow
        }

        return try await runTranslationPipeline(
            text: trimmed,
            targetLanguage: translationTargetLanguage,
            allowStrictRetry: true
        )
    }

    func runRewritePreview(dictatedPrompt: String, sourceText: String) async throws -> String {
        let trimmedPrompt = dictatedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedSource = sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty || !trimmedSource.isEmpty else { return "" }

        return try await runRewritePipeline(
            dictatedText: trimmedPrompt,
            selectedSourceText: trimmedSource,
            conversationHistory: [],
            structuredAnswerOutput: trimmedSource.isEmpty
        )
    }

    func processTranslatedTranscription(_ text: String, sessionID: UUID) {
        guard shouldHandleCallbacks(for: sessionID) else { return }
        let targetLanguage = effectiveSessionTranslationTargetLanguage
        let resolution = resolvedTranslationProviderResolution(
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: false
        )
        VoxtLog.translation(
            "Translation flow started. inputChars=\(text.count), targetLanguage=\(targetLanguage.instructionName), translationModelProvider=\(translationModelProvider.rawValue), resolvedProvider=\(resolution.provider.rawValue)"
        )
        setEnhancingState(true)
        overlayState.transcribedText = text
        _Concurrency.Task<Void, Never> {
            defer {
                self.setEnhancingState(false)
            }

            let llmStartedAt = Date()
            do {
                let translated = try await self.runTranslationPipeline(
                    text: text,
                    targetLanguage: targetLanguage,
                    allowStrictRetry: false
                )
                guard self.shouldHandleCallbacks(for: sessionID) else { return }
                let llmDuration = Date().timeIntervalSince(llmStartedAt)
                if self.looksUntranslated(source: text, result: translated) {
                    VoxtLog.translationWarning("Translation output may be untranslated. sourceChars=\(text.count), outputChars=\(translated.count)")
                }
                VoxtLog.translation("Translation flow succeeded. outputChars=\(translated.count), llmDurationSec=\(String(format: "%.3f", llmDuration))")
                self.commitTranscription(translated, llmDurationSeconds: llmDuration) { [weak self] in
                    guard let self, self.shouldHandleCallbacks(for: sessionID) else { return }
                    self.finishSession(after: 0)
                }
            } catch {
                guard self.shouldHandleCallbacks(for: sessionID) else { return }
                VoxtLog.translationWarning("Translation flow failed without committing raw text: \(error)")
                self.failCurrentTextTransformSession(
                    self.textTransformFailureMessage(
                        for: error,
                        fallbackMessage: AppLocalization.localizedString("Translation failed. Try again after checking the selected model.")
                    )
                )
            }
        }
    }

    func beginSelectedTextTranslationIfPossible() -> Bool {
        guard !isSessionActive else { return false }
        guard let selectedText = selectedContentTextFromSystemSelection() else {
            return false
        }

        let sourceWindowNumber = NSApp.isActive ? NSApp.keyWindow?.windowNumber : nil
        pendingSessionFinishTask?.cancel()
        pendingSessionFinishTask = nil
        silenceMonitorTask?.cancel()
        silenceMonitorTask = nil
        pauseLLMTask?.cancel()
        pauseLLMTask = nil
        releaseResidualRecordingResources(reason: "selected-text-translation-begin")
        resetSessionTranslationState()
        overlayState.reset()
        let frontmostApplication = NSWorkspace.shared.frontmostApplication
        let frontmostBundleID = frontmostApplication?.bundleIdentifier
        let ownBundleID = Bundle.main.bundleIdentifier
        if let frontmostBundleID,
           frontmostBundleID != ownBundleID {
            sessionTargetApplicationBundleID = frontmostBundleID
        } else {
            sessionTargetApplicationBundleID = nil
        }
        sessionTargetApplicationPID = sessionTargetApplicationBundleID == nil ? nil : frontmostApplication?.processIdentifier
        selectedTextTranslationHadWritableFocusedInput = hasWritableFocusedTextInput()
        overlayState.transcribedText = selectedText
        overlayState.setAnswerTranslationSourceText(selectedText)
        overlayState.statusMessage = ""
        overlayState.presentRecording(iconMode: .translation)
        overlayWindow.show(state: overlayState, position: overlayPosition)

        let startedAt = Date()
        isSessionActive = true
        isSelectedTextTranslationFlow = true
        recordingLifecycle.begin()
        invalidateActiveLLMRequest()
        sessionOutputMode = .translation
        recordingRequestedAt = startedAt
        recordingStartedAt = startedAt
        recordingStoppedAt = startedAt
        transcriptionProcessingStartedAt = nil
        transcriptionResultReceivedAt = nil
        sessionFinalOutputDeliveredAt = nil
        sessionLLMExecutionTimings = []
        enhancementContextSnapshot = nil
        lastEnhancementPromptContext = nil
        sessionOutputDestinationContext = nil

        if interactionSoundsEnabled {
            interactionSoundPlayer.playStartAsync()
        }

        OnboardingSessionEvent.started(
            id: activeRecordingSessionID,
            kind: .selectedTextTranslation,
            windowNumber: sourceWindowNumber
        ).post()
        VoxtLog.translation("Selected text translation started. inputChars=\(selectedText.count)")
        prewarmSelectedTextTranslationLLMIfNeeded(targetLanguage: effectiveSessionTranslationTargetLanguage)
        processSelectedTextTranslation(selectedText)
        return true
    }

    func processRewriteTranscription(_ text: String, sessionID: UUID) {
        guard shouldHandleCallbacks(for: sessionID) else { return }
        let isConversationContinuation = overlayState.isRewriteConversationActive
        let selectedSourceText: String
        let conversationHistory: [RewriteConversationPromptTurn]
        let prefersStructuredAnswerOutput: Bool
        let previousConversationResponseID: String?
        let conversationResponseContextKey = currentRewriteResponsesConversationContextKey

        if isConversationContinuation {
            selectedSourceText = ""
            conversationHistory = overlayState.rewriteConversationPromptHistory
            rewriteSessionHasSelectedSourceText = false
            prefersStructuredAnswerOutput = false
            if overlayState.rewriteConversationRemoteContextKey == conversationResponseContextKey {
                previousConversationResponseID = overlayState.rewriteConversationRemoteResponseID
            } else {
                previousConversationResponseID = nil
                overlayState.invalidateRewriteConversationRemoteContext()
            }
            overlayState.stageConversationUserPrompt(text)
        } else {
            selectedSourceText = rewriteSessionSelectedSourceText
            rewriteSessionHasSelectedSourceText = !selectedSourceText.isEmpty
            conversationHistory = []
            prefersStructuredAnswerOutput = shouldUseStructuredRewriteAnswerOutput(
                hasSelectedSourceText: rewriteSessionHasSelectedSourceText
            )
            previousConversationResponseID = nil
            overlayState.stageConversationUserPrompt(text, sourceText: selectedSourceText)
        }
        let conversationResponseContextGeneration = overlayState.rewriteConversationRemoteContextGeneration
        VoxtLog.translation(
            "Rewrite flow started. promptChars=\(text.count), selectedSourceChars=\(selectedSourceText.count), rewriteModelProvider=\(rewriteModelProvider.rawValue), structuredAnswerOutput=\(prefersStructuredAnswerOutput), conversationHistoryTurns=\(conversationHistory.count)"
        )
        setEnhancingState(true)
        let requestID = beginLLMRequest()
        let progressHandler: (@Sendable (String) -> Void)? = if isConversationContinuation {
            { [weak self] partialOutput in
                Task { @MainActor [weak self] in
                    guard let self,
                          self.shouldHandleCallbacks(for: sessionID),
                          self.isCurrentLLMRequest(requestID)
                    else { return }
                    self.presentRewriteConversationStreamingPreview(content: partialOutput)
                }
            }
        } else {
            nil
        }

        runTrackedLLMRequest(requestID) {
            defer {
                if self.isCurrentLLMRequest(requestID) {
                    self.setEnhancingState(false)
                }
            }

            let llmStartedAt = Date()
            var latestConversationResponseID: String?
            do {
                var rewritten: String
                let rewriteResult = try await self.runRewritePipeline(
                    dictatedText: text,
                    selectedSourceText: selectedSourceText,
                    conversationHistory: conversationHistory,
                    structuredAnswerOutput: prefersStructuredAnswerOutput,
                    previousConversationResponseID: previousConversationResponseID,
                    onProgress: progressHandler,
                    onResponseID: { responseID in
                        guard self.isCurrentLLMRequest(requestID) else { return }
                        latestConversationResponseID = responseID
                    }
                )
                rewritten = rewriteResult
                if prefersStructuredAnswerOutput,
                   self.shouldRetryStructuredRewriteAnswer(for: rewritten, dictatedPrompt: text) {
                    VoxtLog.translationWarning("Rewrite structured answer was missing usable content; retrying in direct-answer mode.")
                    if let retried = try? await self.rewriteText(
                        dictatedPrompt: text,
                        sourceText: selectedSourceText,
                        conversationHistory: conversationHistory,
                        structuredAnswerOutput: true,
                        forceNonEmptyAnswer: true,
                        previousConversationResponseID: previousConversationResponseID,
                        onProgress: nil,
                        onResponseID: { responseID in
                            guard self.isCurrentLLMRequest(requestID) else { return }
                            latestConversationResponseID = responseID
                        }
                    ) {
                        rewritten = retried
                    }
                }
                if !prefersStructuredAnswerOutput {
                    let normalized = RewriteAnswerContentNormalizer.normalizePlainTextAnswer(rewritten)
                    if normalized != rewritten.trimmingCharacters(in: .whitespacesAndNewlines) {
                        VoxtLog.translationWarning(
                            """
                            Rewrite plain-text answer normalized before delivery.
                            [raw]
                            \(VoxtLog.llmPreview(rewritten))
                            [normalized]
                            \(VoxtLog.llmPreview(normalized))
                            """
                        )
                    }
                    rewritten = normalized
                }
                if prefersStructuredAnswerOutput,
                   self.shouldRetryStructuredRewriteAnswer(for: rewritten, dictatedPrompt: text) {
                    VoxtLog.translationWarning("Rewrite structured answer still unusable after retry; failing without committing fallback text.")
                    throw TextTransformFailure.rewriteRejectedByGuard(reason: "Structured rewrite answer was empty or unusable after retry.")
                }
                let shouldRetryPlainAnswer = !prefersStructuredAnswerOutput && (
                    RewriteAnswerContentNormalizer.isUnusablePlainTextAnswer(rewritten, dictatedPrompt: text) ||
                    RewriteAnswerContentNormalizer.repeatsLatestAssistantAnswer(
                        rewritten,
                        dictatedPrompt: text,
                        conversationHistory: conversationHistory
                    )
                )
                if shouldRetryPlainAnswer {
                    VoxtLog.translationWarning("Rewrite plain-text answer was empty, unusable, or repeated the latest assistant reply; retrying with corrective guidance.")
                    if let retried = try? await self.rewriteText(
                        dictatedPrompt: text,
                        sourceText: selectedSourceText,
                        conversationHistory: conversationHistory,
                        structuredAnswerOutput: false,
                        forceNonEmptyAnswer: true,
                        previousConversationResponseID: previousConversationResponseID,
                        onProgress: progressHandler,
                        onResponseID: { responseID in
                            guard self.isCurrentLLMRequest(requestID) else { return }
                            latestConversationResponseID = responseID
                        }
                    ) {
                        rewritten = RewriteAnswerContentNormalizer.normalizePlainTextAnswer(retried)
                    }
                }
                let plainAnswerStillRejected = !prefersStructuredAnswerOutput && (
                    RewriteAnswerContentNormalizer.isUnusablePlainTextAnswer(rewritten, dictatedPrompt: text) ||
                    RewriteAnswerContentNormalizer.repeatsLatestAssistantAnswer(
                        rewritten,
                        dictatedPrompt: text,
                        conversationHistory: conversationHistory
                    )
                )
                if plainAnswerStillRejected {
                    VoxtLog.translationWarning("Rewrite plain-text answer remained unusable or repeated after retry; failing without committing fallback text.")
                    throw TextTransformFailure.rewriteRejectedByGuard(reason: "Plain rewrite answer was empty, matched the prompt, or repeated the latest assistant reply after retry.")
                }
                guard self.shouldHandleCallbacks(for: sessionID), self.isCurrentLLMRequest(requestID) else { return }
                if let latestConversationResponseID {
                    let stored = self.overlayState.storeRewriteConversationRemoteContext(
                        responseID: latestConversationResponseID,
                        contextKey: conversationResponseContextKey,
                        expectedGeneration: conversationResponseContextGeneration
                    )
                    if !stored {
                        VoxtLog.translation(
                            "Discarded stale rewrite response context after remote provider configuration changed."
                        )
                    }
                }
                if isConversationContinuation,
                   !self.overlayState.isStreamingAnswer,
                   !rewritten.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    await self.presentRewriteConversationPseudoStreamingPreview(
                        content: rewritten,
                        sessionID: sessionID
                    )
                }
                let llmDuration = Date().timeIntervalSince(llmStartedAt)
                VoxtLog.translation("Rewrite flow succeeded. outputChars=\(rewritten.count), llmDurationSec=\(String(format: "%.3f", llmDuration))")
                self.commitTranscription(rewritten, llmDurationSeconds: llmDuration) { [weak self] in
                    guard let self,
                          self.shouldHandleCallbacks(for: sessionID),
                          self.isCurrentLLMRequest(requestID)
                    else { return }
                    self.finishSession(after: 0)
                }
            } catch {
                guard self.shouldHandleCallbacks(for: sessionID), self.isCurrentLLMRequest(requestID) else { return }
                VoxtLog.translationWarning("Rewrite flow failed without committing fallback text: \(error)")
                let message = self.textTransformFailureMessage(
                    for: error,
                    fallbackMessage: AppLocalization.localizedString("Rewrite failed. Try again after checking the selected model.")
                )
                if isConversationContinuation {
                    self.failCurrentRewriteConversationTurn(message)
                } else {
                    self.failCurrentTextTransformSession(message)
                }
            }
        }
    }

    private func processSelectedTextTranslation(_ text: String) {
        setEnhancingState(true)
        let requestID = beginLLMRequest()
        runTrackedLLMRequest(requestID) {
            defer {
                if self.isCurrentLLMRequest(requestID) {
                    self.setEnhancingState(false)
                }
            }

            let llmStartedAt = Date()
            do {
                let translated = try await self.runTranslationPipeline(
                    text: text,
                    targetLanguage: self.translationTargetLanguage,
                    allowStrictRetry: true
                )
                guard self.isCurrentLLMRequest(requestID) else { return }
                let llmDuration = Date().timeIntervalSince(llmStartedAt)
                if self.looksUntranslated(source: text, result: translated) {
                    VoxtLog.translationWarning("Selected text translation output may be untranslated. inputChars=\(text.count), outputChars=\(translated.count)")
                }
                VoxtLog.translation("Selected text translation succeeded. outputChars=\(translated.count), llmDurationSec=\(String(format: "%.3f", llmDuration))")
                self.overlayState.transcribedText = translated
                self.commitTranscription(translated, llmDurationSeconds: llmDuration) { [weak self] in
                    guard let self, self.isCurrentLLMRequest(requestID) else { return }
                    self.finishSession(after: 0)
                }
            } catch {
                guard self.isCurrentLLMRequest(requestID) else { return }
                VoxtLog.translationWarning("Selected text translation failed without committing original selected text: \(error)")
                self.failCurrentTextTransformSession(
                    self.textTransformFailureMessage(
                        for: error,
                        fallbackMessage: AppLocalization.localizedString("Translation failed. Try again after checking the selected model.")
                    )
                )
            }
        }
    }

}
