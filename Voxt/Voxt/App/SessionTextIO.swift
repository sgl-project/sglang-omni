// SessionTextIO.swift
// Provides Session Text IO for app lifecycle and routing.

import Foundation
import AppKit
import ApplicationServices

extension AppDelegate {
    enum AnswerOverlayInjectionMode: Sendable {
        case standard
        case selectedTextTranslation
    }

    private enum SessionOutputDelivery {
        case typeText
        case answerOverlay
        case selectedTextTranslationResultWindow
    }

    // MARK: - Session Text I/O
    // Keeps clipboard/AX/paste simulation logic isolated from recording orchestration.

    func commitTranscription(
        _ text: String,
        llmDurationSeconds: TimeInterval?,
        onDeliveryCompleted: (() -> Void)? = nil
    ) {
        let sessionID = activeRecordingSessionID
        guard recordingLifecycle.claimOutput(for: sessionID) else {
            VoxtLog.input("Skipping duplicate or cancelled commit for current session output.")
            return
        }
        let sessionOutputMode = sessionOutputMode
        let userMainLanguage = userMainLanguage
        let callbackDecision = Self.sessionCallbackHandlingDecision(
            requestedSessionID: sessionID,
            activeSessionID: activeRecordingSessionID,
            isSessionCancellationRequested: isSessionCancellationRequested
        )
        guard callbackDecision == .accept else {
            VoxtLog.inputWarning(
                """
                Commit transcription abandoned after session invalidation. reason=\(callbackDecision.logDescription), sessionID=\(sessionID.uuidString), activeSessionID=\(activeRecordingSessionID.uuidString), outputMode=\(RecordingSessionSupport.outputLabel(for: sessionOutputMode)), chars=\(text.count), stopped=\(recordingStoppedAt != nil)
                """
            )
            return
        }

        VoxtLog.input("Commit transcription entered. characters=\(text.count)", verbose: true)

        // Prepare once, then persist this exact snapshot after delivery completes.
        // Recomputing from mutable session state could record a different result.
        let context = Self.preparedDeliveryContext(
            originalText: text,
            llmDurationSeconds: llmDurationSeconds,
            sessionOutputMode: sessionOutputMode,
            userMainLanguage: userMainLanguage,
            matcher: dictionaryStore.makeMatcherIfEnabled(for: text, activeGroupID: activeDictionaryGroupID()),
            usesConservativeEvidence: shouldUseConservativeDictionaryEvidenceForCurrentSession(),
            automaticReplacementEnabled: UserDefaults.standard.bool(
                forKey: AppPreferenceKey.dictionaryHighConfidenceCorrectionEnabled
            )
        )
        cacheLatestInjectableOutputText(context.outputText)
        logUnicodeReplacementCharactersIfNeeded(
            stage: "commit",
            inputText: text,
            outputText: context.outputText,
            outputMode: sessionOutputMode
        )

        VoxtLog.input(
            "Commit transcription prepared payload. inputChars=\(text.count), outputChars=\(context.outputText.count), hasRewritePayload=\(context.rewriteAnswerPayload != nil), dictionaryMatches=\(context.dictionaryMatches.count), dictionaryCorrections=\(context.dictionaryCorrectedTerms.count)",
            verbose: true
        )

        deliverCommittedOutput(context) { [weak self] didInject, didTriggerAutoKeyPress, outputDestinationContext in
            guard let self, self.recordingLifecycle.accepts(sessionID), !self.isApplicationTerminating else { return }
            self.finalizeCommittedOutputPostDelivery(
                deliveredContext: context,
                outputMode: sessionOutputMode,
                didInject: didInject,
                didTriggerAutoKeyPress: didTriggerAutoKeyPress,
                outputDestinationContext: outputDestinationContext
            )
            onDeliveryCompleted?()
        }
    }

    private func logUnicodeReplacementCharactersIfNeeded(
        stage: String,
        inputText: String,
        outputText: String,
        outputMode: SessionOutputMode
    ) {
        let marker = Character("\u{FFFD}")
        let inputCount = inputText.reduce(into: 0) { count, character in
            if character == marker { count += 1 }
        }
        let outputCount = outputText.reduce(into: 0) { count, character in
            if character == marker { count += 1 }
        }
        guard inputCount > 0 || outputCount > 0 else { return }

        VoxtLog.llmDebug(
            "Unicode replacement characters detected before delivery. stage=\(stage), outputMode=\(RecordingSessionSupport.outputLabel(for: outputMode)), inputReplacementChars=\(inputCount), outputReplacementChars=\(outputCount), inputChars=\(inputText.count), outputChars=\(outputText.count)"
        )
    }

    private func beginOverlayOutputDelivery() {
        overlayState.isRequesting = true
        overlayState.isCompleting = false
        if overlayState.displayMode != .answer {
            overlayState.displayMode = .processing
        }
    }

    private func endOverlayOutputDelivery() {
        overlayState.isRequesting = false
    }

    private func deliverCommittedOutput(
        _ context: SessionFinalizeContext,
        completion: ((Bool, Bool, OutputDestinationContext?) -> Void)? = nil
    ) {
        let sessionID = activeRecordingSessionID
        let delivery = resolvedOutputDelivery(for: context)
        let deliveryLabel: String
        switch delivery {
        case .typeText:
            deliveryLabel = "typeText"
        case .answerOverlay:
            deliveryLabel = "answerOverlay"
        case .selectedTextTranslationResultWindow:
            deliveryLabel = "selectedTextTranslationResultWindow"
        }
        VoxtLog.input(
            "Deliver committed output started. delivery=\(deliveryLabel), characters=\(context.outputText.count)",
            verbose: true
        )

        switch delivery {
        case .typeText:
            let autoKeyPressHotkey = sessionOutputMode == .transcription
                ? autoKeyPressHotkeyForCurrentAppBranchSession()
                : nil
            let outputGeneration = recordingLifecycle.outputGeneration
            beginOverlayOutputDelivery()
            typeText(context.outputText, isValid: { [weak self] in
                self?.recordingLifecycle.accepts(sessionID) == true
            }) { [weak self] didInject in
                guard let self, self.recordingLifecycle.accepts(sessionID), !self.isApplicationTerminating else { return }
                self.sessionFinalOutputDeliveredAt = Date()
                self.endOverlayOutputDelivery()
                let outputDestinationContext = didInject
                    ? self.captureCurrentOutputDestinationContext()
                    : nil
                self.sessionOutputDestinationContext = outputDestinationContext
                if didInject, let autoKeyPressHotkey {
                    self.pressAutoKeyAfterTextInjection(autoKeyPressHotkey, outputGeneration: outputGeneration)
                }
                OnboardingSessionEvent.delivered(id: sessionID, text: context.outputText, succeeded: didInject).post()
                completion?(
                    didInject,
                    didInject && autoKeyPressHotkey != nil,
                    outputDestinationContext
                )
            }
        case .answerOverlay:
            if overlayState.isRewriteConversationActive, context.rewriteAnswerPayload == nil {
                presentRewriteConversationAnswerOverlay(content: context.outputText)
            } else {
                let payload = resolvedAnswerPayload(for: context)
                presentRewriteAnswerOverlay(title: payload.title, content: payload.content)
            }
            sessionFinalOutputDeliveredAt = Date()
            completion?(false, false, nil)
        case .selectedTextTranslationResultWindow:
            presentSelectedTextTranslationAnswerOverlay(content: context.outputText)
            sessionFinalOutputDeliveredAt = Date()
            OnboardingSessionEvent.delivered(id: sessionID, text: context.outputText, succeeded: true).post()
            completion?(false, false, nil)
        }
    }

    private func finalizeCommittedOutputPostDelivery(
        deliveredContext: SessionFinalizeContext,
        outputMode: SessionOutputMode,
        didInject: Bool,
        didTriggerAutoKeyPress: Bool,
        outputDestinationContext: OutputDestinationContext?
    ) {
        let deliveredText = deliveredContext.outputText
        let displayTitle = deliveredContext.rewriteAnswerPayload?.trimmedTitle
        let dictionaryMatches = deliveredContext.dictionaryMatches
        let dictionaryCorrectedTerms = deliveredContext.dictionaryCorrectedTerms
        let dictionaryCorrectionSnapshots = deliveredContext.dictionaryCorrectionSnapshots
        let llmDurationSeconds = deliveredContext.llmDurationSeconds
        let rewriteConversationTurns = outputMode == .rewrite
            ? overlayState.rewriteConversationTurns
            : []
        let asrSummary = sessionASRSummary(for: outputMode)
        let timingSnapshot = SessionTimingSummarySnapshot(
            transcriptionCapturePipeline: transcriptionCapturePipeline,
            captureStageLabels: transcriptionCapturePipeline.stageLabels,
            asrProvider: asrSummary.provider,
            asrModel: asrSummary.model,
            captureMetrics: currentTranscriptionCaptureMetrics(),
            recordingRequestedAt: recordingRequestedAt,
            recordingStartedAt: recordingStartedAt,
            recordingStoppedAt: recordingStoppedAt,
            transcriptionResultReceivedAt: transcriptionResultReceivedAt,
            firstLiveASRPartialReceivedAt: firstLiveASRPartialReceivedAt,
            sessionFinalOutputDeliveredAt: sessionFinalOutputDeliveredAt,
            llmExecutions: sessionLLMExecutionTimings
        )

        let historyEntryID = appendHistoryIfNeeded(
            text: deliveredText,
            outputMode: outputMode,
            outputDestinationContext: outputDestinationContext,
            displayTitle: displayTitle,
            llmDurationSeconds: llmDurationSeconds,
            dictionaryHitTerms: Self.orderedUniqueDictionaryTerms(from: dictionaryMatches.map(\.term)),
            dictionaryCorrectedTerms: Self.orderedUniqueDictionaryTerms(from: dictionaryCorrectedTerms),
            dictionaryCorrectionSnapshots: dictionaryCorrectionSnapshots,
            // Automatic suggestion discovery is retired; keep the history field
            // for older records. Explicit history scans add directly to DictionaryStore.
            dictionarySuggestedTerms: [],
            rewriteConversationTurns: rewriteConversationTurns
        )
        overlayState.latestHistoryEntryID = historyEntryID
        if didInject {
            let dictionaryScope = currentDictionaryScope()
            let reinforcedTerms = dictionaryStore.incrementOccurrences(
                in: deliveredText,
                activeGroupID: dictionaryScope.groupID
            )
            if !reinforcedTerms.isEmpty {
                VoxtLog.input(
                    "Dictionary occurrences reinforced from delivered text. terms=\(reinforcedTerms.joined(separator: ", "))",
                    verbose: true
                )
            }
        }
        scheduleAutomaticDictionaryLearningIfNeeded(
            insertedText: deliveredText,
            outputMode: outputMode,
            didInject: didInject,
            didTriggerAutoKeyPress: didTriggerAutoKeyPress,
            historyEntryID: historyEntryID
        )
        dictionaryStore.recordMatches(dictionaryMatches)
        VoxtLog.input(
            "Deliver committed output finalized. historyEntryID=\(historyEntryID?.uuidString ?? "nil"), characters=\(deliveredText.count)",
            verbose: true
        )
        logSessionTimingSummaryIfPossible(
            snapshot: timingSnapshot,
            deliveredText: deliveredText,
            outputMode: outputMode,
            didInject: didInject
        )
    }

    private func resolvedOutputDelivery(for context: SessionFinalizeContext) -> SessionOutputDelivery {
        if shouldPresentSelectedTextTranslationAnswerOverlay() {
            return .selectedTextTranslationResultWindow
        }

        if shouldAutoInjectSelectedTextTranslationResult() {
            return .typeText
        }

        return shouldPresentRewriteAnswerOverlay(hasSelectedSourceText: rewriteSessionHasSelectedSourceText)
            ? SessionOutputDelivery.answerOverlay
            : SessionOutputDelivery.typeText
    }

    static func shouldPresentSelectedTextTranslationAnswerOverlay(
        sessionOutputMode: SessionOutputMode,
        isSelectedTextTranslationFlow: Bool,
        showResultWindow: Bool
    ) -> Bool {
        sessionOutputMode == .translation &&
            isSelectedTextTranslationFlow &&
            showResultWindow
    }

    func shouldPresentSelectedTextTranslationAnswerOverlay() -> Bool {
        Self.shouldPresentSelectedTextTranslationAnswerOverlay(
            sessionOutputMode: sessionOutputMode,
            isSelectedTextTranslationFlow: isSelectedTextTranslationFlow,
            showResultWindow: showSelectedTextTranslationResultWindow
        )
    }

    static func shouldAutoInjectSelectedTextTranslationResult(
        sessionOutputMode: SessionOutputMode,
        isSelectedTextTranslationFlow: Bool,
        showResultWindow: Bool
    ) -> Bool {
        sessionOutputMode == .translation &&
            isSelectedTextTranslationFlow &&
            !showResultWindow
    }

    func shouldAutoInjectSelectedTextTranslationResult() -> Bool {
        Self.shouldAutoInjectSelectedTextTranslationResult(
            sessionOutputMode: sessionOutputMode,
            isSelectedTextTranslationFlow: isSelectedTextTranslationFlow,
            showResultWindow: showSelectedTextTranslationResultWindow
        )
    }

    static func shouldPresentRewriteAnswerOverlay(
        sessionOutputMode: SessionOutputMode,
        hasSelectedSourceText _: Bool
    ) -> Bool {
        sessionOutputMode == .rewrite
    }

    func shouldPresentRewriteAnswerOverlay(hasSelectedSourceText: Bool) -> Bool {
        Self.shouldPresentRewriteAnswerOverlay(
            sessionOutputMode: sessionOutputMode,
            hasSelectedSourceText: hasSelectedSourceText
        )
    }

    static func shouldUseStructuredRewriteAnswerOutput(
        sessionOutputMode: SessionOutputMode,
        hasSelectedSourceText: Bool
    ) -> Bool {
        sessionOutputMode == .rewrite && !hasSelectedSourceText
    }

    func shouldUseStructuredRewriteAnswerOutput(hasSelectedSourceText: Bool) -> Bool {
        Self.shouldUseStructuredRewriteAnswerOutput(
            sessionOutputMode: sessionOutputMode,
            hasSelectedSourceText: hasSelectedSourceText
        )
    }

    private func resolvedAnswerPayload(for context: SessionFinalizeContext) -> RewriteAnswerPayload {
        if let payload = context.rewriteAnswerPayload,
           !payload.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           !payload.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return payload
        }

        if let placeholderTitle = emptyRewriteAnswerPlaceholderTitle(from: context.outputText) {
            return RewriteAnswerPayload(
                title: placeholderTitle,
                content: "Unable to generate answer."
            )
        }

        return RewriteAnswerPayload(
            title: AppLocalization.localizedString("AI Answer"),
            content: context.outputText
        )
    }

    private func presentRewriteAnswerOverlay(title: String, content: String) {
        let resolvedPayload = normalizedRewriteAnswerPayload(
            RewriteAnswerPayload(title: title, content: content)
        )
        let trimmedContent = resolvedPayload.trimmedContent
        guard !trimmedContent.isEmpty else { return }

        if autoCopyWhenNoFocusedInput {
            writeTextToPasteboard(trimmedContent)
        }

        answerOverlayInjectionMode = .standard
        configureAnswerOverlayInjectionHandler()
        let canInjectIntoFocusedInput = resolvedCanInjectIntoFocusedInputForRewriteAnswer(logResult: true)
        overlayState.presentAnswer(
            title: resolvedPayload.trimmedTitle.isEmpty
                ? AppLocalization.localizedString("AI Answer")
                : resolvedPayload.trimmedTitle,
            content: trimmedContent,
            canInject: canInjectIntoFocusedInput
        )
        overlayWindow.show(state: overlayState, position: overlayPosition)
    }

    private func presentSelectedTextTranslationAnswerOverlay(content: String) {
        let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedContent.isEmpty else { return }

        if autoCopyWhenNoFocusedInput {
            writeTextToPasteboard(trimmedContent)
        }

        answerOverlayInjectionMode = .selectedTextTranslation
        configureAnswerOverlayInjectionHandler()

        overlayState.configureSessionTranslationTargetLanguage(
            translationTargetLanguage,
            allowsSwitching: true
        )
        overlayState.presentAnswer(
            title: AppLocalization.localizedString("Translation"),
            content: trimmedContent,
            canInject: true
        )
        overlayWindow.show(state: overlayState, position: overlayPosition)
    }

    private func presentRewriteConversationAnswerOverlay(content: String) {
        let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedContent.isEmpty else { return }

        if autoCopyWhenNoFocusedInput {
            writeTextToPasteboard(trimmedContent)
        }

        answerOverlayInjectionMode = .standard
        configureAnswerOverlayInjectionHandler()
        let canInjectIntoFocusedInput = resolvedCanInjectIntoFocusedInputForRewriteAnswer(logResult: true)
        overlayState.presentConversationAnswer(
            content: trimmedContent,
            canInject: canInjectIntoFocusedInput
        )
        overlayWindow.show(state: overlayState, position: overlayPosition)
    }

    func presentRewriteConversationStreamingPreview(content: String) {
        let normalizedContent = RewriteAnswerContentNormalizer.normalizePlainTextStreamingPreview(content)
        guard !normalizedContent.isEmpty else { return }

        configureAnswerOverlayInjectionHandler()
        let canInjectIntoFocusedInput =
            overlayState.latestCompletedAnswerPayload != nil
                ? resolvedCanInjectIntoFocusedInputForRewriteAnswer(logResult: false)
                : false

        overlayState.presentStreamingConversationAnswer(
            content: normalizedContent,
            canInject: canInjectIntoFocusedInput
        )
        overlayWindow.show(state: overlayState, position: overlayPosition)
    }

    private func normalizedRewriteAnswerPayload(_ payload: RewriteAnswerPayload) -> RewriteAnswerPayload {
        RewriteAnswerPayloadParser.normalize(payload)
    }

    func dismissAnswerOverlay() {
        guard overlayState.displayMode == .answer else { return }
        recordingLifecycle.invalidateOutputDelivery()
        if isSessionActive {
            cancelActiveRecordingSession()
        }
        cancelPendingSelectedTextTranslationRefresh()
        releaseResidualRecordingResources(reason: "dismiss-answer-overlay")
        let generation = recordingLifecycle.outputGeneration
        overlayWindow.hide { [weak self] in
            guard let self, self.recordingLifecycle.acceptsOutputGeneration(generation) else { return }
            self.overlayWindow.onRequestInject = nil
            self.overlayState.reset()
            self.answerOverlayInjectionMode = .standard
            self.sessionTargetApplicationPID = nil
            self.sessionTargetApplicationBundleID = nil
            self.selectedTextTranslationHadWritableFocusedInput = false
        }
    }

    func injectAnswerOverlayContent() {
        let trimmed = overlayState.latestCompletedAnswerPayload?.trimmedContent ?? ""
        guard !trimmed.isEmpty else { return }
        guard overlayState.canInjectAnswer else { return }
        VoxtLog.input("Answer overlay inject requested. chars=\(trimmed.count), canInject=\(overlayState.canInjectAnswer)")

        injectAnswerOverlayContent(trimmed, mode: answerOverlayInjectionMode)
    }

    private func injectAnswerOverlayContent(_ text: String, mode: AnswerOverlayInjectionMode) {
        let generation = recordingLifecycle.outputGeneration
        let target = sessionTextInjectionTarget
        let historyEntryID = overlayState.latestHistoryEntryID
        let isValid: @MainActor () -> Bool = { [weak self] in
            guard let self, !self.isApplicationTerminating else { return false }
            return self.recordingLifecycle.acceptsOutputGeneration(generation)
        }
        VoxtLog.input("Answer overlay inject will hide overlay before paste. chars=\(text.count)")
        overlayWindow.onRequestInject = nil
        overlayWindow.hide(animated: false) { [weak self] in
            guard let self, isValid() else { return }
            self.overlayState.canInjectAnswer = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) { [weak self] in
                guard let self, isValid() else { return }
                if mode == .selectedTextTranslation {
                    let activationRestored = self.activateSelectedTextTranslationInjectionTargetIfNeeded(target)
                    VoxtLog.input(
                        "Selected text translation overlay inject target prepared. activationRestored=\(activationRestored)"
                    )
                }
                self.typeText(
                    text,
                    restoreSessionTarget: mode == .standard,
                    target: target,
                    isValid: isValid
                ) { [weak self] didInject in
                    guard let self, isValid() else { return }
                    VoxtLog.input("Answer overlay inject completed. didInject=\(didInject)")
                    if didInject {
                        self.updateHistoryOutputDestinationAfterInjection(historyEntryID: historyEntryID)
                        self.overlayState.reset()
                        self.answerOverlayInjectionMode = .standard
                        self.sessionTargetApplicationPID = nil
                        self.sessionTargetApplicationBundleID = nil
                        self.selectedTextTranslationHadWritableFocusedInput = false
                    } else {
                        self.configureAnswerOverlayInjectionHandler()
                        self.overlayState.canInjectAnswer = true
                        self.overlayWindow.show(state: self.overlayState, position: self.overlayPosition)
                    }
                }
            }
        }
    }

    @discardableResult
    private func activateSelectedTextTranslationInjectionTargetIfNeeded(_ target: TextInjectionTarget) -> Bool {
        let ownBundleID = Bundle.main.bundleIdentifier
        let frontmostApplication = NSWorkspace.shared.frontmostApplication
        let frontmostBundleID = frontmostApplication?.bundleIdentifier
        if frontmostBundleID != ownBundleID,
           frontmostBundleID == target.bundleIdentifier {
            return false
        }

        if let targetPID = target.processIdentifier,
           let targetApplication = NSRunningApplication(processIdentifier: targetPID),
           !targetApplication.isTerminated {
            VoxtLog.input(
                "Selected text translation overlay restoring target app before paste. bundleID=\(targetApplication.bundleIdentifier ?? "unknown"), pid=\(targetPID)"
            )
            return targetApplication.activate(options: [])
        }

        if let targetBundleID = target.bundleIdentifier,
           let targetApplication = NSRunningApplication.runningApplications(withBundleIdentifier: targetBundleID)
            .first(where: { !$0.isTerminated }) {
            VoxtLog.input(
                "Selected text translation overlay restoring target app by bundle ID before paste. bundleID=\(targetBundleID), pid=\(targetApplication.processIdentifier)"
            )
            return targetApplication.activate(options: [])
        }

        VoxtLog.input(
            "Selected text translation overlay target restore skipped. frontmostBundleID=\(frontmostBundleID ?? "nil"), targetBundleID=\(target.bundleIdentifier ?? "nil"), targetPID=\(target.processIdentifier.map(String.init) ?? "nil")"
        )
        return false
    }

    func showCurrentTranscriptionDetailWindow() {
        guard let historyEntryID = overlayState.latestHistoryEntryID else {
            VoxtLog.inputWarning("Transcription detail open skipped: latest history entry ID was unavailable.")
            return
        }
        showTranscriptionDetailWindow(for: historyEntryID)
    }

    private func updateHistoryOutputDestinationAfterInjection(historyEntryID: UUID?) {
        guard let historyEntryID,
              let outputDestinationContext = captureCurrentOutputDestinationContext()
        else {
            return
        }
        _ = historyStore.updateOutputDestination(
            for: historyEntryID,
            focusedAppName: outputDestinationContext.appName,
            focusedAppBundleID: outputDestinationContext.bundleID,
            browserURLHost: outputDestinationContext.browserURLHost,
            browserURLOrigin: outputDestinationContext.browserURLOrigin
        )
    }

    private func configureAnswerOverlayInjectionHandler() {
        overlayWindow.onRequestInject = { [weak self] in
            guard let self else { return }
            let generation = self.recordingLifecycle.outputGeneration
            Task { @MainActor [weak self] in
                guard let self, self.recordingLifecycle.acceptsOutputGeneration(generation) else { return }
                self.injectAnswerOverlayContent()
            }
        }
    }

    private func resolvedCanInjectIntoFocusedInputForRewriteAnswer(logResult: Bool) -> Bool {
        let liveHasWritableFocusedInput = hasWritableFocusedTextInput()
        let hasFallbackInjectTarget = rewriteSessionFallbackInjectBundleID != nil
        let canInjectIntoFocusedInput =
            rewriteSessionHadWritableFocusedInput ||
            liveHasWritableFocusedInput ||
            hasFallbackInjectTarget
        if logResult {
            VoxtLog.input(
                "Rewrite answer overlay inject check. sessionHadWritableFocusedInput=\(rewriteSessionHadWritableFocusedInput), liveHasWritableFocusedInput=\(liveHasWritableFocusedInput), fallbackBundleID=\(rewriteSessionFallbackInjectBundleID ?? "nil"), canInject=\(canInjectIntoFocusedInput)"
            )
        }
        return canInjectIntoFocusedInput
    }

    private func emptyRewriteAnswerPlaceholderTitle(from text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        let title = (object["title"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let content = (object["content"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !title.isEmpty, content.isEmpty else { return nil }
        return title
    }

}
