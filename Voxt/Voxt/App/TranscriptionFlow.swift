// TranscriptionFlow.swift
// Provides Transcription Flow for app lifecycle and routing.

import Foundation

extension AppDelegate {
    private struct TranscriptionEnhanceStage: SessionPipelineStage {
        let transform: @MainActor (String) async throws -> String

        var name: String { "transcriptionEnhance" }

        func run(context: SessionPipelineContext) async throws -> SessionPipelineContext {
            var next = context
            next.workingText = try await transform(context.workingText)
            return next
        }
    }

    // MARK: - Standard Transcription Flow

    func processStandardTranscription(_ text: String, sessionID: UUID) {
        guard shouldHandleCallbacks(for: sessionID) else { return }
        VoxtLog.asr("Standard transcription flow entered. characters=\(text.count), enhancementMode=\(enhancementMode.rawValue)", verbose: true)

        switch enhancementMode {
        case .off:
            setEnhancingState(false)
            overlayState.transcribedText = text
            VoxtLog.asr("Standard transcription committing raw text immediately. characters=\(text.count)", verbose: true)
            commitTranscription(text, llmDurationSeconds: nil) { [weak self] in
                self?.finishSession(after: 0)
            }

        case .customLLM:
            guard let enhancementRepo = resolvedTranscriptionEnhancementLocalRepo(),
                  customLLMManager.canAttemptInference(repo: enhancementRepo) else {
                VoxtLog.asrWarning("Custom LLM selected but local model is not installed. Using raw transcription.")
                showOverlayStatus(
                    AppLocalization.localizedString("Custom LLM model is not installed. Open Settings > Model to install it."),
                    clearAfter: 2.5
                )
                setEnhancingState(false)
                overlayState.transcribedText = text
                VoxtLog.asr("Standard transcription falling back to raw text because custom model is unavailable. characters=\(text.count)", verbose: true)
                commitTranscription(text, llmDurationSeconds: nil) { [weak self] in
                    self?.finishSession(after: 0)
                }
                return
            }
            runStandardTranscriptionPipelineAsync(
                text,
                sessionID: sessionID
            )

        case .appleIntelligence, .remoteLLM:
            runStandardTranscriptionPipelineAsync(
                text,
                sessionID: sessionID
            )
        }
    }

    private func runStandardTranscriptionPipelineAsync(
        _ text: String,
        sessionID: UUID
    ) {
        setEnhancingState(true)
        overlayState.transcribedText = text
        let requestID = beginLLMRequest()
        runTrackedLLMRequest(requestID) {
            defer {
                if self.isCurrentLLMRequest(requestID) {
                    self.setEnhancingState(false)
                }
            }

            let llmStartedAt = Date()
            if let asrAt = self.transcriptionResultReceivedAt {
                let handoffMs = Int(llmStartedAt.timeIntervalSince(asrAt) * 1000)
                VoxtLog.asr("Enhancement handoff. mode=\(self.enhancementMode.rawValue), handoffMs=\(max(handoffMs, 0)), inputChars=\(text.count)", verbose: true)
            } else {
                VoxtLog.asr("Enhancement handoff. mode=\(self.enhancementMode.rawValue), handoffMs=unknown, inputChars=\(text.count)", verbose: true)
            }
            do {
                let enhanced = try await self.runStandardTranscriptionPipeline(text: text)
                guard self.shouldHandleCallbacks(for: sessionID), self.isCurrentLLMRequest(requestID) else { return }
                let llmDuration = Date().timeIntervalSince(llmStartedAt)
                VoxtLog.asr("Enhancement completed. mode=\(self.enhancementMode.rawValue), inputChars=\(text.count), outputChars=\(enhanced.count), llmDurationSec=\(String(format: "%.3f", llmDuration))")
                self.overlayState.transcribedText = enhanced
                self.commitTranscription(enhanced, llmDurationSeconds: llmDuration) { [weak self] in
                    self?.finishSession(after: 0)
                }
            } catch {
                guard self.shouldHandleCallbacks(for: sessionID), self.isCurrentLLMRequest(requestID) else { return }
                VoxtLog.asrWarning("Standard transcription pipeline enhancement failed, using raw text: \(error)")
                self.overlayState.transcribedText = text
                self.commitTranscription(text, llmDurationSeconds: nil) { [weak self] in
                    self?.finishSession(after: 0)
                }
            }
        }
    }

    func runStandardTranscriptionPipeline(
        text: String
    ) async throws -> String {
        let sessionID = activeRecordingSessionID
        let runner = SessionPipelineRunner(
            stages: [
                TranscriptionEnhanceStage(transform: { [weak self] value in
                    guard let self else { return value }
                    return try await self.enhanceTextForCurrentMode(
                        value,
                        sessionID: sessionID
                    )
                })
            ]
        )
        let result = try await runner.run(initial: SessionPipelineContext(originalText: text, workingText: text))
        return result.workingText
    }

    func enhanceTextForCurrentMode(
        _ text: String,
        sessionID: UUID? = nil
    ) async throws -> String {
        guard let provider = resolvedTranscriptionEnhancementProvider() else {
            return text
        }
        let basePolicy = DictionaryGlossaryPurpose.enhancement.selectionPolicy
        let provisionalStrategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: text,
            promptCharacterCount: 0,
            baseGlossarySelectionPolicy: basePolicy,
            capabilities: llmProviderModelCapabilities(for: provider)
        )
        let promptResolution = resolvedEnhancementPrompt(
            rawTranscription: text,
            glossarySelectionPolicy: provisionalStrategy.glossarySelectionPolicy
        )
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: text,
            promptCharacterCount: promptResolution.content.count +
                (promptResolution.dictionaryGlossary?.count ?? 0),
            baseGlossarySelectionPolicy: basePolicy,
            capabilities: llmProviderModelCapabilities(for: provider)
        )
        VoxtLog.asr(
            "Task LLM strategy resolved. inputChars=\(text.count), \(strategy.logLabel)",
            verbose: true
        )
        if promptResolution.delivery == EnhancementPromptResolution.Delivery.skipEnhancement {
            if let sessionID,
               autoKeyPressHotkeyForCurrentAppBranchSession() != nil {
                applyEnhancementOverlayIconIfNeeded(
                    match: lastEnhancementPromptContext?.overlayIconMatch,
                    sessionID: sessionID
                )
            }
            return text
        }
        if let sessionID {
            applyEnhancementOverlayIconIfNeeded(match: promptResolution.overlayIconMatch, sessionID: sessionID)
        }

        if enhancementMode == .remoteLLM {
            let context = resolvedRemoteLLMContext(forTranslation: false)
            guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: context.provider,
                stored: remoteLLMConfigurations
            ) else {
                VoxtLog.asrWarning("Enhancement provider remoteLLM unavailable: no configured model.")
                return text
            }
            VoxtLog.llmDebug(
                "Remote LLM enhancement request. provider=\(context.provider.rawValue), model=\(context.configuration.model)"
            )
        }

        guard let plan = buildEnhancementExecutionPlan(
            rawText: text,
            promptResolution: promptResolution,
            providerOverride: provider,
            executionStrategy: strategy
        ) else {
            return text
        }
        let enhanced = try await executeLLMExecutionPlan(plan)
        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: enhanced,
            originalText: text,
            strategy: strategy
        )
        if guarded.didFallback {
            VoxtLog.asrWarning(
                "Enhancement truncation guard restored raw text. inputChars=\(text.count), outputChars=\(enhanced.count), reason=\(guarded.reason ?? "unknown"), strategy=\(strategy.logLabel)"
            )
        }
        return guarded.text
    }
}
