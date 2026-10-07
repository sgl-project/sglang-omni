// TranslationSupport.swift
// Provides Translation Support for app lifecycle and routing.

import Foundation

extension AppDelegate {
    enum TextTransformFailure: LocalizedError {
        case translationModelNotInstalled
        case translationRemoteModelNotConfigured
        case translationProviderUnavailable
        case translationPlanUnavailable
        case translationRejectedByGuard(reason: String?)
        case rewriteModelNotInstalled
        case rewriteRemoteModelNotConfigured
        case rewriteRejectedByGuard(reason: String?)

        var errorDescription: String? {
            switch self {
            case .translationModelNotInstalled:
                return AppLocalization.localizedString("Translation failed because the selected local LLM model is not installed.")
            case .translationRemoteModelNotConfigured:
                return AppLocalization.localizedString("Translation failed because no remote LLM model is configured.")
            case .translationProviderUnavailable:
                return AppLocalization.localizedString("Translation failed because the selected translation provider is unavailable for this request.")
            case .translationPlanUnavailable:
                return AppLocalization.localizedString("Translation failed because Voxt could not prepare the translation request.")
            case .translationRejectedByGuard:
                return AppLocalization.localizedString("Translation failed because the model output looked incomplete.")
            case .rewriteModelNotInstalled:
                return AppLocalization.localizedString("Rewrite failed because the selected local LLM model is not installed.")
            case .rewriteRemoteModelNotConfigured:
                return AppLocalization.localizedString("Rewrite failed because no remote LLM model is configured.")
            case .rewriteRejectedByGuard:
                return AppLocalization.localizedString("Rewrite failed because the model output looked incomplete.")
            }
        }

        var recoverySuggestion: String? {
            switch self {
            case .translationModelNotInstalled, .rewriteModelNotInstalled:
                return AppLocalization.localizedString("Open Settings > Model to install it.")
            case .translationRemoteModelNotConfigured, .rewriteRemoteModelNotConfigured:
                return AppLocalization.localizedString("Configure a provider in Settings > Model.")
            case .translationRejectedByGuard, .rewriteRejectedByGuard,
                 .translationProviderUnavailable, .translationPlanUnavailable:
                return nil
            }
        }
    }

    enum VariablePromptDelivery: Equatable {
        case systemPrompt
        case userMessage
    }

    struct VariablePromptResolution: Equatable {
        let content: String
        let dictionaryGlossary: String?
        let delivery: VariablePromptDelivery
        let promptProfile: String
    }

    func isStoredRemoteLLMConfigured(_ provider: RemoteLLMProvider) -> Bool {
        RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
            provider: provider,
            stored: remoteLLMConfigurations
        )
    }

    func resolvedRemoteLLMContext(forTranslation: Bool) -> (provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration) {
        let provider: RemoteLLMProvider
        if forTranslation, let translationProvider = translationRemoteLLMProvider {
            provider = translationProvider
        } else {
            provider = remoteLLMSelectedProvider
        }

        let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: provider,
            stored: remoteLLMConfigurations
        )
        return (provider, configuration)
    }

    func resolvedRemoteLLMContext(forRewrite: Bool) -> (provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration) {
        let provider: RemoteLLMProvider
        if forRewrite, let rewriteProvider = rewriteRemoteLLMProvider {
            provider = rewriteProvider
        } else {
            provider = remoteLLMSelectedProvider
        }

        let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
            provider: provider,
            stored: remoteLLMConfigurations
        )
        return (provider, configuration)
    }

    var currentRewriteResponsesConversationContextKey: String? {
        guard rewriteModelProvider == .remoteLLM else { return nil }
        let context = resolvedRemoteLLMContext(forRewrite: true)
        guard context.provider.usesResponsesAPI else { return nil }
        return [
            context.provider.rawValue,
            context.configuration.endpoint,
            context.configuration.model
        ].joined(separator: "|")
    }

    func translateText(
        _ text: String,
        targetLanguage: TranslationTargetLanguage,
        providerResolution: TranslationProviderResolution? = nil
    ) async throws -> String {
        let resolution = providerResolution ?? resolvedTranslationProviderResolution(
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: isSelectedTextTranslationFlow
        )
        let modelProvider = resolution.provider
        let providerOverride: LLMExecutionProvider?
        switch modelProvider {
        case .customLLM:
            providerOverride = .customLLM(repo: translationCustomLLMRepo)
        case .localGGUF:
            providerOverride = .localGGUF(modelID: translationGGUFModelID)
        case .remoteLLM:
            let context = resolvedRemoteLLMContext(forTranslation: true)
            providerOverride = .remote(provider: context.provider, configuration: context.configuration)
        }
        let basePolicy = DictionaryGlossaryPurpose.translation.selectionPolicy
        let provisionalStrategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: text,
            promptCharacterCount: 0,
            baseGlossarySelectionPolicy: basePolicy,
            capabilities: providerOverride.map(llmProviderModelCapabilities(for:)) ?? .unknown
        )
        let promptResolution = resolvedTranslationPrompt(
            targetLanguage: targetLanguage,
            sourceText: text,
            strict: false,
            glossarySelectionPolicy: provisionalStrategy.glossarySelectionPolicy
        )
        let translationModel = translationModelLogDescriptor(for: modelProvider)
        VoxtLog.llmDebug(
            "Translation request. promptChars=\(promptResolution.content.count), inputChars=\(text.count), provider=\(modelProvider.rawValue), selectedProvider=\(translationModelProvider.rawValue), translationModel=\(translationModel), delivery=\(String(describing: promptResolution.delivery)), promptProfile=\(promptResolution.promptProfile)"
        )

        if modelProvider == .customLLM {
            let translationRepo = translationCustomLLMRepo
            guard customLLMManager.canAttemptInference(repo: translationRepo) else {
                VoxtLog.translationWarning("Translation provider customLLM unavailable: model not downloaded. repo=\(translationRepo)")
                throw TextTransformFailure.translationModelNotInstalled
            }
            VoxtLog.translation("Translation provider selected: customLLM")
        } else if modelProvider == .localGGUF {
            guard ggufTranslationModelManager.canAttemptInference(id: translationGGUFModelID) else {
                VoxtLog.translationWarning("Translation provider localGGUF unavailable: model not downloaded. modelID=\(translationGGUFModelID.rawValue)")
                throw TextTransformFailure.translationModelNotInstalled
            }
            VoxtLog.translation("Translation provider selected: localGGUF(\(translationGGUFModelID.rawValue))")
        } else if modelProvider == .remoteLLM {
            let context = resolvedRemoteLLMContext(forTranslation: true)
            guard isStoredRemoteLLMConfigured(context.provider) else {
                VoxtLog.translationWarning("Translation provider remoteLLM unavailable: no configured model.")
                throw TextTransformFailure.translationRemoteModelNotConfigured
            }
            VoxtLog.translation("Translation provider selected: remoteLLM(\(context.provider.rawValue))")
        } else {
            throw TextTransformFailure.translationProviderUnavailable
        }

        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: text,
            promptCharacterCount: promptResolution.content.count + (promptResolution.dictionaryGlossary?.count ?? 0),
            baseGlossarySelectionPolicy: basePolicy,
            capabilities: providerOverride.map(llmProviderModelCapabilities(for:)) ?? .unknown
        )
        guard let plan = buildTranslationExecutionPlan(
            sourceText: text,
            targetLanguage: targetLanguage,
            promptResolution: promptResolution,
            modelProvider: modelProvider,
            providerOverride: providerOverride,
            executionStrategy: strategy
        ) else {
            throw TextTransformFailure.translationPlanUnavailable
        }
        let translated = try await executeLLMExecutionPlan(plan)
        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: translated,
            originalText: text,
            strategy: strategy
        )
        if guarded.didFallback {
            VoxtLog.translationWarning(
                "Translation truncation guard restored source text. inputChars=\(text.count), outputChars=\(translated.count), reason=\(guarded.reason ?? "unknown"), strategy=\(strategy.logLabel)"
            )
            throw TextTransformFailure.translationRejectedByGuard(reason: guarded.reason)
        }
        return guarded.text
    }

    func rewriteText(
        dictatedPrompt: String,
        sourceText: String,
        conversationHistory: [RewriteConversationPromptTurn],
        structuredAnswerOutput: Bool,
        forceNonEmptyAnswer: Bool = false,
        previousConversationResponseID: String? = nil,
        onProgress: (@Sendable (String) -> Void)? = nil,
        onResponseID: ((String) -> Void)? = nil
    ) async throws -> String {
        let directAnswerMode = sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let trimmedConversationHistory = trimmedRewriteConversationHistory(
            conversationHistory
        )
        let modelProvider = rewriteModelProvider
        let remoteContext = rewriteModelProvider == .remoteLLM ? resolvedRemoteLLMContext(forRewrite: true) : nil
        let hasProviderConversationState =
            !trimmedConversationHistory.isEmpty ||
            !(previousConversationResponseID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "").isEmpty
        let shouldUseProviderManagedConversation =
            (remoteContext?.provider.usesResponsesAPI ?? false) &&
            directAnswerMode &&
            !structuredAnswerOutput &&
            hasProviderConversationState
        let shouldUseRoleMessageConversation =
            directAnswerMode &&
            !structuredAnswerOutput &&
            !trimmedConversationHistory.isEmpty &&
            !shouldUseProviderManagedConversation
        if shouldUseRoleMessageConversation {
            let latestTurn = trimmedConversationHistory.last
            VoxtLog.translation(
                """
                Rewrite continue conversation context prepared. turns=\(trimmedConversationHistory.count), latestTitle=\(VoxtLog.llmPreview(latestTurn?.resultTitle ?? "")), latestContent=\(VoxtLog.llmPreview(latestTurn?.resultContent ?? "")), currentPrompt=\(VoxtLog.llmPreview(dictatedPrompt))
                """
            )
        }
        let promptResolution = shouldUseRoleMessageConversation
            ? VariablePromptResolution(
                content: resolvedRewriteConversationPrompt(forceNonEmptyAnswer: forceNonEmptyAnswer),
                dictionaryGlossary: nil,
                delivery: .systemPrompt,
                promptProfile: "rewrite-conversation"
            )
            : resolvedRewritePrompt(
                dictatedPrompt: dictatedPrompt,
                sourceText: sourceText,
                conversationHistory: shouldUseProviderManagedConversation ? [] : trimmedConversationHistory,
                structuredAnswerOutput: structuredAnswerOutput,
                directAnswerMode: directAnswerMode,
                forceNonEmptyAnswer: forceNonEmptyAnswer,
                glossarySelectionPolicy: nil
            )
        let rewriteRepo = rewriteCustomLLMRepo
        VoxtLog.llmDebug(
            "Rewrite request. promptChars=\(promptResolution.content.count), dictatedChars=\(dictatedPrompt.count), sourceChars=\(sourceText.count), provider=\(modelProvider.rawValue), rewriteRepo=\(rewriteRepo), structuredAnswerOutput=\(structuredAnswerOutput), directAnswerMode=\(directAnswerMode), forceNonEmptyAnswer=\(forceNonEmptyAnswer), delivery=\(String(describing: promptResolution.delivery))"
        )
        if shouldUseProviderManagedConversation, let remoteContext {
            VoxtLog.translation(
                "Rewrite continue using Responses API. provider=\(remoteContext.provider.rawValue), endpoint=\(remoteContext.configuration.endpoint)"
            )
        } else if modelProvider == .remoteLLM,
                  directAnswerMode,
                  !structuredAnswerOutput,
                  onProgress != nil,
                  let remoteContext {
            VoxtLog.translation(
                "Rewrite continue using chat completions stream. provider=\(remoteContext.provider.rawValue), endpoint=\(remoteContext.configuration.endpoint)"
            )
        }

        if modelProvider == .customLLM {
            guard customLLMManager.canAttemptInference(repo: rewriteRepo) else {
                VoxtLog.translationWarning("Rewrite provider customLLM unavailable: model not downloaded. repo=\(rewriteRepo)")
                throw TextTransformFailure.rewriteModelNotInstalled
            }
        } else {
            let context = remoteContext ?? resolvedRemoteLLMContext(forRewrite: true)
            guard isStoredRemoteLLMConfigured(context.provider) else {
                VoxtLog.translationWarning("Rewrite provider remoteLLM unavailable: no configured model.")
                throw TextTransformFailure.rewriteRemoteModelNotConfigured
            }
        }

        let conversationHistoryForPlan: [RewriteConversationPromptTurn]
        switch modelProvider {
        case .customLLM:
            conversationHistoryForPlan = shouldUseRoleMessageConversation ? trimmedConversationHistory : []
        case .remoteLLM:
            switch promptResolution.delivery {
            case .systemPrompt:
                conversationHistoryForPlan = (shouldUseProviderManagedConversation || shouldUseRoleMessageConversation) ? trimmedConversationHistory : []
            case .userMessage:
                conversationHistoryForPlan = shouldUseProviderManagedConversation ? trimmedConversationHistory : []
            }
        }

        let providerOverride: LLMExecutionProvider
        switch modelProvider {
        case .customLLM:
            providerOverride = .customLLM(repo: rewriteRepo)
        case .remoteLLM:
            let context = remoteContext ?? resolvedRemoteLLMContext(forRewrite: true)
            providerOverride = .remote(provider: context.provider, configuration: context.configuration)
        }
        let strategyInputText = directAnswerMode ? dictatedPrompt : sourceText
        var strategyPromptCharacterCount = promptResolution.content.count
        strategyPromptCharacterCount += promptResolution.dictionaryGlossary?.count ?? 0
        let providerCapabilities = llmProviderModelCapabilities(for: providerOverride)
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .rewrite,
            rawText: strategyInputText,
            promptCharacterCount: strategyPromptCharacterCount,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.rewrite.selectionPolicy,
            capabilities: providerCapabilities
        )
        let plan = buildRewriteExecutionPlan(
            dictatedPrompt: dictatedPrompt,
            sourceText: sourceText,
            promptResolution: promptResolution,
            modelProvider: modelProvider,
            conversationHistory: conversationHistoryForPlan,
            previousResponseID: shouldUseProviderManagedConversation ? previousConversationResponseID : nil,
            structuredAnswerOutput: structuredAnswerOutput,
            providerOverride: providerOverride,
            executionStrategy: strategy
        )
        let rewritten = try await executeLLMExecutionPlan(
            plan,
            onPartialText: onProgress,
            onResponseID: onResponseID
        )
        let originalText = sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? dictatedPrompt
            : sourceText
        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: rewritten,
            originalText: originalText,
            strategy: strategy
        )
        if guarded.didFallback {
            VoxtLog.translationWarning(
                "Rewrite truncation guard restored source text. inputChars=\(originalText.count), outputChars=\(rewritten.count), reason=\(guarded.reason ?? "unknown"), strategy=\(strategy.logLabel)"
            )
            throw TextTransformFailure.rewriteRejectedByGuard(reason: guarded.reason)
        }
        return guarded.text
    }

    func translateTextStrict(
        _ text: String,
        targetLanguage: TranslationTargetLanguage
    ) async throws -> String {
        let resolution = resolvedTranslationProviderResolution(
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: isSelectedTextTranslationFlow
        )
        let modelProvider = resolution.provider
        let providerOverride: LLMExecutionProvider?
        switch modelProvider {
        case .customLLM:
            providerOverride = .customLLM(repo: translationCustomLLMRepo)
        case .localGGUF:
            providerOverride = .localGGUF(modelID: translationGGUFModelID)
        case .remoteLLM:
            let context = resolvedRemoteLLMContext(forTranslation: true)
            providerOverride = .remote(provider: context.provider, configuration: context.configuration)
        }
        let basePolicy = DictionaryGlossaryPurpose.translation.selectionPolicy
        let provisionalStrategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: text,
            promptCharacterCount: 0,
            baseGlossarySelectionPolicy: basePolicy,
            capabilities: providerOverride.map(llmProviderModelCapabilities(for:)) ?? .unknown
        )
        let promptResolution = resolvedTranslationPrompt(
            targetLanguage: targetLanguage,
            sourceText: text,
            strict: true,
            glossarySelectionPolicy: provisionalStrategy.glossarySelectionPolicy
        )
        let translationModel = translationModelLogDescriptor(for: modelProvider)
        VoxtLog.llmDebug(
            "Strict translation retry. promptChars=\(promptResolution.content.count), inputChars=\(text.count), provider=\(modelProvider.rawValue), selectedProvider=\(translationModelProvider.rawValue), translationModel=\(translationModel), delivery=\(String(describing: promptResolution.delivery)), promptProfile=\(promptResolution.promptProfile)"
        )

        if modelProvider == .customLLM {
            let translationRepo = translationCustomLLMRepo
            guard customLLMManager.canAttemptInference(repo: translationRepo) else {
                throw TextTransformFailure.translationModelNotInstalled
            }
        } else if modelProvider == .localGGUF {
            guard ggufTranslationModelManager.canAttemptInference(id: translationGGUFModelID) else {
                throw TextTransformFailure.translationModelNotInstalled
            }
        } else if modelProvider == .remoteLLM {
            let context = resolvedRemoteLLMContext(forTranslation: true)
            guard isStoredRemoteLLMConfigured(context.provider) else {
                throw TextTransformFailure.translationRemoteModelNotConfigured
            }
        } else {
            throw TextTransformFailure.translationProviderUnavailable
        }

        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: text,
            promptCharacterCount: promptResolution.content.count + (promptResolution.dictionaryGlossary?.count ?? 0),
            baseGlossarySelectionPolicy: basePolicy,
            capabilities: providerOverride.map(llmProviderModelCapabilities(for:)) ?? .unknown
        )
        guard let plan = buildTranslationExecutionPlan(
            sourceText: text,
            targetLanguage: targetLanguage,
            promptResolution: promptResolution,
            modelProvider: modelProvider,
            providerOverride: providerOverride,
            executionStrategy: strategy
        ) else {
            throw TextTransformFailure.translationPlanUnavailable
        }
        let translated = try await executeLLMExecutionPlan(plan)
        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: translated,
            originalText: text,
            strategy: strategy
        )
        if guarded.didFallback {
            VoxtLog.translationWarning(
                "Strict translation truncation guard restored source text. inputChars=\(text.count), outputChars=\(translated.count), reason=\(guarded.reason ?? "unknown"), strategy=\(strategy.logLabel)"
            )
            throw TextTransformFailure.translationRejectedByGuard(reason: guarded.reason)
        }
        return guarded.text
    }

    static func effectiveTranslationTargetLanguage(
        savedTargetLanguage: TranslationTargetLanguage,
        sessionOverride: TranslationTargetLanguage?,
        isSelectedTextTranslation: Bool
    ) -> TranslationTargetLanguage {
        guard !isSelectedTextTranslation, let sessionOverride else {
            return savedTargetLanguage
        }
        return sessionOverride
    }

    var effectiveSessionTranslationTargetLanguage: TranslationTargetLanguage {
        Self.effectiveTranslationTargetLanguage(
            savedTargetLanguage: translationTargetLanguage,
            sessionOverride: sessionTranslationTargetLanguageOverride,
            isSelectedTextTranslation: isSelectedTextTranslationFlow
        )
    }

    func resolvedTranslationProviderResolution(
        targetLanguage: TranslationTargetLanguage,
        isSelectedTextTranslation: Bool
    ) -> TranslationProviderResolution {
        Self.resolvedSessionTranslationProviderResolution(
            lockedResolution: activeSessionTranslationProviderResolution,
            selectedProvider: translationModelProvider,
            fallbackProvider: translationFallbackModelProvider,
            transcriptionEngine: transcriptionEngine,
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: isSelectedTextTranslation
        )
    }

    static func resolvedSessionTranslationProviderResolution(
        lockedResolution: TranslationProviderResolution?,
        selectedProvider: TranslationModelProvider,
        fallbackProvider: TranslationModelProvider,
        transcriptionEngine: TranscriptionEngine,
        targetLanguage: TranslationTargetLanguage,
        isSelectedTextTranslation: Bool
    ) -> TranslationProviderResolution {
        if !isSelectedTextTranslation,
           let lockedResolution {
            return lockedResolution
        }

        return TranslationProviderResolver.resolve(
            selectedProvider: selectedProvider,
            fallbackProvider: fallbackProvider,
            transcriptionEngine: transcriptionEngine,
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: isSelectedTextTranslation
        )
    }

    func prepareMicrophoneTranslationSessionState() {
        let persistedTargetLanguage = translationTargetLanguage
        let resolution = TranslationProviderResolver.resolve(
            selectedProvider: translationModelProvider,
            fallbackProvider: translationFallbackModelProvider,
            transcriptionEngine: transcriptionEngine,
            targetLanguage: persistedTargetLanguage,
            isSelectedTextTranslation: false
        )

        sessionTranslationTargetLanguageOverride = persistedTargetLanguage
        activeSessionTranslationProviderResolution = resolution
        overlayState.configureSessionTranslationTargetLanguage(
            persistedTargetLanguage,
            allowsSwitching: Self.shouldAllowSessionTranslationLanguageSwitching(
                sessionOutputMode: .translation,
                isSelectedTextTranslationFlow: false
            )
        )
    }

    func resetSessionTranslationState() {
        cancelPendingSelectedTextTranslationRefresh()
        sessionTranslationTargetLanguageOverride = nil
        activeSessionTranslationProviderResolution = nil
        overlayState.configureSessionTranslationTargetLanguage(nil, allowsSwitching: false)
    }

    static func shouldAllowSessionTranslationLanguageSwitching(
        sessionOutputMode: SessionOutputMode,
        isSelectedTextTranslationFlow: Bool
    ) -> Bool {
        sessionOutputMode == .translation &&
            !isSelectedTextTranslationFlow
    }

    func toggleSessionTranslationTargetPicker() {
        guard overlayState.allowsSessionTranslationLanguageSwitching else { return }
        if overlayState.isSessionTranslationTargetPickerPresented {
            overlayState.dismissSessionTranslationTargetPicker()
        } else {
            overlayState.presentSessionTranslationTargetPicker()
        }
    }

    func selectSessionTranslationTargetLanguage(_ language: TranslationTargetLanguage) {
        guard overlayState.allowsSessionTranslationLanguageSwitching else { return }
        let previousLanguage = overlayState.sessionTranslationTargetLanguage
        let shouldRefreshDisplayedTranslation = shouldRefreshDisplayedTranslationAnswer(
            targetLanguage: language,
            previousLanguage: previousLanguage
        )

        if shouldRefreshDisplayedTranslation {
            overlayState.configureSessionTranslationTargetLanguage(language, allowsSwitching: true)
            overlayState.dismissSessionTranslationTargetPicker()
            refreshDisplayedTranslationAnswer(
                targetLanguage: language,
                previousLanguage: previousLanguage
            )
            return
        }

        if isSessionActive {
            sessionTranslationTargetLanguageOverride = language
        } else {
            UserDefaults.standard.set(language.rawValue, forKey: AppPreferenceKey.translationTargetLanguage)
        }
        overlayState.configureSessionTranslationTargetLanguage(language, allowsSwitching: true)
        overlayState.dismissSessionTranslationTargetPicker()
    }

    func dismissSessionTranslationTargetPicker() {
        overlayState.dismissSessionTranslationTargetPicker()
    }

    func shouldRefreshDisplayedTranslationAnswer(
        targetLanguage: TranslationTargetLanguage,
        previousLanguage: TranslationTargetLanguage?
    ) -> Bool {
        guard overlayState.displayMode == .answer,
              overlayState.sessionIconMode == .translation,
              overlayState.answerInteractionMode == .singleResult
        else {
            return false
        }
        guard targetLanguage != previousLanguage else { return false }
        return !overlayState.answerTranslationSourceText.isEmpty
    }

    func refreshDisplayedTranslationAnswer(
        targetLanguage: TranslationTargetLanguage,
        previousLanguage: TranslationTargetLanguage?
    ) {
        let sourceText = overlayState.answerTranslationSourceText
        guard !sourceText.isEmpty else { return }

        cancelPendingSelectedTextTranslationRefresh()
        overlayState.isRequesting = true

        let refreshID = UUID()
        selectedTextTranslationRefreshID = refreshID
        pendingSelectedTextTranslationRefreshTask = Task { @MainActor [weak self] in
            guard let self else { return }

            defer {
                if self.selectedTextTranslationRefreshID == refreshID {
                    self.overlayState.isRequesting = false
                    self.pendingSelectedTextTranslationRefreshTask = nil
                }
            }

            do {
                let translated = try await self.runTranslationPipeline(
                    text: sourceText,
                    targetLanguage: targetLanguage,
                    allowStrictRetry: true
                )
                guard !Task.isCancelled,
                      self.selectedTextTranslationRefreshID == refreshID,
                      self.overlayState.sessionTranslationTargetLanguage == targetLanguage
                else {
                    return
                }

                self.overlayState.replaceCurrentAnswer(
                    title: AppLocalization.localizedString("Translation"),
                    content: translated
                )
            } catch {
                guard !Task.isCancelled,
                      self.selectedTextTranslationRefreshID == refreshID
                else {
                    return
                }

                VoxtLog.translationWarning(
                    "Displayed translation refresh failed. sourceChars=\(sourceText.count), targetLanguage=\(targetLanguage.instructionName), error=\(error)"
                )

                if let previousLanguage {
                    if self.isSessionActive {
                        self.sessionTranslationTargetLanguageOverride = previousLanguage
                    } else {
                        UserDefaults.standard.set(
                            previousLanguage.rawValue,
                            forKey: AppPreferenceKey.translationTargetLanguage
                        )
                    }
                    self.overlayState.configureSessionTranslationTargetLanguage(
                        previousLanguage,
                        allowsSwitching: true
                    )
                }
            }
        }
    }

    func cancelPendingSelectedTextTranslationRefresh() {
        selectedTextTranslationRefreshID = UUID()
        pendingSelectedTextTranslationRefreshTask?.cancel()
        pendingSelectedTextTranslationRefreshTask = nil
        overlayState.isRequesting = false
    }

    func resolvedTranslationPrompt(
        targetLanguage: TranslationTargetLanguage,
        sourceText: String,
        strict: Bool,
        glossarySelectionPolicy: DictionaryGlossarySelectionPolicy?
    ) -> VariablePromptResolution {
        let basePrompt = TranslationPromptBuilder.build(
            systemPrompt: translationSystemPrompt,
            targetLanguage: targetLanguage,
            sourceText: sourceText,
            userMainLanguagePromptValue: userMainLanguagePromptValue,
            strict: strict
        )
        let glossary = dictionaryGlossaryText(
            for: sourceText,
            purpose: .translation,
            selectionPolicy: glossarySelectionPolicy
        )
        return VariablePromptResolution(
            content: basePrompt,
            dictionaryGlossary: glossary,
            delivery: deliveryForTemplate(
                translationSystemPrompt,
                variableTokens: ["{{SOURCE_TEXT}}"]
            ),
            promptProfile: "standard"
        )
    }

    func resolvedRewritePrompt(
        dictatedPrompt: String,
        sourceText: String,
        conversationHistory: [RewriteConversationPromptTurn],
        structuredAnswerOutput: Bool,
        directAnswerMode: Bool,
        forceNonEmptyAnswer: Bool,
        glossarySelectionPolicy: DictionaryGlossarySelectionPolicy?
    ) -> VariablePromptResolution {
        let resolved = RewritePromptBuilder.build(
            systemPrompt: rewriteSystemPrompt,
            dictatedPrompt: dictatedPrompt,
            sourceText: sourceText,
            conversationHistory: conversationHistory,
            structuredAnswerOutput: structuredAnswerOutput,
            directAnswerMode: directAnswerMode,
            forceNonEmptyAnswer: forceNonEmptyAnswer
        )
        let glossary = dictionaryGlossaryText(
            for: "\(dictatedPrompt)\n\(sourceText)",
            purpose: .rewrite,
            selectionPolicy: glossarySelectionPolicy
        )
        return VariablePromptResolution(
            content: resolved,
            dictionaryGlossary: glossary,
            delivery: deliveryForTemplate(
                rewriteSystemPrompt,
                variableTokens: ["{{DICTATED_PROMPT}}", "{{SOURCE_TEXT}}"]
            ),
            promptProfile: "standard"
        )
    }

    private func deliveryForTemplate(
        _ template: String,
        variableTokens: [String]
    ) -> VariablePromptDelivery {
        variableTokens.contains { template.contains($0) } ? .userMessage : .systemPrompt
    }

    private func translationModelLogDescriptor(for provider: TranslationModelProvider) -> String {
        switch provider {
        case .customLLM:
            return translationCustomLLMRepo
        case .localGGUF:
            return translationGGUFModelID.rawValue
        case .remoteLLM:
            let context = resolvedRemoteLLMContext(forTranslation: true)
            return "\(context.provider.rawValue):\(context.configuration.model)"
        }
    }

    func resolvedRewriteConversationPrompt(forceNonEmptyAnswer: Bool) -> String {
        let retryConstraint = forceNonEmptyAnswer
            ? """
            Retry rule:
            - A previous answer was empty, quoted-empty, or otherwise unusable.
            - This time, you must return a non-empty plain-text answer.
            - Do not return surrounding quotes.
            """
            : ""

        let base = """
        You are Voxt's follow-up voice conversation assistant.

        The previous conversation is provided as chat messages.
        Respond to the latest user message directly based on that conversation.

        Rules:
        1. Treat the latest user message as a follow-up to the previous assistant reply.
        2. If the user omits context with a short follow-up like “那大同呢”, infer the missing subject from the conversation history.
        3. Treat short confirmations such as “对”, “是”, or “yes” as answers to the latest assistant question, incorporate them, and continue the original task.
        4. Never repeat or paraphrase the latest assistant reply as the next answer, and never ask the same clarification twice.
        5. Do not ask the user to confirm a place, date, subject, or qualifier that already appears in the conversation.
        6. Ask a clarification question only when essential information is genuinely absent from the whole conversation.
        7. If current information cannot be verified because live lookup is unavailable, state that limitation directly instead of treating the request as ambiguous.
        8. Return plain text only.
        9. Do not return JSON, field names, markdown, or surrounding quotes.
        10. Do not return an empty string.
        """

        let prompt = [base, retryConstraint]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")

        return prompt
    }

    private func trimmedRewriteConversationHistory(
        _ turns: [RewriteConversationPromptTurn]
    ) -> [RewriteConversationPromptTurn] {
        guard !turns.isEmpty else { return [] }
        let budget = (maxTurns: 4, maxCharacters: 1_600)

        let candidates = Array(turns.suffix(budget.maxTurns)).reversed()
        var selectedReversed: [RewriteConversationPromptTurn] = []
        var consumedCharacters = 0

        for turn in candidates {
            let turnCharacters = turn.modelUserMessage.count + turn.resultContent.count
            let separatorCost = selectedReversed.isEmpty ? 0 : 4
            let nextCount = consumedCharacters + separatorCost + turnCharacters

            if !selectedReversed.isEmpty && nextCount > budget.maxCharacters {
                break
            }

            selectedReversed.append(turn)
            consumedCharacters = nextCount
        }

        return selectedReversed.reversed()
    }

    func shouldRetryStructuredRewriteAnswer(for text: String, dictatedPrompt: String) -> Bool {
        RewriteAnswerContentNormalizer.isUnusableStructuredAnswer(
            text,
            dictatedPrompt: dictatedPrompt
        )
    }

    func serializedRewriteAnswerPayload(_ payload: RewriteAnswerPayload) -> String? {
        let object: [String: String] = [
            "title": payload.title,
            "content": payload.content
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else {
            return nil
        }
        return text
    }

    func runTranslationPipeline(
        text: String,
        targetLanguage: TranslationTargetLanguage,
        allowStrictRetry: Bool
    ) async throws -> String {
        let stages = TranslationSessionPipelineBuilder.makeTranslationStages(
            targetLanguage: targetLanguage,
            allowStrictRetry: allowStrictRetry,
            translate: { [weak self] value, targetLanguage in
                guard let self else { throw CancellationError() }
                return try await self.translateText(value, targetLanguage: targetLanguage)
            },
            shouldRetry: { [weak self] source, result in
                guard let self else { return false }
                if self.looksUntranslated(source: source, result: result) {
                    VoxtLog.translationWarning("Selected text translation first-pass looks untranslated. Retrying with strict translation prompt.")
                    return true
                }
                return false
            },
            strictTranslate: { [weak self] value, targetLanguage in
                guard let self else { throw CancellationError() }
                return try await self.translateTextStrict(value, targetLanguage: targetLanguage)
            }
        )
        let runner = SessionPipelineRunner(stages: stages)
        let initial = SessionPipelineContext(originalText: text, workingText: text)
        let result = try await runner.run(initial: initial)
        return result.workingText
    }

    func runRewritePipeline(
        dictatedText: String,
        selectedSourceText: String,
        conversationHistory: [RewriteConversationPromptTurn],
        structuredAnswerOutput: Bool,
        previousConversationResponseID: String? = nil,
        onProgress: (@Sendable (String) -> Void)? = nil,
        onResponseID: ((String) -> Void)? = nil
    ) async throws -> String {
        let stages = TranslationSessionPipelineBuilder.makeRewriteStages(
            sourceText: selectedSourceText,
            rewrite: { [weak self] dictatedPrompt, sourceText in
                guard let self else { throw CancellationError() }
                return try await self.rewriteText(
                    dictatedPrompt: dictatedPrompt,
                    sourceText: sourceText,
                    conversationHistory: conversationHistory,
                    structuredAnswerOutput: structuredAnswerOutput,
                    previousConversationResponseID: previousConversationResponseID,
                    onProgress: onProgress,
                    onResponseID: onResponseID
                )
            }
        )
        let runner = SessionPipelineRunner(stages: stages)
        let initial = SessionPipelineContext(originalText: dictatedText, workingText: dictatedText)
        let result = try await runner.run(initial: initial)
        return result.workingText
    }

    func presentRewriteConversationPseudoStreamingPreview(
        content: String,
        sessionID: UUID
    ) async {
        let normalized = RewriteAnswerContentNormalizer.normalizePlainTextAnswer(content)
        guard !normalized.isEmpty else { return }

        // Large answers should appear immediately; dozens of staged updates stall the
        // overlay and spinner on the main thread without adding useful feedback.
        guard normalized.count < 360 else {
            guard shouldHandleCallbacks(for: sessionID) else { return }
            presentRewriteConversationStreamingPreview(content: normalized)
            return
        }

        let characters = Array(normalized)
        let stageCount = min(4, max(2, Int(ceil(Double(characters.count) / 96.0))))
        let chunkSize = max(24, Int(ceil(Double(characters.count) / Double(stageCount))))

        var rendered = ""
        var index = 0
        while index < characters.count {
            guard shouldHandleCallbacks(for: sessionID) else { return }
            let upperBound = min(index + chunkSize, characters.count)
            rendered += String(characters[index..<upperBound])
            presentRewriteConversationStreamingPreview(content: rendered)
            index = upperBound
            guard index < characters.count else { continue }
            do {
                try await Task.sleep(for: .milliseconds(45))
            } catch {
                return
            }
        }
    }

    func looksUntranslated(source: String, result: String) -> Bool {
        let sourceTrimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let resultTrimmed = result.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sourceTrimmed.isEmpty, !resultTrimmed.isEmpty else { return false }
        return sourceTrimmed.caseInsensitiveCompare(resultTrimmed) == .orderedSame
    }
}
