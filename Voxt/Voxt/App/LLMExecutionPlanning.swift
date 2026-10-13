// LLMExecutionPlanning.swift
// Provides LLMExecution Planning for app lifecycle and routing.

import Foundation

extension AppDelegate {
    func resolvedTranscriptionEnhancementTextSelection() -> FeatureModelSelectionID.TextSelection? {
        transcriptionFeatureSettings.llmEnabled
            ? transcriptionFeatureSettings.llmSelectionID.textSelection
            : nil
    }

    func resolvedTranscriptionEnhancementLocalRepo() -> String? {
        guard case .localLLM(let repo) = resolvedTranscriptionEnhancementTextSelection() else {
            return enhancementMode == .customLLM ? customLLMManager.currentModelRepo : nil
        }
        return CustomLLMModelManager.canonicalModelRepo(repo)
    }

    func resolvedTranscriptionEnhancementProvider(
        providerOverride: LLMExecutionProvider? = nil
    ) -> LLMExecutionProvider? {
        if let providerOverride {
            return providerOverride
        }

        switch enhancementMode {
        case .off:
            return nil
        case .appleIntelligence:
            return .appleIntelligence
        case .customLLM, .remoteLLM:
            break
        }

        switch resolvedTranscriptionEnhancementTextSelection() {
        case .appleIntelligence:
            return .appleIntelligence
        case .localLLM(let repo):
            return .customLLM(repo: CustomLLMModelManager.canonicalModelRepo(repo))
        case .remoteLLM(let provider):
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            return .remote(provider: provider, configuration: configuration)
        case .none:
            switch enhancementMode {
            case .off:
                return nil
            case .appleIntelligence:
                return .appleIntelligence
            case .customLLM:
                return .customLLM(repo: customLLMManager.currentModelRepo)
            case .remoteLLM:
                let context = resolvedRemoteLLMContext(forTranslation: false)
                return .remote(provider: context.provider, configuration: context.configuration)
            }
        }
    }

    func buildEnhancementExecutionPlan(
        rawText: String,
        promptResolution: EnhancementPromptResolution,
        providerOverride: LLMExecutionProvider? = nil,
        executionStrategy: TaskLLMExecutionStrategy
    ) -> LLMExecutionPlan? {
        guard let provider = resolvedTranscriptionEnhancementProvider(providerOverride: providerOverride) else {
            return nil
        }

        let delivery: LLMExecutionDelivery
        switch promptResolution.delivery {
        case .systemPrompt:
            delivery = .systemPrompt
        case .userMessage:
            delivery = .userMessage
        case .skipEnhancement:
            return nil
        }

        return LLMExecutionPlan(
            task: .enhancement(rawText: rawText),
            provider: provider,
            delivery: delivery,
            promptContent: promptResolution.content,
            fallbackText: rawText,
            executionStrategy: executionStrategy,
            outputTokenBudgetHint: executionStrategy.outputTokenBudgetHint,
            contextBlocks: compactLLMContextBlocks([
                LLMContextBlock(
                    kind: .input,
                    title: "Raw transcription",
                    content: rawText,
                    isStablePrefixCandidate: false
                ),
                glossaryContextBlock(promptResolution.dictionaryGlossary, purpose: .enhancement)
            ]),
            conversationHistory: [],
            previousResponseID: nil,
            responseFormat: nil
        )
    }

    func buildTranslationExecutionPlan(
        sourceText: String,
        targetLanguage: TranslationTargetLanguage,
        promptResolution: VariablePromptResolution,
        modelProvider: TranslationModelProvider,
        providerOverride: LLMExecutionProvider? = nil,
        executionStrategy: TaskLLMExecutionStrategy
    ) -> LLMExecutionPlan? {
        let provider: LLMExecutionProvider
        if let providerOverride {
            provider = providerOverride
        } else {
            switch modelProvider {
            case .customLLM:
                provider = .customLLM(repo: translationCustomLLMRepo)
            case .localGGUF:
                provider = .localGGUF(modelID: translationGGUFModelID)
            case .remoteLLM:
                let context = resolvedRemoteLLMContext(forTranslation: true)
                provider = .remote(provider: context.provider, configuration: context.configuration)
            }
        }

        return LLMExecutionPlan(
            task: .translation(sourceText: sourceText, targetLanguage: targetLanguage),
            provider: provider,
            delivery: llmExecutionDelivery(for: promptResolution.delivery),
            promptContent: promptResolution.content,
            fallbackText: sourceText,
            executionStrategy: executionStrategy,
            outputTokenBudgetHint: executionStrategy.outputTokenBudgetHint,
            contextBlocks: compactLLMContextBlocks([
                LLMContextBlock(
                    kind: .input,
                    title: "Source text",
                    content: sourceText,
                    isStablePrefixCandidate: false
                ),
                glossaryContextBlock(promptResolution.dictionaryGlossary, purpose: .translation)
            ]),
            conversationHistory: [],
            previousResponseID: nil,
            responseFormat: nil
        )
    }

    func buildRewriteExecutionPlan(
        dictatedPrompt: String,
        sourceText: String,
        promptResolution: VariablePromptResolution,
        modelProvider: RewriteModelProvider,
        conversationHistory: [RewriteConversationPromptTurn],
        previousResponseID: String?,
        structuredAnswerOutput: Bool,
        providerOverride: LLMExecutionProvider? = nil,
        executionStrategy: TaskLLMExecutionStrategy
    ) -> LLMExecutionPlan {
        let provider: LLMExecutionProvider
        if let providerOverride {
            provider = providerOverride
        } else {
            switch modelProvider {
            case .customLLM:
                provider = .customLLM(repo: rewriteCustomLLMRepo)
            case .remoteLLM:
                let context = resolvedRemoteLLMContext(forRewrite: true)
                provider = .remote(provider: context.provider, configuration: context.configuration)
            }
        }

        let responseFormat: RemoteLLMRuntimeClient.OpenAICompatibleResponseFormat?
        if case .remote(let remoteProvider, _) = provider,
           structuredAnswerOutput,
           remoteProvider == .deepseek || (
               remoteProvider.usesResponsesAPI &&
               LLMProviderCapabilityRegistry.capabilities(for: remoteProvider).supportsResponseFormat
           ) {
            responseFormat = .jsonObject
        } else {
            responseFormat = nil
        }

        return LLMExecutionPlan(
            task: .rewrite(
                dictatedPrompt: dictatedPrompt,
                sourceText: sourceText,
                structuredAnswerOutput: structuredAnswerOutput
            ),
            provider: provider,
            delivery: llmExecutionDelivery(for: promptResolution.delivery),
            promptContent: promptResolution.content,
            fallbackText: sourceText,
            executionStrategy: executionStrategy,
            outputTokenBudgetHint: executionStrategy.outputTokenBudgetHint,
            contextBlocks: compactLLMContextBlocks([
                LLMContextBlock(
                    kind: .input,
                    title: "Spoken instruction",
                    content: dictatedPrompt,
                    isStablePrefixCandidate: false
                ),
                sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? nil
                    : LLMContextBlock(
                        kind: .input,
                        title: "Selected source text",
                        content: sourceText,
                        isStablePrefixCandidate: false
                    ),
                sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? LLMContextBlock(
                        kind: .metadata,
                        title: "Direct-answer runtime context",
                        content: RewriteDirectAnswerRuntimeGuidance.content(
                            liveInformationAccess: rewriteProviderHasLiveInformationAccess(provider)
                        ),
                        isStablePrefixCandidate: false
                    )
                    : nil,
                glossaryContextBlock(promptResolution.dictionaryGlossary, purpose: .rewrite),
                rewriteConversationContextBlock(conversationHistory)
            ]),
            conversationHistory: conversationHistory,
            previousResponseID: previousResponseID,
            responseFormat: responseFormat
        )
    }

    func executeLLMExecutionPlan(
        _ plan: LLMExecutionPlan,
        onPartialText: (@Sendable (String) -> Void)? = nil,
        onResponseID: ((String) -> Void)? = nil
    ) async throws -> String {
        let compiledRequest = LLMExecutionPlanCompiler.compile(plan)
        let executionStartedAt = Date()
        VoxtLog.llmInfo(
            "LLM execution plan. task=\(plan.taskLabel), provider=\(llmExecutionProviderLabel(plan.provider)), delivery=\(String(describing: plan.delivery)), promptChars=\(plan.promptCharacterCount), inputChars=\(plan.primaryInputCharacterCount), blocks=\(plan.contextBlocks.count), strategy=\(plan.executionStrategy.logLabel)"
        )

        let output: String
        switch plan.executionStrategy.mode {
        case .singlePass:
            output = try await executeSingleLLMExecutionPlan(
                plan,
                compiledRequest: compiledRequest,
                onPartialText: onPartialText,
                onResponseID: onResponseID
            )
        case .segmented:
            output = try await executeSegmentedLLMExecutionPlan(
                plan,
                onPartialText: onPartialText,
                onResponseID: onResponseID
            )
        }
        recordSessionLLMExecutionTiming(
            taskLabel: plan.taskLabel,
            provider: plan.provider,
            startedAt: executionStartedAt
        )
        let sanitized = LLMVisibleOutputSanitizer.sanitize(
            output,
            fallbackText: plan.fallbackText,
            taskKind: visibleOutputSanitizerTaskKind(for: plan.task)
        )
        if sanitized.didFallback || sanitized.didExtractFinalOutput || sanitized.didRemoveProcessText {
            VoxtLog.llmWarning(
                "LLM visible output sanitized. task=\(plan.taskLabel), fallback=\(sanitized.didFallback), extractedFinal=\(sanitized.didExtractFinalOutput), removedProcess=\(sanitized.didRemoveProcessText), inputChars=\(plan.primaryInputCharacterCount), outputChars=\(output.count), finalChars=\(sanitized.text.count)"
            )
        }
        return sanitized.text
    }

    private func visibleOutputSanitizerTaskKind(for task: LLMExecutionTaskPayload) -> LLMVisibleOutputSanitizer.TaskKind {
        switch task {
        case .enhancement:
            return .enhancement
        case .translation:
            return .translation
        case .rewrite:
            return .rewrite
        case .transcriptSummary:
            return .generic
        }
    }

    func llmProviderModelCapabilities(for provider: LLMExecutionProvider) -> LLMProviderModelCapabilities {
        switch provider {
        case .appleIntelligence:
            return .unknown
        case .customLLM:
            return .unknown
        case .localGGUF:
            return .unknown
        case .remote:
            return .unknown
        }
    }

    private func executeSingleLLMExecutionPlan(
        _ plan: LLMExecutionPlan,
        compiledRequest: LLMCompiledRequest,
        onPartialText: (@Sendable (String) -> Void)? = nil,
        onResponseID: ((String) -> Void)? = nil
    ) async throws -> String {
        switch plan.provider {
        case .appleIntelligence:
            guard let enhancer else { return plan.fallbackText }
            return try await enhancer.executeCompiledRequest(compiledRequest)

        case .customLLM(let repo):
            return try await customLLMManager.executeCompiledRequest(
                compiledRequest,
                repo: repo,
                onPartialText: onPartialText
            )

        case .localGGUF(let modelID):
            return try await ggufTranslationModelManager.executeCompiledRequest(
                compiledRequest,
                modelID: modelID,
                onPartialText: onPartialText
            )

        case .remote(let provider, let configuration):
            return try await RemoteLLMRuntimeClient().executeCompiledRequest(
                compiledRequest,
                provider: provider,
                configuration: configuration,
                onPartialText: onPartialText,
                onResponseID: onResponseID
            )
        }
    }

    private func executeSegmentedLLMExecutionPlan(
        _ plan: LLMExecutionPlan,
        onPartialText: (@Sendable (String) -> Void)? = nil,
        onResponseID: ((String) -> Void)? = nil
    ) async throws -> String {
        let segments = segmentPrimaryInput(
            of: plan.task,
            limit: plan.executionStrategy.segmentationCharacterLimit ?? TaskLLMStrategyResolver.longTextThreshold
        )
        guard segments.count > 1 else {
            return try await executeSingleLLMExecutionPlan(
                plan,
                compiledRequest: LLMExecutionPlanCompiler.compile(plan),
                onPartialText: onPartialText,
                onResponseID: onResponseID
            )
        }

        var outputs: [String] = []
        outputs.reserveCapacity(segments.count)
        for segment in segments {
            let segmentedPlan = replacePrimaryInput(of: plan, with: segment)
            let segmentOutput = try await executeSingleLLMExecutionPlan(
                segmentedPlan,
                compiledRequest: LLMExecutionPlanCompiler.compile(segmentedPlan),
                onPartialText: nil,
                onResponseID: onResponseID
            )
            outputs.append(segmentOutput)
        }
        let joined = outputs.joined()
        onPartialText?(joined)
        return joined
    }

    private func segmentPrimaryInput(
        of task: LLMExecutionTaskPayload,
        limit: Int
    ) -> [String] {
        let text: String
        switch task {
        case .enhancement(let rawText):
            text = rawText
        case .translation(let sourceText, _):
            text = sourceText
        case .rewrite(_, let sourceText, _):
            guard !sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
            text = sourceText
        case .transcriptSummary(let transcript, _):
            return [transcript]
        }
        return TextSegmentationSupport.segment(text: text, limit: limit)
    }

    private func replacePrimaryInput(
        of plan: LLMExecutionPlan,
        with segment: String
    ) -> LLMExecutionPlan {
        let segmentedTask: LLMExecutionTaskPayload
        switch plan.task {
        case .enhancement:
            segmentedTask = .enhancement(rawText: segment)
        case .translation(_, let targetLanguage):
            segmentedTask = .translation(sourceText: segment, targetLanguage: targetLanguage)
        case .rewrite(let dictatedPrompt, _, let structuredAnswerOutput):
            segmentedTask = .rewrite(
                dictatedPrompt: dictatedPrompt,
                sourceText: segment,
                structuredAnswerOutput: structuredAnswerOutput
            )
        case .transcriptSummary(_, let request):
            segmentedTask = .transcriptSummary(
                transcript: segment,
                request: request
            )
        }

        let segmentedBlocks = plan.contextBlocks.map { block -> LLMContextBlock in
            guard block.kind == .input else { return block }
            switch plan.task {
            case .enhancement:
                return LLMContextBlock(
                    kind: .input,
                    title: "Raw transcription",
                    content: segment,
                    isStablePrefixCandidate: block.isStablePrefixCandidate
                )
            case .translation:
                return LLMContextBlock(
                    kind: .input,
                    title: "Source text",
                    content: segment,
                    isStablePrefixCandidate: block.isStablePrefixCandidate
                )
            case .rewrite(let dictatedPrompt, _, _):
                if block.title == "Selected source text" {
                    return LLMContextBlock(
                        kind: .input,
                        title: block.title,
                        content: segment,
                        isStablePrefixCandidate: block.isStablePrefixCandidate
                    )
                }
                if block.title == "Spoken instruction" {
                    return LLMContextBlock(
                        kind: .input,
                        title: block.title,
                        content: dictatedPrompt,
                        isStablePrefixCandidate: block.isStablePrefixCandidate
                    )
                }
                return block
            case .transcriptSummary:
                return LLMContextBlock(
                    kind: .input,
                    title: "Transcript",
                    content: segment,
                    isStablePrefixCandidate: block.isStablePrefixCandidate
                )
            }
        }

        return LLMExecutionPlan(
            task: segmentedTask,
            provider: plan.provider,
            delivery: plan.delivery,
            promptContent: plan.promptContent,
            fallbackText: segment,
            executionStrategy: plan.executionStrategy,
            outputTokenBudgetHint: plan.outputTokenBudgetHint,
            contextBlocks: segmentedBlocks,
            conversationHistory: plan.conversationHistory,
            previousResponseID: plan.previousResponseID,
            responseFormat: plan.responseFormat
        )
    }

    private func llmExecutionDelivery(for delivery: VariablePromptDelivery) -> LLMExecutionDelivery {
        switch delivery {
        case .systemPrompt:
            return .systemPrompt
        case .userMessage:
            return .userMessage
        }
    }

    private func compactLLMContextBlocks(_ blocks: [LLMContextBlock?]) -> [LLMContextBlock] {
        blocks.compactMap { block in
            guard let block else { return nil }
            return block.trimmedContent.isEmpty ? nil : block
        }
    }

    private func glossaryContextBlock(
        _ glossary: String?,
        purpose: DictionaryGlossaryPurpose
    ) -> LLMContextBlock? {
        guard let body = DictionaryGlossaryPromptComposer.body(glossary: glossary, purpose: purpose) else {
            return nil
        }

        return LLMContextBlock(
            kind: .glossary,
            title: "Dictionary Guidance",
            content: body,
            isStablePrefixCandidate: true
        )
    }

    private func rewriteConversationContextBlock(
        _ turns: [RewriteConversationPromptTurn]
    ) -> LLMContextBlock? {
        guard !turns.isEmpty else { return nil }

        let segments = turns.compactMap { turn -> String? in
            let userPrompt = turn.modelUserMessage
            let resultContent = turn.resultContent.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !userPrompt.isEmpty || !resultContent.isEmpty else {
                return nil
            }

            var lines: [String] = []
            if !userPrompt.isEmpty {
                lines.append("User: \(userPrompt)")
            }
            if !resultContent.isEmpty {
                lines.append("Assistant: \(resultContent)")
            }
            return lines.joined(separator: "\n")
        }

        guard !segments.isEmpty else { return nil }
        return LLMContextBlock(
            kind: .conversation,
            title: "Previous conversation",
            content: segments.joined(separator: "\n\n"),
            isStablePrefixCandidate: false
        )
    }

    private func rewriteProviderHasLiveInformationAccess(_ provider: LLMExecutionProvider) -> Bool {
        guard case .remote(let remoteProvider, let configuration) = provider else { return false }
        return configuration.searchEnabled && remoteProvider.supportsHostedSearch
    }
    private func llmExecutionProviderLabel(_ provider: LLMExecutionProvider) -> String {
        switch provider {
        case .appleIntelligence:
            return "appleIntelligence"
        case .customLLM(let repo):
            return "customLLM(\(repo))"
        case .localGGUF(let modelID):
            return "localGGUF(\(modelID.rawValue))"
        case .remote(let provider, let configuration):
            return "remote(\(provider.rawValue):\(configuration.model))"
        }
    }

    private func recordSessionLLMExecutionTiming(
        taskLabel: String,
        provider: LLMExecutionProvider,
        startedAt: Date
    ) {
        let diagnostics = provider.isCustomLLM ? customLLMManager.lastRunDiagnostics : nil
        let completedAt: Date
        let firstChunkAt: Date?
        if let diagnostics {
            completedAt = startedAt.addingTimeInterval(Double(diagnostics.totalElapsedMs) / 1000)
            firstChunkAt = diagnostics.overallFirstChunkMs.map {
                startedAt.addingTimeInterval(Double($0) / 1000)
            }
        } else {
            completedAt = Date()
            firstChunkAt = nil
        }

        sessionLLMExecutionTimings.append(
            SessionLLMExecutionTiming(
                taskLabel: taskLabel,
                providerLabel: llmExecutionProviderLabel(provider),
                startedAt: startedAt,
                firstChunkAt: firstChunkAt,
                completedAt: completedAt,
                diagnostics: diagnostics
            )
        )
    }
}

private extension LLMExecutionProvider {
    var isCustomLLM: Bool {
        if case .customLLM = self {
            return true
        }
        return false
    }
}
