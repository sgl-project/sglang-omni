// LLMSmokeTest.swift
// Provides LLMSmoke Test for app lifecycle and routing.

import Foundation
import Darwin

extension AppDelegate {
    private enum LLMSmokeTask: String {
        case enhancement
        case translation
        case rewrite
    }

    private struct LLMSmokeIterationResult {
        let iteration: Int
        let output: String
        let elapsedMs: Int
        let diagnostics: CustomLLMRunDiagnostics?
    }

    func maybeRunLLMSmokeAndTerminate() -> Bool {
        let environment = ProcessInfo.processInfo.environment
        let smokeTask = LLMSmokeTask(
            rawValue: environment["VOXT_LLM_SMOKE_TASK"]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased() ?? "enhancement"
        ) ?? .enhancement
        let enhancementText = environment["VOXT_LLM_SMOKE_ENHANCEMENT_TEXT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let translationText = environment["VOXT_LLM_SMOKE_TRANSLATION_TEXT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let rewritePrompt = environment["VOXT_LLM_SMOKE_REWRITE_PROMPT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let rewriteSourceText = environment["VOXT_LLM_SMOKE_REWRITE_SOURCE_TEXT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        let shouldRun: Bool = switch smokeTask {
        case .enhancement:
            !(enhancementText ?? "").isEmpty
        case .translation:
            !(translationText ?? "").isEmpty
        case .rewrite:
            !(rewritePrompt ?? "").isEmpty
        }
        guard shouldRun else {
            return false
        }

        let requestedRepo = environment["VOXT_LLM_SMOKE_LOCAL_REPO"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let repo = requestedRepo.map(CustomLLMModelManager.canonicalModelRepo(_:))
            ?? customLLMManager.currentModelRepo
        let iterations = max(
            1,
            Int(environment["VOXT_LLM_SMOKE_ITERATIONS"] ?? "") ?? 1
        )
        let shouldPrewarm = ["1", "true", "yes"].contains(
            (environment["VOXT_LLM_SMOKE_PREWARM"] ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
        )
        let prefillStepOverride = Int(environment["VOXT_LLM_SMOKE_PREFILL_STEP"] ?? "")
        let translationTargetLanguage = resolvedSmokeTranslationTargetLanguage(
            environment["VOXT_LLM_SMOKE_TARGET_LANGUAGE"]
        ) ?? .english

        Task { @MainActor [weak self] in
            guard let self else { return }
            let startedAt = Date()
            do {
                guard self.customLLMManager.canAttemptInference(repo: repo) else {
                    throw NSError(
                        domain: "Voxt.LLMSmoke",
                        code: 404,
                        userInfo: [NSLocalizedDescriptionKey: "Local model is not downloaded: \(repo)"]
                    )
                }

                let plan: LLMExecutionPlan
                switch smokeTask {
                case .enhancement:
                    let rawInput = enhancementText ?? ""
                    let strategy = TaskLLMStrategyResolver.resolve(
                        taskKind: .transcriptionEnhancement,
                        rawText: rawInput,
                        promptCharacterCount: 0,
                        baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.enhancement.selectionPolicy,
                        capabilities: self.llmProviderModelCapabilities(for: .customLLM(repo: repo))
                    )
                    let promptResolution = self.resolvedEnhancementPrompt(
                        rawTranscription: rawInput,
                        glossarySelectionPolicy: strategy.glossarySelectionPolicy
                    )
                    guard let builtPlan = self.buildEnhancementExecutionPlan(
                        rawText: rawInput,
                        promptResolution: promptResolution,
                        providerOverride: .customLLM(repo: repo),
                        executionStrategy: strategy
                    ) else {
                        throw NSError(
                            domain: "Voxt.LLMSmoke",
                            code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "Unable to build enhancement execution plan."]
                        )
                    }
                    plan = builtPlan
                case .translation:
                    let sourceText = translationText ?? ""
                    let strategy = TaskLLMStrategyResolver.resolve(
                        taskKind: .translation,
                        rawText: sourceText,
                        promptCharacterCount: 0,
                        baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.translation.selectionPolicy,
                        capabilities: self.llmProviderModelCapabilities(for: .customLLM(repo: repo))
                    )
                    let promptResolution = self.resolvedTranslationPrompt(
                        targetLanguage: translationTargetLanguage,
                        sourceText: sourceText,
                        strict: false,
                        glossarySelectionPolicy: strategy.glossarySelectionPolicy
                    )
                    guard let builtPlan = self.buildTranslationExecutionPlan(
                        sourceText: sourceText,
                        targetLanguage: translationTargetLanguage,
                        promptResolution: promptResolution,
                        modelProvider: .customLLM,
                        providerOverride: .customLLM(repo: repo),
                        executionStrategy: strategy
                    ) else {
                        throw NSError(
                            domain: "Voxt.LLMSmoke",
                            code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "Unable to build translation execution plan."]
                        )
                    }
                    plan = builtPlan
                case .rewrite:
                    let dictatedPrompt = rewritePrompt ?? ""
                    let strategy = TaskLLMStrategyResolver.resolve(
                        taskKind: .rewrite,
                        rawText: rewriteSourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? dictatedPrompt : rewriteSourceText,
                        promptCharacterCount: 0,
                        baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.rewrite.selectionPolicy,
                        capabilities: self.llmProviderModelCapabilities(for: .customLLM(repo: repo))
                    )
                    let promptResolution = self.resolvedRewritePrompt(
                        dictatedPrompt: dictatedPrompt,
                        sourceText: rewriteSourceText,
                        conversationHistory: [],
                        structuredAnswerOutput: false,
                        directAnswerMode: rewriteSourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                        forceNonEmptyAnswer: false,
                        glossarySelectionPolicy: strategy.glossarySelectionPolicy
                    )
                    plan = self.buildRewriteExecutionPlan(
                        dictatedPrompt: dictatedPrompt,
                        sourceText: rewriteSourceText,
                        promptResolution: promptResolution,
                        modelProvider: .customLLM,
                        conversationHistory: [],
                        previousResponseID: nil,
                        structuredAnswerOutput: false,
                        providerOverride: .customLLM(repo: repo),
                        executionStrategy: strategy
                    )
                }

                let compiledRequest = LLMExecutionPlanCompiler.compile(plan)
                let originalTuning = self.customLLMManager.generationTuning
                self.customLLMManager.generationTuning = CustomLLMGenerationTuning(
                    prefillStepSizeOverride: prefillStepOverride,
                    maxTokensOverride: nil
                )
                defer {
                    self.customLLMManager.generationTuning = originalTuning
                }
                if shouldPrewarm {
                    try await self.customLLMManager.prewarmModel(repo: repo)
                }

                var iterationResults: [LLMSmokeIterationResult] = []
                iterationResults.reserveCapacity(iterations)
                for iteration in 1...iterations {
                    let iterationStartedAt = Date()
                    let output = try await self.customLLMManager.executeCompiledRequest(
                        compiledRequest,
                        repo: repo
                    )
                    let elapsedMs = Int(Date().timeIntervalSince(iterationStartedAt) * 1000)
                    iterationResults.append(
                        LLMSmokeIterationResult(
                            iteration: iteration,
                            output: output,
                            elapsedMs: elapsedMs,
                            diagnostics: self.customLLMManager.lastRunDiagnostics
                        )
                    )
                }
                let totalElapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                let finalResult = iterationResults.last

                print("[VOXT_SMOKE] task=\(smokeTask.rawValue)")
                print("[VOXT_SMOKE] repo=\(repo)")
                print("[VOXT_SMOKE] iterations=\(iterations)")
                print("[VOXT_SMOKE] prewarm=\(shouldPrewarm)")
                print("[VOXT_SMOKE] prefillStepOverride=\(prefillStepOverride.map(String.init) ?? "auto")")
                print("[VOXT_SMOKE] delivery=\(String(describing: plan.delivery))")
                print("[VOXT_SMOKE] promptChars=\(plan.promptCharacterCount)")
                print("[VOXT_SMOKE] inputChars=\(plan.primaryInputCharacterCount)")
                print("[VOXT_SMOKE] instructionChars=\(compiledRequest.instructions.count)")
                print("[VOXT_SMOKE] requestPromptChars=\(compiledRequest.prompt.count)")
                print("[VOXT_SMOKE] overallElapsedMs=\(totalElapsedMs)")
                if let finalResult {
                    print("[VOXT_SMOKE] outputChars=\(finalResult.output.count)")
                }

                for result in iterationResults {
                    print("[VOXT_SMOKE][iteration \(result.iteration)] elapsedMs=\(result.elapsedMs)")
                    if let diagnostics = result.diagnostics {
                        printIterationDiagnostics(diagnostics, iteration: result.iteration)
                    }
                }

                printAggregateMetrics(iterationResults)
                print("[VOXT_SMOKE][instructions]")
                print(compiledRequest.instructions)
                print("[VOXT_SMOKE][prompt]")
                print(compiledRequest.prompt)
                print("[VOXT_SMOKE][output]")
                print(finalResult?.output ?? "")
                fflush(stdout)
                self.requestApplicationTermination(exitStatus: 0)
            } catch {
                print("[VOXT_SMOKE][error] \(error.localizedDescription)")
                fflush(stdout)
                self.requestApplicationTermination(exitStatus: 1)
            }
        }

        return true
    }

    private func printIterationDiagnostics(
        _ diagnostics: CustomLLMRunDiagnostics,
        iteration: Int
    ) {
        let prefix = "[VOXT_SMOKE][iteration \(iteration)]"
        print("\(prefix) containerSource=\(diagnostics.containerLoadSource.rawValue)")
        print("\(prefix) containerLoadMs=\(diagnostics.containerLoadMs)")
        print("\(prefix) setupMs=\(diagnostics.setupMs)")
        print("\(prefix) modelElapsedMs=\(diagnostics.modelElapsedMs)")
        print("\(prefix) totalElapsedMs=\(diagnostics.totalElapsedMs)")
        print("\(prefix) firstChunkMs=\(diagnostics.firstChunkMs.map(String.init) ?? "n/a")")
        print("\(prefix) overallFirstChunkMs=\(diagnostics.overallFirstChunkMs.map(String.init) ?? "n/a")")
        print("\(prefix) promptTokens=\(diagnostics.promptTokens.map(String.init) ?? "n/a")")
        print("\(prefix) completionTokens=\(diagnostics.completionTokens.map(String.init) ?? "n/a")")
        print("\(prefix) prefillMs=\(diagnostics.prefillMs.map(String.init) ?? "n/a")")
        print("\(prefix) generationMs=\(diagnostics.generationMs.map(String.init) ?? "n/a")")
        print("\(prefix) modelOverheadMs=\(diagnostics.modelOverheadMs.map(String.init) ?? "n/a")")
        print("\(prefix) totalOverheadMs=\(diagnostics.totalOverheadMs.map(String.init) ?? "n/a")")
    }

    private func printAggregateMetrics(_ results: [LLMSmokeIterationResult]) {
        guard !results.isEmpty else { return }
        let elapsedValues = results.map(\.elapsedMs)
        print("[VOXT_SMOKE][summary] avgElapsedMs=\(average(elapsedValues))")
        print("[VOXT_SMOKE][summary] minElapsedMs=\(elapsedValues.min() ?? 0)")
        print("[VOXT_SMOKE][summary] maxElapsedMs=\(elapsedValues.max() ?? 0)")

        let hotResults = Array(results.dropFirst())
        if !hotResults.isEmpty {
            print("[VOXT_SMOKE][summary] hotAvgElapsedMs=\(average(hotResults.map(\.elapsedMs)))")
        }

        printAverageSummary(results, label: "containerLoadMs") { $0.containerLoadMs }
        printAverageSummary(results, label: "setupMs") { $0.setupMs }
        printOptionalAverageSummary(results, label: "firstChunkMs") { $0.firstChunkMs }
        printOptionalAverageSummary(results, label: "overallFirstChunkMs") { $0.overallFirstChunkMs }
        printOptionalAverageSummary(results, label: "prefillMs") { $0.prefillMs }
        printOptionalAverageSummary(results, label: "generationMs") { $0.generationMs }
        printOptionalAverageSummary(results, label: "modelOverheadMs") { $0.modelOverheadMs }
        printOptionalAverageSummary(results, label: "totalOverheadMs") { $0.totalOverheadMs }
        printOptionalAverageSummary(results, label: "promptTokens") { $0.promptTokens }
        printOptionalAverageSummary(results, label: "completionTokens") { $0.completionTokens }
    }

    private func printAverageSummary(
        _ results: [LLMSmokeIterationResult],
        label: String,
        value: (CustomLLMRunDiagnostics) -> Int
    ) {
        let values = results.compactMap(\.diagnostics).map(value)
        guard !values.isEmpty else { return }
        print("[VOXT_SMOKE][summary] avg\(label.prefix(1).uppercased())\(label.dropFirst())=\(average(values))")
    }

    private func printOptionalAverageSummary(
        _ results: [LLMSmokeIterationResult],
        label: String,
        value: (CustomLLMRunDiagnostics) -> Int?
    ) {
        let values = results.compactMap(\.diagnostics).compactMap(value)
        guard !values.isEmpty else { return }
        print("[VOXT_SMOKE][summary] avg\(label.prefix(1).uppercased())\(label.dropFirst())=\(average(values))")
        let hotValues = Array(values.dropFirst())
        if !hotValues.isEmpty {
            print("[VOXT_SMOKE][summary] hotAvg\(label.prefix(1).uppercased())\(label.dropFirst())=\(average(hotValues))")
        }
    }

    private func average(_ values: [Int]) -> Int {
        guard !values.isEmpty else { return 0 }
        let total = values.reduce(0, +)
        return Int((Double(total) / Double(values.count)).rounded())
    }

    private func resolvedSmokeTranslationTargetLanguage(_ rawValue: String?) -> TranslationTargetLanguage? {
        let normalized = rawValue?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard let normalized, !normalized.isEmpty else { return nil }
        return TranslationTargetLanguage.allCases.first {
            $0.rawValue.lowercased() == normalized ||
            $0.instructionName.lowercased() == normalized ||
            $0.title.lowercased() == normalized
        }
    }
}
