// HyMT2TranslationBenchmarkTests.swift
// Measures Hy-MT2 translation through the app's local translation managers.
//
// Opt-in: runs only with VOXT_RUN_MODEL_TESTS=1 and VOXT_BENCH_OUT set. Run one
// backend per process so load time and peak memory are not shared between arms.
//
//   VOXT_MODEL_STORAGE_ROOT   model storage root holding gguf-translation/ or mlx-llm/
//   VOXT_BENCH_BACKEND        gguf or mlx
//   VOXT_BENCH_MODEL          GGUF model ID or MLX repo
//   VOXT_BENCH_INPUT          JSONL of {id, target, source}
//   VOXT_BENCH_OUT            output directory
//   VOXT_BENCH_RUN            run label, e.g. mlx-4bit-r1

import Darwin
import MLX
import XCTest
@testable import Voxt

@MainActor
final class HyMT2TranslationBenchmarkTests: XCTestCase {
    private struct Item: Decodable {
        let id: String
        let target: String
        let source: String
    }

    private struct Result: Encodable {
        let id: String
        let output: String
        let totalMs: Double
        let firstPartialMs: Double?
        let mlx: MLXTimings?
    }

    private struct MLXTimings: Encodable {
        let setupMs: Int
        let promptTokens: Int?
        let completionTokens: Int?
        let prefillMs: Int?
        let generationMs: Int?
        let modelOverheadMs: Int?
    }

    private struct Summary: Encodable {
        let run: String
        let backend: String
        let model: String
        let items: Int
        let coldFirstRequestMs: Double
        let warmFirstRequestMs: Double
        let baselineFootprintBytes: UInt64
        let peakFootprintBytes: UInt64
        let baselineResidentBytes: UInt64
        let peakResidentBytes: UInt64
        let mlxPeakBytes: Int
    }

    private final class PartialClock: @unchecked Sendable {
        private let lock = NSLock()
        private var firstPartialAt: TimeInterval?

        func mark() {
            lock.lock()
            defer { lock.unlock() }
            if firstPartialAt == nil {
                firstPartialAt = ProcessInfo.processInfo.systemUptime
            }
        }

        var value: TimeInterval? {
            lock.lock()
            defer { lock.unlock() }
            return firstPartialAt
        }
    }

    func testTranslateFixedSet() async throws {
        try ModelTestGate.requireEnabled("Hy-MT2 translation benchmark")
        let environment = ProcessInfo.processInfo.environment
        func value(_ key: String) -> String? {
            let trimmed = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return trimmed.isEmpty ? nil : trimmed
        }
        guard let backend = value("VOXT_BENCH_BACKEND"),
              let model = value("VOXT_BENCH_MODEL"),
              let input = value("VOXT_BENCH_INPUT"),
              let output = value("VOXT_BENCH_OUT"),
              let run = value("VOXT_BENCH_RUN") else {
            throw XCTSkip("Set VOXT_BENCH_BACKEND, VOXT_BENCH_MODEL, VOXT_BENCH_INPUT, VOXT_BENCH_OUT and VOXT_BENCH_RUN.")
        }
        ModelTestGate.configureStorageRoot(for: self)

        let decoder = JSONDecoder()
        let items = try String(contentsOfFile: input, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .map { try decoder.decode(Item.self, from: Data($0.utf8)) }
        XCTAssertFalse(items.isEmpty)

        let translate = try await translator(backend: backend, model: model)
        let baseline = Self.memory()

        // The second warm-up isolates model load from the first request.
        let warmup = Item(id: "warmup", target: TranslationTargetLanguage.chineseSimplified.rawValue, source: "Hello, world.")
        let cold = try await timedTranslation(warmup, translate: translate)
        let warm = try await timedTranslation(warmup, translate: translate)

        var results: [Result] = []
        for item in items {
            results.append(try await timedTranslation(item, translate: translate))
        }

        let summary = Summary(
            run: run,
            backend: backend,
            model: model,
            items: results.count,
            coldFirstRequestMs: cold.totalMs,
            warmFirstRequestMs: warm.totalMs,
            baselineFootprintBytes: baseline.footprint,
            peakFootprintBytes: Self.memory().peakFootprint,
            baselineResidentBytes: baseline.resident,
            peakResidentBytes: Self.memory().peakResident,
            mlxPeakBytes: backend == "mlx" ? Memory.peakMemory : 0
        )

        let outputURL = URL(fileURLWithPath: output, isDirectory: true)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let lines = try results.map { String(decoding: try encoder.encode($0), as: UTF8.self) }
        try (lines.joined(separator: "\n") + "\n").write(
            to: outputURL.appendingPathComponent("\(run).jsonl"),
            atomically: true,
            encoding: .utf8
        )
        try encoder.encode(summary).write(to: outputURL.appendingPathComponent("\(run).summary.json"))
    }

    private typealias Translate = (LLMCompiledRequest, @escaping @Sendable (String) -> Void) async throws -> (output: String, mlx: MLXTimings?)

    private func translator(backend: String, model: String) async throws -> Translate {
        switch backend {
        case "gguf":
            guard let id = GGUFTranslationModelID(rawValue: model) else {
                throw XCTSkip("Unknown GGUF model ID: \(model)")
            }
            let manager = GGUFTranslationModelManager(modelID: id)
            guard try await manager.refreshInstallation(id: id).isInstalled else {
                throw XCTSkip("GGUF model is not installed: \(model)")
            }
            addTeardownBlock { await manager.shutdownForApplicationTermination() }
            return { request, onPartial in
                (try await manager.executeCompiledRequest(request, modelID: id, onPartialText: onPartial), nil)
            }
        case "mlx":
            let manager = CustomLLMModelManager(modelRepo: model)
            guard try await manager.refreshInstallation(repo: model).isInstalled else {
                throw XCTSkip("MLX model is not installed: \(model)")
            }
            addTeardownBlock { await manager.shutdownForApplicationTermination() }
            return { request, onPartial in
                let output = try await manager.executeCompiledRequest(request, repo: model, onPartialText: onPartial)
                let diagnostics = manager.lastRunDiagnostics
                return (output, diagnostics.map {
                    MLXTimings(
                        setupMs: $0.setupMs,
                        promptTokens: $0.promptTokens,
                        completionTokens: $0.completionTokens,
                        prefillMs: $0.prefillMs,
                        generationMs: $0.generationMs,
                        modelOverheadMs: $0.modelOverheadMs
                    )
                })
            }
        default:
            throw XCTSkip("Unknown backend: \(backend)")
        }
    }

    private func timedTranslation(_ item: Item, translate: Translate) async throws -> Result {
        guard let target = TranslationTargetLanguage(rawValue: item.target) else {
            throw XCTSkip("Unknown target language: \(item.target)")
        }
        let request = compiledTranslationRequest(source: item.source, target: target)
        let clock = PartialClock()
        let startedAt = ProcessInfo.processInfo.systemUptime
        let translation = try await translate(request) { _ in clock.mark() }
        let finishedAt = ProcessInfo.processInfo.systemUptime
        let visible = LLMVisibleOutputSanitizer.sanitize(
            translation.output,
            fallbackText: item.source,
            taskKind: .translation
        ).text
        return Result(
            id: item.id,
            output: visible,
            totalMs: (finishedAt - startedAt) * 1000,
            firstPartialMs: clock.value.map { ($0 - startedAt) * 1000 },
            mlx: translation.mlx
        )
    }

    // Mirrors AppDelegate.buildTranslationExecutionPlan with the default prompt
    // and no dictionary glossary.
    private func compiledTranslationRequest(
        source: String,
        target: TranslationTargetLanguage
    ) -> LLMCompiledRequest {
        let template = AppPreferenceKey.defaultTranslationPrompt
        let prompt = TranslationPromptBuilder.build(
            systemPrompt: template,
            targetLanguage: target,
            sourceText: source,
            userMainLanguagePromptValue: "English",
            strict: false
        )
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: source,
            promptCharacterCount: prompt.count,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.translation.selectionPolicy,
            capabilities: .unknown
        )
        let delivery: LLMExecutionDelivery =
            TextTransformPromptSupport.deliveryForTemplate(template, variableTokens: ["{{SOURCE_TEXT}}"]) == .userMessage
            ? .userMessage
            : .systemPrompt
        let plan = LLMExecutionPlan(
            task: .translation(sourceText: source, targetLanguage: target),
            provider: .appleIntelligence,
            delivery: delivery,
            promptContent: prompt,
            fallbackText: source,
            executionStrategy: strategy,
            outputTokenBudgetHint: strategy.outputTokenBudgetHint,
            contextBlocks: [
                LLMContextBlock(kind: .input, title: "Source text", content: source, isStablePrefixCandidate: false)
            ],
            conversationHistory: [],
            previousResponseID: nil,
            responseFormat: nil
        )
        return LLMExecutionPlanCompiler.compile(plan)
    }

    // llama.cpp maps GGUF weights from disk; clean file-backed pages count toward
    // resident size but not toward the physical footprint.
    private static func memory() -> (footprint: UInt64, peakFootprint: UInt64, resident: UInt64, peakResident: UInt64) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return (0, 0, 0, 0) }
        return (
            info.phys_footprint,
            UInt64(info.ledger_phys_footprint_peak),
            info.resident_size,
            info.resident_size_peak
        )
    }
}
