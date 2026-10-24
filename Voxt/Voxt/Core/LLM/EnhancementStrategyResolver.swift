// EnhancementStrategyResolver.swift
// Provides Enhancement Strategy Resolver for LLM requests and response handling.

import Foundation

enum TaskLLMKind: String, Equatable {
    case transcriptionEnhancement
    case translation
    case rewrite
    case transcriptSummary

    var logLabel: String { rawValue }

    var outputTokenMultiplier: Double {
        switch self {
        case .transcriptionEnhancement:
            return 1.10
        case .translation, .rewrite:
            return 1.35
        case .transcriptSummary:
            return 0.45
        }
    }
}

enum TaskLLMExecutionMode: String, Equatable {
    case singlePass
    case segmented
}

enum TaskLLMContextBudgetPolicy: String, Equatable {
    case standard
    case reducedForLongInput
}

struct LLMProviderModelCapabilities: Equatable {
    let maxContextTokens: Int?
    let maxOutputTokens: Int?

    static let unknown = LLMProviderModelCapabilities(
        maxContextTokens: nil,
        maxOutputTokens: nil
    )
}

struct TaskLLMTruncationGuardPolicy: Equatable {
    let isEnabled: Bool
    let minimumCoverageRatio: Double
    let prefixCoverageRatio: Double
    let absoluteSlack: Int

    static let disabled = TaskLLMTruncationGuardPolicy(
        isEnabled: false,
        minimumCoverageRatio: 0,
        prefixCoverageRatio: 0,
        absoluteSlack: 0
    )
}

struct TaskLLMExecutionStrategy: Equatable {
    let taskKind: TaskLLMKind
    let rawTextCharacterCount: Int
    let promptCharacterCount: Int
    let mode: TaskLLMExecutionMode
    let contextBudgetPolicy: TaskLLMContextBudgetPolicy
    let glossarySelectionPolicy: DictionaryGlossarySelectionPolicy
    let outputTokenBudgetHint: Int?
    let segmentationCharacterLimit: Int?
    let truncationGuard: TaskLLMTruncationGuardPolicy

    var logLabel: String {
        [
            "task=\(taskKind.logLabel)",
            "rawChars=\(rawTextCharacterCount)",
            "promptChars=\(promptCharacterCount)",
            "mode=\(mode.rawValue)",
            "contextBudget=\(contextBudgetPolicy.rawValue)",
            "outputBudgetHint=\(outputTokenBudgetHint.map(String.init) ?? "n/a")",
            "segmentLimit=\(segmentationCharacterLimit.map(String.init) ?? "n/a")",
            "truncationGuard=\(truncationGuard.isEnabled)"
        ].joined(separator: ",")
    }
}

enum TaskLLMStrategyResolver {
    static let longTextThreshold = 300

    static func resolve(
        taskKind: TaskLLMKind,
        rawText: String,
        promptCharacterCount: Int,
        baseGlossarySelectionPolicy: DictionaryGlossarySelectionPolicy,
        capabilities: LLMProviderModelCapabilities
    ) -> TaskLLMExecutionStrategy {
        let rawTextCharacterCount = normalizedCharacterCount(rawText)
        let isLongText = rawTextCharacterCount > longTextThreshold
        let estimatedOutputTokens = estimatedOutputTokens(
            taskKind: taskKind,
            rawTextCharacterCount: rawTextCharacterCount
        )

        let mode: TaskLLMExecutionMode
        if isLongText,
           let maxOutputTokens = capabilities.maxOutputTokens,
           estimatedOutputTokens >= max(1, Int(Double(maxOutputTokens) * 0.85)) {
            mode = .segmented
        } else {
            mode = .singlePass
        }

        let outputTokenBudgetHint: Int?
        if let maxOutputTokens = capabilities.maxOutputTokens {
            outputTokenBudgetHint = min(maxOutputTokens, estimatedOutputTokens)
        } else {
            outputTokenBudgetHint = nil
        }

        return TaskLLMExecutionStrategy(
            taskKind: taskKind,
            rawTextCharacterCount: rawTextCharacterCount,
            promptCharacterCount: promptCharacterCount,
            mode: mode,
            contextBudgetPolicy: isLongText ? .reducedForLongInput : .standard,
            glossarySelectionPolicy: isLongText
                ? baseGlossarySelectionPolicy.reducedForLongInput()
                : baseGlossarySelectionPolicy,
            outputTokenBudgetHint: outputTokenBudgetHint,
            segmentationCharacterLimit: mode == .segmented ? max(longTextThreshold, 280) : nil,
            truncationGuard: isLongText
                ? truncationGuardPolicy(for: taskKind)
                : .disabled
        )
    }

    static func applyTruncationGuard(
        outputText: String,
        originalText: String,
        strategy: TaskLLMExecutionStrategy
    ) -> (text: String, didFallback: Bool, reason: String?) {
        let policy = strategy.truncationGuard
        guard policy.isEnabled else {
            return (outputText, false, nil)
        }

        let normalizedOriginal = normalizedComparableText(originalText)
        let normalizedOutput = normalizedComparableText(outputText)
        guard !normalizedOriginal.isEmpty, !normalizedOutput.isEmpty else {
            return (originalText, true, "emptyComparableText")
        }

        let originalCount = normalizedOriginal.count
        let outputCount = normalizedOutput.count
        // Accept either proportional coverage or a small absolute drop; cleanup can
        // legitimately remove many filler tokens without being truncated.
        let minimumCoverage = min(
            Int((Double(originalCount) * policy.minimumCoverageRatio).rounded(.down)),
            originalCount - policy.absoluteSlack
        )
        let coverageRatio = Double(outputCount) / Double(max(1, originalCount))
        let isSuspiciousPrefix = normalizedOriginal.hasPrefix(normalizedOutput) &&
            coverageRatio < policy.prefixCoverageRatio
        let isSuspiciouslyShort = outputCount < minimumCoverage ||
            isSuspiciousSameScriptTranslation(
                original: normalizedOriginal,
                output: normalizedOutput,
                coverageRatio: coverageRatio,
                strategy: strategy
            )

        guard isSuspiciousPrefix || isSuspiciouslyShort else {
            return (outputText, false, nil)
        }

        let reason = [
            "normalizedInputChars=\(originalCount)",
            "normalizedOutputChars=\(outputCount)",
            "minimumCoverage=\(minimumCoverage)",
            "coverageRatio=\(String(format: "%.3f", coverageRatio))",
            "prefix=\(isSuspiciousPrefix)",
            "short=\(isSuspiciouslyShort)"
        ].joined(separator: ",")
        return (originalText, true, reason)
    }

    private static func estimatedOutputTokens(
        taskKind: TaskLLMKind,
        rawTextCharacterCount: Int
    ) -> Int {
        let safeCharacters = max(1, rawTextCharacterCount)
        return Int((Double(safeCharacters) * taskKind.outputTokenMultiplier).rounded(.up))
    }

    private static func truncationGuardPolicy(for taskKind: TaskLLMKind) -> TaskLLMTruncationGuardPolicy {
        switch taskKind {
        case .transcriptionEnhancement, .transcriptSummary:
            return .disabled
        case .translation, .rewrite:
            return TaskLLMTruncationGuardPolicy(
                isEnabled: true,
                minimumCoverageRatio: 0,
                prefixCoverageRatio: 0.90,
                absoluteSlack: 24
            )
        }
    }

    private static func isSuspiciousSameScriptTranslation(
        original: String,
        output: String,
        coverageRatio: Double,
        strategy: TaskLLMExecutionStrategy
    ) -> Bool {
        guard strategy.taskKind == .translation,
              coverageRatio < 0.35
        else {
            return false
        }

        let originalScript = dominantScript(in: original)
        guard originalScript != .other else { return false }
        return originalScript == dominantScript(in: output)
    }

    private enum DominantScript {
        case latin
        case han
        case kana
        case other
    }

    private static func dominantScript(in text: String) -> DominantScript {
        var latinCount = 0
        var hanCount = 0
        var kanaCount = 0

        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x0041...0x005A, 0x0061...0x007A:
                latinCount += 1
            case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF:
                hanCount += 1
            case 0x3040...0x309F, 0x30A0...0x30FF:
                kanaCount += 1
            default:
                continue
            }
        }

        let ranked = [
            (DominantScript.latin, latinCount),
            (DominantScript.han, hanCount),
            (DominantScript.kana, kanaCount)
        ].max { lhs, rhs in
            lhs.1 < rhs.1
        }

        guard let ranked, ranked.1 > 0 else {
            return .other
        }
        return ranked.0
    }

    private static func normalizedCharacterCount(_ text: String) -> Int {
        normalizedComparableText(text).count
    }

    private static func normalizedComparableText(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
