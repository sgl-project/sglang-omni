// CustomLLMModelSupport.swift
// Provides Custom LLMModel Support for model catalog and storage support.

import Foundation
import HuggingFace

struct CustomLLMModelBehavior: Equatable {
    let family: CustomLLMModelFamily
    let disablesThinking: Bool

    static let thinkingOffAdditionalContext: [String: any Sendable] = [
        "enable_thinking": false,
        "reasoning_effort": "low"
    ]

    static let thinkingOnAdditionalContext: [String: any Sendable] = [
        "enable_thinking": true,
        "reasoning_effort": "medium"
    ]

}

enum CustomLLMModelBehaviorResolver {
    static func behavior(for repo: String) -> CustomLLMModelBehavior {
        let family = CustomLLMModelFamily.resolve(for: repo)
        return CustomLLMModelBehavior(
            family: family,
            disablesThinking: disablesThinking(for: repo, family: family)
        )
    }

    private static func disablesThinking(
        for repo: String,
        family: CustomLLMModelFamily
    ) -> Bool {
        if family == .qwen3 {
            return true
        }
        return false
    }
}

enum CustomLLMTaskKind: Equatable {
    case enhancement
    case translation
    case rewrite
    case dictionaryHistoryScan

    var logLabel: String {
        switch self {
        case .enhancement: return "enhance"
        case .translation: return "translate"
        case .rewrite: return "rewrite"
        case .dictionaryHistoryScan: return "dictionaryHistoryScan"
        }
    }

    var tokenBudgetMultiplier: Double {
        switch self {
        case .enhancement:
            return 1.10
        case .translation, .rewrite:
            return 1.35
        case .dictionaryHistoryScan:
            return 2.20
        }
    }
}

struct CustomLLMLogSection: Equatable {
    let label: String
    let content: String
}

enum CustomLLMContainerLoadSource: String, Equatable {
    case reusedLoaded
    case loadedFromDisk
}

struct CustomLLMRunDiagnostics: Equatable {
    let repo: String
    let taskLabel: String
    let containerLoadSource: CustomLLMContainerLoadSource
    let containerLoadMs: Int
    let setupMs: Int
    let modelElapsedMs: Int
    let totalElapsedMs: Int
    let firstChunkMs: Int?
    let overallFirstChunkMs: Int?
    let promptTokens: Int?
    let completionTokens: Int?
    let prefillMs: Int?
    let generationMs: Int?
    let modelOverheadMs: Int?
    let totalOverheadMs: Int?
}

struct CustomLLMGenerationTuning: Equatable {
    let prefillStepSizeOverride: Int?
    let maxTokensOverride: Int?

    static let `default` = CustomLLMGenerationTuning(prefillStepSizeOverride: nil, maxTokensOverride: nil)
}

/// Bounds UI preview work without dropping the final result or delaying the first chunk.
nonisolated struct LocalLLMPartialDelivery {
    private var lastPublishedAt: TimeInterval?
    private let minimumInterval: TimeInterval = 0.05

    mutating func shouldPublish(at uptime: TimeInterval) -> Bool {
        if let lastPublishedAt, uptime - lastPublishedAt < minimumInterval {
            return false
        }
        lastPublishedAt = uptime
        return true
    }
}

struct LLMOutputRepetition: Equatable {
    let repeatedUnit: String
    let repetitionCount: Int
    let truncatedText: String
}

struct LLMOutputRepetitionGuard {
    var maximumUnitLength = 48
    var minimumRepetitionCount = 6
    var minimumRunCharacterCount = 48
    var shortUnitMinimumRepetitionCount = 10
    var shortUnitMinimumRunCharacterCount = 24

    func repeatedSuffix(in text: String) -> LLMOutputRepetition? {
        let characterCount = text.count
        guard characterCount >= shortUnitMinimumRunCharacterCount else { return nil }

        let longestUnit = min(maximumUnitLength, characterCount / minimumRepetitionCount)
        guard longestUnit > 0 else { return nil }

        for unitLength in 1...longestUnit {
            guard let unitStart = text.index(text.endIndex, offsetBy: -unitLength, limitedBy: text.startIndex) else {
                continue
            }
            let unit = String(text[unitStart...])
            guard !unit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }

            var repetitions = 1
            var runStart = unitStart
            while let previousStart = text.index(runStart, offsetBy: -unitLength, limitedBy: text.startIndex),
                  text[previousStart..<runStart] == text[unitStart..<text.endIndex] {
                repetitions += 1
                runStart = previousStart
            }

            let runCharacterCount = repetitions * unitLength
            let isShortUnitRun = unitLength <= 4 &&
                repetitions >= shortUnitMinimumRepetitionCount &&
                runCharacterCount >= shortUnitMinimumRunCharacterCount
            let isGeneralRun = repetitions >= minimumRepetitionCount &&
                runCharacterCount >= minimumRunCharacterCount

            guard isShortUnitRun || isGeneralRun else { continue }

            let keepEnd = text.index(text.endIndex, offsetBy: -unitLength * (repetitions - 1))
            return LLMOutputRepetition(
                repeatedUnit: unit,
                repetitionCount: repetitions,
                truncatedText: String(text[..<keepEnd])
            )
        }

        return nil
    }
}

struct CustomLLMRequestPlan: Equatable {
    let kind: CustomLLMTaskKind
    let repo: String
    let instructions: String
    let prompt: String
    let inputCharacterCount: Int
    let maxTokensOverride: Int?
    let conversationHistory: [RewriteConversationPromptTurn]
    let logMode: String?
    let contentLogSections: [CustomLLMLogSection]
    let resultFallback: String
    let responseExtractionMode: CustomLLMResponseExtractionMode
}

enum CustomLLMResponseExtractionMode: Equatable {
    case textResultPayloadOrNormalizedText
    case normalizedRawText
}

enum CustomLLMRequestPlanBuilder {
    static func compiled(
        request: LLMCompiledRequest,
        repo: String
    ) -> CustomLLMRequestPlan {
        let kind: CustomLLMTaskKind
        switch request.taskLabel {
        case "enhancement":
            kind = .enhancement
        case "translation":
            kind = .translation
        case "rewrite":
            kind = .rewrite
        default:
            kind = .enhancement
        }

        let usesUserMessageMode = request.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        var sections: [CustomLLMLogSection] = [
            CustomLLMLogSection(
                label: "system_prompt",
                content: usesUserMessageMode ? "<empty>" : request.instructions
            ),
            CustomLLMLogSection(label: "input", content: request.debugInput)
        ]
        sections.append(
            CustomLLMLogSection(
                label: usesUserMessageMode ? "user_message_prompt" : "request_content",
                content: request.prompt
            )
        )

        return CustomLLMRequestPlan(
            kind: kind,
            repo: repo,
            instructions: request.instructions,
            prompt: request.prompt,
            inputCharacterCount: request.inputCharacterCount,
            maxTokensOverride: request.outputTokenBudgetHint,
            conversationHistory: request.conversationHistory,
            logMode: usesUserMessageMode ? "userMessage" : nil,
            contentLogSections: sections,
            resultFallback: request.fallbackText,
            responseExtractionMode: .textResultPayloadOrNormalizedText
        )
    }

    static func userPromptEnhancement(
        prompt: String,
        repo: String
    ) -> CustomLLMRequestPlan {
        CustomLLMRequestPlan(
            kind: .enhancement,
            repo: repo,
            instructions: "",
            prompt: prompt,
            inputCharacterCount: prompt.count,
            maxTokensOverride: nil,
            conversationHistory: [],
            logMode: "userMessage",
            contentLogSections: [
                CustomLLMLogSection(label: "system_prompt", content: "<empty>"),
                CustomLLMLogSection(label: "input", content: prompt)
            ],
            resultFallback: "",
            responseExtractionMode: .textResultPayloadOrNormalizedText
        )
    }

    static func dictionaryHistoryScan(
        prompt: String,
        repo: String,
        structuredOutputPrompt: (String) -> String
    ) -> CustomLLMRequestPlan {
        let requestPrompt = structuredOutputPrompt(prompt)
        return CustomLLMRequestPlan(
            kind: .dictionaryHistoryScan,
            repo: repo,
            instructions: "",
            prompt: requestPrompt,
            inputCharacterCount: prompt.count,
            maxTokensOverride: nil,
            conversationHistory: [],
            logMode: "dictionaryHistoryScan",
            contentLogSections: [
                CustomLLMLogSection(label: "system_prompt", content: "<empty>"),
                CustomLLMLogSection(label: "input", content: prompt),
                CustomLLMLogSection(label: "request_content", content: requestPrompt)
            ],
            resultFallback: "[]",
            responseExtractionMode: .normalizedRawText
        )
    }
}

enum CustomLLMModelFamily: Equatable {
    case qwen3
    case glm4
    case mistral
    case gemma
    case other

    var logLabel: String {
        switch self {
        case .qwen3: return "qwen3"
        case .glm4: return "glm4"
        case .mistral: return "mistral"
        case .gemma: return "gemma"
        case .other: return "other"
        }
    }

    static func resolve(for repo: String) -> CustomLLMModelFamily {
        let normalizedRepo = CustomLLMModelCatalog.canonicalModelRepo(repo).lowercased()
        if normalizedRepo.contains("qwen3") { return .qwen3 }
        if normalizedRepo.contains("glm-4") || normalizedRepo.contains("glm4") {
            return .glm4
        }
        if normalizedRepo.contains("mistral") || normalizedRepo.contains("ministral") { return .mistral }
        if normalizedRepo.contains("gemma") { return .gemma }
        return .other
    }
}

enum CustomLLMOutputSanitizer {
    static func normalizeResultText(_ output: String) -> String {
        LLMVisibleOutputSanitizer.sanitize(
            output,
            fallbackText: "",
            taskKind: .generic
        ).text
    }

    static func unwrapCodeFenceIfNeeded(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("```"), trimmed.hasSuffix("```") else {
            return trimmed
        }
        var lines = trimmed.components(separatedBy: .newlines)
        guard lines.count >= 2 else { return trimmed }
        lines.removeFirst()
        if let last = lines.last, last.trimmingCharacters(in: .whitespacesAndNewlines) == "```" {
            lines.removeLast()
        }
        return lines.joined(separator: "\n")
    }
}

struct CustomLLMModelCatalog {
    struct Option: Identifiable, Hashable {
        let id: String
        let title: String
        let description: String
    }

    private struct PresentationMetadata {
        let ratingText: String
        let tagKeys: [String]
    }

    nonisolated static let defaultModelRepo = "mlx-community/Qwen3.5-4B-OptiQ-4bit"

    // Migration only: no compatibility-only runtime models.
    nonisolated private static let compatibilityAliases: [String: String] = [
        "Qwen/Qwen2-1.5B-Instruct": "mlx-community/Qwen3.5-2B-4bit",
        "Qwen/Qwen2.5-3B-Instruct": "mlx-community/Qwen3.5-4B-OptiQ-4bit",
        "mlx-community/Qwen2.5-VL-3B-Instruct-4bit": "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "mlx-community/Qwen3-0.6B-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/Qwen3-1.7B-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/Qwen3-4B-4bit": "mlx-community/Qwen3.5-4B-OptiQ-4bit",
        "mlx-community/Qwen3-8B-4bit": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "mlx-community/Qwen3.5-4B-4bit": "mlx-community/Qwen3.5-4B-OptiQ-4bit",
        "mlx-community/Qwen3.5-0.8B-OptiQ-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/gemma-2-2b-it-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-2-9b-it-4bit": "mlx-community/gemma-4-e4b-it-4bit",
        "mlx-community/gemma-3-1b-it-qat-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-3n-E2B-it-lm-4bit": "mlx-community/gemma-4-e2b-it-4bit",
        "mlx-community/gemma-3n-E4B-it-lm-4bit": "mlx-community/gemma-4-e4b-it-4bit",
        "mlx-community/Qwen3-30B-A3B-4bit": "mlx-community/Qwen3.6-27B-4bit",
        "Qwen/Qwen3-8B-4bit": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "Qwen/Qwen2.5-7B-Instruct": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "mlx-community/Qwen3.5-2B-MLX-4bit": "mlx-community/Qwen3.5-2B-4bit",
        "mlx-community/Qwen3.5-0.8B-4bit-OptiQ": "mlx-community/Qwen3.5-2B-4bit",
    ]

    nonisolated private static let visibleModels: [Option] = [
        Option(
            id: "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
            title: "Qwen3 VL 4B Instruct (4bit)",
            description: "Qwen3 VL architecture for local text generation."
        ),
        Option(
            id: "mlx-community/Qwen3.5-2B-4bit",
            title: "Qwen3.5 2B (4bit)",
            description: "Official Qwen3.5 local model using the upstream-supported inference path."
        ),
        Option(
            id: "mlx-community/Qwen3.5-4B-OptiQ-4bit",
            title: "Qwen3.5 4B OptiQ (4bit)",
            description: "Mixed-precision Qwen3.5 variant tuned for a stronger quality-to-size balance on home Macs."
        ),
        Option(
            id: "mlx-community/Qwen3.5-9B-OptiQ-4bit",
            title: "Qwen3.5 9B OptiQ (4bit)",
            description: "Higher-quality Qwen3.5 option for higher-memory Macs using Apple-Silicon-optimized mixed precision."
        ),
        Option(
            id: "mlx-community/GLM-4-9B-0414-4bit",
            title: "GLM 4 9B",
            description: "GLM-4 model variant with strong multilingual instruction following."
        ),
        Option(
            id: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
            title: "Mistral 3 3B",
            description: "Current compact Mistral-family model for lightweight non-Qwen local generation."
        ),
        Option(
            id: "mlx-community/LFM2-1.2B-4bit",
            title: "LFM2 1.2B (4bit)",
            description: "Very lightweight LFM2 model for low-memory local generation."
        ),
        Option(
            id: "mlx-community/LFM2-8B-A1B-3bit-MLX",
            title: "LFM2 8B A1B (3bit)",
            description: "Compact LFM2 MoE model with a small active-parameter footprint."
        ),
        Option(
            id: "mlx-community/Qwen3.6-27B-4bit",
            title: "Qwen3.6 27B (4bit)",
            description: "High-end Qwen3.6 model for large-memory Macs and stronger local quality."
        ),
        Option(
            id: "mlx-community/gemma-4-e2b-it-4bit",
            title: "Gemma 4 E2B IT (4bit)",
            description: "Compact Gemma 4 model for local text generation."
        ),
        Option(
            id: "mlx-community/gemma-4-e4b-it-4bit",
            title: "Gemma 4 E4B IT (4bit)",
            description: "Higher-capacity Gemma 4 option for stronger local text generation."
        ),
        Option(
            id: "mlx-community/gemma-4-12B-it-OptiQ-4bit",
            title: "Gemma 4 12B IT OptiQ (4bit)",
            description: "High-end Gemma 4 model for stronger local text quality on higher-memory Macs."
        ),
    ]

    nonisolated private static let allModels: [Option] = visibleModels

    nonisolated static let availableModels: [Option] = allModels

    nonisolated static let supportedModels: [Option] = allModels

    nonisolated private static let presentationByRepo: [String: PresentationMetadata] = [
        "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit": PresentationMetadata(ratingText: "4.7", tagKeys: ["Balanced"]),
        "mlx-community/Qwen3.5-4B-OptiQ-4bit": PresentationMetadata(ratingText: "4.8", tagKeys: ["Balanced"]),
        "mlx-community/Qwen3.5-9B-OptiQ-4bit": PresentationMetadata(ratingText: "4.9", tagKeys: ["Accurate"]),
        "mlx-community/GLM-4-9B-0414-4bit": PresentationMetadata(ratingText: "4.7", tagKeys: ["Accurate"]),
        "mlx-community/Ministral-3-3B-Instruct-2512-4bit": PresentationMetadata(ratingText: "4.5", tagKeys: ["Balanced"]),
        "mlx-community/gemma-4-e2b-it-4bit": PresentationMetadata(ratingText: "4.3", tagKeys: ["Fast"]),
        "mlx-community/gemma-4-e4b-it-4bit": PresentationMetadata(ratingText: "4.6", tagKeys: ["Balanced"]),
        "mlx-community/gemma-4-12B-it-OptiQ-4bit": PresentationMetadata(ratingText: "4.8", tagKeys: ["Accurate"]),
        "mlx-community/LFM2-1.2B-4bit": PresentationMetadata(ratingText: "4.0", tagKeys: ["Fast"]),
        "mlx-community/LFM2-8B-A1B-3bit-MLX": PresentationMetadata(ratingText: "4.4", tagKeys: ["Balanced"]),
        "mlx-community/Qwen3.6-27B-4bit": PresentationMetadata(ratingText: "4.9", tagKeys: ["Accurate"]),
    ]

    nonisolated private static let knownRemoteSizeBytesByRepo: [String: Int64] = [
        "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit": 4_900_000_000,
        "mlx-community/Qwen3.5-2B-4bit": 1_742_261_128,
        "mlx-community/Qwen3.5-4B-OptiQ-4bit": 2_970_000_000,
        "mlx-community/Qwen3.5-9B-OptiQ-4bit": 6_040_000_000,
        "mlx-community/GLM-4-9B-0414-4bit": 5_309_031_270,
        "mlx-community/Ministral-3-3B-Instruct-2512-4bit": 2_745_210_934,
        "mlx-community/gemma-4-e2b-it-4bit": 3_581_101_896,
        "mlx-community/gemma-4-e4b-it-4bit": 5_217_361_182,
        "mlx-community/gemma-4-12B-it-OptiQ-4bit": 8_964_644_918,
        "mlx-community/LFM2-1.2B-4bit": 663_392_070,
        "mlx-community/LFM2-8B-A1B-3bit-MLX": 4_176_559_875,
        "mlx-community/Qwen3.6-27B-4bit": 16_081_490_064,
    ]

    nonisolated static func canonicalModelRepo(_ repo: String) -> String {
        let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultModelRepo }
        return compatibilityAliases[trimmed] ?? trimmed
    }

    /// Only this alias renames the same checkpoint; retired model settings must not
    /// be copied onto a different replacement model.
    nonisolated static func preservesGenerationSettings(for repo: String) -> Bool {
        availableModels.contains { $0.id == repo }
            || repo == "mlx-community/Qwen3.5-2B-MLX-4bit"
    }

    nonisolated static func option(for repo: String) -> Option? {
        let canonicalRepo = canonicalModelRepo(repo)
        return supportedModels.first(where: { $0.id == canonicalRepo })
    }

    nonisolated static func displayModels(including repo: String? = nil) -> [Option] {
        guard let repo else { return availableModels }
        return displayModels(includingInstalled: [repo])
    }

    nonisolated static func displayModels(includingInstalled _: Set<String>) -> [Option] {
        availableModels
    }

    nonisolated static func displayTitle(for repo: String) -> String {
        option(for: repo)?.title ?? repo
    }

    nonisolated static func description(for repo: String) -> String? {
        option(for: repo)?.description
    }

    nonisolated static func ratingText(for repo: String) -> String {
        presentationByRepo[canonicalModelRepo(repo)]?.ratingText ?? "4.0"
    }

    nonisolated static func catalogTagKeys(for repo: String) -> [String] {
        presentationByRepo[canonicalModelRepo(repo)]?.tagKeys ?? []
    }

    nonisolated static func isSupportedModelRepo(_ repo: String) -> Bool {
        option(for: repo) != nil
    }

    /// Loading architecture, not an image-input capability. Voxt sends text only.
    nonisolated static func requiresVLMFactory(repo: String) -> Bool {
        switch canonicalModelRepo(repo) {
        case "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
             "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
             "mlx-community/gemma-4-e2b-it-4bit",
             "mlx-community/gemma-4-e4b-it-4bit",
             "mlx-community/gemma-4-12B-it-OptiQ-4bit":
            return true
        default:
            return false
        }
    }

    nonisolated static func fallbackRemoteSizeText(repo: String) -> String? {
        fallbackRemoteSizeInfo(repo: repo)?.text
    }

    nonisolated static func fallbackRemoteSizeInfo(repo: String) -> (bytes: Int64, text: String)? {
        guard let bytes = knownRemoteSizeBytesByRepo[canonicalModelRepo(repo)] else { return nil }
        return (bytes, CustomLLMModelStorageSupport.formatByteCount(bytes))
    }
}

enum CustomLLMModelStorageSupport {
    nonisolated private static let remoteSizeCachePreferenceKey = "customLLMRemoteSizeCache"

    nonisolated static func formatByteCount(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    nonisolated static func loadPersistedRemoteSizeCache() -> [String: String] {
        guard let data = UserDefaults.standard.data(forKey: remoteSizeCachePreferenceKey),
              let decoded = try? JSONDecoder().decode([String: String].self, from: data) else {
            return [:]
        }
        return decoded
    }

    nonisolated static func savePersistedRemoteSizeCache(_ cache: [String: String]) {
        guard let data = try? JSONEncoder().encode(cache) else { return }
        UserDefaults.standard.set(data, forKey: remoteSizeCachePreferenceKey)
    }

    nonisolated static func destinationFileURL(for entryPath: String, under directory: URL) throws -> URL {
        let base = directory.standardizedFileURL
        let destination = base.appendingPathComponent(entryPath).standardizedFileURL
        let basePrefix = base.path.hasSuffix("/") ? base.path : "\(base.path)/"
        guard destination.path.hasPrefix(basePrefix) else {
            throw NSError(
                domain: "Voxt.CustomLLM",
                code: 1002,
                userInfo: [NSLocalizedDescriptionKey: "Invalid model file path: \(entryPath)"]
            )
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        return destination
    }

    nonisolated static func cacheDirectory(for repo: String, rootDirectory: URL) -> URL? {
        guard let repoID = Repo.ID(rawValue: repo) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        return rootDirectory
            .appendingPathComponent("mlx-llm")
            .appendingPathComponent(modelSubdir)
    }

    nonisolated static func isModelDirectoryValid(_ directory: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: directory.path) else { return false }
        let rootConfig = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: rootConfig.path),
              let rootConfigData = try? Data(contentsOf: rootConfig),
              (try? JSONSerialization.jsonObject(with: rootConfigData)) != nil
        else {
            return false
        }

        guard ModelWeightFileValidation.hasCompleteIndex(in: directory),
              let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            return false
        }
        for case let url as URL in enumerator where url.pathExtension.lowercased() == "safetensors" {
            if let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
               values.isRegularFile == true, (values.fileSize ?? 0) > 0 { return true }
        }
        return false
    }

    nonisolated static func clearHubCache(for repoID: Repo.ID, rootDirectory: URL = HubCache.default.cacheDirectory) {
        let cache = HubCache(cacheDirectory: rootDirectory)
        let repoDir = cache.repoDirectory(repo: repoID, kind: .model)
        let metadataDir = cache.metadataDirectory(repo: repoID, kind: .model)
        try? FileManager.default.removeItem(at: repoDir)
        try? FileManager.default.removeItem(at: metadataDir)
    }
}
