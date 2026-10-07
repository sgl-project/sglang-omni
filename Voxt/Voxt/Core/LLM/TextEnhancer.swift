// TextEnhancer.swift
// Provides Text Enhancer for LLM requests and response handling.

import Foundation
import FoundationModels

enum AppleIntelligenceUnavailableReason: Equatable {
    case appleIntelligenceNotEnabled
    case deviceNotEligible
    case modelNotReady
    case unknown(String)

    var message: String {
        switch self {
        case .appleIntelligenceNotEnabled:
            return AppLocalization.localizedString(
                "Turn on Apple Intelligence in System Settings to use this model."
            )
        case .deviceNotEligible:
            return AppLocalization.localizedString(
                "Apple Intelligence is not available for this Mac or region."
            )
        case .modelNotReady:
            return AppLocalization.localizedString(
                "Apple Intelligence is still preparing its on-device model. Try again shortly."
            )
        case .unknown(let detail):
            return detail.isEmpty
                ? AppLocalization.localizedString("Apple Intelligence is currently unavailable.")
                : AppLocalization.localizedString("Apple Intelligence is currently unavailable.") + " (\(detail))"
        }
    }

    var logDescription: String {
        switch self {
        case .appleIntelligenceNotEnabled:
            return "appleIntelligenceNotEnabled"
        case .deviceNotEligible:
            return "deviceNotEligible"
        case .modelNotReady:
            return "modelNotReady"
        case .unknown(let detail):
            return "unknown(\(detail))"
        }
    }
}

enum AppleIntelligenceAvailability: Equatable {
    case unsupportedOS
    case available
    case unavailable(AppleIntelligenceUnavailableReason)

    var isAvailable: Bool {
        self == .available
    }

    var disabledReason: String? {
        switch self {
        case .unsupportedOS:
            return AppLocalization.localizedString("Apple Intelligence requires macOS 26 or later.")
        case .available:
            return nil
        case .unavailable(let reason):
            return reason.message
        }
    }

    var logDescription: String {
        switch self {
        case .unsupportedOS:
            return "unsupportedOS"
        case .available:
            return "available"
        case .unavailable(let reason):
            return "unavailable(\(reason.logDescription))"
        }
    }

    @MainActor
    static var current: Self {
        guard #available(macOS 26.0, *) else {
            return .unsupportedOS
        }

        switch SystemLanguageModel.default.availability {
        case .available:
            return .available
        case .unavailable(let reason):
            switch reason {
            case .appleIntelligenceNotEnabled:
                return .unavailable(.appleIntelligenceNotEnabled)
            case .deviceNotEligible:
                return .unavailable(.deviceNotEligible)
            case .modelNotReady:
                return .unavailable(.modelNotReady)
            @unknown default:
                return .unavailable(.unknown(String(describing: reason)))
            }
        }
    }
}

@MainActor
protocol TextEnhancing: AnyObject {
    func enhance(_ rawText: String, systemPrompt: String) async throws -> String
    func enhance(userPrompt: String) async throws -> String
    func dictionaryHistoryScanTerms(userPrompt: String) async throws -> [String]
    func executeCompiledRequest(_ request: LLMCompiledRequest) async throws -> String
    func translate(
        _ text: String,
        targetLanguage: TranslationTargetLanguage,
        systemPrompt: String
    ) async throws -> String
}

/// Uses Apple Intelligence (on-device Foundation Models) to clean up
/// and enhance raw speech transcription output.
@available(macOS 26.0, *)
@MainActor
class TextEnhancer: TextEnhancing {
    @Generable
    struct EnhancementOutput {
        var resultText: String
    }

    @Generable
    struct DictionaryHistoryScanGeneratedItem {
        var term: String
    }

    @Generable
    struct DictionaryHistoryScanGeneratedOutput {
        var terms: [DictionaryHistoryScanGeneratedItem]
    }

    /// Whether Apple Intelligence is available on this device.
    static var isAvailable: Bool {
        availability.isAvailable
    }

    /// Current Apple Intelligence runtime availability and its user-facing reason.
    static var availability: AppleIntelligenceAvailability {
        AppleIntelligenceAvailability.current
    }

    /// Enhances raw transcribed text by fixing grammar, punctuation,
    /// and formatting while preserving the original meaning.
    /// - Parameters:
    ///   - rawText: The raw transcription output to clean up.
    ///   - systemPrompt: The system prompt that instructs the model how to enhance the text.
    func enhance(_ rawText: String, systemPrompt: String) async throws -> String {
        guard TextEnhancer.isAvailable else {
            return rawText
        }

        let session = LanguageModelSession(
            instructions: systemPrompt
        )

        let response = try await session.respond(
            to: """
            Clean up this transcription while preserving meaning and style.
            Input:
            \(rawText)
            """,
            generating: EnhancementOutput.self
        )

        let enhanced = Self.normalizeResultText(response.content.resultText)
        return enhanced.isEmpty ? rawText : enhanced
    }

    func enhance(userPrompt: String) async throws -> String {
        let input = userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return "" }
        guard TextEnhancer.isAvailable else {
            return input
        }

        let session = LanguageModelSession(instructions: "")
        let response = try await session.respond(
            to: input,
            generating: EnhancementOutput.self
        )

        let enhanced = Self.normalizeResultText(response.content.resultText)
        return enhanced.isEmpty ? input : enhanced
    }

    func dictionaryHistoryScanTerms(userPrompt: String) async throws -> [String] {
        let input = userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return [] }
        guard TextEnhancer.isAvailable else { return [] }

        let session = LanguageModelSession(instructions: "")
        let response = try await session.respond(
            to: input,
            generating: DictionaryHistoryScanGeneratedOutput.self
        )

        return DictionaryHistoryScanResponseParser.normalizeAcceptedTerms(
            from: response.content.terms.map(\.term)
        )
    }

    func executeCompiledRequest(_ request: LLMCompiledRequest) async throws -> String {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return request.fallbackText }
        guard TextEnhancer.isAvailable else { return request.fallbackText }

        let session = LanguageModelSession(instructions: request.instructions)
        let response = try await session.respond(
            to: prompt,
            generating: EnhancementOutput.self
        )

        let result = Self.normalizeResultText(response.content.resultText)
        return result.isEmpty ? request.fallbackText : result
    }

    /// Translates text to the requested target language.
    func translate(
        _ text: String,
        targetLanguage: TranslationTargetLanguage,
        systemPrompt: String
    ) async throws -> String {
        guard TextEnhancer.isAvailable else {
            return text
        }

        let session = LanguageModelSession(
            instructions: systemPrompt
        )

        let response = try await session.respond(
            to: """
            Translate the following text according to the instructions.
            Input:
            \(text)
            """,
            generating: EnhancementOutput.self
        )

        let translated = Self.normalizeResultText(response.content.resultText)
        return translated.isEmpty ? text : translated
    }

    private static func normalizeResultText(_ output: String) -> String {
        LLMVisibleOutputSanitizer.sanitize(
            output,
            fallbackText: "",
            taskKind: .generic
        ).text
    }

    private static func unwrapCodeFenceIfNeeded(_ text: String) -> String {
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
