// LLMVisibleOutputSanitizer.swift
// Provides LLMVisible Output Sanitizer for LLM requests and response handling.

import Foundation

enum LLMVisibleOutputSanitizer {
    enum TaskKind {
        case enhancement
        case translation
        case rewrite
        case generic

        var requiresStrictFinalText: Bool {
            switch self {
            case .enhancement, .translation:
                return true
            case .rewrite, .generic:
                return false
            }
        }
    }

    struct SanitizedOutput: Equatable {
        let text: String
        let didFallback: Bool
        let didExtractFinalOutput: Bool
        let didRemoveProcessText: Bool
    }

    static func sanitize(
        _ output: String,
        fallbackText: String,
        taskKind: TaskKind
    ) -> SanitizedOutput {
        let withoutThink = stripThinkBlocks(from: output)
        let unfenced = unwrapCodeFenceIfNeeded(withoutThink)
        let cleaned = unfenced.trimmingCharacters(in: .whitespacesAndNewlines)
        let removedThink = cleaned != output.trimmingCharacters(in: .whitespacesAndNewlines)

        guard taskKind.requiresStrictFinalText else {
            return SanitizedOutput(
                text: cleaned.isEmpty ? fallbackText : cleaned,
                didFallback: cleaned.isEmpty,
                didExtractFinalOutput: false,
                didRemoveProcessText: removedThink
            )
        }

        if let finalOutput = extractFinalOutput(from: cleaned),
           !containsProcessMarkers(finalOutput) {
            return SanitizedOutput(
                text: finalOutput,
                didFallback: false,
                didExtractFinalOutput: true,
                didRemoveProcessText: true
            )
        }

        if cleaned.isEmpty || containsProcessMarkers(cleaned) {
            return SanitizedOutput(
                text: fallbackText,
                didFallback: true,
                didExtractFinalOutput: false,
                didRemoveProcessText: true
            )
        }

        return SanitizedOutput(
            text: cleaned,
            didFallback: false,
            didExtractFinalOutput: false,
            didRemoveProcessText: removedThink
        )
    }

    static func stripThinkBlocks(from text: String) -> String {
        var cleaned = text
        if let regex = try? NSRegularExpression(pattern: "<think>[\\s\\S]*?</think>", options: [.caseInsensitive]) {
            let range = NSRange(location: 0, length: (cleaned as NSString).length)
            cleaned = regex.stringByReplacingMatches(in: cleaned, options: [], range: range, withTemplate: "")
        }
        cleaned = cleaned.replacingOccurrences(of: "<think>", with: "", options: .caseInsensitive)
        cleaned = cleaned.replacingOccurrences(of: "</think>", with: "", options: .caseInsensitive)
        return cleaned
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
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func extractFinalOutput(from text: String) -> String? {
        let markers = [
            "Final Output:",
            "Final output:",
            "FINAL OUTPUT:",
            "Output:",
            "Result:",
            "Cleaned Text:",
            "Cleaned text:",
            "最终输出：",
            "最终输出:",
            "输出：",
            "输出:"
        ]

        for marker in markers {
            guard let range = text.range(of: marker, options: [.backwards, .caseInsensitive]) else {
                continue
            }
            let candidate = text[range.upperBound...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'`"))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let unfenced = unwrapCodeFenceIfNeeded(String(candidate))
            if !unfenced.isEmpty {
                return unfenced
            }
        }

        return nil
    }

    private static func containsProcessMarkers(_ text: String) -> Bool {
        let normalizedLines = text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }

        let linePrefixes = [
            "thinking process",
            "thought process",
            "analysis",
            "reasoning",
            "role:",
            "assistant role",
            "apply rules",
            "applying rules",
            "step 1",
            "step 2",
            "rule 1",
            "final output",
            "cleaning process",
            "process:",
            "思考过程",
            "推理过程",
            "分析：",
            "分析:",
            "最终输出"
        ]

        if normalizedLines.contains(where: { line in
            linePrefixes.contains(where: { line.hasPrefix($0) })
        }) {
            return true
        }

        let lowercased = text.lowercased()
        let inlineMarkers = [
            "thinking process:",
            "thought process:",
            "apply rules:",
            "final output:",
            "role: voxt",
            "transcription cleanup assistant"
        ]
        return inlineMarkers.contains { lowercased.contains($0) }
    }
}
