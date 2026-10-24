// TranscriptSummarySupport.swift
// Provides Transcript Summary Support for core app behavior.

import Foundation

enum TranscriptSummaryChatRole: String, Codable, Hashable, Sendable {
    case user
    case assistant
}

struct TranscriptSummaryChatMessage: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    let role: TranscriptSummaryChatRole
    let content: String
    let createdAt: Date

    init(
        id: UUID = UUID(),
        role: TranscriptSummaryChatRole,
        content: String,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.createdAt = createdAt
    }
}

struct TranscriptSummarySettingsSnapshot: Codable, Hashable, Sendable {
    let autoGenerate: Bool
    let promptTemplate: String?
    let modelSelectionID: String?

    init(
        autoGenerate: Bool,
        promptTemplate: String? = nil,
        modelSelectionID: String? = nil
    ) {
        self.autoGenerate = autoGenerate
        self.promptTemplate = promptTemplate
        self.modelSelectionID = modelSelectionID
    }
}

struct TranscriptSummarySnapshot: Codable, Hashable, Sendable {
    let title: String
    let body: String
    let todoItems: [String]
    let generatedAt: Date
    let settingsSnapshot: TranscriptSummarySettingsSnapshot
}

struct TranscriptSummaryProviderStatus: Equatable, Sendable {
    let isAvailable: Bool
    let message: String
}

struct TranscriptSummaryModelOption: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let subtitle: String
}

enum TranscriptSummarySupport {
    static let transcriptRecordTemplateVariable = "{{TRANSCRIPT_RECORD}}"
    static let promptTemplateVariables = [
        AppPreferenceKey.asrUserMainLanguageTemplateVariable,
        transcriptRecordTemplateVariable
    ]

    private struct DecodedPayload: Decodable {
        struct SummaryBlock: Decodable {
            let title: String?
            let content: String?
            let body: String?
        }

        let transcriptSummary: SummaryBlock?
        let title: String?
        let body: String?
        let content: String?
        let todoList: [String]?
        let todoItems: [String]?

        enum CodingKeys: String, CodingKey {
            case transcriptSummary = "transcript_summary"
            case title
            case body
            case content
            case todoList = "todo_list"
            case todoItems
        }
    }

    static func defaultPromptTemplate() -> String {
        AppPromptDefaults.text(for: .transcriptSummary, language: .english)
    }

    static func summaryPrompt(
        transcript: String,
        settings: TranscriptSummarySettingsSnapshot,
        userMainLanguage: String
    ) -> String {
        let template = resolvedPromptTemplate(settings.promptTemplate)
        return resolvePromptTemplate(
            template: template,
            userMainLanguage: userMainLanguage,
            transcriptRecord: transcript
        )
    }

    static func resolvedPromptTemplate(_ promptTemplate: String?) -> String {
        AppPromptDefaults.resolvedStoredText(promptTemplate, kind: .transcriptSummary)
    }

    static func transcriptRecord(from values: [String: String]) -> String {
        values[transcriptRecordTemplateVariable]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private static func resolvePromptTemplate(
        template: String,
        userMainLanguage: String,
        transcriptRecord: String
    ) -> String {
        var prompt = template.trimmingCharacters(in: .whitespacesAndNewlines)
        let languageVariable = AppPreferenceKey.asrUserMainLanguageTemplateVariable

        if prompt.contains(languageVariable) {
            prompt = prompt.replacingOccurrences(of: languageVariable, with: userMainLanguage)
        } else {
            prompt += "\n\nUser main language: \(userMainLanguage)"
        }

        if prompt.contains(transcriptRecordTemplateVariable) {
            prompt = prompt.replacingOccurrences(of: transcriptRecordTemplateVariable, with: transcriptRecord)
        } else {
            prompt += "\n\nTranscript:\n\(transcriptRecord)"
        }

        return prompt
    }

    static func followUpPrompt(
        transcript: String,
        summary: TranscriptSummarySnapshot?,
        history: [TranscriptSummaryChatMessage],
        question: String,
        userMainLanguage: String
    ) -> String {
        let trimmedHistory = history
            .map { message in
                let roleLabel = message.role == .user ? "User" : "Assistant"
                return "\(roleLabel): \(message.content)"
            }
            .joined(separator: "\n")
        let summaryBlock: String
        if let summary {
            let todoText = summary.todoItems.isEmpty
                ? "None"
                : summary.todoItems.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
            summaryBlock = """
            Title: \(summary.title)
            Body:
            \(summary.body)

            TODO:
            \(todoText)
            """
        } else {
            summaryBlock = "No generated summary is available yet."
        }

        return AppPromptResourceStore.requiredText(for: .transcriptFollowUp)
            .replacingOccurrences(of: "{{USER_MAIN_LANGUAGE}}", with: userMainLanguage)
            .replacingOccurrences(of: "{{CURRENT_SUMMARY}}", with: summaryBlock)
            .replacingOccurrences(
                of: "{{PREVIOUS_CHAT}}",
                with: trimmedHistory.isEmpty ? "None" : trimmedHistory
            )
            .replacingOccurrences(of: "{{TRANSCRIPT}}", with: transcript)
            .replacingOccurrences(of: "{{QUESTION}}", with: question)
    }

    static func decodeSummary(
        from text: String,
        settings: TranscriptSummarySettingsSnapshot,
        generatedAt: Date = Date()
    ) -> TranscriptSummarySnapshot? {
        let normalized = normalizePayload(text)
        guard let data = normalized.data(using: .utf8),
              let payload = try? JSONDecoder().decode(DecodedPayload.self, from: data)
        else {
            return decodeLooseSummary(from: normalized, settings: settings, generatedAt: generatedAt)
        }
        return snapshot(from: payload, settings: settings, generatedAt: generatedAt)
    }

    static func fallbackSummaryTitle(for settings: TranscriptSummarySettingsSnapshot) -> String {
        AppLocalization.localizedString("Transcript Summary")
    }

    private static func snapshot(
        from payload: DecodedPayload,
        settings: TranscriptSummarySettingsSnapshot,
        generatedAt: Date
    ) -> TranscriptSummarySnapshot? {
        let summaryBlock = payload.transcriptSummary
        let title = decodedText(summaryBlock?.title ?? payload.title)
        let body = decodedText(summaryBlock?.content ?? summaryBlock?.body ?? payload.body ?? payload.content)
        let todoItems = decodedTodoItems(payload.todoList ?? payload.todoItems ?? [])
        guard !body.isEmpty || !todoItems.isEmpty else { return nil }
        return TranscriptSummarySnapshot(
            title: title.isEmpty ? fallbackSummaryTitle(for: settings) : title,
            body: body,
            todoItems: todoItems,
            generatedAt: generatedAt,
            settingsSnapshot: settings
        )
    }

    private static func decodeLooseSummary(
        from text: String,
        settings: TranscriptSummarySettingsSnapshot,
        generatedAt: Date
    ) -> TranscriptSummarySnapshot? {
        let xmlTitle = normalizeEscapedText(firstMatch(
            in: text,
            patterns: [
                #"(?is)<title>\s*([\s\S]*?)\s*</title>"#
            ]
        )).trimmingCharacters(in: .whitespacesAndNewlines)
        let xmlBody = normalizeEscapedText(firstMatch(
            in: text,
            patterns: [
                #"(?is)<content>\s*([\s\S]*?)\s*</content>"#
            ]
        )).trimmingCharacters(in: .whitespacesAndNewlines)
        let xmlTodoBlock = firstMatch(
            in: text,
            patterns: [
                #"(?is)<todo_list>\s*([\s\S]*?)\s*</todo_list>"#
            ]
        )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        let title = !xmlTitle.isEmpty ? xmlTitle : normalizeEscapedText(firstMatch(
            in: text,
            patterns: [
                #"(?is)(?:^|\n)\s*["']?title["']?\s*[:：]\s*["']?(.+?)["']?(?=\n\s*["']?(?:body|summary|content)["']?\s*[:：]|\n{2,}|$)"#
            ]
        ) ?? fallbackSummaryTitle(for: settings))
        let body = !xmlBody.isEmpty ? xmlBody : normalizeEscapedText(firstMatch(
            in: text,
            patterns: [
                #"(?is)(?:^|\n)\s*["']?(?:body|summary|content)["']?\s*[:：]\s*([\s\S]+?)(?=\n\s*["']?(?:todoItems|todos|actionItems)["']?\s*[:：]|\s*$)"#
            ]
        )).trimmingCharacters(in: .whitespacesAndNewlines)
        let todoBlock = xmlTodoBlock.isEmpty ? (firstMatch(
            in: text,
            patterns: [
                #"(?is)(?:^|\n)\s*["']?(?:todoItems|todos|actionItems)["']?\s*[:：]\s*([\s\S]+?)\s*$"#
            ]
        ) ?? "") : xmlTodoBlock
        let todoItems = normalizedLooseTodoItems(
            todoBlock
                .components(separatedBy: .newlines)
                .map { $0.replacingOccurrences(of: #"^[\-\*\d\.\)\s]+"#, with: "", options: .regularExpression) }
        )
        guard !body.isEmpty || !todoItems.isEmpty else { return nil }
        return TranscriptSummarySnapshot(
            title: title.trimmingCharacters(in: .whitespacesAndNewlines),
            body: body,
            todoItems: todoItems,
            generatedAt: generatedAt,
            settingsSnapshot: settings
        )
    }

    private static func decodedText(_ value: String?) -> String {
        value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private static func decodedTodoItems(_ values: [String]) -> [String] {
        values
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func normalizedLooseTodoItems(_ values: [String]) -> [String] {
        values
            .map { normalizeEscapedText($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func normalizeEscapedText(_ value: String?) -> String {
        guard let value else { return "" }
        return value
            .replacingOccurrences(of: "\\r\\n", with: "\n")
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
    }

    private static func normalizePayload(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let unwrapped: String
        if trimmed.hasPrefix("```"), trimmed.hasSuffix("```") {
            var lines = trimmed.components(separatedBy: .newlines)
            guard lines.count >= 2 else { return trimmed }
            lines.removeFirst()
            if lines.last?.trimmingCharacters(in: .whitespacesAndNewlines) == "```" {
                lines.removeLast()
            }
            unwrapped = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            unwrapped = trimmed
        }
        return extractJSONObject(from: unwrapped) ?? unwrapped
    }

    private static func extractJSONObject(from text: String) -> String? {
        guard let startIndex = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var isEscaped = false

        for index in text[startIndex...].indices {
            let character = text[index]
            if inString {
                if isEscaped {
                    isEscaped = false
                    continue
                }
                if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    inString = false
                }
                continue
            }

            if character == "\"" {
                inString = true
                continue
            }
            if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 {
                    return String(text[startIndex...index])
                }
            }
        }
        return nil
    }

    private static func firstMatch(in text: String, patterns: [String]) -> String? {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            guard let match = regex.firstMatch(in: text, options: [], range: range),
                  match.numberOfRanges > 1,
                  let valueRange = Range(match.range(at: 1), in: text)
            else {
                continue
            }
            let value = String(text[valueRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty {
                return value
            }
        }
        return nil
    }
}
