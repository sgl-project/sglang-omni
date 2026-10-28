// EnhancementPromptResolver.swift
// Provides Enhancement Prompt Resolver for core app behavior.

import Foundation

struct EnhancementPromptResolver {
    enum Delivery: Equatable {
        case systemPrompt
        case userMessage
        case skipEnhancement
    }

    enum GlobalFallbackReason: Equatable {
        case appBranchDisabled
        case noGroups
        case browserURLUnavailable(bundleID: String?)
        case browserURLNoMatch(bundleID: String?, url: String)
        case noGroupMatch(bundleID: String?)
    }

    enum Source: Equatable {
        case globalDefault(GlobalFallbackReason)
        case appGroup(groupName: String, bundleID: String)
        case appGroupFallback(groupName: String, bundleID: String)
        case urlGroup(groupName: String, pattern: String, url: String)
        case urlGroupFallback(groupName: String, pattern: String, url: String)
    }

    struct PromptContext: Equatable {
        let focusedAppName: String?
        let matchedGroupID: UUID?
        let matchedAppGroupName: String?
        let matchedURLGroupName: String?
    }

    struct Input {
        let globalPrompt: String
        let rawTranscription: String
        let userMainLanguagePromptValue: String
        let userOtherLanguagesPromptValue: String
        let dictionaryGlossary: String?
        let appEnhancementEnabled: Bool
        let groups: [AppBranchGroup]
        let urlsByID: [UUID: String]
        let frontmostBundleID: String?
        let focusedAppName: String?
        let normalizedActiveURL: String?
        let supportedBrowserBundleIDs: Set<String>
    }

    struct Output: Equatable {
        let content: String
        let delivery: Delivery
        let promptContext: PromptContext
        let source: Source
    }

    static let rawTranscriptionTemplateVariable = "{{RAW_TRANSCRIPTION}}"
    static let userMainLanguageTemplateVariable = "{{USER_MAIN_LANGUAGE}}"

    static func resolve(_ input: Input) -> Output {
        let fallbackPrompt = resolvedGlobalPrompt(input.globalPrompt)
        let fallbackDelivery = fallbackDelivery(for: fallbackPrompt)

        func makeFallback(reason: GlobalFallbackReason) -> Output {
            Output(
                content: resolvedPrompt(
                    template: fallbackPrompt,
                    rawTranscription: input.rawTranscription,
                    userMainLanguagePromptValue: input.userMainLanguagePromptValue,
                    userOtherLanguagesPromptValue: input.userOtherLanguagesPromptValue
                ),
                delivery: fallbackDelivery,
                promptContext: PromptContext(
                    focusedAppName: input.focusedAppName,
                    matchedGroupID: nil,
                    matchedAppGroupName: nil,
                    matchedURLGroupName: nil
                ),
                source: .globalDefault(reason)
            )
        }

        guard input.appEnhancementEnabled else {
            return makeFallback(reason: .appBranchDisabled)
        }

        guard !input.groups.isEmpty else {
            return makeFallback(reason: .noGroups)
        }

        if let bundleID = input.frontmostBundleID,
           input.supportedBrowserBundleIDs.contains(bundleID) {
            guard let normalizedActiveURL = input.normalizedActiveURL else {
                return makeFallback(reason: .browserURLUnavailable(bundleID: bundleID))
            }

            if let match = AppBranchURLPatternService.firstPromptMatch(
                groups: input.groups,
                urlsByID: input.urlsByID,
                normalizedURL: normalizedActiveURL
            ) {
                return Output(
                    content: resolvedPrompt(
                        template: match.prompt,
                        rawTranscription: input.rawTranscription,
                        userMainLanguagePromptValue: input.userMainLanguagePromptValue,
                        userOtherLanguagesPromptValue: input.userOtherLanguagesPromptValue
                    ),
                    delivery: .systemPrompt,
                    promptContext: PromptContext(
                        focusedAppName: input.focusedAppName,
                        matchedGroupID: match.groupID,
                        matchedAppGroupName: nil,
                        matchedURLGroupName: match.groupName
                    ),
                    source: .urlGroup(groupName: match.groupName, pattern: match.pattern, url: normalizedActiveURL)
                )
            }

            if let match = AppBranchURLPatternService.firstGroupMatch(
                groups: input.groups,
                urlsByID: input.urlsByID,
                normalizedURL: normalizedActiveURL
            ) {
                return Output(
                    content: resolvedPrompt(
                        template: fallbackPrompt,
                        rawTranscription: input.rawTranscription,
                        userMainLanguagePromptValue: input.userMainLanguagePromptValue,
                        userOtherLanguagesPromptValue: input.userOtherLanguagesPromptValue
                    ),
                    delivery: fallbackDelivery,
                    promptContext: PromptContext(
                        focusedAppName: input.focusedAppName,
                        matchedGroupID: match.groupID,
                        matchedAppGroupName: nil,
                        matchedURLGroupName: match.groupName
                    ),
                    source: .urlGroupFallback(
                        groupName: match.groupName,
                        pattern: match.pattern,
                        url: normalizedActiveURL
                    )
                )
            }

            return makeFallback(reason: .browserURLNoMatch(bundleID: bundleID, url: normalizedActiveURL))
        }

        if let bundleID = input.frontmostBundleID,
           let group = input.groups.first(where: { $0.appBundleIDs.contains(bundleID) }) {
            let prompt = group.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !prompt.isEmpty {
                return Output(
                    content: resolvedPrompt(
                        template: prompt,
                        rawTranscription: input.rawTranscription,
                        userMainLanguagePromptValue: input.userMainLanguagePromptValue,
                        userOtherLanguagesPromptValue: input.userOtherLanguagesPromptValue
                    ),
                    delivery: .systemPrompt,
                    promptContext: PromptContext(
                        focusedAppName: input.focusedAppName,
                        matchedGroupID: group.id,
                        matchedAppGroupName: group.name,
                        matchedURLGroupName: nil
                    ),
                    source: .appGroup(groupName: group.name, bundleID: bundleID)
                )
            }

            return Output(
                content: resolvedPrompt(
                    template: fallbackPrompt,
                    rawTranscription: input.rawTranscription,
                    userMainLanguagePromptValue: input.userMainLanguagePromptValue,
                    userOtherLanguagesPromptValue: input.userOtherLanguagesPromptValue
                ),
                delivery: fallbackDelivery,
                promptContext: PromptContext(
                    focusedAppName: input.focusedAppName,
                    matchedGroupID: group.id,
                    matchedAppGroupName: group.name,
                    matchedURLGroupName: nil
                ),
                source: .appGroupFallback(groupName: group.name, bundleID: bundleID)
            )
        }

        return makeFallback(reason: .noGroupMatch(bundleID: input.frontmostBundleID))
    }

    private static func resolvedGlobalPrompt(_ prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? AppPromptDefaults.text(for: .enhancement) : trimmed
    }

    private static func fallbackDelivery(for prompt: String) -> Delivery {
        prompt.contains(rawTranscriptionTemplateVariable) ? .userMessage : .systemPrompt
    }

    private static func resolvedPrompt(
        template: String,
        rawTranscription: String,
        userMainLanguagePromptValue: String,
        userOtherLanguagesPromptValue: String
    ) -> String {
        let resolved = template
            .replacingOccurrences(of: rawTranscriptionTemplateVariable, with: rawTranscription)
            .replacingOccurrences(of: userMainLanguageTemplateVariable, with: userMainLanguagePromptValue)
        return [
            resolved,
            enhancementLanguagePreservationRules(
                userMainLanguagePromptValue: userMainLanguagePromptValue,
                userOtherLanguagesPromptValue: userOtherLanguagesPromptValue
            )
        ]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }

    private static func enhancementLanguagePreservationRules(
        userMainLanguagePromptValue: String,
        userOtherLanguagesPromptValue: String
    ) -> String {
        let otherLanguages = normalizedOtherLanguagesPromptValue(userOtherLanguagesPromptValue)
        let otherLanguagesLine = otherLanguages.map {
            "Other user languages: \($0)."
        } ?? "Other user languages: None."

        return """
        Language rule:
        - User main language: \(userMainLanguagePromptValue).
        - \(otherLanguagesLine)
        - Use this only for punctuation, formatting, filler cleanup, and ambiguity resolution.
        - It is guidance only, not a translation target. Preserve the original language mix.
        """
    }

    private static func normalizedOtherLanguagesPromptValue(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard trimmed.caseInsensitiveCompare(DictionaryHistoryScanPromptLanguageSupport.noneValue) != .orderedSame else {
            return nil
        }
        return trimmed
    }
}
