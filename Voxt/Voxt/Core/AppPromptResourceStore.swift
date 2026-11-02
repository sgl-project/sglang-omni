// AppPromptResourceStore.swift
// Provides App Prompt Resource Store for core app behavior.

import Foundation

enum LocalizedPromptResource: String, CaseIterable {
    case enhancement
    case translation
    case rewrite
    case transcriptSummary = "transcript-summary"
    case dictionaryIngest = "dictionary-ingest"
    case dictionaryAutoLearning = "dictionary-auto-learning"
    case dictionaryAutoLearningRuntimeConstraints = "dictionary-auto-learning-runtime-constraints"
    case qwenASRContextBias = "qwen-asr-context-bias"
    case openAIASRHint = "openai-asr-hint"
    case glmASRHint = "glm-asr-hint"
    case onboardingTranscriptionEnhancement = "onboarding-transcription-enhancement"
    case onboardingAppEnhancement = "onboarding-app-enhancement"
}

nonisolated enum SharedPromptResource: String, CaseIterable {
    case mossTimestampedDiarization = "moss-timestamped-diarization"
    case mossSpeakerOnly = "moss-speaker-only"
    case mossPlainText = "moss-plain-text"
    case transcriptFollowUp = "transcript-follow-up"
    case savedTranscriptionFollowUp = "saved-transcription-follow-up"
}

enum AppPromptResourceStore {
    static var registeredLocalizedResourceNames: Set<String> {
        Set(LocalizedPromptResource.allCases.map(\.rawValue))
            .union(FeaturePromptPresetCatalog.localizedResourceNames)
    }

    nonisolated static var registeredSharedResourceNames: Set<String> {
        Set(SharedPromptResource.allCases.map(\.rawValue))
    }

    static func text(for kind: AppPromptKind, language: AppInterfaceLanguage) -> String? {
        text(for: kind.promptResource, language: language)
    }

    static func text(
        for resource: LocalizedPromptResource,
        language: AppInterfaceLanguage
    ) -> String? {
        let languageDirectory = resourceLanguageDirectory(for: language)
        let resourceName = "\(languageDirectory)-\(resource.rawValue)"
        let subdirectory = "Resources/Prompts/\(languageDirectory)"
        return loadText(resourceName: resourceName, subdirectory: subdirectory)
    }

    nonisolated static func text(for resource: SharedPromptResource) -> String? {
        loadText(
            resourceName: resource.rawValue,
            subdirectory: "Resources/Prompts/shared"
        )
    }

    static func requiredText(
        for resource: LocalizedPromptResource,
        language: AppInterfaceLanguage
    ) -> String {
        guard let text = text(for: resource, language: language),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            preconditionFailure("Missing bundled prompt resource \(resource.rawValue) for \(language.rawValue)")
        }
        return text
    }

    nonisolated static func requiredText(for resource: SharedPromptResource) -> String {
        guard let text = text(for: resource),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            preconditionFailure("Missing bundled shared prompt resource \(resource.rawValue)")
        }
        return text
    }

    static func presetText(
        for kind: AppPromptKind,
        presetID: String,
        language: AppInterfaceLanguage
    ) -> String? {
        let languageDirectory = resourceLanguageDirectory(for: language)
        let resourceName = "\(languageDirectory)-\(kind.promptResource.rawValue)-\(presetID)"
        let subdirectory = "Resources/Prompts/\(languageDirectory)"
        return loadText(resourceName: resourceName, subdirectory: subdirectory)
    }

    private static func resourceLanguageDirectory(for language: AppInterfaceLanguage) -> String {
        switch language {
        case .english:
            return "en"
        case .chineseSimplified:
            return "zh-Hans"
        case .japanese:
            return "ja"
        case .system:
            return resourceLanguageDirectory(for: .resolvedSystemLanguage)
        }
    }

    nonisolated private static func loadText(
        resourceName: String,
        subdirectory: String
    ) -> String? {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "txt")
            ?? Bundle.main.url(
                forResource: resourceName,
                withExtension: "txt",
                subdirectory: subdirectory
            ),
            let text = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }
        return text.removingSingleTrailingNewline()
    }
}

extension AppPromptKind {
    var promptResource: LocalizedPromptResource {
        switch self {
        case .enhancement:
            return .enhancement
        case .translation:
            return .translation
        case .rewrite:
            return .rewrite
        case .transcriptSummary:
            return .transcriptSummary
        case .dictionaryIngest:
            return .dictionaryIngest
        case .dictionaryAutoLearning:
            return .dictionaryAutoLearning
        case .qwenASRContextBias:
            return .qwenASRContextBias
        case .openAIASRHint:
            return .openAIASRHint
        case .glmASRHint:
            return .glmASRHint
        }
    }
}

nonisolated private extension String {
    func removingSingleTrailingNewline() -> String {
        if hasSuffix("\r\n") {
            return String(dropLast(2))
        }
        if hasSuffix("\n") {
            return String(dropLast())
        }
        return self
    }
}
