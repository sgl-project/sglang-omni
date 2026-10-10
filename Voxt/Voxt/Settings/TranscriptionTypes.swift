// TranscriptionTypes.swift
// Provides Transcription Types for settings screens.

import SwiftUI

enum TranscriptionEngine: String, CaseIterable, Identifiable {
    case dictation
    case mlxAudio
    case remote

    var id: String { rawValue }

    static func resolved(rawValue: String?) -> TranscriptionEngine {
        if rawValue == "whisperKit" {
            return .mlxAudio
        }
        if rawValue == "sherpaOnnx" {
            return .mlxAudio
        }
        return TranscriptionEngine(rawValue: rawValue ?? "") ?? .mlxAudio
    }

    var titleKey: LocalizedStringKey {
        switch self {
        case .dictation: return "Direct Dictation"
        case .mlxAudio: return "MLX Audio (On-device)"
        case .remote: return "Remote"
        }
    }

    var title: String {
        switch self {
        case .dictation: return AppLocalization.localizedString("Direct Dictation")
        case .mlxAudio: return AppLocalization.localizedString("MLX Audio (On-device)")
        case .remote: return AppLocalization.localizedString("Remote")
        }
    }

    var description: String {
        switch self {
        case .dictation:
            return AppLocalization.localizedString("Uses Apple's built-in speech recognition. Works immediately with no setup.")
        case .mlxAudio:
            return AppLocalization.localizedString("Uses MLX Audio speech models running locally. Requires a one-time model download.")
        case .remote:
            return AppLocalization.localizedString("Uses remote speech recognition providers and cloud-hosted ASR models.")
        }
    }
}

enum EnhancementMode: String, CaseIterable, Identifiable {
    case off
    case appleIntelligence
    case customLLM
    case remoteLLM

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .off: return "Off"
        case .appleIntelligence: return "Apple Intelligence"
        case .customLLM: return "Custom LLM"
        case .remoteLLM: return "Remote LLM"
        }
    }

    var title: String {
        switch self {
        case .off: return AppLocalization.localizedString("Off")
        case .appleIntelligence: return AppLocalization.localizedString("Apple Intelligence")
        case .customLLM: return AppLocalization.localizedString("Custom LLM")
        case .remoteLLM: return AppLocalization.localizedString("Remote LLM")
        }
    }

    static func availableModes(appleIntelligenceAvailable: Bool) -> [EnhancementMode] {
        allCases.filter { mode in
            mode != .appleIntelligence || appleIntelligenceAvailable
        }
    }

    static func resolved(
        storedRawValue: String?,
        appleIntelligenceAvailable: Bool,
        customLLMAvailable: Bool,
        remoteLLMAvailable: Bool
    ) -> EnhancementMode {
        let requestedMode = EnhancementMode(rawValue: storedRawValue ?? "") ?? .off
        guard requestedMode == .appleIntelligence, !appleIntelligenceAvailable else {
            return requestedMode
        }

        if customLLMAvailable {
            return .customLLM
        }

        if remoteLLMAvailable {
            return .remoteLLM
        }

        return .off
    }
}

enum TranslationModelProvider: String, CaseIterable, Identifiable {
    case customLLM
    case localGGUF
    case remoteLLM

    var id: String { rawValue }

    static func resolved(rawValue: String?) -> TranslationModelProvider {
        if rawValue == "whisperKit" {
            return .customLLM
        }
        return TranslationModelProvider(rawValue: rawValue ?? "") ?? .customLLM
    }

    var titleKey: LocalizedStringKey {
        switch self {
        case .customLLM: return "Custom LLM"
        case .localGGUF: return "Local GGUF"
        case .remoteLLM: return "Remote LLM"
        }
    }

    var title: String {
        switch self {
        case .customLLM: return AppLocalization.localizedString("Custom LLM")
        case .localGGUF: return AppLocalization.localizedString("Local GGUF")
        case .remoteLLM: return AppLocalization.localizedString("Remote LLM")
        }
    }
}

struct TranslationProviderResolution: Equatable {
    let provider: TranslationModelProvider
    let fallbackProvider: TranslationModelProvider
}

enum TranslationProviderResolver {
    static func sanitizedFallbackProvider(_ provider: TranslationModelProvider) -> TranslationModelProvider {
        provider
    }

    static func resolve(
        selectedProvider: TranslationModelProvider,
        fallbackProvider: TranslationModelProvider,
        transcriptionEngine: TranscriptionEngine,
        targetLanguage: TranslationTargetLanguage,
        isSelectedTextTranslation: Bool
    ) -> TranslationProviderResolution {
        let sanitizedFallback = sanitizedFallbackProvider(fallbackProvider)
        return TranslationProviderResolution(
            provider: selectedProvider,
            fallbackProvider: sanitizedFallback
        )
    }

    static func warningMessage(
        selectedProvider: TranslationModelProvider,
        transcriptionEngine: TranscriptionEngine,
        targetLanguage: TranslationTargetLanguage
    ) -> String? {
        nil
    }
}

enum RewriteModelProvider: String, CaseIterable, Identifiable {
    case customLLM
    case remoteLLM

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .customLLM: return "Custom LLM"
        case .remoteLLM: return "Remote LLM"
        }
    }

    var title: String {
        switch self {
        case .customLLM: return AppLocalization.localizedString("Custom LLM")
        case .remoteLLM: return AppLocalization.localizedString("Remote LLM")
        }
    }
}

enum OverlayPosition: String, CaseIterable, Identifiable {
    case bottom
    case top

    var id: String { rawValue }

    var titleKey: LocalizedStringKey {
        switch self {
        case .bottom: return "Bottom"
        case .top: return "Top"
        }
    }

    var title: String {
        switch self {
        case .bottom: return AppLocalization.localizedString("Bottom")
        case .top: return AppLocalization.localizedString("Top")
        }
    }
}
