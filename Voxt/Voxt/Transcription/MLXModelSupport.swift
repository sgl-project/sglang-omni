// MLXModelSupport.swift
// Provides MLXModel Support for transcription engines.

import Foundation
import HuggingFace

nonisolated enum MLXLiveMode: Equatable, Sendable {
    case batchPreview
    case nativeQwenLive
    case nativeStreamingLive
    case nativeNemotronLive
}

nonisolated enum MLXLanguageRouting: Equatable, Sendable {
    case unavailable
    case automatic
    case iso6391(requiresExplicitPrimaryLanguage: Bool)
    case languageName
    case localeOrISO6391
}

nonisolated struct MLXASROutputCapability: OptionSet, Sendable {
    let rawValue: UInt

    nonisolated static let text = Self(rawValue: 1 << 0)
    nonisolated static let timestamps = Self(rawValue: 1 << 1)
    nonisolated static let speakerLabels = Self(rawValue: 1 << 2)
    nonisolated static let language = Self(rawValue: 1 << 3)
    nonisolated static let emotion = Self(rawValue: 1 << 4)
    nonisolated static let audioEvents = Self(rawValue: 1 << 5)
}

nonisolated enum MLXASRTimingGranularity: Equatable, Sendable {
    case none
    case chunk
    case sentence
    case word

    nonisolated var providesReliableSegments: Bool {
        switch self {
        case .sentence, .word:
            return true
        case .none, .chunk:
            return false
        }
    }
}

nonisolated struct MLXASRConfigurationCapability: OptionSet, Sendable {
    let rawValue: UInt

    nonisolated static let recognitionPreset = Self(rawValue: 1 << 0)
    nonisolated static let languageRouting = Self(rawValue: 1 << 1)
    nonisolated static let whisperTemperature = Self(rawValue: 1 << 2)
    nonisolated static let qwenContext = Self(rawValue: 1 << 3)
    nonisolated static let senseVoiceITN = Self(rawValue: 1 << 5)
    nonisolated static let cohereLongForm = Self(rawValue: 1 << 6)
    nonisolated static let mossPromptAndOutput = Self(rawValue: 1 << 7)
    nonisolated static let nemotronLatency = Self(rawValue: 1 << 11)
}

nonisolated enum MLXVADPolicy: Equatable, Sendable {
    case standard
    /// Keep full timeline audio for Final (e.g. MOSS). Local VAD may only gate no-speech.
    case preserveTimeline
    /// Model owns segmentation / long-form VAD. External Final must not speech-trim.
    case modelManaged

    var usesExternalFinalSpeechValidation: Bool {
        self != .modelManaged
    }

    /// Whether `finalizationSamples` may replace full PCM with VAD-filtered speech.
    var allowsExternalFinalSpeechTrim: Bool {
        self == .standard
    }
}

nonisolated struct MLXASRKVCachePolicy: Equatable, Sendable {
    let bits: Int
    let groupSize: Int
    let quantizedStart: Int

    nonisolated static let conservativeQwen = Self(
        bits: 8,
        groupSize: 64,
        quantizedStart: 256
    )
}

nonisolated struct MLXASRPurpose: OptionSet, Sendable {
    let rawValue: UInt

    nonisolated static let dictation = Self(rawValue: 1 << 0)
    nonisolated static let meeting = Self(rawValue: 1 << 1)
}

nonisolated struct MLXASRModelCapability: Equatable, Sendable {
    let family: MLXModelFamily
    let supportedLanguageCodes: Set<String>
    let languageRouting: MLXLanguageRouting
    let liveMode: MLXLiveMode
    let isRealtimeCapable: Bool
    let outputCapabilities: MLXASROutputCapability
    let timingGranularity: MLXASRTimingGranularity
    let configurationCapabilities: MLXASRConfigurationCapability
    let vadPolicy: MLXVADPolicy
    let supportedPurposes: MLXASRPurpose
    let kvCachePolicy: MLXASRKVCachePolicy?

    nonisolated var isMultilingual: Bool { supportedLanguageCodes.count > 1 }

    nonisolated func supportsLanguage(code: String) -> Bool {
        supportedLanguageCodes.contains(code.lowercased())
    }

    @MainActor
    func resolvedLanguage(for language: UserMainLanguageOption) -> String? {
        let baseCode = language.baseLanguageCode
        guard supportsLanguage(code: baseCode) else { return nil }

        switch languageRouting {
        case .unavailable, .automatic:
            return nil
        case .iso6391:
            return baseCode
        case .localeOrISO6391:
            switch language.code {
            case "zh-hans":
                return "zh-CN"
            case "zh-hant":
                return "zh-TW"
            default:
                return baseCode
            }
        case .languageName:
            return language.promptName
        }
    }

    nonisolated var requiresExplicitPrimaryLanguage: Bool {
        switch languageRouting {
        case .iso6391(let required):
            return required
        case .languageName, .localeOrISO6391:
            return true
        default:
            return false
        }
    }
}

enum MLXWhisperMigrationSupport {
    nonisolated static let defaultRepo = "mlx-community/whisper-large-v3-turbo"
    nonisolated static let defaultLegacyModelID = "large-v3"

    nonisolated private static let legacyWhisperModelMap: [String: String] = [
        "tiny": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-small-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": defaultRepo,
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    ]

    nonisolated static func canonicalLegacyModelID(_ modelID: String) -> String {
        let raw = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return defaultLegacyModelID }
        var normalized = raw
            .replacingOccurrences(of: "openai_whisper-", with: "")
            .replacingOccurrences(of: "openai/whisper-", with: "")
        if normalized == "large-v3-v20240930" {
            normalized = "large-v3"
        }
        if legacyWhisperModelMap[normalized] != nil {
            return normalized
        }
        return defaultLegacyModelID
    }

    nonisolated static func repo(forLegacyWhisperModelID modelID: String) -> String {
        let canonicalModelID = canonicalLegacyModelID(modelID)
        return legacyWhisperModelMap[canonicalModelID] ?? defaultRepo
    }

    nonisolated static func isWhisperRepo(_ repo: String) -> Bool {
        MLXModelCatalog.capability(for: repo).family == .whisper
    }
}

struct MLXModelCatalog {
    struct Option: Identifiable, Hashable {
        let id: String
        let title: String
        let description: String
    }

    private struct PresentationMetadata {
        let ratingText: String
        let tagKeys: [String]
    }

    nonisolated static let defaultModelRepo = "mlx-community/Qwen3-ASR-0.6B-4bit"

    nonisolated private static let whisperLanguageCodes: Set<String> = [
        "af", "am", "ar", "as", "az", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy",
        "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha",
        "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kk", "km",
        "kn", "ko", "la", "lb", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms",
        "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa",
        "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg",
        "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh",
    ]

    nonisolated private static let qwenLanguageCodes: Set<String> = [
        "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi",
        "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "tl", "fa", "el", "hu",
        "mk", "ro",
    ]

    nonisolated private static let european25LanguageCodes: Set<String> = [
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv",
        "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
    ]

    nonisolated private static let nemotronReadyLanguageCodes: Set<String> = [
        "en", "es", "fr", "it", "pt", "nl", "de", "tr", "ru", "ar", "hi", "ja", "ko", "vi",
        "uk", "pl", "sv", "cs", "no", "da", "bg", "fi", "hr", "sk", "zh", "hu", "ro", "et",
    ]

    nonisolated private static let cohereLanguageCodes: Set<String> = [
        "zh", "en", "ja", "ko", "vi", "ar", "el", "pl", "nl", "pt", "it", "es", "de", "fr",
    ]

    // Migration only: retired weights are never loaded or scanned.
    nonisolated private static let legacyModelRepoMap: [String: String] = [
        "mlx-community/Parakeet-0.6B": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/GLM-ASR-Nano-4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/FireRedASR2": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/Voxtral-Mini-4B-Realtime-2602": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/Voxtral-Mini-4B-Realtime-2602-6bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/whisper-tiny-mlx": "mlx-community/whisper-small-mlx",
        "mlx-community/whisper-base-mlx": "mlx-community/whisper-small-mlx",
        "mlx-community/Qwen3-ASR-0.6B-6bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/Qwen3-ASR-0.6B-8bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/Qwen3-ASR-0.6B-bf16": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/Qwen3-ASR-1.7B-4bit": "mlx-community/Qwen3-ASR-1.7B-6bit",
        "mlx-community/Qwen3-ASR-1.7B-bf16": "mlx-community/Qwen3-ASR-1.7B-6bit",
        "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/Voxtral-Mini-4B-Realtime-6bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "Mediform/canary-1b-v2-mlx-q8": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "UsefulSensors/moonshine-tiny": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "facebook/wav2vec2-base-960h": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "facebook/mms-1b-fl102": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/parakeet-tdt_ctc-110m": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-tdt-0.6b-v2": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-ctc-0.6b": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-rnnt-0.6b": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-tdt-1.1b": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-tdt_ctc-1.1b": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-ctc-1.1b": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-rnnt-1.1b": "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/GLM-ASR-Nano-2512-4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/granite-4.0-1b-speech-5bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "mlx-community/FireRedASR2-AED-mlx": "mlx-community/Qwen3-ASR-0.6B-4bit",
    ]

    nonisolated private static let allModels: [Option] = [
        Option(
            id: "mlx-community/whisper-large-v3-turbo",
            title: "Whisper Large v3 Turbo",
            description: "Fast Whisper large-v3 family model with the best quality-to-latency balance."
        ),
        Option(
            id: "mlx-community/whisper-large-v3-mlx",
            title: "Whisper Large v3",
            description: "Accuracy-first Whisper model with a heavier local footprint."
        ),
        Option(
            id: "mlx-community/whisper-small-mlx",
            title: "Whisper Small",
            description: "Lower-resource Whisper model for lighter local setups."
        ),
        Option(
            id: "mlx-community/Qwen3-ASR-0.6B-4bit",
            title: "Qwen3 0.6B (4bit)",
            description: "Balanced quality and speed with low memory use."
        ),
        Option(
            id: "mlx-community/Qwen3-ASR-1.7B-6bit",
            title: "Qwen3 1.7B (6bit)",
            description: "High-accuracy flagship model with a balanced memory footprint."
        ),
        Option(
            id: "mlx-community/Qwen3-ASR-1.7B-8bit",
            title: "Qwen3 1.7B (8bit)",
            description: "High-precision 1.7B model for stronger recognition quality."
        ),
        Option(
            id: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
            title: "Cohere 03-2026",
            description: "High-accuracy multilingual encoder-decoder model with punctuation enabled."
        ),
        Option(
            id: "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            title: "MOSS",
            description: "One-pass timestamped transcription and speaker-label model for meeting-style audio."
        ),
        Option(
            id: "mlx-community/parakeet-tdt-0.6b-v3",
            title: "Parakeet v3",
            description: "Fast 25-language European ASR with automatic language detection."
        ),
        Option(
            id: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
            title: "Nemotron",
            description: "Streaming ASR model with cache-aware NeMo-family decoding."
        ),
        Option(
            id: "mlx-community/SenseVoiceSmall",
            title: "SenseVoice",
            description: "Fast multilingual model with built-in language and event detection."
        )
    ]

    nonisolated static let availableModels: [Option] = allModels
    nonisolated static let supportedModels: [Option] = allModels

    nonisolated private static let capabilitiesByRepo: [String: MLXASRModelCapability] = {
        var capabilities: [String: MLXASRModelCapability] = [:]

        func register(
            repos: [String],
            family: MLXModelFamily,
            languages: Set<String>,
            routing: MLXLanguageRouting,
            liveMode: MLXLiveMode = .batchPreview,
            realtime: Bool = false,
            outputs: MLXASROutputCapability = [.text],
            timingGranularity: MLXASRTimingGranularity = .none,
            configuration: MLXASRConfigurationCapability = [],
            vadPolicy: MLXVADPolicy = .standard,
            purposes: MLXASRPurpose = [.dictation, .meeting],
            kvCachePolicy: MLXASRKVCachePolicy? = nil
        ) {
            let capability = MLXASRModelCapability(
                family: family,
                supportedLanguageCodes: languages,
                languageRouting: routing,
                liveMode: liveMode,
                isRealtimeCapable: realtime,
                outputCapabilities: outputs,
                timingGranularity: timingGranularity,
                configurationCapabilities: configuration,
                vadPolicy: vadPolicy,
                supportedPurposes: purposes,
                kvCachePolicy: kvCachePolicy
            )
            repos.forEach { capabilities[$0] = capability }
        }

        register(
            repos: [
                "mlx-community/whisper-large-v3-turbo",
                "mlx-community/whisper-large-v3-mlx",
                "mlx-community/whisper-small-mlx",
            ],
            family: .whisper,
            languages: whisperLanguageCodes,
            routing: .iso6391(requiresExplicitPrimaryLanguage: false),
            outputs: [.text, .timestamps],
            timingGranularity: .chunk,
            configuration: [.languageRouting, .whisperTemperature]
        )
        register(
            repos: [
                "mlx-community/Qwen3-ASR-0.6B-4bit",
                "mlx-community/Qwen3-ASR-1.7B-6bit",
                "mlx-community/Qwen3-ASR-1.7B-8bit",
            ],
            family: .qwen3ASR,
            languages: qwenLanguageCodes,
            routing: .languageName,
            liveMode: .nativeQwenLive,
            outputs: [.text, .timestamps],
            timingGranularity: .chunk,
            configuration: [.recognitionPreset, .languageRouting, .qwenContext],
            kvCachePolicy: .conservativeQwen
        )
        register(
            repos: ["beshkenadze/cohere-transcribe-03-2026-mlx-fp16"],
            family: .cohereTranscribe,
            languages: cohereLanguageCodes,
            routing: .iso6391(requiresExplicitPrimaryLanguage: true),
            liveMode: .nativeStreamingLive,
            realtime: true,
            configuration: [.recognitionPreset, .languageRouting, .cohereLongForm],
            vadPolicy: .modelManaged
        )
        register(
            repos: ["OpenMOSS-Team/MOSS-Transcribe-Diarize"],
            family: .mossTranscribeDiarize,
            languages: ["zh", "en"],
            routing: .automatic,
            liveMode: .nativeStreamingLive,
            realtime: true,
            outputs: [.text, .timestamps, .speakerLabels, .audioEvents],
            timingGranularity: .sentence,
            configuration: [.mossPromptAndOutput],
            vadPolicy: .preserveTimeline
        )
        register(
            repos: ["mlx-community/parakeet-tdt-0.6b-v3"],
            family: .parakeet,
            languages: european25LanguageCodes,
            routing: .automatic,
            outputs: [.text, .timestamps],
            timingGranularity: .sentence
        )
        register(
            repos: ["mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"],
            family: .nemotronASR,
            languages: nemotronReadyLanguageCodes,
            routing: .localeOrISO6391,
            liveMode: .nativeNemotronLive,
            realtime: true,
            outputs: [.text, .timestamps, .language],
            timingGranularity: .sentence,
            configuration: [.languageRouting, .nemotronLatency],
            vadPolicy: .modelManaged
        )
        register(
            repos: ["mlx-community/SenseVoiceSmall"],
            family: .senseVoice,
            languages: ["zh", "en", "yue", "ja", "ko"],
            routing: .iso6391(requiresExplicitPrimaryLanguage: false),
            outputs: [.text, .language, .emotion, .audioEvents],
            // Keep default `.standard` so meeting external final-speech validation still runs.
            // Dictation Final avoids PCM speech-trim via family-aware finalizationSamples.
            configuration: [.languageRouting, .senseVoiceITN]
        )
        return capabilities
    }()

    nonisolated private static let presentationByRepo: [String: PresentationMetadata] = [
        "mlx-community/whisper-large-v3-turbo": PresentationMetadata(ratingText: "4.8", tagKeys: ["Multilingual", "Fast", "Balanced"]),
        "mlx-community/whisper-large-v3-mlx": PresentationMetadata(ratingText: "4.9", tagKeys: ["Multilingual", "Accurate"]),
        "mlx-community/whisper-small-mlx": PresentationMetadata(ratingText: "4.5", tagKeys: ["Multilingual", "Fast"]),
        "mlx-community/Qwen3-ASR-0.6B-4bit": PresentationMetadata(ratingText: "4.4", tagKeys: ["Multilingual", "Realtime", "Fast"]),
        "mlx-community/Qwen3-ASR-1.7B-6bit": PresentationMetadata(ratingText: "4.8", tagKeys: ["Multilingual", "Realtime", "Accurate"]),
        "mlx-community/Qwen3-ASR-1.7B-8bit": PresentationMetadata(ratingText: "4.8", tagKeys: ["Multilingual", "Realtime", "Accurate"]),
        "beshkenadze/cohere-transcribe-03-2026-mlx-fp16": PresentationMetadata(ratingText: "4.8", tagKeys: ["Multilingual", "Realtime", "Accurate"]),
        "OpenMOSS-Team/MOSS-Transcribe-Diarize": PresentationMetadata(ratingText: "4.7", tagKeys: ["Multilingual", "Realtime", "Diarization"]),
        "mlx-community/parakeet-tdt-0.6b-v3": PresentationMetadata(ratingText: "4.3", tagKeys: ["Fast"]),
        "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit": PresentationMetadata(ratingText: "4.5", tagKeys: ["Multilingual", "Realtime", "Fast"]),
        "mlx-community/SenseVoiceSmall": PresentationMetadata(ratingText: "4.5", tagKeys: ["Multilingual", "Fast"]),
    ]

    nonisolated private static let knownRemoteSizeBytesByRepo: [String: Int64] = [
        "mlx-community/whisper-large-v3-turbo": 1_617_000_000,
        "mlx-community/whisper-large-v3-mlx": 3_090_319_899,
        "mlx-community/whisper-small-mlx": 486_487_465,
        "mlx-community/Qwen3-ASR-0.6B-4bit": 712_781_279,
        "mlx-community/Qwen3-ASR-1.7B-6bit": 2_037_746_046,
        "mlx-community/Qwen3-ASR-1.7B-8bit": 2_467_859_030,
        "beshkenadze/cohere-transcribe-03-2026-mlx-fp16": 4_132_564_062,
        "OpenMOSS-Team/MOSS-Transcribe-Diarize": 1_833_165_136,
        "mlx-community/parakeet-tdt-0.6b-v3": 2_509_044_141,
        "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit": 760_000_000,
        "mlx-community/SenseVoiceSmall": 936_491_235,
    ]

    nonisolated static func canonicalModelRepo(_ repo: String) -> String {
        legacyModelRepoMap[repo] ?? repo
    }

    nonisolated static func capability(for repo: String) -> MLXASRModelCapability {
        let canonicalRepo = canonicalModelRepo(repo)
        return capabilitiesByRepo[canonicalRepo] ?? fallbackCapability(for: canonicalRepo)
    }

    nonisolated static func hasRegisteredCapability(for repo: String) -> Bool {
        capabilitiesByRepo[canonicalModelRepo(repo)] != nil
    }

    nonisolated private static func fallbackCapability(for repo: String) -> MLXASRModelCapability {
        // Unknown repositories are not inferred from names and cannot select a loader.
        MLXASRModelCapability(
            family: .generic,
            supportedLanguageCodes: [],
            languageRouting: .unavailable,
            liveMode: .batchPreview,
            isRealtimeCapable: false,
            outputCapabilities: [.text],
            timingGranularity: .none,
            configurationCapabilities: [],
            vadPolicy: .standard,
            supportedPurposes: [],
            kvCachePolicy: nil
        )
    }

    nonisolated static func displayTitle(for repo: String) -> String {
        let canonicalRepo = canonicalModelRepo(repo)
        return supportedModels.first(where: { $0.id == canonicalRepo })?.title ?? canonicalRepo
    }

    nonisolated static func description(for repo: String) -> String? {
        let canonicalRepo = canonicalModelRepo(repo)
        return supportedModels.first(where: { $0.id == canonicalRepo })?.description
    }

    nonisolated static func isAvailableModelRepo(_ repo: String) -> Bool {
        let canonicalRepo = canonicalModelRepo(repo)
        return availableModels.contains { $0.id == canonicalRepo }
    }

    nonisolated static func displayModels(includingInstalled _: Set<String>) -> [Option] {
        availableModels
    }

    nonisolated static func isRealtimeCapableModelRepo(_ repo: String) -> Bool {
        capability(for: repo).isRealtimeCapable
    }

    nonisolated static func liveMode(for repo: String) -> MLXLiveMode {
        capability(for: repo).liveMode
    }

    nonisolated static func ratingText(for repo: String) -> String {
        presentationByRepo[canonicalModelRepo(repo)]?.ratingText ?? "4.3"
    }

    nonisolated static func catalogTagKeys(for repo: String) -> [String] {
        presentationByRepo[canonicalModelRepo(repo)]?.tagKeys ?? []
    }

    nonisolated static func isMultilingualModelRepo(_ repo: String) -> Bool {
        capability(for: repo).isMultilingual
    }

    nonisolated static func supportsLanguage(_ code: String, for repo: String) -> Bool {
        capability(for: repo).supportsLanguage(code: code)
    }

    nonisolated static func fallbackRemoteSizeText(repo: String) -> String? {
        fallbackRemoteSizeInfo(repo: repo)?.text
    }

    nonisolated static func fallbackRemoteSizeInfo(repo: String) -> (bytes: Int64, text: String)? {
        let canonicalRepo = canonicalModelRepo(repo)
        guard let bytes = knownRemoteSizeBytesByRepo[canonicalRepo] else { return nil }
        return (bytes, MLXModelStorageSupport.formatByteCount(bytes))
    }
}

enum MLXModelStorageSupport {
    nonisolated private static let remoteSizeCachePreferenceKey = "mlxRemoteSizeCache"

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

    nonisolated static func cacheDirectory(for repo: String, rootDirectory: URL) -> URL? {
        guard let repoID = Repo.ID(rawValue: repo) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        return rootDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
    }

    nonisolated static func hubCache(rootDirectory: URL) -> HubCache {
        HubCache(cacheDirectory: rootDirectory)
    }

    nonisolated static func destinationFileURL(for entryPath: String, under directory: URL) throws -> URL {
        let base = directory.standardizedFileURL
        let destination = base.appendingPathComponent(entryPath).standardizedFileURL
        let basePrefix = base.path.hasSuffix("/") ? base.path : "\(base.path)/"
        guard destination.path.hasPrefix(basePrefix) else {
            throw NSError(
                domain: "MLXModelManager",
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

    nonisolated static func clearHubCache(for repoID: Repo.ID, rootDirectory: URL = HubCache.default.cacheDirectory) {
        let cache = hubCache(rootDirectory: rootDirectory)
        let repoDir = cache.repoDirectory(repo: repoID, kind: .model)
        let metadataDir = cache.metadataDirectory(repo: repoID, kind: .model)
        try? FileManager.default.removeItem(at: repoDir)
        try? FileManager.default.removeItem(at: metadataDir)
    }
}
