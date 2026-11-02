// RemoteModelConfiguration.swift
// Provides Remote Model Configuration for model catalog and storage support.

import Foundation

struct RemoteModelOption: Hashable, Identifiable {
    let id: String
    let title: String
}

enum StepFunASRModelFamily: String, Hashable {
    case stepAudio25ASR
    case stepAudio2ASRPro
    case stepASR11Stream
    case unknown
}

struct StepFunASRModelCapabilities: Hashable {
    let family: StepFunASRModelFamily
    let supportsHotwords: Bool
    let supportsPrompt: Bool
    let usesRealtimeWebSocket: Bool

    static func forModel(_ model: String) -> StepFunASRModelCapabilities {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        if normalized == "stepaudio-2.5-asr" || normalized == "step-asr-1.1-stream" {
            return StepFunASRModelCapabilities(
                family: normalized == "stepaudio-2.5-asr" ? .stepAudio25ASR : .stepASR11Stream,
                supportsHotwords: normalized == "stepaudio-2.5-asr",
                supportsPrompt: normalized == "step-asr-1.1-stream",
                usesRealtimeWebSocket: normalized == "step-asr-1.1-stream"
            )
        }

        if normalized == "stepaudio-2-asr-pro" {
            return StepFunASRModelCapabilities(
                family: .stepAudio2ASRPro,
                supportsHotwords: true,
                supportsPrompt: true,
                usesRealtimeWebSocket: false
            )
        }

        return StepFunASRModelCapabilities(
            family: .unknown,
            supportsHotwords: false,
            supportsPrompt: false,
            usesRealtimeWebSocket: false
        )
    }
}

enum AliyunASRModelFamily: String, Hashable {
    case qwenAudio3
    case qwen3ASR
    case omni
    case funASR
    case paraformer
    case unknown
}

enum AliyunASRLanguageCatalog {
    static let qwenAudioAndFunRealtime = [
        "zh", "en", "ja", "ko", "vi", "th", "id", "ms", "tl", "hi",
        "ar", "fr", "de", "es", "pt", "ru", "it", "nl", "sv", "da",
        "fi", "no", "el", "pl", "cs", "hu", "ro", "bg", "hr", "sk"
    ]

    static let qwenASRRealtime = [
        "zh", "yue", "en", "ja", "de", "ko", "ru", "fr", "pt", "ar",
        "it", "es", "hi", "id", "th", "tr", "uk", "vi", "cs", "da",
        "fil", "fi", "is", "ms", "no", "pl", "sv"
    ]

    static let funRealtime2026 = ["zh", "en", "ja"]
    static let funRealtime2025 = ["zh", "en"]
    static let funFlash8K = ["zh"]

    static func funRealtime(for model: String) -> [String] {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.hasPrefix("fun-asr-realtime-2026-") {
            return funRealtime2026
        }
        if normalized.hasPrefix("fun-asr-realtime-2025-09-") {
            return funRealtime2025
        }
        if normalized.hasPrefix("fun-asr-flash-8k-realtime") {
            return funFlash8K
        }
        return qwenAudioAndFunRealtime
    }
}

struct AliyunASRModelCapabilities: Hashable {
    let family: AliyunASRModelFamily
    let supportsInlineVocabulary: Bool
    let supportsLanguageHints: Bool
    let maximumLanguageHints: Int
    let supportedLanguageCodes: [String]
    let supportsMaxSentenceSilence: Bool
    let supportsServerVAD: Bool
    let supportsManualCommit: Bool
    let supportsSemanticPunctuation: Bool
    let supportsPunctuationPrediction: Bool
    let supportsInverseTextNormalization: Bool
    let supportsDisfluencyRemoval: Bool

    var isFunStyleRealtime: Bool {
        family == .qwenAudio3 || family == .funASR || family == .paraformer
    }

    var supportsAnyModelSettings: Bool {
        supportsInlineVocabulary ||
            supportsLanguageHints ||
            supportsMaxSentenceSilence ||
            supportsServerVAD ||
            supportsManualCommit ||
            supportsSemanticPunctuation ||
            supportsPunctuationPrediction ||
            supportsInverseTextNormalization ||
            supportsDisfluencyRemoval
    }

    static func forModel(_ model: String) -> AliyunASRModelCapabilities {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        if normalized.hasPrefix("qwen-audio-3.0-asr-flash-streaming") {
            return AliyunASRModelCapabilities(
                family: .qwenAudio3,
                supportsInlineVocabulary: true,
                supportsLanguageHints: true,
                maximumLanguageHints: 4,
                supportedLanguageCodes: AliyunASRLanguageCatalog.qwenAudioAndFunRealtime,
                supportsMaxSentenceSilence: true,
                supportsServerVAD: true,
                supportsManualCommit: false,
                supportsSemanticPunctuation: true,
                supportsPunctuationPrediction: false,
                supportsInverseTextNormalization: false,
                supportsDisfluencyRemoval: false,
            )
        }

        if normalized.hasPrefix("qwen3-asr-flash-realtime") {
            return AliyunASRModelCapabilities(
                family: .qwen3ASR,
                supportsInlineVocabulary: false,
                supportsLanguageHints: false,
                maximumLanguageHints: 1,
                supportedLanguageCodes: AliyunASRLanguageCatalog.qwenASRRealtime,
                supportsMaxSentenceSilence: false,
                supportsServerVAD: true,
                supportsManualCommit: true,
                supportsSemanticPunctuation: false,
                supportsPunctuationPrediction: false,
                supportsInverseTextNormalization: false,
                supportsDisfluencyRemoval: false,
            )
        }

        if normalized.hasPrefix("qwen3.5-omni-") || normalized.hasPrefix("qwen-omni-") {
            return AliyunASRModelCapabilities(
                family: .omni,
                supportsInlineVocabulary: false,
                supportsLanguageHints: false,
                maximumLanguageHints: 1,
                supportedLanguageCodes: [],
                supportsMaxSentenceSilence: false,
                supportsServerVAD: true,
                supportsManualCommit: false,
                supportsSemanticPunctuation: false,
                supportsPunctuationPrediction: false,
                supportsInverseTextNormalization: false,
                supportsDisfluencyRemoval: false,
            )
        }

        if normalized.hasPrefix("paraformer-realtime") {
            let isV2 = normalized.contains("v2")
            let supportedLanguages = normalized.contains("-8k-")
                ? ["zh"]
                : ["zh", "en", "ja", "ko", "de", "fr", "ru"]
            return AliyunASRModelCapabilities(
                family: .paraformer,
                supportsInlineVocabulary: false,
                supportsLanguageHints: true,
                maximumLanguageHints: 1,
                supportedLanguageCodes: supportedLanguages,
                supportsMaxSentenceSilence: isV2,
                supportsServerVAD: true,
                supportsManualCommit: false,
                supportsSemanticPunctuation: isV2,
                supportsPunctuationPrediction: isV2,
                supportsInverseTextNormalization: isV2,
                supportsDisfluencyRemoval: true,
            )
        }

        if normalized.hasPrefix("fun-asr") {
            return AliyunASRModelCapabilities(
                family: .funASR,
                supportsInlineVocabulary: false,
                supportsLanguageHints: true,
                maximumLanguageHints: 1,
                supportedLanguageCodes: AliyunASRLanguageCatalog.funRealtime(for: normalized),
                supportsMaxSentenceSilence: true,
                supportsServerVAD: true,
                supportsManualCommit: false,
                supportsSemanticPunctuation: true,
                supportsPunctuationPrediction: false,
                supportsInverseTextNormalization: false,
                supportsDisfluencyRemoval: false,
            )
        }

        return AliyunASRModelCapabilities(
            family: .unknown,
            supportsInlineVocabulary: false,
            supportsLanguageHints: false,
            maximumLanguageHints: 0,
            supportedLanguageCodes: [],
            supportsMaxSentenceSilence: false,
            supportsServerVAD: false,
            supportsManualCommit: false,
            supportsSemanticPunctuation: false,
            supportsPunctuationPrediction: false,
            supportsInverseTextNormalization: false,
            supportsDisfluencyRemoval: false,
        )
    }
}

enum RemoteASRProvider: String, CaseIterable, Identifiable {
    case openAIWhisper
    case doubaoASR
    case glmASR
    case aliyunBailianASR
    case stepFunASR
    case xiaomiMiMoASR
    case googleGeminiASR

    nonisolated var id: String { rawValue }

    var title: String {
        switch self {
        case .openAIWhisper:
            return AppLocalization.localizedString("OpenAI Transcribe")
        case .doubaoASR:
            return AppLocalization.localizedString("Doubao")
        case .glmASR:
            return AppLocalization.localizedString("Z ai")
        case .aliyunBailianASR:
            return AppLocalization.localizedString("Aliyun Bailian ASR")
        case .stepFunASR:
            return AppLocalization.localizedString("StepFun")
        case .xiaomiMiMoASR:
            return AppLocalization.localizedString("Xiaomi MiMo")
        case .googleGeminiASR:
            return AppLocalization.localizedString("Google Gemini")
        }
    }

    var suggestedModel: String {
        switch self {
        case .openAIWhisper:
            return "gpt-4o-mini-transcribe"
        case .doubaoASR:
            return DoubaoASRConfiguration.modelV2
        case .glmASR:
            return "glm-asr-1"
        case .aliyunBailianASR:
            return "fun-asr-realtime"
        case .stepFunASR:
            return "stepaudio-2.5-asr"
        case .xiaomiMiMoASR:
            return "mimo-v2.5-asr"
        case .googleGeminiASR:
            return GeminiLivePayloadSupport.defaultModel
        }
    }

    var modelOptions: [RemoteModelOption] {
        switch self {
        case .openAIWhisper:
            return [
                RemoteModelOption(id: "gpt-4o-mini-transcribe", title: "GPT-4o Mini Transcribe"),
                RemoteModelOption(id: "gpt-4o-transcribe", title: "GPT-4o Transcribe"),
                RemoteModelOption(id: "gpt-4o-transcribe-diarize", title: "GPT-4o Transcribe Diarize"),
                RemoteModelOption(id: "whisper-1", title: "Whisper-1")
            ]
        case .doubaoASR:
            return [
                RemoteModelOption(id: DoubaoASRConfiguration.modelV2, title: "Doubao ASR 2.0 (Hourly)"),
                RemoteModelOption(id: DoubaoASRConfiguration.modelV1, title: "Doubao ASR 1.0 (Hourly)")
            ]
        case .glmASR:
            return [
                RemoteModelOption(id: "glm-asr-2512", title: "GLM-ASR-2512"),
                RemoteModelOption(id: "glm-asr-1", title: "GLM-ASR-1")
            ]
        case .aliyunBailianASR:
            return [
                RemoteModelOption(id: "qwen-audio-3.0-asr-flash-streaming", title: "Qwen-Audio 3.0 ASR Flash Streaming"),
                RemoteModelOption(id: "qwen3-asr-flash-realtime", title: "Qwen3 ASR Flash Realtime"),
                RemoteModelOption(id: "qwen3-asr-flash-realtime-2026-02-10", title: "Qwen3 ASR Flash Realtime (2026-02-10)"),
                RemoteModelOption(id: "qwen3-asr-flash-realtime-2025-10-27", title: "Qwen3 ASR Flash Realtime (2025-10-27)"),
                RemoteModelOption(id: "qwen3.5-omni-flash-realtime", title: "Qwen3.5 Omni Flash Realtime"),
                RemoteModelOption(id: "qwen3.5-omni-plus-realtime", title: "Qwen3.5 Omni Plus Realtime"),
                RemoteModelOption(id: "qwen-omni-turbo-realtime", title: "Qwen Omni Turbo Realtime"),
                RemoteModelOption(id: "fun-asr-realtime", title: "Fun ASR Realtime"),
                RemoteModelOption(id: "fun-asr-realtime-2026-02-28", title: "Fun ASR Realtime (2026-02-28)"),
                RemoteModelOption(id: "fun-asr-realtime-2025-11-07", title: "Fun ASR Realtime (2025-11-07)"),
                RemoteModelOption(id: "fun-asr-realtime-2025-09-15", title: "Fun ASR Realtime (2025-09-15)"),
                RemoteModelOption(id: "fun-asr-flash-8k-realtime", title: "Fun ASR Flash 8k Realtime"),
                RemoteModelOption(id: "fun-asr-flash-8k-realtime-2026-01-28", title: "Fun ASR Flash 8k Realtime (2026-01-28)"),
                RemoteModelOption(id: "paraformer-realtime-v2", title: "Paraformer Realtime V2"),
                RemoteModelOption(id: "paraformer-realtime-v1", title: "Paraformer Realtime V1"),
                RemoteModelOption(id: "paraformer-realtime-8k-v2", title: "Paraformer Realtime 8k V2"),
                RemoteModelOption(id: "paraformer-realtime-8k-v1", title: "Paraformer Realtime 8k V1")
            ]
        case .stepFunASR:
            return [
                RemoteModelOption(id: "stepaudio-2.5-asr", title: "StepAudio 2.5 ASR (SSE)"),
                RemoteModelOption(id: "stepaudio-2-asr-pro", title: "StepAudio 2 ASR Pro (SSE)"),
                RemoteModelOption(id: "step-asr-1.1-stream", title: "Step ASR 1.1 Stream (WebSocket)")
            ]
        case .xiaomiMiMoASR:
            return [
                RemoteModelOption(id: "mimo-v2.5-asr", title: "MiMo V2.5 ASR")
            ]
        case .googleGeminiASR:
            return [
                RemoteModelOption(
                    id: GeminiLivePayloadSupport.defaultModel,
                    title: "Gemini 3.5 Transcribe Live"
                )
            ]
        }
    }
}

enum RemoteLLMProvider: String, CaseIterable, Identifiable {
    case anthropic
    case google
    case openAI
    case codex
    case ollama
    case omlx
    case deepseek
    case openrouter
    case grok
    case zai
    case volcengine
    case kimi
    case lmStudio
    case minimax
    case aliyunBailian
    case stepFun
    case xiaomiMiMo

    var id: String { rawValue }

    nonisolated var usesResponsesAPI: Bool {
        switch self {
        case .openAI, .codex, .volcengine, .aliyunBailian:
            return true
        default:
            return false
        }
    }

    nonisolated var supportsHostedSearch: Bool {
        switch self {
        case .anthropic, .google, .zai, .volcengine, .aliyunBailian:
            return true
        default:
            return false
        }
    }

    nonisolated var defaultSearchEnabled: Bool {
        self == .aliyunBailian
    }

    nonisolated var apiKeyIsOptional: Bool {
        switch self {
        case .codex, .ollama, .omlx:
            return true
        default:
            return false
        }
    }

    var searchToggleDescription: String {
        switch self {
        case .anthropic:
            return AppLocalization.localizedString("Use Anthropic hosted web search when the model needs current information. This may increase latency and usage.")
        case .google:
            return AppLocalization.localizedString("Ground responses with Google Search when Gemini needs fresher web context. This may increase latency and usage.")
        case .zai:
            return AppLocalization.localizedString("Allow GLM to call its hosted web search tool for fresher answers. This may increase latency and usage.")
        case .aliyunBailian:
            return AppLocalization.localizedString("Allow Qwen to use built-in web search. Enabled by default for Aliyun because it is available directly in the official compatible API.")
        default:
            return AppLocalization.localizedString("Use provider-hosted web search when available. This may increase latency and usage.")
        }
    }

    var title: String {
        switch self {
        case .anthropic:
            return AppLocalization.localizedString("Anthropic")
        case .google:
            return AppLocalization.localizedString("Google")
        case .openAI:
            return AppLocalization.localizedString("OpenAI")
        case .codex:
            return AppLocalization.localizedString("Codex (ChatGPT)")
        case .ollama:
            return AppLocalization.localizedString("Ollama")
        case .omlx:
            return "oMLX"
        case .deepseek:
            return AppLocalization.localizedString("DeepSeek")
        case .openrouter:
            return AppLocalization.localizedString("OpenRouter")
        case .grok:
            return AppLocalization.localizedString("xAI (Grok)")
        case .zai:
            return AppLocalization.localizedString("Z.ai")
        case .volcengine:
            return AppLocalization.localizedString("Volcengine")
        case .kimi:
            return AppLocalization.localizedString("Kimi")
        case .lmStudio:
            return AppLocalization.localizedString("LM Studio")
        case .minimax:
            return AppLocalization.localizedString("MiniMax")
        case .aliyunBailian:
            return AppLocalization.localizedString("Aliyun Bailian")
        case .stepFun:
            return AppLocalization.localizedString("StepFun")
        case .xiaomiMiMo:
            return AppLocalization.localizedString("Xiaomi MiMo")
        }
    }

    nonisolated var suggestedModel: String {
        switch self {
        case .anthropic:
            return "claude-sonnet-4-6"
        case .google:
            return "gemini-2.5-pro"
        case .openAI:
            return "gpt-5.2"
        case .codex:
            return "gpt-5.4"
        case .ollama:
            return "qwen2.5"
        case .omlx:
            return "qwen3"
        case .deepseek:
            return "deepseek-flash"
        case .openrouter:
            return "openrouter/auto"
        case .grok:
            return "grok-4"
        case .zai:
            return "glm-5"
        case .volcengine:
            return "doubao-seed-2-0-pro-260215"
        case .kimi:
            return "kimi-k2.5"
        case .lmStudio:
            return "llama3.1"
        case .minimax:
            return "MiniMax-M2.5"
        case .aliyunBailian:
            return "qwen-plus-latest"
        case .stepFun:
            return "step-3.5-flash"
        case .xiaomiMiMo:
            return "mimo-v2.5-pro"
        }
    }

    var modelOptions: [RemoteModelOption] {
        let merged = latestModelOptions + basicModelOptions + advancedModelOptions
        var seen = Set<String>()
        return merged.filter { seen.insert($0.id).inserted }
    }

    var latestModelOptions: [RemoteModelOption] {
        switch self {
        case .anthropic:
            return [
                RemoteModelOption(id: "claude-opus-4-6", title: "Claude Opus 4.6"),
                RemoteModelOption(id: "claude-sonnet-4-6", title: "Claude Sonnet 4.6"),
                RemoteModelOption(id: "claude-opus-4-5-20251101", title: "Claude Opus 4.5"),
                RemoteModelOption(id: "claude-sonnet-4-5-20250929", title: "Claude Sonnet 4.5")
            ]
        case .google:
            return [
                RemoteModelOption(id: "gemini-3.1-pro-preview", title: "Gemini 3.1 Pro Preview"),
                RemoteModelOption(id: "gemini-3-pro-preview", title: "Gemini 3 Pro Preview"),
                RemoteModelOption(id: "gemini-2.5-pro", title: "Gemini 2.5 Pro"),
                RemoteModelOption(id: "gemini-2.5-pro-preview-06-05", title: "Gemini 2.5 Pro Preview 06-05")
            ]
        case .openAI:
            return [
                RemoteModelOption(id: "gpt-5.2", title: "GPT-5.2"),
                RemoteModelOption(id: "gpt-5.2-pro", title: "GPT-5.2 pro"),
                RemoteModelOption(id: "gpt-5.2-codex", title: "GPT-5.2 Codex"),
                RemoteModelOption(id: "gpt-5.1", title: "GPT-5.1"),
                RemoteModelOption(id: "gpt-5.1-codex", title: "GPT-5.1 Codex"),
                RemoteModelOption(id: "gpt-5.1-codex-max", title: "GPT-5.1 Codex Max"),
                RemoteModelOption(id: "gpt-5", title: "GPT-5"),
                RemoteModelOption(id: "gpt-5-pro", title: "GPT-5 pro"),
                RemoteModelOption(id: "gpt-5-mini", title: "GPT-5 mini"),
                RemoteModelOption(id: "gpt-5-nano", title: "GPT-5 nano")
            ]
        case .codex:
            return [
                RemoteModelOption(id: "gpt-5.4", title: "GPT-5.4"),
                RemoteModelOption(id: "gpt-5.5", title: "GPT-5.5"),
                RemoteModelOption(id: "gpt-5.4-mini", title: "GPT-5.4 Mini"),
                RemoteModelOption(id: "gpt-5.3-codex", title: "GPT-5.3 Codex"),
                RemoteModelOption(id: "gpt-5.3-codex-spark", title: "GPT-5.3 Codex Spark"),
                RemoteModelOption(id: "gpt-5.2", title: "GPT-5.2"),
                RemoteModelOption(id: "gpt-5-codex", title: "GPT-5 Codex"),
                RemoteModelOption(id: "gpt-5-codex-mini", title: "GPT-5 Codex Mini"),
                RemoteModelOption(id: "gpt-oss-120b", title: "GPT-OSS 120B"),
                RemoteModelOption(id: "gpt-oss-20b", title: "GPT-OSS 20B")
            ]
        case .ollama:
            return [
                RemoteModelOption(id: "deepseek-v3.1:671b", title: "DeepSeek V3.1"),
                RemoteModelOption(id: "gpt-oss:120b", title: "GPT-OSS 120B"),
                RemoteModelOption(id: "qwen3-coder:480b", title: "Qwen3 Coder 480B"),
                RemoteModelOption(id: "deepseek-r1", title: "DeepSeek R1")
            ]
        case .omlx:
            return [
                RemoteModelOption(id: "Qwen3-Coder-Next-8bit", title: "Qwen3 Coder Next 8bit"),
                RemoteModelOption(id: "gpt-oss-120b-MXFP4-Q8", title: "GPT-OSS 120B MXFP4 Q8"),
                RemoteModelOption(id: "Qwen3.5-122B-A10B-4bit", title: "Qwen3.5 122B A10B 4bit"),
                RemoteModelOption(id: "Step-3.5-Flash-8bit", title: "Step 3.5 Flash 8bit")
            ]
        case .deepseek:
            return [
                RemoteModelOption(id: "deepseek-flash", title: "DeepSeek V4.1 Flash (Recommended)"),
                RemoteModelOption(id: "deepseek-v4-pro", title: "DeepSeek V4 Pro")
            ]
        case .openrouter:
            return [
                RemoteModelOption(id: "openrouter/auto", title: "Auto (best for prompt)"),
                RemoteModelOption(id: "deepseek/deepseek-chat-v3.1", title: "DeepSeek V3.1"),
                RemoteModelOption(id: "openai/gpt-4.1", title: "GPT-4.1"),
                RemoteModelOption(id: "google/gemini-2.5-pro", title: "Gemini 2.5 Pro")
            ]
        case .grok:
            return [
                RemoteModelOption(id: "grok-4-1-fast-reasoning", title: "Grok 4.1 Fast"),
                RemoteModelOption(id: "grok-4-1-fast-non-reasoning", title: "Grok 4.1 Fast (Non-Reasoning)"),
                RemoteModelOption(id: "grok-4", title: "Grok 4 0709")
            ]
        case .zai:
            return [
                RemoteModelOption(id: "glm-5", title: "GLM-5"),
                RemoteModelOption(id: "glm-4.7", title: "GLM-4.7"),
                RemoteModelOption(id: "glm-4.6", title: "GLM-4.6"),
                RemoteModelOption(id: "glm-4.5", title: "GLM-4.5")
            ]
        case .volcengine:
            return [
                RemoteModelOption(id: "doubao-seed-2-0-pro-260215", title: "Doubao Seed 2.0 Pro (260215)"),
                RemoteModelOption(id: "doubao-seed-2-0-lite-260215", title: "Doubao Seed 2.0 Lite (260215)"),
                RemoteModelOption(id: "doubao-seed-2-0-mini-260215", title: "Doubao Seed 2.0 Mini (260215)"),
                RemoteModelOption(id: "doubao-seed-2-0-code-preview-260215", title: "Doubao Seed 2.0 Code Preview (260215)"),
                RemoteModelOption(id: "doubao-seed-1-8-251228", title: "Doubao Seed 1.8 (251228)"),
                RemoteModelOption(id: "glm-4-7-251222", title: "GLM-4.7 (251222)"),
                RemoteModelOption(id: "doubao-seed-code-preview-251028", title: "Doubao Seed Code Preview (251028)"),
                RemoteModelOption(id: "doubao-seed-1-6-lite-251015", title: "Doubao Seed 1.6 Lite (251015)"),
                RemoteModelOption(id: "doubao-seed-1-6-flash-250828", title: "Doubao Seed 1.6 Flash (250828)"),
                RemoteModelOption(id: "doubao-seed-translation-250915", title: "Doubao Seed Translation (250915)"),
                RemoteModelOption(id: "doubao-seed-1-6-vision-250815", title: "Doubao Seed 1.6 Vision (250815)")
            ]
        case .kimi:
            return [
                RemoteModelOption(id: "kimi-k2.5", title: "Kimi K2.5"),
                RemoteModelOption(id: "kimi-k2-thinking", title: "Kimi K2 Thinking"),
                RemoteModelOption(id: "kimi-latest", title: "Kimi Latest")
            ]
        case .lmStudio:
            return [
                RemoteModelOption(id: "llama3.1", title: "Llama 3.1 8B"),
                RemoteModelOption(id: "qwen2.5-14b-instruct", title: "Qwen2.5 14B")
            ]
        case .minimax:
            return [
                RemoteModelOption(id: "MiniMax-M2.5", title: "MiniMax M2.5"),
                RemoteModelOption(id: "MiniMax-M2.5-Lightning", title: "MiniMax M2.5 Lightning"),
                RemoteModelOption(id: "MiniMax-M2.1", title: "MiniMax M2.1")
            ]
        case .aliyunBailian:
            return [
                RemoteModelOption(id: "qwen-max-latest", title: "Qwen Max Latest"),
                RemoteModelOption(id: "qwen-plus-latest", title: "Qwen Plus Latest"),
                RemoteModelOption(id: "qwen-turbo-latest", title: "Qwen Turbo Latest")
            ]
        case .stepFun:
            return [
                RemoteModelOption(id: "step-3.5-flash", title: "Step 3.5 Flash"),
                RemoteModelOption(id: "step-3.5-flash-2603", title: "Step 3.5 Flash 2603"),
                RemoteModelOption(id: "stepaudio-2.5-chat", title: "StepAudio 2.5 Chat")
            ]
        case .xiaomiMiMo:
            return [
                RemoteModelOption(id: "mimo-v2.5-pro", title: "MiMo V2.5 Pro"),
                RemoteModelOption(id: "mimo-v2.5", title: "MiMo V2.5")
            ]
        }
    }

    var basicModelOptions: [RemoteModelOption] {
        switch self {
        case .anthropic:
            return [
                RemoteModelOption(id: "claude-haiku-4-5-20251001", title: "Claude Haiku 4.5"),
                RemoteModelOption(id: "claude-3-haiku-20240307", title: "Claude 3 Haiku")
            ]
        case .google:
            return [
                RemoteModelOption(id: "gemini-2.5-flash", title: "Gemini 2.5 Flash"),
                RemoteModelOption(id: "gemini-2.5-flash-lite", title: "Gemini 2.5 Flash-Lite"),
                RemoteModelOption(id: "gemini-2.0-flash", title: "Gemini 2.0 Flash"),
                RemoteModelOption(id: "gemini-2.0-flash-001", title: "Gemini 2.0 Flash 001"),
                RemoteModelOption(id: "gemini-1.5-flash-002", title: "Gemini 1.5 Flash 002")
            ]
        case .openAI:
            return [
                RemoteModelOption(id: "gpt-4.1", title: "GPT-4.1"),
                RemoteModelOption(id: "gpt-4.1-mini", title: "GPT-4.1 mini"),
                RemoteModelOption(id: "gpt-4o-mini", title: "GPT-4o mini"),
                RemoteModelOption(id: "gpt-4o", title: "GPT-4o")
            ]
        case .codex:
            return []
        case .ollama:
            return [
                RemoteModelOption(id: "qwen2.5", title: "Qwen2.5 7B"),
                RemoteModelOption(id: "qwen3", title: "Qwen3 7B"),
                RemoteModelOption(id: "llama3.1", title: "Llama 3.1 8B"),
                RemoteModelOption(id: "mistral", title: "Mistral 7B"),
                RemoteModelOption(id: "gemma2", title: "Gemma 2 9B")
            ]
        case .omlx:
            return [
                RemoteModelOption(id: "qwen3", title: "Qwen3"),
                RemoteModelOption(id: "llama3.1", title: "Llama 3.1"),
                RemoteModelOption(id: "deepseek-r1", title: "DeepSeek R1"),
                RemoteModelOption(id: "gemma3", title: "Gemma 3"),
                RemoteModelOption(id: "mistral", title: "Mistral")
            ]
        case .deepseek:
            return [
                RemoteModelOption(id: "deepseek-v4-flash", title: "DeepSeek V4 Flash (Compatibility Alias)"),
                RemoteModelOption(id: "deepseek-v4-flash-vision-exp", title: "DeepSeek V4 Flash Vision (Compatibility Alias)"),
                RemoteModelOption(id: "deepseek-chat", title: "DeepSeek Chat (Legacy Alias)")
            ]
        case .openrouter:
            return [
                RemoteModelOption(id: "google/gemini-2.5-flash", title: "Gemini 2.5 Flash"),
                RemoteModelOption(id: "openai/gpt-4.1-mini", title: "GPT-4.1 mini"),
                RemoteModelOption(id: "qwen/qwen3-14b", title: "Qwen3 14B")
            ]
        case .grok:
            return [
                RemoteModelOption(id: "grok-3", title: "Grok 3"),
                RemoteModelOption(id: "grok-3-mini", title: "Grok 3 Mini")
            ]
        case .zai:
            return [
                RemoteModelOption(id: "glm-4.7-flash", title: "GLM-4.7-Flash"),
                RemoteModelOption(id: "glm-4.5-air", title: "GLM-4.5-Air"),
                RemoteModelOption(id: "glm-4.5-airx", title: "GLM-4.5-AirX")
            ]
        case .volcengine:
            return []
        case .kimi:
            return [
                RemoteModelOption(id: "moonshot-v1-8k", title: "Moonshot V1 8K"),
                RemoteModelOption(id: "moonshot-v1-32k", title: "Moonshot V1 32K"),
                RemoteModelOption(id: "moonshot-v1-auto", title: "Moonshot V1 Auto")
            ]
        case .lmStudio:
            return [
                RemoteModelOption(id: "qwen2.5-14b-instruct", title: "Qwen2.5 14B"),
                RemoteModelOption(id: "llama3.1", title: "Llama 3.1 8B")
            ]
        case .minimax:
            return [
                RemoteModelOption(id: "MiniMax-Text-01", title: "MiniMax Text 01"),
                RemoteModelOption(id: "MiniMax-M2", title: "MiniMax M2"),
                RemoteModelOption(id: "MiniMax-M2.1-Lightning", title: "MiniMax M2.1 Lightning")
            ]
        case .aliyunBailian:
            return [
                RemoteModelOption(id: "qwen-plus", title: "Qwen Plus"),
                RemoteModelOption(id: "qwen-turbo", title: "Qwen Turbo")
            ]
        case .stepFun:
            return [
                RemoteModelOption(id: "step-2-mini", title: "Step 2 Mini"),
                RemoteModelOption(id: "step-2-16k", title: "Step 2 16K"),
                RemoteModelOption(id: "step-1-8k", title: "Step 1 8K")
            ]
        case .xiaomiMiMo:
            return []
        }
    }

    var advancedModelOptions: [RemoteModelOption] {
        switch self {
        case .anthropic:
            return [
                RemoteModelOption(id: "claude-opus-4-6", title: "Claude Opus 4.6"),
                RemoteModelOption(id: "claude-opus-4-5-20251101", title: "Claude Opus 4.5")
            ]
        case .google:
            return [
                RemoteModelOption(id: "gemini-2.5-flash", title: "Gemini 2.5 Flash"),
                RemoteModelOption(id: "gemini-1.5-pro-002", title: "Gemini 1.5 Pro 002"),
                RemoteModelOption(id: "gemini-pro-latest", title: "Gemini Pro Latest")
            ]
        case .openAI:
            return [
                RemoteModelOption(id: "gpt-4", title: "GPT-4")
            ]
        case .codex:
            return []
        case .ollama:
            return [
                RemoteModelOption(id: "gpt-oss:120b", title: "GPT-OSS 120B"),
                RemoteModelOption(id: "llama3.1:70b", title: "Llama 3.1 70B"),
                RemoteModelOption(id: "mixtral:8x22b", title: "Mixtral 8x22B")
            ]
        case .omlx:
            return [
                RemoteModelOption(id: "Qwen3-Coder-Next-8bit", title: "Qwen3 Coder Next 8bit"),
                RemoteModelOption(id: "gpt-oss-120b-MXFP4-Q8", title: "GPT-OSS 120B MXFP4 Q8"),
                RemoteModelOption(id: "mixtral", title: "Mixtral")
            ]
        case .deepseek:
            return [RemoteModelOption(id: "deepseek-reasoner", title: "DeepSeek Reasoner (Compatibility Alias)")]
        case .openrouter:
            return [
                RemoteModelOption(id: "openai/gpt-4.1", title: "GPT-4.1"),
                RemoteModelOption(id: "deepseek/deepseek-r1", title: "DeepSeek R1"),
                RemoteModelOption(id: "anthropic/claude-3.5-sonnet", title: "Claude 3.5 Sonnet")
            ]
        case .grok:
            return [
                RemoteModelOption(id: "grok-4-1-fast-reasoning", title: "Grok 4.1 Fast"),
                RemoteModelOption(id: "grok-code-fast-1", title: "Grok Code Fast 1")
            ]
        case .zai:
            return [
                RemoteModelOption(id: "glm-4.7", title: "GLM-4.7"),
                RemoteModelOption(id: "glm-4.6v", title: "GLM-4.6V")
            ]
        case .volcengine:
            return []
        case .kimi:
            return [
                RemoteModelOption(id: "kimi-k2-thinking", title: "Kimi K2 Thinking"),
                RemoteModelOption(id: "moonshot-v1-128k", title: "Moonshot V1 128K")
            ]
        case .lmStudio:
            return [RemoteModelOption(id: "qwen2.5-14b-instruct", title: "Qwen2.5 14B")]
        case .minimax:
            return [
                RemoteModelOption(id: "MiniMax-M2-Stable", title: "MiniMax M2 Stable"),
                RemoteModelOption(id: "MiniMax-M2.1", title: "MiniMax M2.1")
            ]
        case .aliyunBailian:
            return [
                RemoteModelOption(id: "qwen-max", title: "Qwen Max"),
                RemoteModelOption(id: "qwq-plus", title: "QwQ Plus")
            ]
        case .stepFun:
            return [
                RemoteModelOption(id: "step-2-16k-exp", title: "Step 2 16K Exp"),
                RemoteModelOption(id: "step-1-32k", title: "Step 1 32K"),
                RemoteModelOption(id: "step-router-v1", title: "Step Router V1 (Step Plan)")
            ]
        case .xiaomiMiMo:
            return []
        }
    }
}
