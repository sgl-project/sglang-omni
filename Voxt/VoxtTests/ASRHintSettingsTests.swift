// ASRHintSettingsTests.swift
// Provides ASRHint Settings Tests for Voxt test coverage.

import XCTest
import MLXAudioSTT
@testable import Voxt

@MainActor
final class ASRHintSettingsTests: XCTestCase {
    func testLoadSanitizesUnsupportedPromptEditors() {
        let raw = """
        {"mlxAudio":{"followsUserMainLanguage":true,"promptTemplate":"  should be removed  "},"openAIWhisper":{"followsUserMainLanguage":false,"promptTemplate":"  Bias {{USER_MAIN_LANGUAGE}}  "}}
        """

        let loaded = ASRHintSettingsStore.load(from: raw)

        XCTAssertEqual(
            loaded[.mlxAudio]?.promptTemplate,
            ""
        )
        XCTAssertEqual(loaded[.openAIWhisper]?.promptTemplate, "Bias {{USER_MAIN_LANGUAGE}}")
    }

    func testResolvedSettingsFallsBackToDefaults() {
        let settings = ASRHintSettingsStore.resolvedSettings(for: .glmASR, rawValue: nil)

        XCTAssertTrue(settings.followsUserMainLanguage)
        XCTAssertEqual(settings.promptTemplate, AppPromptDefaults.text(for: .glmASRHint))
        XCTAssertEqual(settings.promptTemplate, AppPreferenceKey.asrDictionaryTermsTemplateVariable)
    }

    func testResolveOpenAIUsesBaseLanguageAndResolvedPrompt() {
        let payload = ASRHintResolver.resolve(
            target: .openAIWhisper,
            settings: ASRHintSettings(
                followsUserMainLanguage: true,
                promptTemplate: "Primary {{USER_MAIN_LANGUAGE}}"
            ),
            userLanguageCodes: ["zh-Hant"]
        )

        XCTAssertEqual(payload.language, "zh")
        XCTAssertEqual(payload.prompt, "Primary Traditional Chinese")
    }

    func testResolveMLXAvoidsPromptBiasAndForcedLanguage() {
        let payload = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(
                followsUserMainLanguage: true,
                promptTemplate: "Bias {{USER_MAIN_LANGUAGE}} punctuation"
            ),
            userLanguageCodes: ["zh-Hant"]
        )

        XCTAssertNil(payload.language)
        XCTAssertNil(payload.prompt)
    }

    func testResolvedMLXSettingsDefaultToEmptyPrompt() {
        let settings = ASRHintSettingsStore.resolvedSettings(for: .mlxAudio, rawValue: nil)

        XCTAssertTrue(settings.followsUserMainLanguage)
        XCTAssertEqual(settings.promptTemplate, "")
    }

    func testDefaultASRPromptResolvesToDictionaryTermsOnly() {
        let payload = ASRHintResolver.resolve(
            target: .openAIWhisper,
            settings: ASRHintSettingsStore.resolvedSettings(for: .openAIWhisper, rawValue: nil),
            userLanguageCodes: ["zh-Hans", "en"],
            dictionaryTerms: "Codex\nVoxt"
        )

        XCTAssertEqual(payload.language, nil)
        XCTAssertEqual(payload.prompt, "Codex\nVoxt")
    }

    func testDefaultASRPromptDoesNotAutoAppendLanguageContext() {
        let payload = ASRHintResolver.resolve(
            target: .glmASR,
            settings: ASRHintSettings(
                followsUserMainLanguage: true,
                promptTemplate: ""
            ),
            userLanguageCodes: ["zh-Hans", "en"],
            dictionaryTerms: ""
        )

        XCTAssertNil(payload.prompt)
    }

    func testResolveDoubaoUsesVariantMappingForTraditionalChinese() {
        let payload = ASRHintResolver.resolve(
            target: .doubaoASR,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hant"]
        )

        XCTAssertEqual(payload.language, "zh-CN")
        XCTAssertEqual(payload.chineseOutputVariant, "zh-Hant")
        XCTAssertNil(payload.prompt)
    }

    func testResolveAliyunSupportsModelSpecificMultilingualHints() {
        let payload = ASRHintResolver.resolve(
            target: .aliyunBailianASR,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hans", "en", "zh-Hant", "ja", "ko"],
            mlxModelRepo: "qwen-audio-3.0-asr-flash-streaming"
        )

        XCTAssertEqual(payload.languageHints, ["zh", "en", "ja", "ko"])
        XCTAssertEqual(payload.language, "zh")

        let funPayload = ASRHintResolver.resolve(
            target: .aliyunBailianASR,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hans", "en", "ja"],
            mlxModelRepo: "fun-asr-realtime-2026-02-28"
        )
        XCTAssertEqual(funPayload.languageHints, ["zh"])
    }

    func testResolveAliyunBuildsHotwordsFromContextAndDictionaryTerms() {
        let payload = ASRHintResolver.resolve(
            target: .aliyunBailianASR,
            settings: ASRHintSettings(contextualPhrasesText: "Voxt\nFireRed\nVoxt"),
            userLanguageCodes: ["zh-Hans"],
            dictionaryTerms: "Codex\nFireRed"
        )

        XCTAssertEqual(payload.language, "zh")
        XCTAssertEqual(payload.contextualPhrases, ["Voxt", "FireRed", "Codex"])
    }

    func testResolveStepFunBuildsPromptFromTerms() {
        let payload = ASRHintResolver.resolve(
            target: .stepFunASR,
            settings: ASRHintSettings(contextualPhrasesText: "Voxt\nFireRed\nVoxt"),
            userLanguageCodes: ["zh-Hans"],
            dictionaryTerms: "Codex\nFireRed"
        )

        XCTAssertEqual(payload.language, "zh")
        XCTAssertEqual(payload.contextualPhrases, ["Voxt", "FireRed", "Codex"])
        XCTAssertNotNil(payload.prompt)
        XCTAssertContains(payload.prompt ?? "", "Preserve names")
        XCTAssertContains(payload.prompt ?? "", "Voxt")
        XCTAssertContains(payload.prompt ?? "", "FireRed")
        XCTAssertContains(payload.prompt ?? "", "Codex")
        XCTAssertEqual(payload.prompt?.components(separatedBy: "Voxt").count, 2)
        XCTAssertEqual(payload.prompt?.components(separatedBy: "FireRed").count, 2)
    }

    func testResolveMLXUsesPromptNameForQwenModel() {
        let payload = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hant"],
            mlxModelRepo: "mlx-community/Qwen3-ASR-0.6B-4bit"
        )

        XCTAssertEqual(payload.language, "Traditional Chinese")
    }

    func testResolveMLXKeepsQwenPrimaryLanguageWhenSecondaryLanguagesAreConfigured() {
        let payload = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hans", "en"],
            mlxModelRepo: "mlx-community/Qwen3-ASR-0.6B-4bit"
        )

        XCTAssertEqual(payload.language, "Simplified Chinese")
        XCTAssertEqual(payload.otherLanguages, ["English"])
    }

    func testResolveMLXAllowsQwenAutomaticLanguageWhenFollowingMainLanguageIsDisabled() {
        let payload = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(followsUserMainLanguage: false),
            userLanguageCodes: ["zh-Hans", "en"],
            mlxModelRepo: "mlx-community/Qwen3-ASR-0.6B-4bit"
        )

        XCTAssertNil(payload.language)
    }

    func testResolveMLXUsesSenseVoiceLanguageRoutingOnlyForSupportedLocales() {
        let cantonesePayload = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["yue"],
            mlxModelRepo: "mlx-community/SenseVoiceSmall"
        )
        XCTAssertEqual(cantonesePayload.language, "yue")

        let chinesePayload = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hant"],
            mlxModelRepo: "mlx-community/SenseVoiceSmall"
        )
        XCTAssertEqual(chinesePayload.language, "zh")

        let unsupportedPayload = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["fr"],
            mlxModelRepo: "mlx-community/SenseVoiceSmall"
        )
        XCTAssertNil(unsupportedPayload.language)
    }

    func testResolveMLXUsesCapabilityLanguageRouting() {
        let unsupportedQwen = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["sw"],
            mlxModelRepo: "mlx-community/Qwen3-ASR-0.6B-4bit"
        )
        XCTAssertNil(unsupportedQwen.language)

        let automaticParakeet = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["de"],
            mlxModelRepo: "mlx-community/parakeet-tdt-0.6b-v3"
        )
        XCTAssertNil(automaticParakeet.language)

        let nemotron = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hans", "en"],
            mlxModelRepo: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
        )
        XCTAssertEqual(nemotron.language, "zh-CN")

        let traditionalChineseNemotron = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hant"],
            mlxModelRepo: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
        )
        XCTAssertEqual(traditionalChineseNemotron.language, "zh-TW")

        let automaticVoxtral = ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: ASRHintSettings(),
            userLanguageCodes: ["zh-Hans"],
            mlxModelRepo: "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
        )
        XCTAssertEqual(automaticVoxtral.language, "Simplified Chinese")
    }

    func testQwenLocalTuningDefaultsToDictionaryTermsOnly() {
        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "mlx-community/Qwen3-ASR-1.7B-4bit",
            rawValue: nil
        )

        XCTAssertEqual(settings.qwenContextBias, AppPreferenceKey.asrDictionaryTermsTemplateVariable)
    }


    func testMLXLocalTuningLoadsStoredSettingsWithoutWhisperTemperature() throws {
        let raw = """
        {"whisper":{"preset":"accuracyFirst","qwenContextBias":"","granitePromptBias":"","senseVoiceUseITN":false}}
        """

        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "mlx-community/whisper-large-v3-mlx",
            rawValue: raw
        )

        XCTAssertEqual(settings.preset, .accuracyFirst)
        XCTAssertEqual(settings.whisperTemperature, 0.0)
    }

    func testMLXWhisperTemperatureIsSanitized() throws {
        let stored = MLXLocalTuningSettingsStore.save(
            MLXLocalTuningSettings(whisperTemperature: 1.8),
            for: "mlx-community/whisper-large-v3-mlx",
            rawValue: nil
        )

        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "mlx-community/whisper-large-v3-mlx",
            rawValue: stored
        )

        XCTAssertEqual(settings.whisperTemperature, 1.0)
    }

    func testQwenLocalTuningMigratesLegacyDefaultContextBiasToDictionaryTermsOnly() throws {
        let legacyDefault = AppPromptDefaults.text(for: .qwenASRContextBias, language: .chineseSimplified)
        XCTAssertEqual(legacyDefault, AppPreferenceKey.asrDictionaryTermsTemplateVariable)

        let rawStoredSettings = [
            MLXLocalTuningSettingsStore.familyKey(for: "mlx-community/Qwen3-ASR-1.7B-4bit"): MLXLocalTuningSettings(
                qwenContextBias: """
                说话者的主要语言是 {{USER_MAIN_LANGUAGE}}，其他常用语言是 {{USER_OTHER_LANGUAGES}}。

                请将识别偏向于人名、产品名、技术术语和混合语言内容的正确拼写，并保持与原始发音一致，不要翻译。

                当音频中确实出现这些词时，请优先参考下列词典词汇：
                {{DICTIONARY_TERMS}}
                """
            )
        ]
        let data = try JSONEncoder().encode(rawStoredSettings)
        let stored = try XCTUnwrap(String(data: data, encoding: .utf8))

        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "mlx-community/Qwen3-ASR-1.7B-4bit",
            rawValue: stored
        )

        XCTAssertEqual(settings.qwenContextBias, AppPreferenceKey.asrDictionaryTermsTemplateVariable)
    }

    func testQwenLocalTuningMigratesResolvedLegacyContextBiasToDictionaryTermsOnly() throws {
        let rawStoredSettings = [
            MLXLocalTuningSettingsStore.familyKey(for: "mlx-community/Qwen3-ASR-1.7B-4bit"): MLXLocalTuningSettings(
                qwenContextBias: """
                说话者的主要语言是 Simplified Chinese，其他常用语言是 None specified。

                请将识别偏向于人名、产品名、技术术语和混合语言内容的正确拼写，并保持与原始发音一致，不要翻译。

                当音频中确实出现这些词时，请优先参考下列词典词汇：
                """
            )
        ]
        let data = try JSONEncoder().encode(rawStoredSettings)
        let stored = try XCTUnwrap(String(data: data, encoding: .utf8))

        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "mlx-community/Qwen3-ASR-1.7B-4bit",
            rawValue: stored
        )

        XCTAssertEqual(settings.qwenContextBias, AppPreferenceKey.asrDictionaryTermsTemplateVariable)
    }

    func testKnownQwenContextLeakageIsRemovedFromASROutput() {
        let leaked = "说话者的主要语言是 Simplified Chinese，其他常用语言是 None specified。"

        XCTAssertEqual(MLXTranscriptionPlanning.removingKnownASRContextLeakage(from: leaked), "")

        let mixed = """
        说话者的主要语言是 Simplified Chinese，其他常用语言是 None specified。
        今天要整理 Codex 体验。
        """

        XCTAssertEqual(
            MLXTranscriptionPlanning.removingKnownASRContextLeakage(from: mixed),
            "今天要整理 Codex 体验。"
        )
    }

    func testMLXModelFamilyRecognizesCohereTranscribe() {
        XCTAssertEqual(
            MLXModelFamily.family(for: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"),
            .cohereTranscribe
        )
    }

    func testMLXModelFamilyRecognizesLatestMLXAudioFamilies() {
        XCTAssertEqual(MLXModelFamily.family(for: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"), .nemotronASR)
        XCTAssertEqual(MLXModelFamily.family(for: "OpenMOSS-Team/MOSS-Transcribe-Diarize"), .mossTranscribeDiarize)
        XCTAssertEqual(MLXModelFamily.family(for: "mlx-community/parakeet-tdt-0.6b-v3"), .parakeet)
    }

    func testLatestMLXModelTuningRoundTripsAndSanitizes() {
        let stored = MLXLocalTuningSettingsStore.save(
            MLXLocalTuningSettings(
                cohereLongFormStrategy: .fixedChunks,
                cohereUsePunctuation: false,
                cohereMaxTokens: 4096,
                cohereTemperature: 1.4,
                nemotronStreamLatency: .fast
            ),
            for: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
            rawValue: nil
        )

        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
            rawValue: stored
        )

        XCTAssertEqual(settings.cohereLongFormStrategy, .fixedChunks)
        XCTAssertFalse(settings.cohereUsePunctuation)
        XCTAssertEqual(settings.cohereMaxTokens, 2048)
        XCTAssertEqual(settings.cohereTemperature, 1.0)
        XCTAssertEqual(settings.nemotronStreamLatency, .fast)
    }


    func testStreamingLatencySettingsUseBackwardCompatibleDefaults() {
        let legacy = """
        {"nemotronASR":{"preset":"balanced"},"voxtralRealtime":{"preset":"balanced"}}
        """

        XCTAssertEqual(
            MLXLocalTuningSettingsStore.resolvedSettings(
                for: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
                rawValue: legacy
            ).nemotronStreamLatency,
            .balanced
        )
        XCTAssertNil(MLXLocalTuningSettingsStore.load(from: legacy)["voxtralRealtime"])
    }

    func testMOSSLocalTuningUsesSeparateDictationAndMeetingDefaults() {
        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            rawValue: nil
        )

        XCTAssertEqual(settings.mossOutputMode, .plainText)
        XCTAssertEqual(settings.mossHotwords, AppPreferenceKey.asrDictionaryTermsTemplateVariable)
        XCTAssertEqual(settings.mossCustomPrompt, "")
        XCTAssertEqual(settings.mossMeetingOutputMode, .timestampedDiarization)
        XCTAssertEqual(settings.mossMeetingHotwords, AppPreferenceKey.asrDictionaryTermsTemplateVariable)
        XCTAssertEqual(settings.mossMeetingCustomPrompt, "")
    }

    func testMOSSLocalTuningRoundTripsPromptConfiguration() {
        let stored = MLXLocalTuningSettingsStore.save(
            MLXLocalTuningSettings(
                mossOutputMode: .customPrompt,
                mossHotwords: "  Voxt\nCodex  ",
                mossCustomPrompt: "  Transcribe with concise paragraphs.  ",
                mossMeetingOutputMode: .speakerOnly,
                mossMeetingHotwords: "  Alice\nBob  ",
                mossMeetingCustomPrompt: "  Keep speaker turns.  "
            ),
            for: "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            rawValue: nil
        )

        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            rawValue: stored
        )
        XCTAssertEqual(settings.mossOutputMode, .customPrompt)
        XCTAssertEqual(settings.mossHotwords, "Voxt\nCodex")
        XCTAssertEqual(settings.mossCustomPrompt, "Transcribe with concise paragraphs.")
        XCTAssertEqual(settings.mossMeetingOutputMode, .speakerOnly)
        XCTAssertEqual(settings.mossMeetingHotwords, "Alice\nBob")
        XCTAssertEqual(settings.mossMeetingCustomPrompt, "Keep speaker turns.")
        XCTAssertEqual(settings.mossSettings(for: .dictation).outputMode, .customPrompt)
        XCTAssertEqual(settings.mossSettings(for: .meeting).outputMode, .speakerOnly)
    }

    func testMOSSLocalTuningMigratesLegacyConfigurationToMeetingScope() throws {
        let legacy = """
        {"mossTranscribeDiarize":{"mossOutputMode":"timestampedDiarization","mossHotwords":"Voxt","mossCustomPrompt":"Legacy prompt"}}
        """

        let settings = MLXLocalTuningSettingsStore.resolvedSettings(
            for: "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            rawValue: legacy
        )

        XCTAssertEqual(settings.mossOutputMode, .plainText)
        XCTAssertEqual(settings.mossHotwords, "Voxt")
        XCTAssertEqual(settings.mossMeetingOutputMode, .timestampedDiarization)
        XCTAssertEqual(settings.mossMeetingHotwords, "Voxt")
        XCTAssertEqual(settings.mossMeetingCustomPrompt, "Legacy prompt")
    }

    func testMOSSStructuredSegmentsSanitizeTextWhilePreservingTimingAndSpeaker() {
        let segments = MLXTranscriber.mossStructuredSegments(from: [
            STTTranscriptSegment(
                text: "[S02] Hello [sniff] world",
                startTime: 1.25,
                endTime: 2.75,
                speakerID: "S02"
            ),
            STTTranscriptSegment(
                text: "[S02] [breath]",
                startTime: 2.80,
                endTime: 3.10,
                speakerID: "S02"
            )
        ])

        XCTAssertEqual(
            segments,
            [
                MLXStructuredTranscriptSegment(
                    startSeconds: 1.25,
                    endSeconds: 2.75,
                    speakerID: "S02",
                    text: "Hello world"
                )
            ]
        )
    }

    func testReliableStructuredSegmentsAcceptSentenceTimingAndRejectCoarseChunks() {
        let rawSegments = [
            STTTranscriptSegment(text: " First sentence. ", startTime: 0.2, endTime: 1.4),
            STTTranscriptSegment(text: "", startTime: 1.4, endTime: 2.0),
            STTTranscriptSegment(text: "Invalid timing", startTime: 2.0, endTime: 2.0),
        ]

        XCTAssertEqual(
            MLXTranscriber.reliableStructuredSegments(
                from: rawSegments,
                timingGranularity: .sentence
            ),
            [
                MLXStructuredTranscriptSegment(
                    startSeconds: 0.2,
                    endSeconds: 1.4,
                    text: "First sentence."
                )
            ]
        )
        XCTAssertEqual(
            MLXTranscriber.reliableStructuredSegments(
                from: rawSegments,
                timingGranularity: .chunk
            ),
            []
        )
    }

    func testLiveEndedStructuredSegmentsAlignWithBatchReliability() {
        let qwenOutput = STTOutput(
            text: "hello",
            segments: [
                STTTranscriptSegment(text: "hello", startTime: 0, endTime: 8, language: "English")
            ],
            language: "English",
            languageProvenance: .detected
        )
        XCTAssertTrue(
            MLXTranscriber.structuredSegmentsForLiveEnded(
                output: qwenOutput,
                modelFamily: .qwen3ASR,
                timingGranularity: .chunk
            ).isEmpty
        )

        let nemotronOutput = STTOutput(
            text: "Hello world",
            segments: [
                STTTranscriptSegment(text: "Hello world", startTime: 0.1, endTime: 1.2)
            ]
        )
        XCTAssertEqual(
            MLXTranscriber.structuredSegmentsForLiveEnded(
                output: nemotronOutput,
                modelFamily: .nemotronASR,
                timingGranularity: .sentence
            ),
            [
                MLXStructuredTranscriptSegment(
                    startSeconds: 0.1,
                    endSeconds: 1.2,
                    text: "Hello world"
                )
            ]
        )

        let mossOutput = STTOutput(
            text: "[S01] Hi",
            segments: [
                STTTranscriptSegment(
                    text: "[S01] Hi",
                    startTime: 0.2,
                    endTime: 0.9,
                    speakerID: "S01"
                )
            ]
        )
        XCTAssertEqual(
            MLXTranscriber.structuredSegmentsForLiveEnded(
                output: mossOutput,
                modelFamily: .mossTranscribeDiarize,
                timingGranularity: .sentence
            ),
            [
                MLXStructuredTranscriptSegment(
                    startSeconds: 0.2,
                    endSeconds: 0.9,
                    speakerID: "S01",
                    text: "Hi"
                )
            ]
        )
    }

    func testMOSSPromptSupportUsesOfficialModesAndHotwords() {
        XCTAssertEqual(
            MossASRPromptSupport.resolvedPrompt(
                outputMode: .speakerOnly,
                customPrompt: "",
                hotwords: "Voxt\nCodex"
            ),
            "Transcribe the audio as text using speaker labels such as [S01], [S02], and [S03]. Hotwords: Voxt, Codex"
        )
        XCTAssertEqual(
            MossASRPromptSupport.resolvedPrompt(
                outputMode: .customPrompt,
                customPrompt: "Keep paragraph breaks.",
                hotwords: ""
            ),
            "Keep paragraph breaks."
        )
    }

    func testMOSSMeetingAlwaysUsesStructuredGenerationOutput() {
        XCTAssertEqual(
            MossASRPromptSupport.generationOutputMode(
                requestedOutputMode: .plainText,
                scope: .meeting
            ),
            .timestampedDiarization
        )
        XCTAssertEqual(
            MossASRPromptSupport.generationOutputMode(
                requestedOutputMode: .customPrompt,
                scope: .meeting
            ),
            .timestampedDiarization
        )
        XCTAssertEqual(
            MossASRPromptSupport.generationOutputMode(
                requestedOutputMode: .plainText,
                scope: .dictation
            ),
            .plainText
        )
    }

    func testMOSSMeetingCustomPromptKeepsRequiredStructure() {
        let prompt = MossASRPromptSupport.resolvedPrompt(
            requestedOutputMode: .plainText,
            scope: .meeting,
            customPrompt: "Keep product names verbatim.",
            hotwords: "Voxt"
        )

        XCTAssertContains(prompt, "start with the timestamp and speaker ID")
        XCTAssertContains(prompt, "Additional transcription instructions: Keep product names verbatim.")
        XCTAssertContains(prompt, "Hotwords: Voxt")
    }

    func testMOSSRenderingRemovesOnlyConfiguredStructuredTags() {
        let raw = "[0.48][S01]Welcome everyone[1.66][2.10][S02]Ready to begin[3.25]"

        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(raw, outputMode: .speakerOnly),
            "[S01] Welcome everyone\n[S02] Ready to begin"
        )
        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(raw, outputMode: .plainText),
            "Welcome everyone Ready to begin"
        )
        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(raw, outputMode: .timestampedDiarization),
            raw
        )
    }

    func testMOSSPlainTextFlattensStreamingWindowsWithoutBreakingWordBoundaries() {
        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(
                "零九是一名在互联网公司做\n后端开发的软件工程师\n他每天上班后的第一件事",
                outputMode: .plainText
            ),
            "零九是一名在互联网公司做后端开发的软件工程师他每天上班后的第一件事"
        )
        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(
                "The server is healthy.\nReady to deploy\n(version 2).",
                outputMode: .plainText
            ),
            "The server is healthy. Ready to deploy (version 2)."
        )
    }

    func testMOSSPlainTextHidesAcousticEventsAndIncompleteSpeakerProtocol() {
        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(
                "[0.00][S01]Hello [sniff] there[1.20][1.30][S0",
                outputMode: .plainText
            ),
            "Hello there"
        )

        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(
                "[0.00][S What did you see?",
                outputMode: .plainText
            ),
            "What did you see?"
        )

        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(
                "[0.00][SO Yeah, pretty every week.",
                outputMode: .plainText
            ),
            "Yeah, pretty every week."
        )
    }

    func testMOSSMeetingRenderingPreservesStructuredLineBreaks() {
        let raw = "[0.00][S01]第一段[1.20]\n[1.30][S02]第二段[2.40]"

        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(raw, outputMode: .speakerOnly),
            "[S01] 第一段\n[S02] 第二段"
        )
        XCTAssertEqual(
            MossASRTranscriptRendering.renderedText(raw, outputMode: .timestampedDiarization),
            raw
        )
    }


    func testResolveDictationSettingsUsesMainLanguageAndContextualPhrases() {
        let settings = ASRHintSettings(
            followsUserMainLanguage: true,
            contextualPhrasesText: "Voxt\nFireRed\n Voxt \n",
            prefersOnDeviceRecognition: true,
            addsPunctuation: false,
            reportsPartialResults: false
        )

        let resolved = ASRHintResolver.resolveDictationSettings(
            settings: settings,
            userLanguageCodes: ["zh-Hant"]
        )

        XCTAssertEqual(resolved.localeIdentifier, "zh-TW")
        XCTAssertEqual(resolved.contextualPhrases, ["Voxt", "FireRed", "Voxt"])
        XCTAssertTrue(resolved.prefersOnDeviceRecognition)
        XCTAssertFalse(resolved.addsPunctuation)
        XCTAssertFalse(resolved.reportsPartialResults)
    }

    func testSanitizedDictationContextualPhrasesTrimBlankLines() {
        let settings = ASRHintSettingsStore.sanitized(
            ASRHintSettings(
                contextualPhrasesText: "\n  Voxt  \n\n FireRed ASR \n"
            ),
            for: .dictation
        )

        XCTAssertEqual(settings.contextualPhrasesText, "Voxt\nFireRed ASR")
    }

    func testLanguageSummaryAndOutputVariantDescription() {
        XCTAssertEqual(
            ASRHintResolver.selectedLanguageSummary(["zh-Hans", "en"]),
            "Simplified Chinese, English"
        )
        XCTAssertEqual(
            ASRHintResolver.outputVariantDescription(for: UserMainLanguageOption.option(for: "zh-hant")!),
            AppLocalization.localizedString("Traditional Chinese")
        )
    }
}
