// RemoteASRSupportTests.swift
// Provides Remote ASRSupport Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RemoteASRSupportTests: XCTestCase {
    func testOpenAITranscriptionMultipartFieldsOmitStreamForFileTranscription() {
        let fields = RemoteASRTextSupport.openAITranscriptionMultipartFields(
            model: "gpt-4o-mini-transcribe",
            hintPayload: ResolvedASRHintPayload(
                language: "zh",
                languageHints: ["zh"],
                prompt: "Prefer product names."
            )
        )

        XCTAssertEqual(fields["response_format"], "json")
        XCTAssertEqual(fields["language"], "zh")
        XCTAssertEqual(fields["prompt"], "Prefer product names.")
        XCTAssertNil(fields["stream"])
    }

    func testOpenAITranscriptionMultipartFieldsOmitPromptForDiarizeModel() {
        let fields = RemoteASRTextSupport.openAITranscriptionMultipartFields(
            model: "gpt-4o-transcribe-diarize",
            hintPayload: ResolvedASRHintPayload(
                language: "en",
                languageHints: ["en"],
                prompt: "Ignore for diarize."
            )
        )

        XCTAssertEqual(fields["response_format"], "json")
        XCTAssertEqual(fields["language"], "en")
        XCTAssertNil(fields["prompt"])
        XCTAssertNil(fields["stream"])
    }

    func testRemoteUploadVADPolicyUsesClientVADForFileUploadProviders() {
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .openAIWhisper,
                model: "gpt-4o-mini-transcribe",
                localVADMode: .energy
            ),
            .fileUploadClientVAD
        )
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .glmASR,
                model: "glm-asr-1",
                localVADMode: .energy
            ),
            .fileUploadClientVAD
        )
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .xiaomiMiMoASR,
                model: "mimo-v2.5-asr",
                localVADMode: .energy
            ),
            .fileUploadClientVAD
        )
    }

    func testRemoteUploadVADPolicyDoesNotClientTrimRealtimeServerVADProviders() {
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .aliyunBailianASR,
                model: "qwen3-asr-flash-realtime",
                localVADMode: .energy
            ),
            .realtimeServerVAD
        )
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .stepFunASR,
                model: "step-asr-1.1-stream",
                localVADMode: .energy
            ),
            .realtimeServerVAD
        )
    }

    func testRemoteUploadVADPolicyLeavesRealtimeStreamingProvidersUnchanged() {
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .doubaoASR,
                model: DoubaoASRConfiguration.modelV2,
                localVADMode: .energy
            ),
            .realtimeUnchanged(reason: "doubao-streaming")
        )
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .aliyunBailianASR,
                model: "fun-asr-realtime",
                localVADMode: .energy
            ),
            .realtimeUnchanged(reason: "aliyun-fun-realtime")
        )
    }

    func testRemoteUploadVADPolicyDisablesWhenLocalVADIsOff() {
        XCTAssertEqual(
            RemoteASRAudioUploadVADPolicyResolver.policy(
                provider: .openAIWhisper,
                model: "gpt-4o-mini-transcribe",
                localVADMode: .off
            ),
            .disabled(reason: "local-vad-off")
        )
    }

    func testRemoteUploadPreprocessorExportsShorterClientVADUploadAudio() async throws {
        let sourceURL = HistoryAudioArchiveSupport.temporaryArchiveURL(prefix: "remote-upload-vad-test-source")
        defer { try? FileManager.default.removeItem(at: sourceURL) }

        let sampleRate = HistoryAudioArchiveSupport.targetSampleRate
        let leadingSilence = Array(repeating: Float(0), count: Int(sampleRate * 2))
        let speech = Array(repeating: Float(0.35), count: Int(sampleRate))
        let trailingSilence = Array(repeating: Float(0), count: Int(sampleRate * 2))
        let samples = leadingSilence + speech + trailingSilence
        XCTAssertTrue(try HistoryAudioArchiveSupport.exportWAV(samples: samples, sampleRate: sampleRate, to: sourceURL))

        let preparation = try await RemoteASRAudioUploadPreprocessor.prepareUploadAudio(
            originalFileURL: sourceURL,
            provider: .openAIWhisper,
            configuration: RemoteProviderConfiguration(
                providerID: RemoteASRProvider.openAIWhisper.rawValue,
                model: "gpt-4o-mini-transcribe",
                endpoint: "",
                apiKey: "test"
            ),
            localVADMode: .energy,
            useCase: .transcription
        )
        defer { preparation.cleanupTemporaryUploadFileIfNeeded() }

        XCTAssertTrue(preparation.shouldRequestRemoteASR)
        XCTAssertEqual(preparation.policy, .fileUploadClientVAD)
        XCTAssertNotNil(preparation.temporaryUploadFileURL)
        XCTAssertLessThan(preparation.uploadDurationSeconds ?? 0, preparation.originalDurationSeconds ?? 0)
        XCTAssertGreaterThan(preparation.speechSegmentCount, 0)

        let uploadSamples = try HistoryAudioArchiveSupport.readWAVSamples(from: preparation.uploadFileURL)
        XCTAssertLessThan(uploadSamples.count, samples.count)
        XCTAssertGreaterThan(uploadSamples.count, 0)
    }

    func testExtractStreamErrorMessageReadsNestedErrorPayload() {
        let message = RemoteASRTextSupport.extractStreamErrorMessage(
            fromLine: #"{"error":{"code":"invalid_api_key","message":"API key is invalid"}}"#
        )

        XCTAssertEqual(message, "API key is invalid")
    }

    func testExtractStreamErrorMessageReadsErrorEventPayload() {
        let message = RemoteASRTextSupport.extractStreamErrorMessage(
            fromLine: #"{"event":"error","message":"model is required"}"#
        )

        XCTAssertEqual(message, "model is required")
    }

    func testExtractStreamErrorMessageIgnoresNormalTextPayload() {
        let message = RemoteASRTextSupport.extractStreamErrorMessage(
            fromLine: #"{"text":"hello world"}"#
        )

        XCTAssertNil(message)
    }

    func testStepFunTranscriptionPayloadIncludesLanguageAndHotwords() {
        let payload = StepFunPayloadSupport.transcriptionPayload(
            model: "stepaudio-2.5-asr",
            hintPayload: ResolvedASRHintPayload(
                language: "zh",
                contextualPhrases: ["Voxt", "FireRed"]
            )
        )

        XCTAssertEqual(payload["model"] as? String, "stepaudio-2.5-asr")
        XCTAssertEqual(payload["language"] as? String, "zh")
        XCTAssertEqual(payload["enable_itn"] as? Bool, true)
        XCTAssertEqual(payload["hotwords"] as? [String], ["Voxt", "FireRed"])
        XCTAssertNil(payload["enable_timestamp"])
        XCTAssertNil(payload["prompt"])
        XCTAssertFalse(StepFunPayloadSupport.supportsSSEPrompt(model: "stepaudio-2.5-asr"))
    }

    func testStepFunRealtimeModelDoesNotReceiveHotwords() {
        let payload = StepFunPayloadSupport.transcriptionPayload(
            model: "step-asr-1.1-stream",
            hintPayload: ResolvedASRHintPayload(
                language: "zh",
                contextualPhrases: ["Voxt"]
            )
        )

        XCTAssertNil(payload["hotwords"])
        XCTAssertTrue(StepFunASRModelCapabilities.forModel("step-asr-1.1-stream").usesRealtimeWebSocket)
        XCTAssertTrue(StepFunASRModelCapabilities.forModel("step-asr-1.1-stream").supportsPrompt)
    }

    func testAliyunFunRealtimeParametersDoNotUseAnAliyunVocabularyID() {
        let parameters = AliyunFunRealtimePayloadSupport.parameters(
            model: "fun-asr-realtime",
            hintPayload: ResolvedASRHintPayload(
                languageHints: ["zh", "en"],
                contextualPhrases: ["Voxt", "FireRed"]
            )
        )

        XCTAssertEqual(parameters["sample_rate"] as? Int, 16000)
        XCTAssertEqual(parameters["format"] as? String, "pcm")
        XCTAssertEqual(parameters["language_hints"] as? [String], ["zh"])
        XCTAssertNil(parameters["hotwords"])
        XCTAssertNil(parameters["vocabulary"])
        XCTAssertNil(parameters["vocabulary_id"])
        XCTAssertEqual(parameters["semantic_punctuation_enabled"] as? Bool, false)
    }

    func testAliyunQwenAudioRealtimeParametersUseInlineVocabulary() {
        let parameters = AliyunFunRealtimePayloadSupport.parameters(
            model: "qwen-audio-3.0-asr-flash-streaming",
            hintPayload: ResolvedASRHintPayload(
                languageHints: ["zh", "en", "ja", "de", "ko"],
                contextualPhrases: ["Voxt", "FireRed"]
            )
        )

        XCTAssertEqual(parameters["language_hints"] as? [String], ["zh", "en", "ja", "de"])
        XCTAssertEqual(parameters["vocabulary"] as? [String: Int], ["Voxt": 4, "FireRed": 4])
        XCTAssertEqual(parameters["max_sentence_silence"] as? Int, 1300)
        XCTAssertEqual(parameters["speech_noise_threshold"] as? Double, 0.35)
        XCTAssertEqual(parameters["semantic_punctuation_enabled"] as? Bool, false)
        XCTAssertNil(parameters["vocabulary_id"])
    }

    func testAliyunFunRealtimeContextIsLimitedToSupportedModelsAnd400Characters() {
        let context = AliyunFunRealtimePayloadSupport.context(
            model: "fun-asr-realtime",
            phrases: [String(repeating: "a", count: 500)]
        )
        XCTAssertEqual(context.count, 1)
        let content = context.first?["content"] as? [[String: Any]]
        XCTAssertEqual((content?.first?["text"] as? String)?.count, 400)
        XCTAssertTrue(AliyunFunRealtimePayloadSupport.context(model: "paraformer-realtime-v2", phrases: ["Voxt"]).isEmpty)
    }

    func testAliyunQwenRealtimeSettingsCanDisableServerVAD() throws {
        let payload = AliyunQwenRealtimePayloadSupport.sessionUpdatePayload(
            kind: .qwenASR,
            hintPayload: ResolvedASRHintPayload(language: "zh"),
            settings: AliyunASRModelSettings(useManualCommit: true)
        )
        let session = try XCTUnwrap(payload["session"] as? [String: Any])
        XCTAssertNil(session["turn_detection"])
    }

    func testStepFunProTranscriptionPayloadCanIncludePrompt() {
        let payload = StepFunPayloadSupport.transcriptionPayload(
            model: "stepaudio-2-asr-pro",
            hintPayload: ResolvedASRHintPayload(
                language: "zh",
                prompt: "Prefer these terms.\nVoxt",
                contextualPhrases: ["Voxt"]
            ),
            includePrompt: StepFunPayloadSupport.supportsSSEPrompt(model: "stepaudio-2-asr-pro")
        )

        XCTAssertEqual(payload["model"] as? String, "stepaudio-2-asr-pro")
        XCTAssertEqual(payload["prompt"] as? String, "Prefer these terms.\nVoxt")
        XCTAssertEqual(payload["hotwords"] as? [String], ["Voxt"])
        XCTAssertTrue(StepFunPayloadSupport.supportsSSEPrompt(model: "stepaudio-2-asr-pro"))
    }

    func testStepFunSSEDoneTextIsParsedAsCompletedResult() {
        let delta = StepFunPayloadSupport.parseSSEDataLine(
            #"{"type":"transcript.text.delta","delta":"一二三四五。"}"#
        )
        let completed = StepFunPayloadSupport.parseSSEDataLine(
            #"{"type":"transcript.text.done","text":"12345。"}"#
        )

        XCTAssertEqual(delta, .delta("一二三四五。"))
        XCTAssertEqual(completed, .completed("12345。"))
    }

    func testStepFunRealtimeSessionUpdateUsesWebSocketShape() throws {
        let payload = StepFunPayloadSupport.sessionUpdatePayload(
            model: "step-asr-1.1-stream",
            hintPayload: ResolvedASRHintPayload(
                language: "zh",
                prompt: "Prefer these terms.\nVoxt",
                contextualPhrases: ["Voxt"]
            ),
            useServerVAD: true
        )

        XCTAssertEqual(payload["type"] as? String, "session.update")
        let session = try XCTUnwrap(payload["session"] as? [String: Any])
        let audio = try XCTUnwrap(session["audio"] as? [String: Any])
        let input = try XCTUnwrap(audio["input"] as? [String: Any])
        let transcription = try XCTUnwrap(input["transcription"] as? [String: Any])
        let format = try XCTUnwrap(input["format"] as? [String: Any])

        XCTAssertEqual(transcription["model"] as? String, "step-asr-1.1-stream")
        XCTAssertEqual(transcription["prompt"] as? String, "Prefer these terms.\nVoxt")
        XCTAssertNil(transcription["hotwords"])
        XCTAssertEqual(transcription["full_rerun_on_commit"] as? Bool, true)
        XCTAssertEqual(format["codec"] as? String, "pcm_s16le")
        XCTAssertNotNil(input["turn_detection"])
    }

    func testStepFunRealtimeEndpointRemapsSSEEndpoint() {
        XCTAssertEqual(
            RemoteASREndpointSupport.resolvedStepFunRealtimeEndpoint("https://api.stepfun.com/v1/audio/asr/sse"),
            "wss://api.stepfun.com/v1/realtime/asr/stream"
        )
    }

    func testXiaomiMiMoASRPayloadUsesChatAudioShapeAndLanguage() throws {
        let payload = RemoteASRTextSupport.xiaomiMiMoASRPayload(
            model: "",
            audioData: Data([0x01, 0x02, 0x03]),
            mimeType: "audio/wav",
            hintPayload: ResolvedASRHintPayload(language: "zh")
        )

        XCTAssertEqual(payload["model"] as? String, "mimo-v2.5-asr")
        XCTAssertNil(payload["stream"])

        let options = try XCTUnwrap(payload["asr_options"] as? [String: Any])
        XCTAssertEqual(options["language"] as? String, "zh")

        let messages = try XCTUnwrap(payload["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.first?["role"] as? String, "user")

        let content = try XCTUnwrap(messages.first?["content"] as? [[String: Any]])
        XCTAssertEqual(content.first?["type"] as? String, "input_audio")

        let inputAudio = try XCTUnwrap(content.first?["input_audio"] as? [String: Any])
        XCTAssertEqual(inputAudio["data"] as? String, "data:audio/wav;base64,AQID")
    }

    func testXiaomiMiMoASRPayloadFallsBackToAutoForUnsupportedLanguage() throws {
        let payload = RemoteASRTextSupport.xiaomiMiMoASRPayload(
            model: "mimo-v2.5-asr",
            audioData: Data([0x00]),
            mimeType: "audio/wav",
            hintPayload: ResolvedASRHintPayload(language: "ja")
        )

        let options = try XCTUnwrap(payload["asr_options"] as? [String: Any])
        XCTAssertEqual(options["language"] as? String, "auto")
        XCTAssertNil(payload["stream"])
    }

    func testXiaomiMiMoASREndpointNormalizesChatCompletions() {
        XCTAssertEqual(
            RemoteASREndpointSupport.resolvedXiaomiMiMoASREndpoint(""),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
        XCTAssertEqual(
            RemoteASREndpointSupport.resolvedXiaomiMiMoASREndpoint("https://api.xiaomimimo.com"),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
        XCTAssertEqual(
            RemoteASREndpointSupport.resolvedXiaomiMiMoASREndpoint("https://api.xiaomimimo.com/v1/models"),
            "https://api.xiaomimimo.com/v1/chat/completions"
        )
    }

    func testGeminiLiveEndpointResolutionNormalizesSchemeAndStripsKey() {
        XCTAssertEqual(
            RemoteASREndpointSupport.resolvedGeminiLiveEndpoint(""),
            RemoteASREndpointSupport.geminiLiveDefaultEndpoint
        )
        XCTAssertEqual(
            RemoteASREndpointSupport.resolvedGeminiLiveEndpoint("https://generativelanguage.googleapis.com"),
            RemoteASREndpointSupport.geminiLiveDefaultEndpoint
        )
        XCTAssertEqual(
            RemoteASREndpointSupport.resolvedGeminiLiveEndpoint("https://relay.example.com/ws/bidi?key=secret"),
            "wss://relay.example.com/ws/bidi"
        )
    }

    func testGeminiLiveURLAppendsAPIKeyExactlyOnce() throws {
        let url = try XCTUnwrap(
            RemoteASREndpointSupport.geminiLiveURL(
                endpoint: "wss://relay.example.com/ws/bidi?key=stale",
                apiKey: "fresh-key"
            )
        )
        let components = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let keys = (components.queryItems ?? []).filter { $0.name == "key" }
        XCTAssertEqual(keys.count, 1)
        XCTAssertEqual(keys.first?.value, "fresh-key")
    }

    func testGeminiLiveSetupPayloadQualifiesModelAndCarriesHints() throws {
        let payload = GeminiLivePayloadSupport.setupPayload(
            model: "",
            hintPayload: ResolvedASRHintPayload(
                language: "zh",
                languageHints: ["zh", "zh", "en"],
                contextualPhrases: ["Voxt", "Voxt", "mihomo"]
            )
        )

        let setup = try XCTUnwrap(payload["setup"] as? [String: Any])
        XCTAssertEqual(setup["model"] as? String, "models/\(GeminiLivePayloadSupport.defaultModel)")

        let generation = try XCTUnwrap(setup["generationConfig"] as? [String: Any])
        XCTAssertEqual(generation["responseModalities"] as? [String], ["TEXT"])

        let transcription = try XCTUnwrap(setup["inputAudioTranscription"] as? [String: Any])
        XCTAssertEqual(transcription["mode"] as? String, "SMART")
        XCTAssertEqual(transcription["languageCodes"] as? [String], ["zh", "en"])
        XCTAssertEqual(transcription["customVocabulary"] as? [String], ["Voxt", "mihomo"])
    }

    func testGeminiLiveSetupPayloadOmitsEmptyVocabularyAndKeepsAutoDetect() throws {
        let payload = GeminiLivePayloadSupport.setupPayload(
            model: "models/gemini-3.5-transcribe-live",
            hintPayload: ResolvedASRHintPayload(language: nil)
        )

        let setup = try XCTUnwrap(payload["setup"] as? [String: Any])
        XCTAssertEqual(setup["model"] as? String, "models/gemini-3.5-transcribe-live")

        let transcription = try XCTUnwrap(setup["inputAudioTranscription"] as? [String: Any])
        XCTAssertEqual(transcription["languageCodes"] as? [String], [])
        XCTAssertNil(transcription["customVocabulary"])
    }

    func testGeminiLiveAudioPayloadUsesPCM16MimeType() throws {
        let payload = GeminiLivePayloadSupport.audioPayload(Data([0x01, 0x02, 0x03]))
        let realtime = try XCTUnwrap(payload["realtimeInput"] as? [String: Any])
        let audio = try XCTUnwrap(realtime["audio"] as? [String: Any])
        XCTAssertEqual(audio["mimeType"] as? String, "audio/pcm;rate=16000")
        XCTAssertEqual(audio["data"] as? String, "AQID")
        XCTAssertEqual(
            (GeminiLivePayloadSupport.audioStreamEndPayload["realtimeInput"] as? [String: Any])?["audioStreamEnd"] as? Bool,
            true
        )
    }

    func testGeminiLiveTranscriptUpdateReadsBothProtoCasings() {
        let camel = GeminiLivePayloadSupport.transcriptUpdate(from: [
            "serverContent": [
                "interimInputTranscription": ["text": "你好 "],
                "inputTranscription": ["text": "你好，板爷。"]
            ]
        ])
        XCTAssertEqual(camel.interim, "你好")
        XCTAssertEqual(camel.final, "你好，板爷。")

        let snake = GeminiLivePayloadSupport.transcriptUpdate(from: [
            "server_content": ["input_transcription": ["text": "hello"]]
        ])
        XCTAssertEqual(snake.final, "hello")
        XCTAssertNil(snake.interim)
    }

    func testGeminiLiveSetupCompleteAndErrorDetection() {
        XCTAssertTrue(GeminiLivePayloadSupport.isSetupComplete(["setupComplete": [:]]))
        XCTAssertTrue(GeminiLivePayloadSupport.isSetupComplete(["setup_complete": [:]]))
        XCTAssertFalse(GeminiLivePayloadSupport.isSetupComplete(["serverContent": [:]]))
        XCTAssertEqual(
            GeminiLivePayloadSupport.errorMessage(from: ["error": ["message": "API key not valid"]]),
            "API key not valid"
        )
        XCTAssertNil(GeminiLivePayloadSupport.errorMessage(from: ["serverContent": [:]]))
    }

    func testGeminiLiveTranscriptJoiningSkipsSpacesAroundCJK() {
        XCTAssertEqual(GeminiLiveTranscriptJoining.join(["今天开会", "讨论了配置。"]), "今天开会讨论了配置。")
        XCTAssertEqual(GeminiLiveTranscriptJoining.join(["hello there", "how are you"]), "hello there how are you")
        XCTAssertEqual(GeminiLiveTranscriptJoining.join(["我们用", "mihomo"]), "我们用mihomo")
        XCTAssertEqual(GeminiLiveTranscriptJoining.join(["", "  ", "only"]), "only")
    }
}
