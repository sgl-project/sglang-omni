import XCTest
@testable import Voxt

final class RemoteModelConfigurationASRTests: RemoteModelConfigurationTestCase {
    func testDoubaoConfigurationUsesExpectedDefaults() {
        XCTAssertEqual(DoubaoASRConfiguration.resolvedEndpoint("", model: ""), DoubaoASRConfiguration.defaultNostreamEndpoint)
        XCTAssertEqual(
            DoubaoASRConfiguration.resolvedStreamingEndpoint("", model: DoubaoASRConfiguration.modelV1),
            DoubaoASRConfiguration.defaultStreamingEndpointV1
        )
        XCTAssertEqual(
            DoubaoASRConfiguration.resolvedStreamingEndpoint("", model: ""),
            DoubaoASRConfiguration.defaultStreamingEndpointV2
        )
    }

    func testDoubaoFullRequestPayloadIncludesLanguageAndVariant() {
        let payload = DoubaoASRConfiguration.fullRequestPayload(
            requestID: "req-1",
            userID: "user-1",
            language: "zh-CN",
            chineseOutputVariant: "zh-Hans",
            enableNonstream: true
        )

        let audio = payload["audio"] as? [String: Any]
        let request = payload["request"] as? [String: Any]
        XCTAssertEqual(audio?["language"] as? String, "zh-CN")
        XCTAssertEqual(request?["output_zh_variant"] as? String, "zh-Hans")
        XCTAssertEqual(request?["enable_nonstream"] as? Bool, true)
    }

    func testDoubaoFullRequestPayloadIncludesDictionaryContextAndCorpus() throws {
        let payload = DoubaoASRConfiguration.fullRequestPayload(
            requestID: "req-1",
            userID: "user-1",
            language: "zh-CN",
            chineseOutputVariant: "zh-Hans",
            dictionaryPayload: DoubaoDictionaryRequestPayload(
                hotwords: ["OpenAI"],
                correctWords: ["open ai": "OpenAI"]
            )
        )

        let request = try XCTUnwrap(payload["request"] as? [String: Any])
        let corpus = try XCTUnwrap(request["corpus"] as? [String: Any])

        let contextString = try XCTUnwrap(corpus["context"] as? String)
        let contextData = try XCTUnwrap(contextString.data(using: .utf8))
        let context = try XCTUnwrap(try JSONSerialization.jsonObject(with: contextData) as? [String: Any])
        XCTAssertEqual((context["hotwords"] as? [[String: String]])?.first?["word"], "OpenAI")
        XCTAssertEqual((context["correct_words"] as? [String: String])?["open ai"], "OpenAI")
    }

    func testDoubaoRecommendedStreamingChunkSplitsAndFlushesTrailingPartial() {
        let packetBytes = DoubaoASRConfiguration.recommendedStreamingPacketBytes
        var buffer = Data(repeating: 1, count: packetBytes * 2 + 123)

        let first = DoubaoASRConfiguration.popRecommendedStreamingChunk(from: &buffer, includeTrailingPartial: false)
        let second = DoubaoASRConfiguration.popRecommendedStreamingChunk(from: &buffer, includeTrailingPartial: false)
        let noneYet = DoubaoASRConfiguration.popRecommendedStreamingChunk(from: &buffer, includeTrailingPartial: false)
        let trailing = DoubaoASRConfiguration.popRecommendedStreamingChunk(from: &buffer, includeTrailingPartial: true)

        XCTAssertEqual(first?.count, packetBytes)
        XCTAssertEqual(second?.count, packetBytes)
        XCTAssertNil(noneYet)
        XCTAssertEqual(trailing?.count, 123)
        XCTAssertTrue(buffer.isEmpty)
    }

    func testDoubaoFinalStreamingSequenceUsesNextSequence() {
        XCTAssertEqual(DoubaoASRConfiguration.finalStreamingSequence(nextAudioSequence: 2), -2)
        XCTAssertEqual(DoubaoASRConfiguration.finalStreamingSequence(nextAudioSequence: 16), -16)
    }

    func testDoubaoParserRejectsOversizedCompressedPayload() {
        let oversizedPayload = Data(
            repeating: 0,
            count: DoubaoPacketCodec.maxCompressedPayloadBytes + 1
        )
        let packet = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeFullServerResponse,
            messageFlags: DoubaoProtocol.flagPositiveSequence,
            serialization: DoubaoProtocol.serializationJSON,
            compression: DoubaoProtocol.compressionGzip,
            sequence: 1,
            payload: oversizedPayload
        )

        XCTAssertThrowsError(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet))
    }

    func testDoubaoParserRejectsHighExpansionGzipPayload() throws {
        let largePlaintext = Data(repeating: 65, count: 1_048_577)
        let encoded = try DoubaoPacketCodec.encodePayload(largePlaintext)
        XCTAssertEqual(encoded.compression, DoubaoProtocol.compressionGzip)
        let packet = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeFullServerResponse,
            messageFlags: DoubaoProtocol.flagPositiveSequence,
            serialization: DoubaoProtocol.serializationJSON,
            compression: encoded.compression,
            sequence: 1,
            payload: encoded.payload
        )

        XCTAssertThrowsError(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet))
    }

    func testAliyunASRModelOptionsIncludeOmniRealtimeModels() {
        let ids = Set(RemoteASRProvider.aliyunBailianASR.modelOptions.map(\.id))
        XCTAssertTrue(ids.contains("qwen-audio-3.0-asr-flash-streaming"))
        XCTAssertTrue(ids.contains("qwen3.5-omni-flash-realtime"))
        XCTAssertTrue(ids.contains("qwen3.5-omni-plus-realtime"))
        XCTAssertTrue(ids.contains("qwen-omni-turbo-realtime"))
    }

    func testStepFunASRModelCapabilitiesMatchDocumentedTransportFeatures() {
        let standard = StepFunASRModelCapabilities.forModel("stepaudio-2.5-asr")
        XCTAssertTrue(standard.supportsHotwords)
        XCTAssertFalse(standard.supportsPrompt)
        XCTAssertFalse(standard.usesRealtimeWebSocket)

        let pro = StepFunASRModelCapabilities.forModel("stepaudio-2-asr-pro")
        XCTAssertTrue(pro.supportsHotwords)
        XCTAssertTrue(pro.supportsPrompt)
        XCTAssertFalse(pro.usesRealtimeWebSocket)

        let realtime = StepFunASRModelCapabilities.forModel("step-asr-1.1-stream")
        XCTAssertFalse(realtime.supportsHotwords)
        XCTAssertTrue(realtime.supportsPrompt)
        XCTAssertTrue(realtime.usesRealtimeWebSocket)
    }

    func testAliyunASRModelCapabilitiesSeparateVocabularyAndLanguageSupport() {
        let qwenAudio = AliyunASRModelCapabilities.forModel("qwen-audio-3.0-asr-flash-streaming")
        XCTAssertEqual(qwenAudio.family, .qwenAudio3)
        XCTAssertTrue(qwenAudio.supportsInlineVocabulary)
        XCTAssertEqual(qwenAudio.maximumLanguageHints, 4)
        XCTAssertTrue(qwenAudio.supportedLanguageCodes.contains("vi"))
        XCTAssertTrue(qwenAudio.supportedLanguageCodes.contains("tl"))

        let qwen3 = AliyunASRModelCapabilities.forModel("qwen3-asr-flash-realtime")
        XCTAssertEqual(qwen3.family, .qwen3ASR)
        XCTAssertFalse(qwen3.supportsLanguageHints)
        XCTAssertTrue(qwen3.supportedLanguageCodes.contains("fil"))
        XCTAssertTrue(qwen3.supportsManualCommit)

        let paraformer = AliyunASRModelCapabilities.forModel("paraformer-realtime-v2")
        XCTAssertTrue(paraformer.supportsInverseTextNormalization)
        XCTAssertTrue(paraformer.supportsDisfluencyRemoval)
    }

    func testAliyunRealtimeModelFamilyDetectionSeparatesQwenAndOmni() {
        XCTAssertEqual(
            RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: "qwen3-asr-flash-realtime"),
            .qwenASR
        )
        XCTAssertEqual(
            RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: "qwen3.5-omni-flash-realtime"),
            .omniASR
        )
        XCTAssertEqual(
            RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: "qwen3.5-omni-plus-realtime"),
            .omniASR
        )
        XCTAssertEqual(
            RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: "qwen-omni-turbo-realtime"),
            .omniASR
        )
        XCTAssertNil(RemoteASREndpointSupport.aliyunQwenRealtimeSessionKind(for: "fun-asr-realtime"))
    }

    func testAliyunOmniSessionUpdatePayloadUsesExplicitInputTranscriptionModel() throws {
        let payload = AliyunQwenRealtimePayloadSupport.sessionUpdatePayload(
            kind: .omniASR,
            hintPayload: ResolvedASRHintPayload(language: "zh", languageHints: ["zh"])
        )

        let session = try XCTUnwrap(payload["session"] as? [String: Any])
        let transcription = try XCTUnwrap(session["input_audio_transcription"] as? [String: Any])
        let turnDetection = try XCTUnwrap(session["turn_detection"] as? [String: Any])

        XCTAssertEqual(payload["type"] as? String, "session.update")
        XCTAssertEqual(session["modalities"] as? [String], ["text"])
        XCTAssertEqual(session["input_audio_format"] as? String, "pcm")
        XCTAssertEqual(session["sample_rate"] as? Int, 16000)
        XCTAssertEqual(transcription["model"] as? String, "qwen3-asr-flash-realtime")
        XCTAssertEqual(transcription["language"] as? String, "zh")
        XCTAssertEqual(turnDetection["type"] as? String, "server_vad")
        XCTAssertEqual(turnDetection["threshold"] as? Double, 0.35)
        XCTAssertEqual(turnDetection["silence_duration_ms"] as? Int, 800)
    }

    func testAliyunOmniRealtimeDoesNotRequireManualCommitWhenUsingServerVAD() {
        XCTAssertFalse(AliyunQwenRealtimeSessionKind.omniASR.shouldCommitBeforeFinish)
    }

    func testAliyunQwenSessionUpdatePayloadLeavesTranscriptionModelUnset() throws {
        let payload = AliyunQwenRealtimePayloadSupport.sessionUpdatePayload(
            kind: .qwenASR,
            hintPayload: ResolvedASRHintPayload(language: nil, languageHints: [])
        )

        let session = try XCTUnwrap(payload["session"] as? [String: Any])
        let transcription = try XCTUnwrap(session["input_audio_transcription"] as? [String: Any])

        XCTAssertNil(transcription["model"])
        XCTAssertNil(transcription["language"])
    }

    func testAliyunQwenSessionUpdatePayloadCanDisableTurnDetectionForMeeting() throws {
        let payload = AliyunQwenRealtimePayloadSupport.sessionUpdatePayload(
            kind: .qwenASR,
            hintPayload: ResolvedASRHintPayload(language: nil, languageHints: []),
            includesTurnDetection: false
        )

        let session = try XCTUnwrap(payload["session"] as? [String: Any])

        XCTAssertNil(session["turn_detection"])
    }

    func testDoubaoConfigurationRequiresBothAppIDAndAccessToken() {
        let onlyAppID = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app"
        )
        let onlyAccessToken = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            accessToken: "doubao-token"
        )
        let complete = TestFactories.makeRemoteConfiguration(
            providerID: RemoteASRProvider.doubaoASR.rawValue,
            model: DoubaoASRConfiguration.modelV2,
            appID: "doubao-app",
            accessToken: "doubao-token"
        )

        XCTAssertFalse(onlyAppID.isConfigured)
        XCTAssertFalse(onlyAccessToken.isConfigured)
        XCTAssertTrue(complete.isConfigured)
    }

    func testRemoteASRTextSanitizerRejectsIdentifierLikeStrings() {
        XCTAssertTrue(RemoteASRTextSanitizer.isLikelyIdentifierText("9ff6a1a4-f758-4a87-b761-11508533c499"))
        XCTAssertTrue(RemoteASRTextSanitizer.isLikelyIdentifierText("abc123ef456789ab_cdef1234567890"))
        XCTAssertTrue(RemoteASRTextSanitizer.isLikelyIdentifierText("9ff6a1a4f7584a87b76111508533c499"))
    }

    func testRemoteASRTextSanitizerAllowsNaturalLanguageText() {
        XCTAssertFalse(RemoteASRTextSanitizer.isLikelyIdentifierText("你好"))
        XCTAssertFalse(RemoteASRTextSanitizer.isLikelyIdentifierText("我们今天继续开会"))
        XCTAssertFalse(RemoteASRTextSanitizer.isLikelyIdentifierText("hello world 2026"))
    }

    func testGoogleGeminiASROffersLiveTranscribeModelOnly() {
        XCTAssertEqual(RemoteASRProvider.googleGeminiASR.suggestedModel, "gemini-3.5-transcribe-live")
        XCTAssertEqual(
            RemoteASRProvider.googleGeminiASR.modelOptions.map(\.id),
            ["gemini-3.5-transcribe-live"]
        )
        XCTAssertTrue(RemoteASRProvider.allCases.contains(.googleGeminiASR))
    }
}
