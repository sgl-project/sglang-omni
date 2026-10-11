// OmniTranscriptionRequestTests.swift
// Covers the request Voxt sends to the local Omni server.

import XCTest
@testable import Voxt

final class OmniTranscriptionRequestTests: XCTestCase {
    private func fieldValue(_ name: String, in body: Data) -> String? {
        let text = String(decoding: body, as: UTF8.self)
        guard let range = text.range(of: "name=\"\(name)\"\r\n\r\n") else { return nil }
        let rest = text[range.upperBound...]
        return String(rest[..<(rest.range(of: "\r\n")?.lowerBound ?? rest.endIndex)])
    }

    func testQwenFinalRequestsReproduceTheSwiftAudioLayout() {
        let request = OmniASRRuntime.qwenFinalRequest(
            samples: [0, 0.1],
            sampleRate: 16000,
            language: "English",
            context: "Voxt",
            maxNewTokens: 64
        )
        let body = OmniMultipartBody.transcription(request, modelName: "m", boundary: "b")

        XCTAssertEqual(fieldValue("audio_layout", in: body), "voxt_swift")
        XCTAssertEqual(fieldValue("language", in: body), "English")
        XCTAssertEqual(fieldValue("prompt", in: body), "Voxt")
        XCTAssertEqual(fieldValue("max_new_tokens", in: body), "64")
        XCTAssertEqual(fieldValue("stop_at_end_of_text", in: body), "true")
        XCTAssertEqual(fieldValue("stop_on_token_loop", in: body), "true")
        XCTAssertEqual(fieldValue("include_generation_metadata", in: body), "true")
    }

    func testWhisperFinalRequestsCarryTheBudgetLanguageAndTemperature() {
        let request = OmniASRRuntime.whisperFinalRequest(
            samples: [0, 0.1],
            sampleRate: 16000,
            language: "en",
            maxNewTokens: 320,
            temperature: 0
        )
        let body = OmniMultipartBody.transcription(request, modelName: "m", boundary: "b")

        XCTAssertEqual(fieldValue("language", in: body), "en")
        XCTAssertEqual(fieldValue("max_new_tokens", in: body), "320")
        XCTAssertEqual(fieldValue("temperature", in: body), "0.0")
        XCTAssertNil(fieldValue("audio_layout", in: body))
        XCTAssertNil(fieldValue("stop_at_end_of_text", in: body))
        XCTAssertNil(fieldValue("include_generation_metadata", in: body))
    }

    func testCohereFinalRequestsCarryVoxtsLongFormSettings() {
        let request = OmniASRRuntime.cohereFinalRequest(
            samples: [0, 0.1],
            sampleRate: 16000,
            language: "zh",
            usePunctuation: false,
            maxNewTokens: 1024,
            temperature: 0,
            chunkDuration: 1200,
            minChunkDuration: 1,
            speechSegments: OmniSpeechSegments(
                vadModelDirectory: URL(fileURLWithPath: "/models/silero"),
                threshold: 0.5,
                minSpeechMilliseconds: 220,
                minSilenceMilliseconds: 420,
                speechPadMilliseconds: 180,
                mergeGapSeconds: 1,
                maxChunkSeconds: 24
            )
        )
        let body = OmniMultipartBody.transcription(request, modelName: "m", boundary: "b")

        XCTAssertEqual(fieldValue("language", in: body), "zh")
        XCTAssertEqual(fieldValue("use_punctuation", in: body), "false")
        XCTAssertEqual(fieldValue("max_new_tokens", in: body), "1024")
        XCTAssertEqual(fieldValue("chunk_duration", in: body), "1200.0")
        XCTAssertEqual(fieldValue("min_chunk_duration", in: body), "1.0")
        XCTAssertEqual(fieldValue("vad_model_directory", in: body), "/models/silero")
        XCTAssertEqual(fieldValue("vad_threshold", in: body), "0.5")
        XCTAssertEqual(fieldValue("vad_min_speech_ms", in: body), "220")
        XCTAssertEqual(fieldValue("vad_min_silence_ms", in: body), "420")
        XCTAssertEqual(fieldValue("vad_speech_pad_ms", in: body), "180")
        XCTAssertEqual(fieldValue("vad_merge_gap_seconds", in: body), "1.0")
        XCTAssertEqual(fieldValue("vad_max_chunk_seconds", in: body), "24.0")
    }

    func testParakeetFinalRequestsCarryOnlyVoxtsChunkLength() {
        let request = OmniASRRuntime.parakeetFinalRequest(samples: [0, 0.1], sampleRate: 16000, chunkDuration: 1200)
        let body = OmniMultipartBody.transcription(request, modelName: "m", boundary: "b")

        XCTAssertEqual(fieldValue("chunk_duration", in: body), "1200.0")
        XCTAssertNil(fieldValue("language", in: body))
        XCTAssertNil(fieldValue("max_new_tokens", in: body))
    }

    func testRequestsWithoutALayoutLeaveTheServerDefault() {
        let request = OmniTranscriptionRequest(
            samples: [0],
            sampleRate: 16000,
            language: nil,
            prompt: nil,
            maxNewTokens: nil,
            stopAtEndOfText: false,
            stopOnTokenLoop: false
        )
        let body = OmniMultipartBody.transcription(request, modelName: "m", boundary: "b")

        XCTAssertNil(fieldValue("audio_layout", in: body))
    }

    func testMossIsRoutedToTheOmniServer() {
        XCTAssertEqual(
            OmniASRBackend.modelKindsByRepo["OpenMOSS-Team/MOSS-Transcribe-Diarize"],
            .mossTranscribeDiarize
        )
        XCTAssertEqual(OmniASRModelKind.mossTranscribeDiarize.rawValue, "moss_transcribe_diarize")
    }

    func testDoneEventsCarrySpeakerSegments() throws {
        var parser = OmniTranscriptionStreamParser()
        try parser.consume(line: #"data: {"type":"transcript.text.done","text":"[0.07][S01] Hi[1.20]","segments":[{"start":0.07,"end":1.2,"speaker":"S01","text":"Hi"},{"start":"bad"}]}"#)
        try parser.consume(line: "data: [DONE]")

        let result = try parser.finish()
        XCTAssertEqual(result.text, "[0.07][S01] Hi[1.20]")
        XCTAssertEqual(result.segments, [
            OmniSpeakerSegment(startSeconds: 0.07, endSeconds: 1.2, speakerID: "S01", text: "Hi"),
        ])
    }

    func testDoneEventsWithoutSegmentsHaveNone() throws {
        var parser = OmniTranscriptionStreamParser()
        try parser.consume(line: #"data: {"type":"transcript.text.done","text":"hello"}"#)

        XCTAssertEqual(try parser.finish().segments, [])
    }

    func testUnlabelledSegmentsHaveNoSpeakerForVoxt() {
        let segment = OmniSpeakerSegment(startSeconds: 0, endSeconds: 2, speakerID: "", text: "hi")

        XCTAssertNil(segment.transcriptSegment.speakerID)
        XCTAssertEqual(segment.transcriptSegment.endTime, 2)
    }
}

final class OmniRealtimeSessionUpdateTests: XCTestCase {
    private func session(_ message: String) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any]
        XCTAssertEqual(object?["type"] as? String, "session.update")
        return try XCTUnwrap(object?["session"] as? [String: Any])
    }

    /// Voxt's Swift live session streams continuously without voice detection,
    /// so the server must not wait for a VAD onset before its first decode.
    func testLiveSessionsTurnServerVoiceDetectionOff() throws {
        let session = try session(OmniRealtimeTranscriptionSession.sessionUpdate(language: nil))

        XCTAssertTrue(session.keys.contains("turn_detection"))
        XCTAssertTrue(session["turn_detection"] is NSNull)
        XCTAssertEqual(session["input_audio_format"] as? String, "pcm16")
        XCTAssertNil(session["language"])
    }

    func testLiveSessionsPassTheLanguageHint() throws {
        let session = try session(OmniRealtimeTranscriptionSession.sessionUpdate(language: "English"))

        XCTAssertEqual(session["language"] as? String, "English")
    }

    /// MLXAudio's MOSS session kept each finalized window on its own line and
    /// the pending window on the next; Voxt's MOSS rendering merges the lines.
    func testMossLiveWindowsJoinByLine() {
        var assembler = OmniLiveTranscriptAssembler(joining: .lines)

        XCTAssertEqual(
            assembler.apply(eventIndex: 1, segmentID: 0, text: "Hello", isFinal: true),
            .display(confirmedText: "Hello", provisionalText: "")
        )
        XCTAssertEqual(
            assembler.apply(eventIndex: 2, segmentID: 1, text: ", wor", isFinal: false),
            .display(confirmedText: "Hello\n", provisionalText: ", wor")
        )
        XCTAssertEqual(
            assembler.apply(eventIndex: 3, segmentID: 1, text: " , world ", isFinal: true),
            .display(confirmedText: "Hello\n, world", provisionalText: "")
        )
    }

    func testQwenLiveSegmentsKeepScriptAwareSpacing() {
        var assembler = OmniLiveTranscriptAssembler()

        _ = assembler.apply(eventIndex: 1, segmentID: 0, text: "Hello", isFinal: true)
        XCTAssertEqual(
            assembler.apply(eventIndex: 2, segmentID: 1, text: "world", isFinal: true),
            .display(confirmedText: "Hello world", provisionalText: "")
        )
    }

    func testMossLiveSessionsPassTheTaskPrompt() throws {
        let session = try session(OmniRealtimeTranscriptionSession.sessionUpdate(language: nil, prompt: "Transcribe."))

        XCTAssertEqual(session["prompt"] as? String, "Transcribe.")
        XCTAssertNil(session["language"])
    }
}
