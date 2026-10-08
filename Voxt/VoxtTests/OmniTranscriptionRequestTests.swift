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
}
