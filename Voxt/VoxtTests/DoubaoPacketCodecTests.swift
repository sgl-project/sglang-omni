import XCTest
@testable import Voxt

@MainActor
final class DoubaoPacketCodecTests: XCTestCase {
    func testClientFinalFramesPreserveMeetingAndDictationSequenceLayouts() {
        let meeting = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeAudioOnlyClientRequest,
            messageFlags: DoubaoProtocol.flagLastAudioPacket,
            serialization: DoubaoProtocol.serializationNone,
            compression: DoubaoProtocol.compressionNone,
            sequence: 0,
            payload: Data()
        )
        let dictation = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeAudioOnlyClientRequest,
            messageFlags: DoubaoProtocol.flagNegativeAudioPacket,
            serialization: DoubaoProtocol.serializationNone,
            compression: DoubaoProtocol.compressionNone,
            sequence: -2,
            payload: Data()
        )
        XCTAssertEqual(Array(meeting), [0x11, 0x22, 0, 0, 0, 0, 0, 0])
        XCTAssertEqual(Array(dictation), [0x11, 0x23, 0, 0, 255, 255, 255, 254, 0, 0, 0, 0])
    }

    func testServerFinalWithoutSequencePreservesPayloadLength() throws {
        let packet = DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeFullServerResponse,
            messageFlags: DoubaoProtocol.flagLastAudioPacket,
            serialization: DoubaoProtocol.serializationJSON,
            compression: DoubaoProtocol.compressionNone,
            sequence: 0,
            payload: Data("final text".utf8)
        )
        let response = try XCTUnwrap(try DoubaoPacketCodec.decodeServerPacket(packet))
        XCTAssertEqual(response.payload, Data("final text".utf8))
        XCTAssertNil(response.sequence)
        XCTAssertTrue(response.isFinal)
        XCTAssertTrue(try XCTUnwrap(try RemoteASRTranscriber.parseDoubaoServerPacket(packet)).isFinal)
        XCTAssertTrue(try XCTUnwrap(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet)).isFinal)
    }

    func testDataSliceWithNonzeroIndicesDecodesLikeSocketData() throws {
        let packet = responsePacket(Data("hello".utf8))
        var prefixed = Data([1, 2, 3])
        prefixed.append(packet)
        let response = try XCTUnwrap(try DoubaoPacketCodec.decodeServerPacket(prefixed.dropFirst(3)))
        XCTAssertEqual(response.payload, Data("hello".utf8))
    }

    func testGzipResponseRoundTripsWithNegativeSequence() throws {
        let payload = Data(#"{"result":{"text":"你好"}}"#.utf8)
        let encoded = try DoubaoPacketCodec.encodePayload(payload)
        let response = try XCTUnwrap(try DoubaoPacketCodec.decodeServerPacket(
            responsePacket(encoded.payload, compression: encoded.compression, sequence: -3)
        ))
        XCTAssertEqual(response.payload, payload)
        XCTAssertEqual(response.sequence, -3)
        XCTAssertTrue(response.isFinal)
    }

    func testTruncatedHeaderAndPayloadAreIgnoredWithoutReadingPastEnd() throws {
        let packet = responsePacket(Data(#"{"text":"hello"}"#.utf8))
        for length in 0..<packet.count {
            XCTAssertNil(try DoubaoPacketCodec.decodeServerPacket(Data(packet.prefix(length))), "length=\(length)")
        }
    }

    func testEventHeaderAndEmptyFinalAcknowledgement() throws {
        let packet = Data([0x11, 0xB7, 0, 0, 255, 255, 255, 254, 0, 0, 0, 7])
        let response = try XCTUnwrap(try DoubaoPacketCodec.decodeServerPacket(packet))
        XCTAssertTrue(response.isFinal)
        XCTAssertEqual(response.sequence, -2)
        XCTAssertTrue(response.payload.isEmpty)
    }

    func testInvalidGzipDoesNotFallBackToVisiblePlainText() {
        let packet = responsePacket(Data("not a gzip stream".utf8), compression: DoubaoProtocol.compressionGzip)
        XCTAssertThrowsError(try DoubaoPacketCodec.decodeServerPacket(packet))
        XCTAssertThrowsError(try RemoteASRTranscriber.parseDoubaoServerPacket(packet))
        XCTAssertThrowsError(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet))
    }

    func testCompressionLimitsApplyToDictationAndMeeting() throws {
        let oversized = Data(repeating: 0, count: DoubaoPacketCodec.maxCompressedPayloadBytes + 1)
        let expansion = try DoubaoPacketCodec.encodePayload(Data(repeating: 65, count: 1_048_577))
        for payload in [oversized, expansion.payload] {
            let packet = responsePacket(payload, compression: DoubaoProtocol.compressionGzip)
            XCTAssertThrowsError(try RemoteASRTranscriber.parseDoubaoServerPacket(packet))
            XCTAssertThrowsError(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet))
        }
    }

    func testUnknownCompressionIsRejected() {
        XCTAssertThrowsError(try DoubaoPacketCodec.decodeServerPacket(responsePacket(Data("hello".utf8), compression: 0xF)))
    }

    func testServerErrorIsNotReturnedAsTranscript() {
        // Error frames carry a provider error code before their payload length.
        var packet = Data([0x11, 0xF0, 0x10, 0, 0, 0, 1, 145])
        let payload = Data("unauthorized".utf8)
        packet.append(contentsOf: [0, 0, 0, UInt8(payload.count)])
        packet.append(payload)
        XCTAssertThrowsError(try DoubaoPacketCodec.decodeServerPacket(packet)) { error in
            XCTAssertEqual(error.localizedDescription, "unauthorized")
        }
    }

    func testNegativeWireSequenceCannotBeOverriddenByJSONMetadata() throws {
        let packet = responsePacket(Data(#"{"sequence":1,"text":"final answer"}"#.utf8), sequence: -2)
        XCTAssertTrue(try XCTUnwrap(try RemoteASRTranscriber.parseDoubaoServerPacket(packet)).isFinal)
        XCTAssertTrue(try XCTUnwrap(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet)).isFinal)
    }

    func testOnlyNamedSequenceFieldsAffectFinality() throws {
        let packet = responsePacket(Data(#"{"result":{"duration":-99,"text":"hello"}}"#.utf8))
        XCTAssertFalse(try XCTUnwrap(try RemoteASRTranscriber.parseDoubaoServerPacket(packet)).isFinal)
        XCTAssertFalse(try XCTUnwrap(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet)).isFinal)
        XCTAssertNil(DoubaoPacketCodec.sequence(in: ["sequence": Int64.max]))
        XCTAssertEqual(DoubaoPacketCodec.sequence(in: ["result": ["auto_assigned_sequence": -2]]), -2)
    }

    func testWorkflowSpecificTranscriptProjectionIsPreserved() throws {
        let payload = Data(#"{"result":{"text":"whole transcript","utterances":[{"text":"latest sentence","start_time":1000,"end_time":2000,"definite":true}]}}"#.utf8)
        let packet = responsePacket(payload)
        let dictation = try XCTUnwrap(try RemoteASRTranscriber.parseDoubaoServerPacket(packet))
        let meeting = try XCTUnwrap(try MeetingRemoteAudioSupport.parseDoubaoServerPacket(packet))
        XCTAssertEqual(dictation.text, "whole transcript")
        XCTAssertEqual(meeting.units.first?.text, "latest sentence")
        XCTAssertEqual(meeting.units.first?.startSeconds, 1)
        XCTAssertNil(meeting.fallbackText)
    }

    private func responsePacket(_ payload: Data, compression: UInt8 = 0, sequence: Int32 = 1) -> Data {
        DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeFullServerResponse,
            messageFlags: DoubaoProtocol.flagPositiveSequence,
            serialization: DoubaoProtocol.serializationJSON,
            compression: compression,
            sequence: sequence,
            payload: payload
        )
    }
}
