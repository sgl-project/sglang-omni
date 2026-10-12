// OmniSpeakerDiarizationTests.swift
// Covers the wire format of Sortformer speaker diarization on the native runtime.

import XCTest
@testable import Voxt

final class OmniSpeakerDiarizationTests: XCTestCase {
    private let endpoint = OmniServerEndpoint(host: "127.0.0.1", port: 8123, modelName: "m", serverProcessIdentifier: 1)

    func testStreamURLCarriesVoxtsFeedOptions() throws {
        let url = OmniDiarizationStream.url(endpoint: endpoint, options: .init())
        let components = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))

        XCTAssertEqual(components.scheme, "ws")
        XCTAssertEqual(components.port, 8123)
        XCTAssertEqual(components.path, "/v1/diarization/stream")
        XCTAssertEqual(
            Dictionary(uniqueKeysWithValues: (components.queryItems ?? []).map { ($0.name, $0.value ?? "") }),
            ["threshold": "0.5", "min_duration": "0.0", "merge_gap": "0.18", "spkcache_max": "188", "fifo_max": "188"]
        )
    }

    func testReplyCarriesProbabilitiesSegmentsAndState() throws {
        let reply = #"""
        {"frames":2,"speakers":4,"probabilities":[[0.875,0.0,0.25,1.0],[0.5600000023841858,0.0,0.0,0.0]],
         "segments":[{"start":0.0,"end":0.07999999821186066,"speaker":0}],
         "state":{"fifo_length":63,"spkcache_length":0,"frames_processed":65}}
        """#

        let feed = try OmniDiarizationStream.feed(fromReply: reply)

        XCTAssertEqual(feed.probabilities, [[0.875, 0, 0.25, 1], [0.56, 0, 0, 0]])
        XCTAssertEqual(feed.segments, [.init(start: 0, end: 0.08, speaker: 0)])
        XCTAssertEqual(feed.fifoLength, 63)
        XCTAssertEqual(feed.spkcacheLength, 0)
        XCTAssertEqual(feed.framesProcessed, 65)
    }

    func testMalformedOrRefusedRepliesThrow() {
        XCTAssertThrowsError(try OmniDiarizationStream.feed(fromReply: #"{"error":"A feed is 2 to 480000 samples."}"#)) {
            XCTAssertEqual($0 as? OmniSpeakerDiarizationError, .server("A feed is 2 to 480000 samples."))
        }
        let state = #""state":{"fifo_length":1,"spkcache_length":0,"frames_processed":1}"#
        for reply in [
            #"{"frames":2,"speakers":1,"probabilities":[[0.5]],"segments":[],\#(state)}"#,
            #"{"frames":1,"speakers":1,"probabilities":[[0.5]],"segments":[{"start":1,"end":0,"speaker":0}],\#(state)}"#,
            #"{"frames":1,"speakers":1,"probabilities":[[0.5]],"segments":[{"start":0,"end":1,"speaker":3}],\#(state)}"#,
            #"{"frames":1,"speakers":1,"probabilities":[[0.5]],"segments":[]}"#,
        ] {
            XCTAssertThrowsError(try OmniDiarizationStream.feed(fromReply: reply)) {
                XCTAssertEqual($0 as? OmniSpeakerDiarizationError, .malformedResponse)
            }
        }
    }
}
