// OmniVoiceActivityTests.swift
// Covers the wire formats of Silero VAD on the native runtime.

import XCTest
@testable import Voxt

final class OmniVoiceActivityTests: XCTestCase {
    func testStreamSendsFloat32LittleEndianSamples() {
        let data = OmniVoiceActivityStream.float32LittleEndian([0, 1, -0.5])

        XCTAssertEqual([UInt8](data), [
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x80, 0x3F,
            0x00, 0x00, 0x00, 0xBF,
        ])
    }

    func testStreamReplyCarriesTheLastChunkProbabilityOrNull() throws {
        XCTAssertEqual(try OmniVoiceActivityStream.probability(fromReply: #"{"probability":0.875}"#), 0.875)
        XCTAssertNil(try OmniVoiceActivityStream.probability(fromReply: #"{"probability":null}"#))
        XCTAssertThrowsError(try OmniVoiceActivityStream.probability(fromReply: #"{"error":"x"}"#))
        XCTAssertThrowsError(try OmniVoiceActivityStream.probability(fromReply: #"{"probability":"high"}"#))
    }

    func testSpeechTimestampsAreSampleRanges() throws {
        let response = Data(#"{"sample_rate":16000,"timestamps":[{"start":0,"end":512},{"start":1024,"end":4096}]}"#.utf8)

        XCTAssertEqual(try OmniVoiceActivityRequests.ranges(fromResponse: response), [0 ..< 512, 1024 ..< 4096])
        XCTAssertThrowsError(try OmniVoiceActivityRequests.ranges(fromResponse: Data(#"{"timestamps":[{"start":9,"end":3}]}"#.utf8)))
        XCTAssertThrowsError(try OmniVoiceActivityRequests.ranges(fromResponse: Data(#"{"detail":"x"}"#.utf8)))
    }

    /// Detectors that find the shared server dead at the same time all get
    /// the same replacement.
    func testConcurrentRecoveryStartsOneReplacement() async throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-vad-recovery-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        // Note (Jiaxin Deng): reports ready with its own pid, then exits, so every server "dies" at once.
        let server = scratch.appendingPathComponent("qwen3_asr_server")
        let script = #"""
        #!/bin/sh
        echo "{\"event\":\"ready\",\"host\":\"127.0.0.1\",\"port\":1,\"model_name\":\"m\",\"server_pid\":$$}"
        sleep 0.2
        """#
        try script.write(to: server, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: server.path)
        let configuration = OmniBackendConfiguration(runtimeExecutable: server)
        let shared = OmniSharedModelRuntime(kind: .sileroVAD, configuration: { configuration })

        _ = try await shared.acquire(modelDirectory: scratch)
        for _ in 0 ..< 10 {
            try await Task.sleep(for: .milliseconds(600))
            async let first = shared.acquire(modelDirectory: scratch)
            async let second = shared.acquire(modelDirectory: scratch)
            async let third = shared.acquire(modelDirectory: scratch)
            let endpoints = try await [first, second, third]
            let pids = Set(endpoints.map(\.serverProcessIdentifier))
            XCTAssertEqual(pids.count, 1, "concurrent recoveries started \(pids.count) servers")
        }
        for _ in 0 ..< 31 {
            await shared.release()
        }
    }
}
