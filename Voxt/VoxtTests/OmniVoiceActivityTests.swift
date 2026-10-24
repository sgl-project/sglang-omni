// OmniVoiceActivityTests.swift
// Native Silero VAD wire formats, stream ownership, and cancellation.

import Network
import XCTest
@testable import Voxt

final class OmniVoiceActivityTests: XCTestCase {
    func testDiscardingAStreamClosesItsConnection() async throws {
        let server = try VADTestWebSocketServer(reply: #"{"probability":0.875}"#)
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        var stream: OmniVoiceActivityStream? = OmniVoiceActivityStream(endpoint: server.endpoint)
        _ = try await stream?.probability(samples16k: [0.1])
        stream = nil
        let deadline = ContinuousClock.now + .seconds(1)
        while server.activeConnectionCount != 0, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(server.activeConnectionCount, 0, "A discarded VAD stream retained its connection")
    }

    func testHealthyExchangesKeepTheStreamOpenAndOrdered() async throws {
        let server = try VADTestWebSocketServer(reply: #"{"probability":0.875}"#)
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let stream = OmniVoiceActivityStream(endpoint: server.endpoint)
        for _ in 0..<32 {
            let probability = try await stream.probability(samples16k: [Float](repeating: 0.1, count: 512))
            XCTAssertEqual(probability, 0.875)
        }
        XCTAssertEqual(server.connectionCount, 1)
        XCTAssertEqual(server.messageCount, 32)
        await stream.close()
    }

    @MainActor
    func testCancellingAQueuedRequestAlsoReleasesTheBlockedExchange() async throws {
        let received = expectation(description: "first VAD request received")
        let server = try VADTestWebSocketServer { received.fulfill() }
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let stream = OmniVoiceActivityStream(endpoint: server.endpoint, responseTimeout: .seconds(10))
        let first = Task { try await stream.probability(samples16k: [0.1]) }
        await fulfillment(of: [received], timeout: 5)
        let queued = Task { try await stream.probability(samples16k: [0.2]) }
        try await Task.sleep(for: .milliseconds(20))
        let finished = expectation(description: "both VAD callers released")
        let observer = Task {
            _ = await first.result
            _ = await queued.result
            finished.fulfill()
        }
        queued.cancel()
        let completion = await XCTWaiter.fulfillment(of: [finished], timeout: 0.5)
        await stream.close()
        await observer.value
        XCTAssertEqual(completion, .completed)
        if case .failure(let error) = await queued.result {
            XCTAssertTrue(error is CancellationError)
        } else {
            XCTFail("Cancelled queued request returned a probability")
        }
        XCTAssertEqual(server.messageCount, 1)
    }

    @MainActor
    func testCancellingAnUnansweredVADRequestReleasesTheCaller() async throws {
        let received = expectation(description: "VAD received audio")
        let server = try VADTestWebSocketServer { received.fulfill() }
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let stream = OmniVoiceActivityStream(endpoint: server.endpoint, responseTimeout: .seconds(10))
        let probability = Task { try await stream.probability(samples16k: [Float](repeating: 0.1, count: 512)) }
        await fulfillment(of: [received], timeout: 5)
        let finished = expectation(description: "cancelled VAD request finished")
        let observer = Task {
            _ = await probability.result
            finished.fulfill()
        }
        probability.cancel()
        let completion = await XCTWaiter.fulfillment(of: [finished], timeout: 0.5)
        await stream.close()
        await observer.value
        XCTAssertEqual(completion, .completed)
        if case .failure(let error) = await probability.result {
            XCTAssertTrue(error is CancellationError)
        } else {
            XCTFail("Cancelled VAD request returned a probability")
        }
    }

    @MainActor
    func testAnUnansweredVADRequestFailsAtItsDeadline() async throws {
        let received = expectation(description: "VAD received audio")
        let server = try VADTestWebSocketServer { received.fulfill() }
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let stream = OmniVoiceActivityStream(endpoint: server.endpoint, responseTimeout: .milliseconds(80))
        let probability = Task { try await stream.probability(samples16k: [Float](repeating: 0.1, count: 512)) }
        await fulfillment(of: [received], timeout: 5)
        let finished = expectation(description: "VAD deadline released the caller")
        let observer = Task {
            _ = await probability.result
            finished.fulfill()
        }
        let completion = await XCTWaiter.fulfillment(of: [finished], timeout: 0.5)
        await stream.close()
        await observer.value
        XCTAssertEqual(completion, .completed)
        if case .failure(let error) = await probability.result {
            XCTAssertEqual((error as? URLError)?.code, .timedOut)
        } else {
            XCTFail("Unanswered VAD request returned a probability")
        }
    }

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

final class VADTestWebSocketServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "voxt-vad-silent-server")
    private var connections: [NWConnection] = []
    private var reply: String?
    private var messages = 0
    private var closedConnections: Set<ObjectIdentifier> = []
    let ready = DispatchSemaphore(value: 0)

    init(reply: String? = nil, received: @escaping @Sendable () -> Void = {}) throws {
        self.reply = reply
        let parameters = NWParameters.tcp
        parameters.defaultProtocolStack.applicationProtocols.insert(NWProtocolWebSocket.Options(), at: 0)
        listener = try NWListener(using: parameters, on: .any)
        listener.stateUpdateHandler = { [ready] state in
            if case .ready = state { ready.signal() }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            self.connections.append(connection)
            connection.start(queue: self.queue)
            self.receive(connection, received: received)
        }
        listener.start(queue: queue)
    }

    var endpoint: OmniServerEndpoint {
        OmniServerEndpoint(host: "127.0.0.1", port: Int(listener.port!.rawValue), modelName: "silero", serverProcessIdentifier: 0)
    }

    var connectionCount: Int { queue.sync { connections.count } }
    var messageCount: Int { queue.sync { messages } }
    var activeConnectionCount: Int { queue.sync { connections.count - closedConnections.count } }

    func respond(with reply: String) {
        queue.sync { self.reply = reply }
    }

    private func receive(_ connection: NWConnection, received: @escaping @Sendable () -> Void) {
        connection.receiveMessage { [weak self] data, context, _, error in
            guard let self else { return }
            let receivedMetadata = context?.protocolMetadata(definition: NWProtocolWebSocket.definition) as? NWProtocolWebSocket.Metadata
            guard data != nil, error == nil, receivedMetadata?.opcode != .close else {
                self.closedConnections.insert(ObjectIdentifier(connection))
                connection.cancel()
                return
            }
            self.messages += 1
            received()
            guard let reply = self.reply else { return }
            let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
            let context = NWConnection.ContentContext(identifier: "vad-probability", metadata: [metadata])
            connection.send(
                content: Data(reply.utf8),
                contentContext: context,
                isComplete: true,
                completion: .contentProcessed { [weak self] error in
                    if error == nil { self?.receive(connection, received: received) }
                }
            )
        }
    }

    func stop() {
        queue.sync { connections.forEach { $0.cancel() } }
        listener.cancel()
    }
}
