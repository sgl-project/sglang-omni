// OmniFailurePathTests.swift
// Failure paths of the Omni client that need no model: live socket errors,
// orphaned runtimes, retirement racing a launch, and what errors may say.

import Network
import XCTest
@testable import Voxt

/// A loopback WebSocket server that answers each connection with one scripted
/// realtime event once that connection has sent enough messages, then keeps it open.
private final class LiveEventServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "omni-live-event-server")
    private var connections: [NWConnection] = []
    let ready = DispatchSemaphore(value: 0)

    /// The last event and message count repeat for every connection after the scripted ones.
    init(events: [String], messagesBeforeReply: [Int] = [1]) throws {
        precondition(!events.isEmpty && !messagesBeforeReply.isEmpty)
        let parameters = NWParameters.tcp
        let webSocketOptions = NWProtocolWebSocket.Options()
        webSocketOptions.maximumMessageSize = 16 * 1024 * 1024
        parameters.defaultProtocolStack.applicationProtocols.insert(webSocketOptions, at: 0)
        listener = try NWListener(using: parameters, on: .any)
        listener.stateUpdateHandler = { [ready] state in
            if case .ready = state { ready.signal() }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            self.connections.append(connection)
            let index = self.connections.count - 1
            let event = events[min(index, events.count - 1)]
            let replyAfter = messagesBeforeReply[min(index, messagesBeforeReply.count - 1)]
            connection.start(queue: self.queue)
            Self.receive(on: connection, received: 0, replyAfter: replyAfter, event: event)
        }
        listener.start(queue: queue)
    }

    private static func receive(on connection: NWConnection, received: Int, replyAfter: Int, event: String) {
        connection.receiveMessage { _, _, _, error in
            guard error == nil else { return }
            let count = received + 1
            if count == replyAfter {
                let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
                let context = NWConnection.ContentContext(identifier: "event", metadata: [metadata])
                connection.send(
                    content: Data(event.utf8),
                    contentContext: context,
                    isComplete: true,
                    completion: .contentProcessed { _ in }
                )
            }
            receive(on: connection, received: count, replyAfter: replyAfter, event: event)
        }
    }

    var port: Int { Int(listener.port?.rawValue ?? 0) }
    var connectionCount: Int { queue.sync { connections.count } }

    func stop() {
        connections.forEach { $0.cancel() }
        listener.cancel()
    }
}

private actor StartupAttemptCounter {
    private(set) var value = 0

    func increment() {
        value += 1
    }
}

final class OmniFailurePathTests: XCTestCase {
    /// A failed live ASR setup used to only log: the recording showed a waveform
    /// and never any text, with no retry and nothing shown to the user.
    @MainActor
    func testLiveSetupFailureIsRetriedOnceAndThenReported() async {
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            throw OmniASRRuntimeError.launchFailed("injected startup failure")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        var failures: [String] = []
        transcriber.onRuntimeFailure = { failures.append($0) }
        transcriber.isRecording = true
        transcriber.isModelInitializing = true

        await transcriber.startNativeLiveSession(revision: 0).value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 2)
        XCTAssertNotNil(transcriber.consumePendingRuntimeFailureMessage())
        XCTAssertEqual(failures.count, 1)
        XCTAssertFalse(transcriber.isModelInitializing)
        XCTAssertFalse(manager.hasActiveUse)
        transcriber.isRecording = false
        await manager.shutdownForApplicationTermination()
    }

    @MainActor
    func testTransientLoadFailureRecoversLivePreviewWithoutRestarting() async throws {
        try await verifyLiveRecovery(failFirstLoad: true, recoverySucceeds: true)
    }

    @MainActor
    func testTransientConnectionFailureRecoversLivePreviewWithoutRestarting() async throws {
        try await verifyLiveRecovery(failFirstLoad: false, recoverySucceeds: true)
    }

    @MainActor
    func testRepeatedConnectionFailureIsReportedAndStopsRetrying() async throws {
        try await verifyLiveRecovery(failFirstLoad: false, recoverySucceeds: false)
    }

    /// Load and connection failures share one retry per recording.
    @MainActor
    func testConnectionFailureAfterLoadRetryDoesNotRetryAgain() async throws {
        try await verifyLiveRecovery(failFirstLoad: true, recoverySucceeds: false)
    }

    /// A retry late in a long recording must replay the whole recording, in batches the live
    /// queue accepts: one 40 s batch overflows it, and replaying only the tail truncates the preview.
    @MainActor
    func testConnectionFailureLateInALongRecordingRecoversLivePreview() async throws {
        try await verifyLiveRecovery(failFirstLoad: false, recoverySucceeds: true, bufferedSpeechSeconds: 40)
    }

    /// With buffered speech, the recovered connection answers only after it has received all of it.
    @MainActor
    private func verifyLiveRecovery(
        failFirstLoad: Bool,
        recoverySucceeds: Bool,
        bufferedSpeechSeconds: Double = 0
    ) async throws {
        let previewEvent = """
        {"type":"transcription.segment","segment_id":0,"text":"recovered preview","is_final":false,"event_index":1}
        """
        let failureEvent = """
        {"type":"error","error":{"code":"injected_connection_failure"}}
        """
        let lastEvent = recoverySucceeds ? previewEvent : failureEvent
        let server = try LiveEventServer(
            events: failFirstLoad ? [lastEvent] : [failureEvent, lastEvent],
            messagesBeforeReply: [1, 1 + Int((bufferedSpeechSeconds / MLXTranscriber.nativeLiveFeedBatchSeconds).rounded(.up))]
        )
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-live-recovery-\(UUID().uuidString)")
        let modelDirectory = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let executable = root.appendingPathComponent("server")
        try """
        #!/bin/sh
        printf '{"event":"ready","host":"127.0.0.1","port":\(server.port),"model_name":"m","server_pid":%s}\\n' "$$"
        read -r command
        """.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let runtime = OmniASRRuntime(
            kind: .qwen3ASR,
            modelDirectory: modelDirectory,
            configuration: OmniBackendConfiguration(runtimeExecutable: executable)
        )
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            if failFirstLoad, await attempts.value == 1 {
                throw OmniASRRuntimeError.launchFailed("injected transient failure")
            } else {
                _ = try await runtime.prepare()
                return MLXLoadedModelBox(loaded: .omni(runtime))
            }
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        let result = expectation(description: "live startup settled")
        var failures: [String] = []
        transcriber.onPartialTranscription = { _ in result.fulfill() }
        transcriber.onRuntimeFailure = { message in
            failures.append(message)
            result.fulfill()
        }
        if bufferedSpeechSeconds > 0 {
            let sampleRate = 16_000.0
            transcriber.configureVoiceActivityFinalizationFiltering(enabled: true)
            transcriber.appendVoiceActivityFinalizationFrame(
                ASRVoiceActivityAudioFrame(
                    samples: [Float](repeating: 0.1, count: Int(bufferedSpeechSeconds * sampleRate)),
                    sampleRate: sampleRate,
                    startSeconds: 0,
                    endSeconds: bufferedSpeechSeconds,
                    level: 0.5
                ),
                isSpeech: true
            )
        }
        transcriber.isRecording = true

        await transcriber.startNativeLiveSession(revision: 0).value
        await fulfillment(of: [result], timeout: 10)

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, failFirstLoad ? 2 : 1)
        XCTAssertEqual(server.connectionCount, failFirstLoad ? 1 : 2)
        if recoverySucceeds {
            XCTAssertEqual(transcriber.transcribedText, "recovered preview")
            XCTAssertTrue(failures.isEmpty)
            XCTAssertNil(transcriber.consumePendingRuntimeFailureMessage())
        } else {
            XCTAssertTrue(transcriber.transcribedText.isEmpty)
            XCTAssertEqual(failures.count, 1)
            XCTAssertNotNil(transcriber.consumePendingRuntimeFailureMessage())
        }
        transcriber.isRecording = false
        transcriber.releaseNativeLiveSession(cancelSession: true)
        await manager.shutdownForApplicationTermination()
        await runtime.retire()
    }

    @MainActor
    func testStoppingDuringLiveSetupNeitherRetriesNorReportsAStaleFailure() async {
        let barrier = ManualTaskBarrier()
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            await barrier.wait()
            throw OmniASRRuntimeError.launchFailed("injected stale failure")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        transcriber.onRuntimeFailure = { _ in XCTFail("Stopped setup reported a stale failure") }
        transcriber.isRecording = true
        let setup = transcriber.startNativeLiveSession(revision: 0)
        await barrier.waitUntilEntered()
        transcriber.isRecording = false
        barrier.release()
        await setup.value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 1)
        XCTAssertNil(transcriber.consumePendingRuntimeFailureMessage())
        XCTAssertFalse(manager.hasActiveUse)
        await manager.shutdownForApplicationTermination()
    }

    /// After Stop, capture keeps running while voice activity is flushed; a failure in that
    /// window used to start another live session and could report a failure.
    @MainActor
    func testFailureAfterStopIsPressedNeitherRetriesNorReports() async {
        let barrier = ManualTaskBarrier()
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            await barrier.wait()
            throw OmniASRRuntimeError.launchFailed("injected failure after stop")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        transcriber.onRuntimeFailure = { _ in XCTFail("A failure after Stop was reported") }
        transcriber.isRecording = true
        let setup = transcriber.startNativeLiveSession(revision: 0)
        await barrier.waitUntilEntered()
        transcriber.noteStopRequested()
        barrier.release()
        await setup.value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 1)
        XCTAssertNil(transcriber.consumePendingRuntimeFailureMessage())
        XCTAssertFalse(manager.hasActiveUse)
        transcriber.isRecording = false
        await manager.shutdownForApplicationTermination()
    }

    /// A retry queued by a connection failure, or one whose delay ended, starts a live session
    /// again; after Stop it used to start a second model load while capture was still finishing.
    @MainActor
    func testLiveSessionStartQueuedAfterStopDoesNothing() async {
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            throw OmniASRRuntimeError.launchFailed("a retry ran after stop")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        transcriber.onRuntimeFailure = { _ in XCTFail("A failure after Stop was reported") }
        transcriber.isRecording = true
        transcriber.isModelInitializing = true
        transcriber.noteStopRequested()

        await transcriber.startNativeLiveSession(revision: 0).value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 0)
        XCTAssertFalse(transcriber.isModelInitializing)
        XCTAssertFalse(manager.hasActiveUse)
        transcriber.isRecording = false
        await manager.shutdownForApplicationTermination()
    }

    /// A load that finished after Stop used to open a live session during the stop-time flush.
    @MainActor
    func testLoadFinishingAfterStopOpensNoLiveSession() async throws {
        let server = try LiveEventServer(events: ["""
        {"type":"transcription.segment","segment_id":0,"text":"late session","is_final":false,"event_index":1}
        """])
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-late-load-\(UUID().uuidString)")
        let modelDirectory = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let executable = root.appendingPathComponent("server")
        try """
        #!/bin/sh
        printf '{"event":"ready","host":"127.0.0.1","port":\(server.port),"model_name":"m","server_pid":%s}\\n' "$$"
        read -r command
        """.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let runtime = OmniASRRuntime(
            kind: .qwen3ASR,
            modelDirectory: modelDirectory,
            configuration: OmniBackendConfiguration(runtimeExecutable: executable)
        )
        let loadGate = ManualTaskBarrier()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await loadGate.wait()
            _ = try await runtime.prepare()
            return MLXLoadedModelBox(loaded: .omni(runtime))
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        transcriber.onPartialTranscription = { _ in XCTFail("A live session opened after Stop") }
        transcriber.isRecording = true
        let setup = transcriber.startNativeLiveSession(revision: 0)
        await loadGate.waitUntilEntered()
        transcriber.noteStopRequested()
        loadGate.release()
        await setup.value

        XCTAssertEqual(server.connectionCount, 0)
        XCTAssertFalse(transcriber.isModelInitializing)
        transcriber.isRecording = false
        await manager.shutdownForApplicationTermination()
        await runtime.retire()
    }

    @MainActor
    func testCancelledLiveSetupNeitherRetriesNorReportsAFailure() async {
        let barrier = ManualTaskBarrier()
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            await barrier.wait()
            throw OmniASRRuntimeError.launchFailed("injected cancelled failure")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        transcriber.onRuntimeFailure = { _ in XCTFail("Cancelled setup reported a failure") }
        transcriber.isRecording = true
        let setup = transcriber.startNativeLiveSession(revision: 0)
        await barrier.waitUntilEntered()
        setup.cancel()
        barrier.release()
        await setup.value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 1)
        XCTAssertNil(transcriber.consumePendingRuntimeFailureMessage())
        XCTAssertFalse(manager.hasActiveUse)
        transcriber.isRecording = false
        await manager.shutdownForApplicationTermination()
    }

    /// A server error ends the live session: the stream finishes, so whoever
    /// forwards it releases its runtime lease, and the message carries no text
    /// the server may have echoed from the request.
    func testALiveSocketErrorEventEndsTheSession() async throws {
        let server = try LiveEventServer(events: ["""
        {"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_event", "message": "input_value='U0VDUkVUIEFVRElP'"}}
        """])
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let session = OmniRealtimeTranscriptionSession(
            endpoint: OmniServerEndpoint(host: "127.0.0.1", port: server.port, modelName: "m", serverProcessIdentifier: 0),
            language: nil
        )
        session.feedAudio(samples: [0, 0.1, 0.2])

        let drained = Task { () -> [OmniLiveEvent] in
            var received: [OmniLiveEvent] = []
            for await event in session.events { received.append(event) }
            return received
        }
        let finished = Task { () -> Bool in
            try? await Task.sleep(for: .seconds(5))
            return false
        }
        let events = await withTaskGroup(of: [OmniLiveEvent]?.self) { group in
            group.addTask { await drained.value }
            group.addTask { _ = await finished.value; return nil }
            let first = await group.next() ?? nil
            group.cancelAll()
            drained.cancel()
            return first
        }
        finished.cancel()

        let received = try XCTUnwrap(events, "the live session never ended after a server error")
        guard case .failed(let message) = received.last else {
            return XCTFail("expected a failure event, got \(received)")
        }
        XCTAssertFalse(message.contains("U0VDUkVUIEFVRElP"))
        XCTAssertTrue(message.contains("invalid_event"))
    }

    /// A live feed that ignored how much audio was still unsent could overflow the queue when the
    /// socket stalled, failing the session; the reported capacity is what lets the feed wait instead.
    func testQueueCapacityShrinksWhileAudioIsUnsent() throws {
        let listener = try NWListener(using: .tcp, on: .any)
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            if case .ready = state { ready.signal() }
        }
        listener.newConnectionHandler = { _ in }
        listener.start(queue: DispatchQueue(label: "omni-stalled-server"))
        defer { listener.cancel() }
        XCTAssertEqual(ready.wait(timeout: .now() + 5), .success)
        let session = OmniRealtimeTranscriptionSession(
            endpoint: OmniServerEndpoint(
                host: "127.0.0.1",
                port: Int(listener.port?.rawValue ?? 0),
                modelName: "m",
                serverProcessIdentifier: 0
            ),
            language: nil
        )
        defer { session.cancel() }
        let maximum = OmniRealtimeTranscriptionSession.maximumQueuedSamples

        XCTAssertEqual(session.queueCapacitySamples, maximum)
        session.feedAudio(samples: [Float](repeating: 0.1, count: maximum - 1_000))
        XCTAssertEqual(session.queueCapacitySamples, 1_000)
        session.feedAudio(samples: [Float](repeating: 0.1, count: 1_000))
        XCTAssertEqual(session.queueCapacitySamples, 0)
    }

    func testErrorDescriptionsCarryNoRequestOrTranscriptText() {
        let secret = "dictionary term and transcript text"
        let errors: [OmniTranscriptionError] = [
            .httpStatus(422, "{\"detail\": [{\"input\": \"\(secret)\"}]}"),
            .malformedEvent("{\"type\": \"transcript.text.done\", \"text\": \"\(secret)\"}"),
        ]
        for error in errors {
            let description = error.errorDescription ?? ""
            XCTAssertFalse(description.contains(secret), description)
        }
        XCTAssertTrue(OmniTranscriptionError.httpStatus(422, secret).errorDescription?.contains("422") ?? false)
    }

    @MainActor
    func testTheLedgerReleasesRuntimesNoLoadWillAdopt() async {
        let scratch = FileManager.default.temporaryDirectory
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: URL(fileURLWithPath: "/usr/bin/false")
        )
        let adopted = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
        let orphan = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
        let ledger = OmniRuntimeLedger()
        ledger.track(adopted)
        ledger.adopt(adopted)
        ledger.track(orphan)

        ledger.releaseUnadopted()
        await ledger.waitForRetirements()

        XCTAssertTrue(ledger.pendingRuntimes().isEmpty)
        let orphanState = await orphan.state
        let adoptedState = await adopted.state
        XCTAssertEqual(orphanState, .stopped)
        XCTAssertEqual(adoptedState, .idle)
    }

    /// Retiring a runtime whose launch has not started yet must keep the
    /// server process from starting at all. A smoke check: the window between
    /// prepare() queueing the launch and the launch running cannot be hit on
    /// demand, so this passes without the guard too; it catches leftovers.
    func testRetiringBeforeTheLaunchRunsStartsNoServerProcess() async throws {
        let token = "voxt-omni-retire-race-\(UUID().uuidString)"
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(token, isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let script = scratch.appendingPathComponent("qwen3_asr_server")
        // Note (Jiaxin Deng): stands in for the runtime; exits once Voxt writes or closes its control pipe.
        try "#!/bin/sh\nexec /bin/sh -c 'read line' \(token)\n".write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: script,
            startupTimeoutSeconds: 5
        )

        var preparations: [Task<OmniServerEndpoint, Error>] = []
        for _ in 0..<20 {
            let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
            preparations.append(Task { try await runtime.prepare() })
            await runtime.retire()
        }
        try await Task.sleep(for: .milliseconds(500))
        let leftover = Self.processes(matching: token)
        // Note (Jiaxin Deng): a server started after retirement would wait on its control pipe forever.
        Self.kill(matching: token)
        preparations.forEach { $0.cancel() }

        XCTAssertEqual(leftover, 0, "a server process started after its runtime retired")
    }

    private static func kill(matching token: String) {
        let pkill = Process()
        pkill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        pkill.arguments = ["-f", token]
        try? pkill.run()
        pkill.waitUntilExit()
    }

    func testQwenEnergyCutsMayPassTheChunkEndUnlessThatWouldExceedTheServerLimit() {
        XCTAssertTrue(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 90))
        XCTAssertTrue(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 1195))
        XCTAssertFalse(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 1196))
        XCTAssertFalse(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 1200))
    }

    private static func processes(matching token: String) -> Int {
        let pgrep = Process()
        pgrep.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        pgrep.arguments = ["-f", token]
        let output = Pipe()
        pgrep.standardOutput = output
        try? pgrep.run()
        pgrep.waitUntilExit()
        let text = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        return text.split(whereSeparator: \.isNewline).count
    }
}
