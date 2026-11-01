#if OMNI_SILERO_PORTABLE_TESTS
import Foundation
import OmniModelArtifacts
import XCTest

@testable import Voxt

// Note (assistant): a controlled loopback process supplies protocol sentinels, not model probabilities.
@MainActor
final class SileroVADActorOwnershipTests: XCTestCase {
    private var root: URL!
    private var directory: URL!
    private let samples = [Float](repeating: 0.75, count: 512)

    override func setUp() async throws {
        guard ProcessInfo.processInfo.environment["VOXT_SILERO_HTTP_FIXTURE"] != nil else {
            throw XCTSkip("The portable HTTP process fixture is required.")
        }
        root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-silero-http-\(UUID())")
        directory = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        let header = Data("{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}".utf8)
        var headerByteCount = UInt64(header.count).littleEndian
        var weights = withUnsafeBytes(of: &headerByteCount) { Data($0) }
        weights.append(header)
        weights.append(Data(repeating: 0, count: 4))
        try weights.write(to: directory.appendingPathComponent("model.safetensors"))
        setenv("VOXT_SILERO_FIXTURE_DIRECTORY", directory.path, 1)
        SileroVADModelProvisioner.shared.configure(directory: directory)
    }

    override func tearDown() async throws {
        guard let root, let directory else { return }
        SileroVADModelProvisioner.shared.release()
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        where file.lastPathComponent.hasPrefix("block-") {
            try FileManager.default.removeItem(at: file)
        }
        await OmniSileroVADRuntime.shared.shutdownForApplicationTermination()
        try FileManager.default.removeItem(at: root)
    }

    func testNativeUnavailableRejectsBothDetectorsWithoutProvisioning() async throws {
        let executable = try XCTUnwrap(ProcessInfo.processInfo.environment["VOXT_SILERO_HTTP_FIXTURE"])
        unsetenv("VOXT_SILERO_HTTP_FIXTURE")
        defer { setenv("VOXT_SILERO_HTTP_FIXTURE", executable, 1) }
        SileroVADModelProvisioner.shared.block()
        do {
            _ = try await ASRSileroStreamingVoiceActivityDetector().probability(samples: samples, sampleRate: 16_000)
            XCTFail("Native unavailable must not run another backend")
        } catch MeetingVADModelError.runtimeUnavailable {}
        do {
            _ = try await ASRSileroOfflineVoiceActivityDetector().speechRanges(samples: samples, sampleRate: 16_000)
            XCTFail("Native unavailable must not run another backend")
        } catch MeetingVADModelError.runtimeUnavailable {}
        XCTAssertEqual(SileroVADModelProvisioner.shared.waitingCount, 0)
        XCTAssertEqual(eventCount("starting"), 0)
    }

    func testBothDetectorsShareProcessReaderUntilLastUnloadAndActualExit() async throws {
        let streaming = ASRSileroStreamingVoiceActivityDetector()
        let offline = ASRSileroOfflineVoiceActivityDetector()
        _ = try await streaming.probability(samples: samples, sampleRate: 16_000)
        _ = try await offline.speechRanges(samples: samples, sampleRate: 16_000)
        XCTAssertEqual(eventCount("starting"), 1)
        try assertBusy()
        await streaming.unload()
        try assertBusy()
        try block("exit")
        var finished = false
        let retirement = Task { await offline.unload(); finished = true }
        try await waitUntil { self.eventCount("shutdown") == 1 }
        XCTAssertFalse(finished)
        try assertBusy()
        try release("exit")
        await retirement.value
        try assertMutable()
    }

    func testStreamingLastFrameNilIndependentStreamsAndReset() async throws {
        let detector = ASRSileroStreamingVoiceActivityDetector()
        let partial = try await detector.probability(samples: Array(samples.prefix(300)), sampleRate: 16_000)
        XCTAssertNil(partial)
        let last = try await detector.probability(samples: [Float](repeating: 0, count: 724), sampleRate: 16_000)
        XCTAssertEqual(last, 0.875, "Two completed fixture frames return the last sentinel")
        let unfinished = try await detector.probability(samples: [0], sampleRate: 16_000)
        XCTAssertNil(unfinished, "A previous probability does not replace a nil response")
        let sibling = try await detector.probability(samples: samples, sampleRate: 16_000, streamID: "other")
        XCTAssertEqual(sibling, 0.125)
        await detector.reset()
        try assertBusy()
        let reset = try await detector.probability(samples: samples, sampleRate: 16_000)
        XCTAssertEqual(reset, 0.125)
        XCTAssertEqual(eventCount("starting"), 1)
        await detector.unload()
        try assertMutable()
    }

    func testResetWaitsForBlockedExchangeAndDoesNotAdmitReplacementEarly() async throws {
        let detector = ASRSileroStreamingVoiceActivityDetector()
        try block("stream")
        let pending = Task { try await detector.probability(samples: samples, sampleRate: 16_000) }
        try await waitUntil { self.eventCount("stream") == 1 }
        var resetFinished = false
        let reset = Task { await detector.reset(); resetFinished = true }
        try await Task.sleep(for: .milliseconds(50))
        let replacement = Task { try await detector.probability(samples: samples, sampleRate: 16_000) }
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertFalse(resetFinished)
        XCTAssertEqual(eventCount("opened"), 1)
        try assertBusy()
        try release("stream")
        let stale = try await pending.value
        XCTAssertNil(stale)
        await reset.value
        let next = try await replacement.value
        XCTAssertEqual(next, 0.125)
        XCTAssertEqual(eventCount("starting"), 1)
        await detector.unload()
        try assertMutable()
    }

    func testUnloadWaitsForOfflineCommandAndReplacementWaitsForProcessExit() async throws {
        let detector = ASRSileroOfflineVoiceActivityDetector()
        try block("offline")
        try block("exit")
        let pending = Task { try await detector.speechRanges(samples: samples, sampleRate: 16_000) }
        try await waitUntil { self.eventCount("offline") == 1 }
        var unloadFinished = false
        let unload = Task { await detector.unload(); unloadFinished = true }
        try await Task.sleep(for: .milliseconds(50))
        let replacement = Task { try await detector.speechRanges(samples: samples, sampleRate: 16_000) }
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertFalse(unloadFinished)
        XCTAssertEqual(eventCount("shutdown"), 0)
        XCTAssertEqual(eventCount("offline"), 1)
        try assertBusy()
        try release("offline")
        do { _ = try await pending.value; XCTFail("Unloaded command returned stale ranges") }
        catch is CancellationError {}
        try await waitUntil { self.eventCount("shutdown") == 1 }
        XCTAssertFalse(unloadFinished)
        XCTAssertEqual(eventCount("starting"), 1)
        try assertBusy()
        try release("exit")
        await unload.value
        _ = try await replacement.value
        XCTAssertEqual(eventCount("starting"), 2)
        try assertBusy()
        await detector.unload()
        try assertMutable()
    }

    func testOldCloseCannotReleaseReplacementEndpoint() async throws {
        let detector = ASRSileroStreamingVoiceActivityDetector()
        _ = try await detector.probability(samples: samples, sampleRate: 16_000)
        try block("close")
        try block("exit")
        let unload = Task { await detector.unload() }
        try await waitUntil { self.eventCount("closing") == 1 }
        let replacement = Task { try await detector.probability(samples: samples, sampleRate: 16_000) }
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(eventCount("starting"), 1)
        XCTAssertEqual(eventCount("opened"), 1)
        XCTAssertEqual(eventCount("shutdown"), 0)
        try assertBusy()
        try release("close")
        try await waitUntil { self.eventCount("shutdown") == 1 }
        XCTAssertEqual(eventCount("starting"), 1)
        try release("exit")
        await unload.value
        let next = try await replacement.value
        XCTAssertEqual(next, 0.125)
        XCTAssertEqual(eventCount("starting"), 2)
        try assertBusy()
        await detector.unload()
        try assertMutable()
    }

    func testCancelledExchangeAndQueuedUnloadRetainReaderUntilExit() async throws {
        let detector = ASRSileroStreamingVoiceActivityDetector()
        try block("stream")
        try block("exit")
        let pending = Task { try await detector.probability(samples: samples, sampleRate: 16_000) }
        try await waitUntil { self.eventCount("stream") == 1 }
        pending.cancel()
        let unload = Task { await detector.unload() }
        try release("stream")
        try await waitUntil { self.eventCount("shutdown") == 1 }
        try assertBusy()
        try release("exit")
        _ = try? await pending.value
        await unload.value
        try assertMutable()
    }

    func testInvalidAudioDoesNotAcquireAndStreamingResamplesBeforeTransport() async throws {
        let streaming = ASRSileroStreamingVoiceActivityDetector()
        let offline = ASRSileroOfflineVoiceActivityDetector()
        for rate in [0.0, -1.0, Double.nan, Double.infinity] {
            let probability = try await streaming.probability(samples: samples, sampleRate: rate)
            let ranges = try await offline.speechRanges(samples: samples, sampleRate: rate)
            XCTAssertNil(probability)
            XCTAssertTrue(ranges.isEmpty)
        }
        let emptyProbability = try await streaming.probability(samples: [], sampleRate: 16_000)
        let emptyRanges = try await offline.speechRanges(samples: [], sampleRate: 16_000)
        XCTAssertNil(emptyProbability)
        XCTAssertTrue(emptyRanges.isEmpty)
        XCTAssertEqual(eventCount("starting"), 0)
        let probability = try await streaming.probability(samples: Array(samples.prefix(256)), sampleRate: 8_000)
        XCTAssertEqual(probability, 0.125)
        XCTAssertEqual(events().first { $0["event"] as? String == "stream" }?["sampleCount"] as? Int, 512)
        await streaming.unload()
    }

    func testConcurrentCallsUseOneAcquisitionPerDetector() async throws {
        SileroVADModelProvisioner.shared.block()
        let streaming = ASRSileroStreamingVoiceActivityDetector()
        let offline = ASRSileroOfflineVoiceActivityDetector()
        let first = Task { try await streaming.probability(samples: samples, sampleRate: 16_000, streamID: "one") }
        let second = Task { try await streaming.probability(samples: samples, sampleRate: 16_000, streamID: "two") }
        let third = Task { try await offline.speechRanges(samples: samples, sampleRate: 16_000) }
        let fourth = Task { try await offline.speechRanges(samples: samples, sampleRate: 16_000) }
        try await waitUntil { SileroVADModelProvisioner.shared.waitingCount == 2 }
        SileroVADModelProvisioner.shared.release()
        _ = try await (first.value, second.value, third.value, fourth.value)
        XCTAssertEqual(eventCount("starting"), 1)
        await streaming.unload()
        try assertBusy()
        await offline.unload()
        try assertMutable()
    }

    func testResetAndUnloadInvalidateBlockedProvisioningWithoutStartingProcess() async throws {
        SileroVADModelProvisioner.shared.block()
        let streaming = ASRSileroStreamingVoiceActivityDetector()
        let offline = ASRSileroOfflineVoiceActivityDetector()
        let first = Task { try await streaming.probability(samples: samples, sampleRate: 16_000) }
        let second = Task { try await offline.speechRanges(samples: samples, sampleRate: 16_000) }
        try await waitUntil { SileroVADModelProvisioner.shared.waitingCount == 2 }
        let reset = Task { await streaming.reset() }
        let unload = Task { await offline.unload() }
        try await Task.sleep(for: .milliseconds(50))
        SileroVADModelProvisioner.shared.release()
        do { _ = try await first.value; XCTFail("Reset resurrected streaming acquisition") }
        catch is CancellationError {}
        do { _ = try await second.value; XCTFail("Unload resurrected offline acquisition") }
        catch is CancellationError {}
        await reset.value
        await unload.value
        XCTAssertEqual(eventCount("starting"), 0)
        try assertMutable()
    }

    func testCancellationWhileProvisioningDoesNotStartProcess() async throws {
        SileroVADModelProvisioner.shared.block()
        let streaming = ASRSileroStreamingVoiceActivityDetector()
        let offline = ASRSileroOfflineVoiceActivityDetector()
        let first = Task { try await streaming.probability(samples: samples, sampleRate: 16_000) }
        let second = Task { try await offline.speechRanges(samples: samples, sampleRate: 16_000) }
        try await waitUntil { SileroVADModelProvisioner.shared.waitingCount == 2 }
        first.cancel()
        second.cancel()
        SileroVADModelProvisioner.shared.release()
        do { _ = try await first.value; XCTFail("Cancelled streaming acquisition succeeded") }
        catch is CancellationError {}
        do { _ = try await second.value; XCTFail("Cancelled offline acquisition succeeded") }
        catch is CancellationError {}
        XCTAssertEqual(eventCount("starting"), 0)
        try assertMutable()
    }

    func testUnloadDuringStartupRetainsReaderUntilProcessExit() async throws {
        try block("ready")
        try block("exit")
        let detector = ASRSileroOfflineVoiceActivityDetector()
        let pending = Task { try await detector.speechRanges(samples: samples, sampleRate: 16_000) }
        try await waitUntil { self.eventCount("starting") == 1 }
        let unload = Task { await detector.unload() }
        try await Task.sleep(for: .milliseconds(50))
        try release("ready")
        try await waitUntil { self.eventCount("shutdown") == 1 }
        try assertBusy()
        XCTAssertEqual(eventCount("offline"), 0)
        try release("exit")
        do { _ = try await pending.value; XCTFail("Unload resurrected a starting endpoint") }
        catch is CancellationError {}
        await unload.value
        try assertMutable()
    }

    func testWholeAudioOverThirtySecondsIsOneRequestWithOriginalOptionsAndSeconds() async throws {
        let detector = ASRSileroOfflineVoiceActivityDetector()
        let ranges = try await detector.speechRanges(samples: [Float](repeating: 0, count: 31 * 8_000), sampleRate: 8_000)
        XCTAssertEqual(ranges, [
            ASROfflineSpeechRange(startSeconds: 0.01, endSeconds: 0.5),
            ASROfflineSpeechRange(startSeconds: 2, endSeconds: 3),
        ])
        XCTAssertEqual(eventCount("offline"), 1)
        XCTAssertEqual(events().first { $0["event"] as? String == "offline" }?["sampleCount"] as? Int, 31 * 16_000)
        let multipart = String(decoding: try Data(contentsOf: directory.appendingPathComponent("last-request.multipart")), as: UTF8.self)
        let profile = MeetingSileroVADSensitivity.stored().configuration()
        for (name, value) in [
            ("threshold", String(profile.onsetProbabilityThreshold)),
            ("min_speech_duration_ms", String(Int((profile.minSpeechSeconds * 1_000).rounded(.up)))),
            ("min_silence_duration_ms", String(Int((profile.minSilenceSeconds * 1_000).rounded(.up)))),
            ("speech_pad_ms", String(Int((profile.speechPadSeconds * 1_000).rounded(.up)))),
        ] {
            XCTAssertTrue(multipart.contains("name=\"\(name)\"\r\n\r\n\(value)\r\n"))
        }
        await detector.unload()
    }

    func testStartupFailureReleasesReadersAndAllowsRetry() async throws {
        let failure = directory.appendingPathComponent("fail-start")
        try Data().write(to: failure)
        let streaming = ASRSileroStreamingVoiceActivityDetector()
        let offline = ASRSileroOfflineVoiceActivityDetector()
        do { _ = try await streaming.probability(samples: samples, sampleRate: 16_000); XCTFail("Expected startup failure") }
        catch OmniASRRuntimeError.launchFailed {}
        try assertMutable()
        do { _ = try await offline.speechRanges(samples: samples, sampleRate: 16_000); XCTFail("Expected startup failure") }
        catch OmniASRRuntimeError.launchFailed {}
        try assertMutable()
        try FileManager.default.removeItem(at: failure)
        _ = try await streaming.probability(samples: samples, sampleRate: 16_000)
        _ = try await offline.speechRanges(samples: samples, sampleRate: 16_000)
        XCTAssertEqual(eventCount("starting"), 3)
        await streaming.unload()
        try assertBusy()
        await offline.unload()
        try assertMutable()
    }

    func testCancelledOfflineCommandKeepsProcessReaderThroughRetirement() async throws {
        try block("offline")
        try block("exit")
        let detector = ASRSileroOfflineVoiceActivityDetector()
        let pending = Task { try await detector.speechRanges(samples: samples, sampleRate: 16_000) }
        try await waitUntil { self.eventCount("offline") == 1 }
        pending.cancel()
        let unload = Task { await detector.unload() }
        try await waitUntil { self.eventCount("shutdown") == 1 }
        try assertBusy()
        try release("offline")
        try release("exit")
        _ = try? await pending.value
        await unload.value
        try assertMutable()
    }

    func testHTTPFailureIsNotFallbackAndRetainsReaderUntilUnload() async throws {
        try Data().write(to: directory.appendingPathComponent("fail-http"))
        let detector = ASRSileroOfflineVoiceActivityDetector()
        do { _ = try await detector.speechRanges(samples: samples, sampleRate: 16_000); XCTFail("Expected HTTP failure") }
        catch OmniVoiceActivityError.httpStatus(503) {}
        try assertBusy()
        await detector.unload()
        try assertMutable()
    }

    func testTransportFailureRetiresOldEndpointBeforeReplacement() async throws {
        try Data().write(to: directory.appendingPathComponent("fail-transport"))
        try block("exit")
        let detector = ASRSileroStreamingVoiceActivityDetector()
        let failed = Task { try await detector.probability(samples: samples, sampleRate: 16_000) }
        try await waitUntil { self.eventCount("shutdown") == 1 }
        let replacement = Task { try await detector.probability(samples: samples, sampleRate: 16_000) }
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(eventCount("starting"), 1)
        try assertBusy()
        try FileManager.default.removeItem(at: directory.appendingPathComponent("fail-transport"))
        try release("exit")
        do { _ = try await failed.value; XCTFail("Transport failure was hidden") } catch {}
        let next = try await replacement.value
        XCTAssertEqual(next, 0.125)
        XCTAssertEqual(eventCount("starting"), 2)
        try assertBusy()
        await detector.unload()
        try assertMutable()
    }

    func testDroppingActorsEventuallyRetiresProcessAndReader() async throws {
        var streaming: ASRSileroStreamingVoiceActivityDetector? = ASRSileroStreamingVoiceActivityDetector()
        var offline: ASRSileroOfflineVoiceActivityDetector? = ASRSileroOfflineVoiceActivityDetector()
        _ = try await streaming?.probability(samples: samples, sampleRate: 16_000)
        _ = try await offline?.speechRanges(samples: samples, sampleRate: 16_000)
        streaming = nil
        offline = nil
        try await waitUntil { (try? OmniModelDirectoryLease.acquire(for: self.directory, mode: .exclusiveMutation)) != nil }
        XCTAssertEqual(eventCount("exiting"), 1)
    }

    func testMutationPreventsBothProcessStarts() async throws {
        let mutation = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
        defer { withExtendedLifetime(mutation) {} }
        let streaming = ASRSileroStreamingVoiceActivityDetector()
        let offline = ASRSileroOfflineVoiceActivityDetector()
        do { _ = try await streaming.probability(samples: samples, sampleRate: 16_000); XCTFail("Mutation admitted startup") }
        catch OmniModelStorageError.busy {}
        do { _ = try await offline.speechRanges(samples: samples, sampleRate: 16_000); XCTFail("Mutation admitted startup") }
        catch OmniModelStorageError.busy {}
        XCTAssertEqual(eventCount("starting"), 0)
    }

    private func block(_ name: String) throws {
        try Data().write(to: directory.appendingPathComponent("block-\(name)"))
    }

    private func release(_ name: String) throws {
        try FileManager.default.removeItem(at: directory.appendingPathComponent("block-\(name)"))
    }

    private func events() -> [[String: Any]] {
        guard let bytes = try? Data(contentsOf: directory.appendingPathComponent("events.jsonl")) else { return [] }
        return bytes.split(separator: 10).compactMap { (try? JSONSerialization.jsonObject(with: Data($0))) as? [String: Any] }
    }

    private func eventCount(_ name: String) -> Int {
        events().filter { $0["event"] as? String == name }.count
    }

    private func assertBusy(file: StaticString = #filePath, line: UInt = #line) throws {
        do {
            _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
            XCTFail("An active HTTP process lost its reader", file: file, line: line)
        } catch OmniModelStorageError.busy {}
    }

    private func assertMutable() throws {
        _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
    }

    private func waitUntil(_ predicate: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !predicate(), ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        guard predicate() else {
            XCTFail("Timed out waiting for the HTTP fixture barrier")
            throw CocoaError(.fileReadUnknown)
        }
    }
}
#endif
