import Dispatch
import Foundation
import XCTest
@testable import Voxt

final class MeetingFileInferenceCacheTests: XCTestCase {
    func testCachePolicyPreservesSmallCachesButTrimsTheObservedLargeCache() {
        let threshold = MeetingFileInferenceCache.retainedCacheThresholdBytes
        XCTAssertFalse(MeetingFileInferenceCache.shouldTrim(cacheBytes: 0, underPressure: true))
        XCTAssertFalse(MeetingFileInferenceCache.shouldTrim(cacheBytes: threshold, underPressure: false))
        XCTAssertTrue(MeetingFileInferenceCache.shouldTrim(cacheBytes: threshold + 1, underPressure: false))
        XCTAssertTrue(MeetingFileInferenceCache.shouldTrim(cacheBytes: 1_788_153_124, underPressure: false))
        XCTAssertTrue(MeetingFileInferenceCache.shouldTrim(cacheBytes: 1, underPressure: true))
    }

    func testPressureProbeAcceptsOnlyKnownCurrentLevels() {
        XCTAssertEqual(MeetingMemoryPressureMonitor.constraint(forRawLevel: UInt32(DispatchSource.MemoryPressureEvent.normal.rawValue)), false)
        XCTAssertEqual(MeetingMemoryPressureMonitor.constraint(forRawLevel: UInt32(DispatchSource.MemoryPressureEvent.warning.rawValue)), true)
        XCTAssertEqual(MeetingMemoryPressureMonitor.constraint(forRawLevel: UInt32(DispatchSource.MemoryPressureEvent.critical.rawValue)), true)
        XCTAssertNil(MeetingMemoryPressureMonitor.constraint(forRawLevel: 0))
        XCTAssertNil(MeetingMemoryPressureMonitor.constraint(forRawLevel: .max))
        let mixed = DispatchSource.MemoryPressureEvent.normal.rawValue | DispatchSource.MemoryPressureEvent.warning.rawValue
        XCTAssertNil(MeetingMemoryPressureMonitor.constraint(forRawLevel: UInt32(mixed)))
    }

    func testBoundedFileWorkDoesNotBlockOnWarningButStillBlocksOnCritical() {
        XCTAssertEqual(MeetingMemoryPressureMonitor.fileConstraint(forRawLevel: UInt32(DispatchSource.MemoryPressureEvent.warning.rawValue)), false)
        XCTAssertEqual(MeetingMemoryPressureMonitor.fileConstraint(forRawLevel: UInt32(DispatchSource.MemoryPressureEvent.normal.rawValue)), false)
        XCTAssertEqual(MeetingMemoryPressureMonitor.fileConstraint(forRawLevel: UInt32(DispatchSource.MemoryPressureEvent.critical.rawValue)), true)
        XCTAssertNil(MeetingMemoryPressureMonitor.fileConstraint(forRawLevel: 0))
        XCTAssertNil(MeetingMemoryPressureMonitor.fileConstraint(forRawLevel: .max))
    }

    func testFileWarningPolicyDoesNotRelaxOtherCallers() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readMemoryPressure: { false })
        await coordinator.setMemoryPressureConstrained(true)
        try await coordinator.withPermit(.fileSpeakerAnalysis) {}
        do {
            try await coordinator.withPermit(.summary) {}
            XCTFail("A file policy override must not change the shared warning flag")
        } catch {
            guard let safety = error as? MeetingLocalInferenceCoordinatorError,
                  case .memoryConstrained = safety else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testCacheScopeIsRestoredOnFailureAndNeverInstalledForLiveWork() async throws {
        let probe = FileCacheProbe()
        let coordinator = MeetingLocalInferenceCoordinator(beginFileWorkUnit: {
            probe.beginScope()
            return { probe.endScope() }
        })
        do {
            try await coordinator.withPermit(.fileSpeakerAnalysis) {
                XCTAssertEqual(probe.openScopes, 1)
                throw URLError(.cannotDecodeContentData)
            }
            XCTFail("Expected original error")
        } catch { XCTAssertEqual((error as? URLError)?.code, .cannotDecodeContentData) }
        XCTAssertEqual(probe.openScopes, 0)
        try await coordinator.withPermit(.fileASR) { XCTAssertEqual(probe.openScopes, 1) }
        XCTAssertEqual(probe.openScopes, 0)
        try await coordinator.withPermit(.liveASRFinal) { XCTAssertEqual(probe.openScopes, 0) }
    }

    func testInheritedCacheIsReclaimedBeforeAdmissionAndAllowsConfirmedRecovery() async throws {
        let probe = FileCacheProbe()
        probe.setCache(1_788_153_124)
        let coordinator = MeetingLocalInferenceCoordinator(
            maintainFileCache: { probe.maintain(pressure: $0) },
            readMemoryPressure: { probe.cacheBytes > MeetingFileInferenceCache.retainedCacheThresholdBytes }
        )
        await coordinator.setMemoryPressureConstrained(true)
        try await coordinator.withPermit(.fileSpeakerAnalysis) {
            XCTAssertEqual(probe.cacheBytes, 0)
            XCTAssertEqual(probe.reclamations, 1)
        }
        XCTAssertEqual(probe.reclamations, 1)
    }

    func testFileCacheIsTrimmedOnReturnIncludingThrowAndCancellation() async throws {
        for kind in [MeetingLocalInferenceWorkClass.fileASR, .fileSpeakerAnalysis] {
            let probe = FileCacheProbe()
            let coordinator = MeetingLocalInferenceCoordinator(maintainFileCache: { probe.maintain(pressure: $0) })
            try await coordinator.withPermit(kind) { probe.setCache(1_788_153_124) }
            XCTAssertEqual(probe.cacheBytes, 0)
            do {
                try await coordinator.withPermit(kind) {
                    probe.setCache(1_788_153_124)
                    throw URLError(.cannotDecodeContentData)
                }
                XCTFail("Expected original inference error")
            } catch { XCTAssertEqual((error as? URLError)?.code, .cannotDecodeContentData) }
            XCTAssertEqual(probe.cacheBytes, 0)
            do {
                try await coordinator.withPermit(kind) {
                    probe.setCache(1_788_153_124)
                    throw CancellationError()
                }
                XCTFail("Expected cancellation")
            } catch { XCTAssertTrue(error is CancellationError) }
            XCTAssertEqual(probe.cacheBytes, 0)
            XCTAssertEqual(probe.reclamations, 3)
        }
    }

    func testSustainedOrUnknownPressureIsNotOverriddenByCacheReclamation() async throws {
        for sample in [Optional(true), nil] {
            let probe = FileCacheProbe()
            probe.setCache(1_788_153_124)
            let maintained = expectation(description: "wait boundary reached")
            let coordinator = MeetingLocalInferenceCoordinator(
                maintainFileCache: { pressure in
                    probe.maintain(pressure: pressure)
                    if probe.maintenanceCalls == 1 { maintained.fulfill() }
                },
                readMemoryPressure: { sample }
            )
            await coordinator.setMemoryPressureConstrained(true)
            let task = Task {
                try await coordinator.withPermit(.fileSpeakerAnalysis) {
                    XCTFail("Pressure must remain a blocker")
                }
            }
            await fulfillment(of: [maintained], timeout: 2)
            XCTAssertEqual(probe.cacheBytes, 0)
            task.cancel()
            do { try await task.value; XCTFail("Expected cancellation") }
            catch { XCTAssertTrue(error is CancellationError) }
            XCTAssertEqual(probe.reclamations, 1)
        }
    }

    func testFreshNormalSampleRecoversAnOldNotificationWithoutDisablingProtection() async throws {
        let coordinator = MeetingLocalInferenceCoordinator(readMemoryPressure: { false })
        await coordinator.setMemoryPressureConstrained(true)
        let result = try await coordinator.withPermit(.fileASR) { 42 }
        XCTAssertEqual(result, 42)
    }

    func testCancellationDoesNotReclaimUntilNativeFileWorkActuallyReturns() async throws {
        let probe = FileCacheProbe()
        let gate = FileCacheTestGate()
        let started = expectation(description: "file native work started")
        let coordinator = MeetingLocalInferenceCoordinator(maintainFileCache: { probe.maintain(pressure: $0) })
        let task = Task {
            try await coordinator.withPermit(.fileSpeakerAnalysis) {
                probe.setCache(1_788_153_124)
                started.fulfill()
                await gate.wait() // Simulates native work that ignores Task.cancel.
                try Task.checkCancellation()
            }
        }
        await fulfillment(of: [started], timeout: 2)
        task.cancel()
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(probe.cacheBytes, 1_788_153_124)
        XCTAssertEqual(probe.reclamations, 0)
        await gate.open()
        do { try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(probe.cacheBytes, 0)
        XCTAssertEqual(probe.reclamations, 1)
    }

    func testWaitingFileDoesNotReclaimWhileAnotherNativeOperationHoldsTheLane() async throws {
        let probe = FileCacheProbe()
        let gate = FileCacheTestGate()
        let started = expectation(description: "native operation started")
        let coordinator = MeetingLocalInferenceCoordinator(maintainFileCache: { probe.maintain(pressure: $0) })
        let active = Task {
            try await coordinator.withPermit(.liveASRFinal) {
                probe.setCache(1_788_153_124)
                started.fulfill()
                await gate.wait()
            }
        }
        await fulfillment(of: [started], timeout: 2)
        let waiting = Task {
            try await coordinator.withPermit(.fileSpeakerAnalysis) {
                XCTFail("Cancelled file must not run")
            }
        }
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(probe.maintenanceCalls, 0)
        waiting.cancel()
        do { try await waiting.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        await gate.open()
        try await active.value
        XCTAssertEqual(probe.maintenanceCalls, 0, "Non-file work does not gain cache maintenance")
    }
}

private nonisolated final class FileCacheProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var cache = 0
    private var trims = 0
    private var calls = 0
    private var scopes = 0
    var openScopes: Int { lock.withLock { scopes } }
    func beginScope() { lock.withLock { scopes += 1 } }
    func endScope() { lock.withLock { scopes -= 1 } }
    var cacheBytes: Int { lock.withLock { cache } }
    var reclamations: Int { lock.withLock { trims } }
    var maintenanceCalls: Int { lock.withLock { calls } }
    func setCache(_ bytes: Int) { lock.withLock { cache = bytes } }
    func maintain(pressure: Bool) {
        lock.withLock {
            calls += 1
            if MeetingFileInferenceCache.shouldTrim(cacheBytes: cache, underPressure: pressure) {
                cache = 0
                trims += 1
            }
        }
    }
}

private actor FileCacheTestGate {
    private var isOpen = false
    private var continuation: CheckedContinuation<Void, Never>?
    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { continuation = $0 }
    }
    func open() {
        isOpen = true
        continuation?.resume()
        continuation = nil
    }
}
