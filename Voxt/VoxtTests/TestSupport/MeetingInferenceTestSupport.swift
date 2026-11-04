import Foundation
import XCTest
@testable import Voxt

func waitForInferenceCondition(_ condition: @Sendable () async -> Bool) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: .seconds(5))
    while clock.now < deadline {
        if await condition() { return true }
        await Task.yield()
    }
    return false
}

func waitForQueuedInferenceWork(
    _ expectedCount: Int,
    coordinator: MeetingLocalInferenceCoordinator
) async -> Bool {
    await waitForInferenceCondition { await coordinator.currentStatistics().peakQueuedCount >= expectedCount }
}

nonisolated final class MeetingInferenceHooks: @unchecked Sendable {
    enum Event: Equatable { case trim(Bool), begin, restore, probe }

    private let lock = NSLock()
    private var recordedEvents: [Event] = []
    private var constrained = false
    private var thermal: ProcessInfo.ThermalState = .nominal

    var events: [Event] { lock.withLock { recordedEvents } }
    var memoryConstrained: Bool {
        get { lock.withLock { constrained } }
        set { lock.withLock { constrained = newValue } }
    }
    var thermalState: ProcessInfo.ThermalState {
        get { lock.withLock { thermal } }
        set { lock.withLock { thermal = newValue } }
    }

    func record(_ event: Event) { lock.withLock { recordedEvents.append(event) } }

    func makeCoordinator() -> MeetingLocalInferenceCoordinator {
        MeetingLocalInferenceCoordinator(
            maintainFileCache: { self.record(.trim($0)) },
            readMemoryPressure: { self.memoryConstrained },
            beginFileWorkUnit: {
                self.record(.begin)
                return { self.record(.restore) }
            },
            readThermalState: { self.thermalState }
        )
    }
}

actor MeetingInferenceConcurrencyTracker {
    private var active = 0
    private var peak = 0

    func begin() {
        active += 1
        peak = max(peak, active)
    }

    func end() {
        active -= 1
    }

    func peakCount() -> Int { peak }
}

actor MeetingInferenceGate {
    private var started = false
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        started = true
        await withCheckedContinuation { continuation = $0 }
    }

    func waitUntilStarted() async {
        while !started {
            await Task.yield()
        }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

actor MeetingInferenceOrderRecorder {
    private var recorded: [String] = []

    func append(_ value: String) { recorded.append(value) }
    func values() -> [String] { recorded }
}
