// MeetingMemoryPressureMonitor.swift
// Converts macOS memory-pressure notifications into a bounded-inference safety signal.

import Dispatch
import Foundation
import Darwin

nonisolated final class MeetingMemoryPressureMonitor: @unchecked Sendable {
    private let lock = NSLock()
    private var source: DispatchSourceMemoryPressure?

    /// XNU exposes the current system level in dispatch NOTE_MEMORYSTATUS units.
    /// This read-only probe is optional: unavailable/unknown values must NEVER
    /// clear a previously observed pressure event merely because time passed.
    /// File work is already serial, feed-bounded and cache-limited. A global
    /// warning requests reclamation, not an indefinite stop before loading any
    /// model. Critical pressure still blocks; unknown reads stay conservative.
    static func currentFileConstraint() -> Bool? {
        currentRawLevel().flatMap { fileConstraint(forRawLevel: $0) }
    }

    static func fileConstraint(forRawLevel level: UInt32) -> Bool? {
        switch UInt(level) {
        case DispatchSource.MemoryPressureEvent.normal.rawValue,
             DispatchSource.MemoryPressureEvent.warning.rawValue: return false
        case DispatchSource.MemoryPressureEvent.critical.rawValue: return true
        default: return nil
        }
    }

    private static func currentRawLevel() -> UInt32? {
        var level: UInt32 = 0
        var size = MemoryLayout<UInt32>.size
        guard sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nil, 0) == 0,
              size == MemoryLayout<UInt32>.size else { return nil }
        return level
    }

    static func constraint(forRawLevel level: UInt32) -> Bool? {
        switch UInt(level) {
        case DispatchSource.MemoryPressureEvent.normal.rawValue: return false
        case DispatchSource.MemoryPressureEvent.warning.rawValue,
             DispatchSource.MemoryPressureEvent.critical.rawValue: return true
        default: return nil
        }
    }

    func start(criticalOnly: Bool = false, handler: @escaping @Sendable (Bool) -> Void) {
        lock.lock()
        defer { lock.unlock() }
        guard source == nil else { return }

        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical],
            queue: DispatchQueue.global(qos: .utility)
        )
        source.setEventHandler {
            let event = source.data
            if criticalOnly {
                handler(Self.currentFileConstraint() ?? event.contains(.critical))
            } else {
                handler(Self.constraint(forRawLevel: UInt32(event.rawValue))
                    ?? (event.contains(.warning) || event.contains(.critical)))
            }
        }
        source.resume()
        self.source = source
    }

    deinit {
        source?.cancel()
    }
}
