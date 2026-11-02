// HotkeyCaptureState.swift
// Provides process-local state for shortcut capture coordination.

import Foundation

nonisolated final class HotkeyCaptureState {
    static let shared = HotkeyCaptureState()

    private let lock = NSLock()
    private var captureInProgress = false

    private init() {
        UserDefaults.standard.set(false, forKey: AppPreferenceKey.hotkeyCaptureInProgress)
    }

    var isCaptureInProgress: Bool {
        lock.lock()
        defer { lock.unlock() }
        return captureInProgress
    }

    func setCaptureInProgress(_ isInProgress: Bool) {
        lock.lock()
        captureInProgress = isInProgress
        lock.unlock()
        UserDefaults.standard.set(isInProgress, forKey: AppPreferenceKey.hotkeyCaptureInProgress)
    }

    func refreshFromDefaults() {
        let isInProgress = UserDefaults.standard.bool(forKey: AppPreferenceKey.hotkeyCaptureInProgress)
        lock.lock()
        captureInProgress = isInProgress
        lock.unlock()
    }
}
