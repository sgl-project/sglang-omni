// MainActorSync.swift
// Provides Main Actor Sync for shared utilities.

import Dispatch
import Foundation

enum MainActorSync {
    nonisolated static func run<T>(_ body: @escaping @MainActor () -> T) -> T {
        if Thread.isMainThread {
            return MainActor.assumeIsolated(body)
        }
        return DispatchQueue.main.sync {
            MainActor.assumeIsolated(body)
        }
    }
}
