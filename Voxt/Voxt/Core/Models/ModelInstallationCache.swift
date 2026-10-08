import Foundation

nonisolated struct ModelInstallationSnapshot: Equatable, Sendable {
    var directory: URL?
    var existingDirectory: URL?
    var hasPartialDownload = false
    var allocatedBytes: Int64 = 0
    var isInstalled: Bool { directory != nil }
}

/// Snapshot all security-scoped/root URLs on MainActor before constructing this
/// request. The worker never reads preferences or touches model/runtime objects.
nonisolated struct ModelInstallationRequest: Sendable {
    let directories: [URL]
    let partialDirectories: [URL]
    let validate: @Sendable (URL) -> Bool

    func scan() -> ModelInstallationSnapshot {
        let fm = FileManager.default
        var result = ModelInstallationSnapshot()
        for directory in directories {
            guard !Task.isCancelled else { return result }
            guard fm.fileExists(atPath: directory.path) else { continue }
            if result.existingDirectory == nil { result.existingDirectory = directory }
            if validate(directory) {
                result.directory = directory
                result.allocatedBytes = Self.size(at: directory)
                break
            }
        }
        if !result.isInstalled {
            result.hasPartialDownload = partialDirectories.contains { directory in
                guard fm.fileExists(atPath: directory.path) else { return false }
                if (try? directory.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true { return true }
                guard let files = fm.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]) else { return false }
                for case let file as URL in files {
                    if Task.isCancelled { return false }
                    if (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true { return true }
                }
                return false
            }
        }
        return result
    }

    private static func size(at url: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileAllocatedSizeKey, .totalFileAllocatedSizeKey]
        if let values = try? url.resourceValues(forKeys: keys), values.isRegularFile == true {
            return Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0)
        }
        guard let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: Array(keys)) else { return 0 }
        var bytes: Int64 = 0
        for case let file as URL in files {
            if Task.isCancelled { return bytes }
            guard let values = try? file.resourceValues(forKeys: keys), values.isRegularFile == true else { continue }
            bytes += Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0)
        }
        return bytes
    }
}

/// Coalesces background disk reads. Every invalidation creates a new identity:
/// late workers cannot restore an uninstalled model or an old storage-root state.
@MainActor
final class ModelInstallationCache {
    private struct Pending {
        let id: UUID
        let task: Task<ModelInstallationSnapshot, Never>
    }
    private var values: [String: ModelInstallationSnapshot] = [:]
    private var pending: [String: Pending] = [:]
    private var identities: [String: UUID] = [:]
    private var waiters: [String: [UUID: CheckedContinuation<ModelInstallationSnapshot, Error>]] = [:]
    var onChange: ((String, ModelInstallationSnapshot) -> Void)?

    func peek(_ key: String) -> ModelInstallationSnapshot? { values[key] }
    func needsRequest(_ key: String) -> Bool { values[key] == nil && pending[key] == nil }
    func hasPendingRequest(_ key: String) -> Bool { pending[key] != nil }
    func isChecking(_ key: String) -> Bool { pending[key] != nil || values[key] == nil }

    func request(_ key: String, scan: @escaping @Sendable () -> ModelInstallationSnapshot) {
        guard values[key] == nil, pending[key] == nil else { return }
        let id = UUID()
        let task = Task.detached(priority: .utility) {
            (try? await ModelDiskOperations.perform(scan)) ?? ModelInstallationSnapshot()
        }
        pending[key] = Pending(id: id, task: task)
        identities[key] = id
        Task { [weak self] in
            let value = await task.value
            self?.publish(value, key: key, id: id)
        }
    }

    func value(_ key: String, scan: @escaping @Sendable () -> ModelInstallationSnapshot) async throws -> ModelInstallationSnapshot {
        try Task.checkCancellation()
        request(key, scan: scan)
        if let value = values[key] { return value }
        guard let work = pending[key] else { throw CancellationError() }
        let waiterID = UUID()
        let result: ModelInstallationSnapshot = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                guard !Task.isCancelled, identities[key] == work.id else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                if let value = values[key] {
                    continuation.resume(returning: value)
                } else {
                    waiters[key, default: [:]][waiterID] = continuation
                }
            }
        } onCancel: {
            Task { @MainActor [weak self] in
                self?.waiters[key]?.removeValue(forKey: waiterID)?.resume(throwing: CancellationError())
            }
        }
        try Task.checkCancellation()
        guard identities[key] == work.id else { throw CancellationError() }
        return result
    }

    func invalidate(_ key: String) {
        pending.removeValue(forKey: key)?.task.cancel()
        if let cancelled = waiters.removeValue(forKey: key) {
            for waiter in cancelled.values { waiter.resume(throwing: CancellationError()) }
        }
        values.removeValue(forKey: key)
        identities.removeValue(forKey: key)
    }

    func invalidateAll() {
        for work in pending.values { work.task.cancel() }
        for group in waiters.values {
            for waiter in group.values { waiter.resume(throwing: CancellationError()) }
        }
        waiters.removeAll()
        pending.removeAll()
        values.removeAll()
        identities.removeAll()
    }

    private func publish(_ value: ModelInstallationSnapshot, key: String, id: UUID) {
        guard pending[key]?.id == id else { return }
        pending.removeValue(forKey: key)
        values[key] = value
        let completedWaiters = waiters.removeValue(forKey: key)
        onChange?(key, value)
        if let completedWaiters {
            for waiter in completedWaiters.values {
                if identities[key] == id { waiter.resume(returning: value) }
                else { waiter.resume(throwing: CancellationError()) }
            }
        }
    }
}
