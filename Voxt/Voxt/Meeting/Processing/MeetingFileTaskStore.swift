import Foundation

/// Owns file-task persistence and staging-directory accounting. It does not
/// schedule preparation or inference, so queue control flow stays separate
/// from disk bookkeeping.
@MainActor
final class MeetingFileTaskStore {
    struct PersistedPayload: Codable, Sendable {
        let version: Int
        let tasks: [MeetingFileTask]
    }

    static let maximumStagedSourceBytes: Int64 = 8 * 1024 * 1024 * 1024
    static let maximumTaskCount = 64

    let fileManager: FileManager
    let now: () -> Date
    let storageDirectoryURL: URL
    private let taskFileURL: URL
    private let persistenceCoordinator: AsyncJSONPersistenceCoordinator
    private(set) var reservedStagingBytes: Int64 = 0

    init(
        fileManager: FileManager = .default,
        storageDirectoryURL: URL? = nil,
        now: @escaping () -> Date = Date.init
    ) {
        self.fileManager = fileManager
        self.now = now
        let resolvedStorageDirectoryURL = storageDirectoryURL ?? Self.defaultStorageDirectoryURL(fileManager: fileManager)
        self.storageDirectoryURL = resolvedStorageDirectoryURL
        self.taskFileURL = resolvedStorageDirectoryURL.appendingPathComponent("tasks.json")
        self.persistenceCoordinator = AsyncJSONPersistenceCoordinator(
            label: "com.voxt.meeting-file-task-queue.persistence"
        )
    }

    func loadTasks() -> [MeetingFileTask] {
        do {
            guard fileManager.fileExists(atPath: taskFileURL.path) else { return [] }
            let data = try Data(contentsOf: taskFileURL)
            let payload = try JSONDecoder().decode(PersistedPayload.self, from: data)
            guard payload.version == 1 || payload.version == 2 else { return [] }
            return payload.tasks.filter { task in
                // Persisted paths must stay inside this queue's storage directory.
                [task.stagedFileName, task.legacyStagedFileName].compactMap { $0 }.allSatisfy {
                    !$0.isEmpty && $0 != "." && $0 != ".." && URL(fileURLWithPath: $0).lastPathComponent == $0
                }
            }.map { task in
                let destinationName = task.stagedFileName.hasSuffix(".prepared.wav")
                    ? task.stagedFileName : task.stagedFileName + ".prepared.wav"
                try? fileManager.removeItem(at: storageDirectoryURL.appendingPathComponent(destinationName).appendingPathExtension("partial"))
                guard task.status == .processing || task.status == .waitingForResources || task.status == .cancelling || task.status == .preparing else { return task }
                var restored = task.resetForRetry()
                restored.errorMessage = AppLocalization.localizedString("The task was interrupted and has been queued again.")
                return restored
            }
        } catch {
            return []
        }
    }

    func persist(_ tasks: [MeetingFileTask]) {
        persistenceCoordinator.scheduleWrite(PersistedPayload(version: 2, tasks: tasks), to: taskFileURL)
    }

    func flush(_ tasks: [MeetingFileTask]) {
        persistenceCoordinator.flushWrite(PersistedPayload(version: 2, tasks: tasks), to: taskFileURL)
    }

    func stagedURL(for task: MeetingFileTask) -> URL {
        storageDirectoryURL.appendingPathComponent(task.stagedFileName)
    }

    func restoreStagingReservations() {
        // Include orphaned complete caches and legacy originals in the quota;
        // a lost metadata write must not make their disk usage disappear.
        reservedStagingBytes = 0
        let urls: [URL]
        do {
            urls = try fileManager.contentsOfDirectory(
                at: storageDirectoryURL, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]
            )
        } catch {
            // Fail closed if an existing directory's usage cannot be counted.
            reservedStagingBytes = fileManager.fileExists(atPath: storageDirectoryURL.path) ? .max : 0
            return
        }
        for url in urls where url.lastPathComponent != taskFileURL.lastPathComponent {
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]) else {
                reservedStagingBytes = .max
                return
            }
            guard values.isRegularFile == true else { continue }
            guard let size = values.fileSize, size >= 0 else {
                reservedStagingBytes = .max
                return
            }
            let (sum, overflow) = reservedStagingBytes.addingReportingOverflow(Int64(size))
            reservedStagingBytes = overflow ? Int64.max : sum
        }
    }

    func removeAbandonedPartialFiles() {
        let urls = (try? fileManager.contentsOfDirectory(at: storageDirectoryURL, includingPropertiesForKeys: nil)) ?? []
        for url in urls {
            let name = url.lastPathComponent
            guard name.hasSuffix(".prepared.wav.partial"),
                  UUID(uuidString: String(name.prefix(36))) != nil else { continue }
            try? fileManager.removeItem(at: url)
        }
    }

    private static func defaultStorageDirectoryURL(fileManager: FileManager) -> URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("meeting-file-tasks", isDirectory: true)
    }
}
