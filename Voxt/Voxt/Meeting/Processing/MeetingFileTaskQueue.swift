// MeetingFileTaskQueue.swift
// Provides persistent, serial processing for imported meeting files.

import Combine
import Foundation

@MainActor
final class MeetingFileTaskQueue: ObservableObject {
    typealias Analyzer = @MainActor @Sendable (
        _ sourceURL: URL,
        _ originalFileName: String,
        _ progress: @escaping @MainActor @Sendable (MeetingFileAnalysisProgress) -> Void
    ) async throws -> TranscriptionHistoryEntry
    typealias Preparer = @Sendable (
        _ sourceURL: URL,
        _ destinationURL: URL,
        _ limits: MeetingFilePreparationLimits,
        _ checkpoint: @escaping @Sendable () async throws -> Void,
        _ progress: @escaping @Sendable (Double) async -> Void
    ) async throws -> MeetingImportedAudioFile
    typealias ActiveAnalysisCanceller = @MainActor @Sendable () async -> Void
    typealias CanStartProvider = @MainActor @Sendable () -> Bool
    typealias AnalysisRollback = @MainActor @Sendable (TranscriptionHistoryEntry) -> Void

    @Published private(set) var tasks: [MeetingFileTask]

    private let analyzer: Analyzer
    private let preparer: Preparer
    private let cancelActiveAnalysis: ActiveAnalysisCanceller
    private let canStart: CanStartProvider
    private let rollbackAnalysis: AnalysisRollback
    private let onAnalysisCompleted: @MainActor (UUID, TranscriptionHistoryEntry) -> Void
    private let onTaskRemoved: @MainActor (UUID) -> Void
    private let store: MeetingFileTaskStore
    private var activePreparationID: UUID?
    private var workerTask: Task<Void, Never>?
    private var tickerTask: Task<Void, Never>?
    private var activeTaskID: UUID?
    private var stagingTaskIDs: Set<UUID> = []
    private var stagingTasks: [UUID: Task<Void, Never>] = [:]
    private var reservedStagingBytes: Int64 = 0
    private var isShuttingDown = false
    private var resourceWaitNotificationToken: NSObjectProtocol?

    init(
        analyzer: @escaping Analyzer,
        cancelActiveAnalysis: @escaping ActiveAnalysisCanceller,
        canStart: @escaping CanStartProvider,
        rollbackAnalysis: @escaping AnalysisRollback = { _ in },
        preparer: @escaping Preparer = { source, destination, limits, checkpoint, progress in
            try await MeetingImportedAudioFile.prepare(
                from: source, to: destination, limits: limits, checkpoint: checkpoint, progress: progress
            )
        },
        onAnalysisCompleted: @escaping @MainActor (UUID, TranscriptionHistoryEntry) -> Void = { _, _ in },
        onTaskRemoved: @escaping @MainActor (UUID) -> Void = { _ in },
        fileManager: FileManager = .default,
        storageDirectoryURL: URL? = nil,
        now: @escaping () -> Date = Date.init
    ) {
        self.analyzer = analyzer
        self.preparer = preparer
        self.cancelActiveAnalysis = cancelActiveAnalysis
        self.canStart = canStart
        self.rollbackAnalysis = rollbackAnalysis
        self.onAnalysisCompleted = onAnalysisCompleted
        self.onTaskRemoved = onTaskRemoved
        self.store = MeetingFileTaskStore(fileManager: fileManager, storageDirectoryURL: storageDirectoryURL, now: now)
        self.tasks = store.loadTasks()
        store.restoreStagingReservations()
        store.persist(tasks)
        resourceWaitNotificationToken = NotificationCenter.default.addObserver(
            forName: .voxtMeetingFileResourceWaitDidChange,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            Task { @MainActor [weak self] in
                self?.handleResourceWaitNotification(notification)
            }
        }
        store.removeAbandonedPartialFiles()
    }

    var hasActiveTasks: Bool {
        tasks.contains { !$0.isTerminal }
    }

    var hasFinishedTasks: Bool {
        tasks.contains(where: \.isTerminal)
    }

    func task(id: UUID) -> MeetingFileTask? {
        tasks.first { $0.id == id }
    }

    /// An unfinished speaker pass must not hide an already completed ASR result.
    /// Read the durable checkpoint, never a partial in-flight transcript array.
    func completedTranscriptSegments(taskID: UUID) async -> [MeetingTranscriptSegment]? {
        guard task(id: taskID) != nil,
              let checkpoint = await MeetingFileAnalysisCheckpointStore.shared.load(taskID: taskID),
              checkpoint.taskID == taskID,
              checkpoint.completedDescriptorCount == checkpoint.descriptorCount,
              task(id: taskID) != nil else { return nil }
        let segments = MeetingTranscriptPostProcessor.process(checkpoint.segments)
        return segments.isEmpty ? nil : segments
    }

    func enqueue(urls: [URL]) {
        guard !isShuttingDown else { return }

        for sourceURL in urls {
            guard MeetingFileImportSupport.isSupportedImportFile(at: sourceURL) else { continue }

            guard tasks.count < MeetingFileTaskStore.maximumTaskCount else {
                NotificationCenter.default.post(
                    name: .voxtFeatureSettingsToastRequested,
                    object: nil,
                    userInfo: ["message": AppLocalization.localizedString("The file task queue is full. Clear finished tasks before adding more files.")]
                )
                break
            }
            let taskID = UUID()
            let fileName = sourceURL.lastPathComponent
            let stagedFileName = taskID.uuidString + "-" + fileName + ".prepared.wav"
            tasks.append(
                .queued(
                    id: taskID,
                    fileName: fileName,
                    stagedFileName: stagedFileName,
                    enqueuedAt: store.now()
                )
            )
            stagingTaskIDs.insert(taskID)
            stage(sourceURL: sourceURL, taskID: taskID, stagedFileName: stagedFileName)
        }

        persist()
        startIfNeeded()
    }

    func cancel(taskID: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        guard !tasks[index].isTerminal else { return }
        if activeTaskID == taskID,
           tasks[index].status == .processing || tasks[index].status == .waitingForResources {
            tasks[index].status = .cancelling
            persist()
            Task { @MainActor [weak self] in
                guard let self, self.activeTaskID == taskID,
                      self.task(id: taskID)?.status == .cancelling else { return }
                await self.cancelActiveAnalysis()
            }
            return
        }

        if stagingTaskIDs.contains(taskID) {
            stagingTasks[taskID]?.cancel()
            if activePreparationID == taskID {
                tasks[index].status = .cancelling
                persist()
                return
            }
        }
        tasks[index].status = .cancelled
        tasks[index].completedAt = store.now()
        persist()
        startIfNeeded()
    }

    /// Moves a queued task ahead of the other queued tasks while keeping any
    /// currently processing task in place.
    func prioritize(taskID: UUID) {
        guard let taskIndex = tasks.firstIndex(where: { $0.id == taskID }),
              tasks[taskIndex].status == .queued,
              let firstQueuedIndex = tasks.firstIndex(where: { $0.status == .queued }),
              taskIndex != firstQueuedIndex
        else { return }

        let task = tasks.remove(at: taskIndex)
        let insertionIndex = tasks.firstIndex(where: { $0.status == .queued }) ?? tasks.endIndex
        tasks.insert(task, at: insertionIndex)
        persist()
        startIfNeeded()
    }

    func retry(taskID: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        guard tasks[index].status == .failed || tasks[index].status == .cancelled else { return }
        guard !stagingTaskIDs.contains(taskID) else { return }
        guard store.fileManager.fileExists(atPath: stagedURL(for: tasks[index]).path) else {
            tasks[index].status = .failed
            tasks[index].errorMessage = AppLocalization.localizedString("The staged source file is no longer available.")
            persist()
            return
        }

        tasks[index] = tasks[index].resetForRetry()
        persist()
        startIfNeeded()
    }

    func clearFinishedTasks() {
        let finishedTasks = tasks.filter(\.isTerminal)
        tasks.removeAll(where: \.isTerminal)
        for task in finishedTasks {
            removeFinishedTaskArtifacts(task)
        }
        store.restoreStagingReservations()
        persist()
    }

    func removeFinishedTask(taskID: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }),
              tasks[index].isTerminal else { return }
        let task = tasks.remove(at: index)
        removeFinishedTaskArtifacts(task)
        store.restoreStagingReservations()
        persist()
    }

    func startIfNeeded() {
        guard !isShuttingDown else { return }
        // Lazily upgrade raw files from v1 queues through the same serial lane.
        if stagingTaskIDs.isEmpty,
           let task = tasks.first(where: { !$0.isTerminal }),
           task.status == .queued, task.preparedAudioVersion != 1 {
            stagingTaskIDs.insert(task.id)
            stage(sourceURL: stagedURL(for: task), taskID: task.id, stagedFileName: task.stagedFileName)
        }
        guard workerTask == nil, tasks.contains(where: { !$0.isTerminal }) else { return }
        workerTask = Task { @MainActor [weak self] in
            await self?.runWorker()
        }
        if tickerTask == nil {
            tickerTask = Task { @MainActor [weak self] in
                await self?.runTicker()
            }
        }
    }

    func shutdown() async {
        if let resourceWaitNotificationToken {
            NotificationCenter.default.removeObserver(resourceWaitNotificationToken)
            self.resourceWaitNotificationToken = nil
        }
        isShuttingDown = true
        tickerTask?.cancel()
        tickerTask = nil
        if activeTaskID != nil {
            await cancelActiveAnalysis()
        }
        let stagingTaskIDsToCancel = Array(stagingTasks.keys)
        for taskID in stagingTaskIDsToCancel {
            if let index = tasks.firstIndex(where: { $0.id == taskID }), !tasks[index].isTerminal {
                // Do not allow Clear Finished Tasks to race a decoder/writer.
                tasks[index].status = .cancelling
            }
            stagingTasks[taskID]?.cancel()
        }
        persist()
        if let workerTask {
            await workerTask.value
        }
        workerTask = nil
        let tasksToFinish = Array(stagingTasks.values)
        for stagingTask in tasksToFinish {
            await stagingTask.value
        }
        stagingTasks.removeAll()
        store.flush(tasks)
    }

    private func runWorker() async {
        defer { workerTask = nil }

        while !Task.isCancelled, !isShuttingDown {
            startIfNeeded()
            guard let index = nextRunnableTaskIndex() else {
                guard tasks.contains(where: { !$0.isTerminal }) else { return }
                try? await Task.sleep(for: .milliseconds(200))
                continue
            }
            let taskID = tasks[index].id

            while (!canStart() || activePreparationID != nil), !Task.isCancelled, !isShuttingDown {
                try? await Task.sleep(for: .seconds(1))
            }
            guard !Task.isCancelled, !isShuttingDown else { return }

            guard let queuedTask = task(id: taskID), queuedTask.status == .queued else { continue }
            let preparedURL = stagedURL(for: queuedTask)
            do {
                _ = try await Task.detached(priority: .utility) {
                    try MeetingImportedAudioFile.validatedPreparedFile(at: preparedURL)
                }.value
            } catch {
                if !isShuttingDown, task(id: taskID)?.status == .queued {
                    markFailed(taskID: taskID, error: error)
                    persist()
                }
                continue
            }
            guard !isShuttingDown, !Task.isCancelled else { return }
            guard canStart(), activePreparationID == nil,
                  tasks.first(where: { !$0.isTerminal })?.id == taskID else { continue }
            guard let currentIndex = tasks.firstIndex(where: { $0.id == taskID }),
                  tasks[currentIndex].status == .queued
            else { continue }

            let startDate = store.now()
            tasks[currentIndex].status = .processing
            tasks[currentIndex].startedAt = tasks[currentIndex].startedAt ?? startDate
            tasks[currentIndex].completedAt = nil
            tasks[currentIndex].progressStage = .transcribing
            tasks[currentIndex].status = .processing
            tasks[currentIndex].progressFraction = 0.15
            tasks[currentIndex].processedMediaDurationSeconds = nil
            tasks[currentIndex].processingSpeedSecondsPerSecond = nil
            tasks[currentIndex].speedSampleAt = nil
            tasks[currentIndex].speedSampleProcessedMediaDurationSeconds = nil
            tasks[currentIndex].estimatedTotalSeconds = nil
            tasks[currentIndex].errorMessage = nil
            activeTaskID = taskID
            VoxtLog.meeting("File task analysis started. taskID=\(taskID)")
            persist()

            do {
                let task = tasks[currentIndex]
                let entry = try await MeetingFileTaskContext.$taskID.withValue(taskID) {
                    try await analyzer(stagedURL(for: task), task.fileName) { [weak self] progress in
                        guard let self else { return }
                        self.apply(progress: progress, to: taskID)
                    }
                }
                guard let finishedIndex = tasks.firstIndex(where: { $0.id == taskID }) else { continue }
                let shouldRollback = isShuttingDown || tasks[finishedIndex].status == .cancelling
                if shouldRollback {
                    rollbackAnalysis(entry)
                }
                if isShuttingDown {
                    markInterrupted(taskID: taskID)
                } else if tasks[finishedIndex].status == .cancelling {
                    tasks[finishedIndex].status = .cancelled
                    tasks[finishedIndex].completedAt = store.now()
                } else {
                    // Publish completion before checkpoint cleanup can suspend.
                    // Preview readers can now resolve the durable history entry.
                    tasks[finishedIndex].status = .completed
                    tasks[finishedIndex].completedAt = store.now()
                    tasks[finishedIndex].progressStage = .saving
                    tasks[finishedIndex].progressFraction = 1
                    tasks[finishedIndex].historyEntryID = entry.id
                    VoxtLog.meeting("File task completed. taskID=\(taskID), stage=\(tasks[finishedIndex].progressStage.diagnosticName)")
                    SystemNotificationSupport.post(
                        title: AppLocalization.localizedString("File conversion completed"),
                        body: AppLocalization.format("%@ has been converted successfully.", tasks[finishedIndex].fileName),
                        userInfo: [
                            "fileTaskID": taskID.uuidString,
                            "historyEntryID": entry.id.uuidString
                        ]
                    )
                    onAnalysisCompleted(taskID, entry)
                    await MeetingFileAnalysisCheckpointStore.shared.clear(taskID: taskID)
                }
            } catch is CancellationError {
                if isShuttingDown {
                    markInterrupted(taskID: taskID)
                } else {
                    markCancelled(taskID: taskID)
                }
            } catch {
                markFailed(taskID: taskID, error: error)
            }

            activeTaskID = nil
            persist()
        }
    }

    private func runTicker() async {
        defer { tickerTask = nil }
        while !Task.isCancelled, !isShuttingDown {
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled else { return }
            guard hasActiveTasks else { return }
            objectWillChange.send()
        }
    }

    private func nextRunnableTaskIndex() -> Int? {
        guard let index = tasks.firstIndex(where: { !$0.isTerminal }) else { return nil }
        let task = tasks[index]
        guard task.status == .queued, task.preparedAudioVersion == 1,
              !stagingTaskIDs.contains(task.id) else { return nil }
        guard store.fileManager.fileExists(atPath: stagedURL(for: task).path) else {
            markFailed(
                taskID: task.id,
                error: NSError(
                    domain: "Voxt.MeetingFileTaskQueue",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("The staged source file is no longer available.")]
                )
            )
            return nextRunnableTaskIndex()
        }
        return index
    }

    private func apply(progress: MeetingFileAnalysisProgress, to taskID: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }),
              tasks[index].status == .processing || tasks[index].status == .preparing
        else { return }
        let sampleDate = store.now()
        let previousStage = tasks[index].progressStage
        tasks[index].progressStage = progress.stage
        tasks[index].progressFraction = min(max(progress.fractionCompleted, 0), 1)
        if let mediaDurationSeconds = progress.mediaDurationSeconds {
            tasks[index].mediaDurationSeconds = mediaDurationSeconds
        }
        if let processedMediaDurationSeconds = progress.processedMediaDurationSeconds {
            tasks[index].processedMediaDurationSeconds = processedMediaDurationSeconds
        }
        if previousStage != progress.stage {
            VoxtLog.meeting("File task stage changed. taskID=\(taskID), stage=\(progress.stage.diagnosticName)")
        }
        updateProcessingSpeed(for: &tasks[index], at: sampleDate)
        tasks[index].estimatedTotalSeconds = MeetingFileTask.updatedEstimatedTotalSeconds(
            current: tasks[index].estimatedTotalSeconds,
            elapsed: tasks[index].elapsedSeconds(now: store.now()),
            progressFraction: tasks[index].progressFraction,
            mediaDurationSeconds: tasks[index].mediaDurationSeconds,
            processedMediaDurationSeconds: tasks[index].processedMediaDurationSeconds,
            stage: progress.stage,
            processingSpeed: tasks[index].processingSpeedSecondsPerSecond
        )
        persist()
    }

    private func updateProcessingSpeed(for task: inout MeetingFileTask, at sampleDate: Date) {
        guard task.progressStage == .transcribing,
              let processedDuration = task.processedMediaDurationSeconds,
              processedDuration > 0
        else { return }

        if let previousDate = task.speedSampleAt,
           let previousProcessedDuration = task.speedSampleProcessedMediaDurationSeconds {
            let wallDuration = sampleDate.timeIntervalSince(previousDate)
            let audioDuration = processedDuration - previousProcessedDuration
            if wallDuration > 0, audioDuration > 0 {
                let instantaneousSpeed = audioDuration / wallDuration
                if let existingSpeed = task.processingSpeedSecondsPerSecond {
                    task.processingSpeedSecondsPerSecond = existingSpeed * 0.7 + instantaneousSpeed * 0.3
                } else {
                    task.processingSpeedSecondsPerSecond = instantaneousSpeed
                }
            }
        }

        task.speedSampleAt = sampleDate
        task.speedSampleProcessedMediaDurationSeconds = processedDuration
    }

    private func handleResourceWaitNotification(_ notification: Notification) {
        guard let rawTaskID = notification.userInfo?["taskID"] as? String,
              let taskID = UUID(uuidString: rawTaskID),
              let index = tasks.firstIndex(where: { $0.id == taskID }),
              activeTaskID == taskID
        else { return }
        let isWaiting = notification.userInfo?["isWaiting"] as? Bool ?? false
        guard !tasks[index].isTerminal else { return }
        if isWaiting, tasks[index].status == .processing {
            tasks[index].status = .waitingForResources
            persist()
        } else if !isWaiting, tasks[index].status == .waitingForResources {
            tasks[index].status = .processing
            persist()
        }
    }

    private func markCancelled(taskID: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        tasks[index].status = .cancelled
        tasks[index].completedAt = store.now()
        VoxtLog.meeting("File task cancelled. taskID=\(taskID), stage=\(tasks[index].progressStage.diagnosticName)")
    }

    private func markInterrupted(taskID: UUID) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        var task = tasks[index].resetForRetry()
        task.errorMessage = AppLocalization.localizedString("The task was interrupted and has been queued again.")
        tasks[index] = task
    }

    private func markFailed(taskID: UUID, error: Error) {
        guard let index = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        tasks[index].status = .failed
        tasks[index].completedAt = store.now()
        tasks[index].errorMessage = error.localizedDescription
        VoxtLog.meetingError(
            "File task failed. taskID=\(taskID), stage=\(tasks[index].progressStage.diagnosticName), \(MeetingFileTaskDiagnostics.errorSummary(error))"
        )
        SystemNotificationSupport.post(
            title: AppLocalization.localizedString("File conversion failed"),
            body: AppLocalization.format("%@: %@", tasks[index].fileName, error.localizedDescription)
        )
    }

    private func removeFinishedTaskArtifacts(_ task: MeetingFileTask) {
        onTaskRemoved(task.id)
        Task { await MeetingFileAnalysisCheckpointStore.shared.clear(taskID: task.id) }
        try? store.fileManager.removeItem(at: stagedURL(for: task))
        if let legacyName = task.legacyStagedFileName {
            try? store.fileManager.removeItem(at: store.storageDirectoryURL.appendingPathComponent(legacyName))
        }
    }

    private func stage(sourceURL: URL, taskID: UUID, stagedFileName: String) {
        // Retain the provider URL and scope, but read no audio until admission.
        let didStartAccessing = sourceURL.startAccessingSecurityScopedResource()
        let preparedName = stagedFileName.hasSuffix(".prepared.wav")
            ? stagedFileName : stagedFileName + ".prepared.wav"
        let destinationURL = store.storageDirectoryURL.appendingPathComponent(preparedName)
        let preparer = self.preparer
        let fileManager = store.fileManager

        let stagingTask = Task { @MainActor [weak self] in
            var createdDestination = false
            defer {
                if didStartAccessing { sourceURL.stopAccessingSecurityScopedResource() }
                if let self {
                    self.stagingTasks.removeValue(forKey: taskID)
                    self.stagingTaskIDs.remove(taskID)
                    if self.activePreparationID == taskID { self.activePreparationID = nil }
                    self.store.restoreStagingReservations()
                    self.persist()
                    self.startIfNeeded()
                }
            }
            do {
                guard let self else { throw CancellationError() }
                while true {
                    try Task.checkCancellation()
                    guard !self.isShuttingDown, let task = self.task(id: taskID), !task.isTerminal else {
                        throw CancellationError()
                    }
                    if self.activePreparationID == nil, self.activeTaskID == nil,
                       self.tasks.first(where: { !$0.isTerminal })?.id == taskID,
                       self.canStart() { break }
                    try await Task.sleep(for: .milliseconds(200))
                }
                self.activePreparationID = taskID
                guard let index = self.tasks.firstIndex(where: { $0.id == taskID }) else { throw CancellationError() }
                self.tasks[index].status = .preparing
                self.tasks[index].startedAt = self.store.now()
                self.tasks[index].progressStage = .preparing
                self.tasks[index].progressFraction = 0
                self.persist()
                try fileManager.createDirectory(at: self.store.storageDirectoryURL, withIntermediateDirectories: true)
                self.store.restoreStagingReservations()
                let remainingBytes = max(0, MeetingFileTaskStore.maximumStagedSourceBytes - self.store.reservedStagingBytes)
                var limits = MeetingFilePreparationLimits()
                limits.maximumOutputBytes = min(limits.maximumOutputBytes, remainingBytes)
                let preparationLimits = limits
                createdDestination = !fileManager.fileExists(atPath: destinationURL.path)
                let shouldPrepare = createdDestination
                let preparationTask = Task.detached(priority: .utility) { [weak self] in
                    try await MeetingFileTaskContext.$taskID.withValue(taskID) {
                        if !shouldPrepare {
                            // Recover a renamed cache even if metadata wasn't flushed.
                            return try MeetingImportedAudioFile.validatedPreparedFile(at: destinationURL)
                        }
                        let before = try fileManager.attributesOfItem(atPath: sourceURL.path)
                        guard before[.type] as? FileAttributeType == .typeRegular else {
                            throw MeetingFileTaskStagingError.sourceUnavailable
                        }
                        let audio = try await preparer(sourceURL, destinationURL, preparationLimits, {
                            guard let self else { throw CancellationError() }
                            try await self.waitForPreparationAvailability()
                        }, { fraction in
                            await self?.apply(
                                progress: MeetingFileAnalysisProgress(stage: .preparing, stageFraction: fraction),
                                to: taskID
                            )
                        })
                        let after = try fileManager.attributesOfItem(atPath: sourceURL.path)
                        guard (before[.size] as? NSNumber) == (after[.size] as? NSNumber),
                              (before[.modificationDate] as? Date) == (after[.modificationDate] as? Date),
                              (before[.systemFileNumber] as? NSNumber) == (after[.systemFileNumber] as? NSNumber),
                              (before[.systemNumber] as? NSNumber) == (after[.systemNumber] as? NSNumber) else {
                            throw MeetingFileTaskStagingError.sourceUnavailable
                        }
                        return audio
                    }
                }
                let audio = try await withTaskCancellationHandler {
                    try await preparationTask.value
                } onCancel: { preparationTask.cancel() }
                try Task.checkCancellation()
                guard let updatedIndex = self.tasks.firstIndex(where: { $0.id == taskID }) else { throw CancellationError() }
                if preparedName != stagedFileName {
                    self.tasks[updatedIndex].legacyStagedFileName = stagedFileName
                }
                self.tasks[updatedIndex].stagedFileName = preparedName
                self.tasks[updatedIndex].preparedAudioVersion = 1
                self.tasks[updatedIndex].mediaDurationSeconds = audio.durationSeconds
                self.tasks[updatedIndex].progressFraction = 0.15
                self.tasks[updatedIndex].status = .queued
            } catch {
                if createdDestination {
                    try? fileManager.removeItem(at: destinationURL)
                    try? fileManager.removeItem(at: destinationURL.appendingPathExtension("partial"))
                }
                if let self, self.isShuttingDown,
                   let index = self.tasks.firstIndex(where: { $0.id == taskID }) {
                    self.tasks[index].status = .failed
                    self.tasks[index].completedAt = self.store.now()
                    self.tasks[index].errorMessage = AppLocalization.localizedString("The meeting file could not be staged before the app closed.")
                } else if error is CancellationError {
                    self?.markCancelled(taskID: taskID)
                } else {
                    self?.markFailed(taskID: taskID, error: error)
                }
            }
        }
        stagingTasks[taskID] = stagingTask
    }

    private func waitForPreparationAvailability() async throws {
        while !canStart() {
            try Task.checkCancellation()
            guard !isShuttingDown else { throw CancellationError() }
            try await Task.sleep(for: .milliseconds(250))
        }
        try Task.checkCancellation()
        guard !isShuttingDown else { throw CancellationError() }
    }

    private func stagedURL(for task: MeetingFileTask) -> URL {
        store.stagedURL(for: task)
    }

    private func persist() {
        store.persist(tasks)
    }
}
