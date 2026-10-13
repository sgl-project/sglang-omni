import Foundation
import XCTest
@testable import Voxt

@MainActor
final class MeetingFileTaskQueueTests: XCTestCase {
    func testShutdownFlushesLatestTaskStateToDisk() async throws {
        let storage = try TemporaryDirectory()
        let sourceURL = try makeSourceFile(named: "shutdown.wav")
        let taskFileURL = storage.url
            .appendingPathComponent("tasks", isDirectory: true)
            .appendingPathComponent("tasks.json")
        var cancelRequested = false

        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in
                while !cancelRequested {
                    try await Task.sleep(for: .milliseconds(10))
                }
                throw CancellationError()
            },
            cancelActiveAnalysis: {
                cancelRequested = true
            },
            canStart: { true },
            storageDirectoryURL: storage.url.appendingPathComponent("tasks", isDirectory: true)
        )

        queue.enqueue(urls: [sourceURL])
        try await waitUntil(queue, status: .processing, at: 0)
        await queue.shutdown()

        struct PersistedPayload: Decodable {
            let tasks: [MeetingFileTask]
        }
        let payload = try JSONDecoder().decode(
            PersistedPayload.self,
            from: Data(contentsOf: taskFileURL)
        )
        let persistedTask = try XCTUnwrap(payload.tasks.first)
        XCTAssertEqual(persistedTask.status, .queued)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: storage.url
                    .appendingPathComponent("tasks", isDirectory: true)
                    .appendingPathComponent(persistedTask.stagedFileName)
                    .path
            )
        )
    }

    func testQueueProcessesFilesInFIFOOrderWithOnlyOneActiveAnalyzer() async throws {
        let storage = try TemporaryDirectory()
        let firstURL = try makeSourceFile(named: "first.wav")
        let secondURL = try makeSourceFile(named: "second.wav")
        var processedNames: [String] = []
        var activeAnalyses = 0
        var maximumActiveAnalyses = 0
        var completedTaskIDs: [UUID] = []
        var removedTaskIDs: [UUID] = []

        let queue = MeetingFileTaskQueue(
            analyzer: { sourceURL, _, progress in
                activeAnalyses += 1
                maximumActiveAnalyses = max(maximumActiveAnalyses, activeAnalyses)
                processedNames.append(sourceURL.path.contains("first.wav") ? "first" : "second")
                progress(MeetingFileAnalysisProgress(stage: .transcribing, stageFraction: 0.5))
                try await Task.sleep(for: .milliseconds(30))
                progress(MeetingFileAnalysisProgress(stage: .saving, stageFraction: 1))
                activeAnalyses -= 1
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {},
            canStart: { true },
            onAnalysisCompleted: { taskID, _ in completedTaskIDs.append(taskID) },
            onTaskRemoved: { removedTaskIDs.append($0) },
            storageDirectoryURL: storage.url.appendingPathComponent("tasks", isDirectory: true)
        )

        queue.enqueue(urls: [firstURL, secondURL])
        try await waitUntilAllTasksAreTerminal(queue)

        XCTAssertEqual(processedNames, ["first", "second"])
        XCTAssertEqual(maximumActiveAnalyses, 1)
        XCTAssertEqual(queue.tasks.map(\.status), [.completed, .completed])
        XCTAssertTrue(queue.tasks.allSatisfy { $0.historyEntryID != nil })
        XCTAssertEqual(completedTaskIDs, queue.tasks.map(\.id))
        queue.clearFinishedTasks()
        XCTAssertEqual(removedTaskIDs, completedTaskIDs)
        await queue.shutdown()
    }

    func testRemoveFinishedTaskRemovesOnlyTheRequestedTask() async throws {
        let storage = try TemporaryDirectory()
        let firstURL = try makeSourceFile(named: "remove-first.wav")
        let secondURL = try makeSourceFile(named: "remove-second.wav")
        var removedTaskIDs: [UUID] = []

        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in Self.makeHistoryEntry() },
            cancelActiveAnalysis: {},
            canStart: { true },
            onTaskRemoved: { removedTaskIDs.append($0) },
            storageDirectoryURL: storage.url.appendingPathComponent("tasks", isDirectory: true)
        )

        queue.enqueue(urls: [firstURL, secondURL])
        try await waitUntilAllTasksAreTerminal(queue)
        let firstTaskID = try XCTUnwrap(queue.tasks.first?.id)
        let secondTaskID = try XCTUnwrap(queue.tasks.last?.id)

        queue.removeFinishedTask(taskID: firstTaskID)

        XCTAssertNil(queue.task(id: firstTaskID))
        XCTAssertEqual(queue.task(id: secondTaskID)?.status, .completed)
        XCTAssertEqual(removedTaskIDs, [firstTaskID])
        await queue.shutdown()
    }

    func testCancellingCurrentTaskContinuesWithNextQueuedTask() async throws {
        let storage = try TemporaryDirectory()
        let firstURL = try makeSourceFile(named: "cancel-me.wav")
        let secondURL = try makeSourceFile(named: "continue.wav")
        var cancelRequested = false

        let queue = MeetingFileTaskQueue(
            analyzer: { sourceURL, _, progress in
                if sourceURL.path.contains("cancel-me.wav") {
                    while !cancelRequested {
                        try await Task.sleep(for: .milliseconds(10))
                    }
                    throw CancellationError()
                }
                progress(MeetingFileAnalysisProgress(stage: .saving, stageFraction: 1))
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {
                cancelRequested = true
            },
            canStart: { true },
            storageDirectoryURL: storage.url.appendingPathComponent("tasks", isDirectory: true)
        )

        queue.enqueue(urls: [firstURL, secondURL])
        try await waitUntil(queue, status: .processing, at: 0)
        queue.cancel(taskID: try XCTUnwrap(queue.tasks.first?.id))
        try await waitUntilAllTasksAreTerminal(queue)

        XCTAssertEqual(queue.tasks.map(\.status), [.cancelled, .completed])
        await queue.shutdown()
    }

    func testCancellingAfterAnalyzerPersistedResultRollsBackHistoryEntry() async throws {
        let storage = try TemporaryDirectory()
        let sourceURL = try makeSourceFile(named: "persisted-then-cancelled.wav")
        var cancelRequested = false
        var rolledBackEntryIDs: [UUID] = []
        let persistedEntry = Self.makeHistoryEntry()

        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in
                while !cancelRequested {
                    try await Task.sleep(for: .milliseconds(10))
                }
                return persistedEntry
            },
            cancelActiveAnalysis: {
                cancelRequested = true
            },
            canStart: { true },
            rollbackAnalysis: { entry in
                rolledBackEntryIDs.append(entry.id)
            },
            storageDirectoryURL: storage.url.appendingPathComponent("tasks", isDirectory: true)
        )

        queue.enqueue(urls: [sourceURL])
        try await waitUntil(queue, status: .processing, at: 0)
        queue.cancel(taskID: try XCTUnwrap(queue.tasks.first?.id))
        try await waitUntilAllTasksAreTerminal(queue)

        XCTAssertEqual(queue.tasks.first?.status, .cancelled)
        XCTAssertEqual(rolledBackEntryIDs, [persistedEntry.id])
        XCTAssertNil(queue.tasks.first?.historyEntryID)
        await queue.shutdown()
    }

    func testCancellingQueuedTaskDoesNotAffectTheActiveTask() async throws {
        let storage = try TemporaryDirectory()
        let firstURL = try makeSourceFile(named: "active.wav")
        let secondURL = try makeSourceFile(named: "queued-cancel.wav")
        var releaseActiveTask = false

        let queue = MeetingFileTaskQueue(
            analyzer: { sourceURL, _, progress in
                if sourceURL.path.contains("active.wav") {
                    while !releaseActiveTask {
                        try await Task.sleep(for: .milliseconds(10))
                    }
                }
                progress(MeetingFileAnalysisProgress(stage: .saving, stageFraction: 1))
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {},
            canStart: { true },
            storageDirectoryURL: storage.url.appendingPathComponent("tasks", isDirectory: true)
        )

        queue.enqueue(urls: [firstURL, secondURL])
        try await waitUntil(queue, status: .processing, at: 0)
        let queuedTaskID = try XCTUnwrap(queue.tasks.first(where: { $0.status == .queued })?.id)

        queue.cancel(taskID: queuedTaskID)

        XCTAssertEqual(queue.task(id: queuedTaskID)?.status, .cancelled)
        releaseActiveTask = true
        try await waitUntilAllTasksAreTerminal(queue)
        XCTAssertEqual(queue.tasks.map(\.status), [.completed, .cancelled])
        await queue.shutdown()
    }

    func testPrioritizingQueuedTaskMovesItAheadOfOtherQueuedTasks() async throws {
        let storage = try TemporaryDirectory()
        let firstURL = try makeSourceFile(named: "first-priority.wav")
        let secondURL = try makeSourceFile(named: "second-priority.wav")
        let priorityURL = try makeSourceFile(named: "priority.wav")
        var processedNames: [String] = []
        var releaseFirstTask = false

        let queue = MeetingFileTaskQueue(
            analyzer: { sourceURL, _, progress in
                if sourceURL.path.contains("first-priority.wav") {
                    while !releaseFirstTask {
                        try await Task.sleep(for: .milliseconds(10))
                    }
                }
                if sourceURL.path.contains("first-priority.wav") {
                    processedNames.append("first-priority.wav")
                } else if sourceURL.path.contains("second-priority.wav") {
                    processedNames.append("second-priority.wav")
                } else {
                    processedNames.append("priority.wav")
                }
                progress(MeetingFileAnalysisProgress(stage: .saving, stageFraction: 1))
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {},
            canStart: { true },
            storageDirectoryURL: storage.url.appendingPathComponent("tasks", isDirectory: true)
        )

        queue.enqueue(urls: [firstURL, secondURL, priorityURL])
        try await waitUntil(queue, status: .processing, at: 0)
        let priorityTaskID = try XCTUnwrap(
            queue.tasks.first(where: { $0.fileName == priorityURL.lastPathComponent })?.id
        )

        queue.prioritize(taskID: priorityTaskID)

        XCTAssertEqual(
            queue.tasks.map(\.fileName),
            [firstURL.lastPathComponent, priorityURL.lastPathComponent, secondURL.lastPathComponent]
        )
        releaseFirstTask = true
        try await waitUntilAllTasksAreTerminal(queue)
        XCTAssertEqual(
            processedNames,
            ["first-priority.wav", "priority.wav", "second-priority.wav"]
        )
        await queue.shutdown()
    }

    func testPreparationAndAnalysisNeverOverlapAndReceiveCanonicalAudio() async throws {
        let storage = try TemporaryDirectory()
        let first = try makeSourceFile(named: "prepare-first.wav")
        let second = try makeSourceFile(named: "prepare-second.wav")
        let probe = FilePreparationProbe()
        var originalNames: [String] = []
        let queue = MeetingFileTaskQueue(
            analyzer: { url, originalName, _ in
                XCTAssertNotNil(MeetingFileTaskContext.taskID)
                await probe.begin("analysis")
                originalNames.append(originalName)
                let audio = try MeetingImportedAudioFile.validatedPreparedFile(at: url)
                XCTAssertGreaterThan(audio.sampleCount, 0)
                try await Task.sleep(for: .milliseconds(20))
                await probe.end()
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {},
            canStart: { true },
            preparer: { source, destination, limits, checkpoint, progress in
                XCTAssertNotNil(MeetingFileTaskContext.taskID)
                await probe.begin("preparation")
                let audio = try await MeetingImportedAudioFile.prepare(
                    from: source, to: destination, limits: limits, checkpoint: checkpoint, progress: progress
                )
                await probe.end()
                return audio
            },
            storageDirectoryURL: storage.url
        )
        queue.enqueue(urls: [first, second])
        try await waitUntilAllTasksAreTerminal(queue)
        let maximumActive = await probe.maximumActive
        let events = await probe.events
        XCTAssertEqual(maximumActive, 1)
        XCTAssertEqual(events, ["preparation", "analysis", "preparation", "analysis"])
        XCTAssertEqual(originalNames, [first.lastPathComponent, second.lastPathComponent])
        XCTAssertTrue(queue.tasks.allSatisfy { $0.preparedAudioVersion == 1 && $0.status == .completed })
        await queue.shutdown()
    }

    func testRetryReusesPreparedAudioAfterOriginalWasRemoved() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "retry-cache.wav")
        let probe = FilePreparationProbe()
        var analyses = 0
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in
                analyses += 1
                if analyses == 1 { throw URLError(.cannotParseResponse) }
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {},
            canStart: { true },
            preparer: { source, destination, limits, checkpoint, progress in
                await probe.begin("preparation")
                let audio = try await MeetingImportedAudioFile.prepare(
                    from: source, to: destination, limits: limits, checkpoint: checkpoint, progress: progress
                )
                await probe.end()
                return audio
            },
            storageDirectoryURL: storage.url
        )
        queue.enqueue(urls: [source])
        try await waitUntil(queue, status: .failed, at: 0)
        try FileManager.default.removeItem(at: source)
        let taskID = try XCTUnwrap(queue.tasks.first?.id)
        queue.retry(taskID: taskID)
        try await waitUntil(queue, status: .completed, at: 0)
        let events = await probe.events
        XCTAssertEqual(events, ["preparation"])
        XCTAssertEqual(analyses, 2)
        let preparedURL = storage.url.appendingPathComponent(try XCTUnwrap(queue.tasks.first?.stagedFileName))
        XCTAssertTrue(FileManager.default.fileExists(atPath: preparedURL.path))
        queue.clearFinishedTasks()
        XCTAssertFalse(FileManager.default.fileExists(atPath: preparedURL.path))
        await queue.shutdown()
    }

    func testCancellationDuringPreparationNeverStartsAnalyzer() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "cancel-preparation.wav")
        let gate = ManualTaskBarrier()
        var analysisCount = 0
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in analysisCount += 1; return Self.makeHistoryEntry() },
            cancelActiveAnalysis: {},
            canStart: { true },
            preparer: { source, destination, limits, checkpoint, progress in
                await gate.wait()
                try Task.checkCancellation()
                return try await MeetingImportedAudioFile.prepare(
                    from: source, to: destination, limits: limits, checkpoint: checkpoint, progress: progress
                )
            },
            storageDirectoryURL: storage.url
        )
        queue.enqueue(urls: [source])
        await gate.waitUntilEntered()
        let task = try XCTUnwrap(queue.tasks.first)
        queue.cancel(taskID: task.id)
        XCTAssertEqual(queue.tasks.first?.status, .cancelling)
        queue.clearFinishedTasks()
        XCTAssertEqual(queue.tasks.count, 1)
        gate.release()
        try await waitUntil(queue, status: .cancelled, at: 0)
        XCTAssertEqual(analysisCount, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: storage.url.appendingPathComponent(task.stagedFileName).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
        await queue.shutdown()
    }

    func testLegacyRawQueueIsPreparedAndCacheSurvivesRelaunch() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "legacy.wav")
        let stagedName = "legacy-staged.wav"
        try FileManager.default.copyItem(at: source, to: storage.url.appendingPathComponent(stagedName))
        struct Payload: Encodable { let version: Int; let tasks: [MeetingFileTask] }
        let task = MeetingFileTask.queued(fileName: "legacy.wav", stagedFileName: stagedName)
        try JSONEncoder().encode(Payload(version: 1, tasks: [task])).write(to: storage.url.appendingPathComponent("tasks.json"))
        let firstQueue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in throw URLError(.cannotParseResponse) },
            cancelActiveAnalysis: {}, canStart: { true }, storageDirectoryURL: storage.url
        )
        firstQueue.startIfNeeded()
        try await waitUntil(firstQueue, status: .failed, at: 0)
        XCTAssertEqual(firstQueue.tasks.first?.preparedAudioVersion, 1)
        XCTAssertEqual(firstQueue.tasks.first?.legacyStagedFileName, stagedName)
        await firstQueue.shutdown()
        try FileManager.default.removeItem(at: source)
        try FileManager.default.removeItem(at: storage.url.appendingPathComponent(stagedName))
        let secondQueue = MeetingFileTaskQueue(
            analyzer: { _, originalName, _ in
                XCTAssertEqual(originalName, "legacy.wav")
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {}, canStart: { true },
            preparer: { _, _, _, _, _ in
                XCTFail("A completed cache must not be decoded again")
                throw URLError(.cannotDecodeContentData)
            },
            storageDirectoryURL: storage.url
        )
        secondQueue.retry(taskID: task.id)
        try await waitUntil(secondQueue, status: .completed, at: 0)
        await secondQueue.shutdown()
    }

    func testRelaunchRecoversPromotedAudioAndRemovesIncompleteSibling() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "recover.wav")
        let taskID = UUID()
        let name = "\(taskID.uuidString)-recover.wav.prepared.wav"
        let destination = storage.url.appendingPathComponent(name)
        _ = try await MeetingImportedAudioFile.prepare(from: source, to: destination)
        try FileManager.default.removeItem(at: source)
        let partial = destination.appendingPathExtension("partial")
        try Data([1, 2]).write(to: partial)
        var task = MeetingFileTask.queued(id: taskID, fileName: "recover.wav", stagedFileName: name)
        task.status = .preparing
        struct Payload: Encodable { let version: Int; let tasks: [MeetingFileTask] }
        try JSONEncoder().encode(Payload(version: 2, tasks: [task])).write(to: storage.url.appendingPathComponent("tasks.json"))
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in Self.makeHistoryEntry() },
            cancelActiveAnalysis: {}, canStart: { true },
            preparer: { _, _, _, _, _ in
                XCTFail("Promoted audio should be recovered without the original")
                throw URLError(.fileDoesNotExist)
            },
            storageDirectoryURL: storage.url
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        queue.startIfNeeded()
        try await waitUntil(queue, status: .completed, at: 0)
        XCTAssertEqual(queue.tasks.first?.preparedAudioVersion, 1)
        await queue.shutdown()
    }

    func testCorruptedPreparedCacheNeverReachesAnalyzer() async throws {
        let storage = try TemporaryDirectory()
        var task = MeetingFileTask.queued(fileName: "bad.wav", stagedFileName: "bad.prepared.wav")
        task.preparedAudioVersion = 1
        try Data([1, 2, 3]).write(to: storage.url.appendingPathComponent(task.stagedFileName))
        struct Payload: Encodable { let version: Int; let tasks: [MeetingFileTask] }
        try JSONEncoder().encode(Payload(version: 2, tasks: [task])).write(to: storage.url.appendingPathComponent("tasks.json"))
        var analyses = 0
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in analyses += 1; return Self.makeHistoryEntry() },
            cancelActiveAnalysis: {}, canStart: { true }, storageDirectoryURL: storage.url
        )
        queue.startIfNeeded()
        try await waitUntil(queue, status: .failed, at: 0)
        XCTAssertEqual(analyses, 0)
        await queue.shutdown()
    }

    func testQueueAdmissionBoundsPendingPreparations() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "bounded-queue.wav")
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in XCTFail("No analysis is admitted"); return Self.makeHistoryEntry() },
            cancelActiveAnalysis: {}, canStart: { false }, storageDirectoryURL: storage.url
        )
        queue.enqueue(urls: Array(repeating: source, count: 70))
        XCTAssertEqual(queue.tasks.count, 64)
        XCTAssertTrue(queue.tasks.allSatisfy { $0.status == .queued })
        await queue.shutdown()
        XCTAssertTrue(queue.tasks.allSatisfy(\.isTerminal))
    }

    func testResourceWaitNotificationUpdatesTaskStatusWithoutLosingProgress() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "resource-wait.wav")
        let gate = ManualTaskBarrier()
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in
                await gate.wait()
                return Self.makeHistoryEntry()
            },
            cancelActiveAnalysis: {}, canStart: { true }, storageDirectoryURL: storage.url
        )
        queue.enqueue(urls: [source])
        try await waitUntil(queue, status: .processing, at: 0)
        let taskID = try XCTUnwrap(queue.tasks.first?.id)
        NotificationCenter.default.post(
            name: .voxtMeetingFileResourceWaitDidChange,
            object: nil,
            userInfo: ["taskID": taskID.uuidString, "isWaiting": true]
        )
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(queue.task(id: taskID)?.status, .waitingForResources)

        NotificationCenter.default.post(
            name: .voxtMeetingFileResourceWaitDidChange,
            object: nil,
            userInfo: ["taskID": taskID.uuidString, "isWaiting": false]
        )
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(queue.task(id: taskID)?.status, .processing)
        gate.release()
        try await waitUntil(queue, status: .completed, at: 0)
        await queue.shutdown()
    }

    func testFailedAnalysisPreservesStageAndCauseForDiagnosis() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "failed-transcription.wav")
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, progress in
                progress(MeetingFileAnalysisProgress(
                    stage: .transcribing,
                    stageFraction: 0.25,
                    mediaDurationSeconds: 7_200,
                    processedMediaDurationSeconds: 1_800
                ))
                throw MeetingLocalInferenceCoordinatorError.memoryConstrained
            },
            cancelActiveAnalysis: {}, canStart: { true }, storageDirectoryURL: storage.url
        )
        queue.enqueue(urls: [source])
        try await waitUntil(queue, status: .failed, at: 0)
        let failed = try XCTUnwrap(queue.tasks.first)
        XCTAssertEqual(failed.progressStage, .transcribing)
        XCTAssertEqual(failed.processedMediaDurationSeconds, 1_800)
        XCTAssertEqual(failed.mediaDurationSeconds, 7_200)
        XCTAssertEqual(failed.errorMessage, MeetingLocalInferenceCoordinatorError.memoryConstrained.localizedDescription)
        XCTAssertNotNil(failed.completedAt)
        await queue.shutdown()

        struct Payload: Decodable { let tasks: [MeetingFileTask] }
        let payload = try JSONDecoder().decode(
            Payload.self, from: Data(contentsOf: storage.url.appendingPathComponent("tasks.json"))
        )
        XCTAssertEqual(payload.tasks.first?.progressStage, failed.progressStage)
        XCTAssertEqual(payload.tasks.first?.errorMessage, failed.errorMessage)
        XCTAssertEqual(payload.tasks.first?.processedMediaDurationSeconds, failed.processedMediaDurationSeconds)
    }

    func testOnlyCompletedASRCheckpointCanBePreviewed() async throws {
        let storage = try TemporaryDirectory()
        let source = try makeSourceFile(named: "preview.wav")
        let queue = MeetingFileTaskQueue(
            analyzer: { _, _, _ in throw CancellationError() },
            cancelActiveAnalysis: {}, canStart: { false }, storageDirectoryURL: storage.url
        )
        queue.enqueue(urls: [source])
        let id = try XCTUnwrap(queue.tasks.first?.id)
        let store = MeetingFileAnalysisCheckpointStore.shared
        let segment = MeetingTranscriptSegment(speaker: .them, startSeconds: 0, endSeconds: 1, text: "ASR result")
        await store.save(MeetingFileASRCheckpoint(
            schemaVersion: 1, taskID: id, preparedAudioSampleCount: 32_000, descriptorCount: 2,
            modelFingerprint: "test", completedDescriptorCount: 1, segments: [segment], updatedAt: Date()
        ))
        let partial = await queue.completedTranscriptSegments(taskID: id)
        XCTAssertNil(partial)
        await store.save(MeetingFileASRCheckpoint(
            schemaVersion: 1, taskID: id, preparedAudioSampleCount: 32_000, descriptorCount: 2,
            modelFingerprint: "test", completedDescriptorCount: 2, segments: [segment], updatedAt: Date()
        ))
        let complete = await queue.completedTranscriptSegments(taskID: id)
        XCTAssertEqual(complete?.first?.text, "ASR result")
        XCTAssertEqual(complete?.first?.startSeconds, 0)
        XCTAssertEqual(complete?.first?.endSeconds, 1)
        await store.clear(taskID: id)
        let cleared = await queue.completedTranscriptSegments(taskID: id)
        XCTAssertNil(cleared)
        await queue.shutdown()
    }

    func testTaskEstimateAndRetryReset() {
        let enqueuedAt = Date(timeIntervalSince1970: 100)
        var task = MeetingFileTask.queued(
            fileName: "meeting.wav",
            stagedFileName: "staged.wav",
            enqueuedAt: enqueuedAt
        )
        task.status = .processing
        task.startedAt = Date(timeIntervalSince1970: 110)
        task.progressFraction = 0.25

        XCTAssertEqual(task.elapsedSeconds(now: Date(timeIntervalSince1970: 130)), 20)
        let estimatedRemaining = try? XCTUnwrap(
            task.estimatedRemainingSeconds(now: Date(timeIntervalSince1970: 130))
        )
        XCTAssertEqual(estimatedRemaining ?? -1, 60, accuracy: 0.001)

        task.status = .failed
        task.errorMessage = "failure"
        task.historyEntryID = UUID()
        let retry = task.resetForRetry()
        XCTAssertEqual(retry.status, .queued)
        XCTAssertNil(retry.startedAt)
        XCTAssertNil(retry.completedAt)
        XCTAssertNil(retry.errorMessage)
        XCTAssertNil(retry.historyEntryID)
        XCTAssertEqual(retry.progressFraction, 0)
        XCTAssertNil(retry.estimatedTotalSeconds)
    }

    func testTaskEstimateUsesConservativeAnchorAndNeverIncreases() {
        let initial = MeetingFileTask.updatedEstimatedTotalSeconds(
            current: nil,
            elapsed: 10,
            progressFraction: 0.1
        )
        XCTAssertEqual(initial ?? -1, 135, accuracy: 0.001)

        let fasterPhase = MeetingFileTask.updatedEstimatedTotalSeconds(
            current: initial,
            elapsed: 40,
            progressFraction: 0.5
        )
        XCTAssertEqual(fasterPhase ?? -1, 108, accuracy: 0.001)

        let slowerPhase = MeetingFileTask.updatedEstimatedTotalSeconds(
            current: fasterPhase,
            elapsed: 50,
            progressFraction: 0.3
        )
        XCTAssertEqual(slowerPhase ?? -1, fasterPhase ?? -2, accuracy: 0.001)

        let remainingAtAnchor = MeetingFileTask.queued(
            fileName: "meeting.wav",
            stagedFileName: "staged.wav"
        )
        var processingTask = remainingAtAnchor
        processingTask.status = .processing
        processingTask.startedAt = Date(timeIntervalSince1970: 100)
        processingTask.estimatedTotalSeconds = slowerPhase
        XCTAssertEqual(
            processingTask.estimatedRemainingSeconds(now: Date(timeIntervalSince1970: 150)) ?? -1,
            58,
            accuracy: 0.001
        )
    }

    func testTaskEstimateRemainsPositiveWhilePostTranscriptionStagesAreRunning() {
        let estimate = MeetingFileTask.updatedEstimatedTotalSeconds(
            current: nil,
            elapsed: 40,
            progressFraction: 0.78,
            mediaDurationSeconds: 2_520,
            processedMediaDurationSeconds: 2_520,
            stage: .transcribing
        )

        var task = MeetingFileTask.queued(
            fileName: "meeting.wav",
            stagedFileName: "staged.wav"
        )
        task.status = .processing
        task.startedAt = Date(timeIntervalSince1970: 100)
        task.progressFraction = 0.90
        task.estimatedTotalSeconds = estimate

        XCTAssertGreaterThan(task.estimatedRemainingSeconds(now: Date(timeIntervalSince1970: 150)) ?? 0, 0)

        let identifyingEstimate = MeetingFileTask.updatedEstimatedTotalSeconds(
            current: estimate,
            elapsed: 70,
            progressFraction: 0.90,
            mediaDurationSeconds: 2_520,
            processedMediaDurationSeconds: 2_520,
            stage: .identifyingSpeakers
        )
        task.estimatedTotalSeconds = identifyingEstimate
        XCTAssertGreaterThan(task.estimatedRemainingSeconds(now: Date(timeIntervalSince1970: 170)) ?? 0, 0)

        let recoveredEstimate = MeetingFileTask.updatedEstimatedTotalSeconds(
            current: 0,
            elapsed: 10,
            progressFraction: 0.20
        )
        XCTAssertGreaterThan(recoveredEstimate ?? 0, 0)
    }

    func testTaskPersistenceRoundTripsProgressAndHistoryID() throws {
        var task = MeetingFileTask.queued(
            fileName: "meeting.wav",
            stagedFileName: "staged.wav",
            enqueuedAt: Date(timeIntervalSince1970: 100)
        )
        task.status = .completed
        task.startedAt = Date(timeIntervalSince1970: 110)
        task.completedAt = Date(timeIntervalSince1970: 140)
        task.progressStage = .saving
        task.progressFraction = 1
        task.historyEntryID = UUID()

        try XCTAssertJSONRoundTrip(task)
    }

    private func makeSourceFile(named name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("Voxt-Meeting-Task-\(UUID().uuidString)-\(name)")
        try MeetingAudioChunkWAVExporter.write(
            samples: Array(repeating: Float.zero, count: 160),
            sampleRate: 16_000,
            to: url
        )
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func waitUntilAllTasksAreTerminal(_ queue: MeetingFileTaskQueue) async throws {
        for _ in 0..<1_000 {
            if !queue.tasks.contains(where: { !$0.isTerminal }) {
                return
            }
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTFail("Timed out waiting for the meeting file queue")
    }

    private func waitUntil(
        _ queue: MeetingFileTaskQueue,
        status: MeetingFileTaskStatus,
        at index: Int
    ) async throws {
        for _ in 0..<1_000 {
            if queue.tasks.indices.contains(index), queue.tasks[index].status == status {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Timed out waiting for task status (status)")
    }

    private static func makeHistoryEntry() -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: UUID(),
            text: "Test transcript",
            createdAt: Date(),
            transcriptionEngine: "test",
            transcriptionModel: "test",
            enhancementMode: "off",
            enhancementModel: "",
            kind: .transcript,
            isTranslation: false,
            audioDurationSeconds: nil,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }
}

private actor FilePreparationProbe {
    private var active = 0
    private(set) var maximumActive = 0
    private(set) var events: [String] = []

    func begin(_ event: String) {
        active += 1
        maximumActive = max(maximumActive, active)
        events.append(event)
    }

    func end() { active -= 1 }
}
