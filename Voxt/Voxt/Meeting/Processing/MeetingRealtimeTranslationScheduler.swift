// MeetingRealtimeTranslationScheduler.swift
// Serializes realtime meeting translation so local models cannot fan out into concurrent inference.

import Foundation

nonisolated enum MeetingTranslationExecutionScope: Sendable, Equatable {
    case localInference
    case externalRequest
}

nonisolated struct MeetingTranslationOperation: @unchecked Sendable {
    let executionScope: MeetingTranslationExecutionScope
    let run: @MainActor () async throws -> String

    static func cancelled() -> MeetingTranslationOperation {
        MeetingTranslationOperation(executionScope: .externalRequest) {
            throw CancellationError()
        }
    }
}

@MainActor
final class MeetingRealtimeTranslationScheduler {
    enum SubmissionResult: Equatable {
        case accepted
        case updated
        case duplicate
        case overloaded
    }

    struct Statistics: Equatable, Sendable {
        var submittedCount = 0
        var completedCount = 0
        var failedCount = 0
        var cancelledCount = 0
        var overloadedCount = 0
        var peakScheduledCount = 0
        var inferenceBatchCount = 0
        var peakBatchSize = 0
    }

    private struct Job: @unchecked Sendable {
        let token: UUID
        let segmentID: UUID
        let sourceText: String
        let targetLanguage: TranslationTargetLanguage
        let operation: MeetingTranslationOperation
        let completion: @MainActor (Result<String, Error>) -> Void
    }

    private struct Revision: Equatable {
        let sourceText: String
        let targetLanguage: TranslationTargetLanguage
        let executionScope: MeetingTranslationExecutionScope
    }

    /// One active microbatch plus at most two pending microbatches.
    private static let maximumScheduledJobs = 9
    private static let maximumBatchJobs = 4
    private static let maximumBatchCharacters = 240

    private let workClass: MeetingLocalInferenceWorkClass
    private var jobs: [Job] = []
    private var nextJobIndex = 0
    private var scheduledTokensBySegmentID: [UUID: UUID] = [:]
    private var scheduledRevisionsBySegmentID: [UUID: Revision] = [:]
    private var activeSegmentID: UUID?
    private var workerTask: Task<Void, Never>?
    private var generation = 0
    private(set) var statistics = Statistics()

    init(workClass: MeetingLocalInferenceWorkClass = .realtimeTranslation) {
        self.workClass = workClass
    }

    nonisolated deinit {}

    @discardableResult
    func submit(
        segmentID: UUID,
        sourceText: String,
        targetLanguage: TranslationTargetLanguage,
        operation: MeetingTranslationOperation,
        completion: @escaping @MainActor (Result<String, Error>) -> Void
    ) -> SubmissionResult {
        let revision = Revision(
            sourceText: sourceText.trimmingCharacters(in: .whitespacesAndNewlines),
            targetLanguage: targetLanguage,
            executionScope: operation.executionScope
        )
        if scheduledRevisionsBySegmentID[segmentID] == revision {
            return .duplicate
        }
        let isRevisionUpdate = scheduledTokensBySegmentID[segmentID] != nil
        guard isRevisionUpdate || scheduledTokensBySegmentID.count < Self.maximumScheduledJobs else {
            statistics.overloadedCount += 1
            return .overloaded
        }

        let token = UUID()
        scheduledTokensBySegmentID[segmentID] = token
        scheduledRevisionsBySegmentID[segmentID] = revision
        let job = Job(
            token: token,
            segmentID: segmentID,
            sourceText: sourceText,
            targetLanguage: targetLanguage,
            operation: operation,
            completion: completion
        )
        if isRevisionUpdate,
           let pendingIndex = jobs.indices.first(where: {
               $0 >= nextJobIndex && jobs[$0].segmentID == segmentID
           }) {
            jobs[pendingIndex] = job
        } else {
            jobs.append(job)
        }
        statistics.submittedCount += 1
        statistics.peakScheduledCount = max(statistics.peakScheduledCount, scheduledTokensBySegmentID.count)
        startWorkerIfNeeded()
        return isRevisionUpdate ? .updated : .accepted
    }

    func cancel(segmentID: UUID) {
        guard scheduledTokensBySegmentID.removeValue(forKey: segmentID) != nil else { return }
        scheduledRevisionsBySegmentID[segmentID] = nil
        statistics.cancelledCount += 1
    }

    func cancelAll() {
        generation += 1
        statistics.cancelledCount += scheduledTokensBySegmentID.count
        scheduledTokensBySegmentID.removeAll()
        scheduledRevisionsBySegmentID.removeAll()
        jobs.removeAll(keepingCapacity: false)
        nextJobIndex = 0
        activeSegmentID = nil
        workerTask?.cancel()
    }

    func flush() async {
        while let workerTask {
            await workerTask.value
        }
    }

    func currentStatistics() -> Statistics {
        statistics
    }

    func resetStatistics() {
        statistics = Statistics()
    }

    private func startWorkerIfNeeded() {
        guard workerTask == nil else { return }
        let workerGeneration = generation
        workerTask = Task { @MainActor [weak self] in
            await self?.drain(workerGeneration: workerGeneration)
        }
    }

    private func drain(workerGeneration: Int) async {
        defer {
            activeSegmentID = nil
            workerTask = nil
            compactQueue(force: true)
            if nextJobIndex < jobs.count {
                startWorkerIfNeeded()
            }
        }

        while workerGeneration == generation, !Task.isCancelled, let batch = popNextBatch() {
            guard let firstJob = batch.first else { continue }
            activeSegmentID = firstJob.segmentID
            statistics.inferenceBatchCount += 1
            statistics.peakBatchSize = max(statistics.peakBatchSize, batch.count)
            let results = await execute(batch)

            activeSegmentID = nil
            for (job, result) in results {
                guard workerGeneration == generation,
                      !Task.isCancelled,
                      scheduledTokensBySegmentID[job.segmentID] == job.token
                else {
                    continue
                }
                scheduledTokensBySegmentID[job.segmentID] = nil
                scheduledRevisionsBySegmentID[job.segmentID] = nil
                switch result {
                case .success:
                    statistics.completedCount += 1
                case .failure:
                    statistics.failedCount += 1
                }
                job.completion(result)
            }
        }
    }

    private func execute(_ batch: [Job]) async -> [(Job, Result<String, Error>)] {
        guard let executionScope = batch.first?.operation.executionScope else { return [] }
        switch executionScope {
        case .localInference:
            do {
                return try await MeetingLocalInferenceCoordinator.shared.withPermit(workClass) {
                    await Self.executeOperations(batch)
                }
            } catch {
                return batch.map { ($0, .failure(error)) }
            }
        case .externalRequest:
            return await Self.executeOperations(batch)
        }
    }

    private static func executeOperations(_ batch: [Job]) async -> [(Job, Result<String, Error>)] {
        var output: [(Job, Result<String, Error>)] = []
        output.reserveCapacity(batch.count)
        for job in batch {
            do {
                output.append((job, .success(try await job.operation.run())))
            } catch {
                output.append((job, .failure(error)))
            }
        }
        return output
    }

    private func popNextBatch() -> [Job]? {
        guard let first = popNextJob() else { return nil }
        var batch = [first]
        var characterCount = first.sourceText.count

        while batch.count < Self.maximumBatchJobs,
              let candidate = peekNextValidJob(),
              candidate.targetLanguage == first.targetLanguage,
              candidate.operation.executionScope == first.operation.executionScope,
              characterCount + candidate.sourceText.count <= Self.maximumBatchCharacters {
            guard let accepted = popNextJob() else { break }
            batch.append(accepted)
            characterCount += accepted.sourceText.count
        }
        return batch
    }

    private func peekNextValidJob() -> Job? {
        while nextJobIndex < jobs.count {
            let job = jobs[nextJobIndex]
            if scheduledTokensBySegmentID[job.segmentID] == job.token {
                return job
            }
            nextJobIndex += 1
            compactQueue(force: false)
        }
        return nil
    }

    private func popNextJob() -> Job? {
        while nextJobIndex < jobs.count {
            let job = jobs[nextJobIndex]
            nextJobIndex += 1
            compactQueue(force: false)
            if scheduledTokensBySegmentID[job.segmentID] == job.token {
                return job
            }
        }
        return nil
    }

    private func compactQueue(force: Bool) {
        guard nextJobIndex > 0,
              force || (nextJobIndex >= 16 && nextJobIndex * 2 >= jobs.count)
        else {
            return
        }
        jobs.removeFirst(nextJobIndex)
        nextJobIndex = 0
    }
}
