import Foundation

final class ResumableDownloadAttemptDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    enum ResponseDecision {
        case stream(fileHandle: FileHandle, initialState: ResumableDownloadState, rangeSupported: Bool, resumedFromBytes: Int64)
        case restartFromZero(reason: String)
        case completedExisting(ResumableDownloadResult)
    }

    private let progress: Progress
    private let responseHandler: @Sendable (HTTPURLResponse) throws -> ResponseDecision
    private let stateWriter: @Sendable (ResumableDownloadState) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<ResumableDownloadAttemptOutcome, Error>?
    private var task: URLSessionTask?
    private var fileHandle: FileHandle?
    private var currentState: ResumableDownloadState?
    private var controlledOutcome: ResumableDownloadAttemptOutcome?
    private var lastProgressAt = Date()
    private var totalBytesWritten: Int64 = 0
    private var rangeSupported = false
    private var resumedFromBytes: Int64 = 0
    private var hasFinished = false
    private var lastPersistedBytes: Int64 = 0
    private var lastPersistedAt = Date.distantPast

    init(
        progress: Progress,
        responseHandler: @escaping @Sendable (HTTPURLResponse) throws -> ResponseDecision,
        stateWriter: @escaping @Sendable (ResumableDownloadState) -> Void
    ) {
        self.progress = progress
        self.responseHandler = responseHandler
        self.stateWriter = stateWriter
    }

    func attach(task: URLSessionTask, continuation: CheckedContinuation<ResumableDownloadAttemptOutcome, Error>) {
        lock.lock()
        self.task = task
        self.continuation = continuation
        lock.unlock()
    }

    func progressTimedOut(stallTimeout: Duration) {
        lock.lock()
        guard !hasFinished else {
            lock.unlock()
            return
        }
        let elapsed = Date().timeIntervalSince(lastProgressAt)
        guard elapsed >= ResumableModelDownloadSupport.seconds(from: stallTimeout) else {
            lock.unlock()
            return
        }
        controlledOutcome = .recoverableFailure(reason: "stall-timeout")
        let task = self.task
        lock.unlock()
        task?.cancel()
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let httpResponse = response as? HTTPURLResponse else {
            controlledOutcome = .recoverableFailure(reason: ResumableDownloadError.badServerResponse.localizedDescription)
            completionHandler(.cancel)
            return
        }

        do {
            switch try responseHandler(httpResponse) {
            case .stream(let fileHandle, let initialState, let rangeSupported, let resumedFromBytes):
                lock.lock()
                self.fileHandle = fileHandle
                self.currentState = initialState
                self.rangeSupported = rangeSupported
                self.resumedFromBytes = resumedFromBytes
                self.totalBytesWritten = resumedFromBytes
                self.lastPersistedBytes = resumedFromBytes
                self.lastPersistedAt = Date()
                self.lastProgressAt = Date()
                lock.unlock()
                progress.totalUnitCount = max(initialState.expectedSize, 1)
                progress.completedUnitCount = max(resumedFromBytes, 0)
                stateWriter(initialState)
                completionHandler(.allow)
            case .restartFromZero(let reason):
                controlledOutcome = .restartFromZero(reason: reason)
                completionHandler(.cancel)
            case .completedExisting(let result):
                controlledOutcome = .completed(result)
                completionHandler(.cancel)
            }
        } catch {
            controlledOutcome = .fatal(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        let fileHandle = self.fileHandle
        lock.unlock()

        guard let fileHandle else { return }
        do {
            try fileHandle.write(contentsOf: data)
            let chunkBytes = Int64(data.count)
            lock.lock()
            totalBytesWritten += chunkBytes
            lastProgressAt = Date()
            let totalBytesWritten = self.totalBytesWritten
            let currentState = self.currentState
            lock.unlock()

            progress.completedUnitCount = max(totalBytesWritten, 0)
            maybePersistProgress(totalBytesWritten: totalBytesWritten, currentState: currentState)
        } catch {
            controlledOutcome = .fatal(error)
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        hasFinished = true
        let continuation = self.continuation
        self.continuation = nil
        let controlledOutcome = self.controlledOutcome
        let currentState = self.currentState
        let fileHandle = self.fileHandle
        let totalBytesWritten = self.totalBytesWritten
        let rangeSupported = self.rangeSupported
        let resumedFromBytes = self.resumedFromBytes
        lock.unlock()

        try? fileHandle?.close()

        if let currentState {
            stateWriter(
                ResumableDownloadState(
                    relativePath: currentState.relativePath,
                    sourceURL: currentState.sourceURL,
                    expectedSize: currentState.expectedSize,
                    etag: currentState.etag,
                    downloadedBytes: totalBytesWritten,
                    updatedAt: Date(),
                    rangeSupported: currentState.rangeSupported
                )
            )
        }

        guard let continuation else { return }

        if let controlledOutcome {
            switch controlledOutcome {
            case .fatal(let error):
                continuation.resume(throwing: error)
            default:
                continuation.resume(returning: controlledOutcome)
            }
            return
        }

        if let error {
            if (error as? URLError)?.code == .cancelled {
                continuation.resume(throwing: CancellationError())
            } else {
                continuation.resume(throwing: error)
            }
            return
        }

        continuation.resume(
            returning: .completed(
                ResumableDownloadResult(
                    bytesDownloaded: totalBytesWritten,
                    resumedFromBytes: resumedFromBytes,
                    rangeSupported: rangeSupported
                )
            )
        )
    }

    private func maybePersistProgress(totalBytesWritten: Int64, currentState: ResumableDownloadState?) {
        guard let currentState else { return }
        let now = Date()
        let shouldPersist = totalBytesWritten - lastPersistedBytes >= 5 * 1024 * 1024
            || now.timeIntervalSince(lastPersistedAt) >= 5
        guard shouldPersist else { return }
        lock.lock()
        lastPersistedBytes = totalBytesWritten
        lastPersistedAt = now
        lock.unlock()
        stateWriter(
            ResumableDownloadState(
                relativePath: currentState.relativePath,
                sourceURL: currentState.sourceURL,
                expectedSize: currentState.expectedSize,
                etag: currentState.etag,
                downloadedBytes: totalBytesWritten,
                updatedAt: now,
                rangeSupported: currentState.rangeSupported
            )
        )
    }
}
