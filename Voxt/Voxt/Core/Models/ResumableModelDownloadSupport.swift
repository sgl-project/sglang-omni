import Foundation
import CFNetwork

enum ResumableModelDownloadSupport {
    static func download(
        _ descriptor: ResumableDownloadDescriptor,
        progress: Progress
    ) async throws -> ResumableDownloadResult {
        let fileManager = FileManager.default
        let partURL = partialFileURL(for: descriptor.destinationURL)
        let stateURL = stateFileURL(for: partURL)
        try fileManager.createDirectory(
            at: descriptor.destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let supportsByteResume = max(descriptor.expectedSize ?? 0, 0) >= descriptor.policy.resumeThresholdBytes
        var recoveryAttempts = 0
        var restartFromZeroCount = 0

        while true {
            try Task.checkCancellation()

            if !supportsByteResume {
                try purgePartialArtifacts(for: descriptor.destinationURL)
            }

            let partialSize = fileSize(at: partURL)
            let loadedState = loadState(from: stateURL)
            let preparedState = try prepareState(
                descriptor: descriptor,
                supportsByteResume: supportsByteResume,
                partialURL: partURL,
                partialSize: partialSize,
                loadedState: loadedState
            )

            do {
                let result = try await performAttempt(
                    descriptor: descriptor,
                    partURL: partURL,
                    stateURL: stateURL,
                    existingState: preparedState,
                    progress: progress
                )
                try finalizeDownload(
                    descriptor: descriptor,
                    partURL: partURL,
                    stateURL: stateURL,
                    result: result
                )
                return result
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as ResumableDownloadError {
                if error.isRetryable {
                    recoveryAttempts += 1
                    guard recoveryAttempts <= descriptor.policy.maxRecoveryAttempts else {
                        throw error
                    }
                    VoxtLog.modelWarning("Resumable download retry \(recoveryAttempts)/\(descriptor.policy.maxRecoveryAttempts): \(descriptor.relativePath) (\(error.localizedDescription))")
                    try? await Task.sleep(for: backoffDuration(policy: descriptor.policy, attempt: recoveryAttempts))
                    continue
                }
                throw error
            } catch let error as URLError {
                recoveryAttempts += 1
                guard recoveryAttempts <= descriptor.policy.maxRecoveryAttempts,
                      MLXModelDownloadSupport.isRetryableTransportError(error)
                else {
                    throw error
                }
                VoxtLog.modelWarning("Resumable download retry \(recoveryAttempts)/\(descriptor.policy.maxRecoveryAttempts): \(descriptor.relativePath) (\(error.localizedDescription))")
                try? await Task.sleep(for: backoffDuration(policy: descriptor.policy, attempt: recoveryAttempts))
            } catch {
                if let restartReason = (error as? ResumableDownloadLoopError)?.restartReason {
                    restartFromZeroCount += 1
                    guard restartFromZeroCount <= descriptor.policy.maxRecoveryAttempts else {
                        throw error
                    }
                    VoxtLog.modelWarning("Resumable download restart from zero: \(descriptor.relativePath) (\(restartReason))")
                    try purgePartialArtifacts(for: descriptor.destinationURL)
                    continue
                }
                if let recoverableReason = (error as? ResumableDownloadLoopError)?.recoverableReason {
                    recoveryAttempts += 1
                    guard recoveryAttempts <= descriptor.policy.maxRecoveryAttempts else {
                        throw error
                    }
                    VoxtLog.modelWarning("Resumable download recoverable retry \(recoveryAttempts)/\(descriptor.policy.maxRecoveryAttempts): \(descriptor.relativePath) (\(recoverableReason))")
                    try? await Task.sleep(for: backoffDuration(policy: descriptor.policy, attempt: recoveryAttempts))
                    continue
                }
                throw error
            }
        }
    }

    static func purgePartialArtifacts(for destinationURL: URL) throws {
        let fileManager = FileManager.default
        let partURL = partialFileURL(for: destinationURL)
        let sidecarURL = stateFileURL(for: partURL)
        if fileManager.fileExists(atPath: partURL.path) {
            try? fileManager.removeItem(at: partURL)
        }
        if fileManager.fileExists(atPath: sidecarURL.path) {
            try? fileManager.removeItem(at: sidecarURL)
        }
    }

    private static func performAttempt(
        descriptor: ResumableDownloadDescriptor,
        partURL: URL,
        stateURL: URL,
        existingState: ResumableDownloadState?,
        progress: Progress
    ) async throws -> ResumableDownloadResult {
        let initialBytes = fileSize(at: partURL)
        let shouldResume = existingState?.rangeSupported == true && initialBytes > 0
        let expectedTotal = max(existingState?.expectedSize ?? descriptor.expectedSize ?? 0, 0)

        progress.totalUnitCount = max(expectedTotal, 1)
        progress.completedUnitCount = max(initialBytes, 0)

        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 60 * 60
        configuration.waitsForConnectivity = false
        if descriptor.disableProxy {
            configuration.connectionProxyDictionary = [
                kCFNetworkProxiesHTTPEnable as String: false,
                kCFNetworkProxiesHTTPSEnable as String: false,
                kCFNetworkProxiesSOCKSEnable as String: false,
            ]
        }
        if let protocolClasses = descriptor.protocolClasses {
            configuration.protocolClasses = protocolClasses
        }

        let responseHandler: @Sendable (HTTPURLResponse) throws -> ResumableDownloadAttemptDelegate.ResponseDecision = { response in
            let metadata = ResumableResponseMetadata(response: response)
            switch metadata.statusCode {
            case 200:
                if shouldResume {
                    return .restartFromZero(reason: "server-ignored-range")
                }

                let remoteETag = metadata.etag
                guard let remoteETag, !remoteETag.isEmpty else {
                    throw ResumableDownloadError.missingETag
                }

                let actualExpectedSize = max(metadata.contentLength ?? descriptor.expectedSize ?? 0, 0)
                if descriptor.requiresExpectedSizeMatch,
                   let descriptorExpected = descriptor.expectedSize,
                   descriptorExpected > 0,
                   actualExpectedSize > 0,
                   descriptorExpected != actualExpectedSize
                {
                    throw ResumableDownloadError.inconsistentExpectedSize(expected: descriptorExpected, actual: actualExpectedSize)
                }

                if FileManager.default.fileExists(atPath: partURL.path) {
                    try? FileManager.default.removeItem(at: partURL)
                }
                FileManager.default.createFile(atPath: partURL.path, contents: nil)
                let fileHandle = try FileHandle(forWritingTo: partURL)
                let initialState = ResumableDownloadState(
                    relativePath: descriptor.relativePath,
                    sourceURL: descriptor.sourceURL.absoluteString,
                    expectedSize: actualExpectedSize,
                    etag: remoteETag,
                    downloadedBytes: 0,
                    updatedAt: Date(),
                    rangeSupported: metadata.acceptRanges
                )
                return .stream(
                    fileHandle: fileHandle,
                    initialState: initialState,
                    rangeSupported: metadata.acceptRanges,
                    resumedFromBytes: 0
                )
            case 206:
                guard shouldResume else {
                    return .restartFromZero(reason: "unexpected-partial-content")
                }
                guard let contentRange = metadata.contentRange else {
                    throw ResumableDownloadError.invalidContentRange(response.value(forHTTPHeaderField: "Content-Range") ?? "")
                }
                guard contentRange.start == initialBytes else {
                    return .restartFromZero(reason: "content-range-mismatch")
                }
                if let stateETag = existingState?.etag,
                   let remoteETag = metadata.etag,
                   remoteETag != stateETag
                {
                    return .restartFromZero(reason: "etag-changed")
                }
                guard let existingState else {
                    throw ResumableDownloadError.inconsistentPartialState
                }
                let fileHandle = try FileHandle(forWritingTo: partURL)
                try fileHandle.seekToEnd()
                let updatedState = ResumableDownloadState(
                    relativePath: existingState.relativePath,
                    sourceURL: descriptor.sourceURL.absoluteString,
                    expectedSize: contentRange.total ?? existingState.expectedSize,
                    etag: metadata.etag ?? existingState.etag,
                    downloadedBytes: initialBytes,
                    updatedAt: Date(),
                    rangeSupported: true
                )
                return .stream(
                    fileHandle: fileHandle,
                    initialState: updatedState,
                    rangeSupported: true,
                    resumedFromBytes: initialBytes
                )
            case 416:
                if let completedSize = existingState?.expectedSize ?? descriptor.expectedSize,
                   completedSize > 0,
                   initialBytes == completedSize
                {
                    return .completedExisting(
                        ResumableDownloadResult(
                            bytesDownloaded: initialBytes,
                            resumedFromBytes: initialBytes,
                            rangeSupported: true
                        )
                    )
                }
                return .restartFromZero(reason: "range-not-satisfiable")
            default:
                throw ResumableDownloadError.unexpectedStatusCode(metadata.statusCode)
            }
        }

        let delegate = ResumableDownloadAttemptDelegate(
            progress: progress,
            responseHandler: responseHandler
        ) { state in
            saveState(state, to: stateURL)
        }
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        var request = URLRequest(url: descriptor.sourceURL)
        request.setValue(descriptor.userAgent, forHTTPHeaderField: "User-Agent")
        if let bearerToken = descriptor.bearerToken, !bearerToken.isEmpty {
            request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        }
        if shouldResume {
            request.setValue("bytes=\(initialBytes)-", forHTTPHeaderField: "Range")
            if let etag = existingState?.etag, !etag.isEmpty {
                request.setValue(etag, forHTTPHeaderField: "If-Range")
            }
            VoxtLog.modelInfo("Resumable download resuming: file=\(descriptor.relativePath), offset=\(initialBytes)")
        } else {
            VoxtLog.modelInfo("Resumable download starting: file=\(descriptor.relativePath), url=\(descriptor.sourceURL.absoluteString)")
        }

        let task = session.dataTask(with: request)
        return try await withTaskCancellationHandler(operation: {
            let watchdog = Task { [policy = descriptor.policy] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: policy.stallPollInterval)
                    delegate.progressTimedOut(stallTimeout: policy.stallTimeout)
                }
            }
            defer {
                watchdog.cancel()
                session.invalidateAndCancel()
            }
            let outcome = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<ResumableDownloadAttemptOutcome, Error>) in
                delegate.attach(task: task, continuation: continuation)
                task.resume()
            }

            switch outcome {
            case .completed(let result):
                return result
            case .restartFromZero(let reason):
                throw ResumableDownloadLoopError.restartFromZero(reason: reason)
            case .recoverableFailure(let reason):
                throw ResumableDownloadLoopError.recoverable(reason: reason)
            case .fatal(let error):
                throw error
            }
        }, onCancel: {
            task.cancel()
            session.invalidateAndCancel()
        })
    }

    private static func prepareState(
        descriptor: ResumableDownloadDescriptor,
        supportsByteResume: Bool,
        partialURL: URL,
        partialSize: Int64,
        loadedState: ResumableDownloadState?
    ) throws -> ResumableDownloadState? {
        guard partialSize > 0 else {
            if loadedState != nil {
                try purgePartialArtifacts(for: descriptor.destinationURL)
            }
            return nil
        }

        guard supportsByteResume else {
            try purgePartialArtifacts(for: descriptor.destinationURL)
            return nil
        }

        guard let loadedState,
              !loadedState.etag.isEmpty,
              loadedState.expectedSize > 0
        else {
            try purgePartialArtifacts(for: descriptor.destinationURL)
            return nil
        }

        if descriptor.requiresExpectedSizeMatch,
           let expectedSize = descriptor.expectedSize,
           expectedSize > 0,
           loadedState.expectedSize != expectedSize
        {
            try purgePartialArtifacts(for: descriptor.destinationURL)
            return nil
        }

        if partialSize > loadedState.expectedSize {
            try purgePartialArtifacts(for: descriptor.destinationURL)
            return nil
        }

        if partialSize == loadedState.expectedSize {
            return ResumableDownloadState(
                relativePath: loadedState.relativePath,
                sourceURL: loadedState.sourceURL,
                expectedSize: loadedState.expectedSize,
                etag: loadedState.etag,
                downloadedBytes: partialSize,
                updatedAt: Date(),
                rangeSupported: loadedState.rangeSupported
            )
        }

        return ResumableDownloadState(
            relativePath: loadedState.relativePath,
            sourceURL: loadedState.sourceURL,
            expectedSize: loadedState.expectedSize,
            etag: loadedState.etag,
            downloadedBytes: partialSize,
            updatedAt: Date(),
            rangeSupported: loadedState.rangeSupported
        )
    }

    private static func finalizeDownload(
        descriptor: ResumableDownloadDescriptor,
        partURL: URL,
        stateURL: URL,
        result: ResumableDownloadResult
    ) throws {
        if descriptor.requiresExpectedSizeMatch {
            let expectedSize = descriptor.expectedSize ?? result.bytesDownloaded
            if expectedSize > 0, result.bytesDownloaded != expectedSize {
                throw ResumableDownloadError.inconsistentExpectedSize(expected: expectedSize, actual: result.bytesDownloaded)
            }
        }

        if FileManager.default.fileExists(atPath: descriptor.destinationURL.path) {
            try? FileManager.default.removeItem(at: descriptor.destinationURL)
        }
        try FileManager.default.moveItem(at: partURL, to: descriptor.destinationURL)
        try? FileManager.default.removeItem(at: stateURL)
        VoxtLog.modelInfo("Resumable download completed: file=\(descriptor.relativePath), bytes=\(result.bytesDownloaded), resumedFrom=\(result.resumedFromBytes)")
    }

    private static func partialFileURL(for destinationURL: URL) -> URL {
        destinationURL.appendingPathExtension("part")
    }

    private static func stateFileURL(for partURL: URL) -> URL {
        partURL.appendingPathExtension("json")
    }

    private nonisolated static func loadState(from url: URL) -> ResumableDownloadState? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let relativePath = object["relativePath"] as? String,
              let sourceURL = object["sourceURL"] as? String,
              let expectedSize = object["expectedSize"] as? NSNumber,
              let etag = object["etag"] as? String,
              let downloadedBytes = object["downloadedBytes"] as? NSNumber,
              let updatedAtInterval = object["updatedAt"] as? NSNumber,
              let rangeSupported = object["rangeSupported"] as? Bool
        else {
            return nil
        }
        return ResumableDownloadState(
            relativePath: relativePath,
            sourceURL: sourceURL,
            expectedSize: expectedSize.int64Value,
            etag: etag,
            downloadedBytes: downloadedBytes.int64Value,
            updatedAt: Date(timeIntervalSince1970: updatedAtInterval.doubleValue),
            rangeSupported: rangeSupported
        )
    }

    private nonisolated static func saveState(_ state: ResumableDownloadState, to url: URL) {
        let object: [String: Any] = [
            "relativePath": state.relativePath,
            "sourceURL": state.sourceURL,
            "expectedSize": state.expectedSize,
            "etag": state.etag,
            "downloadedBytes": state.downloadedBytes,
            "updatedAt": state.updatedAt.timeIntervalSince1970,
            "rangeSupported": state.rangeSupported,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else { return }
        try? data.write(to: url, options: .atomic)
    }

    private nonisolated static func fileSize(at url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }

    private static func backoffDuration(policy: ResumableDownloadPolicy, attempt: Int) -> Duration {
        let exponent = max(0, attempt - 1)
        let rawSeconds = min(
            policy.initialBackoffSeconds * pow(2, Double(exponent)),
            policy.maxBackoffSeconds
        )
        let jitter = min(0.25 * rawSeconds, 1.0)
        let randomized = rawSeconds + Double.random(in: 0 ... jitter)
        return .milliseconds(Int64(randomized * 1000))
    }

    nonisolated static func seconds(from duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }
}
