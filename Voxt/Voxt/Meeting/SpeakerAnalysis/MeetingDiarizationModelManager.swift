// MeetingDiarizationModelManager.swift
// Provides Meeting Diarization Model Manager for meeting speaker analysis.

import Combine
import Foundation
import HuggingFace
import MLXAudioVAD

@MainActor
final class MeetingDiarizationModelManager: ObservableObject {
    enum State: Equatable {
        case notDownloaded
        case downloaded
        case downloading(progress: Double, detail: String?)
        case error(String)

        var isDownloading: Bool {
            if case .downloading = self {
                return true
            }
            return false
        }
    }

    @Published private(set) var selectedMode = MeetingDiarizationMode.stored()
    @Published private(set) var state: State = .notDownloaded
    @Published private(set) var remoteSizeText = MeetingDiarizationMode.stored().fallbackRemoteSizeText

    private var downloadTask: Task<Void, Never>?
    private var sizeTask: Task<Void, Never>?
    private let installationCache = ModelInstallationCache()
    private var storageDirectories: [URL] = []
    private var storageRevision = UUID()
    private var downloadStorageRevision: UUID?
    private var ensureInstallationRequested = false

    init() {
        installationCache.onChange = { [weak self] _, snapshot in
            guard let self, self.downloadTask == nil else { return }
            self.state = snapshot.isInstalled ? .downloaded : .notDownloaded
            if self.ensureInstallationRequested {
                self.ensureInstallationRequested = false
                if !snapshot.isInstalled { self.downloadSelectedModel() }
            }
        }
        ensureSelectedModelInstalled()
    }

    func refresh() {
        selectedMode = MeetingDiarizationMode.stored()
        remoteSizeText = selectedMode.fallbackRemoteSizeText
        let directories = MeetingSortformerModelStorage.readableDirectories()
        if directories != storageDirectories {
            storageRevision = UUID()
            downloadTask?.cancel()
            state = .notDownloaded
            installationCache.invalidateAll()
            storageDirectories = directories
        }
        let request = ModelInstallationRequest(directories: directories, partialDirectories: [], validate: {
            MeetingSortformerModelStorage.isValidModelDirectory($0)
        })
        installationCache.request("sortformer") { request.scan() }
        if downloadTask == nil, let snapshot = installationCache.peek("sortformer") {
            state = snapshot.isInstalled ? .downloaded : .notDownloaded
        }
        if sizeTask == nil { fetchSortformerRemoteSize() }
    }

    func ensureSelectedModelInstalled() {
        refresh()
        guard !state.isDownloading else { return }
        guard let snapshot = installationCache.peek("sortformer") else {
            ensureInstallationRequested = true
            return
        }
        if !snapshot.isInstalled { downloadSelectedModel() }
    }

    func downloadSelectedModel() {
        guard downloadTask == nil else { return }
        refresh()
        if case .downloaded = state { return }
        let revision = storageRevision
        downloadStorageRevision = revision
        installationCache.invalidate("sortformer")
        ensureInstallationRequested = false
        let mode = MeetingDiarizationMode.stored()
        selectedMode = mode
        remoteSizeText = mode.fallbackRemoteSizeText
        state = .downloading(progress: 0, detail: nil)
        SystemNotificationSupport.requestAuthorizationIfNeeded()
        downloadTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.downloadTask = nil
                self.installationCache.invalidate("sortformer")
                self.downloadStorageRevision = nil
                if revision != self.storageRevision { self.refresh() }
            }
            do {
                try Task.checkCancellation()
                guard revision == self.storageRevision else { throw CancellationError() }
                _ = try ModelStorageDirectoryManager.requireWriteRootURL()
                _ = try await self.downloadSortformerWithFallback()
                try Task.checkCancellation()
                guard revision == self.storageRevision else { throw CancellationError() }
                self.state = .downloaded
                SystemNotificationSupport.postModelDownloadSucceeded(modelName: mode.title)
                VoxtLog.meeting("Meeting diarization model ready. mode=\(mode.rawValue)")
            } catch is CancellationError {
                guard revision == self.storageRevision else { return }
                self.state = .notDownloaded
            } catch {
                guard revision == self.storageRevision else { return }
                self.state = .error(error.localizedDescription)
                SystemNotificationSupport.postModelDownloadFailed(
                    modelName: mode.title,
                    message: error.localizedDescription
                )
                VoxtLog.meetingError("Meeting diarization model install failed. mode=\(mode.rawValue), error=\(error.localizedDescription)")
            }
        }
    }

    private func fetchSortformerRemoteSize() {
        sizeTask?.cancel()
        sizeTask = Task { [weak self] in
            guard let self else { return }
            let preferredBaseURL = Self.preferredHubBaseURL()
            do {
                let info = try await MLXModelDownloadSupport.fetchModelSizeInfo(
                    repo: MeetingSortformerModelStorage.repo,
                    baseURL: preferredBaseURL,
                    userAgent: MLXModelManager.hubUserAgent,
                    formatByteCount: MLXModelStorageSupport.formatByteCount
                )
                guard !Task.isCancelled else { return }
                self.remoteSizeText = info.text
            } catch {
                guard !Task.isCancelled else { return }
                self.remoteSizeText = MeetingDiarizationMode.sortformerV2.fallbackRemoteSizeText
            }
        }
    }

    private func downloadSortformerWithFallback() async throws -> URL {
        let preferredBaseURL = Self.preferredHubBaseURL()
        do {
            return try await downloadSortformer(using: preferredBaseURL)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            try Task.checkCancellation()
            guard let fallback = Self.fallbackHubBaseURL(from: preferredBaseURL) else {
                throw error
            }
            VoxtLog.meetingWarning(
                "Primary meeting Sortformer download endpoint failed. Retrying with mirror. baseURL=\(preferredBaseURL.absoluteString), error=\(error.localizedDescription)"
            )
            MeetingSortformerModelStorage.clearHubCache()
            return try await downloadSortformer(using: fallback)
        }
    }

    private func downloadSortformer(using baseURL: URL) async throws -> URL {
        guard let repoID = HuggingFace.Repo.ID(rawValue: MeetingSortformerModelStorage.repo),
              let modelDir = MeetingSortformerModelStorage.writeModelDirectory(),
              let tempDir = MeetingSortformerModelStorage.downloadTempDirectory()
        else {
            throw NSError(
                domain: "Voxt.MeetingDiarization",
                code: 2000,
                userInfo: [NSLocalizedDescriptionKey: "Invalid meeting diarization model identifier."]
            )
        }

        let installed = await Task.detached(priority: .utility) {
            MeetingSortformerModelStorage.isValidModelDirectory(modelDir)
        }.value
        if installed { return modelDir }

        let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        let session = MLXModelDownloadSupport.makeDownloadSession(for: baseURL)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        let entries = try await MLXModelDownloadSupport.fetchModelEntries(
            repo: repoID.description,
            baseURL: baseURL,
            session: session,
            userAgent: MLXModelManager.hubUserAgent
        )
        guard !entries.isEmpty else {
            throw MLXModelDownloadSupport.DownloadValidationError.emptyFileList
        }

        let totalBytes = max(entries.reduce(Int64(0)) { $0 + max($1.size ?? 0, 0) }, 1)
        let totalFiles = entries.count
        var completedBytes: Int64 = 0

        for (index, entry) in entries.enumerated() {
            try Task.checkCancellation()
            let expectedBytes = max(entry.size ?? 0, 0)
            let progress = Progress(totalUnitCount: max(expectedBytes, 1))
            let baseCompletedBytes = completedBytes
            updateDownloadingState(
                progress: Double(completedBytes) / Double(totalBytes),
                detail: ModelDownloadProgressFormatter.fileProgressText(
                    currentFile: entry.path,
                    completedFiles: index,
                    totalFiles: totalFiles
                )
            )

            let sampler = Task { [weak self] in
                let startTime = Date()
                while !Task.isCancelled {
                    let inFlight = ModelDownloadProgress.inFlightBytes(
                        progress: progress,
                        expectedFileBytes: expectedBytes,
                        startTime: startTime
                    )
                    let currentCompleted = min(baseCompletedBytes + inFlight, totalBytes)
                    await MainActor.run {
                        guard !Task.isCancelled else { return }
                        self?.updateDownloadingState(
                            progress: Double(currentCompleted) / Double(totalBytes),
                            detail: ModelDownloadProgressFormatter.progressText(
                                completed: currentCompleted,
                                total: totalBytes
                            )
                        )
                    }
                    try? await Task.sleep(for: .milliseconds(200))
                }
            }
            defer { sampler.cancel() }

            let destination = try MLXModelStorageSupport.destinationFileURL(for: entry.path, under: tempDir)
            if MLXModelDownloadSupport.canReuseExistingDownload(
                at: destination,
                expectedSize: entry.size,
                fileManager: .default
            ) {
                let fileSize = Int64((try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
                completedBytes += max(expectedBytes, fileSize)
            } else {
                let remoteURL = try MLXModelDownloadSupport.fileResolveURL(
                    baseURL: baseURL,
                    repo: repoID.description,
                    path: entry.path
                )
                _ = try await ResumableModelDownloadSupport.download(
                    ResumableDownloadDescriptor(
                        sourceURL: remoteURL,
                        destinationURL: destination,
                        relativePath: entry.path,
                        expectedSize: expectedBytes > 0 ? expectedBytes : nil,
                        userAgent: MLXModelManager.hubUserAgent,
                        bearerToken: token,
                        disableProxy: MLXModelDownloadSupport.isMirrorHost(baseURL)
                    ),
                    progress: progress
                )
                completedBytes += max(expectedBytes, max(progress.completedUnitCount, 0))
            }

            updateDownloadingState(
                progress: min(1, Double(completedBytes) / Double(totalBytes)),
                detail: ModelDownloadProgressFormatter.fileProgressText(
                    currentFile: nil,
                    completedFiles: index + 1,
                    totalFiles: totalFiles
                )
            )
        }

        try await Task.detached(priority: .userInitiated) {
            guard MeetingSortformerModelStorage.isValidModelDirectory(tempDir) else {
                throw MLXModelDownloadSupport.DownloadValidationError.missingFiles
            }
            if FileManager.default.fileExists(atPath: modelDir.path) {
                try FileManager.default.removeItem(at: modelDir)
            }
            try FileManager.default.moveItem(at: tempDir, to: modelDir)
            _ = try SortformerModel.fromModelDirectory(modelDir)
        }.value
        return modelDir
    }

    private func updateDownloadingState(progress: Double, detail: String?) {
        guard state.isDownloading, downloadStorageRevision == storageRevision else { return }
        state = .downloading(progress: min(max(progress, 0), 1), detail: detail)
    }

    private static func preferredHubBaseURL() -> URL {
        let useMirror = UserDefaults.standard.object(forKey: AppPreferenceKey.useHfMirror) as? Bool ?? false
        return useMirror ? MLXModelManager.mirrorHubBaseURL : MLXModelManager.defaultHubBaseURL
    }

    private static func fallbackHubBaseURL(from baseURL: URL) -> URL? {
        guard baseURL.host?.contains("hf-mirror.com") != true else { return nil }
        return MLXModelManager.mirrorHubBaseURL
    }
}
