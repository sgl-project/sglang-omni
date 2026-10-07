// MLXModelDownloadSupport.swift
// Provides MLXModel Download Support for transcription engines.

import Foundation
import CFNetwork
import HuggingFace

enum MLXModelDownloadSupport {
    private static let modelEntryAllowedExtensions: Set<String> = ["safetensors", "json", "txt", "wav", "jinja", "model", "mvn"]
    nonisolated static let whisperTokenizerAssetPaths: [String] = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "normalizer.json",
        "generation_config.json",
    ]
    nonisolated enum DownloadValidationError: LocalizedError {
        case missingFiles
        case sizeMismatch(expected: Int64, actual: Int64)
        case emptyFileList

        var errorDescription: String? {
            switch self {
            case .missingFiles:
                return "Downloaded files are incomplete."
            case .sizeMismatch(let expected, let actual):
                let expectedText = ByteCountFormatter.string(fromByteCount: expected, countStyle: .file)
                let actualText = ByteCountFormatter.string(fromByteCount: actual, countStyle: .file)
                return "Download incomplete (expected ~\(expectedText), got \(actualText))."
            case .emptyFileList:
                return "No downloadable files were found for this model."
            }
        }
    }

    enum DownloadNetworkError: LocalizedError {
        case mirrorRejected(statusCode: Int)
        case modelUnavailable(repo: String, statusCode: Int)
        case metadataRequestFailed(statusCode: Int)
        case invalidServerResponse

        var errorDescription: String? {
            switch self {
            case .mirrorRejected(let statusCode):
                return "China mirror rejected request (HTTP \(statusCode))."
            case .modelUnavailable(let repo, let statusCode):
                return "Model repository unavailable (\(repo), HTTP \(statusCode))."
            case .metadataRequestFailed(let statusCode):
                return "Model metadata request failed (HTTP \(statusCode))."
            case .invalidServerResponse:
                return "Invalid response from model server."
            }
        }
    }

    struct ModelFileEntry: Hashable {
        let path: String
        let size: Int64?
    }

    static func canReuseExistingDownload(
        at destinationURL: URL,
        expectedSize: Int64?,
        fileManager: FileManager
    ) -> Bool {
        guard fileManager.fileExists(atPath: destinationURL.path),
              let fileSize = (try? destinationURL.resourceValues(forKeys: [.fileSizeKey]).fileSize)
        else {
            return false
        }

        let size = Int64(fileSize)
        if let expectedSize, expectedSize > 0 {
            return size == expectedSize
        }
        return size > 0
    }

    static func isRetryableTransportError(_ error: Error) -> Bool {
        if let urlError = error as? URLError {
            switch urlError.code {
            case .cancelled,
                 .timedOut,
                 .networkConnectionLost,
                 .notConnectedToInternet,
                 .cannotFindHost,
                 .cannotConnectToHost,
                 .dnsLookupFailed,
                 .resourceUnavailable,
                 .cannotLoadFromNetwork,
                 .badServerResponse:
                return true
            default:
                return false
            }
        }

        if let httpError = error as? HTTPClientError {
            switch httpError {
            case .requestError, .unexpectedError:
                return true
            case .responseError(let response, _):
                return response.statusCode >= 500 || response.statusCode == 429 || response.statusCode == 408
            case .decodingError:
                return false
            }
        }

        return false
    }

    static func pauseMessageForInterruptedDownload(_ error: Error) -> String? {
        if let conflictMessage = VoxtNetworkSession.directModeConflictMessage(for: error) {
            return conflictMessage
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            switch nsError.code {
            case NSURLErrorNotConnectedToInternet:
                return AppLocalization.localizedString("Network issue detected. Check your connection, then click Continue to resume.")
            case NSURLErrorTimedOut,
                 NSURLErrorCannotConnectToHost,
                 NSURLErrorNetworkConnectionLost,
                 NSURLErrorCannotFindHost,
                 NSURLErrorDNSLookupFailed,
                 NSURLErrorResourceUnavailable,
                 NSURLErrorCannotLoadFromNetwork:
                return AppLocalization.localizedString("Network issue detected. Check your network or proxy settings, then click Continue to resume.")
            default:
                break
            }
        }

        if let loopError = error as? ResumableDownloadLoopError,
           loopError.recoverableReason == "stall-timeout"
        {
            return AppLocalization.localizedString("Download stalled due to a network issue. Click Continue to resume.")
        }

        return nil
    }

    static func makeDownloadSession(for baseURL: URL) -> URLSession {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 60 * 60
        configuration.waitsForConnectivity = false

        if isMirrorHost(baseURL) {
            configuration.connectionProxyDictionary = [
                kCFNetworkProxiesHTTPEnable as String: false,
                kCFNetworkProxiesHTTPSEnable as String: false,
                kCFNetworkProxiesSOCKSEnable as String: false,
            ]
        }

        return URLSession(configuration: configuration)
    }

    static func fetchModelEntries(
        repo: String,
        baseURL: URL,
        session: URLSession,
        userAgent: String
    ) async throws -> [ModelFileEntry] {
        guard let encoded = repo.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else {
            throw URLError(.badURL)
        }
        let base = baseURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let url = URL(string: "\(base)/api/models/\(encoded)/tree/main?recursive=1")
        else {
            throw URLError(.badURL)
        }

        var request = URLRequest(url: url)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 30

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw DownloadNetworkError.invalidServerResponse
        }
        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            if isMirrorHost(baseURL), [401, 403].contains(httpResponse.statusCode) {
                throw DownloadNetworkError.mirrorRejected(statusCode: httpResponse.statusCode)
            }
            if [401, 404].contains(httpResponse.statusCode) {
                throw DownloadNetworkError.modelUnavailable(repo: repo, statusCode: httpResponse.statusCode)
            }
            throw DownloadNetworkError.metadataRequestFailed(statusCode: httpResponse.statusCode)
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
        return json.compactMap { item in
            guard (item["type"] as? String) == "file" else { return nil }
            let path = (item["path"] as? String) ?? ""
            let ext = path.split(separator: ".").last.map(String.init) ?? ""
            guard modelEntryAllowedExtensions.contains(ext.lowercased()) else { return nil }
            let size: Int64?
            if let raw = item["size"] as? Int {
                size = Int64(raw)
            } else if let raw = item["size"] as? Int64 {
                size = raw
            } else {
                size = nil
            }
            return ModelFileEntry(path: path, size: size)
        }
    }

    static func fetchModelSizeInfo(
        repo: String,
        baseURL: URL,
        userAgent: String,
        formatByteCount: @Sendable (Int64) -> String
    ) async throws -> (bytes: Int64, text: String) {
        let entries = try await fetchModelEntries(
            repo: repo,
            baseURL: baseURL,
            session: makeDownloadSession(for: baseURL),
            userAgent: userAgent
        )
        let total = entries.reduce(Int64(0)) { partial, entry in
            partial + max(entry.size ?? 0, 0)
        }

        guard total > 0 else { return (0, "Unknown") }
        return (total, formatByteCount(total))
    }

    static func fileResolveURL(baseURL: URL, repo: String, path: String) throws -> URL {
        let base = baseURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let encodedPath = path
            .split(separator: "/")
            .map { component in
                String(component).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String(component)
            }
            .joined(separator: "/")
        guard let url = URL(string: "\(base)/\(repo)/resolve/main/\(encodedPath)?download=true") else {
            throw URLError(.badURL)
        }
        return url
    }

    /// Validation belongs to the downloaded repo, never the UI's selected repo.
    nonisolated static func validationSizeState(for repo: String) -> MLXModelManager.ModelSizeState {
        guard let size = MLXModelCatalog.fallbackRemoteSizeInfo(repo: repo) else {
            return .error("Size unavailable")
        }
        return .ready(bytes: size.bytes, text: size.text)
    }

    @concurrent
    nonisolated static func validateDownloadedModelInBackground(
        at url: URL,
        repo: String? = nil,
        sizeState: MLXModelManager.ModelSizeState,
        downloadSizeTolerance: Double,
        fileManager: FileManager
    ) async throws {
        try await ModelDiskOperations.perform {
            try validateDownloadedModel(at: url, repo: repo, sizeState: sizeState,
                downloadSizeTolerance: downloadSizeTolerance, fileManager: fileManager)
        }
    }

    nonisolated static func validateDownloadedModel(
        at url: URL,
        repo: String? = nil,
        sizeState: MLXModelManager.ModelSizeState,
        downloadSizeTolerance: Double,
        fileManager: FileManager
    ) throws {
        let files = allFiles(at: url, fileManager: fileManager)
        let hasWeights = files.contains { file in
            guard file.pathExtension.lowercased() == "safetensors" else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        }
        let configValid = files.contains { file in
            guard file.lastPathComponent.lowercased() == "config.json" else { return false }
            guard let data = try? Data(contentsOf: file) else { return false }
            return (try? JSONSerialization.jsonObject(with: data)) != nil
        }

        guard hasWeights, configValid, ModelWeightFileValidation.hasCompleteIndex(in: url) else {
            throw DownloadValidationError.missingFiles
        }

        if !hasRequiredAuxiliaryFiles(at: url, repo: repo, files: files, fileManager: fileManager) {
            throw DownloadValidationError.missingFiles
        }

        if case .ready(let expectedBytes, _) = sizeState,
           expectedBytes > 0,
           let actualBytesRaw = try? fileManager.allocatedSizeOfDirectory(at: url)
        {
            let actualBytes = Int64(actualBytesRaw)
            let minimumBytes = Int64(Double(expectedBytes) * downloadSizeTolerance)
            if actualBytes < minimumBytes {
                throw DownloadValidationError.sizeMismatch(expected: expectedBytes, actual: actualBytes)
            }
        }
    }

    nonisolated static func clearDirectory(at url: URL, fileManager: FileManager) throws {
        guard fileManager.fileExists(atPath: url.path) else { return }
        do {
            try fileManager.removeItem(at: url)
        } catch {
            let contents = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            for item in contents {
                try? fileManager.removeItem(at: item)
            }
            try fileManager.removeItem(at: url)
        }
    }

    nonisolated static func isModelDirectoryValid(
        _ directory: URL,
        repo: String? = nil,
        fileManager: FileManager
    ) -> Bool {
        guard fileManager.fileExists(atPath: directory.path) else { return false }

        if let topLevelItems = try? fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey]) {
            let malformed = topLevelItems.contains { item in
                let isDirectory = (try? item.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
                guard isDirectory else { return false }
                let ext = item.pathExtension.lowercased()
                return ext == "json" || ext == "safetensors" || ext == "txt" || ext == "wav"
            }
            if malformed {
                return false
            }
        }

        let files = allFiles(at: directory, fileManager: fileManager)
        let hasWeights = files.contains { file in
            guard file.pathExtension.lowercased() == "safetensors" else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        }
        let rootConfig = directory.appendingPathComponent("config.json")
        guard fileManager.fileExists(atPath: rootConfig.path),
              let rootConfigData = try? Data(contentsOf: rootConfig),
              (try? JSONSerialization.jsonObject(with: rootConfigData)) != nil
        else {
            return false
        }

        return hasWeights && ModelWeightFileValidation.hasCompleteIndex(in: directory) && hasRequiredAuxiliaryFiles(
            at: directory,
            repo: repo,
            files: files,
            fileManager: fileManager
        )
    }

    static func missingRequiredRepairEntries(
        at directory: URL,
        repo: String,
        availableEntries: [ModelFileEntry],
        fileManager: FileManager
    ) -> [ModelFileEntry] {
        guard resolvedModelType(at: directory, repo: repo, fileManager: fileManager) == "sensevoice" else {
            return []
        }

        let hasTokenizerAsset = hasSenseVoiceTokenizerAssets(at: directory, fileManager: fileManager)
        let hasCMVNAsset = fileManager.fileExists(atPath: directory.appendingPathComponent("am.mvn").path)
        guard !hasTokenizerAsset || !hasCMVNAsset else { return [] }

        return availableEntries.filter { entry in
            let path = entry.path.lowercased()
            if !hasTokenizerAsset,
               (path.hasSuffix(".model") || path.hasSuffix("/tokenizer.json") || path.hasSuffix("/tokens.json")) {
                return true
            }
            if !hasCMVNAsset, path.hasSuffix("/am.mvn") || path == "am.mvn" {
                return true
            }
            return false
        }
    }

    nonisolated static func whisperTokenizerRepo(for repo: String) -> String? {
        let lowercasedRepo = repo.lowercased()
        guard lowercasedRepo.contains("whisper") else { return nil }
        if lowercasedRepo.contains("tiny") {
            return "openai/whisper-tiny"
        }
        if lowercasedRepo.contains("base") {
            return "openai/whisper-base"
        }
        if lowercasedRepo.contains("small") {
            return "openai/whisper-small"
        }
        if lowercasedRepo.contains("medium") {
            return "openai/whisper-medium"
        }
        if lowercasedRepo.contains("large-v2") {
            return "openai/whisper-large-v2"
        }
        return "openai/whisper-large-v3"
    }

    nonisolated static func missingWhisperTokenizerAssetPaths(
        at directory: URL,
        fileManager: FileManager
    ) -> [String] {
        whisperTokenizerAssetPaths.filter { path in
            !fileManager.fileExists(atPath: directory.appendingPathComponent(path).path)
        }
    }

    static func isMirrorHost(_ url: URL) -> Bool {
        url.host?.contains("hf-mirror.com") == true
    }

    private nonisolated static func allFiles(at root: URL, fileManager: FileManager) -> [URL] {
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var files: [URL] = []
        for case let fileURL as URL in enumerator {
            let isRegular = (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile ?? false
            if isRegular {
                files.append(fileURL)
            }
        }
        return files
    }

    private nonisolated static func hasRequiredAuxiliaryFiles(
        at directory: URL,
        repo: String?,
        files _: [URL],
        fileManager: FileManager
    ) -> Bool {
        switch resolvedModelType(at: directory, repo: repo, fileManager: fileManager) {
        case "whisper":
            return missingWhisperTokenizerAssetPaths(at: directory, fileManager: fileManager).isEmpty
        case "sensevoice":
            return hasSenseVoiceTokenizerAssets(at: directory, fileManager: fileManager)
                && fileManager.fileExists(atPath: directory.appendingPathComponent("am.mvn").path)
        default:
            guard let repo, whisperTokenizerRepo(for: repo) != nil else { return true }
            return missingWhisperTokenizerAssetPaths(at: directory, fileManager: fileManager).isEmpty
        }
    }

    private nonisolated static func resolvedModelType(
        at directory: URL,
        repo: String?,
        fileManager: FileManager
    ) -> String? {
        let configURL = directory.appendingPathComponent("config.json")
        if fileManager.fileExists(atPath: configURL.path),
           let data = try? Data(contentsOf: configURL),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let modelType = object["model_type"] as? String,
           !modelType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return modelType.lowercased()
        }
        return repo?.lowercased().contains("sensevoice") == true ? "sensevoice" : nil
    }

    private nonisolated static func hasSenseVoiceTokenizerAssets(at directory: URL, fileManager: FileManager) -> Bool {
        if fileManager.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path) {
            return true
        }
        if fileManager.fileExists(atPath: directory.appendingPathComponent("tokens.json").path) {
            return true
        }
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }
        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension.lowercased() == "model" {
                return true
            }
        }
        return false
    }
}
