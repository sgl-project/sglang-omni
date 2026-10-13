// HistoryAudioArchiveService.swift
// Provides History Audio Archive Service for history storage and display support.

import Foundation

@MainActor
protocol HistoryAudioArchiveManaging: AnyObject {
    func importArchive(
        from sourceURL: URL,
        kind: TranscriptionHistoryKind,
        preferredFileName: String?
    ) throws -> String
    func replaceArchive(for entry: TranscriptionHistoryEntry, with sourceURL: URL) throws -> String
    func audioURL(for entry: TranscriptionHistoryEntry) -> URL?
    func removeArchive(for entry: TranscriptionHistoryEntry)
    func removeArchive(relativePath: String?)
    func exportAllArchives(
        to destinationDirectoryURL: URL,
        forEachHistoryBatch: (([TranscriptionHistoryEntry]) -> Void) -> Void
    ) throws -> HistoryAudioExportSummary
    func storageStats(audioPaths: [String]) -> HistoryAudioStorageStats
    func rootURL() throws -> URL
}

final class HistoryAudioArchiveService: HistoryAudioArchiveManaging {
    private let fileManager: FileManager

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    func importArchive(
        from sourceURL: URL,
        kind: TranscriptionHistoryKind,
        preferredFileName: String? = nil
    ) throws -> String {
        let resolvedFileName = sanitizedFileName(
            preferredFileName?.trimmingCharacters(in: .whitespacesAndNewlines),
            fallbackKind: kind
        )
        let relativePath = "\(folderName(for: kind))/\(resolvedFileName)"
        let destinationURL = try rootURL().appendingPathComponent(relativePath)
        try fileManager.createDirectory(at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.moveItem(at: sourceURL, to: destinationURL)
        return relativePath
    }

    func replaceArchive(for entry: TranscriptionHistoryEntry, with sourceURL: URL) throws -> String {
        let relativePath = entry.audioRelativePath ?? entry.transcriptAudioRelativePath
        let resolvedRelativePath = relativePath?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? relativePath!
            : "\(folderName(for: entry.kind))/\(sanitizedFileName(nil, fallbackKind: entry.kind))"
        let destinationURL = try rootURL().appendingPathComponent(resolvedRelativePath)
        try fileManager.createDirectory(at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.moveItem(at: sourceURL, to: destinationURL)
        return resolvedRelativePath
    }

    func audioURL(for entry: TranscriptionHistoryEntry) -> URL? {
        let relativePath = entry.audioRelativePath ?? entry.transcriptAudioRelativePath
        guard let relativePath, !relativePath.isEmpty else {
            return nil
        }
        return try? rootURL().appendingPathComponent(relativePath)
    }

    func removeArchive(for entry: TranscriptionHistoryEntry) {
        let relativePath = entry.audioRelativePath ?? entry.transcriptAudioRelativePath
        removeArchive(relativePath: relativePath)
    }

    func removeArchive(relativePath: String?) {
        guard let relativePath, !relativePath.isEmpty else { return }
        do {
            let url = try rootURL().appendingPathComponent(relativePath)
            if fileManager.fileExists(atPath: url.path) {
                try fileManager.removeItem(at: url)
            }
        } catch {
            return
        }
    }

    func exportAllArchives(
        to destinationDirectoryURL: URL,
        forEachHistoryBatch: (([TranscriptionHistoryEntry]) -> Void) -> Void
    ) throws -> HistoryAudioExportSummary {
        try fileManager.createDirectory(at: destinationDirectoryURL, withIntermediateDirectories: true)

        var exportedCount = 0
        var skippedCount = 0
        var failedCount = 0

        forEachHistoryBatch { batch in
            for entry in batch {
                guard let sourceURL = audioURL(for: entry),
                      fileManager.fileExists(atPath: sourceURL.path)
                else {
                    skippedCount += 1
                    continue
                }

                do {
                    let folderURL = destinationDirectoryURL.appendingPathComponent(folderName(for: entry.kind), isDirectory: true)
                    try fileManager.createDirectory(at: folderURL, withIntermediateDirectories: true)
                    let destinationURL = folderURL.appendingPathComponent(exportFileName(for: entry))
                    if fileManager.fileExists(atPath: destinationURL.path) {
                        try fileManager.removeItem(at: destinationURL)
                    }
                    try fileManager.copyItem(at: sourceURL, to: destinationURL)
                    exportedCount += 1
                } catch {
                    failedCount += 1
                }
            }
        }

        return HistoryAudioExportSummary(
            exportedCount: exportedCount,
            skippedCount: skippedCount,
            failedCount: failedCount
        )
    }

    func storageStats(audioPaths: [String]) -> HistoryAudioStorageStats {
        Self.storageStats(rootURL: try? rootURL(), audioPaths: audioPaths)
    }

    func rootURL() throws -> URL {
        try HistoryAudioStorageDirectoryManager.ensureRootDirectoryExists()
    }

    nonisolated static func storageStats(rootURL: URL?, audioPaths: [String]) -> HistoryAudioStorageStats {
        var storedFileCount = 0
        var totalBytes: Int64 = 0
        let fileManager = FileManager.default

        for relativePath in audioPaths {
            guard let sourceURL = rootURL?.appendingPathComponent(relativePath),
                  fileManager.fileExists(atPath: sourceURL.path)
            else {
                continue
            }

            storedFileCount += 1
            let fileSize = (try? sourceURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            totalBytes += Int64(fileSize)
        }

        return HistoryAudioStorageStats(
            storedFileCount: storedFileCount,
            totalBytes: totalBytes
        )
    }

    private func folderName(for kind: TranscriptionHistoryKind) -> String {
        switch kind {
        case .normal:
            return "transcription"
        case .translation:
            return "translation"
        case .rewrite:
            return "rewrite"
        case .transcript:
            return "transcript"
        }
    }

    private func sanitizedFileName(_ preferredFileName: String?, fallbackKind: TranscriptionHistoryKind) -> String {
        let trimmedPreferred = preferredFileName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let baseName = trimmedPreferred.isEmpty ? "\(folderName(for: fallbackKind))-\(UUID().uuidString)" : trimmedPreferred
        let filtered = baseName.map { character -> Character in
            if character.isLetter || character.isNumber || character == "-" || character == "_" {
                return character
            }
            return "-"
        }
        let normalized = String(filtered).trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
        let resolved = normalized.isEmpty ? "\(folderName(for: fallbackKind))-\(UUID().uuidString)" : normalized
        return resolved.hasSuffix(".wav") ? resolved : "\(resolved).wav"
    }

    private func exportFileName(for entry: TranscriptionHistoryEntry) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: entry.createdAt))-\(folderName(for: entry.kind))-\(entry.id.uuidString).wav"
    }
}

struct HistoryAudioExportSummary: Equatable {
    let exportedCount: Int
    let skippedCount: Int
    let failedCount: Int
}

struct HistoryAudioStorageStats: Equatable {
    let storedFileCount: Int
    let totalBytes: Int64
}
