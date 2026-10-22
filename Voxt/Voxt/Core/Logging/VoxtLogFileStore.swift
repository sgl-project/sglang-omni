// VoxtLogFileStore.swift
// Stores local rolling diagnostic logs for user export.

import Darwin
import Foundation

nonisolated final class VoxtLogFileStore: @unchecked Sendable {
    nonisolated static let shared = VoxtLogFileStore()
    private nonisolated static let queue = DispatchQueue(label: "app.voxt.logging.file-store", qos: .utility)

    private let fileManager: FileManager
    private let maxFileBytes: UInt64
    private let retainedArchiveCount: Int
    private let baseDirectoryURL: URL?
    private var recentLines: [String] = []
    private let maxMemoryLines = 2_000
    private var didPrepareStorage = false

    init(
        fileManager: FileManager = FileManager(),
        maxFileBytes: UInt64 = 2 * 1_024 * 1_024,
        retainedArchiveCount: Int = 5,
        baseDirectoryURL: URL? = nil
    ) {
        self.fileManager = fileManager
        self.maxFileBytes = maxFileBytes
        self.retainedArchiveCount = retainedArchiveCount
        self.baseDirectoryURL = baseDirectoryURL
    }

    var logsDirectoryURL: URL {
        if let baseDirectoryURL {
            return baseDirectoryURL
        }
        let supportDirectory = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let base = supportDirectory ?? fileManager.temporaryDirectory
        return base
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("Logs", isDirectory: true)
    }

    var currentLogURL: URL {
        logsDirectoryURL.appendingPathComponent("current.log")
    }

    var archiveDirectoryURL: URL {
        logsDirectoryURL.appendingPathComponent("archive", isDirectory: true)
    }

    var legacyDirectoryURL: URL {
        logsDirectoryURL.appendingPathComponent("legacy", isDirectory: true)
    }

    func append(_ line: String) {
        Self.queue.async { [weak self] in
            self?.appendSynchronously(line)
        }
    }

    func latestLogUpdateDate() -> Date? {
        Self.queue.sync {
            prepareStorageIfNeeded()
            return (try? fileManager.attributesOfItem(atPath: currentLogURL.path)[.modificationDate]) as? Date
        }
    }

    func latestLines(limit: Int) -> [String] {
        Self.queue.sync {
            prepareStorageIfNeeded()
            let selectedLimit = max(1, limit)
            var lines = recentLines
            if lines.count < selectedLimit {
                lines = readPersistedLines(limit: selectedLimit)
            }
            return Array(lines.suffix(selectedLimit))
        }
    }

    func flush() {
        Self.queue.sync {
            prepareStorageIfNeeded()
        }
    }

    private func appendSynchronously(_ line: String) {
        prepareStorageIfNeeded()
        recentLines.append(line)
        if recentLines.count > maxMemoryLines {
            recentLines = Array(recentLines.suffix(maxMemoryLines))
        }

        rotateIfNeeded(incomingLine: line)
        appendLineData(Data((line + "\n").utf8))
    }

    private func appendLineData(_ data: Data) {
        let descriptor = open(currentLogURL.path, O_WRONLY | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { return }
        defer { close(descriptor) }

        data.withUnsafeBytes { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            var bytesWritten = 0
            while bytesWritten < buffer.count {
                let result = write(
                    descriptor,
                    baseAddress.advanced(by: bytesWritten),
                    buffer.count - bytesWritten
                )
                guard result > 0 else { return }
                bytesWritten += result
            }
        }
    }

    private func prepareStorageIfNeeded() {
        guard !didPrepareStorage else { return }
        didPrepareStorage = true
        _ = try? fileManager.createDirectory(at: logsDirectoryURL, withIntermediateDirectories: true)
        _ = try? fileManager.createDirectory(at: archiveDirectoryURL, withIntermediateDirectories: true)
        migrateLegacyLogIfNeeded()
        removePersistedUnitTestLaunchLogsIfNeeded()
        recentLines = readPersistedLines(limit: maxMemoryLines)
    }

    private func migrateLegacyLogIfNeeded() {
        let legacyLogURL = logsDirectoryURL.appendingPathComponent("voxt.log")
        guard fileManager.fileExists(atPath: legacyLogURL.path) else { return }
        try? fileManager.createDirectory(at: legacyDirectoryURL, withIntermediateDirectories: true)
        let dateText = legacyDateFormatter.string(from: Date())
        let destination = legacyDirectoryURL.appendingPathComponent("voxt-legacy-\(dateText).log")
        if fileManager.fileExists(atPath: destination.path) {
            try? fileManager.removeItem(at: destination)
        }
        try? fileManager.moveItem(at: legacyLogURL, to: destination)
    }

    private func rotateIfNeeded(incomingLine: String) {
        let currentSize = ((try? fileManager.attributesOfItem(atPath: currentLogURL.path)[.size]) as? NSNumber)?.uint64Value ?? 0
        let incomingSize = UInt64(Data((incomingLine + "\n").utf8).count)
        guard currentSize + incomingSize > maxFileBytes, currentSize > 0 else { return }

        for index in stride(from: retainedArchiveCount, through: 1, by: -1) {
            let source = archiveDirectoryURL.appendingPathComponent("voxt-\(index).log")
            let destination = archiveDirectoryURL.appendingPathComponent("voxt-\(index + 1).log")
            if fileManager.fileExists(atPath: destination.path) {
                try? fileManager.removeItem(at: destination)
            }
            if fileManager.fileExists(atPath: source.path) {
                try? fileManager.moveItem(at: source, to: destination)
            }
        }

        let firstArchive = archiveDirectoryURL.appendingPathComponent("voxt-1.log")
        if fileManager.fileExists(atPath: firstArchive.path) {
            try? fileManager.removeItem(at: firstArchive)
        }
        try? fileManager.moveItem(at: currentLogURL, to: firstArchive)

        let overflowArchive = archiveDirectoryURL.appendingPathComponent("voxt-\(retainedArchiveCount + 1).log")
        if fileManager.fileExists(atPath: overflowArchive.path) {
            try? fileManager.removeItem(at: overflowArchive)
        }
    }

    private func readPersistedLines(limit: Int) -> [String] {
        var lines: [String] = []
        for url in readableLogURLs() {
            guard let text = try? String(contentsOf: url, encoding: .utf8), !text.isEmpty else { continue }
            lines.append(contentsOf: Self.removingUnitTestLaunchLines(from: text.split(whereSeparator: \.isNewline).map(String.init)))
            if lines.count > limit * 2 {
                lines = Array(lines.suffix(limit))
            }
        }
        return Array(lines.suffix(limit))
    }

    private func readableLogURLs() -> [URL] {
        let archived = (1...retainedArchiveCount)
            .reversed()
            .map { archiveDirectoryURL.appendingPathComponent("voxt-\($0).log") }
        return archived + [currentLogURL]
    }

    private func removePersistedUnitTestLaunchLogsIfNeeded() {
        for url in readableLogURLs() where fileManager.fileExists(atPath: url.path) {
            guard let text = try? String(contentsOf: url, encoding: .utf8), !text.isEmpty else { continue }
            let originalLines = text.split(whereSeparator: \.isNewline).map(String.init)
            let cleanedLines = Self.removingUnitTestLaunchLines(from: originalLines)
            guard cleanedLines.count != originalLines.count else { continue }
            try? cleanedLines.joined(separator: "\n").appending(cleanedLines.isEmpty ? "" : "\n").write(
                to: url,
                atomically: true,
                encoding: .utf8
            )
        }
    }

    static func removingUnitTestLaunchLines(from lines: [String]) -> [String] {
        var removedIndexes = Set<Int>()
        for (index, line) in lines.enumerated() where line.contains("Voxt launch running under XCTest") {
            removedIndexes.insert(index)
            let runtimeIndex = index - 1
            if runtimeIndex >= 0, lines[runtimeIndex].contains("[app] Runtime system version:") {
                removedIndexes.insert(runtimeIndex)
            }
            let launchIndex = index - 2
            if launchIndex >= 0, lines[launchIndex].contains("[app] Voxt launching.") {
                removedIndexes.insert(launchIndex)
            }
        }
        guard !removedIndexes.isEmpty else { return lines }
        return lines.enumerated().compactMap { removedIndexes.contains($0.offset) ? nil : $0.element }
    }

    private var legacyDateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd"
        return formatter
    }
}
