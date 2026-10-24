// ModelDownloadProgressFormatter.swift
// Provides Model Download Progress Formatter for model settings.

import Foundation

enum ModelDownloadProgressFormatter {
    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        return formatter
    }()

    static func progressText(completed: Int64, total: Int64) -> String {
        let completedText = byteFormatter.string(fromByteCount: completed)
        if total > 0 {
            return AppLocalization.format(
                "Downloaded: %@ / %@",
                completedText,
                byteFormatter.string(fromByteCount: total)
            )
        }
        return AppLocalization.format("Downloaded: %@", completedText)
    }

    static func byteProgressText(completed: Int64, total: Int64) -> String {
        let completedText = byteFormatter.string(fromByteCount: completed)
        guard total > 0 else {
            return completedText
        }
        return "\(completedText) / \(byteFormatter.string(fromByteCount: total))"
    }

    static func fileProgressText(
        currentFile: String?,
        currentFileCompleted: Int64 = 0,
        currentFileTotal: Int64 = 0,
        completedFiles: Int,
        totalFiles: Int
    ) -> String {
        let filesText: String
        if totalFiles > 0 {
            filesText = AppLocalization.format("%d/%d files", completedFiles, totalFiles)
        } else {
            filesText = AppLocalization.format("%d files", completedFiles)
        }

        guard let currentFile, !currentFile.isEmpty else {
            if totalFiles > 0, completedFiles >= totalFiles {
                return AppLocalization.format("Finalizing download… (%@)", filesText)
            }
            return AppLocalization.format("Preparing download… (%@)", filesText)
        }

        let fileName = (currentFile as NSString).lastPathComponent
        guard currentFileTotal > 0 else {
            return AppLocalization.format("Downloading: %@ (%@)", fileName, filesText)
        }

        let currentProgressText = progressText(
            completed: min(max(currentFileCompleted, 0), currentFileTotal),
            total: currentFileTotal
        )
        return AppLocalization.format("Downloading: %@ (%@ · %@)", fileName, currentProgressText, filesText)
    }

    static func pausedFileProgressText(
        currentFile: String?,
        currentFileCompleted: Int64 = 0,
        currentFileTotal: Int64 = 0,
        completedFiles: Int,
        totalFiles: Int
    ) -> String {
        let filesText: String
        if totalFiles > 0 {
            filesText = AppLocalization.format("%d/%d files", completedFiles, totalFiles)
        } else {
            filesText = AppLocalization.format("%d files", completedFiles)
        }

        guard let currentFile, !currentFile.isEmpty else {
            return AppLocalization.format("Paused. Ready to continue. (%@)", filesText)
        }

        let fileName = (currentFile as NSString).lastPathComponent
        guard currentFileTotal > 0 else {
            return AppLocalization.format("Paused: %@ (%@)", fileName, filesText)
        }

        let currentProgressText = progressText(
            completed: min(max(currentFileCompleted, 0), currentFileTotal),
            total: currentFileTotal
        )
        return AppLocalization.format("Paused: %@ (%@ · %@)", fileName, currentProgressText, filesText)
    }
}
