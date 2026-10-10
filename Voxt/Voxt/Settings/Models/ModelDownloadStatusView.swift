// ModelDownloadStatusView.swift
// Provides Model Download Status View for model settings.

import SwiftUI

struct ModelDownloadStatusSnapshot: Equatable {
    private enum Kind: Equatable {
        case standard
        case customLLM
    }

    let progress: Double
    private let isPaused: Bool
    private let kind: Kind
    private let completed: Int64
    private let total: Int64
    private let currentFile: String?
    private let currentFileCompleted: Int64
    private let currentFileTotal: Int64
    private let completedFiles: Int
    private let totalFiles: Int
    private let pauseMessage: String?

    private var displayedPercent: Int {
        let clampedProgress = max(0, min(progress, 1))
        let rawPercent = Int(clampedProgress * 100)
        return min(rawPercent, 99)
    }

    var titleText: String {
        let formatKey: String
        if isPaused {
            formatKey = kind == .customLLM ? "Custom LLM paused: %d%% • %@" : "Paused: %d%% • %@"
        } else {
            formatKey = kind == .customLLM ? "Custom LLM downloading: %d%% • %@" : "Downloading: %d%% • %@"
        }
        return AppLocalization.format(
            formatKey,
            displayedPercent,
            ModelDownloadProgressFormatter.progressText(completed: completed, total: total)
        )
    }

    var detailText: String {
        if isPaused {
            let progressText = ModelDownloadProgressFormatter.pausedFileProgressText(
                currentFile: currentFile,
                currentFileCompleted: currentFileCompleted,
                currentFileTotal: currentFileTotal,
                completedFiles: completedFiles,
                totalFiles: totalFiles
            )
            guard let pauseMessage, !pauseMessage.isEmpty else {
                return progressText
            }
            guard !progressText.isEmpty else { return pauseMessage }
            return AppLocalization.format("%@ • %@", pauseMessage, progressText)
        }
        return ModelDownloadProgressFormatter.fileProgressText(
            currentFile: currentFile,
            currentFileCompleted: currentFileCompleted,
            currentFileTotal: currentFileTotal,
            completedFiles: completedFiles,
            totalFiles: totalFiles
        )
    }

    static func fromMLXState(_ state: MLXModelManager.ModelState, pauseMessage: String? = nil) -> Self? {
        switch state {
        case .downloading(let progress, let completed, let total, let currentFile, let completedFiles, let totalFiles):
            return .init(
                progress: progress,
                isPaused: false,
                kind: .standard,
                completed: completed,
                total: total,
                currentFile: currentFile,
                currentFileCompleted: 0,
                currentFileTotal: 0,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                pauseMessage: nil
            )
        case .paused(let progress, let completed, let total, let currentFile, let completedFiles, let totalFiles):
            return .init(
                progress: progress,
                isPaused: true,
                kind: .standard,
                completed: completed,
                total: total,
                currentFile: currentFile,
                currentFileCompleted: 0,
                currentFileTotal: 0,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                pauseMessage: pauseMessage
            )
        default:
            return nil
        }
    }

    static func fromCustomLLMState(_ state: CustomLLMModelManager.ModelState, pauseMessage: String? = nil) -> Self? {
        switch state {
        case .downloading(let progress, let completed, let total, let currentFile, let completedFiles, let totalFiles):
            return .init(
                progress: progress,
                isPaused: false,
                kind: .customLLM,
                completed: completed,
                total: total,
                currentFile: currentFile,
                currentFileCompleted: 0,
                currentFileTotal: 0,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                pauseMessage: nil
            )
        case .paused(let progress, let completed, let total, let currentFile, let completedFiles, let totalFiles):
            return .init(
                progress: progress,
                isPaused: true,
                kind: .customLLM,
                completed: completed,
                total: total,
                currentFile: currentFile,
                currentFileCompleted: 0,
                currentFileTotal: 0,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                pauseMessage: pauseMessage
            )
        default:
            return nil
        }
    }

    static func fromGGUFState(_ state: GGUFTranslationModelManager.ModelState, pauseMessage: String? = nil) -> Self? {
        switch state {
        case .downloading(let progress, let completed, let total, let currentFile, let completedFiles, let totalFiles):
            return .init(
                progress: progress,
                isPaused: false,
                kind: .standard,
                completed: completed,
                total: total,
                currentFile: currentFile,
                currentFileCompleted: 0,
                currentFileTotal: 0,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                pauseMessage: nil
            )
        case .paused(let progress, let completed, let total, let currentFile, let completedFiles, let totalFiles):
            return .init(
                progress: progress,
                isPaused: true,
                kind: .standard,
                completed: completed,
                total: total,
                currentFile: currentFile,
                currentFileCompleted: 0,
                currentFileTotal: 0,
                completedFiles: completedFiles,
                totalFiles: totalFiles,
                pauseMessage: pauseMessage
            )
        default:
            return nil
        }
    }
}

struct ModelDownloadStatusView: View {
    let status: ModelDownloadStatusSnapshot

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: max(0, min(status.progress, 1)))
                .controlSize(.small)

            Text(status.titleText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()

            Text(status.detailText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
