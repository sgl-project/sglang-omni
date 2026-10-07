// ModelDownloadPresentationSupport.swift
// Provides Model Download Presentation Support for model settings.

import Foundation

enum ModelDownloadPresentationSupport {
    static func statusText(
        downloadState: DownloadState,
        errorMessage: String? = nil
    ) -> String {
        switch downloadState {
        case .idle:
            if let errorMessage, !errorMessage.isEmpty {
                return "Error: \(errorMessage)"
            }
            return ""
        case .downloading(let completed, let total):
            return AppLocalization.format(
                "Downloading %@",
                ModelDownloadProgressFormatter.byteProgressText(completed: completed, total: total)
            )
        case .paused(let completed, let total, let pauseMessage):
            let progressText = ModelDownloadProgressFormatter.byteProgressText(completed: completed, total: total)
            if let pauseMessage, !pauseMessage.isEmpty {
                return AppLocalization.format("%@ • %@", pauseMessage, progressText)
            }
            return AppLocalization.format("Paused %@", progressText)
        }
    }

    enum DownloadState {
        case idle
        case downloading(completed: Int64, total: Int64)
        case paused(completed: Int64, total: Int64, pauseMessage: String?)
    }
}
