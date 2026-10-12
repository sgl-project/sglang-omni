// VoxtLogExportStore.swift
// Provides display and export payloads for local diagnostic logs.

import Foundation

enum VoxtLogExportStore {
    nonisolated static func latestLogUpdateDate() -> Date? {
        VoxtLogFileStore.shared.latestLogUpdateDate()
    }

    nonisolated static func latestLogDisplayText(limit: Int = 1_000, metadataProvider: () -> String) -> String {
        let selectedLines = VoxtLogFileStore.shared.latestLines(limit: limit)
        let unavailableText = MainActorSync.run {
            AppLocalization.localizedString("No logs available")
        }
        let appMetaTitle = MainActorSync.run {
            AppLocalization.localizedString("App Meta")
        }
        let logText = selectedLines.isEmpty
            ? "[Voxt] <\(unavailableText)>"
            : selectedLines.joined(separator: "\n")

        return [
            logText,
            "========== \(appMetaTitle) ==========",
            VoxtLogRedactor.redact(metadataProvider())
        ].joined(separator: "\n\n")
    }
}
