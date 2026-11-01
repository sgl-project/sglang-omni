// ModelDownloadBadgeSupport.swift
// Provides Model Download Badge Support for settings shell.

import Foundation

enum SettingsModelDownloadBadgeSupport {
    static func activeDownloadCount(
        mlxActiveDownloadRepos: Set<String>,
        customLLMActiveDownloadRepos: Set<String>
    ) -> Int {
        mlxActiveDownloadRepos.count + customLLMActiveDownloadRepos.count
    }
}
