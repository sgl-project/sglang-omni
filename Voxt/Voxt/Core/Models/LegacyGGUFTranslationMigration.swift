// LegacyGGUFTranslationMigration.swift
// Moves retired llama.cpp Hy-MT2 translation selections onto the MLX checkpoints.

import Foundation

enum LegacyGGUFTranslationMigration {
    nonisolated static let selectionPrefix = "local-gguf-translation:"
    nonisolated static let providerRawValue = "localGGUF"
    nonisolated static let q8ModelID = "tencent/Hy-MT2-1.8B-GGUF#Hy-MT2-1.8B-Q8_0.gguf"
    nonisolated static let defaultRepo = "mlx-community/Hy-MT2-1.8B-4bit"
    nonisolated static let q8Repo = "mlx-community/Hy-MT2-1.8B-8bit"
    nonisolated private static let storageDirectoryName = "gguf-translation"

    /// Q8_0 keeps its precision tier; Q4_K_M and unknown IDs map to the 4-bit default.
    nonisolated static func repo(forLegacyModelID modelID: String?) -> String {
        modelID?.trimmingCharacters(in: .whitespacesAndNewlines) == q8ModelID ? q8Repo : defaultRepo
    }

    /// Runs after `FeatureSettingsStore.migrateIfNeeded`, which has already consumed
    /// the legacy model ID. MLX cannot load the GGUF weights, so reclaim their space.
    static func removeRetiredArtifacts(
        defaults: UserDefaults = .standard,
        fileManager: FileManager = .default
    ) {
        defaults.removeObject(forKey: AppPreferenceKey.legacyTranslationGGUFModelID)
        for root in ModelStorageDirectoryManager.resolvedReadableRootURLs(defaults: defaults) {
            let directory = root.appendingPathComponent(storageDirectoryName, isDirectory: true)
            guard fileManager.fileExists(atPath: directory.path) else { continue }
            do {
                try fileManager.removeItem(at: directory)
                VoxtLog.model("Removed retired GGUF translation models. path=\(directory.path)")
            } catch {
                VoxtLog.modelWarning("Failed to remove retired GGUF translation models. path=\(directory.path), error=\(error.localizedDescription)")
            }
        }
    }
}
