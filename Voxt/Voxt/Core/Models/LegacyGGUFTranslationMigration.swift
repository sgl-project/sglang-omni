// LegacyGGUFTranslationMigration.swift
// Moves retired llama.cpp Hy-MT2 translation selections onto the MLX checkpoints and reclaims the GGUF weights.

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

    /// The XCTest host shares the user's model roots, so only the shipping app reclaims them.
    @discardableResult
    static func reclaimRetiredArtifactsAfterLaunch(defaults: UserDefaults = .standard) -> Task<Void, Never>? {
        guard !VoxtRuntimeEnvironment.isRunningUnitTests else { return nil }
        return Task { await reclaimRetiredArtifacts(defaults: defaults) }
    }

    /// Keeps the GGUF weights until the mapped MLX checkpoint is installed, so a migrated user never loses offline translation.
    static func reclaimRetiredArtifacts(defaults: UserDefaults = .standard) async {
        let replacementRepo = repo(forLegacyModelID: defaults.string(forKey: AppPreferenceKey.legacyTranslationGGUFModelID))
        let rootURLs = ModelStorageDirectoryManager.resolvedReadableRootURLs(defaults: defaults)
        do {
            let retiredDirectories = try await ModelDiskOperations.perform {
                rootURLs
                    .map { $0.appendingPathComponent(storageDirectoryName, isDirectory: true) }
                    .filter { FileManager.default.fileExists(atPath: $0.path) }
            }
            if !retiredDirectories.isEmpty {
                let isReplacementInstalled = try await ModelDiskOperations.perform {
                    rootURLs.contains { rootURL in
                        CustomLLMModelStorageSupport.cacheDirectory(for: replacementRepo, rootDirectory: rootURL)
                            .map(CustomLLMModelStorageSupport.isModelDirectoryValid) ?? false
                    }
                }
                guard isReplacementInstalled else {
                    VoxtLog.model("Keeping retired GGUF translation models until \(replacementRepo) is installed.")
                    return
                }
                try await ModelDiskOperations.remove(retiredDirectories)
                VoxtLog.model("Removed retired GGUF translation models. paths=\(retiredDirectories.map(\.path))")
            }
            defaults.removeObject(forKey: AppPreferenceKey.legacyTranslationGGUFModelID)
        } catch {
            VoxtLog.modelWarning("Failed to remove retired GGUF translation models. error=\(error.localizedDescription)")
        }
    }
}
