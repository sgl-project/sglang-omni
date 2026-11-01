// SileroVADModelLoader.swift
// Authorizes and structurally validates Silero model directory readers.

import Foundation
import OmniModelArtifacts

extension SileroVADModelSupport {
    nonisolated static func validatedModelDirectoryLease(from directory: URL) async throws -> OmniModelDirectoryLease {
        try await ModelDiskOperations.perform {
            let directoryLease = try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead)
            try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directoryLease.directory)
            return directoryLease
        }
    }
}
