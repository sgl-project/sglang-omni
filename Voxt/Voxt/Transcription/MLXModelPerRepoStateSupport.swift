// MLXModelPerRepoStateSupport.swift
// Provides MLXModel Per Repo State Support for transcription engines.

import Foundation

enum MLXModelPerRepoStateSupport {
    static func resolvedState(
        for repo: String,
        currentRepo: String,
        currentState: MLXModelManager.ModelState,
        storedStates: [String: MLXModelManager.ModelState],
        isDownloaded: (String) -> Bool,
        hasResumableDownload: (String) -> Bool
    ) -> MLXModelManager.ModelState {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
        if let existing = storedStates[canonicalRepo] {
            return existing
        }
        if canonicalRepo == currentRepo {
            return currentState
        }
        if isDownloaded(canonicalRepo) {
            return .downloaded
        }
        if hasResumableDownload(canonicalRepo) {
            return .paused(
                progress: 0,
                completed: 0,
                total: 0,
                currentFile: nil,
                completedFiles: 0,
                totalFiles: 0
            )
        }
        return .notDownloaded
    }

    static func applyState(
        _ newState: MLXModelManager.ModelState,
        for repo: String,
        currentRepo: String,
        currentState: inout MLXModelManager.ModelState,
        storedStates: inout [String: MLXModelManager.ModelState]
    ) {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
        if storedStates[canonicalRepo] != newState {
            storedStates[canonicalRepo] = newState
        }
        if canonicalRepo == currentRepo, currentState != newState {
            currentState = newState
        }
    }

    static func pausedStatusMessage(
        for repo: String,
        storedMessages: [String: String]
    ) -> String? {
        storedMessages[MLXModelManager.canonicalModelRepo(repo)]
    }

    static func applyPausedStatusMessage(
        _ message: String?,
        for repo: String,
        currentRepo: String,
        currentMessage: inout String?,
        storedMessages: inout [String: String]
    ) {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
        let existing = storedMessages[canonicalRepo]
        if existing != message {
            if let message {
                storedMessages[canonicalRepo] = message
            } else {
                storedMessages.removeValue(forKey: canonicalRepo)
            }
        }
        if canonicalRepo == currentRepo, currentMessage != message {
            currentMessage = message
        }
    }

    static func clearState(
        for repo: String,
        currentRepo: String,
        currentPausedStatusMessage: inout String?,
        storedStates: inout [String: MLXModelManager.ModelState],
        storedMessages: inout [String: String]
    ) {
        let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
        storedStates.removeValue(forKey: canonicalRepo)
        storedMessages.removeValue(forKey: canonicalRepo)
        if canonicalRepo == currentRepo {
            currentPausedStatusMessage = nil
        }
    }

    static func resetStorageRootState(
        currentPausedStatusMessage: inout String?,
        storedStates: inout [String: MLXModelManager.ModelState],
        storedMessages: inout [String: String]
    ) {
        storedStates.removeAll()
        storedMessages.removeAll()
        currentPausedStatusMessage = nil
    }
}

extension MLXModelPerRepoStateSupport {
    static func resolvedState(
        for repo: String,
        currentRepo: String,
        currentState: CustomLLMModelManager.ModelState,
        storedStates: [String: CustomLLMModelManager.ModelState],
        isDownloaded: (String) -> Bool,
        hasResumableDownload: (String) -> Bool
    ) -> CustomLLMModelManager.ModelState {
        let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        if let existing = storedStates[canonicalRepo] {
            return existing
        }
        if canonicalRepo == currentRepo {
            return currentState
        }
        if isDownloaded(canonicalRepo) {
            return .downloaded
        }
        if hasResumableDownload(canonicalRepo) {
            return .paused(
                progress: 0,
                completed: 0,
                total: 0,
                currentFile: nil,
                completedFiles: 0,
                totalFiles: 0
            )
        }
        return .notDownloaded
    }

    static func applyState(
        _ newState: CustomLLMModelManager.ModelState,
        for repo: String,
        currentRepo: String,
        currentState: inout CustomLLMModelManager.ModelState,
        storedStates: inout [String: CustomLLMModelManager.ModelState]
    ) {
        let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        if storedStates[canonicalRepo] != newState {
            storedStates[canonicalRepo] = newState
        }
        if canonicalRepo == currentRepo, currentState != newState {
            currentState = newState
        }
    }

    static func pausedStatusMessage(
        for repo: String,
        storedMessages: [String: String],
        canonicalize: (String) -> String
    ) -> String? {
        storedMessages[canonicalize(repo)]
    }

    static func applyPausedStatusMessage(
        _ message: String?,
        for repo: String,
        currentRepo: String,
        currentMessage: inout String?,
        storedMessages: inout [String: String],
        canonicalize: (String) -> String
    ) {
        let canonicalRepo = canonicalize(repo)
        let existing = storedMessages[canonicalRepo]
        if existing != message {
            if let message {
                storedMessages[canonicalRepo] = message
            } else {
                storedMessages.removeValue(forKey: canonicalRepo)
            }
        }
        if canonicalRepo == currentRepo, currentMessage != message {
            currentMessage = message
        }
    }

    static func clearCustomLLMState(
        for repo: String,
        currentRepo: String,
        currentPausedStatusMessage: inout String?,
        storedStates: inout [String: CustomLLMModelManager.ModelState],
        storedMessages: inout [String: String]
    ) {
        let canonicalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        storedStates.removeValue(forKey: canonicalRepo)
        storedMessages.removeValue(forKey: canonicalRepo)
        if canonicalRepo == currentRepo {
            currentPausedStatusMessage = nil
        }
    }

    static func resetCustomLLMStorageRootState(
        currentPausedStatusMessage: inout String?,
        storedStates: inout [String: CustomLLMModelManager.ModelState],
        storedMessages: inout [String: String]
    ) {
        storedStates.removeAll()
        storedMessages.removeAll()
        currentPausedStatusMessage = nil
    }
}
