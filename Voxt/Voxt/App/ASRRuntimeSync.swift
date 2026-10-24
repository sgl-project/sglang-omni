// ASRRuntimeSync.swift
// Provides ASRRuntime Sync for app lifecycle and routing.

import Foundation

extension AppDelegate {
    func synchronizeRuntimeASRStateForSession(outputMode: SessionOutputMode) {
        synchronizeRuntimeASRState(for: asrSelectionID(for: outputMode))
    }

    func synchronizeRuntimeASRStateForMeeting() {
        synchronizeRuntimeASRState(for: meetingFeatureSettings.asrSelectionID)
    }

    private func asrSelectionID(for outputMode: SessionOutputMode) -> FeatureModelSelectionID {
        switch outputMode {
        case .transcription:
            return transcriptionFeatureSettings.asrSelectionID
        case .translation:
            return translationFeatureSettings.asrSelectionID
        case .rewrite:
            return rewriteFeatureSettings.asrSelectionID
        }
    }

    private func synchronizeRuntimeASRState(for selectionID: FeatureModelSelectionID) {
        switch selectionID.asrSelection {
        case .mlx(let repo):
            let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
            let previousRepo = MLXModelManager.canonicalModelRepo(mlxModelManager.currentModelRepo)
            guard canonicalRepo != previousRepo else { return }

            VoxtLog.asr(
                "Synchronizing MLX runtime model. previous=\(previousRepo), current=\(canonicalRepo)"
            )
            mlxModelManager.updateModel(repo: canonicalRepo)
            mlxTranscriber = nil

        case .dictation, .remote, .none:
            return
        }
    }
}
