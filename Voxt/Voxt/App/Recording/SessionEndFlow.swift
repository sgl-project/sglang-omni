// SessionEndFlow.swift
// Provides Session End Flow for recording session routing.

import AppKit
import Foundation

extension AppDelegate {
    private func resetEndedSessionState() {
        let shouldPreserveTranslationAnswerControls =
            sessionOutputMode == .translation &&
            overlayState.displayMode == .answer

        recordingLifecycle.invalidateCallbacks()
        invalidateActiveLLMRequest()
        isSessionActive = false
        sessionOutputMode = .transcription
        isSelectedTextTranslationFlow = false
        if !shouldPreserveTranslationAnswerControls {
            sessionTargetApplicationPID = nil
            sessionTargetApplicationBundleID = nil
            selectedTextTranslationHadWritableFocusedInput = false
        }
        enhancementContextSnapshot = nil
        sessionOutputDestinationContext = nil
        rewriteSessionHasSelectedSourceText = false
        rewriteSessionSelectedSourceText = ""
        rewriteSessionHadWritableFocusedInput = false
        rewriteSessionFallbackInjectBundleID = nil
        sessionTranslationTargetLanguageOverride = nil
        activeSessionTranslationProviderResolution = nil
        resetVoxtNoteSessionRuntimeState()
        if !shouldPreserveTranslationAnswerControls {
            overlayState.configureSessionTranslationTargetLanguage(nil, allowsSwitching: false)
        }
        overlayState.isCompleting = false
        if overlayState.displayMode != .answer {
            overlayState.reset()
        }
        pendingSessionFinishTask = nil
    }

    private func beginSessionEndExecution(for sessionID: UUID, trigger: String) -> Bool {
        let decision = recordingLifecycle.beginEnding(sessionID)
        switch decision {
        case .execute:
            return true
        case .skipDuplicateInFlight:
            VoxtLog.asr(
                "Session end pipeline ignored because the same session is already ending. sessionID=\(sessionID.uuidString), trigger=\(trigger)"
            )
            return false
        case .skipAlreadyCompleted:
            VoxtLog.asr(
                "Session end pipeline ignored because the same session has already ended. sessionID=\(sessionID.uuidString), trigger=\(trigger)"
            )
            return false
        case .skipStale:
            VoxtLog.asr("Session end pipeline ignored for an obsolete session. sessionID=\(sessionID.uuidString), trigger=\(trigger)")
            return false
        }
    }

    private func completeSessionEndExecution(for sessionID: UUID) {
        recordingLifecycle.completeEnding(sessionID)
    }

    @MainActor
    func executeSessionEndPipeline(for sessionID: UUID, trigger: String) {
        guard beginSessionEndExecution(for: sessionID, trigger: trigger) else { return }
        defer {
            completeSessionEndExecution(for: sessionID)
        }

        VoxtLog.asr(
            "Session end pipeline started. sessionID=\(sessionID.uuidString), trigger=\(trigger), displayMode=\(overlayState.displayMode), overlayVisible=\(overlayWindow.isVisible)",
            verbose: true
        )
        OnboardingSessionEvent.ended(id: sessionID, message: overlayState.statusMessage).post()
        if overlayState.displayMode != .answer {
            overlayWindow.hide(animated: false)
        }
        systemAudioMuteController.restoreSystemAudioIfNeeded()
        if interactionSoundsEnabled {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                self.interactionSoundPlayer.playEnd()
            }
        }
        resetEndedSessionState()
        releaseResidualRecordingResources(reason: "session-end-pipeline", preservePendingHistoryAudio: true)
        scheduleDeepIdleMemoryReclamation()
        VoxtLog.asr(
            "Session end pipeline completed. sessionID=\(sessionID.uuidString), overlayVisible=\(overlayWindow.isVisible)",
            verbose: true
        )
        meetingFileTaskQueue.startIfNeeded()
    }
}
