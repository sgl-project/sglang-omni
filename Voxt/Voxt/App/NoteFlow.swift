// NoteFlow.swift
// Provides Note Flow for app lifecycle and routing.

import Foundation
import AppKit
import Carbon

extension AppDelegate {
    enum TranscriptionCaptureSessionMode {
        case standard
        case noteSession
    }

    func configureVoxtNoteSessionRuntimeStateForNewRecording(
        mode: TranscriptionCaptureSessionMode = .standard
    ) {
        transcriptionCaptureSessionMode = mode
        configureTranscriptionCapturePipelineForCurrentSession()
        liveTranscriptSegmentationState.reset()
        overlayState.setTranscribedTextTransformer { [weak self] rawText in
            self?.resolvedLiveTranscriptDisplayText(from: rawText) ?? rawText
        }
    }

    func resetVoxtNoteSessionRuntimeState() {
        transcriptionCaptureSessionMode = .standard
        configureTranscriptionCapturePipelineForCurrentSession()
        liveTranscriptSegmentationState.reset()
        overlayState.setTranscribedTextTransformer(nil)
    }

    func handleNoteHotkeyDown() {
        guard FeatureSettingsStore.availability().notesEnabled else { return }
        cancelPendingTranscriptionStart()
        guard !blockNonMeetingRecordingWhileMeetingIsActive(source: "noteHotkey") else { return }
        guard noteStore.isAvailable else {
            showFloatingToast(
                AppLocalization.localizedString("Notes are unavailable. Open Note settings to repair storage."),
                kind: .warning,
                clearAfter: 3.2
            )
            return
        }

        let ignoresSelectedText = SelectedTextSystemSelectionSupport.ignoresSelectedTextFlow(
            bundleID: NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        )
        let selectedText: String? = if isSessionActive || ignoresSelectedText {
            nil
        } else {
            selectedContentTextFromSystemSelection()
        }

        if ignoresSelectedText {
            VoxtLog.hotkey("Note hotkey bypassing selected-text flow for Obsidian.")
        }

        let action = NoteHotkeyActionResolver.resolve(
            state: .init(
                isSessionActive: isSessionActive,
                sessionOutputMode: sessionOutputMode,
                isPanelVisible: noteWindowManager.isVisible,
                canStopSession: !isSessionStopInProgress && !shouldIgnoreTapStop(),
                hasSelectedText: selectedText != nil
            )
        )
        switch action {
        case .captureSelectedText:
            guard let selectedText, !selectedText.isEmpty else { return }
            guard appendVoxtNote(
                text: selectedText,
                sessionID: UUID(),
                source: .selection
            ) else {
                showFloatingToast(
                    AppLocalization.localizedString("The selected text could not be saved as a note."),
                    kind: .warning,
                    clearAfter: 3.2
                )
                return
            }
            showFloatingToast(AppLocalization.localizedString("Note created."))
            VoxtLog.hotkey("Note hotkey captured selected text. characters=\(selectedText.count)")
        case .stopRecordingAsNote:
            transcriptionCaptureSessionMode = .noteSession
            configureTranscriptionCapturePipelineForCurrentSession()
            VoxtLog.hotkey("Note hotkey ending active transcription as a note.")
            endRecording()
        case .startNoteRecording:
            noteWindowManager.hide()
            VoxtLog.hotkey("Note hotkey starting note transcription from visible panel.")
            beginRecording(outputMode: .transcription, transcriptionCaptureMode: .noteSession)
        case .revealPanel:
            VoxtLog.hotkey("Note hotkey revealing note panel.")
            noteWindowManager.show()
        case .ignore:
            break
        }
    }

    @discardableResult
    func captureLiveTranscriptNoteIfPossible(reason: String) -> Bool {
        guard isSessionActive, sessionOutputMode == .transcription else { return false }
        guard noteStore.isAvailable else {
            showFloatingToast(
                AppLocalization.localizedString("Notes are unavailable. Open Note settings to repair storage."),
                kind: .warning,
                clearAfter: 3.2
            )
            return false
        }
        let previousSegmentationState = liveTranscriptSegmentationState
        let rawText = currentSessionRawTranscribedText()
        let capturedText = liveTranscriptSegmentationState.freezeCurrentSegment(
            using: rawText,
            markerLabel: voxtNoteBoundaryMarkerLabel()
        )
            ?? overlayState.transcribedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedText = capturedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            VoxtLog.info("Voxt note capture skipped because current transcript tail is empty. reason=\(reason)")
            return false
        }

        guard appendVoxtNote(text: trimmedText, sessionID: activeRecordingSessionID) else {
            liveTranscriptSegmentationState = previousSegmentationState
            showFloatingToast(
                AppLocalization.localizedString("The note could not be saved. Your transcript was kept unchanged."),
                kind: .warning,
                clearAfter: 3.2
            )
            return false
        }

        transcriptionCaptureSessionMode = .noteSession
        configureTranscriptionCapturePipelineForCurrentSession()
        overlayState.sessionIconMode = currentRecordingOverlayIconMode
        refreshVoxtNoteTranscriptDisplay()
        VoxtLog.info("Voxt note captured. reason=\(reason), characters=\(trimmedText.count)")
        return true
    }

    @discardableResult
    func captureTrailingVoxtNoteIfNeeded(finalRawText: String) -> Bool {
        guard transcriptionCaptureSessionMode == .noteSession else { return false }
        guard noteStore.isAvailable else { return false }
        let trimmedFinalText = finalRawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedFinalText.isEmpty else { return false }

        let previousSegmentationState = liveTranscriptSegmentationState
        let capturedText = liveTranscriptSegmentationState.freezeCurrentSegment(using: trimmedFinalText)
        let trimmedCapturedText = capturedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmedCapturedText.isEmpty else {
            refreshVoxtNoteTranscriptDisplay()
            return false
        }

        guard appendVoxtNote(text: trimmedCapturedText, sessionID: activeRecordingSessionID) else {
            liveTranscriptSegmentationState = previousSegmentationState
            return false
        }
        refreshVoxtNoteTranscriptDisplay()
        VoxtLog.info("Voxt note trailing segment captured at session end. characters=\(trimmedCapturedText.count)")
        return true
    }

    func refreshVoxtNoteTranscriptDisplay() {
        overlayState.refreshDisplayedTranscribedText()
    }

    var isCurrentTranscriptionNoteSessionActive: Bool {
        transcriptionCaptureSessionMode == .noteSession
    }

    private var isCurrentTranscriptionCaptureLive: Bool {
        guard recordingStoppedAt == nil else { return false }
        guard transcriptionCapturePipeline.usesLiveDisplay else { return false }
        switch transcriptionEngine {
        case .dictation:
            return speechTranscriber.isRecording
        case .mlxAudio:
            return mlxTranscriber?.isRecording == true
        case .remote:
            return remoteASRTranscriber.isRecording
        }
    }

    private func resolvedLiveTranscriptDisplayText(from rawText: String) -> String {
        guard isSessionActive, sessionOutputMode == .transcription else {
            return rawText
        }
        guard transcriptionCapturePipeline == .noteSession else {
            return rawText
        }
        return liveTranscriptSegmentationState.displayText(for: rawText)
    }

    func configureTranscriptionCapturePipelineForCurrentSession() {
        transcriptionCapturePipeline = TranscriptionCapturePipeline.resolve(
            realtimeTextDisplayEnabled: realtimeTextDisplayEnabled,
            captureSessionMode: transcriptionCaptureSessionMode
        )
    }

    func currentSessionRawTranscribedText() -> String {
        switch transcriptionEngine {
        case .dictation:
            return speechTranscriber.transcribedText
        case .mlxAudio:
            return mlxTranscriber?.currentWorkingTranscriptText ?? ""
        case .remote:
            return remoteASRTranscriber.transcribedText
        }
    }

    func currentTranscriptionCaptureMetrics() -> TranscriptionCaptureMetrics? {
        switch transcriptionEngine {
        case .mlxAudio:
            return mlxTranscriber?.lastCaptureMetrics
        case .dictation, .remote:
            return nil
        }
    }

    private func voxtNoteBoundaryMarkerLabel() -> String {
        guard let recordingStartedAt else { return "00:00" }
        let elapsed = max(0, Int(Date().timeIntervalSince(recordingStartedAt)))
        let minutes = elapsed / 60
        let seconds = elapsed % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }
}
