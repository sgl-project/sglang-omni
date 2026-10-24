// RecordingOverlayFlow.swift
// Provides Recording Overlay Flow for recording session routing.

import Foundation

extension AppDelegate {
    var currentRecordingOverlayIconMode: OverlaySessionIconMode {
        RecordingSessionSupport.overlayIconMode(
            for: sessionOutputMode,
            isNoteSession: isCurrentTranscriptionNoteSessionActive
        )
    }

    func showOverlayStatus(_ message: String, clearAfter seconds: TimeInterval = 2.4) {
        overlayStatusClearTask?.cancel()
        overlayState.statusMessage = message
        overlayState.presentRecording(iconMode: currentRecordingOverlayIconMode)
        overlayStatusClearTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            if self.overlayState.statusMessage == message {
                self.overlayState.statusMessage = ""
            }
            self.overlayStatusClearTask = nil
        }
    }

    func showOverlayReminder(_ message: String, autoHideAfter seconds: TimeInterval = 2.4) {
        overlayReminderTask?.cancel()
        overlayStatusClearTask?.cancel()
        overlayState.reset()
        overlayState.statusMessage = message
        overlayState.presentRecording(iconMode: currentRecordingOverlayIconMode)
        overlayWindow.show(state: overlayState, position: overlayPosition)

        overlayReminderTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self.overlayWindow.hide()
            self.overlayState.reset()
            self.overlayReminderTask = nil
        }
    }

    func showFloatingToast(
        _ message: String,
        kind: FloatingToastKind = .success,
        clearAfter seconds: TimeInterval = 1.8
    ) {
        toastDismissTask?.cancel()
        toastWindow.show(message: message, kind: kind, position: overlayPosition)
        toastDismissTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(seconds))
            guard !Task.isCancelled else { return }
            self.toastWindow.hide()
            self.toastDismissTask = nil
        }
    }

    func setEnhancingState(_ isEnhancing: Bool) {
        overlayState.isEnhancing = isEnhancing
        if overlayState.displayMode != .answer {
            overlayState.displayMode = isEnhancing ? .processing : .recording
        }
        setActiveRecordingTranscriberEnhancingState(isEnhancing)
    }
}
