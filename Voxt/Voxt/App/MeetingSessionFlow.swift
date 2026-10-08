// MeetingSessionFlow.swift
// Provides Meeting Session Flow for app lifecycle and routing.

import Foundation
import AppKit
import AVFoundation

extension AppDelegate {
    func blockNonMeetingRecordingWhileMeetingIsActive(source: String) -> Bool {
        guard meetingSessionCoordinator.isActive,
              !meetingSessionCoordinator.isAnalyzingImportedFile
        else {
            return false
        }
        VoxtLog.meeting("Non-meeting recording blocked because meeting is active. source=\(source)")
        VoxtLog.hotkey("Non-meeting recording blocked: meeting active. source=\(source)")
        if meetingSessionCoordinator.overlayState.isPresented {
            meetingOverlayWindow.show(
                state: meetingSessionCoordinator.overlayState,
                position: overlayPosition
            )
        }
        return true
    }

    func handleMeetingHotkeyDown() {
        guard FeatureSettingsStore.availability().meetingEnabled else { return }
        guard !isApplicationTerminating else { return }
        VoxtLog.meeting(
            "Meeting hotkey invoked. isMeetingActive=\(meetingSessionCoordinator.isActive), isSessionActive=\(isSessionActive)"
        )
        VoxtLog.hotkey(
            "Hotkey callback meetingDown. isMeetingActive=\(meetingSessionCoordinator.isActive), isSessionActive=\(isSessionActive)"
        )

        cancelPendingTranscriptionStart()

        if meetingSessionCoordinator.isAnalyzingImportedFile {
            showOverlayStatus(
                AppLocalization.localizedString(
                    "Wait for the meeting file analysis to finish before starting Meeting Notes."
                ),
                clearAfter: 2.2
            )
            return
        }

        if meetingSessionCoordinator.isActive {
            if meetingSessionCoordinator.overlayState.isCloseConfirmationPresented {
                dismissMeetingSessionCloseConfirmation()
            } else {
                requestMeetingSessionCloseConfirmation()
            }
            return
        }

        guard !isSessionActive else {
            showOverlayStatus(
                AppLocalization.localizedString("Finish the current recording before starting Meeting Notes."),
                clearAfter: 2.2
            )
            return
        }

        Task { @MainActor [weak self] in
            await self?.startMeetingSession()
        }
    }

    func stopMeetingSession(
        closeOverlayImmediately: Bool = true,
        closeLiveDetailImmediately: Bool = true
    ) {
        hotkeyManager.setCommonStopKeyEnabled(false)
        pendingMeetingStartupTask?.cancel()
        pendingMeetingStartupTask = nil

        if meetingSessionCoordinator.isStartingUp &&
            !meetingSessionCoordinator.overlayState.isRecording &&
            !meetingSessionCoordinator.overlayState.isPaused {
            meetingSessionCoordinator.overlayState.isCloseConfirmationPresented = false
            meetingSessionCoordinator.overlayState.isCaptureModePickerPresented = false
            meetingSessionCoordinator.overlayState.isRealtimeTranslationLanguagePickerPresented = false
            if closeLiveDetailImmediately {
                meetingDetailWindowManager.closeLiveWindow()
            }
            meetingSessionCoordinator.cancelPendingStart()
            if closeOverlayImmediately {
                meetingOverlayWindow.hide()
            }
            return
        }

        guard meetingSessionCoordinator.isActive else {
            if closeOverlayImmediately {
                meetingOverlayWindow.hide()
            }
            return
        }
        meetingSessionCoordinator.overlayState.isCloseConfirmationPresented = false
        meetingSessionCoordinator.overlayState.isCaptureModePickerPresented = false
        meetingSessionCoordinator.overlayState.isRealtimeTranslationLanguagePickerPresented = false
        if closeLiveDetailImmediately {
            meetingDetailWindowManager.closeLiveWindow()
        }
        if closeOverlayImmediately {
            meetingOverlayWindow.hide()
        }
        meetingSessionCoordinator.stop(
            shouldFlushPendingAudio: pendingMeetingSessionCompletionDisposition != .discard
        )
    }

    func requestMeetingSessionCloseConfirmation() {
        guard meetingSessionCoordinator.isActive else { return }
        let closeDecision = MeetingSessionClosePlanner.resolve(
            hasTranscriptSegments: !meetingSessionCoordinator.overlayState.segments.isEmpty,
            hasCapturedAudio: meetingSessionCoordinator.hasCapturedAudio
        )
        if closeDecision == .discard {
            cancelMeetingSessionWithoutSaving()
            return
        }
        if meetingSessionCoordinator.overlayState.isCollapsed {
            meetingSessionCoordinator.setCollapsed(false)
        }
        meetingSessionCoordinator.overlayState.isCaptureModePickerPresented = false
        meetingSessionCoordinator.overlayState.isRealtimeTranslationLanguagePickerPresented = false
        meetingSessionCoordinator.overlayState.isCloseConfirmationPresented = true
    }

    func dismissMeetingSessionCloseConfirmation() {
        guard meetingSessionCoordinator.isActive else { return }
        meetingSessionCoordinator.overlayState.isCloseConfirmationPresented = false
    }

    func cancelMeetingSessionWithoutSaving() {
        guard meetingSessionCoordinator.isActive else {
            meetingOverlayWindow.hide()
            return
        }
        pendingMeetingSessionCompletionDisposition = .discard
        stopMeetingSession()
    }

    func finishMeetingSessionAndOpenDetail() {
        guard meetingSessionCoordinator.isActive else { return }
        pendingMeetingSessionCompletionDisposition = .saveAndOpenDetail
        showLiveMeetingDetailWindow()
        stopMeetingSession(closeOverlayImmediately: true, closeLiveDetailImmediately: false)
    }

    func toggleMeetingOverlayCollapse() {
        meetingSessionCoordinator.setCollapsed(!meetingSessionCoordinator.overlayState.isCollapsed)
    }

    func toggleMeetingPause() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            if self.meetingSessionCoordinator.overlayState.isPaused {
                if let failureMessage = await self.meetingSessionCoordinator.resume() {
                    VoxtLog.meetingWarning("Meeting resume failed: \(failureMessage)")
                    self.showOverlayReminder(failureMessage)
                }
            } else {
                await self.meetingSessionCoordinator.pause()
            }
        }
    }

    func toggleMeetingCaptureModePicker() {
        guard meetingSessionCoordinator.isActive else { return }
        let overlayState = meetingSessionCoordinator.overlayState
        overlayState.isCloseConfirmationPresented = false
        overlayState.isRealtimeTranslationLanguagePickerPresented = false
        overlayState.isCaptureModePickerPresented.toggle()
    }

    func dismissMeetingCaptureModePicker() {
        guard meetingSessionCoordinator.isActive else { return }
        meetingSessionCoordinator.overlayState.isCaptureModePickerPresented = false
    }

    func exportMeetingTranscript() {
        guard meetingSessionCoordinator.canExport else { return }

        do {
            try MeetingTranscriptExporter.export(
                segments: meetingSessionCoordinator.overlayState.segments,
                defaultFilename: meetingExportFilename()
            )
        } catch {
            showOverlayReminder(AppLocalization.format("Export failed: %@", error.localizedDescription))
        }
    }

    func copyMeetingSegment(_ segment: MeetingTranscriptSegment) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(MeetingTranscriptFormatter.copyString(for: segment), forType: .string)
    }

    func showLiveMeetingDetailWindow() {
        guard meetingSessionCoordinator.isActive else { return }
        meetingDetailWindowManager.presentLiveMeeting(
            state: meetingSessionCoordinator.overlayState,
            initialSummarySettings: currentMeetingSummarySettingsSnapshot(),
            summaryModelOptionsProvider: { @MainActor in
                self.meetingSummaryModelOptions()
            },
            summarySettingsProvider: { @MainActor in
                self.currentMeetingSummarySettingsSnapshot()
            },
            translationHandler: { @MainActor text, targetLanguage in
                self.makeMeetingTranslationOperation(text, targetLanguage: targetLanguage)
            }
        )
    }

    func handleMeetingRealtimeTranslationToggle(_ isEnabled: Bool) {
        meetingSessionCoordinator.overlayState.isCaptureModePickerPresented = false
        guard isEnabled else {
            meetingSessionCoordinator.overlayState.isRealtimeTranslationLanguagePickerPresented = false
            meetingSessionCoordinator.setRealtimeTranslateEnabled(false)
            return
        }

        meetingSessionCoordinator.overlayState.realtimeTranslationDraftLanguageRaw =
            (resolvedMeetingRealtimeTranslationTargetLanguage() ?? .english).rawValue
        meetingSessionCoordinator.overlayState.isRealtimeTranslationLanguagePickerPresented = true
        meetingSessionCoordinator.setRealtimeTranslateEnabled(false)
    }

    func handleMeetingCaptureModeSelection(_ mode: MeetingCaptureMode) {
        guard validatePermissionsForMeetingCaptureMode(mode) else { return }
        meetingSessionCoordinator.overlayState.isCaptureModePickerPresented = false

        Task { @MainActor [weak self] in
            guard let self else { return }
            if let failureMessage = await self.meetingSessionCoordinator.setCaptureMode(mode) {
                VoxtLog.meetingWarning("Meeting capture mode switch failed: \(failureMessage)")
                self.showOverlayReminder(failureMessage)
            }
        }
    }

    func confirmMeetingRealtimeTranslationLanguageSelection() {
        let rawValue = meetingSessionCoordinator.overlayState.realtimeTranslationDraftLanguageRaw
        guard let language = TranslationTargetLanguage(rawValue: rawValue) else {
            cancelMeetingRealtimeTranslationLanguageSelection()
            return
        }

        UserDefaults.standard.set(
            language.rawValue,
            forKey: AppPreferenceKey.meetingRealtimeTranslationTargetLanguage
        )
        meetingSessionCoordinator.overlayState.isRealtimeTranslationLanguagePickerPresented = false
        meetingSessionCoordinator.setRealtimeTranslateEnabled(true)
    }

    func cancelMeetingRealtimeTranslationLanguageSelection() {
        meetingSessionCoordinator.overlayState.isRealtimeTranslationLanguagePickerPresented = false
        meetingSessionCoordinator.setRealtimeTranslateEnabled(false)
    }

    private func startMeetingSession() async {
        VoxtLog.meeting("Meeting session start requested.")
        guard preflightPermissionsForMeeting() else { return }

        if let storageFailureMessage = await meetingSessionCoordinator.preflightMeetingCaptureStorage() {
            VoxtLog.meetingWarning("Meeting start blocked by audio storage preflight: \(storageFailureMessage)")
            showOverlayReminder(storageFailureMessage)
            scheduleDeepIdleMemoryReclamation()
            return
        }

        pendingMeetingSessionCompletionDisposition = .save

        meetingSessionCoordinator.onSessionFinished = { [weak self] result in
            guard let self else { return false }
            let handled = self.handleMeetingSessionFinished(result)
            self.scheduleDeepIdleMemoryReclamation()
            return handled
        }

        meetingSessionCoordinator.prepareForStart()
        meetingOverlayWindow.show(
            state: meetingSessionCoordinator.overlayState,
            position: overlayPosition
        )

        pendingMeetingStartupTask?.cancel()
        pendingMeetingStartupTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                if self.pendingMeetingStartupTask?.isCancelled != false {
                    self.pendingMeetingStartupTask = nil
                }
            }
            let failureMessage = await self.meetingSessionCoordinator.start()
            guard !Task.isCancelled else {
                self.pendingMeetingStartupTask = nil
                return
            }
            self.pendingMeetingStartupTask = nil
            if let failureMessage {
                self.hotkeyManager.setCommonStopKeyEnabled(false)
                VoxtLog.meetingWarning("Meeting start failed: \(failureMessage)")
                self.meetingOverlayWindow.hide()
                self.showOverlayReminder(failureMessage)
                self.scheduleDeepIdleMemoryReclamation()
            } else {
                self.hotkeyManager.setCommonStopKeyEnabled(true)
            }
        }
    }

    private func preflightPermissionsForMeeting() -> Bool {
        prepareSettingsForMeetingRuntime()
        synchronizeRuntimeASRStateForMeeting()

        if isSessionActive {
            VoxtLog.meeting("Meeting start blocked because another recording session is active.")
            showOverlayStatus(
                AppLocalization.localizedString("Finish the current recording before starting Meeting Notes."),
                clearAfter: 2.2
            )
            return false
        }

        let remoteConfiguration = RemoteModelConfigurationStore.resolvedASRConfiguration(
            provider: remoteASRSelectedProvider,
            stored: remoteASRConfigurations
        )
        let localASRStartContext = currentLocalASRStartContext()
        let startDecision = MeetingStartPlanner.resolve(
            selectedEngine: transcriptionEngine,
            selectedMLXRepo: localASRStartContext.selectedMLXRepo,
            activeMLXDownloadRepo: localASRStartContext.activeMLXDownloadRepo,
            isSelectedMLXModelDownloaded: localASRStartContext.isSelectedMLXModelDownloaded,
            mlxModelState: localASRStartContext.mlxModelState,
            remoteASRConfiguration: remoteConfiguration
        )
        guard case .start = startDecision else {
            if case .blocked(let reason) = startDecision {
                VoxtLog.meetingWarning("Meeting start blocked: \(reason.logDescription)")
                showOverlayReminder(reason.userMessage)
            }
            return false
        }

        let captureMode = MeetingCaptureMode.stored()
        guard validatePermissionsForMeetingCaptureMode(captureMode) else {
            return false
        }

        if !AccessibilityPermissionManager.isTrusted() {
            VoxtLog.meetingWarning("Meeting start proceeding without accessibility trust. Some injection shortcuts may be unavailable.")
            showOverlayStatus(
                AppLocalization.localizedString("Please enable required permissions in Settings > Permissions."),
                clearAfter: 2.2
            )
        }

        return true
    }

    private func validatePermissionsForMeetingCaptureMode(_ mode: MeetingCaptureMode) -> Bool {
        if mode.usesMicrophone && AVCaptureDevice.authorizationStatus(for: .audio) != .authorized {
            VoxtLog.meetingWarning("Meeting capture mode requires microphone permission. mode=\(mode.rawValue)")
            showOverlayReminder(
                AppLocalization.localizedString("Microphone permission is required. Enable it in Settings > Permissions.")
            )
            return false
        }

        // Starting the public Core Audio tap after this user action triggers
        // system-audio authorization. No private TCC preflight or cached grant
        // can reliably determine access (including after revocation).

        return true
    }

}
