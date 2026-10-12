// HotkeyLifecycle.swift
// Provides Hotkey Lifecycle for app lifecycle and routing.

import AppKit
import Carbon

final class OverlayShortcutEventGate: @unchecked Sendable {
    struct KeySignature: Hashable {
        let keyCode: UInt16
        let modifiersRawValue: UInt

        init(keyCode: UInt16, modifiers: NSEvent.ModifierFlags) {
            self.keyCode = keyCode
            self.modifiersRawValue = modifiers.intersection(.hotkeyRelevant).rawValue
        }
    }

    private let lock = NSLock()
    private var signatures: Set<KeySignature> = [
        KeySignature(keyCode: UInt16(kVK_Escape), modifiers: [])
    ]

    func update(
        answerContinueShortcut: HotkeyPreference.Hotkey
    ) {
        var updatedSignatures: Set<KeySignature> = [
            KeySignature(keyCode: UInt16(kVK_Escape), modifiers: [])
        ]
        updatedSignatures.insert(
            KeySignature(
                keyCode: answerContinueShortcut.keyCode,
                modifiers: answerContinueShortcut.modifiers
            )
        )
        lock.lock()
        signatures = updatedSignatures
        lock.unlock()
    }

    func shouldDispatch(_ event: NSEvent) -> Bool {
        shouldDispatch(
            keyCode: event.keyCode,
            modifiers: event.modifierFlags,
            isRepeat: event.isARepeat
        )
    }

    func shouldDispatch(
        keyCode: UInt16,
        modifiers: NSEvent.ModifierFlags,
        isRepeat: Bool = false
    ) -> Bool {
        guard !isRepeat else { return false }
        let signature = KeySignature(keyCode: keyCode, modifiers: modifiers)
        lock.lock()
        let shouldDispatch = signatures.contains(signature)
        lock.unlock()
        return shouldDispatch
    }
}

@MainActor
extension AppDelegate {
    enum SessionCallbackHandlingDecision: Equatable {
        case accept
        case rejectStale
        case rejectCancelled

        var logDescription: String {
            switch self {
            case .accept:
                return "accept"
            case .rejectStale:
                return "stale-session"
            case .rejectCancelled:
                return "cancelled-session"
            }
        }
    }

    func setupHotkey() {
        // Callback contract:
        // - HotkeyManager only emits normalized events (transcriptionDown/up, translationDown/up, rewriteDown/up).
        // - AppDelegate owns business decisions (start/stop session, selected-text fast path, mode rules).
        hotkeyManager.onKeyDownWithBehavior = { [weak self] behavior in
            guard let self else { return }
            self.postHotkeyDidTrigger(kind: "transcription", behavior: behavior)
            self.handleTranscriptionHotkeyDown(behavior: behavior)
        }
        hotkeyManager.onKeyUpWithBehavior = { [weak self] behavior in
            guard let self else { return }
            self.handleTranscriptionHotkeyUp(behavior: behavior)
        }
        hotkeyManager.onTranslationKeyDownWithBehavior = { [weak self] behavior in
            guard let self else { return }
            self.postHotkeyDidTrigger(kind: "translation", behavior: behavior)
            self.handleTranslationHotkeyDown(behavior: behavior)
        }
        hotkeyManager.onTranslationKeyUpWithBehavior = { [weak self] behavior in
            guard let self else { return }
            self.handleTranslationHotkeyUp(behavior: behavior)
        }
        hotkeyManager.onRewriteKeyDownWithBehavior = { [weak self] behavior in
            guard let self else { return }
            self.postHotkeyDidTrigger(kind: "rewrite", behavior: behavior)
            self.handleRewriteHotkeyDown(behavior: behavior)
        }
        hotkeyManager.onRewriteKeyUpWithBehavior = { [weak self] behavior in
            guard let self else { return }
            self.handleRewriteHotkeyUp(behavior: behavior)
        }
        hotkeyManager.onMeetingKeyDownWithBehavior = { [weak self] behavior in
            guard let self else { return }
            self.postHotkeyDidTrigger(kind: "meeting", behavior: behavior)
            self.handleMeetingHotkeyDown()
        }
        hotkeyManager.onCustomPasteKeyDown = { [weak self] in
            guard let self else { return }
            self.handleCustomPasteHotkeyDown()
        }
        hotkeyManager.onNoteKeyDown = { [weak self] in
            guard let self else { return }
            self.postHotkeyDidTrigger(kind: "note", behavior: .tap)
            self.handleNoteHotkeyDown()
        }
        hotkeyManager.onCommonStopKeyDown = { [weak self] in
            guard let self else { return }
            self.handleCommonStopHotkeyDown()
        }
        hotkeyManager.onEscapeKeyDown = { [weak self] in
            guard let self, self.shouldConsumeEscapeShortcut() else { return false }
            Task { @MainActor [weak self] in
                _ = self?.handleEscapeShortcut()
            }
            return true
        }
        hotkeyManager.start()
        VoxtLog.hotkey("Hotkey callbacks configured.")
    }

    func setupLifecycleRecoveryObservers() {
        let workspaceNotificationCenter = NSWorkspace.shared.notificationCenter

        workspaceWillSleepObserver = workspaceNotificationCenter.addObserver(
            forName: NSWorkspace.willSleepNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.systemAudioMuteController.restoreSystemAudioIfNeeded()
                self?.interactionSoundPlayer.reset()
                self?.scheduleHotkeyTransientStateReset(reason: "workspaceWillSleep")
            }
        }

        workspaceDidWakeObserver = workspaceNotificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.interactionSoundPlayer.reset()
                self?.interactionSoundPlayer.prewarm()
                self?.scheduleHotkeyTransientStateReset(reason: "workspaceDidWake")
            }
        }

        workspaceSessionDidBecomeActiveObserver = workspaceNotificationCenter.addObserver(
            forName: NSWorkspace.sessionDidBecomeActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.scheduleHotkeyTransientStateReset(reason: "workspaceSessionDidBecomeActive")
            }
        }

        workspaceSessionDidResignActiveObserver = workspaceNotificationCenter.addObserver(
            forName: NSWorkspace.sessionDidResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.scheduleHotkeyTransientStateReset(reason: "workspaceSessionDidResignActive")
            }
        }
    }

    func scheduleHotkeyTransientStateReset(reason: String) {
        Task { @MainActor [weak self] in
            self?.hotkeyManager.resetTransientState(reason: reason)
        }
    }

    func setupEscapeKeyMonitoring() {
        refreshOverlayShortcutEventGate()
        // Global NSEvent monitors require Input Monitoring on some macOS
        // versions. Global Escape is handled by HotkeyManager's Accessibility-
        // backed event tap; keep only the local monitor as an in-app fallback.
        let overlayShortcutEventGate = self.overlayShortcutEventGate
        localEscapeKeyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard overlayShortcutEventGate.shouldDispatch(event) else { return event }
            return self?.handleOverlayShortcutEvent(event, shouldConsume: true) ?? event
        }
    }

    func refreshOverlayShortcutEventGate() {
        overlayShortcutEventGate.update(
            answerContinueShortcut: rewriteContinueShortcutSettings.hotkey
        )
    }

    func handleOverlayShortcutEvent(_ event: NSEvent, shouldConsume: Bool = false) -> NSEvent? {
        let startedAt = Date()
        defer {
            let elapsed = Date().timeIntervalSince(startedAt)
            if elapsed >= 0.04 {
                VoxtLog.hotkey(
                    "Overlay shortcut handling was slow. elapsedMs=\(Int(elapsed * 1000)), keyCode=\(event.keyCode), modifiers=\(event.modifierFlags.intersection(.hotkeyRelevant).rawValue), consume=\(shouldConsume)"
                )
            }
        }

        if shouldHandleAnswerOverlayContinueShortcut(event),
           overlayWindow.handleAnswerSpaceShortcut() {
            return shouldConsume ? nil : event
        }

        guard event.keyCode == UInt16(kVK_Escape) else { return event }
        guard handleEscapeShortcut() else { return event }
        return shouldConsume ? nil : event
    }

    func shouldConsumeEscapeShortcut() -> Bool {
        guard UserDefaults.standard.object(forKey: AppPreferenceKey.escapeKeyCancelsOverlaySession) as? Bool ?? true else {
            return false
        }
        if overlayState.displayMode == .answer {
            return true
        }
        // Escape must also cancel a debounced wake before recording becomes
        // active. Once recording starts it cancels every trigger mode, not
        // only tap mode.
        if pendingTranscriptionStartTask != nil {
            return true
        }
        guard isSessionActive else { return false }
        guard !isSelectedTextTranslationFlow else { return false }
        return true
    }

    func handleEscapeShortcut() -> Bool {
        guard shouldConsumeEscapeShortcut() else { return false }
        if pendingTranscriptionStartTask != nil {
            cancelPendingTranscriptionStart()
            hotkeyManager.resetTransientState(reason: "escape-cancelled-pending-wake")
            return true
        }
        if isSessionActive {
            let shouldRestoreRewriteConversation = overlayState.isRewriteConversationActive
            cancelActiveRecordingSession()
            if shouldRestoreRewriteConversation {
                overlayState.restoreLatestCompletedRewriteConversation()
            }
            return true
        }
        if overlayState.displayMode == .answer {
            dismissAnswerOverlay()
            return true
        }
        return false
    }

    func shouldHandleAnswerOverlayContinueShortcut(_ event: NSEvent) -> Bool {
        guard event.type == .keyDown else { return false }
        guard !event.isARepeat else { return false }
        let shortcut = rewriteContinueShortcutSettings.hotkey
        guard event.keyCode == shortcut.keyCode else { return false }
        let modifiers = event.modifierFlags.intersection(.hotkeyRelevant)
        guard modifiers == shortcut.modifiers else { return false }
        return overlayState.answerSpaceShortcutAction != nil
    }

    func shouldIgnoreTapStop() -> Bool {
        guard let startedAt = recordingStartedAt else { return false }
        let elapsed = Date().timeIntervalSince(startedAt)
        return elapsed < tapStopGuardInterval
    }

    var isSessionStopInProgress: Bool {
        isSessionActive && recordingStoppedAt != nil
    }

    func handleTranscriptionTapDown() {
        guard !blockNonMeetingRecordingWhileMeetingIsActive(source: "transcriptionTap") else { return }
        if isSessionActive {
            guard !shouldIgnoreTapStop() else { return }
            endRecording()
            return
        }
        guard !addSelectedShortTextToDictionaryIfPossible() else { return }
        pendingTranscriptionHotkeyStartBehavior = .tap
        beginRecording(outputMode: .transcription)
    }

    func handleTranslationTapDown() {
        guard !blockNonMeetingRecordingWhileMeetingIsActive(source: "translationTap") else { return }
        if isSessionActive {
            guard sessionOutputMode == .translation else {
                VoxtLog.info("Tap translation down ignored: active session belongs to transcription.", verbose: true)
                return
            }
            guard !shouldIgnoreTapStop() else { return }
            endRecording()
            return
        }
        beginRecording(outputMode: .translation)
    }

    func handleCommonStopHotkeyDown() {
        cancelPendingTranscriptionStart()

        // Imported file analysis also uses the meeting coordinator, but it must
        // not take ownership of the common stop shortcut. Core live recording
        // and translation must always be able to stop while a file task yields
        // in the background.
        if meetingSessionCoordinator.isActive,
           !meetingSessionCoordinator.isAnalyzingImportedFile {
            hotkeyManager.cancelPendingDoubleTapCandidate(reason: "commonStopMeeting")
            handleMeetingHotkeyDown()
            return
        }

        guard isSessionActive else { return }
        switch sessionOutputMode {
        case .translation, .rewrite, .transcription:
            hotkeyManager.cancelPendingDoubleTapCandidate(reason: "commonStopRecording")
            endRecording()
        }
    }

    func handleTranscriptionHotkeyDown(behavior: HotkeyPreference.TriggerBehavior = .tap) {
        if behavior == .longPress {
            isTranscriptionLongPressHotkeyDown = true
            handleTranscriptionTriggerDown(
                triggerBehavior: behavior,
                triggerMode: behavior.legacyTriggerMode,
                allowsDoubleTapRewrite: false,
                source: "hotkey",
                pendingStartDelay: transcriptionStartDebounceInterval,
                pendingStartShouldStart: { [weak self] in
                    self?.isTranscriptionLongPressHotkeyDown == true
                }
            )
            return
        }

        handleTranscriptionTriggerDown(
            triggerBehavior: behavior,
            triggerMode: behavior.legacyTriggerMode,
            allowsDoubleTapRewrite: false,
            source: "hotkey"
        )
    }

    private func handleTranscriptionTriggerDown(
        triggerBehavior: HotkeyPreference.TriggerBehavior,
        triggerMode: HotkeyPreference.TriggerMode,
        allowsDoubleTapRewrite: Bool,
        source: String,
        pendingStartDelay: TimeInterval? = nil,
        pendingStartShouldStart: (() -> Bool)? = nil
    ) {
        VoxtLog.hotkey(
            "Trigger callback transcriptionDown. source=\(source), mode=\(triggerMode.rawValue), isSessionActive=\(isSessionActive), sessionOutput=\(sessionOutputMode == .translation ? "translation" : "transcription"), pendingStart=\(pendingTranscriptionStartTask != nil)",
            verbose: true
        )
        guard !blockNonMeetingRecordingWhileMeetingIsActive(source: "\(source)TranscriptionDown") else { return }
        if allowsDoubleTapRewrite {
            let doubleTapRewriteAction = TranscriptionDoubleTapRewriteResolver.resolve(
                state: TranscriptionDoubleTapRewriteResolver.State(
                    triggerMode: triggerMode,
                    rewriteActivationMode: HotkeyPreference.loadRewriteActivationMode(),
                    isSessionActive: isSessionActive,
                    hasPendingTranscriptionStart: pendingTranscriptionStartTask != nil
                )
            )
            switch doubleTapRewriteAction {
            case .useStandardHandling:
                break
            case .scheduleDelayedTranscriptionStart:
                let delay = NSEvent.doubleClickInterval
                VoxtLog.hotkey("Transcription tap entering double-tap rewrite wait window. delaySec=\(delay)")
                pendingTranscriptionHotkeyStartBehavior = triggerBehavior
                schedulePendingTranscriptionStart(
                    delay: delay,
                    reason: "doubleTapRewriteWait"
                )
                return
            case .startRewrite:
                VoxtLog.hotkey("Transcription second tap detected; starting rewrite instead of transcription.")
                cancelPendingTranscriptionStart()
                beginRecording(outputMode: .rewrite)
                return
            }
        }
        let actions = HotkeyActionResolver.resolveTranscriptionDown(
            state: HotkeyActionResolver.State(
                triggerMode: triggerMode,
                isSessionActive: isSessionActive,
                sessionOutputMode: sessionOutputMode,
                hasPendingTranscriptionStart: pendingTranscriptionStartTask != nil,
                isSelectedTextTranslationFlow: isSelectedTextTranslationFlow,
                canStopTapSession: !shouldIgnoreTapStop() && !isSessionStopInProgress
            )
        )
        if actions.contains(where: { $0 == .startTranscription || $0 == .scheduleTranscriptionStart }),
           addSelectedShortTextToDictionaryIfPossible() {
            return
        }
        for action in actions {
            if action == .scheduleTranscriptionStart, let pendingStartDelay {
                pendingTranscriptionHotkeyStartBehavior = triggerBehavior
                schedulePendingTranscriptionStart(
                    delay: pendingStartDelay,
                    reason: "\(source)LongPressDebounce",
                    shouldStart: pendingStartShouldStart
                )
            } else {
                if action == .startTranscription || action == .scheduleTranscriptionStart {
                    pendingTranscriptionHotkeyStartBehavior = triggerBehavior
                }
                performHotkeyAction(action)
            }
        }
    }

    func handleTranscriptionHotkeyUp(behavior: HotkeyPreference.TriggerBehavior = .tap) {
        if behavior == .longPress {
            isTranscriptionLongPressHotkeyDown = false
        }
        handleTranscriptionTriggerUp(triggerMode: behavior.legacyTriggerMode, source: "hotkey")
    }

    private func handleTranscriptionTriggerUp(
        triggerMode: HotkeyPreference.TriggerMode,
        source: String
    ) {
        guard triggerMode == .longPress else { return }
        VoxtLog.hotkey(
            "Trigger callback transcriptionUp. source=\(source), isSessionActive=\(isSessionActive), sessionOutput=\(sessionOutputMode == .translation ? "translation" : "transcription"), pendingStart=\(pendingTranscriptionStartTask != nil)",
            verbose: true
        )
        let actions = HotkeyActionResolver.resolveTranscriptionUp(
            state: HotkeyActionResolver.State(
                triggerMode: triggerMode,
                isSessionActive: isSessionActive,
                sessionOutputMode: sessionOutputMode,
                hasPendingTranscriptionStart: pendingTranscriptionStartTask != nil,
                isSelectedTextTranslationFlow: isSelectedTextTranslationFlow,
                canStopTapSession: !shouldIgnoreTapStop() && !isSessionStopInProgress
            )
        )
        for action in actions {
            performHotkeyAction(action)
        }
    }

    func handleTranslationHotkeyDown(behavior: HotkeyPreference.TriggerBehavior = .tap) {
        guard FeatureSettingsStore.availability().translationEnabled else { return }
        handleTranslationTriggerDown(triggerMode: behavior.legacyTriggerMode, source: "hotkey")
    }

    private func handleTranslationTriggerDown(
        triggerMode: HotkeyPreference.TriggerMode,
        source: String
    ) {
        VoxtLog.info(
            "Translation trigger invoked. source=\(source), mode=\(triggerMode.rawValue), isSessionActive=\(isSessionActive), pendingStart=\(pendingTranscriptionStartTask != nil)"
        )
        VoxtLog.hotkey(
            "Trigger callback translationDown. source=\(source), mode=\(triggerMode.rawValue), isSessionActive=\(isSessionActive), sessionOutput=\(sessionOutputMode == .translation ? "translation" : "transcription"), pendingStart=\(pendingTranscriptionStartTask != nil)",
            verbose: true
        )
        guard !blockNonMeetingRecordingWhileMeetingIsActive(source: "\(source)TranslationDown") else { return }
        let actions = HotkeyActionResolver.resolveTranslationDown(
            state: HotkeyActionResolver.State(
                triggerMode: triggerMode,
                isSessionActive: isSessionActive,
                sessionOutputMode: sessionOutputMode,
                hasPendingTranscriptionStart: pendingTranscriptionStartTask != nil,
                isSelectedTextTranslationFlow: isSelectedTextTranslationFlow,
                canStopTapSession: !shouldIgnoreTapStop() && !isSessionStopInProgress
            )
        )
        for action in actions where action == .cancelPendingTranscriptionStart {
            performHotkeyAction(action)
        }
        guard !isSessionActive else {
            VoxtLog.info("Translation hotkey ignored because a session is already active.")
            VoxtLog.hotkey("Translation down ignored: session already active.")
            return
        }

        if beginSelectedTextTranslationIfPossible() {
            VoxtLog.hotkey("Translation down handled by selected-text translation flow.")
            return
        }

        VoxtLog.info("Translation hotkey dispatching microphone translation start.")
        for action in actions {
            guard action != .cancelPendingTranscriptionStart else { continue }
            performHotkeyAction(action)
        }
    }

    func handleTranslationHotkeyUp(behavior: HotkeyPreference.TriggerBehavior = .tap) {
        handleTranslationTriggerUp(triggerMode: behavior.legacyTriggerMode, source: "hotkey")
    }

    private func handleTranslationTriggerUp(
        triggerMode: HotkeyPreference.TriggerMode,
        source: String
    ) {
        guard triggerMode == .longPress else { return }
        VoxtLog.hotkey(
            "Trigger callback translationUp. source=\(source), isSessionActive=\(isSessionActive), sessionOutput=\(sessionOutputMode == .translation ? "translation" : "transcription"), selectedTextFlow=\(isSelectedTextTranslationFlow)",
            verbose: true
        )
        let actions = HotkeyActionResolver.resolveTranslationUp(
            state: HotkeyActionResolver.State(
                triggerMode: triggerMode,
                isSessionActive: isSessionActive,
                sessionOutputMode: sessionOutputMode,
                hasPendingTranscriptionStart: pendingTranscriptionStartTask != nil,
                isSelectedTextTranslationFlow: isSelectedTextTranslationFlow,
                canStopTapSession: !shouldIgnoreTapStop() && !isSessionStopInProgress
            )
        )
        for action in actions {
            performHotkeyAction(action)
        }
    }

    func handleRewriteHotkeyDown(behavior: HotkeyPreference.TriggerBehavior = .tap) {
        guard FeatureSettingsStore.availability().rewriteEnabled else { return }
        handleRewriteTriggerDown(triggerMode: behavior.legacyTriggerMode, source: "hotkey")
    }

    private func handleRewriteTriggerDown(
        triggerMode: HotkeyPreference.TriggerMode,
        source: String
    ) {
        VoxtLog.info(
            "Rewrite trigger invoked. source=\(source), mode=\(triggerMode.rawValue), isSessionActive=\(isSessionActive), pendingStart=\(pendingTranscriptionStartTask != nil)"
        )
        VoxtLog.hotkey(
            "Trigger callback rewriteDown. source=\(source), mode=\(triggerMode.rawValue), isSessionActive=\(isSessionActive), sessionOutput=\(sessionOutputModeLabel), pendingStart=\(pendingTranscriptionStartTask != nil)",
            verbose: true
        )
        guard !blockNonMeetingRecordingWhileMeetingIsActive(source: "\(source)RewriteDown") else { return }

        cancelPendingTranscriptionStart()
        if isSessionActive {
            if sessionOutputMode == .transcription && shouldIgnoreTapStop() {
                VoxtLog.hotkey("Rewrite down reinterpreting freshly started transcription session as rewrite.")
                cancelActiveRecordingSession()
                beginRecording(outputMode: .rewrite)
                return
            }

            VoxtLog.info("Rewrite hotkey ignored because a session is already active.")
            VoxtLog.hotkey("Rewrite down ignored: session already active.")
            return
        }

        VoxtLog.info("Rewrite hotkey dispatching rewrite recording start.")
        beginRecording(outputMode: .rewrite)
    }

    func handleRewriteHotkeyUp(behavior: HotkeyPreference.TriggerBehavior = .tap) {
        handleRewriteTriggerUp(triggerMode: behavior.legacyTriggerMode, source: "hotkey")
    }

    private func handleRewriteTriggerUp(
        triggerMode: HotkeyPreference.TriggerMode,
        source: String
    ) {
        guard triggerMode == .longPress else { return }
        VoxtLog.hotkey(
            "Trigger callback rewriteUp. source=\(source), isSessionActive=\(isSessionActive), sessionOutput=\(sessionOutputModeLabel)",
            verbose: true
        )
        guard isSessionActive, sessionOutputMode == .rewrite else { return }
        endRecording()
    }

    private func postHotkeyDidTrigger(kind: String, behavior: HotkeyPreference.TriggerBehavior) {
        NotificationCenter.default.post(
            name: .voxtHotkeyDidTrigger,
            object: nil,
            userInfo: [
                "kind": kind,
                "behavior": behavior.rawValue
            ]
        )
    }

    func handleCustomPasteHotkeyDown() {
        guard customPasteHotkeyEnabled else { return }
        injectLatestResultByCustomPasteHotkey()
    }

    func performHotkeyAction(_ action: HotkeyActionResolver.Action) {
        switch action {
        case .ignore:
            return
        case .stopRecording:
            endRecording()
        case .startTranscription:
            beginRecording(outputMode: .transcription)
        case .startTranslation:
            beginRecording(outputMode: .translation)
        case .scheduleTranscriptionStart:
            schedulePendingTranscriptionStart()
        case .cancelPendingTranscriptionStart:
            cancelPendingTranscriptionStart()
        }
    }

    func schedulePendingTranscriptionStart() {
        schedulePendingTranscriptionStart(
            delay: transcriptionStartDebounceInterval,
            reason: "longPressDebounce"
        )
    }

    func schedulePendingTranscriptionStart(
        delay: TimeInterval,
        reason: String,
        shouldStart: (() -> Bool)? = nil
    ) {
        guard !blockNonMeetingRecordingWhileMeetingIsActive(source: "pendingTranscriptionStart:\(reason)") else { return }
        VoxtLog.hotkey("Scheduling pending transcription start. delaySec=\(delay), reason=\(reason)")
        pendingTranscriptionStartTask?.cancel()
        pendingTranscriptionStartTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await Task.sleep(for: .seconds(delay))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            guard !self.isSessionActive else {
                VoxtLog.hotkey("Pending transcription start dropped: session already active.")
                self.pendingTranscriptionStartTask = nil
                return
            }
            guard !self.blockNonMeetingRecordingWhileMeetingIsActive(source: "pendingTranscriptionStartFire:\(reason)") else {
                self.pendingTranscriptionStartTask = nil
                return
            }
            if let shouldStart, !shouldStart() {
                VoxtLog.hotkey("Pending transcription start dropped: trigger is no longer active. reason=\(reason)")
                self.pendingTranscriptionStartTask = nil
                return
            }
            self.pendingTranscriptionStartTask = nil
            VoxtLog.hotkey("Pending transcription start fired.")
            self.beginRecording(outputMode: .transcription)
        }
    }

    func cancelPendingTranscriptionStart() {
        if pendingTranscriptionStartTask != nil {
            VoxtLog.hotkey("Canceled pending transcription start.")
        }
        pendingTranscriptionStartTask?.cancel()
        pendingTranscriptionStartTask = nil
        pendingTranscriptionHotkeyStartBehavior = nil
    }

    nonisolated static func sessionCallbackHandlingDecision(
        requestedSessionID: UUID,
        activeSessionID: UUID,
        isSessionCancellationRequested: Bool
    ) -> SessionCallbackHandlingDecision {
        guard requestedSessionID == activeSessionID else {
            return .rejectStale
        }
        guard !isSessionCancellationRequested else {
            return .rejectCancelled
        }
        return .accept
    }

    func shouldHandleCallbacks(for sessionID: UUID) -> Bool {
        switch Self.sessionCallbackHandlingDecision(
            requestedSessionID: sessionID,
            activeSessionID: activeRecordingSessionID,
            isSessionCancellationRequested: isSessionCancellationRequested
        ) {
        case .accept:
            return true
        case .rejectStale:
            VoxtLog.info("Ignoring stale session callback. sessionID=\(sessionID.uuidString)", verbose: true)
            return false
        case .rejectCancelled:
            VoxtLog.info("Ignoring callback for cancelled session. sessionID=\(sessionID.uuidString)", verbose: true)
            return false
        }
    }

    var sessionOutputModeLabel: String {
        switch sessionOutputMode {
        case .transcription:
            return "transcription"
        case .translation:
            return "translation"
        case .rewrite:
            return "rewrite"
        }
    }
}
