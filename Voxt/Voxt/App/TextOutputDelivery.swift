import Foundation
import AppKit
import ApplicationServices

nonisolated struct TextInjectionTarget: Sendable {
    let processIdentifier: Int32?
    let bundleIdentifier: String?
}

extension AppDelegate {
    var sessionTextInjectionTarget: TextInjectionTarget {
        TextInjectionTarget(
            processIdentifier: sessionTargetApplicationPID,
            bundleIdentifier: sessionTargetApplicationBundleID
        )
    }

    @discardableResult
    private func restoreSessionTargetApplicationIfNeeded(_ target: TextInjectionTarget) -> Bool {
        guard let ownBundleID = Bundle.main.bundleIdentifier else { return false }
        let frontmostApplication = NSWorkspace.shared.frontmostApplication
        let frontmostBundleID = frontmostApplication?.bundleIdentifier
        let frontmostPID = frontmostApplication?.processIdentifier
        let ownPID = ProcessInfo.processInfo.processIdentifier

        guard frontmostPID == ownPID || frontmostBundleID == ownBundleID else {
            return false
        }

        if let targetPID = target.processIdentifier,
           let targetApplication = NSRunningApplication(processIdentifier: targetPID),
           !targetApplication.isTerminated {
            VoxtLog.input(
                "Restoring focus to session target app before text injection. bundleID=\(targetApplication.bundleIdentifier ?? "unknown"), pid=\(targetPID)"
            )
            return targetApplication.activate(options: [])
        }

        if let targetBundleID = target.bundleIdentifier,
           let targetApplication = NSRunningApplication.runningApplications(withBundleIdentifier: targetBundleID)
            .first(where: { !$0.isTerminated }) {
            VoxtLog.input(
                "Restoring focus to session target app by bundle ID before text injection. bundleID=\(targetBundleID), pid=\(targetApplication.processIdentifier)"
            )
            return targetApplication.activate(options: [])
        }

        VoxtLog.input(
            "Session target app restoration skipped: target app unavailable. targetBundleID=\(target.bundleIdentifier ?? "nil"), targetPID=\(target.processIdentifier.map(String.init) ?? "nil")"
        )
        return false
    }

    func typeText(
        _ text: String,
        restoreSessionTarget: Bool = true,
        target: TextInjectionTarget? = nil,
        isValid: @escaping @MainActor () -> Bool = { true },
        completion: TextInjectionTransaction.Completion? = nil
    ) {
        guard !text.isEmpty, !isApplicationTerminating, isValid() else {
            completion?(false)
            return
        }

        let outputGeneration = recordingLifecycle.outputGeneration
        let injectionStartedAt = Date()
        let accessibilityTrusted = AccessibilityPermissionManager.isTrusted()
        let keepResultInClipboard = autoCopyWhenNoFocusedInput

        guard accessibilityTrusted else {
            writeTextToPasteboard(text)
            promptForAccessibilityPermission()
            VoxtLog.inputWarning("Accessibility permission missing. Transcription copied; paste manually after granting permission.")
            completion?(false)
            return
        }

        let activationRestored = restoreSessionTarget
            ? restoreSessionTargetApplicationIfNeeded(target ?? sessionTextInjectionTarget)
            : false
        let activationDelay: TimeInterval = activationRestored ? 0.04 : 0
        VoxtLog.input(
            "Text injection prepared. characters=\(text.count), activationRestored=\(activationRestored), restoreSessionTarget=\(restoreSessionTarget), activationDelayMs=\(Int(activationDelay * 1000))",
            verbose: true
        )
        let transaction = TextInjectionTransaction(
            isValid: { [weak self] in
                guard let self, !self.isApplicationTerminating,
                      self.recordingLifecycle.acceptsOutputGeneration(outputGeneration)
                else { return false }
                return isValid()
            },
            inject: { [weak self] onInjected in
                guard let self else { onInjected(false); return }
                self.pasteTextByShortcut(
                    text,
                    keepResultInClipboard: keepResultInClipboard,
                    completion: onInjected
                )
            },
            completion: { didInject in
                let elapsedMs = Int(Date().timeIntervalSince(injectionStartedAt) * 1000)
                VoxtLog.input(
                    "Text injection attempt completed. characters=\(text.count), elapsedMs=\(elapsedMs), didInject=\(didInject)"
                )
                completion?(didInject)
            }
        )
        DispatchQueue.main.asyncAfter(deadline: .now() + activationDelay) {
            transaction.perform()
        }
    }

    func pressAutoKeyAfterTextInjection(
        _ hotkey: HotkeyPreference.Hotkey,
        outputGeneration: UUID,
        delay: TimeInterval = 0.12
    ) {
        guard let targetPID = NSWorkspace.shared.frontmostApplication?.processIdentifier else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, !self.isApplicationTerminating,
                  self.recordingLifecycle.acceptsOutputGeneration(outputGeneration),
                  NSWorkspace.shared.frontmostApplication?.processIdentifier == targetPID
            else { return }
            guard AccessibilityPermissionManager.isTrusted() else {
                VoxtLog.inputWarning("Auto Key skipped: accessibility permission missing.")
                return
            }
            guard let source = CGEventSource(stateID: .hidSystemState) else {
                VoxtLog.error("Auto Key failed: unable to create CGEventSource")
                return
            }
            guard case .keyboard(let keyCode) = hotkey.input,
                  keyCode != HotkeyPreference.modifierOnlyKeyCode
            else {
                VoxtLog.inputWarning("Auto Key skipped: unsupported shortcut input.")
                return
            }

            let cgKeyCode = CGKeyCode(keyCode)
            let flags = Self.cgEventFlags(for: hotkey.modifiers)
            let keyDown = CGEvent(keyboardEventSource: source, virtualKey: cgKeyCode, keyDown: true)
            let keyUp = CGEvent(keyboardEventSource: source, virtualKey: cgKeyCode, keyDown: false)

            guard let keyDown, let keyUp else {
                VoxtLog.error("Auto Key failed: unable to create key events")
                return
            }

            keyDown.flags = flags
            keyUp.flags = flags
            HotkeyEventSupport.markAsVoxtInjected(keyDown)
            HotkeyEventSupport.markAsVoxtInjected(keyUp)
            keyDown.post(tap: .cgAnnotatedSessionEventTap)
            keyUp.post(tap: .cgAnnotatedSessionEventTap)
            VoxtLog.input(
                "Auto Key event posted after text injection. hotkey=\(HotkeyPreference.displayString(for: hotkey, distinguishModifierSides: false))",
                verbose: true
            )
        }
    }

    private static func cgEventFlags(for modifiers: NSEvent.ModifierFlags) -> CGEventFlags {
        var flags: CGEventFlags = []
        if modifiers.contains(.control) {
            flags.insert(.maskControl)
        }
        if modifiers.contains(.option) {
            flags.insert(.maskAlternate)
        }
        if modifiers.contains(.shift) {
            flags.insert(.maskShift)
        }
        if modifiers.contains(.command) {
            flags.insert(.maskCommand)
        }
        if modifiers.contains(.function) {
            flags.insert(.maskSecondaryFn)
        }
        return flags
    }

    func writeTextToPasteboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    func cacheLatestInjectableOutputText(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        latestInjectableOutputText = trimmed
    }

    private func resolvedLatestInjectableOutputText() -> String? {
        let cached = latestInjectableOutputText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !cached.isEmpty {
            return cached
        }

        let historyText = historyStore.latestEntryText()?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return historyText.isEmpty ? nil : historyText
    }

    func injectLatestResultByCustomPasteHotkey() {
        guard let latestText = resolvedLatestInjectableOutputText() else {
            showOverlayStatus(AppLocalization.localizedString("No recent result available to paste yet."), clearAfter: 2.0)
            return
        }

        typeText(latestText)
    }

    private func pasteTextByShortcut(
        _ text: String,
        keepResultInClipboard: Bool,
        completion: TextInjectionTransaction.Completion
    ) {
        guard let source = CGEventSource(stateID: .hidSystemState) else {
            writeTextToPasteboard(text)
            VoxtLog.error("typeText fallback failed: unable to create CGEventSource")
            completion(false)
            return
        }

        let vKeyCode: CGKeyCode = 0x09
        let cmdDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true)
        cmdDown?.flags = .maskCommand
        let cmdUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false)
        cmdUp?.flags = .maskCommand

        guard cmdDown != nil, cmdUp != nil else {
            writeTextToPasteboard(text)
            VoxtLog.error("typeText fallback failed: unable to create key events")
            completion(false)
            return
        }

        let restoration = pasteboardTextWriter.write(
            text, to: .general, restorePrevious: !keepResultInClipboard
        )
        HotkeyEventSupport.markAsVoxtInjected(cmdDown)
        HotkeyEventSupport.markAsVoxtInjected(cmdUp)
        cmdDown?.post(tap: .cgAnnotatedSessionEventTap)
        cmdUp?.post(tap: .cgAnnotatedSessionEventTap)
        completion(true)

        guard let restoration else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
            self?.pasteboardTextWriter.restoreIfUnchanged(restoration)
        }
    }

    private func promptForAccessibilityPermission() {
        _ = AccessibilityPermissionManager.request(prompt: true)
    }
}
