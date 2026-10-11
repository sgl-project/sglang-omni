// FeatureShortcutComponents.swift
// Provides Feature Shortcut Components for feature settings.

import SwiftUI

extension Notification.Name {
    static let voxtFeatureSettingsToastRequested = Notification.Name("voxtFeatureSettingsToastRequested")
}

struct FeatureShortcutCaptureRow: View {
    let title: String
    let detail: String
    var showsHeader = true
    var inputWidth: CGFloat = 320
    @Binding var hotkey: HotkeyPreference.Hotkey
    let defaultHotkey: HotkeyPreference.Hotkey

    @State private var isRecording = false
    @State private var recorderMessage: String?
    @State private var pendingCapturedHotkey: HotkeyPreference.Hotkey?

    var body: some View {
        VStack(alignment: .leading, spacing: showsHeader ? 12 : 0) {
            if showsHeader {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    if !detail.isEmpty {
                        Text(detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            SettingsShortcutCaptureField(
                title: LocalizedStringKey(title),
                hotkey: pendingCapturedHotkey ?? hotkey,
                isRecording: isRecording,
                isPendingConfirmation: pendingCapturedHotkey != nil,
                distinguishModifierSides: false,
                controlWidth: inputWidth,
                onFocus: {
                    pendingCapturedHotkey = nil
                    isRecording = true
                    showShortcutToast(featureSettingsLocalized("Type your shortcut now. Press Esc to cancel recording."))
                },
                onReset: {
                    hotkey = defaultHotkey
                    pendingCapturedHotkey = nil
                    isRecording = false
                    recorderMessage = nil
                },
                onCancelPending: {
                    pendingCapturedHotkey = nil
                    isRecording = false
                    recorderMessage = nil
                    dismissShortcutToast()
                },
                onConfirmPending: {
                    if let pendingCapturedHotkey {
                        hotkey = pendingCapturedHotkey
                    }
                    self.pendingCapturedHotkey = nil
                    isRecording = false
                    recorderMessage = nil
                    dismissShortcutToast()
                }
            )

            HotkeyRecorderView(
                isRecording: $isRecording,
                onCapture: { capturedHotkey in
                    pendingCapturedHotkey = capturedHotkey
                    recorderMessage = nil
                    showShortcutToast(featureSettingsLocalized("Shortcut captured. Press another shortcut to replace it, or choose Confirm / Cancel."))
                },
                onCancelCapture: {
                    pendingCapturedHotkey = nil
                    isRecording = false
                    recorderMessage = nil
                    dismissShortcutToast()
                },
                onRecorderMessageChange: { messageKey in
                    DispatchQueue.main.async {
                        recorderMessage = messageKey
                        if let messageKey {
                            showShortcutToast(featureSettingsLocalized(messageKey))
                        }
                    }
                }
            )
            .frame(width: 0, height: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func showShortcutToast(_ message: String) {
        NotificationCenter.default.post(
            name: .voxtFeatureSettingsToastRequested,
            object: nil,
            userInfo: ["message": message]
        )
    }

    private func dismissShortcutToast() {
        NotificationCenter.default.post(
            name: .voxtFeatureSettingsToastRequested,
            object: nil,
            userInfo: ["message": ""]
        )
    }
}

struct FeatureNoteShortcutRow: View {
    let title: String
    let detail: String
    var showsHeader = false
    var inputWidth: CGFloat = 263.2
    @Binding var shortcut: TranscriptionNoteTriggerSettings

    var body: some View {
        FeatureShortcutCaptureRow(
            title: title,
            detail: detail,
            showsHeader: showsHeader,
            inputWidth: inputWidth,
            hotkey: Binding(
                get: { shortcut.hotkey },
                set: { capturedHotkey in
                    shortcut = TranscriptionNoteTriggerSettings(
                        keyCode: capturedHotkey.keyCode,
                        modifiers: capturedHotkey.modifiers,
                        sidedModifiers: capturedHotkey.sidedModifiers
                    )
                }
            ),
            defaultHotkey: TranscriptionNoteTriggerSettings.defaultShortcut.hotkey
        )
    }
}

struct FeatureContinueShortcutRow: View {
    let title: String
    let detail: String
    var showsHeader = false
    @Binding var shortcut: TranscriptionContinueShortcutSettings

    var body: some View {
        FeatureShortcutCaptureRow(
            title: title,
            detail: detail,
            showsHeader: showsHeader,
            hotkey: Binding(
                get: { shortcut.hotkey },
                set: { capturedHotkey in
                    shortcut = TranscriptionContinueShortcutSettings(
                        keyCode: capturedHotkey.keyCode,
                        modifiers: capturedHotkey.modifiers,
                        sidedModifiers: capturedHotkey.sidedModifiers
                    )
                }
            ),
            defaultHotkey: TranscriptionContinueShortcutSettings.defaultShortcut.hotkey
        )
    }
}
