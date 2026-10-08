import SwiftUI
import AppKit

extension OnboardingGuideView {
    func shortcutSheet(for kind: OnboardingGuideShortcutKind) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            OnboardingShortcutCaptureRow(
                title: AppLocalization.format("%@ %@", kind.title, AppLocalization.localizedString("Shortcut")),
                detail: AppLocalization.localizedString("Capture a new shortcut for this workflow. The change is saved immediately after confirmation."),
                shortcut: shortcutBinding(for: kind),
                defaultHotkey: kind.defaultHotkey
            )

            HStack {
                Spacer()
                Button(AppLocalization.localizedString("Done")) {
                    editingShortcut = nil
                    updateFocusedField()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
            }
        }
        .settingsDialogChrome(width: 460, cornerRadius: OnboardingGuideStyle.modalCornerRadius, onClose: {
            editingShortcut = nil
            updateFocusedField()
        })
    }

    private func shortcutBinding(for kind: OnboardingGuideShortcutKind) -> Binding<HotkeyPreference.HotkeyBinding> {
        Binding(
            get: {
                shortcutBindings(for: kind).first ?? HotkeyPreference.HotkeyBinding(
                    hotkey: kind.defaultHotkey,
                    behavior: .tap
                )
            },
            set: { binding in
                hotkeyPresetRaw = HotkeyPreference.Preset.custom.rawValue
                var bindings = shortcutBindings(for: kind)
                if bindings.isEmpty {
                    bindings = [binding]
                } else {
                    bindings[0] = binding
                }
                saveShortcutBindings(bindings, for: kind)
            }
        )
    }

    func shortcutDisplay(for kind: OnboardingGuideShortcutKind) -> String {
        let hotkey: HotkeyPreference.Hotkey
        switch kind {
        case .transcription:
            hotkey = shortcutBindings(for: kind).first?.hotkey ?? HotkeyPreference.load()
        case .translation:
            hotkey = shortcutBindings(for: kind).first?.hotkey ?? HotkeyPreference.loadTranslation()
        case .rewrite:
            hotkey = shortcutBindings(for: kind).first?.hotkey ?? HotkeyPreference.loadRewrite()
        case .meeting:
            hotkey = shortcutBindings(for: kind).first?.hotkey ?? HotkeyPreference.loadMeeting()
        }
        return HotkeyPreference.displayString(for: hotkey, distinguishModifierSides: distinguishModifierSides)
    }

    func shortcutBindings(for kind: OnboardingGuideShortcutKind) -> [HotkeyPreference.HotkeyBinding] {
        switch kind {
        case .transcription:
            return HotkeyPreference.loadTranscriptionBindings()
        case .translation:
            return HotkeyPreference.loadTranslationBindings()
        case .rewrite:
            return HotkeyPreference.loadRewriteBindings()
        case .meeting:
            return HotkeyPreference.loadMeetingBindings()
        }
    }

    private func saveShortcutBindings(
        _ bindings: [HotkeyPreference.HotkeyBinding],
        for kind: OnboardingGuideShortcutKind
    ) {
        switch kind {
        case .transcription:
            HotkeyPreference.saveTranscriptionBindings(bindings)
        case .translation:
            HotkeyPreference.saveTranslationBindings(bindings)
        case .rewrite:
            HotkeyPreference.saveRewriteBindings(bindings)
        case .meeting:
            HotkeyPreference.saveMeetingBindings(bindings)
        }
    }
}

enum OnboardingGuideShortcutKind: String, CaseIterable, Identifiable {
    case transcription
    case translation
    case rewrite
    case meeting

    var id: String { rawValue }

    var title: String {
        switch self {
        case .transcription:
            return AppLocalization.localizedString("Transcription")
        case .translation:
            return AppLocalization.localizedString("Translation")
        case .rewrite:
            return AppLocalization.localizedString("Rewrite")
        case .meeting:
            return AppLocalization.localizedString("Meeting")
        }
    }

    var defaultHotkey: HotkeyPreference.Hotkey {
        switch self {
        case .transcription:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultKeyCode,
                modifiers: HotkeyPreference.defaultModifiers,
                sidedModifiers: []
            )
        case .translation:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultTranslationKeyCode,
                modifiers: HotkeyPreference.defaultTranslationModifiers,
                sidedModifiers: []
            )
        case .rewrite:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultRewriteKeyCode,
                modifiers: HotkeyPreference.defaultRewriteModifiers,
                sidedModifiers: []
            )
        case .meeting:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultMeetingKeyCode,
                modifiers: HotkeyPreference.defaultMeetingModifiers,
                sidedModifiers: []
            )
        }
    }
}

private struct OnboardingShortcutCaptureRow: View {
    let title: String
    let detail: String
    @Binding var shortcut: HotkeyPreference.HotkeyBinding
    let defaultHotkey: HotkeyPreference.Hotkey

    @State private var isRecording = false
    @State private var pendingCapturedHotkey: HotkeyPreference.Hotkey?

    private var behaviorSelection: Binding<HotkeyPreference.TriggerBehavior> {
        Binding(
            get: { shortcut.behavior },
            set: { behavior in
                shortcut.behavior = behavior
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                if !detail.isEmpty {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            GeometryReader { proxy in
                let pickerWidth: CGFloat = 112
                HStack(alignment: .center, spacing: 8) {
                    SettingsShortcutCaptureField(
                        title: LocalizedStringKey(""),
                        hotkey: pendingCapturedHotkey ?? shortcut.hotkey,
                        isRecording: isRecording,
                        isPendingConfirmation: pendingCapturedHotkey != nil,
                        distinguishModifierSides: false,
                        showsTitle: false,
                        controlWidth: max(280, proxy.size.width - pickerWidth - 8),
                        onFocus: {
                            pendingCapturedHotkey = nil
                            isRecording = true
                        },
                        onReset: {
                            shortcut.hotkey = defaultHotkey
                            shortcut.behavior = .tap
                            pendingCapturedHotkey = nil
                            isRecording = false
                        },
                        onCancelPending: {
                            pendingCapturedHotkey = nil
                            isRecording = false
                        },
                        onConfirmPending: {
                            if let pendingCapturedHotkey {
                                shortcut.hotkey = pendingCapturedHotkey
                            }
                            pendingCapturedHotkey = nil
                            isRecording = false
                        }
                    )

                    SettingsMenuPicker(
                        selection: behaviorSelection,
                        options: HotkeyPreference.TriggerBehavior.allCases.map { behavior in
                            SettingsMenuOption(value: behavior, title: behavior.title)
                        },
                        selectedTitle: shortcut.behavior.title,
                        width: pickerWidth,
                        allowsCompactWidth: true,
                        usesCompactInsets: true
                    )
                }
            }
            .frame(height: 34)

            HotkeyRecorderView(
                isRecording: $isRecording,
                onCapture: { capturedHotkey in
                    pendingCapturedHotkey = capturedHotkey
                },
                onCancelCapture: {
                    pendingCapturedHotkey = nil
                    isRecording = false
                },
                onRecorderMessageChange: { _ in }
            )
            .frame(width: 0, height: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
