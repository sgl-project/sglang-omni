// HotkeySettingsView.swift
// Provides Hotkey Settings View for settings screens.

import SwiftUI
import AppKit
import Carbon

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

enum HotkeyShortcutKind: String, CaseIterable, Hashable {
    case transcription
    case note
    case translation
    case rewrite
    case meeting

    var titleKey: LocalizedStringKey {
        switch self {
        case .transcription:
            return "Transcription"
        case .note:
            return "Notes"
        case .translation:
            return "Translation"
        case .meeting:
            return "Meeting"
        case .rewrite:
            return "Rewrite"
        }
    }

    var localizedTitle: String {
        switch self {
        case .transcription:
            return localized("Transcription")
        case .note:
            return localized("Notes")
        case .translation:
            return localized("Translation")
        case .meeting:
            return localized("Meeting")
        case .rewrite:
            return localized("Rewrite")
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
        case .note:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultNoteKeyCode,
                modifiers: HotkeyPreference.defaultNoteModifiers,
                sidedModifiers: HotkeyPreference.defaultNoteSidedModifiers
            )
        case .translation:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultTranslationKeyCode,
                modifiers: HotkeyPreference.defaultTranslationModifiers,
                sidedModifiers: []
            )
        case .meeting:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultMeetingKeyCode,
                modifiers: HotkeyPreference.defaultMeetingModifiers,
                sidedModifiers: []
            )
        case .rewrite:
            return HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultRewriteKeyCode,
                modifiers: HotkeyPreference.defaultRewriteModifiers,
                sidedModifiers: []
            )
        }
    }
}

enum HotkeyShortcutVisibility {
    static func visibleKinds(defaults: UserDefaults = .standard) -> [HotkeyShortcutKind] {
        let availability = FeatureSettingsStore.availability(defaults: defaults)
        var kinds: [HotkeyShortcutKind] = [.transcription]
        if availability.notesEnabled {
            kinds.append(.note)
        }
        if availability.translationEnabled {
            kinds.append(.translation)
        }
        if availability.rewriteEnabled {
            kinds.append(.rewrite)
        }
        if availability.meetingEnabled {
            kinds.append(.meeting)
        }
        return kinds
    }
}

struct HotkeySettingsView: View {
    private enum RecordingField: Equatable {
        case binding(HotkeyShortcutKind, UUID)
        case customPaste
    }

    @AppStorage(AppPreferenceKey.hotkeyInputType) private var hotkeyInputType = HotkeyPreference.Hotkey.Input.Kind.keyboard.rawValue
    @AppStorage(AppPreferenceKey.hotkeyKeyCode) private var hotkeyKeyCode = Int(HotkeyPreference.defaultKeyCode)
    @AppStorage(AppPreferenceKey.hotkeyMouseButtonNumber) private var hotkeyMouseButtonNumber = HotkeyPreference.middleMouseButtonNumber
    @AppStorage(AppPreferenceKey.hotkeyModifiers) private var hotkeyModifiers = Int(HotkeyPreference.defaultModifiers.rawValue)
    @AppStorage(AppPreferenceKey.hotkeySidedModifiers) private var hotkeySidedModifiers = 0
    @AppStorage(AppPreferenceKey.translationHotkeyInputType) private var translationHotkeyInputType = HotkeyPreference.Hotkey.Input.Kind.keyboard.rawValue
    @AppStorage(AppPreferenceKey.translationHotkeyKeyCode) private var translationHotkeyKeyCode = Int(HotkeyPreference.defaultTranslationKeyCode)
    @AppStorage(AppPreferenceKey.translationHotkeyMouseButtonNumber) private var translationHotkeyMouseButtonNumber = HotkeyPreference.middleMouseButtonNumber
    @AppStorage(AppPreferenceKey.translationHotkeyModifiers) private var translationHotkeyModifiers = Int(HotkeyPreference.defaultTranslationModifiers.rawValue)
    @AppStorage(AppPreferenceKey.translationHotkeySidedModifiers) private var translationHotkeySidedModifiers = 0
    @AppStorage(AppPreferenceKey.meetingHotkeyInputType) private var meetingHotkeyInputType = HotkeyPreference.Hotkey.Input.Kind.keyboard.rawValue
    @AppStorage(AppPreferenceKey.meetingHotkeyKeyCode) private var meetingHotkeyKeyCode = Int(HotkeyPreference.defaultMeetingKeyCode)
    @AppStorage(AppPreferenceKey.meetingHotkeyMouseButtonNumber) private var meetingHotkeyMouseButtonNumber = HotkeyPreference.middleMouseButtonNumber
    @AppStorage(AppPreferenceKey.meetingHotkeyModifiers) private var meetingHotkeyModifiers = Int(HotkeyPreference.defaultMeetingModifiers.rawValue)
    @AppStorage(AppPreferenceKey.meetingHotkeySidedModifiers) private var meetingHotkeySidedModifiers = 0
    @AppStorage(AppPreferenceKey.rewriteHotkeyInputType) private var rewriteHotkeyInputType = HotkeyPreference.Hotkey.Input.Kind.keyboard.rawValue
    @AppStorage(AppPreferenceKey.rewriteHotkeyKeyCode) private var rewriteHotkeyKeyCode = Int(HotkeyPreference.defaultRewriteKeyCode)
    @AppStorage(AppPreferenceKey.rewriteHotkeyMouseButtonNumber) private var rewriteHotkeyMouseButtonNumber = HotkeyPreference.middleMouseButtonNumber
    @AppStorage(AppPreferenceKey.rewriteHotkeyModifiers) private var rewriteHotkeyModifiers = Int(HotkeyPreference.defaultRewriteModifiers.rawValue)
    @AppStorage(AppPreferenceKey.rewriteHotkeySidedModifiers) private var rewriteHotkeySidedModifiers = 0
    @AppStorage(AppPreferenceKey.rewriteHotkeyActivationMode) private var rewriteHotkeyActivationMode = HotkeyPreference.defaultRewriteActivationMode.rawValue
    @AppStorage(AppPreferenceKey.customPasteHotkeyEnabled) private var customPasteHotkeyEnabled = false
    @AppStorage(AppPreferenceKey.customPasteHotkeyInputType) private var customPasteHotkeyInputType = HotkeyPreference.Hotkey.Input.Kind.keyboard.rawValue
    @AppStorage(AppPreferenceKey.customPasteHotkeyKeyCode) private var customPasteHotkeyKeyCode = Int(HotkeyPreference.defaultCustomPasteKeyCode)
    @AppStorage(AppPreferenceKey.customPasteHotkeyMouseButtonNumber) private var customPasteHotkeyMouseButtonNumber = HotkeyPreference.middleMouseButtonNumber
    @AppStorage(AppPreferenceKey.customPasteHotkeyModifiers) private var customPasteHotkeyModifiers = Int(HotkeyPreference.defaultCustomPasteModifiers.rawValue)
    @AppStorage(AppPreferenceKey.customPasteHotkeySidedModifiers) private var customPasteHotkeySidedModifiers = 0
    @AppStorage(AppPreferenceKey.hotkeyTriggerMode) private var hotkeyTriggerMode = HotkeyPreference.defaultTriggerMode.rawValue
    @AppStorage(AppPreferenceKey.hotkeyDistinguishModifierSides) private var distinguishModifierSides = HotkeyPreference.defaultDistinguishModifierSides
    @AppStorage(AppPreferenceKey.hotkeyPreset) private var hotkeyPreset = HotkeyPreference.defaultPreset.rawValue
    @AppStorage(AppPreferenceKey.escapeKeyCancelsOverlaySession) private var escapeKeyCancelsOverlaySession = true
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.featureSettings) private var featureSettingsRaw = ""
    @State private var recordingField: RecordingField?
    @State private var pendingCapturedField: RecordingField?
    @State private var pendingCapturedHotkey: HotkeyPreference.Hotkey?
    @State private var recorderMessageKey: String?
    @State private var hotkeyToastMessage = ""
    @State private var hotkeyToastDismissTask: Task<Void, Never>?
    @State private var transcriptionBindings = HotkeyPreference.loadTranscriptionBindings()
    @State private var noteBindings = HotkeyPreference.loadNoteBindings()
    @State private var translationBindings = HotkeyPreference.loadTranslationBindings()
    @State private var meetingBindings = HotkeyPreference.loadMeetingBindings()
    @State private var rewriteBindings = HotkeyPreference.loadRewriteBindings()

    private var hotkeyBinding: Binding<UInt16> {
        Binding(
            get: { UInt16(hotkeyKeyCode) },
            set: {
                hotkeyKeyCode = Int($0)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var modifierBinding: Binding<NSEvent.ModifierFlags> {
        Binding(
            get: { NSEvent.ModifierFlags(rawValue: UInt(hotkeyModifiers)).intersection(.hotkeyRelevant) },
            set: {
                hotkeyModifiers = Int($0.rawValue)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var currentHotkey: HotkeyPreference.Hotkey {
        HotkeyPreference.Hotkey(
            input: hotkeyInput,
            modifiers: modifierBinding.wrappedValue,
            sidedModifiers: sidedModifierBinding.wrappedValue
        )
    }

    private var hotkeyInput: HotkeyPreference.Hotkey.Input {
        resolvedInput(
            inputType: hotkeyInputType,
            keyCode: hotkeyKeyCode,
            mouseButtonNumber: hotkeyMouseButtonNumber
        )
    }

    private var sidedModifierBinding: Binding<SidedModifierFlags> {
        Binding(
            get: { SidedModifierFlags(rawValue: hotkeySidedModifiers).filtered(by: modifierBinding.wrappedValue) },
            set: { hotkeySidedModifiers = $0.filtered(by: modifierBinding.wrappedValue).rawValue }
        )
    }

    private var translationHotkeyBinding: Binding<UInt16> {
        Binding(
            get: { UInt16(translationHotkeyKeyCode) },
            set: {
                translationHotkeyKeyCode = Int($0)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var translationModifierBinding: Binding<NSEvent.ModifierFlags> {
        Binding(
            get: { NSEvent.ModifierFlags(rawValue: UInt(translationHotkeyModifiers)).intersection(.hotkeyRelevant) },
            set: {
                translationHotkeyModifiers = Int($0.rawValue)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var currentTranslationHotkey: HotkeyPreference.Hotkey {
        HotkeyPreference.Hotkey(
            input: translationHotkeyInput,
            modifiers: translationModifierBinding.wrappedValue,
            sidedModifiers: translationSidedModifierBinding.wrappedValue
        )
    }

    private var translationHotkeyInput: HotkeyPreference.Hotkey.Input {
        resolvedInput(
            inputType: translationHotkeyInputType,
            keyCode: translationHotkeyKeyCode,
            mouseButtonNumber: translationHotkeyMouseButtonNumber
        )
    }

    private var translationSidedModifierBinding: Binding<SidedModifierFlags> {
        Binding(
            get: { SidedModifierFlags(rawValue: translationHotkeySidedModifiers).filtered(by: translationModifierBinding.wrappedValue) },
            set: { translationHotkeySidedModifiers = $0.filtered(by: translationModifierBinding.wrappedValue).rawValue }
        )
    }

    private var meetingHotkeyBinding: Binding<UInt16> {
        Binding(
            get: { UInt16(meetingHotkeyKeyCode) },
            set: {
                meetingHotkeyKeyCode = Int($0)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var meetingModifierBinding: Binding<NSEvent.ModifierFlags> {
        Binding(
            get: { NSEvent.ModifierFlags(rawValue: UInt(meetingHotkeyModifiers)).intersection(.hotkeyRelevant) },
            set: {
                meetingHotkeyModifiers = Int($0.rawValue)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var currentMeetingHotkey: HotkeyPreference.Hotkey {
        HotkeyPreference.Hotkey(
            input: meetingHotkeyInput,
            modifiers: meetingModifierBinding.wrappedValue,
            sidedModifiers: meetingSidedModifierBinding.wrappedValue
        )
    }

    private var meetingHotkeyInput: HotkeyPreference.Hotkey.Input {
        resolvedInput(
            inputType: meetingHotkeyInputType,
            keyCode: meetingHotkeyKeyCode,
            mouseButtonNumber: meetingHotkeyMouseButtonNumber
        )
    }

    private var meetingSidedModifierBinding: Binding<SidedModifierFlags> {
        Binding(
            get: { SidedModifierFlags(rawValue: meetingHotkeySidedModifiers).filtered(by: meetingModifierBinding.wrappedValue) },
            set: { meetingHotkeySidedModifiers = $0.filtered(by: meetingModifierBinding.wrappedValue).rawValue }
        )
    }

    private var rewriteHotkeyBinding: Binding<UInt16> {
        Binding(
            get: { UInt16(rewriteHotkeyKeyCode) },
            set: {
                rewriteHotkeyKeyCode = Int($0)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var rewriteModifierBinding: Binding<NSEvent.ModifierFlags> {
        Binding(
            get: { NSEvent.ModifierFlags(rawValue: UInt(rewriteHotkeyModifiers)).intersection(.hotkeyRelevant) },
            set: {
                rewriteHotkeyModifiers = Int($0.rawValue)
                hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
            }
        )
    }

    private var currentRewriteHotkey: HotkeyPreference.Hotkey {
        HotkeyPreference.Hotkey(
            input: rewriteHotkeyInput,
            modifiers: rewriteModifierBinding.wrappedValue,
            sidedModifiers: rewriteSidedModifierBinding.wrappedValue
        )
    }

    private var rewriteHotkeyInput: HotkeyPreference.Hotkey.Input {
        resolvedInput(
            inputType: rewriteHotkeyInputType,
            keyCode: rewriteHotkeyKeyCode,
            mouseButtonNumber: rewriteHotkeyMouseButtonNumber
        )
    }

    private var rewriteSidedModifierBinding: Binding<SidedModifierFlags> {
        Binding(
            get: { SidedModifierFlags(rawValue: rewriteHotkeySidedModifiers).filtered(by: rewriteModifierBinding.wrappedValue) },
            set: { rewriteHotkeySidedModifiers = $0.filtered(by: rewriteModifierBinding.wrappedValue).rawValue }
        )
    }

    private var customPasteHotkeyBinding: Binding<UInt16> {
        Binding(
            get: { UInt16(customPasteHotkeyKeyCode) },
            set: { customPasteHotkeyKeyCode = Int($0) }
        )
    }

    private var customPasteModifierBinding: Binding<NSEvent.ModifierFlags> {
        Binding(
            get: { NSEvent.ModifierFlags(rawValue: UInt(customPasteHotkeyModifiers)).intersection(.hotkeyRelevant) },
            set: { customPasteHotkeyModifiers = Int($0.rawValue) }
        )
    }

    private var currentCustomPasteHotkey: HotkeyPreference.Hotkey {
        HotkeyPreference.Hotkey(
            input: customPasteHotkeyInput,
            modifiers: customPasteModifierBinding.wrappedValue,
            sidedModifiers: customPasteSidedModifierBinding.wrappedValue
        )
    }

    private var customPasteHotkeyInput: HotkeyPreference.Hotkey.Input {
        resolvedInput(
            inputType: customPasteHotkeyInputType,
            keyCode: customPasteHotkeyKeyCode,
            mouseButtonNumber: customPasteHotkeyMouseButtonNumber
        )
    }

    private var customPasteSidedModifierBinding: Binding<SidedModifierFlags> {
        Binding(
            get: { SidedModifierFlags(rawValue: customPasteHotkeySidedModifiers).filtered(by: customPasteModifierBinding.wrappedValue) },
            set: { customPasteHotkeySidedModifiers = $0.filtered(by: customPasteModifierBinding.wrappedValue).rawValue }
        )
    }

    private var isRecordingBinding: Binding<Bool> {
        Binding(
            get: { recordingField != nil },
            set: { isRecording in
                if !isRecording {
                    recordingField = nil
                }
            }
        )
    }

    private var triggerModeBinding: Binding<HotkeyPreference.TriggerMode> {
        Binding(
            get: {
                rewriteActivationState.enforcedTriggerMode(
                    from: HotkeyPreference.TriggerMode(rawValue: hotkeyTriggerMode) ?? HotkeyPreference.defaultTriggerMode
                )
            },
            set: {
                hotkeyTriggerMode = rewriteActivationState.enforcedTriggerMode(from: $0).rawValue
            }
        )
    }

    private var rewriteActivationState: HotkeyRewriteActivationState {
        HotkeyRewriteActivationState(rawValue: rewriteHotkeyActivationMode)
    }

    private var isRewriteDoubleTapWakeEnabled: Bool {
        rewriteActivationState.isDoubleTapWakeEnabled
    }

    private var rewriteDoubleTapDisplayText: String {
        rewriteActivationState.displayText(
            for: currentHotkey,
            distinguishModifierSides: distinguishModifierSides
        )
    }

    private var validationMessages: [HotkeySettingsValidation.Message] {
        let visibleKinds = Set(HotkeyShortcutVisibility.visibleKinds())
        return HotkeySettingsValidation.messages(
            for: .init(
                transcriptionBindings: transcriptionBindings,
                translationBindings: visibleKinds.contains(.translation) ? translationBindings : [],
                meetingBindings: visibleKinds.contains(.meeting) ? meetingBindings : [],
                rewriteBindings: visibleKinds.contains(.rewrite) ? rewriteBindings : [],
                customPasteHotkey: customPasteHotkeyEnabled ? currentCustomPasteHotkey : nil,
                noteBindings: visibleKinds.contains(.note) ? noteBindings : []
            )
        )
    }

    private var presetBinding: Binding<HotkeyPreference.Preset> {
        Binding(
            get: { HotkeyPreference.Preset(rawValue: hotkeyPreset) ?? .custom },
            set: { applyPreset($0) }
        )
    }

    private func resolvedInput(
        inputType: String,
        keyCode: Int,
        mouseButtonNumber: Int
    ) -> HotkeyPreference.Hotkey.Input {
        if HotkeyPreference.Hotkey.Input.Kind(rawValue: inputType) == .mouseButton {
            return .mouseButton(mouseButtonNumber)
        }
        return .keyboard(UInt16(keyCode))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(alignment: .center, spacing: 12) {
                        Text(localized("Preset"))
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.primary.opacity(0.92))
                        Spacer()
                        SettingsMenuPicker(
                            selection: presetBinding,
                            options: HotkeyPreference.Preset.allCases.map { preset in
                                SettingsMenuOption(value: preset, title: preset.title)
                            },
                            selectedTitle: presetBinding.wrappedValue.title,
                            width: 220
                        )
                    }

                    let _ = featureSettingsRaw
                    ForEach(HotkeyShortcutVisibility.visibleKinds(), id: \.self) { kind in
                        switch kind {
                        case .transcription:
                            hotkeyBindingGroup(.transcription, bindings: transcriptionBindings)
                        case .note:
                            hotkeyBindingGroup(.note, bindings: noteBindings)
                        case .translation:
                            hotkeyBindingGroup(.translation, bindings: translationBindings)
                        case .rewrite:
                            hotkeyBindingGroup(.rewrite, bindings: rewriteBindings)
                        case .meeting:
                            hotkeyBindingGroup(.meeting, bindings: meetingBindings)
                        }
                    }

                    ForEach(validationMessages) { message in
                        Text(message.text)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }

                    HotkeyRecorderView(
                        isRecording: isRecordingBinding,
                        onCapture: { capturedHotkey in
                            guard let field = recordingField else { return }
                            guard HotkeyPreference.isAllowedGlobalShortcut(capturedHotkey) else {
                                showHotkeyToast(localized("Keyboard shortcuts must include at least one modifier key."))
                                return
                            }
                            pendingCapturedField = field
                            pendingCapturedHotkey = capturedHotkey
                            showHotkeyToast(localized("Shortcut captured. Press another shortcut to replace it, or choose Confirm / Cancel."))
                        },
                        onCancelCapture: {
                            discardPendingCapture()
                            recordingField = nil
                        },
                        onRecorderMessageChange: { messageKey in
                            guard recorderMessageKey != messageKey else { return }
                            DispatchQueue.main.async {
                                recorderMessageKey = messageKey
                                if let messageKey {
                                    showHotkeyToast(localized(messageKey))
                                }
                            }
                        }
                    )
                    .frame(width: 0, height: 0)

                    if customPasteHotkeyEnabled {
                        SettingsShortcutCaptureField(
                            title: "Custom Paste",
                            hotkey: displayedHotkey(for: .customPaste, current: currentCustomPasteHotkey),
                            isRecording: recordingField == .customPaste,
                            isPendingConfirmation: isPendingConfirmation(for: .customPaste),
                            distinguishModifierSides: true,
                            onFocus: { beginRecording(.customPaste) },
                            onReset: {
                                customPasteHotkeyInputType = HotkeyPreference.Hotkey.Input.Kind.keyboard.rawValue
                                customPasteHotkeyBinding.wrappedValue = HotkeyPreference.defaultCustomPasteKeyCode
                                customPasteModifierBinding.wrappedValue = HotkeyPreference.defaultCustomPasteModifiers
                                customPasteSidedModifierBinding.wrappedValue = []
                            },
                            onCancelPending: discardPendingCapture,
                            onConfirmPending: confirmPendingCapture
                        )
                    }

                    GeneralSectionDivider()
                        .padding(.top, 2)

                    HStack(alignment: .center, spacing: 18) {
                        Text(localized("Use Esc to Cancel"))
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.primary.opacity(0.92))
                        Spacer()
                        Toggle("", isOn: $escapeKeyCancelsOverlaySession)
                            .labelsHidden()
                            .toggleStyle(.switch)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

                    VoiceEndCommandSettingsSection()
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .id(interfaceLanguageRaw)
        .overlay(alignment: .top) {
            if !hotkeyToastMessage.isEmpty {
                ModelDebugToast(message: hotkeyToastMessage) {
                    dismissHotkeyToast()
                }
                .padding(.top, 8)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: hotkeyToastMessage)
        .onChange(of: customPasteHotkeyEnabled) { _, enabled in
            guard !enabled else { return }
            if recordingField == .customPaste || pendingCapturedField == .customPaste {
                discardPendingCapture()
            }
        }
    }

    private func applyPreset(_ preset: HotkeyPreference.Preset) {
        discardPendingCapture()
        hotkeyPreset = preset.rawValue
        guard HotkeyPreference.applyPreset(preset) != nil else { return }
        distinguishModifierSides = true
        transcriptionBindings = HotkeyPreference.loadTranscriptionBindings()
        noteBindings = HotkeyPreference.loadNoteBindings()
        translationBindings = HotkeyPreference.loadTranslationBindings()
        meetingBindings = HotkeyPreference.loadMeetingBindings()
        rewriteBindings = HotkeyPreference.loadRewriteBindings()
    }

    @ViewBuilder
    private func hotkeyBindingGroup(
        _ kind: HotkeyShortcutKind,
        bindings: [HotkeyPreference.HotkeyBinding]
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(bindings.enumerated()), id: \.element.id) { index, binding in
                let field = RecordingField.binding(kind, binding.id)
                HStack(alignment: .center, spacing: 8) {
                    SettingsShortcutCaptureField(
                        title: index == 0 ? kind.titleKey : LocalizedStringKey(""),
                        hotkey: displayedHotkey(for: field, current: binding.hotkey),
                        isRecording: recordingField == field,
                        isPendingConfirmation: isPendingConfirmation(for: field),
                        distinguishModifierSides: true,
                        controlWidth: 200,
                        onFocus: { beginRecording(field) },
                        onReset: { resetBinding(kind: kind, id: binding.id) },
                        onCancelPending: discardPendingCapture,
                        onConfirmPending: confirmPendingCapture
                    )

                    if kind == .note {
                        SettingsFixedSelectionBlock(
                            title: localized("Tap"),
                            width: 70,
                            usesCompactInsets: true
                        )
                    } else {
                        SettingsMenuPicker(
                            selection: behaviorBinding(kind: kind, id: binding.id),
                            options: HotkeyPreference.TriggerBehavior.allCases.map { behavior in
                                SettingsMenuOption(value: behavior, title: behavior.title)
                            },
                            selectedTitle: binding.behavior.title,
                            width: 70,
                            allowsCompactWidth: true,
                            usesCompactInsets: true
                        )
                    }

                    if index == 0 {
                        Button {
                            addBinding(kind)
                        } label: {
                            Image(systemName: "plus")
                                .frame(width: 14, height: 14)
                        }
                        .buttonStyle(SettingsCompactIconButtonStyle(size: 34))
                        .help(localized("Add"))
                    } else {
                        Button {
                            removeBinding(kind: kind, id: binding.id)
                        } label: {
                            Image(systemName: "trash")
                                .frame(width: 14, height: 14)
                        }
                        .buttonStyle(SettingsCompactIconButtonStyle(tone: .destructive, size: 34))
                        .help(localized("Delete"))
                    }
                }
            }
        }
    }

    private func beginRecording(_ field: RecordingField) {
        pendingCapturedField = nil
        pendingCapturedHotkey = nil
        recordingField = field
        showHotkeyToast(localized("Type your shortcut now. Press Esc to cancel recording."))
    }

    private func isPendingConfirmation(for field: RecordingField) -> Bool {
        pendingCapturedField == field && pendingCapturedHotkey != nil
    }

    private func displayedHotkey(for field: RecordingField, current: HotkeyPreference.Hotkey) -> HotkeyPreference.Hotkey {
        guard pendingCapturedField == field, let pendingCapturedHotkey else {
            return current
        }
        return pendingCapturedHotkey
    }

    private func discardPendingCapture() {
        recorderMessageKey = nil
        pendingCapturedField = nil
        pendingCapturedHotkey = nil
        recordingField = nil
        dismissHotkeyToast()
    }

    private func confirmPendingCapture() {
        guard let field = pendingCapturedField, let hotkey = pendingCapturedHotkey else { return }
        guard HotkeyPreference.isAllowedGlobalShortcut(hotkey) else {
            discardPendingCapture()
            showHotkeyToast(localized("Keyboard shortcuts must include at least one modifier key."))
            return
        }

        switch field {
        case .binding(let kind, let id):
            updateBinding(kind: kind, id: id) { binding in
                binding.hotkey = hotkey
            }
        case .customPaste:
            assign(hotkey.input, inputType: &customPasteHotkeyInputType, keyCode: &customPasteHotkeyKeyCode, mouseButtonNumber: &customPasteHotkeyMouseButtonNumber)
            customPasteModifierBinding.wrappedValue = hotkey.modifiers
            customPasteSidedModifierBinding.wrappedValue =
                hotkey.keyCode == HotkeyPreference.modifierOnlyKeyCode || hotkey.isMouseButton ? hotkey.sidedModifiers : []
        }

        hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
        pendingCapturedField = nil
        pendingCapturedHotkey = nil
        recordingField = nil
        dismissHotkeyToast()
    }

    private func showHotkeyToast(_ message: String, duration: TimeInterval = 2.2) {
        hotkeyToastDismissTask?.cancel()
        hotkeyToastMessage = message
        hotkeyToastDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(duration))
            guard !Task.isCancelled else { return }
            hotkeyToastMessage = ""
        }
    }

    private func dismissHotkeyToast() {
        hotkeyToastDismissTask?.cancel()
        hotkeyToastMessage = ""
    }

    private func assign(
        _ input: HotkeyPreference.Hotkey.Input,
        inputType: inout String,
        keyCode: inout Int,
        mouseButtonNumber: inout Int
    ) {
        inputType = input.kind.rawValue
        switch input {
        case .keyboard(let value):
            keyCode = Int(value)
        case .mouseButton(let value):
            mouseButtonNumber = value
        }
    }

    private func behaviorBinding(
        kind: HotkeyShortcutKind,
        id: UUID
    ) -> Binding<HotkeyPreference.TriggerBehavior> {
        Binding(
            get: {
                bindings(for: kind).first { $0.id == id }?.behavior ?? .tap
            },
            set: { behavior in
                updateBinding(kind: kind, id: id) { binding in
                    binding.behavior = behavior
                }
            }
        )
    }

    private func addBinding(_ kind: HotkeyShortcutKind) {
        var next = bindings(for: kind)
        next.append(.init(hotkey: kind.defaultHotkey, behavior: .tap))
        setBindings(next, for: kind)
        hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
    }

    private func removeBinding(kind: HotkeyShortcutKind, id: UUID) {
        let current = bindings(for: kind)
        guard current.count > 1 else { return }
        setBindings(current.filter { $0.id != id }, for: kind)
        if recordingField == .binding(kind, id) || pendingCapturedField == .binding(kind, id) {
            discardPendingCapture()
        }
        hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
    }

    private func resetBinding(kind: HotkeyShortcutKind, id: UUID) {
        updateBinding(kind: kind, id: id) { binding in
            binding.hotkey = kind.defaultHotkey
            binding.behavior = .tap
        }
        hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
    }

    private func updateBinding(
        kind: HotkeyShortcutKind,
        id: UUID,
        mutate: (inout HotkeyPreference.HotkeyBinding) -> Void
    ) {
        var next = bindings(for: kind)
        guard let index = next.firstIndex(where: { $0.id == id }) else { return }
        mutate(&next[index])
        setBindings(next, for: kind)
        hotkeyPreset = HotkeyPreference.Preset.custom.rawValue
    }

    private func bindings(for kind: HotkeyShortcutKind) -> [HotkeyPreference.HotkeyBinding] {
        switch kind {
        case .transcription:
            return transcriptionBindings
        case .note:
            return noteBindings
        case .translation:
            return translationBindings
        case .meeting:
            return meetingBindings
        case .rewrite:
            return rewriteBindings
        }
    }

    private func setBindings(
        _ bindings: [HotkeyPreference.HotkeyBinding],
        for kind: HotkeyShortcutKind
    ) {
        switch kind {
        case .transcription:
            transcriptionBindings = bindings
            HotkeyPreference.saveTranscriptionBindings(bindings)
        case .note:
            let normalized = bindings.map {
                HotkeyPreference.HotkeyBinding(id: $0.id, hotkey: $0.hotkey, behavior: .tap)
            }
            noteBindings = normalized.isEmpty
                ? [.init(hotkey: kind.defaultHotkey, behavior: .tap)]
                : normalized
            HotkeyPreference.saveNoteBindings(noteBindings)
        case .translation:
            translationBindings = bindings
            HotkeyPreference.saveTranslationBindings(bindings)
        case .meeting:
            meetingBindings = bindings
            HotkeyPreference.saveMeetingBindings(bindings)
        case .rewrite:
            rewriteBindings = bindings
            HotkeyPreference.saveRewriteBindings(bindings)
        }
    }
}
