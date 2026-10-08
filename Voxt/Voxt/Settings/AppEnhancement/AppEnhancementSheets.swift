// AppEnhancementSheets.swift
// Provides App Enhancement Sheets for app enhancement settings.

import SwiftUI
import AppKit

struct GroupEditorSheet: View {
    let title: String
    let actionTitle: String
    @Binding var name: String
    @Binding var prompt: String
    @Binding var autoKeyPressEnabled: Bool
    @Binding var autoKeyPressHotkey: HotkeyPreference.Hotkey
    let errorMessage: String?
    let onCancel: () -> Void
    let onSave: () -> Void
    @State private var selectedPresetID = Self.placeholderPresetID
    @State private var isAutoKeyRecording = false
    @State private var pendingAutoKeyHotkey: HotkeyPreference.Hotkey?

    private static let placeholderPresetID = "choose-template"
    private let dialogPadding: CGFloat = 0

    private struct PromptPreset: Identifiable {
        let id: String
        let title: String
        let prompt: String
    }

    private var presets: [PromptPreset] {
        [
            PromptPreset(
                id: "slack",
                title: AppLocalization.localizedString("Slack / Chat"),
                prompt: AppLocalization.localizedString("Slack / Chat prompt preset")
            ),
            PromptPreset(
                id: "email",
                title: AppLocalization.localizedString("Email"),
                prompt: AppLocalization.localizedString("Email prompt preset")
            ),
            PromptPreset(
                id: "ide",
                title: AppLocalization.localizedString("IDE / Terminal"),
                prompt: AppLocalization.localizedString("IDE / Terminal prompt preset")
            ),
            PromptPreset(
                id: "docs",
                title: AppLocalization.localizedString("Docs / Notes"),
                prompt: AppLocalization.localizedString("Docs / Notes prompt preset")
            )
        ]
    }

    private var presetOptions: [SettingsMenuOption<String>] {
        [
            SettingsMenuOption(
                value: Self.placeholderPresetID,
                title: AppLocalization.localizedString("Choose template...")
            )
        ] + presets.map { preset in
            SettingsMenuOption(value: preset.id, title: preset.title)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title)
                .font(.title3.weight(.semibold))
                .padding(.horizontal, dialogPadding)
                .padding(.top, dialogPadding)
                .padding(.bottom, 10)

            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(AppLocalization.localizedString("Name"))
                        .font(.headline)
                    TextField(AppLocalization.localizedString("Enter group name"), text: $name)
                        .textFieldStyle(.plain)
                        .settingsFieldSurface(minHeight: 34)
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text(AppLocalization.localizedString("Prompt"))
                        .font(.headline)

                    VStack(alignment: .leading, spacing: 8) {
                        Text(AppLocalization.localizedString("Starter templates"))
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)

                        SettingsMenuPicker(
                            selection: $selectedPresetID,
                            options: presetOptions,
                            selectedTitle: AppLocalization.localizedString("Choose template..."),
                            width: 220
                        )
                        .onChange(of: selectedPresetID) { _, newValue in
                            applyPreset(id: newValue)
                        }
                    }

                    PromptEditorView(
                        text: $prompt,
                        height: 160,
                        contentPadding: 8,
                        variables: ModelSettingsPromptVariables.appEnhancement,
                        variablesLayout: .twoColumns,
                        variablesTitle: PromptAuthoringGuidance.optionalVariablesTitle
                    )
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .center, spacing: 12) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(AppLocalization.localizedString("Auto Key"))
                                .font(.headline)

                            Text(AppLocalization.localizedString("Automatically press this key after transcription ends."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Spacer(minLength: 0)

                        if autoKeyPressEnabled {
                            SettingsShortcutCaptureField(
                                title: "",
                                hotkey: pendingAutoKeyHotkey ?? autoKeyPressHotkey,
                                isRecording: isAutoKeyRecording,
                                isPendingConfirmation: pendingAutoKeyHotkey != nil,
                                distinguishModifierSides: false,
                                showsTitle: false,
                                controlWidth: 240,
                                onFocus: {
                                    pendingAutoKeyHotkey = nil
                                    isAutoKeyRecording = true
                                },
                                onReset: {
                                    autoKeyPressHotkey = AppBranchGroup.defaultAutoKeyPressHotkey
                                    pendingAutoKeyHotkey = nil
                                    isAutoKeyRecording = false
                                },
                                onCancelPending: {
                                    pendingAutoKeyHotkey = nil
                                    isAutoKeyRecording = false
                                },
                                onConfirmPending: {
                                    if let pendingAutoKeyHotkey {
                                        autoKeyPressHotkey = pendingAutoKeyHotkey
                                    }
                                    pendingAutoKeyHotkey = nil
                                    isAutoKeyRecording = false
                                }
                            )
                        }

                        Toggle("", isOn: $autoKeyPressEnabled)
                            .labelsHidden()
                            .toggleStyle(.switch)
                            .onChange(of: autoKeyPressEnabled) { _, enabled in
                                if enabled, autoKeyPressHotkey.keyCode == HotkeyPreference.modifierOnlyKeyCode {
                                    autoKeyPressHotkey = AppBranchGroup.defaultAutoKeyPressHotkey
                                }
                                if !enabled {
                                    pendingAutoKeyHotkey = nil
                                    isAutoKeyRecording = false
                                }
                            }
                    }

                    HotkeyRecorderView(
                        isRecording: $isAutoKeyRecording,
                        onCapture: { capturedHotkey in
                            pendingAutoKeyHotkey = capturedHotkey
                        },
                        onCancelCapture: {
                            pendingAutoKeyHotkey = nil
                            isAutoKeyRecording = false
                        },
                        onRecorderMessageChange: { _ in }
                    )
                    .frame(width: 0, height: 0)
                }

                if let errorMessage, !errorMessage.isEmpty {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, dialogPadding)
            .padding(.bottom, 8)

            Divider()

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Cancel"), action: onCancel)
                    .buttonStyle(SettingsPillButtonStyle())
                    .keyboardShortcut(.cancelAction)

                Button(actionTitle, action: onSave)
                    .buttonStyle(SettingsPrimaryButtonStyle())
                    .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, dialogPadding)
            .padding(.top, 8)
            .padding(.bottom, dialogPadding)
        }
        .settingsDialogChrome(onClose: onCancel)
    }

    private func applyPreset(id: String) {
        guard
            id != Self.placeholderPresetID,
            let preset = presets.first(where: { $0.id == id })
        else {
            return
        }

        prompt = preset.prompt
        DispatchQueue.main.async {
            selectedPresetID = Self.placeholderPresetID
        }
    }
}

struct URLBatchEditorSheet: View {
    let title: String
    let actionTitle: String
    @Binding var text: String
    let errorMessage: String?
    let onCancel: () -> Void
    let onSave: () -> Void
    @State private var testInput = ""
    @State private var isTestInputHovered = false
    @State private var testFaviconOrigin: String?
    @State private var testFaviconImage: NSImage?
    @State private var isLoadingTestFavicon = false
    @State private var testFaviconLookupID = UUID()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.title3.weight(.semibold))

            VStack(alignment: .leading, spacing: 6) {
                Text(AppLocalization.localizedString("URL Patterns"))
                    .font(.headline)
                PromptEditorView(text: $text, height: 180, contentPadding: 8)

                Text(AppLocalization.localizedString("One wildcard pattern per line."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 6) {
                    Text(AppLocalization.localizedString("Pattern Test"))
                        .font(.headline)

                    TextField(AppLocalization.localizedString("Paste a URL to test"), text: $testInput)
                        .textFieldStyle(.plain)
                        .padding(.horizontal, 10)
                        .frame(minHeight: 34)
                        .background(
                            RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                                .fill(testFieldFillColor)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                                .strokeBorder(testFieldBorderColor, lineWidth: 1)
                        )
                        .contentShape(RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous))
                        .onHover { isTestInputHovered = $0 }

                    HStack(spacing: 6) {
                        testFeedbackIcon

                        Text(testFeedbackText)
                            .font(.caption)
                            .foregroundStyle(testFeedbackColor)
                    }
                    .onChange(of: testInput) { _, _ in
                        refreshTestFavicon()
                    }
                    .onChange(of: text) { _, _ in
                        refreshTestFavicon()
                    }
                }
            }

            if let errorMessage, !errorMessage.isEmpty {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Spacer(minLength: 6)

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Cancel"), action: onCancel)
                    .buttonStyle(SettingsPillButtonStyle())
                    .keyboardShortcut(.cancelAction)

                Button(actionTitle, action: onSave)
                    .buttonStyle(SettingsPrimaryButtonStyle())
                    .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(onClose: onCancel)
    }

    private var normalizedPatterns: [String] {
        text
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map(AppBranchURLPatternService.normalizedPattern)
            .filter(AppBranchURLPatternService.isValidWildcardURLPattern)
    }

    private var normalizedCandidate: String? {
        AppBranchURLPatternService.normalizedURLForMatching(testInput)
    }

    private var matchedPattern: String? {
        guard let normalizedCandidate else { return nil }
        return normalizedPatterns.first {
            AppBranchURLPatternService.wildcardMatches(pattern: $0, candidate: normalizedCandidate)
        }
    }

    private var testFieldFillColor: Color {
        switch testStatus {
        case .idle:
            return SettingsUIStyle.controlFillColor
        case .matched:
            return Color.green.opacity(0.12)
        case .unmatched:
            return Color.red.opacity(0.10)
        }
    }

    private var testFieldBorderColor: Color {
        switch testStatus {
        case .idle:
            return isTestInputHovered ? SettingsUIStyle.controlHoverBorderColor : SettingsUIStyle.subtleBorderColor
        case .matched:
            return Color.green.opacity(0.7)
        case .unmatched:
            return Color.red.opacity(0.7)
        }
    }

    private var testFeedbackColor: Color {
        switch testStatus {
        case .idle:
            return .secondary
        case .matched:
            return Color.green
        case .unmatched:
            return Color.red
        }
    }

    private var testFeedbackText: String {
        if testInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return AppLocalization.localizedString("Enter a URL or text to verify whether it matches the patterns above.")
        }
        if normalizedPatterns.isEmpty {
            return AppLocalization.localizedString("Add at least one valid wildcard pattern above to test matching.")
        }
        if let matchedPattern {
            return AppLocalization.format("Matched pattern: %@", matchedPattern)
        }
        return AppLocalization.localizedString("No pattern matched.")
    }

    @ViewBuilder
    private var testFeedbackIcon: some View {
        switch testStatus {
        case .matched:
            if let testFaviconImage {
                Image(nsImage: testFaviconImage)
                    .resizable()
                    .frame(width: 16, height: 16)
                    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
            } else if isLoadingTestFavicon {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.55)
                    .frame(width: 16, height: 16)
            } else {
                Image(systemName: "globe")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(testFeedbackColor)
                    .frame(width: 16, height: 16)
            }
        case .unmatched:
            Image(systemName: "xmark.circle")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(testFeedbackColor)
                .frame(width: 16, height: 16)
        case .idle:
            Image(systemName: "globe")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(testFeedbackColor)
                .frame(width: 16, height: 16)
        }
    }

    private var testStatus: URLPatternTestStatus {
        guard !testInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .idle }
        guard !normalizedPatterns.isEmpty, normalizedCandidate != nil else { return .idle }
        return matchedPattern == nil ? .unmatched : .matched
    }

    private var testFaviconOriginCandidate: String? {
        guard testStatus == .matched else { return nil }
        let trimmed = testInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let urlString = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
        return EnhancementOverlayIconResolver.faviconOrigin(fromPageURL: urlString)
    }

    private func refreshTestFavicon() {
        guard let origin = testFaviconOriginCandidate else {
            testFaviconOrigin = nil
            testFaviconImage = nil
            isLoadingTestFavicon = false
            testFaviconLookupID = UUID()
            return
        }

        if origin == testFaviconOrigin, testFaviconImage != nil {
            return
        }

        testFaviconOrigin = origin
        testFaviconLookupID = UUID()
        let lookupID = testFaviconLookupID

        if let cached = EnhancementOverlayIconResolver.cachedFavicon(forOrigin: origin) {
            testFaviconImage = cached
            isLoadingTestFavicon = false
            return
        }

        testFaviconImage = nil
        isLoadingTestFavicon = true

        Task {
            let image = await EnhancementOverlayIconResolver.favicon(forOrigin: origin)
            await MainActor.run {
                guard testFaviconLookupID == lookupID else { return }
                testFaviconImage = image
                isLoadingTestFavicon = false
            }
        }
    }
}

private enum URLPatternTestStatus {
    case idle
    case matched
    case unmatched
}

struct URLDetailSheet: View {
    let pattern: String
    let onClose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(AppLocalization.localizedString("URL Detail"))
                .font(.title3.weight(.semibold))

            Text(pattern)
                .font(.system(size: 13))
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
                .background(
                    RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                        .fill(SettingsUIStyle.controlFillColor)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                        .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                )

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Close"), action: onClose)
                    .buttonStyle(SettingsPrimaryButtonStyle())
                    .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(onClose: onClose)
    }
}
