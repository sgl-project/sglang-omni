// GeneralSettingsSections.swift
// Provides General Settings Sections for settings screens.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

private func localizedKey(_ key: String) -> LocalizedStringKey {
    LocalizedStringKey(localized(key))
}

struct GeneralAudioCard: View {
    let microphoneState: MicrophoneResolvedState
    @Binding var interactionSoundsEnabled: Bool
    @Binding var muteSystemAudioWhileRecording: Bool
    @Binding var interactionSoundPreset: InteractionSoundPreset
    let onTrySound: () -> TimeInterval
    let onManageMicrophones: () -> Void
    let onViewPriorityList: () -> Void

    var body: some View {
        GeneralSettingsCard(title: localizedKey("Audio")) {
            GeneralFieldRow(
                title: localizedKey("Microphone"),
                description: localizedKey("Reorder microphones to control device priority. Auto Switch only applies when devices connect or disconnect.")
            ) {
                if microphoneState.hasAvailableDevices {
                    SettingsSelectionButton(width: 272, action: onManageMicrophones) {
                        HStack(spacing: 0) {
                            Text(microphoneState.activeDevice?.name ?? localized("No available microphone devices"))
                                .lineLimit(1)
                        }
                    }
                } else {
                    Text(localized("No available microphone devices"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.red)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule(style: .continuous)
                                .fill(Color.red.opacity(0.10))
                        )
                }
            }

            if !microphoneState.hasAvailableDevices, microphoneState.hasTrackedDevices {
                HStack {
                    Spacer()
                    Button(localized("View Priority List"), action: onViewPriorityList)
                        .buttonStyle(SettingsPillButtonStyle())
                }
            }

            GeneralInteractionSoundsRow(
                interactionSoundsEnabled: $interactionSoundsEnabled,
                interactionSoundPreset: $interactionSoundPreset,
                onTrySound: onTrySound
            )

            GeneralToggleRow(
                title: localizedKey("Mute output device while recording"),
                description: localizedKey("Mutes all sounds on the current output device, including Voxt. Some devices do not support software mute. No system audio recording permission is needed."),
                isOn: $muteSystemAudioWhileRecording
            )


        }
    }
}

private struct GeneralInteractionSoundsRow: View {
    @Binding var interactionSoundsEnabled: Bool
    @Binding var interactionSoundPreset: InteractionSoundPreset
    let onTrySound: () -> TimeInterval

    @State private var isPreviewPlaying = false
    @State private var previewToken = UUID()

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text(localizedKey("Interaction Sounds"))
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.primary.opacity(0.92))

                Text(localizedKey("Play start and completion sounds."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 18)

            HStack(alignment: .center, spacing: 13) {
                if interactionSoundsEnabled {
                    HStack(spacing: 5) {
                        SettingsMenuPicker(
                            selection: $interactionSoundPreset,
                            options: InteractionSoundPreset.allCases.map { preset in
                                SettingsMenuOption(value: preset, title: preset.title)
                            },
                            selectedTitle: interactionSoundPreset.title,
                            width: 150,
                            allowsCompactWidth: true
                        )

                        Button(action: playPreview) {
                            SettingsSoundPreviewIcon()
                                .foregroundStyle(isPreviewPlaying ? Color.accentColor : Color.primary.opacity(0.86))
                                .frame(width: 16, height: 16)
                        }
                        .buttonStyle(SettingsCompactIconButtonStyle(size: 34))
                        .help(localized("Try Sound"))
                    }
                }

                Toggle("", isOn: $interactionSoundsEnabled)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }

    private func playPreview() {
        let duration = max(onTrySound(), 0.65)
        let token = UUID()
        previewToken = token
        isPreviewPlaying = true

        Task {
            try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
            await MainActor.run {
                guard previewToken == token else { return }
                isPreviewPlaying = false
            }
        }
    }
}



struct GeneralTranscriptionUICard: View {
    @Binding var overlayPosition: OverlayPosition
    @Binding var overlayCardOpacity: Int
    @Binding var overlayCardCornerRadius: Int
    @Binding var overlayScreenEdgeInset: Int

    var body: some View {
        GeneralSettingsCard(title: localizedKey("Floating Window Style")) {
            GeneralOverlayStylePreviewCard(
                opacity: overlayCardOpacity,
                cornerRadius: overlayCardCornerRadius
            )

            overlayFieldRow {
                overlayNumberField(
                    title: localizedKey("Opacity"),
                    value: $overlayCardOpacity,
                    range: 0...100,
                    width: 90,
                    unit: "%"
                )
            } right: {
                overlayNumberField(
                    title: localizedKey("Corner Radius"),
                    value: $overlayCardCornerRadius,
                    range: 0...40,
                    width: 90,
                    unit: "pt"
                )
            }

            overlayFieldRow {
                overlayPositionField
            } right: {
                overlayNumberField(
                    title: localizedKey("Edge Distance"),
                    value: $overlayScreenEdgeInset,
                    range: 0...120,
                    width: 90,
                    unit: "pt"
                )
            }
        }
    }

    private func overlayFieldRow<Left: View, Right: View>(
        @ViewBuilder left: @escaping () -> Left,
        @ViewBuilder right: @escaping () -> Right
    ) -> some View {
        GeometryReader { proxy in
            let columnSpacing: CGFloat = 34
            let columnWidth = max((proxy.size.width - columnSpacing) / 2, 0)

            HStack(alignment: .center, spacing: columnSpacing) {
                left()
                    .frame(width: columnWidth, alignment: .leading)
                right()
                    .frame(width: columnWidth, alignment: .leading)
            }
        }
        .frame(height: 34)
    }

    private var overlayPositionField: some View {
        GeneralFieldRow(title: localizedKey("Position")) {
            SettingsMenuPicker(
                selection: $overlayPosition,
                options: OverlayPosition.allCases.map { position in
                    SettingsMenuOption(value: position, title: position.title)
                },
                selectedTitle: overlayPosition.title,
                width: 110,
                allowsCompactWidth: true,
                usesCompactInsets: true
            )
        }
        .frame(maxWidth: .infinity)
    }

    private func overlayNumberField(
        title: LocalizedStringKey,
        value: Binding<Int>,
        range: ClosedRange<Int>,
        width: CGFloat,
        unit: String
    ) -> some View {
        GeneralFieldRow(title: title) {
            ClampedIntegerTextField(
                value: value,
                range: range,
                width: width,
                unit: unit
            )
        }
        .frame(maxWidth: .infinity)
    }
}

private struct GeneralOverlayStylePreviewCard: View {
    let opacity: Int
    let cornerRadius: Int

    private var clampedOpacity: Double {
        Double(min(max(opacity, 0), 100)) / 100.0
    }

    private var clampedCornerRadius: CGFloat {
        CGFloat(min(max(cornerRadius, 0), 40))
    }

    var body: some View {
        ZStack {
            Image("OverlayPreviewBackground")
                .resizable()
                .scaledToFill()

            overlayPreview
        }
        .frame(maxWidth: .infinity)
        .frame(height: 150)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var overlayPreview: some View {
        HStack(spacing: 10) {
            WaveformCompactLeadingStatusIconView(
                isCompleting: false,
                showsInitializationIcon: false,
                shouldAnimate: false,
                compactLeadingIconImage: nil,
                sessionIconMode: .transcription,
                displayMode: .recording
            )
            .frame(width: 16, height: 28, alignment: .center)

            HStack(alignment: .center, spacing: 2.5) {
                ForEach(0..<16, id: \.self) { index in
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(
                            LinearGradient(
                                colors: [Color.white.opacity(0.98), Color.white.opacity(0.80)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .frame(width: 3.2, height: barHeight(at: index))
                        .shadow(color: .white.opacity(0.08), radius: 3, x: 0, y: 0)
                }
            }
            .frame(width: 94, height: 28, alignment: .center)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: clampedCornerRadius, style: .continuous)
                .fill(.black.opacity(clampedOpacity))
                .overlay(
                    RoundedRectangle(cornerRadius: clampedCornerRadius, style: .continuous)
                        .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.20), radius: 18, x: 0, y: 10)
        )
    }

    private func barHeight(at index: Int) -> CGFloat {
        let heights: [CGFloat] = [5, 7, 10, 12, 9, 6, 8, 11]
        return heights[index % heights.count]
    }
}

private struct ClampedIntegerTextField: View {
    @Binding var value: Int
    let range: ClosedRange<Int>
    let width: CGFloat
    let unit: String

    @State private var text: String

    init(value: Binding<Int>, range: ClosedRange<Int>, width: CGFloat, unit: String) {
        _value = value
        self.range = range
        self.width = width
        self.unit = unit
        _text = State(initialValue: String(min(max(value.wrappedValue, range.lowerBound), range.upperBound)))
    }

    var body: some View {
        ZStack(alignment: .trailing) {
            TextField("", text: $text)
                .textFieldStyle(.plain)
                .padding(.trailing, unitWidth + 6)
                .settingsFieldSurface(width: width, alignment: .trailing)
                .multilineTextAlignment(.trailing)
                .onChange(of: text) { _, newValue in
                    let digits = newValue.filter(\.isNumber)
                    guard !digits.isEmpty else {
                        return
                    }

                    let parsed = Int(digits) ?? range.lowerBound
                    let clamped = min(max(parsed, range.lowerBound), range.upperBound)
                    if value != clamped {
                        value = clamped
                    }
                }
                .onSubmit {
                    syncTextToValue()
                }
                .onChange(of: value) { _, newValue in
                    let clamped = min(max(newValue, range.lowerBound), range.upperBound)
                    let normalized = String(clamped)
                    if text != normalized {
                        text = normalized
                    }
                }
                .onAppear {
                    syncTextToValue()
                }

            Text(unit)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .frame(width: unitWidth, alignment: .trailing)
                .padding(.trailing, 10)
                .allowsHitTesting(false)
        }
        .frame(width: visualWidth)
    }

    private func syncTextToValue() {
        let digits = text.filter(\.isNumber)
        let parsed = Int(digits) ?? value
        let clamped = min(max(parsed, range.lowerBound), range.upperBound)
        value = clamped
        text = String(clamped)
    }

    private var unitWidth: CGFloat {
        unit == "%" ? 12 : 16
    }

    private var visualWidth: CGFloat {
        width + 20
    }
}

struct GeneralLanguagesCard: View {
    @Binding var interfaceLanguage: AppInterfaceLanguage
    let userMainLanguageSummary: String
    let onEditUserMainLanguage: () -> Void

    var body: some View {
        GeneralSettingsCard(title: localizedKey("Languages"), spacing: 14) {
            GeneralLanguageSettingBlock(
                title: localizedKey("Interface Language"),
                description: nil
            ) {
                SettingsMenuPicker(
                    selection: $interfaceLanguage,
                    options: AppInterfaceLanguage.allCases.map { language in
                        SettingsMenuOption(value: language, title: language.title)
                    },
                    selectedTitle: interfaceLanguage.title,
                    width: 220
                )
            }

            GeneralLanguageSettingBlock(
                title: localizedKey("Speech Recognition Language"),
                description: localizedKey("Used to guide transcription, punctuation, and language-specific cleanup.")
            ) {
                SettingsSelectionButton(width: 220, action: onEditUserMainLanguage) {
                    HStack(spacing: 0) {
                        Text(userMainLanguageSummary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
            }
        }
    }
}

struct GeneralAppBehaviorCard: View {
    @Binding var autoCopyWhenNoFocusedInput: Bool
    @Binding var realtimeTextDisplayEnabled: Bool
    @Binding var customPasteHotkeyEnabled: Bool
    let customPasteHotkeyDisplayString: String
    @Binding var launchAtLogin: Bool
    @Binding var showInDock: Bool
    @Binding var autoCheckForUpdates: Bool
    let launchAtLoginError: String?

    private var customPasteDescription: String {
        String(format: localized("Paste the latest Voxt result with %@."), customPasteHotkeyDisplayString)
    }

    var body: some View {
        GeneralSettingsCard(title: localizedKey("App Behavior")) {
            GeneralToggleRow(
                title: localizedKey("Show Realtime Text"),
                description: localizedKey("Shows live transcription text while recording."),
                isOn: $realtimeTextDisplayEnabled
            )

            GeneralToggleRow(
                title: localizedKey("Also copy result to clipboard"),
                description: localizedKey("Keeps each completed result in the clipboard."),
                isOn: $autoCopyWhenNoFocusedInput
            )

            GeneralToggleRow(
                title: localizedKey("Enable custom paste shortcut"),
                descriptionText: customPasteDescription,
                isOn: $customPasteHotkeyEnabled
            )

            GeneralToggleRow(
                title: localizedKey("Launch at Login"),
                isOn: $launchAtLogin
            )

            GeneralToggleRow(
                title: localizedKey("Show in Dock"),
                isOn: $showInDock
            )

            GeneralToggleRow(
                title: localizedKey("Automatically check for updates"),
                isOn: $autoCheckForUpdates
            )

            if let launchAtLoginError {
                Text(launchAtLoginError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }
}

struct GeneralLoggingCard: View {
    @Binding var hotkeyDebugLoggingEnabled: Bool
    @Binding var llmDebugLoggingEnabled: Bool
    let onViewLogs: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center, spacing: 12) {
                Text(localizedKey("Logging"))
                    .font(.headline)

                Spacer(minLength: 0)

                Button(localized("View Logs")) {
                    onViewLogs()
                }
                .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10, height: 28))
            }

            GeneralToggleRow(
                title: localizedKey("Enable hotkey debug logs"),
                description: localizedKey("Records hotkey detection, trigger routing, and shortcut handling details for debugging."),
                isOn: $hotkeyDebugLoggingEnabled
            )

            GeneralToggleRow(
                title: localizedKey("Enable model debug logs"),
                description: localizedKey("Records local and remote model details, including LLM, ASR, model downloads, and model routing, for debugging."),
                isOn: $llmDebugLoggingEnabled
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct GeneralVADCard: View {
    @Binding var localVADMode: LocalVADMode

    var body: some View {
        GeneralSettingsCard(title: localizedKey("VAD")) {
            GeneralFieldRow(
                title: localizedKey("VAD Mode"),
                description: LocalizedStringKey(localVADMode.detail)
            ) {
                SettingsMenuPicker(
                    selection: $localVADMode,
                    options: LocalVADMode.allCases.map { mode in
                        SettingsMenuOption(value: mode, title: mode.title)
                    },
                    selectedTitle: localVADMode.title,
                    width: 180
                )
            }
        }
    }
}

struct GeneralProxyCard: View {
    @Binding var networkProxyMode: VoxtNetworkSession.ProxyMode
    @Binding var customProxyScheme: VoxtNetworkSession.ProxyScheme
    @Binding var customProxyHost: String
    @Binding var customProxyPort: String
    @Binding var customProxyUsername: String
    @Binding var customProxyPassword: String

    var body: some View {
        GeneralSettingsCard(title: localizedKey("Proxy")) {
            GeneralFieldRow(
                title: localizedKey("Proxy"),
                description: localizedKey("Controls the proxy used by Voxt app network requests.")
            ) {
                SettingsMenuPicker(
                    selection: $networkProxyMode,
                    options: [
                        SettingsMenuOption(value: .system, title: localized("Follow System")),
                        SettingsMenuOption(value: .disabled, title: localized("Off")),
                        SettingsMenuOption(value: .custom, title: localized("Custom"))
                    ],
                    selectedTitle: networkProxyModeTitle,
                    width: 220
                )
            }

            if networkProxyMode == .custom {
                GeneralFieldRow(title: localizedKey("Protocol")) {
                    SettingsMenuPicker(
                        selection: $customProxyScheme,
                        options: [
                            SettingsMenuOption(value: .http, title: "HTTP"),
                            SettingsMenuOption(value: .https, title: "HTTPS"),
                            SettingsMenuOption(value: .socks5, title: "SOCKS5")
                        ],
                        selectedTitle: customProxySchemeTitle,
                        width: 160
                    )
                }

                proxyField(title: localizedKey("Host"), placeholder: "127.0.0.1", text: $customProxyHost, width: 220)
                proxyField(title: localizedKey("Port"), placeholder: "7890", text: $customProxyPort, width: 120)
                proxyField(title: localizedKey("Username"), placeholder: localized("Optional"), text: $customProxyUsername, width: 220)

                GeneralFieldRow(title: localizedKey("Password")) {
                    SecureField(localized("Optional"), text: $customProxyPassword)
                        .textFieldStyle(.plain)
                        .settingsFieldSurface(width: 220)
                }

                Text(localized("Custom proxy supports HTTP, HTTPS, and SOCKS5 host/port routing. Username and password are saved now, but not injected into requests automatically yet."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

        }
    }

    private func proxyField(title: LocalizedStringKey, placeholder: String, text: Binding<String>, width: CGFloat) -> some View {
        GeneralFieldRow(title: title) {
            TextField(placeholder, text: text)
                .textFieldStyle(.plain)
                .settingsFieldSurface(width: width)
        }
    }
}

private extension GeneralProxyCard {
    var networkProxyModeTitle: String {
        GeneralSettingsData.networkProxyModeTitle(networkProxyMode)
    }

    var customProxySchemeTitle: String {
        GeneralSettingsData.proxySchemeTitle(customProxyScheme)
    }
}

struct GeneralAdvancedCard<Content: View>: View {
    @Binding var isExpanded: Bool
    var onExpand: (() -> Void)?
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Button {
                isExpanded.toggle()
                if isExpanded {
                    onExpand?()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .animation(.easeInOut(duration: 0.14), value: isExpanded)

                    Text(localizedKey("Advanced"))
                        .font(.system(size: 14, weight: .semibold))

                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 14) {
                    content()
                }
                .transition(.opacity)
                .clipped()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct GeneralSectionDivider: View {
    var body: some View {
        Divider()
    }
}

struct GeneralSettingsCard<Content: View>: View {
    let title: Text
    let spacing: CGFloat
    @ViewBuilder let content: () -> Content

    init(
        title: LocalizedStringKey,
        spacing: CGFloat = 16,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.title = Text(title)
        self.spacing = spacing
        self.content = content
    }

    init(
        titleText: String,
        spacing: CGFloat = 16,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.title = Text(titleText)
        self.spacing = spacing
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: spacing) {
            title
                .font(.headline)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct GeneralLanguageSettingBlock<Control: View>: View {
    let title: LocalizedStringKey
    let description: LocalizedStringKey?
    @ViewBuilder let control: () -> Control

    var body: some View {
        GeneralFieldRow(title: title, description: description) {
            control()
        }
    }
}

struct GeneralFieldRow<TrailingContent: View>: View {
    private let titleText: Text
    private let descriptionText: Text?
    @ViewBuilder let trailingContent: () -> TrailingContent

    init(
        title: LocalizedStringKey,
        description: LocalizedStringKey? = nil,
        @ViewBuilder trailingContent: @escaping () -> TrailingContent
    ) {
        titleText = Text(title)
        descriptionText = description.map { Text($0) }
        self.trailingContent = trailingContent
    }

    init(
        titleText: String,
        descriptionText: String? = nil,
        @ViewBuilder trailingContent: @escaping () -> TrailingContent
    ) {
        self.titleText = Text(titleText)
        self.descriptionText = descriptionText.map { Text($0) }
        self.trailingContent = trailingContent
    }

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            VStack(alignment: .leading, spacing: 5) {
                titleText
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.primary.opacity(0.92))

                if let descriptionText {
                    descriptionText
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(alignment: .center, spacing: 8) {
                trailingContent()
            }
            .fixedSize(horizontal: true, vertical: false)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct GeneralToggleRow: View {
    let titleText: Text
    let description: Text?
    @Binding var isOn: Bool

    init(title: LocalizedStringKey, description: LocalizedStringKey, isOn: Binding<Bool>) {
        self.titleText = Text(title)
        self.description = Text(description)
        self._isOn = isOn
    }

    init(title: LocalizedStringKey, descriptionText: String, isOn: Binding<Bool>) {
        self.titleText = Text(title)
        self.description = Text(descriptionText)
        self._isOn = isOn
    }

    init(title: LocalizedStringKey, isOn: Binding<Bool>) {
        self.titleText = Text(title)
        self.description = nil
        self._isOn = isOn
    }

    init(titleText: String, isOn: Binding<Bool>) {
        self.titleText = Text(titleText)
        self.description = nil
        self._isOn = isOn
    }

    var body: some View {
        HStack(alignment: .center, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                titleText
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.primary.opacity(0.92))
                if let description {
                    description
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 18)

            Toggle("", isOn: $isOn)
                .labelsHidden()
                .toggleStyle(.switch)
        }
    }
}
