// ModelSettingsSections.swift
// Provides Model Settings Sections for model settings.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

extension ModelSettingsView {
    @ViewBuilder
    func localASRConfigurationSheet(for target: LocalASRConfigurationTarget) -> some View {
        switch target {
        case .mlx(let repo):
            MLXASRConfigurationSheetView(
                modelRepo: repo,
                modelTitle: mlxModelManager.displayTitle(for: repo),
                capability: MLXModelCatalog.capability(for: repo),
                hintSettings: asrHintSettingsBinding(for: .mlxAudio),
                tuningSettings: mlxLocalTuningSettingsBinding(for: repo),
                userLanguageCodes: selectedUserLanguageCodes
            ) {
                activeLocalASRConfigurationTarget = nil
            }
        }
    }
}

private struct MLXASRConfigurationSheetView: View {
    private static let dictionaryTermsVariable = [
        PromptTemplateVariableDescriptor(
            token: AppPreferenceKey.asrDictionaryTermsTemplateVariable,
            tipKey: "Template tip {{DICTIONARY_TERMS}}"
        )
    ]

    let modelRepo: String
    let modelTitle: String
    let capability: MLXASRModelCapability
    @Binding var hintSettings: ASRHintSettings
    @Binding var tuningSettings: MLXLocalTuningSettings
    @State private var mossUsageScope: MossASRUsageScope = .dictation
    let userLanguageCodes: [String]
    let onDone: () -> Void

    private var family: MLXModelFamily { capability.family }
    private var configurationCapabilities: MLXASRConfigurationCapability {
        capability.configurationCapabilities
    }

    private var mainLanguageSummary: String {
        ASRHintResolver.selectedLanguageSummary(userLanguageCodes)
    }

    private var secondaryLanguageSummary: String {
        ASRHintResolver.secondaryLanguageSummary(userLanguageCodes)
    }

    private var resolvedLanguage: String {
        guard hintSettings.followsUserMainLanguage else {
            return AppLocalization.localizedString("Automatic")
        }
        return ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: hintSettings,
            userLanguageCodes: userLanguageCodes,
            mlxModelRepo: modelRepo
        ).language ?? AppLocalization.localizedString("Automatic")
    }

    private var senseVoiceSupportedLanguageSummary: String {
        AppLocalization.localizedString("Automatic, zh, en, yue, ja, ko")
    }

    private var showsExplicitLanguageMatrixSummary: Bool {
        configurationCapabilities.contains(.senseVoiceITN)
    }

    private var explicitLanguageMatrixSummary: String {
        let codes = capability.supportedLanguageCodes.sorted()
        guard !codes.isEmpty else { return senseVoiceSupportedLanguageSummary }
        let labels = ["Automatic"] + codes
        return labels.joined(separator: ", ")
    }

    private var unsupportedPrimaryLanguageWarning: String? {
        guard hintSettings.followsUserMainLanguage,
              let primary = UserMainLanguageOption.option(for: userLanguageCodes.first ?? "")
        else {
            return nil
        }
        switch capability.languageRouting {
        case .unavailable, .automatic:
            return nil
        case .iso6391, .localeOrISO6391, .languageName:
            guard !capability.supportsLanguage(code: primary.baseLanguageCode) else { return nil }
            return AppLocalization.format(
                "Primary language %@ is not supported by this model. Recognition will fall back to Automatic.",
                primary.title()
            )
        }
    }

    private var showsAutomaticLanguageDetectionSummary: Bool {
        configurationCapabilities.isEmpty
            && (capability.languageRouting == .automatic || family == .generic)
    }

    private var mossOutputModeBinding: Binding<MossASROutputMode> {
        Binding(
            get: {
                switch mossUsageScope {
                case .dictation: tuningSettings.mossOutputMode
                case .meeting: tuningSettings.mossMeetingOutputMode
                }
            },
            set: { mode in
                switch mossUsageScope {
                case .dictation: tuningSettings.mossOutputMode = mode
                case .meeting: tuningSettings.mossMeetingOutputMode = mode
                }
            }
        )
    }

    private var mossHotwordsBinding: Binding<String> {
        Binding(
            get: {
                switch mossUsageScope {
                case .dictation: tuningSettings.mossHotwords
                case .meeting: tuningSettings.mossMeetingHotwords
                }
            },
            set: { value in
                switch mossUsageScope {
                case .dictation: tuningSettings.mossHotwords = value
                case .meeting: tuningSettings.mossMeetingHotwords = value
                }
            }
        )
    }

    private var mossCustomPromptBinding: Binding<String> {
        Binding(
            get: {
                switch mossUsageScope {
                case .dictation: tuningSettings.mossCustomPrompt
                case .meeting: tuningSettings.mossMeetingCustomPrompt
                }
            },
            set: { value in
                switch mossUsageScope {
                case .dictation: tuningSettings.mossCustomPrompt = value
                case .meeting: tuningSettings.mossMeetingCustomPrompt = value
                }
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(localized("MLX ASR Configuration"))
                .font(.title3.weight(.semibold))

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(modelTitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if configurationCapabilities.contains(.recognitionPreset) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(localized("Preset"))
                                .font(.subheadline.weight(.medium))
                            SettingsMenuPicker(
                                selection: Binding(
                                    get: { tuningSettings.preset.rawValue },
                                    set: { rawValue in
                                        guard let preset = LocalASRRecognitionPreset(rawValue: rawValue) else { return }
                                        tuningSettings.preset = preset
                                    }
                                ),
                                options: LocalASRRecognitionPreset.allCases.map {
                                    SettingsMenuOption(value: $0.rawValue, title: $0.title)
                                },
                                selectedTitle: tuningSettings.preset.title,
                                width: 220
                            )
                            Text(tuningSettings.preset.summary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if configurationCapabilities.contains(.whisperTemperature) {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(alignment: .firstTextBaseline) {
                                Text(localized("Temperature"))
                                    .font(.subheadline.weight(.medium))
                                Spacer()
                                Text(String(format: "%.2f", tuningSettings.whisperTemperature))
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            Slider(value: $tuningSettings.whisperTemperature, in: 0...1, step: 0.05)
                            Text(localized("Higher values allow more variation. Keep this near 0 for deterministic dictation."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if configurationCapabilities.contains(.languageRouting) {
                        Toggle(localized("Follow User Main Language"), isOn: $hintSettings.followsUserMainLanguage)
                            .toggleStyle(.switch)

                        HStack(alignment: .top, spacing: 16) {
                            localInfoRow(label: localized("Primary language"), value: mainLanguageSummary)
                            localInfoRow(label: localized("Resolved language"), value: resolvedLanguage)
                        }

                        localInfoRow(label: localized("Other languages"), value: secondaryLanguageSummary)

                        if let unsupportedPrimaryLanguageWarning {
                            Text(unsupportedPrimaryLanguageWarning)
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                    }

                    if showsExplicitLanguageMatrixSummary {
                        localInfoRow(
                            label: localized("Supported routes"),
                            value: explicitLanguageMatrixSummary
                        )
                        Text(localized("This model only accepts explicit language routing for the listed languages here. Any other primary language falls back to Automatic."))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if configurationCapabilities.contains(.nemotronLatency) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(localized("Streaming Latency"))
                                .font(.subheadline.weight(.medium))
                            SettingsMenuPicker(
                                selection: $tuningSettings.nemotronStreamLatency,
                                options: NemotronStreamLatency.allCases.map {
                                    SettingsMenuOption(value: $0, title: $0.title)
                                },
                                selectedTitle: tuningSettings.nemotronStreamLatency.title,
                                width: 240
                            )
                            Text(localized("Smaller chunks update sooner; larger chunks favor recognition accuracy."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }


                    if configurationCapabilities.contains(.mossPromptAndOutput) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(localized("Usage"))
                                .font(.subheadline.weight(.medium))
                            Picker(localized("Usage"), selection: $mossUsageScope) {
                                ForEach(MossASRUsageScope.allCases) { scope in
                                    Text(scope.title).tag(scope)
                                }
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                        }

                        VStack(alignment: .leading, spacing: 8) {
                            Text(localized("Output Format"))
                                .font(.subheadline.weight(.medium))
                            if mossUsageScope == .meeting {
                                Text(localized("Structured Meeting Segments"))
                                Text(localized("MOSS always generates timestamped speaker segments for meetings; the meeting view displays cleaned speech text."))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else {
                                SettingsMenuPicker(
                                    selection: Binding(
                                        get: { mossOutputModeBinding.wrappedValue.rawValue },
                                        set: { rawValue in
                                            guard let mode = MossASROutputMode(rawValue: rawValue) else { return }
                                            mossOutputModeBinding.wrappedValue = mode
                                        }
                                    ),
                                    options: MossASROutputMode.allCases.map {
                                        SettingsMenuOption(value: $0.rawValue, title: $0.title)
                                    },
                                    selectedTitle: mossOutputModeBinding.wrappedValue.title,
                                    width: 240
                                )
                                Text(mossOutputModeBinding.wrappedValue.summary)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }

                        Text(localized("Hotwords"))
                            .font(.subheadline.weight(.medium))
                        PromptEditorView(
                            text: mossHotwordsBinding,
                            height: 90,
                            variables: Self.dictionaryTermsVariable
                        )
                        Text(localized("Names and terms are appended to the MOSS prompt using its official Hotwords format."))
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        if mossUsageScope == .meeting {
                            Text(localized("Recognition Prompt"))
                                .font(.subheadline.weight(.medium))
                            PromptEditorView(text: mossCustomPromptBinding, height: 120)
                            Text(localized("This optional meeting instruction is appended without replacing the required timestamp and speaker structure."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else if mossOutputModeBinding.wrappedValue == .customPrompt {
                            Text(localized("Recognition Prompt"))
                                .font(.subheadline.weight(.medium))
                            PromptEditorView(text: mossCustomPromptBinding, height: 120)
                            Text(localized("This instruction replaces the standard MOSS transcription format prompt."))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if configurationCapabilities.contains(.cohereLongForm) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(localized("Long Audio Segmentation"))
                                .font(.subheadline.weight(.medium))
                            SettingsMenuPicker(
                                selection: Binding(
                                    get: { tuningSettings.cohereLongFormStrategy.rawValue },
                                    set: { rawValue in
                                        guard let strategy = CohereLongFormStrategy(rawValue: rawValue) else { return }
                                        tuningSettings.cohereLongFormStrategy = strategy
                                    }
                                ),
                                options: CohereLongFormStrategy.allCases.map {
                                    SettingsMenuOption(value: $0.rawValue, title: $0.title)
                                },
                                selectedTitle: tuningSettings.cohereLongFormStrategy.title,
                                width: 220
                            )
                            Text(tuningSettings.cohereLongFormStrategy.summary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Toggle(localized("Punctuation and Capitalization"), isOn: $tuningSettings.cohereUsePunctuation)
                            .toggleStyle(.switch)
                        SettingsIntegerStepperField(
                            title: localized("Max Output Tokens"),
                            value: $tuningSettings.cohereMaxTokens,
                            range: 32...2048,
                            step: 32,
                            help: localized("Increase this only when long recordings are being truncated.")
                        )
                        decodingTemperatureControl(value: $tuningSettings.cohereTemperature)
                    }


                    if configurationCapabilities.contains(.qwenContext) {
                        Text(localized("Recognition Context"))
                            .font(.subheadline.weight(.medium))
                        PromptEditorView(text: $tuningSettings.qwenContextBias, height: 110, variables: Self.dictionaryTermsVariable)
                        Text(localized("Concise names, terms, and product vocabulary."))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }


                    if configurationCapabilities.contains(.senseVoiceITN) {
                        Toggle(localized("Enable ITN"), isOn: $tuningSettings.senseVoiceUseITN)
                            .toggleStyle(.switch)
                        Text(localized("ITN lets SenseVoice normalize spoken numbers, dates, and similar expressions into written form."))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if showsAutomaticLanguageDetectionSummary {
                        localInfoRow(
                            label: localized("Language detection"),
                            value: capability.supportedLanguageCodes.count > 1
                                ? localized("Automatic")
                                : capability.supportedLanguageCodes.first.map {
                                    UserMainLanguageOption.option(for: $0)?.title() ?? $0
                                } ?? localized("Checkpoint default")
                        )
                        Text(localized("This model uses checkpoint-defined decoding and does not expose additional controls."))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(.trailing, 4)
            }
            .frame(maxHeight: SettingsUIStyle.modelConfigurationScrollMaxHeight)

            SettingsDialogActionRow {
                Button(localized("Reset to Default")) {
                    hintSettings = ASRHintSettingsStore.defaultSettings(for: .mlxAudio)
                    tuningSettings = MLXLocalTuningSettings.defaults(for: .balanced, family: family)
                }
                .buttonStyle(SettingsPillButtonStyle())
            } trailing: {
                Button(localized("Done")) {
                    onDone()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(
            width: SettingsUIStyle.modelConfigurationDialogWidth,
            maxHeight: SettingsUIStyle.modelConfigurationDialogMaxHeight,
            onClose: onDone
        )
    }

    private func localInfoRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .multilineTextAlignment(.trailing)
        }
    }

    private func decodingTemperatureControl(value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(localized("Temperature"))
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: 0...1, step: 0.05)
            Text(localized("Keep this at 0 for deterministic decoding; higher values sample alternative tokens."))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

private struct SettingsIntegerStepperField: View {
    let title: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let step: Int
    let help: String

    private var clampedValue: Binding<Int> {
        Binding(
            get: { value },
            set: { value = clamped($0) }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 12) {
                Text(title)
                    .font(.subheadline.weight(.medium))

                Spacer(minLength: 12)

                HStack(spacing: 8) {
                    TextField("", value: clampedValue, formatter: Self.integerFormatter)
                        .font(.caption.monospacedDigit())
                        .multilineTextAlignment(.trailing)
                        .textFieldStyle(.plain)
                        .modifier(
                            SettingsFieldSurfaceModifier(
                                width: 72,
                                minHeight: 28,
                                horizontalPadding: 8,
                                alignment: .trailing
                            )
                        )
                        .onSubmit {
                            value = clamped(value)
                        }

                    Stepper("", value: clampedValue, in: range, step: step)
                        .labelsHidden()
                        .controlSize(.small)
                        .fixedSize()
                }
            }

            Text(help)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .onAppear {
            value = clamped(value)
        }
    }

    private func clamped(_ candidate: Int) -> Int {
        min(max(candidate, range.lowerBound), range.upperBound)
    }

    private static let integerFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .none
        formatter.allowsFloats = false
        formatter.minimum = 0
        return formatter
    }()
}
