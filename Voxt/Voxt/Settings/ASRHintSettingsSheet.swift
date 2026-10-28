// ASRHintSettingsSheet.swift
// Provides ASRHint Settings Sheet for settings screens.

import SwiftUI

private func asrHintSettingsLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

@MainActor
struct ASRHintSettingsSheet: View {
    let target: ASRHintTarget
    let userLanguageCodes: [String]
    let mlxModelRepo: String?
    let initialSettings: ASRHintSettings
    let onSave: (ASRHintSettings) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var draftSettings: ASRHintSettings

    init(
        target: ASRHintTarget,
        userLanguageCodes: [String],
        mlxModelRepo: String?,
        initialSettings: ASRHintSettings,
        onSave: @escaping (ASRHintSettings) -> Void
    ) {
        self.target = target
        self.userLanguageCodes = userLanguageCodes
        self.mlxModelRepo = mlxModelRepo
        self.initialSettings = initialSettings
        self.onSave = onSave
        _draftSettings = State(initialValue: initialSettings)
    }

    private var mainLanguage: UserMainLanguageOption {
        ASRHintResolver.selectedLanguageOptions(userLanguageCodes).first ?? UserMainLanguageOption.fallbackOption()
    }

    private var secondaryLanguagePreview: String {
        ASRHintResolver.secondaryLanguageSummary(userLanguageCodes)
    }

    private var resolvedPayload: ResolvedASRHintPayload {
        ASRHintResolver.resolve(
            target: target,
            settings: draftSettings,
            userLanguageCodes: userLanguageCodes,
            mlxModelRepo: mlxModelRepo
        )
    }

    private var resolvedDictationSettings: ResolvedDictationSettings {
        ASRHintResolver.resolveDictationSettings(
            settings: draftSettings,
            userLanguageCodes: userLanguageCodes
        )
    }

    private var promptVariables: [PromptTemplateVariableDescriptor] {
        [
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.asrDictionaryTermsTemplateVariable,
                tipKey: "Template tip {{DICTIONARY_TERMS}}"
            )
        ]
    }

    private var languagePreview: String {
        resolvedPayload.language ?? AppLocalization.localizedString("Automatic")
    }

    private var hintsPreview: String {
        let hints = resolvedPayload.languageHints
        return hints.isEmpty ? AppLocalization.localizedString("Not applied") : hints.joined(separator: ", ")
    }

    private var dictationLocalePreview: String {
        resolvedDictationSettings.localeIdentifier ?? AppLocalization.localizedString("System default")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(target.settingsTitle)
                    .font(.title3.weight(.semibold))
                Text(target.title)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Toggle(
                asrHintSettingsLocalized("Follow User Main Language"),
                isOn: $draftSettings.followsUserMainLanguage
            )

            HStack(alignment: .top, spacing: 16) {
                infoRow(label: asrHintSettingsLocalized("Primary language"), value: mainLanguage.title())
                infoRow(
                    label: asrHintSettingsLocalized(
                        target == .dictation ? "Resolved locale" : "Resolved language"
                    ),
                    value: target == .dictation ? dictationLocalePreview : languagePreview
                )
            }

            infoRow(label: asrHintSettingsLocalized("Other languages"), value: secondaryLanguagePreview)

            if target == .aliyunBailianASR {
                infoRow(label: asrHintSettingsLocalized("Language hints"), value: hintsPreview)
            }

            if target == .doubaoASR {
                infoRow(
                    label: asrHintSettingsLocalized("Chinese output"),
                    value: ASRHintResolver.outputVariantDescription(for: mainLanguage)
                )
            }

            if target == .mlxAudio, let mlxModelRepo, !mlxModelRepo.isEmpty {
                infoRow(label: asrHintSettingsLocalized("Current model"), value: mlxModelRepo)
            }

            if target == .dictation {
                Toggle(
                    asrHintSettingsLocalized("Prefer On-Device Recognition"),
                    isOn: $draftSettings.prefersOnDeviceRecognition
                )
                Toggle(
                    asrHintSettingsLocalized("Add Punctuation"),
                    isOn: $draftSettings.addsPunctuation
                )
                Toggle(
                    asrHintSettingsLocalized("Report Partial Results"),
                    isOn: $draftSettings.reportsPartialResults
                )

                Text(asrHintSettingsLocalized("Contextual Phrases"))
                    .font(.subheadline.weight(.medium))
                PromptEditorView(text: $draftSettings.contextualPhrasesText, height: 120)
                Text(
                    asrHintSettingsLocalized(
                        "Enter one phrase per line. These phrases bias Apple's recognizer toward names, products, and domain terms."
                    )
                )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                infoRow(
                    label: asrHintSettingsLocalized("Phrases count"),
                    value: String(resolvedDictationSettings.contextualPhrases.count)
                )
            }

            Text(target.helpText)
                .font(.caption)
                .foregroundStyle(.secondary)

            if target.supportsPromptEditor {
                Text(asrHintSettingsLocalized("Prompt"))
                    .font(.subheadline.weight(.medium))
                PromptEditorView(
                    text: $draftSettings.promptTemplate,
                    height: 128,
                    variables: promptVariables
                )
            }

            SettingsDialogActionRow {
                Button(asrHintSettingsLocalized("Reset to Default")) {
                    draftSettings = ASRHintSettingsStore.defaultSettings(for: target)
                }
                .buttonStyle(SettingsPillButtonStyle())
                .disabled(draftSettings == initialSettings && initialSettings == ASRHintSettingsStore.defaultSettings(for: target))
            } trailing: {
                Button(asrHintSettingsLocalized("Cancel")) {
                    dismiss()
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(asrHintSettingsLocalized("Save")) {
                    onSave(ASRHintSettingsStore.sanitized(draftSettings, for: target))
                    dismiss()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: 520, onClose: { dismiss() })
    }

    private func infoRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .multilineTextAlignment(.trailing)
        }
    }
}
