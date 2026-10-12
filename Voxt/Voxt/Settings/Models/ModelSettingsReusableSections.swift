// ModelSettingsReusableSections.swift
// Provides Model Settings Reusable Sections for model settings.

import SwiftUI

struct ModelSettingsProviderOption: Identifiable {
    let id: String
    let title: String
}

enum ModelSettingsPromptVariables {
    static let enhancement = [
        PromptTemplateVariableDescriptor(
            token: AppDelegate.userMainLanguageTemplateVariable,
            tipKey: "Template tip {{USER_MAIN_LANGUAGE}}"
        )
    ]

    static let translation = [
        PromptTemplateVariableDescriptor(
            token: "{{TARGET_LANGUAGE}}",
            tipKey: "Template tip {{TARGET_LANGUAGE}}"
        ),
        PromptTemplateVariableDescriptor(
            token: "{{USER_MAIN_LANGUAGE}}",
            tipKey: "Template tip {{USER_MAIN_LANGUAGE}}"
        )
    ]

    static let rewrite: [PromptTemplateVariableDescriptor] = []

    static let appEnhancement = [
        PromptTemplateVariableDescriptor(
            token: AppDelegate.userMainLanguageTemplateVariable,
            tipKey: "Template tip {{USER_MAIN_LANGUAGE}}"
        )
    ]
}

enum PromptAuthoringGuidance {
    static let optionalVariablesTitle = AppLocalization.localizedString("Optional variables")

    static let enhancement = AppLocalization.localizedString(
        "Write stable cleanup rules only. Do not paste raw transcription here. Voxt injects the transcription and glossary automatically."
    )

    static let translation = AppLocalization.localizedString(
        "Write translation rules only. Do not paste source text here. Voxt injects the source text, target language, and glossary automatically."
    )

    static let rewrite = AppLocalization.localizedString(
        "Write rewrite behavior rules only. Do not paste spoken instructions or source text here. Voxt injects both automatically at runtime."
    )

    static let appEnhancement = AppLocalization.localizedString(
        "Recommended: describe only app-specific tone or formatting preferences. Do not paste raw transcription here. Voxt injects the transcription automatically."
    )
}

struct ResettablePromptSection: View {
    let title: LocalizedStringKey
    @Binding var text: String
    let defaultText: String
    let variables: [PromptTemplateVariableDescriptor]
    var guidance: String? = nil
    var variablesTitle: String? = nil
    var promptHeight: CGFloat = 124
    var titleUsesFeatureRowStyle = false
    var promptPresets: [FeaturePromptPreset] = []
    var promptPresetWidth: CGFloat = 200
    var showsResetButton = true
    var selectedPromptPresetID: String?
    var onSelectPromptPreset: ((FeaturePromptPreset) -> Void)?
    var onTextChange: ((String) -> Void)?
    var onFocusChange: ((Bool) -> Void)?

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Text(title)
                .font(titleUsesFeatureRowStyle ? .body.weight(.semibold) : .subheadline.weight(.medium))
                .foregroundStyle(Color.primary.opacity(titleUsesFeatureRowStyle ? 0.92 : 1))
            Spacer()
            if !promptPresets.isEmpty {
                SettingsMenuPicker(
                    selection: promptPresetSelection,
                    options: promptPresetOptions,
                    selectedTitle: selectedPromptPresetTitle,
                    width: promptPresetWidth
                )
            }
            if showsResetButton {
                Button(AppLocalization.localizedString("Reset to Default")) {
                    resetPrompt()
                }
                .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 10))
                .disabled(text == resetPromptText)
            }
        }

        if let guidance, !guidance.isEmpty {
            Text(guidance)
                .font(.caption)
                .foregroundStyle(.secondary)
        }

        PromptEditorView(
            text: $text,
            height: promptHeight,
            variables: variables,
            variablesTitle: variablesTitle,
            onTextChange: onTextChange,
            onFocusChange: onFocusChange
        )
    }

    private var selectedPromptPreset: FeaturePromptPreset? {
        promptPresets.first { $0.id == selectedPromptPresetID }
    }

    private var resetPromptText: String {
        selectedPromptPreset?.prompt ?? promptPresets.first?.prompt ?? defaultText
    }

    private var selectedPromptPresetTitle: String {
        guard let preset = selectedPromptPreset else {
            return AppLocalization.localizedString("Custom")
        }
        guard text != preset.prompt else { return preset.title }
        return String(
            format: AppLocalization.localizedString("%@ (Modified)"),
            preset.title
        )
    }

    private var promptPresetOptions: [SettingsMenuOption<String>] {
        var options = promptPresets.map { preset in
            SettingsMenuOption(
                value: preset.id,
                title: preset.id == selectedPromptPresetID
                    ? selectedPromptPresetTitle
                    : preset.title
            )
        }
        if selectedPromptPreset == nil {
            options.insert(
                SettingsMenuOption(
                    value: FeaturePromptPresetCatalog.customPresetID,
                    title: AppLocalization.localizedString("Custom")
                ),
                at: 0
            )
        }
        return options
    }

    private var promptPresetSelection: Binding<String> {
        Binding(
            get: { selectedPromptPresetID ?? FeaturePromptPresetCatalog.customPresetID },
            set: { newValue in
                guard newValue != FeaturePromptPresetCatalog.customPresetID,
                      let preset = promptPresets.first(where: { $0.id == newValue }) else {
                    return
                }
                onSelectPromptPreset?(preset)
            }
        )
    }

    private func resetPrompt() {
        if let preset = selectedPromptPreset ?? promptPresets.first {
            onSelectPromptPreset?(preset)
        } else {
            text = defaultText
        }
    }
}

struct ModelTaskSettingsCard: View {
    let title: LocalizedStringKey
    let providerPickerTitle: LocalizedStringKey
    let providerOptions: [ModelSettingsProviderOption]
    @Binding var selectedProviderID: String
    let modelLabelText: String
    let modelPickerTitle: LocalizedStringKey
    let modelOptions: [TranslationModelOption]
    let selectedModelBinding: Binding<String>
    let modelDisplayText: String?
    let emptyStateText: String
    let statusMessage: String?
    let statusIsWarning: Bool
    let promptTitle: LocalizedStringKey
    @Binding var promptText: String
    let defaultPromptText: String
    let variables: [PromptTemplateVariableDescriptor]
    let promptGuidance: String

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Text(title)
                    .font(.headline)

                HStack(alignment: .center, spacing: 12) {
                    SettingsMenuPicker(
                        selection: $selectedProviderID,
                        options: providerOptions.map { provider in
                            SettingsMenuOption(value: provider.id, title: provider.title)
                        },
                        selectedTitle: selectedProviderTitle,
                        width: 236
                    )

                    if let modelDisplayText {
                        Text(modelDisplayText)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .settingsFieldSurface(minHeight: 34, horizontalPadding: 12)
                    } else if modelOptions.isEmpty {
                        Text(AppLocalization.localizedString("Not available"))
                            .foregroundStyle(.tertiary)
                            .settingsFieldSurface(minHeight: 34, horizontalPadding: 12)
                    } else {
                        SettingsMenuPicker(
                            selection: selectedModelBinding,
                            options: modelOptions.map { option in
                                SettingsMenuOption(value: option.id, title: option.title)
                            },
                            selectedTitle: selectedModelTitle,
                            width: 260
                        )
                        .id("model-picker-\(selectedProviderID)")
                    }
                }

                if modelOptions.isEmpty {
                    Text(emptyStateText)
                        .font(.caption)
                        .foregroundStyle(.orange)
                }

                if let statusMessage {
                    Text(statusMessage)
                        .font(.caption)
                        .foregroundStyle(statusIsWarning ? .orange : .secondary)
                }

                ResettablePromptSection(
                    title: promptTitle,
                    text: $promptText,
                    defaultText: defaultPromptText,
                    variables: variables,
                    guidance: promptGuidance,
                    variablesTitle: PromptAuthoringGuidance.optionalVariablesTitle
                )
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private extension ModelTaskSettingsCard {
    var selectedProviderTitle: String {
        providerOptions.first(where: { $0.id == selectedProviderID }).map(\.title)
            ?? selectedProviderID
    }

    var selectedModelTitle: String {
        modelOptions.first(where: { $0.id == selectedModelBinding.wrappedValue })?.title
            ?? selectedModelBinding.wrappedValue
    }
}
