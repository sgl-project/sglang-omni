// FeatureLanguageToolSections.swift
// Provides Feature Language Tool Sections for feature settings.

import SwiftUI

extension FeatureSettingsView {
    var translationContent: some View {
        featurePage(
            title: featureSettingsLocalized("Translation"),
            subtitle: featureSettingsLocalized("Configure the speech path, translation engine, target language, and prompt behavior for translation mode."),
            iconKind: .translation,
            pills: translationPills,
            showsHeroHeader: false
        ) {
            FeatureSettingsCard(title: "") {
                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Speech Model"),
                        value: asrSelectionSummary(featureSettings.translation.asrSelectionID),
                        action: { selectorSheet = .translationASR }
                    )
                }

                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Translation Model"),
                        value: translationSelectionSummary(featureSettings.translation.modelSelectionID),
                        action: { selectorSheet = .translationModel }
                    )
                }

                FeatureInlinePickerRow(title: featureSettingsLocalized("Target Language"), detail: "") {
                    SettingsMenuPicker(
                        selection: binding(
                            get: { featureSettings.translation.targetLanguage },
                            set: { featureSettings.translation.targetLanguageRawValue = $0.rawValue }
                        ),
                        options: TranslationTargetLanguage.allCases.map {
                            SettingsMenuOption(value: $0, title: $0.title)
                        },
                        selectedTitle: featureSettings.translation.targetLanguage.title,
                        width: 280
                    )
                }

                FeatureEmbeddedFieldGroup {
                    FeatureToggleRow(
                        title: featureSettingsLocalized("Show Translation Result for Selected Text"),
                        detail: "",
                        isOn: binding(
                            get: { featureSettings.translation.showResultWindow },
                            set: { featureSettings.translation.showResultWindow = $0 }
                        )
                    )
                }

                FeatureSettingSection(title: "", detail: "") {
                    FeaturePromptSection(
                        title: featureSettingsLocalized("Translation Prompt"),
                        text: promptBinding(
                            get: { featureSettings.translation.prompt },
                            set: { featureSettings.translation.prompt = $0 },
                            kind: .translation
                        ),
                        defaultText: AppPromptDefaults.text(for: .translation),
                        kind: .translation,
                        selectedPresetID: Binding(
                            get: { featureSettings.translation.promptPresetID },
                            set: { featureSettings.translation.promptPresetID = $0 }
                        ),
                        variables: ModelSettingsPromptVariables.translation,
                        guidance: "",
                        persistChanges: { prompt in
                            FeatureSettingsStore.saveTranslationPrompt(
                                prompt,
                                presetID: featureSettings.translation.promptPresetID
                            )
                        },
                        persistPresetSelection: { presetID, prompt in
                            featureSettings.translation.promptPresetID = presetID
                            featureSettings.translation.prompt = prompt
                            FeatureSettingsStore.saveTranslationPrompt(prompt, presetID: presetID)
                        }
                    )
                }
            }
        }
    }

    var rewriteContent: some View {
        featurePage(
            title: featureSettingsLocalized("Rewrite"),
            subtitle: featureSettingsLocalized("Set the ASR and text model pairing used for rewrite mode, then tune the rewrite-specific prompt and follow-up shortcut."),
            iconKind: .rewrite,
            pills: rewritePills,
            showsHeroHeader: false
        ) {
            FeatureSettingsCard(title: "") {
                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Speech Model"),
                        value: asrSelectionSummary(featureSettings.rewrite.asrSelectionID),
                        action: { selectorSheet = .rewriteASR }
                    )
                }

                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Enhancement Model"),
                        value: llmSelectionSummary(featureSettings.rewrite.llmSelectionID),
                        action: { selectorSheet = .rewriteLLM }
                    )
                }

                FeatureSettingSection(title: "", detail: "") {
                    FeaturePromptSection(
                        title: featureSettingsLocalized("Rewrite Prompt"),
                        text: promptBinding(
                            get: { featureSettings.rewrite.prompt },
                            set: { featureSettings.rewrite.prompt = $0 },
                            kind: .rewrite
                        ),
                        defaultText: AppPromptDefaults.text(for: .rewrite),
                        kind: .rewrite,
                        selectedPresetID: Binding(
                            get: { featureSettings.rewrite.promptPresetID },
                            set: { featureSettings.rewrite.promptPresetID = $0 }
                        ),
                        variables: ModelSettingsPromptVariables.rewrite,
                        guidance: "",
                        persistChanges: { prompt in
                            FeatureSettingsStore.saveRewritePrompt(
                                prompt,
                                presetID: featureSettings.rewrite.promptPresetID
                            )
                        },
                        persistPresetSelection: { presetID, prompt in
                            featureSettings.rewrite.promptPresetID = presetID
                            featureSettings.rewrite.prompt = prompt
                            FeatureSettingsStore.saveRewritePrompt(prompt, presetID: presetID)
                        }
                    )
                }

                FeatureContinueShortcutRow(
                    title: featureSettingsLocalized("Continue Shortcut"),
                    detail: "",
                    shortcut: binding(
                        get: { featureSettings.rewrite.continueShortcut },
                        set: { featureSettings.rewrite.continueShortcut = $0 }
                    )
                )

            }
        }
    }
}
