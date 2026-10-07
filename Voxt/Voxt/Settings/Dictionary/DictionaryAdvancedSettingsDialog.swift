// DictionaryAdvancedSettingsDialog.swift
// Provides Dictionary Advanced Settings Dialog for dictionary settings.

import SwiftUI

struct DictionaryAdvancedSettingsDialog: View {
    @Binding var dictionaryAutoLearningEnabled: Bool
    @Binding var automaticLearningPromptDraft: String
    @Binding var dictionaryHighConfidenceCorrectionEnabled: Bool
    @Binding var isPresented: Bool
    let onRestoreDefaultAutomaticLearningPrompt: () -> Void
    let onSave: () -> Void

    private let dialogWidth: CGFloat = 520
    private let dialogMaxHeight: CGFloat = 700
    private let contentMaxHeight: CGFloat = 620

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text(AppLocalization.localizedString("Dictionary Advanced Settings"))
                        .font(.title3.weight(.semibold))

                    DictionaryAutoCorrectionToggleSection(
                        dictionaryHighConfidenceCorrectionEnabled: $dictionaryHighConfidenceCorrectionEnabled,
                        dictionaryAutoLearningEnabled: $dictionaryAutoLearningEnabled
                    )

                    DictionaryAutomaticLearningPromptSection(
                        automaticLearningPromptDraft: $automaticLearningPromptDraft,
                        onRestoreDefaultAutomaticLearningPrompt: onRestoreDefaultAutomaticLearningPrompt
                    )
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: contentMaxHeight)

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Done")) {
                    onSave()
                    isPresented = false
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: dialogWidth, maxHeight: dialogMaxHeight, onClose: {
            isPresented = false
        })
    }
}

private struct DictionaryAutoCorrectionToggleSection: View {
    @Binding var dictionaryHighConfidenceCorrectionEnabled: Bool
    @Binding var dictionaryAutoLearningEnabled: Bool

    var body: some View {
        Toggle(
            AppLocalization.localizedString("Allow High-Confidence Auto Correction"),
            isOn: $dictionaryHighConfidenceCorrectionEnabled
        )
        .controlSize(.small)
        .toggleStyle(.switch)

        Text(AppLocalization.localizedString("Use exact dictionary terms for very high-confidence matches."))
            .font(.caption)
            .foregroundStyle(.secondary)

        Toggle(
            AppLocalization.localizedString("Auto-Add Corrected Terms"),
            isOn: $dictionaryAutoLearningEnabled
        )
        .controlSize(.small)
        .toggleStyle(.switch)

        Text(AppLocalization.localizedString("Learn confirmed corrections after text insertion."))
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}

private struct DictionaryAutomaticLearningPromptSection: View {
    @Binding var automaticLearningPromptDraft: String
    let onRestoreDefaultAutomaticLearningPrompt: () -> Void

    private var variables: [PromptTemplateVariableDescriptor] {
        [
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningMainLanguageTemplateVariable,
                tipKey: "Template tip {{USER_MAIN_LANGUAGE}}"
            ),
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningOtherLanguagesTemplateVariable,
                tipKey: "Template tip {{USER_OTHER_LANGUAGES}}"
            ),
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningInsertedTextTemplateVariable,
                tipKey: "Template tip {{INSERTED}}"
            ),
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningBaselineContextTemplateVariable,
                tipKey: "Template tip {{BEFORE_CTX}}"
            ),
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningFinalContextTemplateVariable,
                tipKey: "Template tip {{AFTER_CTX}}"
            ),
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningBaselineFragmentTemplateVariable,
                tipKey: "Template tip {{BEFORE_EDIT}}"
            ),
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningFinalFragmentTemplateVariable,
                tipKey: "Template tip {{AFTER_EDIT}}"
            ),
            PromptTemplateVariableDescriptor(
                token: AppPreferenceKey.automaticDictionaryLearningExistingTermsTemplateVariable,
                tipKey: "Template tip {{EXISTING}}"
            )
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(AppLocalization.localizedString("Correction Listener Prompt"))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                Spacer(minLength: 8)

                Button(AppLocalization.localizedString("Restore Default"), action: onRestoreDefaultAutomaticLearningPrompt)
                    .buttonStyle(SettingsPillButtonStyle())
            }

            PromptEditorView(
                text: $automaticLearningPromptDraft,
                height: 180,
                contentPadding: 2,
                variables: variables
            )

            Text(AppLocalization.localizedString("Used to decide which user corrections should become dictionary terms."))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
