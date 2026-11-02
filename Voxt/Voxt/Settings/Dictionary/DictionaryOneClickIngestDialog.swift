// DictionaryOneClickIngestDialog.swift
// Provides Dictionary One Click Ingest Dialog for dictionary settings.

import SwiftUI

struct DictionaryOneClickIngestDialog: View {
    @Binding var isPresented: Bool
    let pendingHistoryScanCount: Int
    let localModelOptions: [DictionaryHistoryScanModelOption]
    let remoteModelOptions: [DictionaryHistoryScanModelOption]
    @Binding var selectedModelID: String
    @Binding var draftPrompt: String
    let historyScanProgress: DictionaryHistoryScanProgress
    let statusText: String
    let cancellationText: String
    let actionMessage: String?
    let onRestoreDefaultPrompt: () -> Void
    let onSave: () -> Void
    let onStart: () -> Void
    let onCancelRunning: () -> Void

    private let dialogWidth: CGFloat = 560
    private let dialogMaxHeight: CGFloat = 760
    private let contentMaxHeight: CGFloat = 660

    private var modelOptions: [SettingsMenuOption<String>] {
        (localModelOptions + remoteModelOptions).map { option in
            SettingsMenuOption(value: option.id, title: option.title)
        }
    }

    private var selectedModelTitle: String {
        modelOptions.first(where: { $0.value == selectedModelID })?.title
            ?? modelOptions.first?.title
            ?? AppLocalization.localizedString("Select Model")
    }

    private var startButtonTitle: String {
        if historyScanProgress.isRunning {
            return historyScanProgress.isCancellationRequested
                ? AppLocalization.localizedString("Canceling...")
                : AppLocalization.localizedString("Cancel Ingest")
        }
        return AppLocalization.localizedString("Start Ingest")
    }

    private var startButtonDisabled: Bool {
        if historyScanProgress.isRunning {
            return historyScanProgress.isCancellationRequested
        }
        return modelOptions.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text(AppLocalization.localizedString("One-Click Ingest"))
                        .font(.title3.weight(.semibold))

                    Text(
                        AppLocalization.format(
                            "%d new history records are ready for dictionary ingestion.",
                            pendingHistoryScanCount
                        )
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    DictionaryHistoryIngestProgressSection(
                        historyScanProgress: historyScanProgress,
                        statusText: statusText,
                        cancellationText: cancellationText,
                        actionMessage: actionMessage
                    )

                    DictionaryHistoryIngestModelSection(
                        modelOptions: modelOptions,
                        selectedModelID: $selectedModelID,
                        selectedModelTitle: selectedModelTitle,
                        isDisabled: historyScanProgress.isRunning
                    )

                    DictionaryHistoryIngestPromptSection(
                        draftPrompt: $draftPrompt,
                        isDisabled: historyScanProgress.isRunning,
                        onRestoreDefaultPrompt: onRestoreDefaultPrompt
                    )

                    Text(AppLocalization.localizedString("Scan history and add accepted terms to the dictionary."))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: contentMaxHeight)

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Cancel")) {
                    isPresented = false
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(AppLocalization.localizedString("Save")) {
                    onSave()
                }
                .buttonStyle(SettingsPillButtonStyle())
                .disabled(historyScanProgress.isRunning)

                Button(startButtonTitle) {
                    if historyScanProgress.isRunning {
                        onCancelRunning()
                    } else {
                        onStart()
                    }
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .disabled(startButtonDisabled)
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: dialogWidth, maxHeight: dialogMaxHeight, onClose: {
            isPresented = false
        })
    }
}

private struct DictionaryHistoryIngestProgressSection: View {
    let historyScanProgress: DictionaryHistoryScanProgress
    let statusText: String
    let cancellationText: String
    let actionMessage: String?

    var body: some View {
        if historyScanProgress.isRunning {
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(
                    value: Double(historyScanProgress.processedCount),
                    total: Double(max(historyScanProgress.totalCount, 1))
                )

                Text(historyScanProgress.isCancellationRequested ? cancellationText : statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else if let errorMessage = historyScanProgress.errorMessage,
                  !errorMessage.isEmpty {
            Text(errorMessage)
                .font(.caption)
                .foregroundStyle(.red)
        } else if let actionMessage, !actionMessage.isEmpty {
            Text(actionMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

private struct DictionaryHistoryIngestModelSection: View {
    let modelOptions: [SettingsMenuOption<String>]
    @Binding var selectedModelID: String
    let selectedModelTitle: String
    let isDisabled: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Model"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            SettingsMenuPicker(
                selection: $selectedModelID,
                options: modelOptions,
                selectedTitle: selectedModelTitle,
                width: 280
            )
            .disabled(isDisabled || modelOptions.isEmpty)

            if modelOptions.isEmpty {
                Text(AppLocalization.localizedString("No configured local or remote model is available for dictionary ingestion. Configure one in Model settings first."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct DictionaryHistoryIngestPromptSection: View {
    @Binding var draftPrompt: String
    let isDisabled: Bool
    let onRestoreDefaultPrompt: () -> Void

    private let variables = [
        PromptTemplateVariableDescriptor(
            token: "{{USER_MAIN_LANGUAGE}}",
            tipKey: "Template tip {{USER_MAIN_LANGUAGE}}"
        ),
        PromptTemplateVariableDescriptor(
            token: "{{USER_OTHER_LANGUAGES}}",
            tipKey: "Template tip {{USER_OTHER_LANGUAGES}}"
        ),
        PromptTemplateVariableDescriptor(
            token: "{{HISTORY_RECORDS}}",
            tipKey: "Template tip {{HISTORY_RECORDS}}"
        )
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(AppLocalization.localizedString("Ingest Prompt"))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                Spacer(minLength: 8)

                Button(AppLocalization.localizedString("Restore Default"), action: onRestoreDefaultPrompt)
                    .buttonStyle(SettingsPillButtonStyle())
                    .disabled(isDisabled)
            }

            PromptEditorView(
                text: $draftPrompt,
                height: 220,
                contentPadding: 2,
                variables: variables
            )
            .disabled(isDisabled)
        }
    }
}
