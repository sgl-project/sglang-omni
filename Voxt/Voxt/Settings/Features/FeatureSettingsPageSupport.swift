// FeatureSettingsPageSupport.swift
// Provides Feature Settings Page Support for feature settings.

import SwiftUI

func featureSettingsLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

func featureSettingsLocalizedKey(_ key: String) -> LocalizedStringKey {
    LocalizedStringKey(AppLocalization.localizedString(key))
}

extension FeatureSettingsView {
    static let scrollBottomAnchorID = "feature-settings-scroll-bottom-anchor"

    func promptBinding(
        get: @escaping () -> String,
        set: @escaping (String) -> Void,
        kind: AppPromptKind
    ) -> Binding<String> {
        Binding(
            get: {
                AppPromptDefaults.resolvedStoredText(get(), kind: kind)
            },
            set: { newValue in
                set(AppPromptDefaults.canonicalStoredText(newValue, kind: kind))
            }
        )
    }

    var transcriptionPills: [FeatureSummaryPill] {
        [
            FeatureSummaryPill(
                title: featureSettingsLocalized("Audio Model"),
                value: shortSummary(asrSelectionSummary(featureSettings.transcription.asrSelectionID), maxLength: 52)
            ),
            FeatureSummaryPill(
                title: featureSettingsLocalized("Enhancement Model"),
                value: featureSettings.transcription.llmEnabled
                    ? shortSummary(llmSelectionSummary(featureSettings.transcription.llmSelectionID), maxLength: 52)
                    : featureSettingsLocalized("Off")
            )
        ]
    }

    var notePills: [FeatureSummaryPill] {
        [
            FeatureSummaryPill(
                title: featureSettingsLocalized("Trigger"),
                value: shortSummary(
                    HotkeyPreference.displayString(
                        for: HotkeyPreference.loadNoteBindings().first?.hotkey
                            ?? HotkeyPreference.Hotkey(
                                keyCode: HotkeyPreference.defaultNoteKeyCode,
                                modifiers: HotkeyPreference.defaultNoteModifiers,
                                sidedModifiers: HotkeyPreference.defaultNoteSidedModifiers
                            ),
                        distinguishModifierSides: true
                    )
                )
            ),
            FeatureSummaryPill(
                title: featureSettingsLocalized("Model"),
                value: shortSummary(llmSelectionSummary(featureSettings.transcription.notes.titleModelSelectionID))
            )
        ]
    }

    var translationPills: [FeatureSummaryPill] {
        [
            FeatureSummaryPill(title: featureSettingsLocalized("Audio Model"), value: shortSummary(asrSelectionSummary(featureSettings.translation.asrSelectionID))),
            FeatureSummaryPill(title: featureSettingsLocalized("Model"), value: shortSummary(translationSelectionSummary(featureSettings.translation.modelSelectionID))),
            FeatureSummaryPill(title: featureSettingsLocalized("Target"), value: featureSettings.translation.targetLanguage.title)
        ]
    }

    var rewritePills: [FeatureSummaryPill] {
        [
            FeatureSummaryPill(
                title: featureSettingsLocalized("Audio Model"),
                value: shortSummary(asrSelectionSummary(featureSettings.rewrite.asrSelectionID), maxLength: 52)
            ),
            FeatureSummaryPill(
                title: featureSettingsLocalized("Enhancement Model"),
                value: shortSummary(llmSelectionSummary(featureSettings.rewrite.llmSelectionID), maxLength: 52)
            )
        ]
    }

    func shortSummary(_ text: String, maxLength: Int = 28) -> String {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.count > maxLength else { return value }
        return String(value.prefix(max(maxLength - 3, 1))) + "..."
    }

    @ViewBuilder
    func featurePage<Content: View>(
        title: String,
        subtitle: String,
        titleBadge: String? = nil,
        iconKind: SettingsSidebarIconKind? = nil,
        systemImageName: String? = nil,
        pills: [FeatureSummaryPill],
        showsHeroHeader: Bool = true,
        allowsScrolling: Bool = true,
        @ViewBuilder content: @escaping () -> Content
    ) -> some View {
        let page = VStack(alignment: .leading, spacing: 18) {
            if showsHeroHeader || !pills.isEmpty {
                FeatureHeroCard(
                    title: showsHeroHeader ? title : "",
                    titleBadge: showsHeroHeader ? titleBadge : nil,
                    subtitle: showsHeroHeader ? subtitle : "",
                    iconKind: showsHeroHeader ? iconKind : nil,
                    systemImageName: showsHeroHeader ? systemImageName : nil,
                    pills: pills
                )
            }

            content()

            if allowsScrolling {
                Color.clear
                    .frame(height: 1)
                    .id(Self.scrollBottomAnchorID)
            }
        }
        .padding(.top, 2)
        .padding(.bottom, 12)
        .padding(.trailing, SettingsUIStyle.contentScrollTrailingGutter)
        .frame(
            maxWidth: .infinity,
            maxHeight: allowsScrolling ? nil : .infinity,
            alignment: .topLeading
        )

        Group {
            if allowsScrolling {
                ScrollViewReader { proxy in
                    ScrollView {
                        page
                    }
                    .onChange(of: scrollToBottomRequestRevision) { _, _ in
                        DispatchQueue.main.async {
                            withAnimation(.easeInOut(duration: 0.18)) {
                                proxy.scrollTo(Self.scrollBottomAnchorID, anchor: .bottom)
                            }
                        }
                    }
                }
            } else {
                page
            }
        }
        .padding(.trailing, -SettingsUIStyle.contentScrollIndicatorOutset)
        .background(SettingsUIStyle.groupedFillColor.opacity(0.001))
    }

    var transcriptionLLMEnabledBinding: Binding<Bool> {
        binding(
            get: { featureSettings.transcription.llmEnabled },
            set: { featureSettings.transcription.llmEnabled = $0 }
        )
    }

    @ViewBuilder
    var noteObsidianSyncSection: some View {
        FeatureEmbeddedFieldGroup {
            FeatureToggleRow(
                title: featureSettingsLocalized("Enable Obsidian Sync"),
                detail: featureSettingsLocalized("Export Voxt notes into an Obsidian vault by writing Markdown files directly into the selected folder."),
                isOn: binding(
                    get: { featureSettings.transcription.notes.obsidianSync.enabled },
                    set: { featureSettings.transcription.notes.obsidianSync.enabled = $0 }
                ),
                isEmbedded: true
            )

            if featureSettings.transcription.notes.obsidianSync.enabled {
                FeatureDirectorySelectionRow(
                    title: featureSettingsLocalized("Vault Folder"),
                    detail: "",
                    path: featureSettings.transcription.notes.obsidianSync.vaultPath.isEmpty
                        ? featureSettingsLocalized("Not configured")
                        : featureSettings.transcription.notes.obsidianSync.vaultPath,
                    buttonTitle: featureSettingsLocalized("Choose"),
                    action: chooseObsidianVaultDirectory,
                    isEmbedded: true
                )

                FeatureInlineTextFieldRow(
                    title: featureSettingsLocalized("Target Folder"),
                    detail: featureSettingsLocalized("Choose where inside the vault Voxt should write exported notes."),
                    text: binding(
                        get: { featureSettings.transcription.notes.obsidianSync.relativeFolder },
                        set: { featureSettings.transcription.notes.obsidianSync.relativeFolder = $0 }
                    ),
                    placeholder: "Voxt",
                    width: 244,
                    isEmbedded: true
                )

                FeatureInlinePickerRow(
                    title: featureSettingsLocalized("Grouping Mode"),
                    detail: "",
                    isEmbedded: true
                ) {
                    SettingsMenuPicker(
                        selection: binding(
                            get: { featureSettings.transcription.notes.obsidianSync.groupingMode },
                            set: { featureSettings.transcription.notes.obsidianSync.groupingMode = $0 }
                        ),
                        options: ObsidianNoteGroupingMode.allCases.map {
                            SettingsMenuOption(value: $0, title: $0.title)
                        },
                        selectedTitle: featureSettings.transcription.notes.obsidianSync.groupingMode.title,
                        width: 280
                    )
                }
            }
        }
    }

    @ViewBuilder
    var noteRemindersSyncSection: some View {
        FeatureEmbeddedFieldGroup {
            FeatureToggleRow(
                title: featureSettingsLocalized("Enable Reminders Sync"),
                detail: featureSettingsLocalized("Sync Voxt notes into Apple Reminders by creating and updating reminders in the selected list."),
                isOn: binding(
                    get: { featureSettings.transcription.notes.remindersSync.enabled },
                    set: { featureSettings.transcription.notes.remindersSync.enabled = $0 }
                ),
                isEmbedded: true
            )

            if featureSettings.transcription.notes.remindersSync.enabled {
                FeatureInlinePickerRow(
                    title: featureSettingsLocalized("Target List"),
                    detail: featureSettingsLocalized("Choose the Reminders list that should receive synced Voxt notes."),
                    isEmbedded: true
                ) {
                    SettingsSelectionButton(width: 280, action: presentRemindersListSelector) {
                        Text(selectedRemindersListTitle)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
            }
        }
    }
}

private extension ObsidianNoteGroupingMode {
    var title: String {
        switch self {
        case .session:
            return featureSettingsLocalized("Session")
        case .daily:
            return featureSettingsLocalized("Daily")
        case .file:
            return featureSettingsLocalized("Single Note File")
        }
    }
}
