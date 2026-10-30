import SwiftUI

/// A feature that keeps its own speech or text model selection.
struct ModelSceneAssignmentRow: Identifiable {
    enum SelectionKind {
        case asr
        case text
        case translation
    }

    let id: String
    let titleKey: String
    let selector: FeatureModelSelectorSheet
    let selectionKeyPath: WritableKeyPath<FeatureSettings, FeatureModelSelectionID>
    let selectionKind: SelectionKind

    func selection(in settings: FeatureSettings) -> FeatureModelSelectionID {
        settings[keyPath: selectionKeyPath]
    }

    @MainActor
    func entry(
        for selection: FeatureModelSelectionID,
        catalog: FeatureModelCatalogBuilder
    ) -> FeatureModelSelectorEntry? {
        switch selectionKind {
        case .asr:
            guard selection.asrSelection != nil else { return nil }
        case .text:
            guard selection.textSelection != nil else { return nil }
        case .translation:
            guard selection.translationSelection != nil else { return nil }
        }
        return catalog.entries(for: selector).first { $0.selectionID == selection }
    }
}

enum ModelSceneAssignment {
    @MainActor
    static func apply(
        selection: FeatureModelSelectionID,
        rows: [ModelSceneAssignmentRow],
        catalog: FeatureModelCatalogBuilder,
        to settings: inout FeatureSettings
    ) {
        for row in rows where row.entry(for: selection, catalog: catalog)?.isSelectable == true {
            settings[keyPath: row.selectionKeyPath] = selection
        }
    }

    /// The scenes already using the model, or every compatible scene when none do.
    @MainActor
    static func defaultSelection(
        selection: FeatureModelSelectionID,
        rows: [ModelSceneAssignmentRow],
        catalog: FeatureModelCatalogBuilder
    ) -> Set<String> {
        let compatible = rows.filter { $0.entry(for: selection, catalog: catalog)?.isSelectable == true }
        let inUse = compatible.filter { $0.selection(in: catalog.featureSettings) == selection }
        return Set((inUse.isEmpty ? compatible : inUse).map(\.id))
    }
}

enum ASRAssignmentScene: String, CaseIterable, Identifiable {
    case transcription = "Transcription"
    case translation = "Translation"
    case rewrite = "Rewrite"
    case meeting = "Meeting"

    var id: String { rawValue }

    var selectionKeyPath: WritableKeyPath<FeatureSettings, FeatureModelSelectionID> {
        Self.row(for: self).selectionKeyPath
    }

    @MainActor
    func entry(
        for selection: FeatureModelSelectionID,
        catalog: FeatureModelCatalogBuilder
    ) -> FeatureModelSelectorEntry? {
        Self.row(for: self).entry(for: selection, catalog: catalog)
    }

    static var rows: [ModelSceneAssignmentRow] {
        allCases.map(Self.row(for:))
    }

    @MainActor
    static func apply(
        selection: FeatureModelSelectionID,
        scenes: Set<Self>,
        catalog: FeatureModelCatalogBuilder,
        to settings: inout FeatureSettings
    ) {
        ModelSceneAssignment.apply(
            selection: selection,
            rows: scenes.map(Self.row(for:)),
            catalog: catalog,
            to: &settings
        )
    }

    private static func row(for scene: Self) -> ModelSceneAssignmentRow {
        switch scene {
        case .transcription:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .transcriptionASR,
                selectionKeyPath: \.transcription.asrSelectionID,
                selectionKind: .asr
            )
        case .translation:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .translationASR,
                selectionKeyPath: \.translation.asrSelectionID,
                selectionKind: .asr
            )
        case .rewrite:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .rewriteASR,
                selectionKeyPath: \.rewrite.asrSelectionID,
                selectionKind: .asr
            )
        case .meeting:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .meetingASR,
                selectionKeyPath: \.meeting.asrSelectionID,
                selectionKind: .asr
            )
        }
    }
}

enum LLMAssignmentScene: String, CaseIterable, Identifiable {
    case transcription = "Transcription Enhancement"
    case translation = "Translation"
    case rewrite = "Rewrite"
    case meeting = "Meeting Summary"
    case noteTitle = "Note Title Model"

    var id: String { rawValue }

    var selectionKeyPath: WritableKeyPath<FeatureSettings, FeatureModelSelectionID> {
        Self.row(for: self).selectionKeyPath
    }

    @MainActor
    func entry(
        for selection: FeatureModelSelectionID,
        catalog: FeatureModelCatalogBuilder
    ) -> FeatureModelSelectorEntry? {
        Self.row(for: self).entry(for: selection, catalog: catalog)
    }

    static var rows: [ModelSceneAssignmentRow] {
        allCases.map(Self.row(for:))
    }

    @MainActor
    static func apply(
        selection: FeatureModelSelectionID,
        scenes: Set<Self>,
        catalog: FeatureModelCatalogBuilder,
        to settings: inout FeatureSettings
    ) {
        ModelSceneAssignment.apply(
            selection: selection,
            rows: scenes.map(Self.row(for:)),
            catalog: catalog,
            to: &settings
        )
    }

    private static func row(for scene: Self) -> ModelSceneAssignmentRow {
        switch scene {
        case .transcription:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .transcriptionLLM,
                selectionKeyPath: \.transcription.llmSelectionID,
                selectionKind: .text
            )
        case .translation:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .translationModel,
                selectionKeyPath: \.translation.modelSelectionID,
                selectionKind: .translation
            )
        case .rewrite:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .rewriteLLM,
                selectionKeyPath: \.rewrite.llmSelectionID,
                selectionKind: .text
            )
        case .meeting:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .meetingSummary,
                selectionKeyPath: \.meeting.summaryModelSelectionID,
                selectionKind: .text
            )
        case .noteTitle:
            return ModelSceneAssignmentRow(
                id: scene.rawValue,
                titleKey: scene.rawValue,
                selector: .transcriptionNoteTitle,
                selectionKeyPath: \.transcription.notes.titleModelSelectionID,
                selectionKind: .text
            )
        }
    }
}

struct ModelSceneAssignmentSheet: View {
    @Environment(\.dismiss) private var dismiss
    let selection: FeatureModelSelectionID
    let catalog: FeatureModelCatalogBuilder
    let rows: [ModelSceneAssignmentRow]
    let summary: String
    let explanationKey: String
    let onApply: (Set<String>) -> Void
    @State private var selectedIDs: Set<String>

    init(
        selection: FeatureModelSelectionID,
        catalog: FeatureModelCatalogBuilder,
        rows: [ModelSceneAssignmentRow],
        summary: String,
        explanationKey: String,
        onApply: @escaping (Set<String>) -> Void
    ) {
        self.selection = selection
        self.catalog = catalog
        self.rows = rows
        self.summary = summary
        self.explanationKey = explanationKey
        self.onApply = onApply
        _selectedIDs = State(
            initialValue: ModelSceneAssignment.defaultSelection(
                selection: selection,
                rows: rows,
                catalog: catalog
            )
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(AppLocalization.localizedString("Apply to Scenes"))
                .font(.title3.weight(.semibold))
            Text(summary)
                .font(.headline)
            Text(AppLocalization.localizedString(explanationKey))
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(rows) { row in
                        rowView(row)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 360)

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Select All")) {
                    selectedIDs = compatibleIDs
                }
                .buttonStyle(SettingsPillButtonStyle())
                .disabled(compatibleIDs.isEmpty)
            } trailing: {
                Button(AppLocalization.localizedString("Cancel")) {
                    dismiss()
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(AppLocalization.localizedString("Apply")) {
                    onApply(selectedIDs)
                    dismiss()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
                .disabled(selectedIDs.isEmpty)
            }
        }
        .settingsDialogChrome(width: 520, onClose: { dismiss() })
    }

    private var compatibleIDs: Set<String> {
        Set(rows.filter { row in
            row.entry(for: selection, catalog: catalog)?.isSelectable == true
        }.map(\.id))
    }

    @ViewBuilder
    private func rowView(_ row: ModelSceneAssignmentRow) -> some View {
        let entry = row.entry(for: selection, catalog: catalog)
        let isSelectable = entry?.isSelectable == true

        Toggle(isOn: Binding(
            get: { selectedIDs.contains(row.id) },
            set: { enabled in
                if enabled {
                    selectedIDs.insert(row.id)
                } else {
                    selectedIDs.remove(row.id)
                }
            }
        )) {
            VStack(alignment: .leading, spacing: 4) {
                Text(AppLocalization.localizedString(row.titleKey))
                Text(AppLocalization.format("Current model: %@", currentSelectionTitle(row)))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !isSelectable {
                    Text(entry?.disabledReason ?? AppLocalization.localizedString("This model is not supported for this scene."))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .toggleStyle(.checkbox)
        .disabled(!isSelectable)
    }

    private func currentSelectionTitle(_ row: ModelSceneAssignmentRow) -> String {
        let current = row.selection(in: catalog.featureSettings)
        switch row.selectionKind {
        case .asr:
            return catalog.asrSelectionSummary(current)
        case .text:
            return catalog.llmSelectionSummary(current)
        case .translation:
            return catalog.translationSelectionSummary(current)
        }
    }
}
