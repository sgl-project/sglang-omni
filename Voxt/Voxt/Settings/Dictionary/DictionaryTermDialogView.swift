// DictionaryTermDialogView.swift
// Provides Dictionary Term Dialog View for dictionary settings.

import SwiftUI

private func localizedDictionaryTermDialog(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct DictionaryTermDialogView: View {
    let dialog: DictionaryDialog
    let availableCategories: [DictionaryCategory]
    let availableGroups: [AppBranchGroup]
    let onCancel: () -> Void
    let onSave: ([String], [String], UUID, UUID?) throws -> Void
    private let mode: DictionaryTermDialogMode

    @State private var draftTerm: String
    @State private var draftReplacementTermInput = ""
    @State private var draftReplacementTerms: [String]
    @State private var selectedCategoryID: UUID
    @State private var selectedGroupOptionID: String
    @State private var isAdvancedExpanded = false
    @State private var errorMessage: String?

    private static let globalGroupOptionID = "global"

    init(
        dialog: DictionaryDialog,
        availableCategories: [DictionaryCategory],
        availableGroups: [AppBranchGroup],
        onCancel: @escaping () -> Void,
        onSave: @escaping ([String], [String], UUID, UUID?) throws -> Void
    ) {
        self.dialog = dialog
        self.availableCategories = availableCategories
        self.availableGroups = availableGroups
        self.onCancel = onCancel
        self.onSave = onSave
        self.mode = dialog.mode

        switch dialog {
        case .create(let categoryID, let mode):
            _draftTerm = State(initialValue: "")
            _draftReplacementTerms = State(initialValue: [])
            _selectedCategoryID = State(initialValue: categoryID ?? DictionaryCategory.defaultID)
            _selectedGroupOptionID = State(initialValue: Self.globalGroupOptionID)
            _isAdvancedExpanded = State(initialValue: mode == .replacement)
        case .edit(let entry):
            _draftTerm = State(initialValue: entry.term)
            _draftReplacementTerms = State(initialValue: entry.replacementTerms.map(\.text))
            _selectedCategoryID = State(initialValue: entry.categoryID)
            _selectedGroupOptionID = State(initialValue: Self.groupOptionID(for: entry.groupID))
            _isAdvancedExpanded = State(initialValue: !entry.replacementTerms.isEmpty)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(verbatim: dialog.title)
                .font(.title3.weight(.semibold))

            TextField(
                "",
                text: $draftTerm,
                prompt: Text(verbatim: dictionaryTermPlaceholder),
                axis: .vertical
            )
            .font(.system(size: 13))
            .lineLimit(1...5)
            .textFieldStyle(.plain)
            .padding(.vertical, 8)
            .settingsFieldSurface(minHeight: 96, alignment: .topLeading)

            editorOptionsSection

            if let errorMessage, !errorMessage.isEmpty {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            SettingsDialogActionRow {
                if mode == .hotword {
                    SettingsMenuPicker(
                        selection: $selectedCategoryID,
                        options: dictionaryCategoryOptions,
                        selectedTitle: selectedDictionaryCategoryTitle,
                        width: 190
                    )
                    .help(localizedDictionaryTermDialog("Category"))
                }
            } trailing: {
                Button {
                    onCancel()
                } label: {
                    Text(verbatim: localizedDictionaryTermDialog("Cancel"))
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button {
                    save()
                } label: {
                    Text(verbatim: dialog.confirmButtonTitle)
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: 520, onClose: onCancel)
    }

    private var dictionaryTermPlaceholder: String {
        switch dialog {
        case .create:
            switch mode {
            case .hotword:
                return localizedDictionaryTermDialog("One hot word per line")
            case .replacement:
                return localizedDictionaryTermDialog("Replacement target term")
            }
        case .edit:
            switch mode {
            case .hotword:
                return localizedDictionaryTermDialog("Hot Word")
            case .replacement:
                return localizedDictionaryTermDialog("Replacement target term")
            }
        }
    }

    private var dictionaryCategoryOptions: [SettingsMenuOption<UUID>] {
        var options = availableCategories.map { category in
            SettingsMenuOption(value: category.id, title: category.name)
        }
        if !options.contains(where: { $0.value == selectedCategoryID }) {
            options.append(SettingsMenuOption(value: selectedCategoryID, title: localizedDictionaryTermDialog("Missing Category")))
        }
        return options
    }

    private var selectedDictionaryCategoryTitle: String {
        availableCategories.first(where: { $0.id == selectedCategoryID })?.name
            ?? localizedDictionaryTermDialog("Missing Category")
    }

    private var dictionaryGroupOptions: [SettingsMenuOption<String>] {
        var options = [
            SettingsMenuOption(value: Self.globalGroupOptionID, title: localizedDictionaryTermDialog("Global"))
        ]
        options.append(contentsOf: availableGroups.map { group in
            SettingsMenuOption(value: Self.groupOptionID(for: group.id), title: group.name)
        })
        if selectedGroupOptionID != Self.globalGroupOptionID,
           !options.contains(where: { $0.value == selectedGroupOptionID }) {
            options.append(SettingsMenuOption(value: selectedGroupOptionID, title: localizedDictionaryTermDialog("Missing Group")))
        }
        return options
    }

    private var selectedDictionaryGroupTitle: String {
        guard let selectedGroupID else {
            return localizedDictionaryTermDialog("Global")
        }
        return availableGroups.first(where: { $0.id == selectedGroupID })?.name
            ?? localizedDictionaryTermDialog("Missing Group")
    }

    private var selectedGroupID: UUID? {
        guard selectedGroupOptionID != Self.globalGroupOptionID else { return nil }
        return UUID(uuidString: selectedGroupOptionID)
    }

    @ViewBuilder
    private var editorOptionsSection: some View {
        switch mode {
        case .hotword:
            EmptyView()
        case .replacement:
            replacementEditorSection
        }
    }

    private var replacementEditorSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button {
                withAnimation(.easeInOut(duration: 0.14)) {
                    isAdvancedExpanded.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isAdvancedExpanded ? 90 : 0))

                    Text(verbatim: localizedDictionaryTermDialog("Replacement Settings"))
                        .font(.system(size: 13, weight: .semibold))

                    if replacementTermCount > 0 {
                        Text(verbatim: "\(replacementTermCount)")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(Color.secondary.opacity(0.12))
                            )
                    }

                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isAdvancedExpanded {
                VStack(alignment: .leading, spacing: 14) {
                    applicationScopeSection
                    replacementTermsSection
                }
                .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var applicationScopeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(verbatim: localizedDictionaryTermDialog("Application Scope (App Enhancement)"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            SettingsMenuPicker(
                selection: $selectedGroupOptionID,
                options: dictionaryGroupOptions,
                selectedTitle: selectedDictionaryGroupTitle,
                width: 240
            )
        }
    }

    private var replacementTermsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(verbatim: localizedDictionaryTermDialog("Replacement Match Terms"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                TextField(
                    "",
                    text: $draftReplacementTermInput,
                    prompt: Text(verbatim: localizedDictionaryTermDialog("Replacement Match Term"))
                )
                .textFieldStyle(.plain)
                .settingsFieldSurface()
                .onSubmit(addDraftReplacementTerm)

                Button {
                    addDraftReplacementTerm()
                } label: {
                    Text(verbatim: localizedDictionaryTermDialog("Add"))
                }
                .buttonStyle(SettingsPillButtonStyle())
                .disabled(draftReplacementTermInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }

            Text(verbatim: localizedDictionaryTermDialog("Optional phrases that should resolve to this term."))
                .font(.caption)
                .foregroundStyle(.secondary)

            if draftReplacementTerms.isEmpty {
                Text(verbatim: localizedDictionaryTermDialog("No replacement match terms."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                DictionaryEditableTagList(values: draftReplacementTerms) { value in
                    removeDraftReplacementTerm(value)
                }
            }
        }
    }

    private var replacementTermCount: Int {
        draftReplacementTerms.count
    }

    private var trimmedDictionaryTerms: [String] {
        switch dialog {
        case .create:
            if mode == .replacement {
                let term = draftTerm.trimmingCharacters(in: .whitespacesAndNewlines)
                return term.isEmpty ? [] : [term]
            }
            return draftTerm
                .split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        case .edit:
            let term = draftTerm.trimmingCharacters(in: .whitespacesAndNewlines)
            return term.isEmpty ? [] : [term]
        }
    }

    private func save() {
        do {
            let terms = trimmedDictionaryTerms
            guard !terms.isEmpty else {
                errorMessage = AppLocalization.localizedString("Dictionary term cannot be empty.")
                return
            }
            if mode == .replacement && draftReplacementTerms.isEmpty {
                errorMessage = AppLocalization.localizedString("Replacement match term cannot be empty.")
                return
            }
            try onSave(terms, draftReplacementTerms, selectedCategoryID, selectedGroupID)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private static func groupOptionID(for groupID: UUID?) -> String {
        groupID?.uuidString ?? globalGroupOptionID
    }

    private func addDraftReplacementTerm() {
        let display = draftReplacementTermInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = DictionaryStore.normalizeTerm(display)
        guard !display.isEmpty, !normalized.isEmpty else {
            errorMessage = AppLocalization.localizedString("Replacement match term cannot be empty.")
            return
        }

        if trimmedDictionaryTerms.contains(where: { DictionaryStore.normalizeTerm($0) == normalized }) {
            errorMessage = AppLocalization.localizedString("Replacement match term cannot be the same as the dictionary term.")
            return
        }

        if draftReplacementTerms.contains(where: { DictionaryStore.normalizeTerm($0) == normalized }) {
            draftReplacementTermInput = ""
            return
        }

        draftReplacementTerms.append(display)
        draftReplacementTerms.sort { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
        draftReplacementTermInput = ""
        errorMessage = nil
    }

    private func removeDraftReplacementTerm(_ value: String) {
        let normalized = DictionaryStore.normalizeTerm(value)
        draftReplacementTerms.removeAll { DictionaryStore.normalizeTerm($0) == normalized }
        errorMessage = nil
    }
}
