import SwiftUI
import VoxtMobileKit

struct DictionaryView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var entries: [PersonalDictionary.Entry] = []
    @State private var query = ""
    @State private var learnsCorrections = false
    @State private var records: [CorrectionLearningStore.Record] = []
    @State private var editingID: UUID?
    @State private var word = ""
    @State private var editing = false
    @State private var errorMessage: String?

    private var filtered: [PersonalDictionary.Entry] {
        entries.filter { query.isEmpty || $0.term.localizedStandardContains(query) }
    }

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        NavigationStack {
            List {
                if query.isEmpty {
                    Section {
                        Text(VoxtL10n.shared.string("Add names and terms you use. Your Mac receives these spellings with each recording."))
                            .foregroundStyle(.secondary)
                    }
                    if model.macServes(.corrections) {
                        Section {
                            Toggle(VoxtL10n.shared.string("Learn corrected words"), isOn: $learnsCorrections)
                                .accessibilityIdentifier("learnCorrectedWords")
                                .onChange(of: learnsCorrections) { _, enabled in model.learning.isEnabled = enabled }
                        } footer: {
                            Text(VoxtL10n.shared.string("After dictation, completed corrections go to your Mac’s text model to learn names and spellings. Review or undo learned words below."))
                        }
                    }
                    if !records.isEmpty {
                        Section(VoxtL10n.shared.string("Learned words")) {
                            ForEach(records) { record in
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(record.correction.replacement)
                                        Text(VoxtL10n.shared.format("From “%@”", record.correction.original))
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    Button(VoxtL10n.shared.string("Undo")) { model.learning.undo(record.id); reload() }
                                        .buttonStyle(.borderless)
                                        .accessibilityIdentifier("undoLearned-\(record.correction.replacement)")
                                }
                            }
                        }
                    }
                }
                if let errorMessage { Text(errorMessage).foregroundStyle(.red) }
                ForEach(filtered) { entry in
                    Button(entry.term) {
                        editingID = entry.id
                        word = entry.term
                        editing = true
                    }
                    .foregroundStyle(.primary)
                    .swipeActions {
                        Button(VoxtL10n.shared.string("Delete"), role: .destructive) {
                            model.dictionary.delete(entry.id)
                            reload()
                        }
                    }
                }
                if entries.isEmpty { Text(VoxtL10n.shared.string("Your dictionary is empty.")).foregroundStyle(.secondary) }
            }
            .searchable(text: $query, prompt: VoxtL10n.shared.string("Search words"))
            .navigationTitle(VoxtL10n.shared.string("Personal dictionary"))
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { Button(VoxtL10n.shared.string("Done")) { dismiss() }.accessibilityIdentifier("dictionaryDone") }
                ToolbarItem(placement: .topBarTrailing) {
                    Button(VoxtL10n.shared.string("Add word"), systemImage: "plus") {
                        editingID = nil
                        word = ""
                        editing = true
                    }
                    .accessibilityIdentifier("addDictionaryWord")
                }
            }
            .alert(VoxtL10n.shared.string(editingID == nil ? "Add word" : "Edit word"), isPresented: $editing) {
                TextField(VoxtL10n.shared.string("Word or name"), text: $word)
                    .autocorrectionDisabled()
                Button(VoxtL10n.shared.string("Cancel"), role: .cancel) {}
                Button(VoxtL10n.shared.string("Save")) {
                    do {
                        if let editingID { try model.dictionary.rename(editingID, to: word) }
                        else { try model.dictionary.add(word) }
                        errorMessage = nil
                        reload()
                    } catch { errorMessage = error.localizedDescription }
                }
            }
            .task { reload() }
        }
        .tint(VoxtTheme.Colors.ink)
    }

    private func reload() {
        entries = model.dictionary.entries
        records = model.learning.records
        learnsCorrections = model.learning.isEnabled
    }
}
