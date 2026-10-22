import SwiftUI
import VoxtMobileKit

struct PersonalStyleView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var enabled = false
    @State private var records: [PersonalStyleStore.Record] = []
    @State private var confirmClear = false
    private let store = PersonalStyleStore(defaults: VoxtPreferences.shared)
    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        NavigationStack {
            List {
                Section {
                    Toggle(VoxtL10n.shared.string("Learn personal style"), isOn: Binding(
                        get: { enabled }, set: { store.isEnabled = $0; enabled = store.isEnabled }
                    ))
                    .accessibilityIdentifier("learnPersonalStyle")
                    Text(VoxtL10n.shared.string("After you rephrase a dictation, the keyboard can offer to save the change. Review both versions and keep it only if the meaning is unchanged."))
                        .font(.subheadline).foregroundStyle(.secondary)
                    Text(VoxtL10n.shared.string("Saved examples go to your Mac’s selected text model for Automatic dictation, editing and writing. Translation and other styles do not use them. Turning this off keeps your examples here."))
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                Section(VoxtL10n.shared.string("Saved examples")) {
                    if records.isEmpty {
                        Text(VoxtL10n.shared.string("No saved examples yet."))
                            .foregroundStyle(.secondary)
                    }
                    ForEach(records) { record in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(VoxtL10n.shared.string("Before")).font(.caption).foregroundStyle(.secondary)
                            Text(record.example.before)
                            Text(VoxtL10n.shared.string("After")).font(.caption).foregroundStyle(.secondary)
                            Text(record.example.after).accessibilityIdentifier("savedStyleExample")
                        }
                        .padding(.vertical, 6)
                    }
                    .onDelete { offsets in
                        for offset in offsets { store.remove(records[offset].id) }
                        records = store.records
                    }
                }
                if !records.isEmpty {
                    Button(VoxtL10n.shared.string("Clear style examples"), role: .destructive) { confirmClear = true }
                }
            }
            .scrollContentBackground(.hidden)
            .background(VoxtTheme.Colors.screenBackground)
            .navigationTitle(VoxtL10n.shared.string("Personal style"))
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(VoxtL10n.shared.string("Done")) { dismiss() }
                        .accessibilityIdentifier("personalStyleDone")
                }
            }
            .confirmationDialog(VoxtL10n.shared.string("Clear all saved style examples?"), isPresented: $confirmClear, titleVisibility: .visible) {
                Button(VoxtL10n.shared.string("Clear style examples"), role: .destructive) { store.clear(); records = store.records }
                Button(VoxtL10n.shared.string("Cancel"), role: .cancel) { }
            }
            .onAppear { enabled = store.isEnabled; records = store.records }
        }
    }
}
