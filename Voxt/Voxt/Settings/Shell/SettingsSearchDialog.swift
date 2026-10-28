// SettingsSearchDialog.swift
// Provides Settings Search Dialog for settings shell.

import SwiftUI

struct SettingsSearchDialog: View {
    let title: String
    let placeholder: String
    @Binding var query: String
    @Binding var isPresented: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.title3.weight(.semibold))

            TextField(placeholder, text: $query)
                .textFieldStyle(.plain)
                .settingsFieldSurface(minHeight: 34)

            HStack(spacing: 8) {
                Button(AppLocalization.localizedString("Clear")) {
                    query = ""
                }
                .buttonStyle(SettingsPillButtonStyle())
                .disabled(query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                Spacer()

                Button(AppLocalization.localizedString("Done")) {
                    isPresented = false
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: 420, onClose: { isPresented = false })
    }
}
