// RemindersListSelectorDialog.swift
// Provides Reminders List Selector Dialog for settings screens.

import SwiftUI

struct RemindersListSelectorDialog: View {
    @Environment(\.dismiss) private var dismiss

    let title: String
    let entries: [RemindersListDescriptor]
    let selectedIdentifier: String
    let onSelect: (RemindersListDescriptor) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 12) {
                Text(title)
                    .font(.title3.weight(.semibold))
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if entries.isEmpty {
                        Text(AppLocalization.localizedString("No writable reminder lists available."))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.vertical, 24)
                    } else {
                        ForEach(entries) { entry in
                            Button {
                                onSelect(entry)
                                dismiss()
                            } label: {
                                HStack(spacing: 10) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(entry.title)
                                            .font(.subheadline.weight(.semibold))
                                            .foregroundStyle(.primary)
                                        if !entry.sourceTitle.isEmpty, entry.sourceTitle != entry.title {
                                            Text(entry.sourceTitle)
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                        }
                                    }

                                    Spacer(minLength: 0)

                                    if entry.identifier == selectedIdentifier {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(Color.accentColor)
                                    }
                                }
                                .padding(14)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                                        .fill(SettingsUIStyle.groupedFillColor)
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                                        .stroke(
                                            entry.identifier == selectedIdentifier
                                                ? Color.accentColor.opacity(0.45)
                                                : SettingsUIStyle.subtleBorderColor,
                                            lineWidth: 1
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(.vertical, 2)
            }
            .frame(height: 320)
        }
        .settingsDialogChrome(width: 420, onClose: { dismiss() })
    }
}
