// GeneralLanguageSelection.swift
// Provides General Language Selection for settings screens.

import SwiftUI

struct UserMainLanguageSelectionSheet: View {
    let localeIdentifier: String
    let onSave: ([String]) -> Void
    var cornerRadius: CGFloat
    var onClose: (() -> Void)?

    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""
    @State private var draftCodes: [String]

    init(
        selectedCodes: [String],
        localeIdentifier: String,
        cornerRadius: CGFloat = SettingsUIStyle.dialogCornerRadius,
        onClose: (() -> Void)? = nil,
        onSave: @escaping ([String]) -> Void
    ) {
        self.localeIdentifier = localeIdentifier
        self.onSave = onSave
        self.cornerRadius = cornerRadius
        self.onClose = onClose
        _draftCodes = State(initialValue: UserMainLanguageOption.sanitizedSelection(selectedCodes))
    }

    private var locale: Locale {
        Locale(identifier: localeIdentifier)
    }

    private var filteredOptions: [UserMainLanguageOption] {
        UserMainLanguageOption.all
            .filter { $0.matches(searchText, locale: locale) }
            .sorted { lhs, rhs in
                let lhsIndex = draftCodes.firstIndex(of: lhs.code)
                let rhsIndex = draftCodes.firstIndex(of: rhs.code)
                switch (lhsIndex, rhsIndex) {
                case let (left?, right?):
                    return left < right
                case (.some, .none):
                    return true
                case (.none, .some):
                    return false
                case (.none, .none):
                    return lhs.title(locale: locale).localizedCaseInsensitiveCompare(rhs.title(locale: locale)) == .orderedAscending
                }
            }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(AppLocalization.localizedString("Select User Languages"))
                .font(.title3.weight(.semibold))

            TextField(AppLocalization.localizedString("Search languages"), text: $searchText)
                .textFieldStyle(.plain)
                .settingsFieldSurface(minHeight: 34)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(filteredOptions) { option in
                        UserMainLanguageRow(
                            option: option,
                            isSelected: draftCodes.contains(option.code),
                            isPrimary: draftCodes.first == option.code,
                            locale: locale,
                            onToggle: { toggle(option) },
                            onSetPrimary: { setPrimary(option) }
                        )
                    }
                }
            }
            .frame(minHeight: 320)

            if filteredOptions.isEmpty {
                Text(AppLocalization.localizedString("No languages found."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Cancel")) {
                    close()
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(AppLocalization.localizedString("Save")) {
                    onSave(draftCodes)
                    close()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
                .disabled(draftCodes.isEmpty)
            }
        }
        .settingsDialogChrome(width: 460, height: 520, cornerRadius: cornerRadius, onClose: close)
    }

    private func close() {
        if let onClose {
            onClose()
        } else {
            dismiss()
        }
    }

    private func toggle(_ option: UserMainLanguageOption) {
        if let index = draftCodes.firstIndex(of: option.code) {
            draftCodes.remove(at: index)
            if draftCodes.isEmpty {
                draftCodes = [option.code]
            }
            return
        }

        draftCodes.append(option.code)
    }

    private func setPrimary(_ option: UserMainLanguageOption) {
        guard let index = draftCodes.firstIndex(of: option.code) else { return }
        let code = draftCodes.remove(at: index)
        draftCodes.insert(code, at: 0)
    }
}

private struct UserMainLanguageRow: View {
    let option: UserMainLanguageOption
    let isSelected: Bool
    let isPrimary: Bool
    let locale: Locale
    let onToggle: () -> Void
    let onSetPrimary: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Button(action: onToggle) {
                HStack(spacing: 10) {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(isSelected ? Color.accentColor : .secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(option.title(locale: locale))
                        Text(option.promptName)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isSelected {
                Button(action: onSetPrimary) {
                    Image(systemName: isPrimary ? "star.fill" : "star")
                        .foregroundStyle(isPrimary ? Color.yellow : .secondary)
                }
                .buttonStyle(.plain)
                .help(AppLocalization.localizedString("Set as primary language"))
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(SettingsUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
    }
}
