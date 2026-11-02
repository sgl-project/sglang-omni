// DictionarySettingsComponents.swift
// Provides Dictionary Settings Components for dictionary settings.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

enum DictionaryEntriesTab: String, CaseIterable, Identifiable {
    case hotwords
    case replacements

    var id: String { rawValue }

    var titleKey: String {
        switch self {
        case .hotwords:
            return "Hot Words"
        case .replacements:
            return "Replacement Terms"
        }
    }
}

enum DictionaryTermDialogMode: String {
    case hotword
    case replacement
}

struct DictionaryEntriesTabPicker: View {
    @Binding var selectedTab: DictionaryEntriesTab

    var body: some View {
        HStack(spacing: 2) {
            ForEach(DictionaryEntriesTab.allCases) { tab in
                Button {
                    var transaction = Transaction()
                    transaction.disablesAnimations = true
                    withTransaction(transaction) {
                        selectedTab = tab
                    }
                } label: {
                    Text(LocalizedStringKey(tab.titleKey))
                }
                .buttonStyle(DictionarySegmentedButtonStyle(isSelected: selectedTab == tab))
            }
        }
        .padding(2)
        .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
    }
}

private struct DictionarySegmentedButtonStyle: ButtonStyle {
    let isSelected: Bool

    func makeBody(configuration: Configuration) -> some View {
        DictionarySegmentedButtonBody(configuration: configuration, isSelected: isSelected)
    }
}

private struct DictionarySegmentedButtonBody: View {
    let configuration: DictionarySegmentedButtonStyle.Configuration
    let isSelected: Bool

    @State private var isHovered = false

    var body: some View {
        configuration.label
            .font(.system(size: 11.5, weight: .semibold))
            .foregroundStyle(isSelected || isHovered ? Color.accentColor : Color.secondary)
            .padding(.horizontal, 12)
            .frame(height: 28)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(isSelected || isHovered ? Color.accentColor.opacity(0.14) : .clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(isSelected || isHovered ? Color.accentColor.opacity(0.4) : .clear, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .opacity(configuration.isPressed ? 0.9 : 1)
            .onHover { isHovered = $0 }
    }
}

struct DictionaryRow: View {
    let entry: DictionaryEntry
    let onEdit: () -> Void
    let onDelete: () -> Void

    @State private var isHovering = false
    @State private var isDeleteHovering = false

    var body: some View {
        HStack(spacing: 8) {
            Text(entry.term)
                .font(.system(size: 12.5, weight: .regular))
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.trailing, 28)
        }
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: 32, alignment: .center)
        .contentShape(Rectangle())
        .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
        .brightness(isHovering ? 0.035 : 0)
        .overlay {
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .strokeBorder(
                    Color.accentColor.opacity(isHovering ? 0.42 : 0),
                    lineWidth: 1
                )
        }
        .onHover { hovering in
            isHovering = hovering
        }
        .animation(.easeOut(duration: 0.12), value: isHovering)
        .onTapGesture(perform: onEdit)
        .overlay(alignment: .topTrailing) {
            Button(role: .destructive, action: onDelete) {
                Image(systemName: "trash")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(isDeleteHovering ? Color.red : Color.secondary)
                    .frame(width: 20, height: 20)
                    .background(
                        Circle()
                            .fill(Color.red.opacity(isDeleteHovering ? 0.12 : 0))
                    )
            }
            .buttonStyle(.plain)
            .help(localized("Delete"))
            .onHover { hovering in
                isDeleteHovering = hovering
            }
            .padding(6)
        }
    }
}

enum DictionaryDialog: Identifiable {
    case create(categoryID: UUID?, mode: DictionaryTermDialogMode)
    case edit(DictionaryEntry)

    var id: String {
        switch self {
        case .create(let categoryID, let mode):
            return "create-\(mode.rawValue)-\(categoryID?.uuidString ?? "default")"
        case .edit(let entry):
            return "edit-\(entry.id.uuidString)"
        }
    }

    var mode: DictionaryTermDialogMode {
        switch self {
        case .create(_, let mode):
            return mode
        case .edit(let entry):
            return entry.replacementTerms.isEmpty ? .hotword : .replacement
        }
    }

    var title: String {
        switch self {
        case .create(_, let mode):
            switch mode {
            case .hotword:
                return localized("Create Hot Word")
            case .replacement:
                return localized("Create Replacement Term")
            }
        case .edit(let entry):
            return entry.replacementTerms.isEmpty
                ? localized("Edit Hot Word")
                : localized("Edit Replacement Term")
        }
    }

    var confirmButtonTitle: String {
        switch self {
        case .create:
            return localized("Create")
        case .edit:
            return localized("Save")
        }
    }
}

enum DictionaryCategoryDialog: Identifiable {
    case create
    case edit(DictionaryCategory)

    var id: String {
        switch self {
        case .create:
            return "create-category"
        case .edit(let category):
            return "edit-category-\(category.id.uuidString)"
        }
    }

    var title: String {
        switch self {
        case .create:
            return localized("Create Dictionary Category")
        case .edit:
            return localized("Edit Dictionary Category")
        }
    }

    var confirmButtonTitle: String {
        switch self {
        case .create:
            return localized("Create")
        case .edit:
            return localized("Save")
        }
    }
}

struct DictionaryCategoryDialogView: View {
    let dialog: DictionaryCategoryDialog
    let onCancel: () -> Void
    let onSave: (String) throws -> Void

    @State private var draftName: String
    @State private var errorMessage: String?

    init(
        dialog: DictionaryCategoryDialog,
        onCancel: @escaping () -> Void,
        onSave: @escaping (String) throws -> Void
    ) {
        self.dialog = dialog
        self.onCancel = onCancel
        self.onSave = onSave
        switch dialog {
        case .create:
            _draftName = State(initialValue: "")
        case .edit(let category):
            _draftName = State(initialValue: category.name)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(verbatim: dialog.title)
                .font(.title3.weight(.semibold))

            TextField(
                "",
                text: $draftName,
                prompt: Text(verbatim: localized("Category Name"))
            )
            .textFieldStyle(.plain)
            .settingsFieldSurface()
            .onSubmit(save)

            if let errorMessage, !errorMessage.isEmpty {
                Text(errorMessage)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            SettingsDialogActionRow {
                Button(localized("Cancel")) {
                    onCancel()
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(dialog.confirmButtonTitle) {
                    save()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: 420, onClose: onCancel)
    }

    private func save() {
        do {
            try onSave(draftName)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

struct DictionaryEditableTagList: View {
    let values: [String]
    let onRemove: (String) -> Void

    var body: some View {
        DictionaryFlexibleTagLayout(tags: values) { value in
            HStack(spacing: 6) {
                Text(value)
                    .lineLimit(1)
                    .textSelection(.enabled)

                Button {
                    onRemove(value)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.secondary.opacity(0.12))
            )
        }
    }
}

private struct DictionaryFlexibleTagLayout<Content: View>: View {
    let tags: [String]
    let content: (String) -> Content

    var body: some View {
        GeometryReader { proxy in
            generateContent(in: proxy)
        }
        .frame(minHeight: 10)
    }

    private func generateContent(in proxy: GeometryProxy) -> some View {
        var width = CGFloat.zero
        var height = CGFloat.zero

        return ZStack(alignment: .topLeading) {
            ForEach(tags, id: \.self) { tag in
                content(tag)
                    .padding(.trailing, 8)
                    .padding(.bottom, 8)
                    .alignmentGuide(.leading) { dimension in
                        if abs(width - dimension.width) > proxy.size.width {
                            width = 0
                            height -= dimension.height
                        }
                        let result = width
                        width = tag == tags.last ? 0 : width - dimension.width
                        return result
                    }
                    .alignmentGuide(.top) { _ in
                        let result = height
                        if tag == tags.last {
                            height = 0
                        }
                        return result
                    }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
