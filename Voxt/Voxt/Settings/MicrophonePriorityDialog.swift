// MicrophonePriorityDialog.swift
// Provides Microphone Priority Dialog for settings screens.

import SwiftUI
import UniformTypeIdentifiers

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct MicrophonePriorityDialog: View {
    enum Mode: Equatable {
        case managePriority
        case selectionOnly
    }

    let state: MicrophoneResolvedState
    let mode: Mode
    let onUseNow: (String) -> Void
    let onAutoSwitchChanged: (Bool) -> Void
    let onReorderPriority: ([String]) -> Void
    var cornerRadius: CGFloat = SettingsUIStyle.dialogCornerRadius
    var onClose: (() -> Void)?

    @Environment(\.dismiss) private var dismiss
    @State private var orderedEntries: [MicrophoneDisplayEntry]
    @State private var draggedUID: String?

    init(
        state: MicrophoneResolvedState,
        mode: Mode = .managePriority,
        onUseNow: @escaping (String) -> Void,
        onAutoSwitchChanged: @escaping (Bool) -> Void = { _ in },
        onReorderPriority: @escaping ([String]) -> Void = { _ in },
        cornerRadius: CGFloat = SettingsUIStyle.dialogCornerRadius,
        onClose: (() -> Void)? = nil
    ) {
        self.state = state
        self.mode = mode
        self.onUseNow = onUseNow
        self.onAutoSwitchChanged = onAutoSwitchChanged
        self.onReorderPriority = onReorderPriority
        self.cornerRadius = cornerRadius
        self.onClose = onClose
        _orderedEntries = State(initialValue: state.entries)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            if displayedEntries.isEmpty {
                Text(localized("No available microphone devices"))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(
                        maxWidth: .infinity,
                        minHeight: mode == .selectionOnly ? 120 : 180,
                        alignment: .center
                    )
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(Array(displayedEntries.enumerated()), id: \.element.uid) { index, entry in
                            if mode == .selectionOnly {
                                MicrophoneSelectionListRow(
                                    entry: entry,
                                    onSelect: { onUseNow(entry.uid) }
                                )
                            } else {
                                MicrophonePriorityListRow(
                                    entry: entry,
                                    index: index,
                                    onBeginDrag: { beginDrag(for: entry.uid) },
                                    onMoveToTop: { moveEntryToTop(uid: entry.uid) },
                                    onUse: { onUseNow(entry.uid) }
                                )
                                .onDrop(
                                    of: [UTType.text.identifier],
                                    delegate: MicrophonePriorityRowDropDelegate(
                                        targetUID: entry.uid,
                                        entries: $orderedEntries,
                                        draggedUID: $draggedUID,
                                        onReorder: persistReorder
                                    )
                                )
                            }
                        }
                    }
                    .padding(.vertical, 2)
                }
                .frame(height: mode == .selectionOnly ? selectionListHeight : nil)
                .frame(
                    minHeight: mode == .managePriority ? 220 : nil,
                    maxHeight: mode == .managePriority ? 250 : nil
                )
            }

            if mode == .selectionOnly {
                Spacer(minLength: 0)
            }

            SettingsDialogActionRow {
                if mode == .managePriority {
                    Toggle(localized("Auto Switch"), isOn: autoSwitchBinding)
                        .toggleStyle(.switch)
                        .controlSize(.small)
                }
            } trailing: {
                Button(localized("Done")) {
                    close()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(
            width: mode == .selectionOnly ? 460 : 560,
            height: mode == .selectionOnly ? 420 : 380,
            cornerRadius: cornerRadius,
            onClose: close
        )
        .onChange(of: state.entries) { _, newValue in
            orderedEntries = newValue
        }
    }

    private func close() {
        if let onClose {
            onClose()
        } else {
            dismiss()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(localized(mode == .selectionOnly ? "Select Microphone" : "Microphone Priority"))
                .font(.headline)
            Text(state.activeDevice?.name ?? AppLocalization.localizedString("No available microphone devices"))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Text(localized(
                mode == .selectionOnly
                    ? "Choose the microphone Voxt should use."
                    : "Drag to set the preferred microphone order."
            ))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var displayedEntries: [MicrophoneDisplayEntry] {
        if mode == .selectionOnly {
            return orderedEntries.filter(\.isAvailable)
        }
        return orderedEntries
    }

    private var selectionListHeight: CGFloat {
        let rowHeight: CGFloat = 44
        let rowSpacing: CGFloat = 8
        let verticalPadding: CGFloat = 4
        let rowCount = CGFloat(displayedEntries.count)
        let spacingCount = CGFloat(max(displayedEntries.count - 1, 0))
        return min(220, rowCount * rowHeight + spacingCount * rowSpacing + verticalPadding)
    }

    private var autoSwitchBinding: Binding<Bool> {
        Binding(
            get: { state.autoSwitchEnabled },
            set: { onAutoSwitchChanged($0) }
        )
    }

    private func moveEntryToTop(uid: String) {
        guard let sourceIndex = orderedEntries.firstIndex(where: { $0.uid == uid }) else { return }
        var reordered = orderedEntries
        let moved = reordered.remove(at: sourceIndex)
        reordered.insert(moved, at: 0)
        orderedEntries = reordered
        persistReorder(reordered.map(\.uid))
    }

    private func persistReorder(_ orderedUIDs: [String]) {
        onReorderPriority(orderedUIDs)
    }

    private func beginDrag(for uid: String) -> NSItemProvider {
        draggedUID = uid
        return NSItemProvider(object: uid as NSString)
    }
}
