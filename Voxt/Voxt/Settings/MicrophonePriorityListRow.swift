// MicrophonePriorityListRow.swift
// Provides Microphone Priority List Row for settings screens.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct MicrophoneSelectionListRow: View {
    let entry: MicrophoneDisplayEntry
    let onSelect: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 10) {
                TranscriptionModeIconView(
                    color: entry.isActive ? Color.accentColor : Color.primary
                )
                .frame(width: 16, height: 16)
                .accessibilityHidden(true)

                Text(entry.name)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Spacer(minLength: 8)

                if entry.isActive {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Color.accentColor)
                }
            }
            .padding(.horizontal, 12)
            .frame(minHeight: 44)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(backgroundColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .accessibilityLabel(entry.name)
        .accessibilityValue(entry.isActive ? localized("Current Microphone") : "")
    }

    private var backgroundColor: Color {
        if entry.isActive {
            return Color.accentColor.opacity(0.12)
        }
        if isHovered {
            return Color.primary.opacity(0.07)
        }
        return Color(nsColor: .controlBackgroundColor)
    }

    private var borderColor: Color {
        entry.isActive ? Color.accentColor.opacity(0.35) : Color.primary.opacity(0.08)
    }
}

struct MicrophonePriorityListRow: View {
    let entry: MicrophoneDisplayEntry
    let index: Int
    let onBeginDrag: () -> NSItemProvider
    let onMoveToTop: () -> Void
    let onUse: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 18, height: 28)
                .contentShape(Rectangle())
                .onDrag(onBeginDrag)

            VStack(alignment: .leading, spacing: 4) {
                Text(entry.name)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(entry.isAvailable ? .primary : .secondary)

                Text(LocalizedStringKey(entry.status.titleKey))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(statusTint)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(
                        Capsule(style: .continuous)
                            .fill(statusTint.opacity(0.12))
                    )
            }

            Spacer(minLength: 8)

            if index > 0 {
                Button {
                    onMoveToTop()
                } label: {
                    Image(systemName: "arrow.up.to.line.compact")
                        .font(.system(size: 12, weight: .semibold))
                }
                .buttonStyle(.borderless)
                .help(AppLocalization.localizedString("Move to Top"))
            }

            if entry.isAvailable && !entry.isActive {
                Button(localized("Use"), action: onUse)
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(backgroundColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(borderColor, lineWidth: 1)
        )
        .opacity(entry.isAvailable ? 1.0 : 0.74)
    }

    private var backgroundColor: Color {
        if entry.isActive {
            return Color.accentColor.opacity(0.12)
        }
        if entry.isAvailable {
            return Color(nsColor: .controlBackgroundColor)
        }
        return Color(nsColor: .controlBackgroundColor).opacity(0.55)
    }

    private var borderColor: Color {
        if entry.isActive {
            return Color.accentColor.opacity(0.35)
        }
        return Color.primary.opacity(0.08)
    }

    private var statusTint: Color {
        switch entry.status {
        case .inUse:
            return .accentColor
        case .available:
            return .secondary
        case .offline:
            return .red
        }
    }
}

struct MicrophonePriorityRowDropDelegate: DropDelegate {
    let targetUID: String
    @Binding var entries: [MicrophoneDisplayEntry]
    @Binding var draggedUID: String?
    let onReorder: ([String]) -> Void

    func dropEntered(info: DropInfo) {
        guard let draggedUID,
              draggedUID != targetUID,
              let fromIndex = entries.firstIndex(where: { $0.uid == draggedUID }),
              let toIndex = entries.firstIndex(where: { $0.uid == targetUID })
        else {
            return
        }

        entries.move(
            fromOffsets: IndexSet(integer: fromIndex),
            toOffset: toIndex > fromIndex ? toIndex + 1 : toIndex
        )
    }

    func performDrop(info: DropInfo) -> Bool {
        draggedUID = nil
        onReorder(entries.map(\.uid))
        return true
    }

    func dropUpdated(info: DropInfo) -> DropProposal? {
        DropProposal(operation: .move)
    }
}
