import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct HistoryNoteStatusFilterSelect: View {
    @Binding var selection: Set<VoxtNoteStatus>
    @State private var isPresented = false

    private let statuses: [VoxtNoteStatus] = [.todo, .inProgress, .done, .backlog]

    var body: some View {
        SettingsSelectionButton(width: 130, height: 28, allowsCompactWidth: true) {
            isPresented = true
        } label: {
            Text(selectionSummary)
                .lineLimit(1)
        }
        .popover(isPresented: $isPresented, arrowEdge: .top) {
            VStack(alignment: .leading, spacing: 4) {
                filterRow(
                    title: AppLocalization.localizedString("All statuses"),
                    isSelected: selection == Set(statuses)
                ) {
                    selection = Set(statuses)
                }

                Divider()
                    .padding(.vertical, 2)

                ForEach(statuses) { status in
                    filterRow(
                        title: status.title,
                        isSelected: selection.contains(status)
                    ) {
                        selection = HistorySettingsData.toggledNoteStatuses(selection, status: status)
                    }
                }
            }
            .padding(8)
            .frame(width: 190)
        }
        .accessibilityLabel(AppLocalization.localizedString("Status"))
    }

    private var selectionSummary: String {
        if selection == Set(statuses) {
            return AppLocalization.localizedString("All statuses")
        }
        if selection.count == 1, let status = selection.first {
            return status.title
        }
        return AppLocalization.format("%d statuses", selection.count)
    }

    private func filterRow(
        title: String,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Text(title)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Group {
                    if isSelected {
                        Image(systemName: "checkmark")
                    } else {
                        Color.clear
                    }
                }
                .font(.system(size: 10, weight: .semibold))
                .frame(width: 12, height: 12)
            }
            .font(.system(size: 12, weight: isSelected ? .semibold : .regular))
            .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
            .padding(.horizontal, 8)
            .frame(height: 28)
            .background(
                Color.accentColor.opacity(isSelected ? 0.09 : 0),
                in: RoundedRectangle(cornerRadius: 7, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

enum HistoryNoteViewMode: String, CaseIterable, Identifiable {
    case list
    case linearCard

    var id: String { rawValue }

    var title: String {
        switch self {
        case .list:
            return AppLocalization.localizedString("List View")
        case .linearCard:
            return AppLocalization.localizedString("Linear Card View")
        }
    }
}

struct HistoryNoteViewPicker: View {
    @Binding var selection: HistoryNoteViewMode

    var body: some View {
        Button {
            selection = selection == .list ? .linearCard : .list
        } label: {
            AppSVGIcon(
                kind: selection == .list ? .listView : .linearView,
                size: 16
            )
        }
        .buttonStyle(SettingsCompactIconButtonStyle())
        .help(nextModeTitle)
        .accessibilityLabel(AppLocalization.localizedString("Note View"))
        .accessibilityValue(selection.title)
    }

    private var nextModeTitle: String {
        selection == .list
            ? AppLocalization.localizedString("Linear Card View")
            : AppLocalization.localizedString("List View")
    }
}


struct NoteHistorySectionHeader: View {
    let status: VoxtNoteStatus
    let count: Int

    var body: some View {
        HStack(spacing: 7) {
            Group {
                if status == .backlog {
                    VoxtNoteStatusMark(status: .backlog, priority: .none, size: 12)
                } else {
                    Image(systemName: statusSystemImage)
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(statusColor)
                }
            }
                .frame(width: 18, height: 18)
                .background(statusColor.opacity(0.10), in: Circle())

            Text("\(status.title) · \(count)")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .contentTransition(.numericText())

            Spacer(minLength: 0)
        }
        .frame(height: 28)
        .padding(.horizontal, 4)
        .accessibilityElement(children: .combine)
    }

    private var statusSystemImage: String {
        switch status {
        case .todo: return "circle"
        case .inProgress: return "circle.lefthalf.filled"
        case .done: return "checkmark"
        case .backlog: return "clock"
        }
    }

    private var statusColor: Color {
        switch status {
        case .todo: return .blue
        case .inProgress: return .orange
        case .done: return .green
        case .backlog: return .secondary
        }
    }
}

struct NoteHistoryMoreButton: View {
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Image(systemName: "ellipsis")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 28, height: 28)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.primary.opacity(isHovered ? 0.07 : 0))
                )
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help(AppLocalization.localizedString("More"))
        .accessibilityLabel(AppLocalization.localizedString("More"))
    }
}
