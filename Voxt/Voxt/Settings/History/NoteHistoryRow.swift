import SwiftUI
import AppKit
import UniformTypeIdentifiers

private struct NoteHistoryFooterActionIcon: View {
    enum Kind {
        case copy
        case edit
        case more
        case confirm
    }

    let kind: Kind
    var forcedHover: Bool? = nil
    @State private var isHovered = false

    private var showsHover: Bool {
        forcedHover ?? isHovered
    }

    var body: some View {
        icon
            .frame(width: 14, height: 14)
            .frame(width: 22, height: 22)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Color.primary.opacity(showsHover ? 0.07 : 0))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(
                        showsHover ? SettingsUIStyle.subtleBorderColor : .clear,
                        lineWidth: 1
                    )
            )
            .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .onHover { isHovered = $0 }
            .animation(.easeOut(duration: 0.12), value: showsHover)
    }

    @ViewBuilder
    private var icon: some View {
        switch kind {
        case .copy:
            AppSVGIcon(kind: .copy, size: 14)
        case .edit:
            AppSVGIcon(kind: .edit, size: 14)
        case .more:
            AppSVGIcon(kind: .more, size: 14)
        case .confirm:
            Image(systemName: "checkmark")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.secondary)
        }
    }
}

private struct NoteHistoryFooterMenuButton<MenuContent: View>: View {
    let menuContent: MenuContent
    @State private var isHovered = false

    init(@ViewBuilder menuContent: () -> MenuContent) {
        self.menuContent = menuContent()
    }

    var body: some View {
        Menu {
            menuContent
        } label: {
            Color.clear
                .frame(width: 22, height: 22)
                .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .frame(width: 22, height: 22)
        .overlay {
            NoteHistoryFooterActionIcon(kind: .more, forcedHover: isHovered)
                .allowsHitTesting(false)
        }
        .onHover { isHovered = $0 }
        .help(AppLocalization.localizedString("Note actions"))
    }
}

enum NoteHistoryRowLayout: Equatable {
    case list
    case linearCard
}

struct NoteHistoryRow: View {
    @State private var isHovered = false
    @State private var isRenaming = false
    @State private var isDetailPresented = false
    @State private var draftTitle = ""
    @State private var isEditingDetail = false
    @State private var detailTitle = ""
    @State private var detailContent = ""
    @FocusState private var isRenameFocused: Bool

    let item: VoxtNoteItem
    let layout: NoteHistoryRowLayout
    let contentLineLimit: Int
    let fixedHeight: CGFloat?
    let onCopy: () -> Void
    let onDoubleClick: () -> Void
    let onSetStatus: (VoxtNoteStatus) -> Void
    let onSetPriority: (VoxtNotePriority) -> Void
    let onRename: (String) -> Void
    let onUpdateDetails: (String, String) -> Bool
    let onReorder: (UUID) -> Bool
    let onDelete: () -> Void

    var body: some View {
        rowContent
            .padding(layout == .linearCard ? 10 : 0)
            .padding(.horizontal, layout == .list ? 9.5 : 0)
            .padding(.vertical, layout == .list ? 4 : 0)
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(height: fixedHeight)
            .background(
                RoundedRectangle(cornerRadius: HistoryRowStyle.cornerRadius, style: .continuous)
                    .fill(rowFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: HistoryRowStyle.cornerRadius, style: .continuous)
                    .strokeBorder(isHovered ? HistoryRowStyle.hoverBorderColor : HistoryRowStyle.borderColor, lineWidth: 1)
            )
            .contentShape(Rectangle())
            .onTapGesture(count: 2) {
                guard !isRenaming else { return }
                onDoubleClick()
            }
            .help(item.text)
            .onDrag {
                VoxtNoteDragPayload(text: item.text, noteID: item.id).itemProvider()
            } preview: {
                dragPreview
            }
            .onDrop(of: [UTType.utf8PlainText], isTargeted: nil, perform: acceptDrop)
            .onHover { hovering in
                withAnimation(.easeInOut(duration: 0.12)) {
                    isHovered = hovering
                }
            }
            .onChange(of: item.id) { _, _ in
                isRenaming = false
                isDetailPresented = false
                isEditingDetail = false
                draftTitle = ""
                detailTitle = ""
                detailContent = ""
            }
            .popover(isPresented: $isDetailPresented, arrowEdge: .trailing) {
                noteDetail
            }
    }

    @ViewBuilder
    private var rowContent: some View {
        switch layout {
        case .list:
            HStack(alignment: .center, spacing: 10) {
                noteBodyContent(showTimeInTitle: true)
                trailingAction
            }
        case .linearCard:
            VStack(alignment: .leading, spacing: 8) {
                noteBodyContent(showTimeInTitle: false)
                linearFooter
            }
        }
    }

    private func noteBodyContent(showTimeInTitle: Bool) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            if isRenaming {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    TextField(AppLocalization.localizedString("Note title"), text: $draftTitle)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13, weight: .semibold))
                        .focused($isRenameFocused)
                        .onSubmit(commitRename)
                        .onExitCommand(perform: cancelRename)

                    if showTimeInTitle {
                        noteTimeLabel
                    }
                }

                noteContentPreview
            } else {
                VStack(alignment: .leading, spacing: 5) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(item.title)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(item.status == .done ? .secondary : .primary)
                            .strikethrough(item.status == .done, color: .secondary)
                            .lineLimit(layout == .linearCard ? 2 : 1)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        if showTimeInTitle {
                            noteTimeLabel
                        }
                    }

                    noteContentPreview
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .onTapGesture(perform: showDetail)
                .accessibilityAddTraits(.isButton)
                .accessibilityHint(AppLocalization.localizedString("Open note details"))
            }

            if item.priority != .none {
                metadataChip(item.priority.title, color: priorityColor)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var rowFillColor: Color {
        layout == .linearCard ? HistoryRowStyle.linearCardFillColor : HistoryRowStyle.fillColor
    }

    private var linearFooter: some View {
        HStack(alignment: .center, spacing: 8) {
            noteTimeLabel

            Spacer(minLength: 8)

            HStack(spacing: 4) {
                Button(action: onCopy) {
                    NoteHistoryFooterActionIcon(kind: .copy)
                }
                .buttonStyle(.plain)
                .help(AppLocalization.localizedString("Copy"))

                if isRenaming {
                    Button(action: commitRename) {
                        NoteHistoryFooterActionIcon(kind: .confirm)
                    }
                    .buttonStyle(.plain)
                    .help(AppLocalization.localizedString("Save title"))
                } else {
                    Button(action: showEditableDetail) {
                        NoteHistoryFooterActionIcon(kind: .edit)
                    }
                    .buttonStyle(.plain)
                    .help(AppLocalization.localizedString("Edit"))
                }

                NoteHistoryFooterMenuButton {
                    linearMoreActions
                }
            }
        }
    }

    private var noteContentPreview: some View {
        Text(item.text)
            .font(.system(size: 11.5))
            .foregroundStyle(.secondary)
            .lineLimit(contentLineLimit)
            .truncationMode(.tail)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
    }

    private var noteTimeLabel: some View {
        Text(timeText)
            .font(.system(size: 10))
            .foregroundStyle(.tertiary)
            .fixedSize(horizontal: true, vertical: false)
    }

    @ViewBuilder
    private var trailingAction: some View {
        if isRenaming {
            Button(action: commitRename) {
                Image(systemName: "checkmark")
                    .font(.system(size: 9, weight: .bold))
                    .frame(width: 24, height: 24)
                    .foregroundStyle(Color.accentColor)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(AppLocalization.localizedString("Save title"))
        } else {
            AppSVGMenuButton(icon: .more) {
                noteActions
            }
            .opacity(isHovered ? 1 : 0.38)
            .help(AppLocalization.localizedString("Note actions"))
        }
    }

    @ViewBuilder
    private var noteActions: some View {
        Button {
            showDetail()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("View note…"), icon: .viewDetails)
        }

        Button {
            beginRenaming()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Edit title…"), icon: .edit)
        }

        Button {
            onCopy()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Copy"), icon: .copy)
        }

        Menu(AppLocalization.localizedString("Priority")) {
            ForEach(VoxtNotePriority.allCases.reversed(), id: \.self) { priority in
                Button {
                    onSetPriority(priority)
                } label: {
                    menuLabel(priority.title, selected: priority == item.priority)
                }
            }
        }

        Menu(AppLocalization.localizedString("Move to")) {
            ForEach(VoxtNoteStatus.moveMenuOrder) { status in
                Button {
                    onSetStatus(status)
                } label: {
                    menuLabel(status.title, selected: status == item.status)
                }
            }
        }

        Divider()

        Button(role: .destructive) {
            onDelete()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Delete"), icon: .delete, color: .red)
        }
    }

    @ViewBuilder
    private var linearMoreActions: some View {
        Button {
            showDetail()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("View note…"), icon: .viewDetails)
        }

        Button {
            beginRenaming()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Edit title…"), icon: .edit)
        }

        Menu(AppLocalization.localizedString("Priority")) {
            ForEach(VoxtNotePriority.allCases.reversed(), id: \.self) { priority in
                Button {
                    onSetPriority(priority)
                } label: {
                    menuLabel(priority.title, selected: priority == item.priority)
                }
            }
        }

        Menu(AppLocalization.localizedString("Move to")) {
            ForEach(VoxtNoteStatus.moveMenuOrder) { status in
                Button {
                    onSetStatus(status)
                } label: {
                    menuLabel(status.title, selected: status == item.status)
                }
            }
        }

        Divider()

        Button(role: .destructive) {
            onDelete()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Delete"), icon: .delete, color: .red)
        }
    }

    private var dragPreview: some View {
        HStack(spacing: 7) {
            Image(systemName: statusSystemImage)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(statusColor)
            Text(item.title)
                .font(.system(size: 12, weight: .medium))
                .lineLimit(2)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
    }

    private func acceptDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let noteID = providers
            .compactMap(\.suggestedName)
            .compactMap(UUID.init(uuidString:))
            .first
        else {
            return false
        }
        return onReorder(noteID)
    }

    private var noteDetail: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 6) {
                    if isEditingDetail {
                        TextField(AppLocalization.localizedString("Note title"), text: $detailTitle)
                            .textFieldStyle(.plain)
                            .font(.headline)
                            .settingsFieldSurface(minHeight: 32)
                    } else {
                        Text(item.title)
                            .font(.headline)
                            .textSelection(.enabled)
                    }
                    HStack(spacing: 6) {
                        metadataChip(item.status.title, color: statusColor)
                        if item.priority != .none {
                            metadataChip(item.priority.title, color: priorityColor)
                        }
                    }
                }
                Spacer(minLength: 12)
                Button {
                    isDetailPresented = false
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .accessibilityLabel(AppLocalization.localizedString("Close"))
            }

            Divider()

            if isEditingDetail {
                TextEditor(text: $detailContent)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(
                        SettingsUIStyle.controlFillColor,
                        in: RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: SettingsUIStyle.controlCornerRadius, style: .continuous)
                            .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                    )
            } else {
                ScrollView {
                    Text(item.text)
                        .font(.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            HStack {
                Text(item.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if isEditingDetail {
                    Button(AppLocalization.localizedString("Cancel"), action: cancelDetailEditing)
                    Button(AppLocalization.localizedString("Save"), action: saveDetailEditing)
                        .disabled(!canSaveDetail)
                } else {
                    Button(AppLocalization.localizedString("Edit"), action: beginDetailEditing)
                    Button(AppLocalization.localizedString("Copy"), action: onCopy)
                }
            }
        }
        .padding(16)
        .frame(width: 400, height: 320)
    }

    private var statusSystemImage: String {
        switch item.status {
        case .todo: return "circle"
        case .inProgress: return "circle.lefthalf.filled"
        case .done: return "checkmark"
        case .backlog: return "clock"
        }
    }

    private var statusColor: Color {
        switch item.status {
        case .todo: return .blue
        case .inProgress: return .orange
        case .done: return .green
        case .backlog: return .secondary
        }
    }

    private var priorityColor: Color {
        switch item.priority {
        case .none: return .secondary
        case .low: return .blue
        case .medium: return .orange
        case .high: return .red
        }
    }

    private var timeText: String {
        RelativeNoteTimestampFormatter.historyCardTimestamp(for: item.updatedAt)
    }

    private func metadataChip(_ title: String, color: Color) -> some View {
        Text(title)
            .font(.system(size: 9.5, weight: .medium))
            .foregroundStyle(color)
            .padding(.horizontal, 5)
            .frame(height: 17)
            .background(color.opacity(0.08), in: Capsule(style: .continuous))
    }

    private func menuLabel(_ title: String, selected: Bool) -> some View {
        HStack {
            Text(title)
            if selected {
                Image(systemName: "checkmark")
            }
        }
    }

    private func beginRenaming() {
        draftTitle = item.title
        isRenaming = true
        DispatchQueue.main.async { isRenameFocused = true }
    }

    private func showDetail() {
        guard !isRenaming else { return }
        detailTitle = item.title
        detailContent = item.text
        isEditingDetail = false
        isDetailPresented = true
    }

    private func showEditableDetail() {
        guard !isRenaming else { return }
        detailTitle = item.title
        detailContent = item.text
        isEditingDetail = true
        isDetailPresented = true
    }

    private var canSaveDetail: Bool {
        !detailTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !detailContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func beginDetailEditing() {
        detailTitle = item.title
        detailContent = item.text
        isEditingDetail = true
    }

    private func cancelDetailEditing() {
        detailTitle = item.title
        detailContent = item.text
        isEditingDetail = false
    }

    private func saveDetailEditing() {
        guard canSaveDetail,
              onUpdateDetails(detailTitle, detailContent)
        else {
            return
        }
        isEditingDetail = false
    }

    private func cancelRename() {
        isRenaming = false
        draftTitle = ""
    }

    private func commitRename() {
        let title = draftTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !title.isEmpty else {
            cancelRename()
            return
        }
        onRename(title)
        cancelRename()
    }
}
