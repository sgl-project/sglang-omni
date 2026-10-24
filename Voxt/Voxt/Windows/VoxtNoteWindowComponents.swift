// VoxtNoteWindowComponents.swift
// Provides the Peekaboo-style Voxt note panel views.

import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct VoxtNoteDragPayload: Sendable {
    let text: String
    var noteID: UUID? = nil

    func itemProvider() -> NSItemProvider {
        let provider = NSItemProvider()
        let plainText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        provider.suggestedName = noteID?.uuidString
        provider.registerObject(plainText as NSString, visibility: .all)
        provider.registerDataRepresentation(
            forTypeIdentifier: UTType.utf8PlainText.identifier,
            visibility: .all
        ) { completion in
            completion(Data(plainText.utf8), nil)
            return nil
        }
        return provider
    }
}

private enum VoxtNotePanelStyle {
    static let cornerRadius: CGFloat = 18
    static let horizontalPadding: CGFloat = 16
    static let rowHeight: CGFloat = 32
    static let itemSpacing: CGFloat = 4
    static let spring = Animation.spring(response: 0.30, dampingFraction: 0.84)
    static let quick = Animation.easeOut(duration: 0.14)
}

private extension VoxtNotePriority {
    var color: Color {
        switch self {
        case .none:
            return Color.secondary.opacity(0.5)
        case .low:
            return .blue
        case .medium:
            return .orange
        case .high:
            return .red
        }
    }
}

private struct VoxtNotePanelSurface: ViewModifier {
    let translucent: Bool

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content
            .background {
                ZStack {
                    Rectangle()
                        .fill(.ultraThinMaterial)
                        .opacity(translucent ? 1 : 0)
                    Rectangle()
                        .fill(Color(nsColor: .windowBackgroundColor))
                        .opacity(translucent ? 0 : 1)
                }
                .animation(reduceMotion ? nil : .easeInOut(duration: 0.22), value: translucent)
            }
            .clipShape(
                RoundedRectangle(cornerRadius: VoxtNotePanelStyle.cornerRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: VoxtNotePanelStyle.cornerRadius, style: .continuous)
                    .stroke(Color.primary.opacity(0.055), lineWidth: 0.7)
            }
    }
}

struct VoxtNoteWindowView: View {
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @ObservedObject var store: VoxtNoteStore
    @ObservedObject var uiState: VoxtNotePanelUIState
    @ObservedObject var settingsState: VoxtNotePanelSettingsState
    let onOpenSettings: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let _ = interfaceLanguageRaw
        let snapshot = store.snapshot(for: uiState.selectedScope)

        VStack(spacing: 0) {
            header(activeCount: snapshot.activeCount)
            scopePicker

            if snapshot.visibleCount == 0 {
                emptyState
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            } else {
                noteList(sections: snapshot.sections)
                    .transition(.opacity)
            }

            if let error = store.lastErrorMessage {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .padding(.horizontal, VoxtNotePanelStyle.horizontalPadding)
                    .padding(.bottom, 10)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .modifier(VoxtNotePanelSurface(translucent: settingsState.value.isTranslucent))
        .animation(reduceMotion ? nil : VoxtNotePanelStyle.spring, value: store.revision)
        .animation(reduceMotion ? nil : VoxtNotePanelStyle.quick, value: uiState.selectedScope)
    }

    private func header(activeCount: Int) -> some View {
        HStack(spacing: 8) {
            Text("Voxt Notes")
                .font(.system(size: 13, weight: .semibold, design: .rounded))
            Text("· \(activeSubtitle(count: activeCount))")
                .font(.system(size: 10, design: .rounded))
                .foregroundStyle(.tertiary)
                .contentTransition(.numericText())

            Spacer()

            Button(action: onOpenSettings) {
                Image(systemName: "gearshape")
                    .font(.system(size: 10, weight: .semibold))
                    .frame(width: 24, height: 24)
                    .background(Color.primary.opacity(0.06), in: Circle())
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(AppLocalization.localizedString("Settings"))
            .accessibilityLabel(AppLocalization.localizedString("Settings"))
        }
        .padding(.leading, VoxtNotePanelStyle.horizontalPadding)
        .padding(.trailing, VoxtNotePanelStyle.horizontalPadding - 4)
        .frame(height: 44)
    }

    private var scopePicker: some View {
        HStack(spacing: 6) {
            ForEach(VoxtNoteScope.allCases) { scope in
                let isSelected = uiState.selectedScope == scope
                Button {
                    if reduceMotion {
                        uiState.selectScope(scope)
                    } else {
                        withAnimation(VoxtNotePanelStyle.quick) {
                            uiState.selectScope(scope)
                        }
                    }
                } label: {
                    Text(scope.title)
                        .font(.system(
                            size: 10,
                            weight: isSelected ? .semibold : .medium,
                            design: .rounded
                        ))
                        .foregroundStyle(
                            isSelected ? Color(nsColor: .windowBackgroundColor) : Color.secondary
                        )
                        .padding(.horizontal, 10)
                        .frame(height: 22)
                        .background(
                            Color.primary.opacity(isSelected ? 0.9 : 0.035),
                            in: Capsule(style: .continuous)
                        )
                        .contentShape(Capsule(style: .continuous))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(scope.title)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, VoxtNotePanelStyle.horizontalPadding)
        .padding(.bottom, 8)
    }

    private func noteList(sections: [VoxtNoteSectionSnapshot]) -> some View {
        ScrollView {
            LazyVStack(spacing: 7) {
                ForEach(sections) { section in
                    let totalItemCount = section.items.count
                    let visibleItems = section.status == .done
                        ? Array(section.items.prefix(uiState.completedItemLimit))
                        : section.items
                    VoxtNoteSectionView(
                        store: store,
                        uiState: uiState,
                        status: section.status,
                        items: visibleItems,
                        totalItemCount: totalItemCount,
                        onLoadMore: visibleItems.count < totalItemCount ? {
                            uiState.loadMoreCompletedItems(totalCount: totalItemCount)
                        } : nil
                    )
                }
            }
            .padding(.horizontal, VoxtNotePanelStyle.horizontalPadding - 4)
            .padding(.bottom, 14)
        }
        .scrollIndicators(.never)
    }

    private var emptyState: some View {
        VStack(spacing: 5) {
            Text(uiState.selectedScope == .notes
                ? AppLocalization.localizedString("Nothing hiding here")
                : AppLocalization.localizedString("No notes waiting"))
                .font(.system(size: 13, weight: .medium, design: .rounded))
            Text(uiState.selectedScope == .notes
                ? AppLocalization.localizedString("Capture a note during transcription and it will stay close by.")
                : AppLocalization.localizedString("Move a note here to keep it for later."))
                .font(.system(size: 11, design: .rounded))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 24)
        .padding(.bottom, 18)
    }

    private func activeSubtitle(count: Int) -> String {
        switch uiState.selectedScope {
        case .notes:
            return count == 1
                ? AppLocalization.localizedString("1 Active Note")
                : String(format: AppLocalization.localizedString("%d Active Notes"), count)
        case .backlog:
            return count == 1
                ? AppLocalization.localizedString("1 Backlog Note")
                : String(format: AppLocalization.localizedString("%d Backlog Notes"), count)
        }
    }
}

private struct VoxtNoteSectionView: View {
    @ObservedObject var store: VoxtNoteStore
    @ObservedObject var uiState: VoxtNotePanelUIState
    let status: VoxtNoteStatus
    let items: [VoxtNoteItem]
    let totalItemCount: Int
    let onLoadMore: (() -> Void)?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isLoadMoreHovered = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 5) {
                Text("\(status.title) · \(totalItemCount)")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
                    .contentTransition(.numericText())
                Spacer()
            }
            .frame(height: 24)
            .padding(.horizontal, 4)

            VStack(spacing: VoxtNotePanelStyle.itemSpacing) {
                ForEach(items) { item in
                    VoxtNoteListRow(store: store, uiState: uiState, item: item)
                        .transition(
                            .asymmetric(
                                insertion: .move(edge: .top).combined(with: .opacity),
                                removal: .scale(scale: 0.96).combined(with: .opacity)
                            )
                        )
                }

                if let onLoadMore {
                    Button(action: onLoadMore) {
                        Image(systemName: "ellipsis")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 32, height: 24)
                            .background(
                                Color.primary.opacity(isLoadMoreHovered ? 0.07 : 0),
                                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
                            )
                            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: .infinity)
                    .onHover { isLoadMoreHovered = $0 }
                    .accessibilityLabel(AppLocalization.localizedString("More"))
                }
            }
        }
        .animation(reduceMotion ? nil : VoxtNotePanelStyle.spring, value: items)
    }
}

struct VoxtNoteStatusMark: View {
    let status: VoxtNoteStatus
    let priority: VoxtNotePriority
    var size: CGFloat = 15

    var body: some View {
        ZStack {
            switch status {
            case .todo:
                Circle()
                    .stroke(priority.color, lineWidth: 1.5)
                    .frame(width: size, height: size)
            case .inProgress:
                Circle()
                    .stroke(priority.color, lineWidth: 1.5)
                    .frame(width: size, height: size)
                Circle()
                    .trim(from: 0, to: 0.5)
                    .rotation(.degrees(90))
                    .fill(priority.color)
                    .frame(width: size * 0.7, height: size * 0.7)
            case .done:
                Circle()
                    .fill(priority.color)
                    .frame(width: size, height: size)
                Image(systemName: "checkmark")
                    .font(.system(size: size * 0.47, weight: .bold))
                    .foregroundStyle(.white)
            case .backlog:
                Circle()
                    .stroke(
                        priority.color,
                        style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [2.2, 2.2])
                    )
                    .frame(width: size, height: size)
            }
        }
    }
}

private struct VoxtNoteStatusButton: View {
    let item: VoxtNoteItem
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Button(action: action) {
            ZStack {
                VoxtNoteStatusMark(status: item.status, priority: item.priority)
                    .id(item.status)
                    .transition(.scale(scale: 0.65).combined(with: .opacity))
            }
            .frame(width: 22, height: 22)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(primaryActionTitle)
        .animation(reduceMotion ? nil : VoxtNotePanelStyle.spring, value: item.status)
        .accessibilityLabel(primaryActionTitle)
        .accessibilityValue("\(item.priority.title) \(AppLocalization.localizedString("priority"))")
    }

    private var primaryActionTitle: String {
        switch item.status {
        case .backlog:
            return AppLocalization.localizedString("Move to To do")
        case .todo, .inProgress:
            return AppLocalization.localizedString("Mark done")
        case .done:
            return AppLocalization.localizedString("Move back to To do")
        }
    }
}

private struct VoxtNoteListRow: View {
    @ObservedObject var store: VoxtNoteStore
    @ObservedObject var uiState: VoxtNotePanelUIState
    let item: VoxtNoteItem

    @State private var editTitle = ""
    @State private var isHovering = false
    @State private var isDetailEditing = false
    @State private var detailTitle = ""
    @State private var detailContent = ""
    @FocusState private var isRenameFocused: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var isEditing: Bool { uiState.editingNoteID == item.id }

    var body: some View {
        HStack(spacing: 8) {
            VoxtNoteStatusButton(item: item) {
                store.performPrimaryAction(for: item.id)
            }

            Group {
                if isEditing {
                    TextField(AppLocalization.localizedString("Note title"), text: $editTitle, axis: .vertical)
                        .textFieldStyle(.plain)
                        .lineLimit(1...6)
                        .focused($isRenameFocused)
                        .onSubmit(commitRename)
                        .onExitCommand(perform: cancelRename)
                } else {
                    Text(item.title)
                        .lineLimit(nil)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                        .layoutPriority(1)
                        .strikethrough(item.status == .done, color: .secondary)
                        .foregroundStyle(item.status == .done ? .secondary : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                        .onTapGesture(perform: showDetail)
                    .accessibilityAddTraits(.isButton)
                    .accessibilityHint(AppLocalization.localizedString("Open note details"))
                }
            }
            .font(.system(
                size: 13,
                weight: item.status == .inProgress ? .medium : .regular,
                design: .rounded
            ))
            .frame(maxWidth: .infinity, alignment: .leading)

            trailingAction
        }
        .padding(.vertical, 2)
        .frame(minHeight: VoxtNotePanelStyle.rowHeight)
        .padding(.horizontal, 4)
        .background(
            Color.primary.opacity(isHovering ? 0.055 : 0),
            in: RoundedRectangle(cornerRadius: 8, style: .continuous)
        )
        .contentShape(Rectangle())
        .onTapGesture(count: 2, perform: handleDoubleClick)
        .help(item.text)
        .contextMenu { noteActions }
        .onDrag {
            dragItemProvider()
        } preview: {
            dragPreview
        }
        .onDrop(of: [.utf8PlainText], isTargeted: nil) { providers, _ in
            acceptDrop(from: providers)
        }
        .onHover { hovering in
            withAnimation(reduceMotion ? nil : VoxtNotePanelStyle.quick) {
                isHovering = hovering
            }
        }
        .onChange(of: isEditing) { _, nowEditing in
            guard nowEditing else { return }
            editTitle = item.title
            DispatchQueue.main.async { isRenameFocused = true }
        }
        .popover(isPresented: detailBinding, arrowEdge: .trailing) {
            noteDetail
        }
        .accessibilityElement(children: .contain)
    }

    private var detailBinding: Binding<Bool> {
        Binding(
            get: { uiState.detailNoteID == item.id },
            set: { isPresented in
                if isPresented {
                    showDetail()
                } else if uiState.detailNoteID == item.id {
                    uiState.hideDetail()
                    isDetailEditing = false
                }
            }
        )
    }

    private var noteDetail: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                if isDetailEditing {
                    TextField(AppLocalization.localizedString("Note title"), text: $detailTitle)
                        .textFieldStyle(.plain)
                        .font(.headline)
                        .settingsFieldSurface(minHeight: 32)
                } else {
                    Text(item.title)
                        .font(.headline)
                        .textSelection(.enabled)
                }
                Spacer(minLength: 12)
                Button {
                    uiState.hideDetail()
                    isDetailEditing = false
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .accessibilityLabel(AppLocalization.localizedString("Close"))
            }
            if isDetailEditing {
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
                if isDetailEditing {
                    Button(AppLocalization.localizedString("Cancel"), action: cancelDetailEditing)
                    Button(AppLocalization.localizedString("Save"), action: saveDetailEditing)
                        .disabled(!canSaveDetail)
                } else {
                    Button(AppLocalization.localizedString("Edit"), action: beginDetailEditing)
                    Button(AppLocalization.localizedString("Copy")) {
                        copyBody()
                    }
                }
            }
        }
        .padding(16)
        .frame(width: 340, height: 260)
    }

    private var dragPreview: some View {
        HStack(spacing: 7) {
            VoxtNoteStatusMark(status: item.status, priority: item.priority, size: 13)
            Text(item.title)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .lineLimit(2)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
    }

    private func dragItemProvider() -> NSItemProvider {
        uiState.beginDragging(item.id)
        return VoxtNoteDragPayload(text: item.text).itemProvider()
    }

    private func acceptDrop(from providers: [NSItemProvider]) -> Bool {
        guard providers.contains(where: {
            $0.hasItemConformingToTypeIdentifier(UTType.utf8PlainText.identifier)
        }), let draggedNoteID = uiState.draggedNoteID else {
            return false
        }
        uiState.endDragging()
        return store.reorder(noteID: draggedNoteID, relativeTo: item.id)
    }

    @ViewBuilder
    private var trailingAction: some View {
        if isEditing {
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
            .opacity(isHovering ? 1 : 0.38)
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
            uiState.beginEditing(item.id)
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Edit title…"), icon: .edit)
        }

        Button {
            copyBody()
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Copy"), icon: .copy)
        }

        Menu(AppLocalization.localizedString("Priority")) {
            ForEach(VoxtNotePriority.allCases.reversed()) { priority in
                Button {
                    store.setPriority(priority, for: item.id)
                } label: {
                    if item.priority == priority {
                        Label(priority.title, systemImage: "checkmark")
                    } else {
                        Text(priority.title)
                    }
                }
            }
        }

        Menu(AppLocalization.localizedString("Move to")) {
            ForEach(VoxtNoteStatus.moveMenuOrder) { status in
                Button {
                    store.setStatus(status, for: item.id)
                } label: {
                    if item.status == status {
                        Label(status.title, systemImage: "checkmark")
                    } else {
                        Text(status.title)
                    }
                }
            }
        }

        Divider()

        Button(role: .destructive) {
            store.delete(id: item.id)
        } label: {
            AppSVGMenuLabel(title: AppLocalization.localizedString("Delete"), icon: .delete, color: .red)
        }
    }

    private func commitRename() {
        if store.rename(item.id, to: editTitle) {
            uiState.endEditing()
        }
    }

    private func cancelRename() {
        uiState.endEditing()
    }

    private func handleDoubleClick() {
        guard !isEditing else { return }
        store.performDoubleClickAction(for: item.id)
    }

    private var canSaveDetail: Bool {
        !detailTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !detailContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func showDetail() {
        guard !isEditing else { return }
        detailTitle = item.title
        detailContent = item.text
        isDetailEditing = false
        uiState.showDetail(item.id)
    }

    private func beginDetailEditing() {
        detailTitle = item.title
        detailContent = item.text
        isDetailEditing = true
    }

    private func cancelDetailEditing() {
        detailTitle = item.title
        detailContent = item.text
        isDetailEditing = false
    }

    private func saveDetailEditing() {
        guard canSaveDetail,
              store.updateDetails(item.id, title: detailTitle, text: detailContent)
        else {
            return
        }
        isDetailEditing = false
    }

    private func copyBody() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(item.text, forType: .string)
    }
}
