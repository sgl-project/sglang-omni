// DictionarySettingsSections.swift
// Provides Dictionary Settings Sections for dictionary settings.

import SwiftUI
import AppKit

struct DictionaryEntriesCard: View {
    @Binding var selectedTab: DictionaryEntriesTab
    @Binding var selectedHotwordCategoryID: UUID?
    let hotwordSections: [(category: DictionaryCategory, entries: [DictionaryEntry])]
    let replacementEntries: [DictionaryEntry]
    let searchText: String
    let isLoadingEntries: Bool
    let onSearch: () -> Void
    let onClearSearch: () -> Void
    let onCreate: () -> Void
    let onCreateCategory: () -> Void
    let onOpenIngest: () -> Void
    let onOpenSettings: () -> Void
    let onImport: () -> Void
    let onExport: () -> Void
    let onCreateInCategory: (DictionaryCategory) -> Void
    let onEditCategory: (DictionaryCategory) -> Void
    let onDeleteCategory: (DictionaryCategory) -> Void
    let onEdit: (DictionaryEntry) -> Void
    let onDelete: (DictionaryEntry) -> Void

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                toolbar

                if !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    HStack(spacing: 8) {
                        Text(AppLocalization.format("Filtered by \"%@\"", searchText))
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Button(AppLocalization.localizedString("Clear")) {
                            onClearSearch()
                        }
                        .buttonStyle(.plain)
                    }
                }

                if isContentEmpty && isLoadingEntries {
                    ProgressView()
                        .controlSize(.small)
                        .frame(maxWidth: .infinity, minHeight: 180, alignment: .center)
                } else if isContentEmpty {
                    SettingsEmptyStateView(
                        illustration: .dictionary,
                        title: emptyStateTitle,
                        message: emptyStateMessage
                    )
                } else {
                    ScrollView {
                        content
                            .padding(.vertical, 2)
                            .transaction { transaction in
                                transaction.disablesAnimations = true
                                transaction.animation = nil
                            }
                    }
                    .frame(maxWidth: .infinity, minHeight: 180, maxHeight: .infinity, alignment: .top)
                }

            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(maxHeight: .infinity, alignment: .top)
    }

    @ViewBuilder
    private var toolbar: some View {
        if selectedTab == .hotwords, let selectedHotwordSection {
            HStack(spacing: 8) {
                Button {
                    selectedHotwordCategoryID = nil
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(SettingsCompactIconButtonStyle(size: 30))
                .accessibilityLabel(AppLocalization.localizedString("Back"))

                VStack(alignment: .leading, spacing: 2) {
                    Text(selectedHotwordSection.category.name)
                        .font(.system(size: 13.5, weight: .semibold))
                        .lineLimit(1)
                    Text(AppLocalization.format("%d terms", selectedHotwordSection.entries.count))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 12)

                DictionaryHeaderIconButton(
                    accessibilityLabel: AppLocalization.localizedString("Search Current Category"),
                    action: onSearch
                ) {
                    SettingsSearchIconView()
                }

                DictionaryHeaderIconButton(
                    accessibilityLabel: AppLocalization.localizedString("Create Hot Word in This Category"),
                    action: { onCreateInCategory(selectedHotwordSection.category) }
                ) {
                    DictionaryActionIcon(kind: .createTerm, size: 15)
                }

                DictionaryHeaderIconButton(
                    accessibilityLabel: AppLocalization.localizedString("Edit Dictionary Category"),
                    action: { onEditCategory(selectedHotwordSection.category) }
                ) {
                    AppSVGIcon(kind: .edit, size: 15)
                }

                DictionaryHeaderIconButton(
                    accessibilityLabel: AppLocalization.localizedString("Delete Dictionary Category"),
                    action: {
                        if !selectedHotwordSection.category.isDefault {
                            onDeleteCategory(selectedHotwordSection.category)
                        }
                    }
                ) {
                    DictionaryActionIcon(kind: .deleteCategory, size: 15)
                }
            }
        } else {
            HStack {
                DictionaryEntriesTabPicker(selectedTab: $selectedTab)

                Spacer(minLength: 12)

                DictionaryHeaderIconButton(
                    accessibilityLabel: searchAccessibilityLabel,
                    action: onSearch
                ) {
                    SettingsSearchIconView()
                }

                if selectedTab == .hotwords {
                    DictionaryHeaderIconButton(
                        accessibilityLabel: AppLocalization.localizedString("Create Category"),
                        action: onCreateCategory
                    ) {
                        DictionaryActionIcon(kind: .createCategory, size: 15)
                    }
                }

                DictionaryHeaderIconButton(
                    accessibilityLabel: createTermAccessibilityLabel,
                    action: onCreate
                ) {
                    DictionaryActionIcon(kind: .createTerm, size: 15)
                }

                Rectangle()
                    .fill(SettingsUIStyle.subtleBorderColor)
                    .frame(width: 1, height: 20)
                    .padding(.horizontal, 4)

                DictionaryHeaderIconButton(
                    accessibilityLabel: AppLocalization.localizedString("One-Click Ingest"),
                    action: onOpenIngest
                ) {
                    SettingsOneClickIngestIconView(size: 15)
                }

                DictionaryHeaderIconButton(
                    accessibilityLabel: AppLocalization.localizedString("Dictionary Advanced Settings"),
                    action: onOpenSettings
                ) {
                    SettingsSparkleSettingsIconView(size: 15)
                }

                DictionaryHeaderActionMenuButton(
                    actions: [
                        DictionaryHeaderMenuAction(
                            title: AppLocalization.localizedString("Import"),
                            handler: onImport
                        ),
                        DictionaryHeaderMenuAction(
                            title: AppLocalization.localizedString("Export"),
                            handler: onExport
                        )
                    ],
                    accessibilityLabel: AppLocalization.localizedString("More")
                )
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch selectedTab {
        case .hotwords:
            if let selectedHotwordSection {
                HotwordCategoryDetail(
                    entries: selectedHotwordSection.entries,
                    onEditEntry: onEdit,
                    onDeleteEntry: onDelete
                )
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 180, maximum: 260), spacing: 10, alignment: .top)],
                    alignment: .leading,
                    spacing: 10
                ) {
                    ForEach(displayedHotwordSections, id: \.category.id) { section in
                        HotwordCategoryCard(
                            category: section.category,
                            entries: section.entries,
                            onOpen: { selectedHotwordCategoryID = section.category.id }
                        )
                    }
                }
            }
        case .replacements:
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 240, maximum: 360), spacing: 8, alignment: .top)],
                alignment: .leading,
                spacing: 8
            ) {
                ForEach(replacementEntries) { entry in
                    DictionaryReplacementRow(
                        entry: entry,
                        onEdit: { onEdit(entry) },
                        onDelete: { onDelete(entry) }
                    )
                }
            }
        }
    }

    private var displayedHotwordSections: [(category: DictionaryCategory, entries: [DictionaryEntry])] {
        guard isSearchActive else { return hotwordSections }
        return hotwordSections.filter { !$0.entries.isEmpty }
    }

    private var selectedHotwordSection: (category: DictionaryCategory, entries: [DictionaryEntry])? {
        guard let selectedHotwordCategoryID else { return nil }
        return hotwordSections.first(where: { $0.category.id == selectedHotwordCategoryID })
    }

    private var isContentEmpty: Bool {
        switch selectedTab {
        case .hotwords:
            if selectedHotwordSection != nil {
                return false
            }
            return displayedHotwordSections.allSatisfy { $0.entries.isEmpty }
        case .replacements:
            return replacementEntries.isEmpty
        }
    }

    private var createTermAccessibilityLabel: String {
        switch selectedTab {
        case .hotwords:
            return AppLocalization.localizedString("Create Hot Word")
        case .replacements:
            return AppLocalization.localizedString("Create Replacement Term")
        }
    }

    private var searchAccessibilityLabel: String {
        switch selectedTab {
        case .hotwords:
            return AppLocalization.localizedString("Search Hot Words")
        case .replacements:
            return AppLocalization.localizedString("Search Replacement Terms")
        }
    }

    private var emptyStateTitle: String {
        if !isSearchActive {
            switch selectedTab {
            case .hotwords:
                return AppLocalization.localizedString("No hot words yet")
            case .replacements:
                return AppLocalization.localizedString("No replacement terms yet")
            }
        }
        return AppLocalization.localizedString("No matching dictionary terms")
    }

    private var emptyStateMessage: String {
        if !isSearchActive {
            switch selectedTab {
            case .hotwords:
                return AppLocalization.localizedString("Create a hot word to help Voxt recognize names, jargon, and product words.")
            case .replacements:
                return AppLocalization.localizedString("Create a replacement term to normalize final transcription results.")
            }
        }
        return AppLocalization.localizedString("Try another keyword or clear the search filter.")
    }

    private var isSearchActive: Bool {
        !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

}

private struct HotwordCategoryCard: View {
    let category: DictionaryCategory
    let entries: [DictionaryEntry]
    let onOpen: () -> Void

    @State private var isHovering = false

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            VStack(alignment: .leading, spacing: 3) {
                Text(category.name)
                    .font(.system(size: 13.5, weight: .semibold))
                    .lineLimit(1)

                Text(AppLocalization.format("%d terms", entries.count))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 8)

            Image(systemName: "chevron.right")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .frame(maxWidth: .infinity, minHeight: 58, alignment: .center)
        .contentShape(Rectangle())
        .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
        .brightness(isHovering ? 0.035 : 0)
        .overlay {
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .strokeBorder(Color.accentColor.opacity(isHovering ? 0.42 : 0), lineWidth: 1)
        }
        .onHover { isHovering = $0 }
        .onTapGesture(perform: onOpen)
        .animation(.easeOut(duration: 0.12), value: isHovering)
    }
}

private struct HotwordCategoryDetail: View {
    let entries: [DictionaryEntry]
    let onEditEntry: (DictionaryEntry) -> Void
    let onDeleteEntry: (DictionaryEntry) -> Void

    var body: some View {
        Group {
            if entries.isEmpty {
                Text(AppLocalization.localizedString("No dictionary terms yet."))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 160, maximum: 240), spacing: 8, alignment: .top)],
                    alignment: .leading,
                    spacing: 8
                ) {
                ForEach(entries) { entry in
                    DictionaryRow(
                        entry: entry,
                        onEdit: { onEditEntry(entry) },
                        onDelete: { onDeleteEntry(entry) }
                    )
                }
                }
            }
        }
    }
}

private struct DictionaryReplacementRow: View {
    let entry: DictionaryEntry
    let onEdit: () -> Void
    let onDelete: () -> Void

    @State private var isHovering = false
    @State private var isDeleteHovering = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(entry.term)
                    .font(.system(size: 12.5, weight: .semibold))
                    .lineLimit(1)

                Text(scopeText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Spacer(minLength: 28)
            }

            Text(replacementText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .settingsCardSurface(cornerRadius: SettingsUIStyle.compactCornerRadius, fillOpacity: 1)
        .brightness(isHovering ? 0.035 : 0)
        .overlay {
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .strokeBorder(Color.accentColor.opacity(isHovering ? 0.42 : 0), lineWidth: 1)
        }
        .onHover { isHovering = $0 }
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
            .help(AppLocalization.localizedString("Delete"))
            .onHover { isDeleteHovering = $0 }
            .padding(6)
        }
    }

    private var replacementText: String {
        entry.replacementTerms.map(\.text).joined(separator: ", ")
    }

    private var scopeText: String {
        guard entry.groupID != nil else {
            return AppLocalization.localizedString("Global")
        }
        return entry.groupNameSnapshot ?? AppLocalization.localizedString("Missing Group")
    }
}

private struct DictionaryHeaderIconButton<Icon: View>: View {
    let accessibilityLabel: String
    let action: () -> Void
    @ViewBuilder let icon: () -> Icon

    var body: some View {
        Button(action: action) {
            icon()
                .frame(width: 15, height: 15)
                .frame(width: 30, height: 30)
        }
        .buttonStyle(DictionaryHeaderIconButtonStyle())
        .accessibilityLabel(accessibilityLabel)
    }
}

private struct DictionaryHeaderIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        DictionaryHeaderIconButtonStyleBody(configuration: configuration)
    }
}

private struct DictionaryHeaderIconButtonStyleBody: View {
    let configuration: ButtonStyle.Configuration
    @State private var isHovered = false

    var body: some View {
        configuration.label
            .foregroundStyle(Color.secondary)
            .background(fillColor)
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .onHover { isHovered = $0 }
    }

    private var fillColor: Color {
        if configuration.isPressed {
            return SettingsUIStyle.subtleFillColor.opacity(0.92)
        }
        if isHovered {
            return SettingsUIStyle.subtleFillColor.opacity(0.72)
        }
        return SettingsUIStyle.subtleFillColor
    }
}

private struct DictionaryActionIcon: View {
    enum Kind {
        case createCategory
        case deleteCategory
        case createTerm
    }

    let kind: Kind
    var size: CGFloat = 18

    var body: some View {
        DictionaryActionIconShape(kind: kind)
            .fill(kind == .deleteCategory ? Color.red : Color.secondary)
            .frame(width: size, height: size)
            .accessibilityHidden(true)
    }
}

private struct DictionaryActionIconShape: Shape {
    let kind: DictionaryActionIcon.Kind

    func path(in rect: CGRect) -> Path {
        let scale = min(rect.width, rect.height) / 24
        let transform = CGAffineTransform(
            a: scale,
            b: 0,
            c: 0,
            d: scale,
            tx: rect.midX - 12 * scale,
            ty: rect.midY - 12 * scale
        )

        var path = Path()
        switch kind {
        case .createCategory:
            addFolderPlus(to: &path)
        case .deleteCategory:
            addFolderDelete(to: &path)
        case .createTerm:
            addBookmarkPlus(to: &path)
        }
        return path.applying(transform)
    }

    private func addFolderPlus(to path: inout Path) {
        path.move(to: CGPoint(x: 12.0601, y: 17.25))
        path.addCurve(to: CGPoint(x: 11.3101, y: 16.5), control1: CGPoint(x: 11.6501, y: 17.25), control2: CGPoint(x: 11.3101, y: 16.91))
        path.addLine(to: CGPoint(x: 11.3101, y: 11.5))
        path.addCurve(to: CGPoint(x: 12.0601, y: 10.75), control1: CGPoint(x: 11.3101, y: 11.09), control2: CGPoint(x: 11.6501, y: 10.75))
        path.addCurve(to: CGPoint(x: 12.8101, y: 11.5), control1: CGPoint(x: 12.4701, y: 10.75), control2: CGPoint(x: 12.8101, y: 11.09))
        path.addLine(to: CGPoint(x: 12.8101, y: 16.5))
        path.addCurve(to: CGPoint(x: 12.0601, y: 17.25), control1: CGPoint(x: 12.8101, y: 16.91), control2: CGPoint(x: 12.4701, y: 17.25))
        path.closeSubpath()

        path.move(to: CGPoint(x: 14.5, y: 14.75))
        path.addLine(to: CGPoint(x: 9.5, y: 14.75))
        path.addCurve(to: CGPoint(x: 8.75, y: 14), control1: CGPoint(x: 9.09, y: 14.75), control2: CGPoint(x: 8.75, y: 14.41))
        path.addCurve(to: CGPoint(x: 9.5, y: 13.25), control1: CGPoint(x: 8.75, y: 13.59), control2: CGPoint(x: 9.09, y: 13.25))
        path.addLine(to: CGPoint(x: 14.5, y: 13.25))
        path.addCurve(to: CGPoint(x: 15.25, y: 14), control1: CGPoint(x: 14.91, y: 13.25), control2: CGPoint(x: 15.25, y: 13.59))
        path.addCurve(to: CGPoint(x: 14.5, y: 14.75), control1: CGPoint(x: 15.25, y: 14.41), control2: CGPoint(x: 14.91, y: 14.75))
        path.closeSubpath()

        addFolderOutline(to: &path)
    }

    private func addFolderDelete(to path: inout Path) {
        path.move(to: CGPoint(x: 13.81, y: 16.4799))
        path.addCurve(to: CGPoint(x: 13.28, y: 16.2599), control1: CGPoint(x: 13.62, y: 16.4799), control2: CGPoint(x: 13.43, y: 16.4099))
        path.addLine(to: CGPoint(x: 9.74, y: 12.7199))
        path.addCurve(to: CGPoint(x: 9.74, y: 11.6599), control1: CGPoint(x: 9.45, y: 12.4299), control2: CGPoint(x: 9.45, y: 11.9499))
        path.addCurve(to: CGPoint(x: 10.8, y: 11.6599), control1: CGPoint(x: 10.03, y: 11.3699), control2: CGPoint(x: 10.51, y: 11.3699))
        path.addLine(to: CGPoint(x: 14.34, y: 15.1999))
        path.addCurve(to: CGPoint(x: 14.34, y: 16.2599), control1: CGPoint(x: 14.63, y: 15.4899), control2: CGPoint(x: 14.63, y: 15.9699))
        path.addCurve(to: CGPoint(x: 13.81, y: 16.4799), control1: CGPoint(x: 14.19, y: 16.3999), control2: CGPoint(x: 14, y: 16.4799))
        path.closeSubpath()

        path.move(to: CGPoint(x: 10.2299, y: 16.5199))
        path.addCurve(to: CGPoint(x: 9.6999, y: 16.2999), control1: CGPoint(x: 10.0399, y: 16.5199), control2: CGPoint(x: 9.8499, y: 16.4499))
        path.addCurve(to: CGPoint(x: 9.6999, y: 15.2399), control1: CGPoint(x: 9.4099, y: 16.0099), control2: CGPoint(x: 9.4099, y: 15.5299))
        path.addLine(to: CGPoint(x: 13.2399, y: 11.6999))
        path.addCurve(to: CGPoint(x: 14.2999, y: 11.6999), control1: CGPoint(x: 13.5299, y: 11.4099), control2: CGPoint(x: 14.0099, y: 11.4099))
        path.addCurve(to: CGPoint(x: 14.2999, y: 12.7599), control1: CGPoint(x: 14.5899, y: 11.9899), control2: CGPoint(x: 14.5899, y: 12.4699))
        path.addLine(to: CGPoint(x: 10.7599, y: 16.2999))
        path.addCurve(to: CGPoint(x: 10.2299, y: 16.5199), control1: CGPoint(x: 10.6199, y: 16.4399), control2: CGPoint(x: 10.4199, y: 16.5199))
        path.closeSubpath()

        addFolderOutline(to: &path)
    }

    private func addFolderOutline(to path: inout Path) {
        path.move(to: CGPoint(x: 17, y: 22.75))
        path.addLine(to: CGPoint(x: 7, y: 22.75))
        path.addCurve(to: CGPoint(x: 1.25, y: 17), control1: CGPoint(x: 2.59, y: 22.75), control2: CGPoint(x: 1.25, y: 21.41))
        path.addLine(to: CGPoint(x: 1.25, y: 7))
        path.addCurve(to: CGPoint(x: 7, y: 1.25), control1: CGPoint(x: 1.25, y: 2.59), control2: CGPoint(x: 2.59, y: 1.25))
        path.addLine(to: CGPoint(x: 8.5, y: 1.25))
        path.addCurve(to: CGPoint(x: 11.5, y: 2.75), control1: CGPoint(x: 10.25, y: 1.25), control2: CGPoint(x: 10.8, y: 1.82))
        path.addLine(to: CGPoint(x: 13, y: 4.75))
        path.addCurve(to: CGPoint(x: 14, y: 5.25), control1: CGPoint(x: 13.33, y: 5.19), control2: CGPoint(x: 13.38, y: 5.25))
        path.addLine(to: CGPoint(x: 17, y: 5.25))
        path.addCurve(to: CGPoint(x: 22.75, y: 11), control1: CGPoint(x: 21.41, y: 5.25), control2: CGPoint(x: 22.75, y: 6.59))
        path.addLine(to: CGPoint(x: 22.75, y: 17))
        path.addCurve(to: CGPoint(x: 17, y: 22.75), control1: CGPoint(x: 22.75, y: 21.41), control2: CGPoint(x: 21.41, y: 22.75))
        path.closeSubpath()

        path.move(to: CGPoint(x: 7, y: 2.75))
        path.addCurve(to: CGPoint(x: 2.75, y: 7), control1: CGPoint(x: 3.43, y: 2.75), control2: CGPoint(x: 2.75, y: 3.43))
        path.addLine(to: CGPoint(x: 2.75, y: 17))
        path.addCurve(to: CGPoint(x: 7, y: 21.25), control1: CGPoint(x: 2.75, y: 20.57), control2: CGPoint(x: 3.43, y: 21.25))
        path.addLine(to: CGPoint(x: 17, y: 21.25))
        path.addCurve(to: CGPoint(x: 21.25, y: 17), control1: CGPoint(x: 20.57, y: 21.25), control2: CGPoint(x: 21.25, y: 20.57))
        path.addLine(to: CGPoint(x: 21.25, y: 11))
        path.addCurve(to: CGPoint(x: 17, y: 6.75), control1: CGPoint(x: 21.25, y: 7.43), control2: CGPoint(x: 20.57, y: 6.75))
        path.addLine(to: CGPoint(x: 14, y: 6.75))
        path.addCurve(to: CGPoint(x: 11.8, y: 5.65), control1: CGPoint(x: 12.72, y: 6.75), control2: CGPoint(x: 12.3, y: 6.31))
        path.addLine(to: CGPoint(x: 10.3, y: 3.65))
        path.addCurve(to: CGPoint(x: 8.5, y: 2.75), control1: CGPoint(x: 9.78, y: 2.96), control2: CGPoint(x: 9.63, y: 2.75))
        path.addLine(to: CGPoint(x: 7, y: 2.75))
        path.closeSubpath()
    }

    private func addBookmarkPlus(to path: inout Path) {
        path.move(to: CGPoint(x: 14.5, y: 11.4004))
        path.addLine(to: CGPoint(x: 9.5, y: 11.4004))
        path.addCurve(to: CGPoint(x: 8.75, y: 10.6504), control1: CGPoint(x: 9.09, y: 11.4004), control2: CGPoint(x: 8.75, y: 11.0604))
        path.addCurve(to: CGPoint(x: 9.5, y: 9.9004), control1: CGPoint(x: 8.75, y: 10.2404), control2: CGPoint(x: 9.09, y: 9.9004))
        path.addLine(to: CGPoint(x: 14.5, y: 9.9004))
        path.addCurve(to: CGPoint(x: 15.25, y: 10.6504), control1: CGPoint(x: 14.91, y: 9.9004), control2: CGPoint(x: 15.25, y: 10.2404))
        path.addCurve(to: CGPoint(x: 14.5, y: 11.4004), control1: CGPoint(x: 15.25, y: 11.0604), control2: CGPoint(x: 14.91, y: 11.4004))
        path.closeSubpath()

        path.move(to: CGPoint(x: 12, y: 13.9609))
        path.addCurve(to: CGPoint(x: 11.25, y: 13.2109), control1: CGPoint(x: 11.59, y: 13.9609), control2: CGPoint(x: 11.25, y: 13.6209))
        path.addLine(to: CGPoint(x: 11.25, y: 8.2109))
        path.addCurve(to: CGPoint(x: 12, y: 7.4609), control1: CGPoint(x: 11.25, y: 7.8009), control2: CGPoint(x: 11.59, y: 7.4609))
        path.addCurve(to: CGPoint(x: 12.75, y: 8.2109), control1: CGPoint(x: 12.41, y: 7.4609), control2: CGPoint(x: 12.75, y: 7.8009))
        path.addLine(to: CGPoint(x: 12.75, y: 13.2109))
        path.addCurve(to: CGPoint(x: 12, y: 13.9609), control1: CGPoint(x: 12.75, y: 13.6209), control2: CGPoint(x: 12.41, y: 13.9609))
        path.closeSubpath()

        path.move(to: CGPoint(x: 19.0701, y: 22.75))
        path.addCurve(to: CGPoint(x: 17.4601, y: 22.29), control1: CGPoint(x: 18.5601, y: 22.75), control2: CGPoint(x: 18.0001, y: 22.6))
        path.addLine(to: CGPoint(x: 12.5801, y: 19.58))
        path.addCurve(to: CGPoint(x: 11.4301, y: 19.58), control1: CGPoint(x: 12.2901, y: 19.42), control2: CGPoint(x: 11.7201, y: 19.42))
        path.addLine(to: CGPoint(x: 6.5501, y: 22.29))
        path.addCurve(to: CGPoint(x: 3.7801, y: 22.44), control1: CGPoint(x: 5.5601, y: 22.84), control2: CGPoint(x: 4.5501, y: 22.9))
        path.addCurve(to: CGPoint(x: 2.5701, y: 19.95), control1: CGPoint(x: 3.0101, y: 21.99), control2: CGPoint(x: 2.5701, y: 21.08))
        path.addLine(to: CGPoint(x: 2.5701, y: 5.86))
        path.addCurve(to: CGPoint(x: 7.1801, y: 1.25), control1: CGPoint(x: 2.5701, y: 3.32), control2: CGPoint(x: 4.6401, y: 1.25))
        path.addLine(to: CGPoint(x: 16.8301, y: 1.25))
        path.addCurve(to: CGPoint(x: 21.4401, y: 5.86), control1: CGPoint(x: 19.3701, y: 1.25), control2: CGPoint(x: 21.4401, y: 3.32))
        path.addLine(to: CGPoint(x: 21.4401, y: 19.95))
        path.addCurve(to: CGPoint(x: 20.2301, y: 22.44), control1: CGPoint(x: 21.4401, y: 21.08), control2: CGPoint(x: 21.0001, y: 21.99))
        path.addCurve(to: CGPoint(x: 19.0701, y: 22.75), control1: CGPoint(x: 19.8801, y: 22.65), control2: CGPoint(x: 19.4801, y: 22.75))
        path.closeSubpath()

        path.move(to: CGPoint(x: 12.0001, y: 17.96))
        path.addCurve(to: CGPoint(x: 13.3001, y: 18.27), control1: CGPoint(x: 12.4701, y: 17.96), control2: CGPoint(x: 12.9301, y: 18.06))
        path.addLine(to: CGPoint(x: 18.1801, y: 20.98))
        path.addCurve(to: CGPoint(x: 19.4601, y: 21.15), control1: CGPoint(x: 18.6901, y: 21.27), control2: CGPoint(x: 19.1601, y: 21.33))
        path.addCurve(to: CGPoint(x: 19.9301, y: 19.95), control1: CGPoint(x: 19.7601, y: 20.97), control2: CGPoint(x: 19.9301, y: 20.54))
        path.addLine(to: CGPoint(x: 19.9301, y: 5.86))
        path.addCurve(to: CGPoint(x: 16.8201, y: 2.75), control1: CGPoint(x: 19.9301, y: 4.15), control2: CGPoint(x: 18.5301, y: 2.75))
        path.addLine(to: CGPoint(x: 7.1801, y: 2.75))
        path.addCurve(to: CGPoint(x: 4.0701, y: 5.86), control1: CGPoint(x: 5.4701, y: 2.75), control2: CGPoint(x: 4.0701, y: 4.15))
        path.addLine(to: CGPoint(x: 4.0701, y: 19.95))
        path.addCurve(to: CGPoint(x: 4.5401, y: 21.15), control1: CGPoint(x: 4.0701, y: 20.54), control2: CGPoint(x: 4.2401, y: 20.98))
        path.addCurve(to: CGPoint(x: 5.8201, y: 20.98), control1: CGPoint(x: 4.8401, y: 21.32), control2: CGPoint(x: 5.3101, y: 21.27))
        path.addLine(to: CGPoint(x: 10.7001, y: 18.27))
        path.addCurve(to: CGPoint(x: 12.0001, y: 17.96), control1: CGPoint(x: 11.0701, y: 18.06), control2: CGPoint(x: 11.5301, y: 17.96))
        path.closeSubpath()
    }
}
