// FeatureModelSelectorDialog.swift
// Provides Feature Model Selector Dialog for feature settings.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct FeatureModelSelectorDialog: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @AppStorage(AppPreferenceKey.hiddenModelVisibilityIDs) private var hiddenModelVisibilityIDsRaw = ""

    let title: String
    let entries: [FeatureModelSelectorEntry]
    let selectedID: FeatureModelSelectionID
    let onSelect: (FeatureModelSelectionID) -> Void

    @State private var selectedTags = Set<String>()
    @State private var expandedGroupIDs = Set<String>()
    @State private var collapsedGroupIDs = Set<String>()

    private var defaultSelectedTags: Set<String> {
        FeatureModelSelectorFiltering.defaultSelectedTags(entries: entries)
    }

    private var hiddenModelVisibilityIDs: Set<String> {
        ModelVisibilityStore.decode(hiddenModelVisibilityIDsRaw)
    }

    private var locationScopedEntriesForTags: [FeatureModelSelectorEntry] {
        FeatureModelSelectorFiltering.locationScopedEntries(entries: entries, selectedTags: selectedTags)
    }

    private var availableTags: [String] {
        FeatureModelSelectorFiltering.availableTags(
            entries: entries,
            selectedTags: selectedTags,
            locationScopedEntries: locationScopedEntriesForTags,
            hiddenIDs: hiddenModelVisibilityIDs
        )
    }

    private var availableTagGroups: [[String]] {
        let available = Set(availableTags)
        var groups = [[String]]()
        let locationGroup = FeatureSelectorTagPriority.groups[0].filter { available.contains($0) }
        if !locationGroup.isEmpty {
            groups.append(locationGroup)
        }
        groups.append(
            contentsOf: FeatureSelectorTagPriority.groups.dropFirst()
                .map { $0.filter { available.contains($0) } }
                .filter { !$0.isEmpty }
        )
        return groups
    }

    private var filteredEntries: [FeatureModelSelectorEntry] {
        FeatureModelSelectorFiltering.filteredEntries(
            entries: entries,
            selectedTags: selectedTags,
            hiddenIDs: hiddenModelVisibilityIDs
        )
    }

    private var displayItems: [FeatureModelSelectorDisplayItem] {
        LocalModelSeriesGrouping.featureSelectorItems(from: filteredEntries, selectedID: selectedID)
    }

    private var selectedEntry: FeatureModelSelectorEntry? {
        entries.first(where: { $0.selectionID == selectedID })
    }

    private var selectionReminder: String? {
        guard let selectedEntry, !selectedEntry.isSelectable else { return nil }
        guard let disabledReason = selectedEntry.disabledReason?.trimmingCharacters(in: .whitespacesAndNewlines),
              !disabledReason.isEmpty else { return nil }
        return disabledReason
    }

    var body: some View {
        dialogContent
            .settingsDialogChrome(width: 640, onClose: { dismiss() })
            .onAppear(perform: initializeDefaultTags)
            .id(interfaceLanguageRaw)
    }

    private var dialogContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            dialogHeader
            selectionReminderView
            tagFilters
            modelEntries
        }
    }

    private var dialogHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.title3.weight(.semibold))
            }
        }
    }

    @ViewBuilder
    private var selectionReminderView: some View {
        if let selectionReminder {
            HStack(spacing: 6) {
                Image(systemName: "info.circle.fill")
                    .font(.caption)
                    .foregroundStyle(Color.accentColor)
                Text(selectionReminder)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
        }
    }

    @ViewBuilder
    private var tagFilters: some View {
        if !availableTags.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Array(availableTagGroups.enumerated()), id: \.offset) { index, group in
                        HStack(spacing: 8) {
                            ForEach(group, id: \.self) { tag in
                                FeatureModelTagChip(
                                    title: tag,
                                    isSelected: selectedTags.contains(tag)
                                ) {
                                    toggleTag(tag)
                                }
                            }
                        }

                        if index < availableTagGroups.count - 1 {
                            Rectangle()
                                .fill(SettingsUIStyle.subtleBorderColor.opacity(0.95))
                                .frame(width: 1, height: 20)
                                .padding(.horizontal, 4)
                        }
                    }
                }
                .padding(.vertical, 2)
            }
        }
    }

    private var modelEntries: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                if filteredEntries.isEmpty {
                    FeatureSelectorEmptyState()
                } else {
                    ForEach(displayItems) { item in
                        displayItemView(item)
                    }
                }
            }
            .padding(.vertical, 2)
        }
        .frame(height: 400)
    }

    private func toggleTag(_ tag: String) {
        selectedTags = FeatureModelSelectorFiltering.toggledTags(
            current: selectedTags,
            tag: tag,
            entries: entries,
            hiddenIDs: hiddenModelVisibilityIDs
        )
    }

    private func initializeDefaultTags() {
        guard selectedTags.isEmpty else { return }
        selectedTags = defaultSelectedTags
    }

    @ViewBuilder
    private func displayItemView(_ item: FeatureModelSelectorDisplayItem) -> some View {
        switch item {
        case .row(let entry):
            FeatureModelSelectorRow(
                entry: entry,
                isSelected: entry.selectionID == selectedID,
                onSelect: {
                    onSelect(entry.selectionID)
                    dismiss()
                },
                onConfigure: FeatureModelConfigurationRouting.canConfigure(entry.selectionID)
                    ? { openConfiguration(for: entry.selectionID) }
                    : nil,
                visibilityAction: visibilityAction(for: entry)
            )
        case .group(let group):
            FeatureModelSelectorGroupCard(
                group: group,
                selectedID: selectedID,
                isExpanded: isGroupExpanded(group),
                onToggle: { toggleGroup(group) },
                onSelect: { selectionID in
                    onSelect(selectionID)
                    dismiss()
                },
                onConfigure: { selectionID in
                    openConfiguration(for: selectionID)
                },
                visibilityActionForEntry: { visibilityAction(for: $0) }
            )
        }
    }

    private func visibilityAction(for entry: FeatureModelSelectorEntry) -> ModelTableAction? {
        guard !entry.filterTags.contains(localized("Installed")) else { return nil }
        let isHidden = ModelVisibilityStore.isModelHidden(entry.id, in: hiddenModelVisibilityIDs)
        if isHidden {
            return ModelTableAction(title: localized("Show")) {
                showModel(entry.id)
            }
        }
        return ModelTableAction(title: localized("Hide")) {
            hideModel(entry.id)
        }
    }

    private func updateHiddenModelVisibility(_ update: (inout Set<String>) -> Void) {
        var values = hiddenModelVisibilityIDs
        update(&values)
        hiddenModelVisibilityIDsRaw = ModelVisibilityStore.encode(values)
    }

    private func hideModel(_ id: String) {
        updateHiddenModelVisibility { $0.insert(ModelVisibilityStore.modelKey(id)) }
    }

    private func showModel(_ id: String) {
        updateHiddenModelVisibility { $0.remove(ModelVisibilityStore.modelKey(id)) }
    }

    private func openConfiguration(for selectionID: FeatureModelSelectionID) {
        dismiss()
        DispatchQueue.main.async {
            AppDelegate.shared?.openMainWindow(
                target: SettingsNavigationTarget(
                    tab: .model,
                    modelSelectionID: selectionID
                )
            )
        }
    }

    private func isGroupExpanded(_ group: FeatureModelSelectorGroupSection) -> Bool {
        if expandedGroupIDs.contains(group.id) {
            return true
        }
        if collapsedGroupIDs.contains(group.id) {
            return false
        }
        return group.defaultExpanded
    }

    private func toggleGroup(_ group: FeatureModelSelectorGroupSection) {
        let isExpanded = isGroupExpanded(group)
        if group.defaultExpanded {
            if isExpanded {
                collapsedGroupIDs.insert(group.id)
            } else {
                collapsedGroupIDs.remove(group.id)
            }
            expandedGroupIDs.remove(group.id)
            return
        }

        if isExpanded {
            expandedGroupIDs.remove(group.id)
        } else {
            expandedGroupIDs.insert(group.id)
        }
        collapsedGroupIDs.remove(group.id)
    }
}

enum FeatureModelSelectorFiltering {
    static var statusFilterTags: Set<String> {
        Set<String>([localized("Installed"), localized("Configured"), localized("In Use"), localized("Hidden")])
    }

    static func defaultSelectedTags(entries _: [FeatureModelSelectorEntry]) -> Set<String> {
        []
    }

    static func locationScopedEntries(
        entries: [FeatureModelSelectorEntry],
        selectedTags: Set<String>
    ) -> [FeatureModelSelectorEntry] {
        if selectedTags.contains(localized("Local")) {
            return entries.filter { $0.filterTags.contains(localized("Local")) }
        }
        if selectedTags.contains(localized("Remote")) {
            return entries.filter { $0.filterTags.contains(localized("Remote")) }
        }
        return entries
    }

    static func availableTags(
        entries: [FeatureModelSelectorEntry],
        selectedTags: Set<String>,
        locationScopedEntries overrideEntries: [FeatureModelSelectorEntry]? = nil,
        hiddenIDs: Set<String> = []
    ) -> [String] {
        let scopedEntries = overrideEntries ?? self.locationScopedEntries(entries: entries, selectedTags: selectedTags)
        let locationTags = Set<String>(entries.flatMap(\.filterTags)).intersection(FeatureSelectorTagPriority.locationTags)
        var tagSet = locationTags.union(Set<String>(scopedEntries.flatMap(\.filterTags)))
        if entries.contains(where: { isHidden($0, hiddenIDs: hiddenIDs) }) {
            tagSet.insert(localized("Hidden"))
        }
        return FeatureSelectorTagPriority.priority.compactMap { tagSet.contains($0) ? $0 : nil }
    }

    static func filteredEntries(
        entries: [FeatureModelSelectorEntry],
        selectedTags: Set<String>,
        hiddenIDs: Set<String> = []
    ) -> [FeatureModelSelectorEntry] {
        let hiddenTag = localized("Hidden")
        let visibilityFiltered = entries.filter { entry in
            let hidden = isHidden(entry, hiddenIDs: hiddenIDs)
            return selectedTags.contains(hiddenTag) ? hidden : !hidden
        }
        let matchingEntries: [FeatureModelSelectorEntry]
        if selectedTags.isEmpty {
            matchingEntries = visibilityFiltered
        } else {
            let selectedStatusTags = selectedTags
                .intersection(statusFilterTags)
                .subtracting([hiddenTag])
            let requiredTags = selectedTags.subtracting(statusFilterTags)
            matchingEntries = visibilityFiltered.filter { entry in
                let entryTags = Set(entry.filterTags)
                guard requiredTags.isSubset(of: entryTags) else { return false }
                if selectedStatusTags.isEmpty {
                    return true
                }
                return !entryTags.intersection(selectedStatusTags).isEmpty
            }
        }

        return matchingEntries.enumerated()
            .sorted { lhs, rhs in
                let lhsInUse = !lhs.element.usageLocations.isEmpty
                let rhsInUse = !rhs.element.usageLocations.isEmpty
                if lhsInUse != rhsInUse {
                    return lhsInUse && !rhsInUse
                }
                return lhs.offset < rhs.offset
            }
            .map(\.element)
    }

    static func toggledTags(
        current: Set<String>,
        tag: String,
        entries: [FeatureModelSelectorEntry],
        hiddenIDs: Set<String> = []
    ) -> Set<String> {
        var next = current
        if next.contains(tag) {
            next.remove(tag)
        } else {
            if FeatureSelectorTagPriority.locationTags.contains(tag) {
                next.subtract(FeatureSelectorTagPriority.locationTags)
            }
            if statusFilterTags.contains(tag) {
                next.subtract(statusFilterTags)
            }
            next.insert(tag)
        }
        return next.intersection(Set<String>(availableTags(
            entries: entries,
            selectedTags: next,
            hiddenIDs: hiddenIDs
        )))
    }

    private static func isHidden(
        _ entry: FeatureModelSelectorEntry,
        hiddenIDs: Set<String>
    ) -> Bool {
        !entry.filterTags.contains(localized("Installed"))
            && ModelVisibilityStore.isModelHidden(entry.id, in: hiddenIDs)
    }
}

enum FeatureModelConfigurationRouting {
    static func canConfigure(_ selectionID: FeatureModelSelectionID) -> Bool {
        if let selection = selectionID.asrSelection {
            switch selection {
            case .dictation:
                return false
            case .mlx, .remote:
                return true
            }
        }

        if let selection = selectionID.textSelection {
            switch selection {
            case .appleIntelligence:
                return false
            case .localLLM, .remoteLLM:
                return true
            }
        }

        if case .localGGUF? = selectionID.translationSelection {
            return false
        }
        return false
    }
}

private struct FeatureModelSelectorRow: View {
    let entry: FeatureModelSelectorEntry
    let isSelected: Bool
    let onSelect: () -> Void
    let onConfigure: (() -> Void)?
    let visibilityAction: ModelTableAction?
    let titleOverride: String?
    let showsEngine: Bool
    let showsTags: Bool
    let showsIcon: Bool
    private let surface: FeatureModelSelectorRowSurface

    private var hintText: String? {
        if let disabledReason = entry.disabledReason, !entry.isSelectable {
            return disabledReason
        }

        let trimmed = entry.statusText
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let genericStatuses = Set([
            localized("Installed"),
            localized("Not installed"),
            localized("Configured"),
            localized("Not configured"),
            localized("Available on this Mac"),
            localized("Works immediately with no model download.")
        ] as [String])

        guard !trimmed.isEmpty, !genericStatuses.contains(trimmed) else { return nil }
        return trimmed
    }

    init(
        entry: FeatureModelSelectorEntry,
        isSelected: Bool,
        onSelect: @escaping () -> Void,
        onConfigure: (() -> Void)? = nil,
        visibilityAction: ModelTableAction? = nil,
        titleOverride: String? = nil,
        showsEngine: Bool = true,
        showsTags: Bool = true,
        showsIcon: Bool = true,
        surface: FeatureModelSelectorRowSurface = .card
    ) {
        self.entry = entry
        self.isSelected = isSelected
        self.onSelect = onSelect
        self.onConfigure = onConfigure
        self.visibilityAction = visibilityAction
        self.titleOverride = titleOverride
        self.showsEngine = showsEngine
        self.showsTags = showsTags
        self.showsIcon = showsIcon
        self.surface = surface
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .center, spacing: 8) {
                    if showsIcon {
                        ModelLogoView(
                            key: entry.modelLogoKey,
                            fallbackTitle: titleOverride ?? entry.title,
                            size: 18
                        )
                    }

                    Text(titleOverride ?? entry.title)
                        .font(.headline)

                    if let badgeText = entry.badgeText {
                        ModelBadgeView(badgeText: badgeText, showsCapsuleForText: true)
                    }

                    if showsEngine {
                        Text(entry.engine)
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(SettingsUIStyle.groupedFillColor)
                            )
                    }

                    if let hintText {
                        Text(hintText)
                            .font(.caption.weight(entry.isSelectable ? .regular : .semibold))
                            .foregroundStyle(entry.isSelectable ? Color.secondary : Color.orange)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }

                    Spacer(minLength: 0)
                }

                HStack(spacing: 12) {
                    FeatureSelectorMetaText(title: localized("Size"), value: entry.sizeText)
                    FeatureSelectorMetaText(title: localized("Score"), value: entry.ratingText)
                    if !entry.usageLocations.isEmpty {
                        FeatureSelectorMetaText(
                            title: localized("Usage"),
                            value: entry.usageLocations.joined(separator: " · ")
                        )
                    }
                }

                if showsTags {
                    FeatureSelectorTagStrip(tags: entry.displayTags)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 0)

            HStack(spacing: 8) {
                if let onConfigure {
                    Button(localized("Configure"), action: onConfigure)
                        .buttonStyle(SettingsCompactActionButtonStyle())
                }

                if isSelected {
                    HStack(spacing: 5) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 11, weight: .semibold))
                        Text(localized("Selected"))
                            .font(.caption.weight(.semibold))
                    }
                    .foregroundStyle(Color.accentColor)
                    .padding(.horizontal, 9)
                    .frame(height: 28)
                    .background(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .fill(Color.accentColor.opacity(0.10))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .strokeBorder(Color.accentColor.opacity(0.24), lineWidth: 1)
                    )
                } else if entry.isSelectable {
                    Button(localized("Select")) {
                        onSelect()
                    }
                    .buttonStyle(SettingsCompactActionButtonStyle())
                }

                if let visibilityAction {
                    Button(visibilityAction.title, action: visibilityAction.handler)
                        .buttonStyle(SettingsCompactActionButtonStyle())
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .modifier(FeatureModelSelectorRowSurfaceModifier(surface: surface, isSelected: isSelected))
    }
}

private enum FeatureModelSelectorRowSurface {
    case card
    case listItem
}

private struct FeatureModelSelectorRowSurfaceModifier: ViewModifier {
    let surface: FeatureModelSelectorRowSurface
    let isSelected: Bool

    func body(content: Content) -> some View {
        switch surface {
        case .card:
            content
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(SettingsUIStyle.controlFillColor.opacity(0.94))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(
                            isSelected ? Color.accentColor.opacity(0.34) : SettingsUIStyle.modelCardBorderColor,
                            lineWidth: 1
                        )
                )
        case .listItem:
            content
                .background(Color.clear)
        }
    }
}

private struct FeatureModelSelectorGroupCard: View {
    let group: FeatureModelSelectorGroupSection
    let selectedID: FeatureModelSelectionID
    let isExpanded: Bool
    let onToggle: () -> Void
    let onSelect: (FeatureModelSelectionID) -> Void
    let onConfigure: (FeatureModelSelectionID) -> Void
    let visibilityActionForEntry: (FeatureModelSelectorEntry) -> ModelTableAction?

    var body: some View {
        groupContent
            .clipped()
            .animation(.easeInOut(duration: 0.18), value: isExpanded)
            .padding(.horizontal, 12)
            .padding(.vertical, 11)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(SettingsUIStyle.modelGroupFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(SettingsUIStyle.modelCardBorderColor, lineWidth: 1)
            )
    }

    private var groupContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            groupHeader
            expandedEntries
        }
    }

    private var groupHeader: some View {
        HStack(alignment: .top, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    onToggle()
                }
            } label: {
                VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .center, spacing: 8) {
                    ModelLogoView(key: group.modelLogoKey, fallbackTitle: group.title, size: 18)

                    Text(group.title)
                        .font(.headline)

                    if let badgeText = group.badgeText {
                        ModelBadgeView(badgeText: badgeText, showsCapsuleForText: true)
                    }

                    Text(group.engine)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(
                            Capsule(style: .continuous)
                                .fill(SettingsUIStyle.groupedFillColor)
                        )

                    Spacer(minLength: 0)

                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 18, height: 18)
                }

                HStack(spacing: 12) {
                    FeatureSelectorMetaText(title: localized("Models"), value: "\(group.entries.count)")
                    FeatureSelectorMetaText(title: localized("Installed"), value: "\(group.installedCount)/\(group.entries.count)")
                    FeatureSelectorMetaText(title: localized("Score"), value: group.ratingText)
                    if !group.usageLocations.isEmpty {
                        FeatureSelectorMetaText(
                            title: localized("Usage"),
                            value: group.usageLocations.joined(separator: " · ")
                        )
                    }
                }

                if !group.tags.isEmpty {
                    FeatureSelectorTagStrip(tags: group.tags)
                }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

        }
    }

    @ViewBuilder
    private var expandedEntries: some View {
        if isExpanded {
            VStack(spacing: 0) {
                ForEach(Array(group.entries.enumerated()), id: \.element.id) { index, entry in
                    FeatureModelSelectorRow(
                        entry: entry,
                        isSelected: entry.selectionID == selectedID,
                        onSelect: { onSelect(entry.selectionID) },
                        onConfigure: FeatureModelConfigurationRouting.canConfigure(entry.selectionID)
                            ? { onConfigure(entry.selectionID) }
                            : nil,
                        visibilityAction: visibilityActionForEntry(entry),
                        titleOverride: entry.groupedVariantTitle,
                        showsEngine: false,
                        showsTags: false,
                        showsIcon: false,
                        surface: .listItem
                    )

                    if index < group.entries.count - 1 {
                        Divider()
                            .padding(.leading, 12)
                    }
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(SettingsUIStyle.modelGroupListFillColor)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(SettingsUIStyle.modelCardBorderColor, lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .clipped()
            .transition(.opacity)
        }
    }
}

private struct FeatureSelectorEmptyState: View {
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "line.3.horizontal.decrease.circle")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(localized("No models match the selected tags."))
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 32)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(SettingsUIStyle.controlFillColor.opacity(0.9))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
    }
}

private struct FeatureModelTagChip: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    Capsule(style: .continuous)
                        .fill(isSelected || isHovered ? Color.accentColor.opacity(0.18) : SettingsUIStyle.controlFillColor)
                )
                .overlay(
                    Capsule(style: .continuous)
                        .stroke(isSelected || isHovered ? Color.accentColor.opacity(0.28) : SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                )
                .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected || isHovered ? Color.accentColor : .primary)
        .onHover { isHovered = $0 }
    }
}

private struct FeatureSelectorMetaText: View {
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.weight(.medium))
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .fixedSize(horizontal: false, vertical: true)
        .layoutPriority(1)
    }
}

private struct FeatureSelectorTagStrip: View {
    let tags: [String]

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 6) {
                ForEach(tags, id: \.self) { tag in
                    tagChip(tag)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 6) {
                ForEach(Array(tags.prefix(5)), id: \.self) { tag in
                    tagChip(tag)
                }
                if tags.count > 5 {
                    tagChip("+\(tags.count - 5)")
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func tagChip(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule(style: .continuous)
                    .fill(SettingsUIStyle.groupedFillColor)
            )
            .overlay(
                Capsule(style: .continuous)
                    .stroke(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
            )
    }
}

private enum FeatureSelectorTagPriority {
    static var locationTags: Set<String> {
        Set<String>([localized("Local"), localized("Remote")])
    }

    static var groups: [[String]] {
        [
            [localized("Local"), localized("Remote")],
            [localized("Fast"), localized("Balanced"), localized("Accurate"), localized("Realtime")],
            [localized("Installed"), localized("Configured"), localized("In Use"), localized("Hidden")]
        ]
    }

    static var priority: [String] {
        groups.flatMap { $0 }
    }
}
