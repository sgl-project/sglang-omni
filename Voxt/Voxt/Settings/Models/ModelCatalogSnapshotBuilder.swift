// ModelCatalogSnapshotBuilder.swift
// Provides Model Catalog Snapshot Builder for model settings.

import Foundation

struct ModelSettingsCatalogSnapshot {
    let allEntries: [ModelCatalogEntry]
    let availableTags: [String]
    let availableTagGroups: [[String]]
    let filteredEntries: [ModelCatalogEntry]
    let displayItems: [ModelCatalogDisplayItem]

    static let empty = ModelSettingsCatalogSnapshot(
        allEntries: [],
        availableTags: [],
        availableTagGroups: [],
        filteredEntries: [],
        displayItems: []
    )
}

enum ModelSettingsCatalogSnapshotBuilder {
    static func build(
        entries: [ModelCatalogEntry],
        selectedTags: Set<String>,
        hiddenIDs: Set<String> = []
    ) -> ModelSettingsCatalogSnapshot {
        let prioritizedEntries = prioritize(entries)
        let locationTags = Set(prioritizedEntries.flatMap(\.filterTags)).intersection(ModelCatalogTag.locationTags)
        let locationScopedEntries = filterEntriesByLocationTag(
            prioritizedEntries,
            selectedTags: selectedTags
        )

        var tagSet = locationTags.union(Set(locationScopedEntries.flatMap(\.filterTags)))
        if prioritizedEntries.contains(where: { isHidden($0, hiddenIDs: hiddenIDs) }) {
            tagSet.insert(AppLocalization.localizedString("Hidden"))
        }
        let availableTags = ModelCatalogTag.priority.compactMap { tagSet.contains($0) ? $0 : nil }
        let availableTagGroups = groupedAvailableTags(from: availableTags)
        let filteredEntries = filterEntries(
            prioritizedEntries,
            selectedTags: selectedTags,
            hiddenIDs: hiddenIDs
        )

        return ModelSettingsCatalogSnapshot(
            allEntries: prioritizedEntries,
            availableTags: availableTags,
            availableTagGroups: availableTagGroups,
            filteredEntries: filteredEntries,
            displayItems: LocalModelSeriesGrouping.modelCatalogItems(from: filteredEntries)
        )
    }

    private static func prioritize(_ entries: [ModelCatalogEntry]) -> [ModelCatalogEntry] {
        entries.enumerated()
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

    private static func filterEntriesByLocationTag(
        _ entries: [ModelCatalogEntry],
        selectedTags: Set<String>
    ) -> [ModelCatalogEntry] {
        let localTag = AppLocalization.localizedString("Local")
        if selectedTags.contains(localTag) {
            return entries.filter { $0.filterTags.contains(localTag) }
        }

        let remoteTag = AppLocalization.localizedString("Remote")
        if selectedTags.contains(remoteTag) {
            return entries.filter { $0.filterTags.contains(remoteTag) }
        }

        return entries
    }

    private static func groupedAvailableTags(from availableTags: [String]) -> [[String]] {
        let available = Set(availableTags)
        var availableTagGroups = [[String]]()

        let locationGroup = ModelCatalogTag.groups[0].filter { available.contains($0) }
        if !locationGroup.isEmpty {
            availableTagGroups.append(locationGroup)
        }

        availableTagGroups.append(
            contentsOf: ModelCatalogTag.groups.dropFirst()
                .map { $0.filter { available.contains($0) } }
                .filter { !$0.isEmpty }
        )

        return availableTagGroups
    }

    private static func filterEntries(
        _ entries: [ModelCatalogEntry],
        selectedTags: Set<String>,
        hiddenIDs: Set<String>
    ) -> [ModelCatalogEntry] {
        let hiddenTag = AppLocalization.localizedString("Hidden")
        let visibilityFiltered = entries.filter { entry in
            let hidden = isHidden(entry, hiddenIDs: hiddenIDs)
            return selectedTags.contains(hiddenTag) ? hidden : !hidden
        }
        guard !selectedTags.isEmpty else { return visibilityFiltered }
        let selectedStatusTags = selectedTags
            .intersection(ModelCatalogTag.statusFilterTags)
            .subtracting([hiddenTag])
        let requiredTags = selectedTags.subtracting(ModelCatalogTag.statusFilterTags)
        return visibilityFiltered.filter { entry in
            let entryTags = Set(entry.filterTags)
            guard requiredTags.isSubset(of: entryTags) else { return false }
            if selectedStatusTags.isEmpty {
                return true
            }
            return !entryTags.intersection(selectedStatusTags).isEmpty
        }
    }

    private static func isHidden(
        _ entry: ModelCatalogEntry,
        hiddenIDs: Set<String>
    ) -> Bool {
        !entry.filterTags.contains(AppLocalization.localizedString("Installed"))
            && ModelVisibilityStore.isModelHidden(entry.id, in: hiddenIDs)
    }
}
