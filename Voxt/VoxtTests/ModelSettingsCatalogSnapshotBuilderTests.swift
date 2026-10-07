// ModelSettingsCatalogSnapshotBuilderTests.swift
// Provides Model Settings Catalog Snapshot Builder Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class ModelSettingsCatalogSnapshotBuilderTests: XCTestCase {
    func testModelCatalogStatusTagsAreMutuallyExclusive() {
        let inUseTag = AppLocalization.localizedString("In Use")

        let installedOnly = ModelCatalogTag.toggledTags(current: [], tag: installedTag)
        XCTAssertEqual(installedOnly, [installedTag])

        let configuredOnly = ModelCatalogTag.toggledTags(current: installedOnly, tag: configuredTag)
        XCTAssertEqual(configuredOnly, [configuredTag])

        let inUseOnly = ModelCatalogTag.toggledTags(current: configuredOnly, tag: inUseTag)
        XCTAssertEqual(inUseOnly, [inUseTag])

        XCTAssertTrue(ModelCatalogTag.toggledTags(current: inUseOnly, tag: inUseTag).isEmpty)
    }

    func testBuildPrioritizesEntriesAlreadyInUse() {
        let snapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [
                makeEntry(id: "local-idle", filterTags: [localTag, installedTag]),
                makeEntry(id: "remote-in-use", filterTags: [remoteTag, configuredTag], usageLocations: ["Translation"])
            ],
            selectedTags: []
        )

        XCTAssertEqual(snapshot.allEntries.map(\.id), ["remote-in-use", "local-idle"])
    }

    func testBuildKeepsBothLocationTagsVisibleWhenFilteringToLocal() {
        let snapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [
                makeEntry(id: "local", filterTags: [localTag, fastTag]),
                makeEntry(id: "remote", filterTags: [remoteTag, configuredTag])
            ],
            selectedTags: [localTag]
        )

        XCTAssertEqual(snapshot.availableTagGroups.first, [localTag, remoteTag])
        XCTAssertEqual(snapshot.filteredEntries.map(\.id), ["local"])
    }

    func testBuildFiltersEntriesBySelectedTagSubset() {
        let snapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [
                makeEntry(id: "installed-local", filterTags: [localTag, installedTag]),
                makeEntry(id: "plain-local", filterTags: [localTag]),
                makeEntry(id: "configured-remote", filterTags: [remoteTag, configuredTag])
            ],
            selectedTags: [localTag, installedTag]
        )

        XCTAssertEqual(snapshot.filteredEntries.map(\.id), ["installed-local"])
    }

    func testBuildAppliesSingleSelectedStatusFilter() {
        let snapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [
                makeEntry(id: "installed-local", filterTags: [localTag, installedTag]),
                makeEntry(id: "configured-remote", filterTags: [remoteTag, configuredTag]),
                makeEntry(id: "plain-local", filterTags: [localTag])
            ],
            selectedTags: [configuredTag]
        )

        XCTAssertEqual(snapshot.filteredEntries.map(\.id), ["configured-remote"])
    }

    func testBuildHidesModelByDefaultAndShowsItWithHiddenFilter() {
        let hiddenID = "local-hidden"
        let hiddenIDs = Set([ModelVisibilityStore.modelKey(hiddenID)])

        let defaultSnapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [
                makeEntry(id: hiddenID, filterTags: [localTag]),
                makeEntry(id: "local-visible", filterTags: [localTag])
            ],
            selectedTags: [],
            hiddenIDs: hiddenIDs
        )
        XCTAssertEqual(defaultSnapshot.filteredEntries.map(\.id), ["local-visible"])
        XCTAssertTrue(defaultSnapshot.availableTags.contains(hiddenTag))

        let hiddenSnapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [
                makeEntry(id: hiddenID, filterTags: [localTag]),
                makeEntry(id: "local-visible", filterTags: [localTag])
            ],
            selectedTags: [hiddenTag],
            hiddenIDs: hiddenIDs
        )
        XCTAssertEqual(hiddenSnapshot.filteredEntries.map(\.id), [hiddenID])
    }

    func testBuildRemovesAGroupWhenAllChildEntriesAreHidden() {
        let first = makeEntry(id: "qwen3-small", title: "Qwen3 Small", engine: "MLX Audio", filterTags: [localTag])
        let second = makeEntry(id: "qwen3-large", title: "Qwen3 Large", engine: "MLX Audio", filterTags: [localTag])
        let snapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [first, second],
            selectedTags: [],
            hiddenIDs: [
                ModelVisibilityStore.modelKey(first.id),
                ModelVisibilityStore.modelKey(second.id)
            ]
        )

        XCTAssertTrue(snapshot.filteredEntries.isEmpty)
        XCTAssertTrue(snapshot.displayItems.isEmpty)
    }

    func testInstalledEntryCannotBeHiddenByStoredVisibilityID() {
        let installedID = "installed-model"
        let snapshot = ModelSettingsCatalogSnapshotBuilder.build(
            entries: [makeEntry(id: installedID, filterTags: [localTag, installedTag])],
            selectedTags: [],
            hiddenIDs: [ModelVisibilityStore.modelKey(installedID)]
        )

        XCTAssertEqual(snapshot.filteredEntries.map(\.id), [installedID])
    }

    private func makeEntry(
        id: String,
        title: String? = nil,
        engine: String = "MLX",
        filterTags: [String],
        usageLocations: [String] = []
    ) -> ModelCatalogEntry {
        ModelCatalogEntry(
            id: id,
            title: title ?? id,
            engine: engine,
            sizeText: "",
            ratingText: "",
            filterTags: filterTags,
            displayTags: filterTags,
            statusText: "",
            usageLocations: usageLocations,
            badgeText: nil,
            primaryAction: nil,
            secondaryActions: []
        )
    }

    private var localTag: String { AppLocalization.localizedString("Local") }
    private var remoteTag: String { AppLocalization.localizedString("Remote") }
    private var fastTag: String { AppLocalization.localizedString("Fast") }
    private var installedTag: String { AppLocalization.localizedString("Installed") }
    private var configuredTag: String { AppLocalization.localizedString("Configured") }
    private var hiddenTag: String { AppLocalization.localizedString("Hidden") }
}
