// FeatureModelSelectorFilteringTests.swift
// Provides Feature Model Selector Filtering Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class FeatureModelSelectorFilteringTests: XCTestCase {
    func testDefaultSelectedTagsDoesNotPreselectMutuallyExclusiveStatus() {
        let entries = [
            makeEntry(
                id: .dictation,
                filterTags: [localized("Local"), localized("Installed")]
            ),
            makeEntry(
                id: .remoteLLM(.openAI),
                filterTags: [localized("Remote"), localized("Configured")]
            ),
            makeEntry(
                id: .localLLM("repo"),
                filterTags: [localized("Local"), localized("Fast")]
            )
        ]

        let tags = FeatureModelSelectorFiltering.defaultSelectedTags(entries: entries)

        XCTAssertTrue(tags.isEmpty)
    }

    func testToggledTagsTreatsLocalAndRemoteAsMutuallyExclusive() {
        let entries = [
            makeEntry(
                id: .dictation,
                filterTags: [localized("Local"), localized("Installed")]
            ),
            makeEntry(
                id: .remoteASR(.openAIWhisper),
                filterTags: [localized("Remote"), localized("Configured")]
            )
        ]

        let localOnly = FeatureModelSelectorFiltering.toggledTags(
            current: [],
            tag: localized("Local"),
            entries: entries
        )
        XCTAssertEqual(localOnly, Set([localized("Local")]))

        let remoteOnly = FeatureModelSelectorFiltering.toggledTags(
            current: localOnly,
            tag: localized("Remote"),
            entries: entries
        )
        XCTAssertEqual(remoteOnly, Set([localized("Remote")]))

        let cleared = FeatureModelSelectorFiltering.toggledTags(
            current: remoteOnly,
            tag: localized("Remote"),
            entries: entries
        )
        XCTAssertTrue(cleared.isEmpty)
    }

    func testToggledTagsTreatsStatusFiltersAsMutuallyExclusive() {
        let installed = localized("Installed")
        let configured = localized("Configured")
        let inUse = localized("In Use")
        let entries = [
            makeEntry(id: .localLLM("installed"), filterTags: [installed]),
            makeEntry(id: .remoteLLM(.openAI), filterTags: [configured]),
            makeEntry(id: .mlx("in-use"), filterTags: [inUse])
        ]

        let installedOnly = FeatureModelSelectorFiltering.toggledTags(
            current: [], tag: installed, entries: entries
        )
        XCTAssertEqual(installedOnly, [installed])

        let configuredOnly = FeatureModelSelectorFiltering.toggledTags(
            current: installedOnly, tag: configured, entries: entries
        )
        XCTAssertEqual(configuredOnly, [configured])

        let inUseOnly = FeatureModelSelectorFiltering.toggledTags(
            current: configuredOnly, tag: inUse, entries: entries
        )
        XCTAssertEqual(inUseOnly, [inUse])

        let cleared = FeatureModelSelectorFiltering.toggledTags(
            current: inUseOnly, tag: inUse, entries: entries
        )
        XCTAssertTrue(cleared.isEmpty)
    }

    func testFilteredEntriesAppliesSingleSelectedStatus() {
        let entries = [
            makeEntry(
                id: .localLLM("installed"),
                filterTags: [localized("Local"), localized("Installed")],
                usageLocations: []
            ),
            makeEntry(
                id: .remoteLLM(.openAI),
                filterTags: [localized("Remote"), localized("Configured")],
                usageLocations: [localized("Translation")]
            ),
            makeEntry(
                id: .localLLM("idle-configured"),
                filterTags: [localized("Local"), localized("Configured")],
                usageLocations: []
            )
        ]

        let filtered = FeatureModelSelectorFiltering.filteredEntries(
            entries: entries,
            selectedTags: Set([localized("Configured")])
        )

        XCTAssertEqual(filtered.map(\.selectionID), [
            .remoteLLM(.openAI),
            .localLLM("idle-configured")
        ])
    }


    func testAvailableTagsDoNotExposeMultilingualFilter() {
        let entries = [
            makeEntry(
                id: .mlx("mlx-community/Qwen3-ASR-0.6B-4bit"),
                filterTags: [localized("Local"), localized("Multilingual"), localized("Fast")]
            )
        ]

        let availableTags = FeatureModelSelectorFiltering.availableTags(
            entries: entries,
            selectedTags: []
        )

        XCTAssertFalse(availableTags.contains(localized("Multilingual")))
        XCTAssertEqual(availableTags, [localized("Local"), localized("Fast")])
    }

    func testHiddenEntriesAreExcludedByDefaultAndAvailableWithHiddenFilter() {
        let hiddenEntry = makeEntry(
            id: .localLLM("hidden"),
            filterTags: [localized("Local"), localized("Fast")]
        )
        let visibleEntry = makeEntry(
            id: .localLLM("visible"),
            filterTags: [localized("Local"), localized("Fast")]
        )
        let hiddenIDs = Set([ModelVisibilityStore.modelKey(hiddenEntry.id)])

        XCTAssertEqual(
            FeatureModelSelectorFiltering.filteredEntries(
                entries: [hiddenEntry, visibleEntry],
                selectedTags: [],
                hiddenIDs: hiddenIDs
            ).map(\.id),
            [visibleEntry.id]
        )
        XCTAssertEqual(
            FeatureModelSelectorFiltering.filteredEntries(
                entries: [hiddenEntry, visibleEntry],
                selectedTags: [localized("Hidden")],
                hiddenIDs: hiddenIDs
            ).map(\.id),
            [hiddenEntry.id]
        )
    }

    private func makeEntry(
        id: FeatureModelSelectionID,
        filterTags: [String],
        usageLocations: [String] = []
    ) -> FeatureModelSelectorEntry {
        FeatureModelSelectorEntry(
            selectionID: id,
            title: id.rawValue,
            engine: "engine",
            sizeText: "size",
            ratingText: "4.0",
            filterTags: filterTags,
            displayTags: filterTags,
            statusText: "",
            usageLocations: usageLocations,
            badgeText: nil,
            isSelectable: true,
            disabledReason: nil
        )
    }

    private func localized(_ key: String) -> String {
        AppLocalization.localizedString(key)
    }
}
