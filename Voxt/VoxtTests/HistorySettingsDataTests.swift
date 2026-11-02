// HistorySettingsDataTests.swift
// Provides History Settings Data Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class HistorySettingsDataTests: XCTestCase {
    func testHistoryTabsUseExpectedDisplayOrder() {
        XCTAssertEqual(
            HistoryFilterTab.allCases,
            [.transcription, .translation, .rewrite, .note, .transcript]
        )
    }

    func testHistoryTabsReuseCorrespondingFeatureTabs() {
        XCTAssertEqual(
            HistoryFilterTab.allCases.map(\.correspondingFeatureTab.rawValue),
            [
                FeatureSettingsTab.transcription,
                .translation,
                .rewrite,
                .note,
                .meeting
            ].map(\.rawValue)
        )
    }

    func testFilteredEntriesUsesSelectedHistoryTab() {
        let entries = [
            makeHistoryEntry(kind: .normal, text: "a"),
            makeHistoryEntry(kind: .translation, text: "b"),
            makeHistoryEntry(kind: .rewrite, text: "c"),
            makeHistoryEntry(kind: .transcript, text: "d")
        ]

        XCTAssertEqual(
            HistorySettingsData.filteredEntries(for: .transcription, allEntries: entries).map(\.text),
            ["a"]
        )
        XCTAssertEqual(
            HistorySettingsData.filteredEntries(for: .translation, allEntries: entries).map(\.text),
            ["b"]
        )
        XCTAssertEqual(
            HistorySettingsData.filteredEntries(for: .rewrite, allEntries: entries).map(\.text),
            ["c"]
        )
        XCTAssertEqual(
            HistorySettingsData.filteredEntries(for: .note, allEntries: entries).map(\.text),
            []
        )
    }

    func testEmptyStatePrefersHistoryAndNoteSpecificMessages() {
        let note = makeNote(title: "todo")
        let entry = makeHistoryEntry(kind: .normal, text: "Voxt")

        XCTAssertEqual(
            HistorySettingsData.emptyState(
                selectedFilter: .note,
                allEntries: [],
                filteredEntries: [],
                notes: []
            ),
            .noNotes
        )
        XCTAssertEqual(
            HistorySettingsData.emptyState(
                selectedFilter: .transcription,
                allEntries: [],
                filteredEntries: [],
                notes: [note]
            ),
            .noHistory
        )
        XCTAssertEqual(
            HistorySettingsData.emptyState(
                selectedFilter: .rewrite,
                allEntries: [entry],
                filteredEntries: [],
                notes: [note]
            ),
            .noEntriesInCategory
        )
        XCTAssertEqual(
            HistorySettingsData.emptyState(
                selectedFilter: .note,
                allEntries: [entry],
                filteredEntries: [entry],
                notes: [note]
            ),
            .none
        )
    }

    func testPaginationHelpersRespectLimits() {
        let values = Array(0..<5)

        XCTAssertEqual(HistorySettingsData.visibleEntries(from: values, visibleLimit: 3), [0, 1, 2])
        XCTAssertTrue(HistorySettingsData.hasMoreItems(in: values, visibleLimit: 3))
        XCTAssertEqual(HistorySettingsData.nextVisibleLimit(currentLimit: 3, pageSize: 2, totalCount: 5), 5)
        XCTAssertEqual(HistorySettingsData.normalizedVisibleLimit(currentLimit: 1, pageSize: 4, totalCount: 2), 4)
    }

    func testNoteFilteringSupportsSingleAndMultipleStatuses() {
        let todo = makeNote(title: "Todo", status: .todo, updatedAt: 10)
        let progress = makeNote(title: "Progress", status: .inProgress, updatedAt: 30)
        let done = makeNote(title: "Done", status: .done, updatedAt: 20)

        XCTAssertEqual(
            HistorySettingsData.filteredNotes(
                [todo, progress, done],
                statuses: [.inProgress],
                query: ""
            ).map(\.id),
            [progress.id]
        )
        XCTAssertEqual(
            HistorySettingsData.filteredNotes(
                [todo, progress, done],
                statuses: [.inProgress, .done],
                query: ""
            ).map(\.id),
            [progress.id, done.id]
        )
    }

    func testNoteSectionsUseFloatPanelStatusOrderAndPreserveItemOrder() {
        let todo = makeNote(title: "Todo", status: .todo, updatedAt: 10)
        let progress = makeNote(title: "Progress", status: .inProgress, updatedAt: 30)
        let done = makeNote(title: "Done", status: .done, updatedAt: 20)
        let backlog = makeNote(title: "Backlog", status: .backlog, updatedAt: 40)

        let sections = HistorySettingsData.noteSections(
            from: [todo, progress, done, backlog]
        )

        XCTAssertEqual(sections.map(\.status), [.inProgress, .todo, .done, .backlog])
        XCTAssertEqual(sections.flatMap(\.items).map(\.id), [progress.id, todo.id, done.id, backlog.id])
    }

    func testLinearNoteSectionsUseBacklogFirstOrder() {
        XCTAssertEqual(
            HistorySettingsData.linearNoteSectionOrder,
            [.backlog, .inProgress, .todo, .done]
        )
    }

    func testLinearNotesOnlyLimitCompletedItems() {
        let todoNotes = (0..<3).map {
            makeNote(title: "Todo \($0)", status: .todo, updatedAt: TimeInterval($0))
        }
        let doneNotes = (0..<25).map {
            makeNote(title: "Done \($0)", status: .done, updatedAt: TimeInterval($0))
        }
        let notes = todoNotes + doneNotes

        XCTAssertEqual(
            HistorySettingsData.visibleLinearNotes(
                from: notes,
                status: .todo,
                completedVisibleLimit: 20
            ).count,
            3
        )
        XCTAssertEqual(
            HistorySettingsData.visibleLinearNotes(
                from: notes,
                status: .done,
                completedVisibleLimit: 20
            ).count,
            20
        )
    }

    func testNoteStatusToggleNeverLeavesAnEmptySelection() {
        XCTAssertEqual(
            HistorySettingsData.toggledNoteStatuses([.todo], status: .todo),
            [.todo]
        )
        XCTAssertEqual(
            HistorySettingsData.toggledNoteStatuses([.todo, .done], status: .done),
            [.todo]
        )
        XCTAssertEqual(
            HistorySettingsData.toggledNoteStatuses([.todo], status: .done),
            [.todo, .done]
        )
    }
}

private extension HistorySettingsDataTests {
    func makeHistoryEntry(kind: TranscriptionHistoryKind, text: String) -> TranscriptionHistoryEntry {
        TranscriptionHistoryEntry(
            id: UUID(),
            text: text,
            createdAt: Date(timeIntervalSince1970: 1),
            transcriptionEngine: "engine",
            transcriptionModel: "model",
            enhancementMode: "mode",
            enhancementModel: "enhanced",
            kind: kind,
            isTranslation: kind == .translation,
            audioDurationSeconds: nil,
            transcriptionProcessingDurationSeconds: nil,
            llmDurationSeconds: nil,
            focusedAppName: nil,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: nil,
            remoteASRModel: nil,
            remoteASREndpoint: nil,
            remoteLLMProvider: nil,
            remoteLLMModel: nil,
            remoteLLMEndpoint: nil,
            audioRelativePath: nil,
            whisperWordTimings: nil,
            dictionaryHitTerms: [],
            dictionaryCorrectedTerms: [],
            dictionarySuggestedTerms: []
        )
    }

    func makeNote(
        title: String,
        status: VoxtNoteStatus = .todo,
        updatedAt: TimeInterval = 1
    ) -> VoxtNoteItem {
        VoxtNoteItem(
            id: UUID(),
            sessionID: UUID(),
            createdAt: Date(timeIntervalSince1970: 1),
            text: title,
            title: title,
            titleGenerationState: .generated,
            updatedAt: Date(timeIntervalSince1970: updatedAt),
            status: status
        )
    }
}
