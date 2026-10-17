// DevelopmentDataSeeder.swift
// Provides Development Data Seeder for core app behavior.

import Foundation

#if DEBUG
enum DevelopmentDataSeeder {
    static func seed(dictionaryCount: Int, historyCount: Int) throws {
        let dictionaryRepository = DictionaryRepository(migrateLegacyJSON: false)
        let historyRepository = HistoryRepository(migrateLegacyJSON: false)

        try dictionaryRepository.upsertAll(makeDictionaryEntries(count: dictionaryCount))
        try historyRepository.upsertAll(makeHistoryEntries(count: historyCount))
    }

    private static func makeDictionaryEntries(count: Int) -> [DictionaryEntry] {
        let now = Date()
        return (0..<count).map { index in
            let groupID = index.isMultiple(of: 5) ? deterministicUUID(prefix: 9, index: index % 20) : nil
            let groupName = groupID.map { _ in "Seed Group \(index % 20)" }
            return DictionaryEntry(
                id: deterministicUUID(prefix: 1, index: index),
                term: "Seed Term \(index)",
                normalizedTerm: "seed term \(index)",
                groupID: groupID,
                groupNameSnapshot: groupName,
                source: index.isMultiple(of: 3) ? .auto : .manual,
                createdAt: now.addingTimeInterval(TimeInterval(-index)),
                updatedAt: now.addingTimeInterval(TimeInterval(-index)),
                lastMatchedAt: index.isMultiple(of: 4) ? now.addingTimeInterval(TimeInterval(-index / 2)) : nil,
                matchCount: index % 37,
                status: .active,
                observedVariants: [
                    ObservedVariant(
                        id: deterministicUUID(prefix: 2, index: index),
                        text: "Seed Variant \(index)",
                        normalizedText: "seed variant \(index)",
                        count: max(1, index % 13),
                        lastSeenAt: now.addingTimeInterval(TimeInterval(-index / 3)),
                        confidence: index.isMultiple(of: 7) ? .high : .medium
                    )
                ],
                replacementTerms: [
                    DictionaryReplacementTerm(
                        id: deterministicUUID(prefix: 3, index: index),
                        text: "Seed Alias \(index)",
                        normalizedText: "seed alias \(index)"
                    )
                ]
            )
        }
    }

    private static func makeHistoryEntries(count: Int) -> [TranscriptionHistoryEntry] {
        let now = Date()
        return (0..<count).map { index in
            let kind = historyKind(for: index)
            let createdAt = now.addingTimeInterval(TimeInterval(-index * 60))
            let appName = index.isMultiple(of: 4) ? "Seed Safari" : "Seed Notes"
            let dictionaryTerm = "Seed Term \(index % max(1, min(count, 20_000)))"
            return TranscriptionHistoryEntry(
                id: deterministicUUID(prefix: 4, index: index),
                text: "Seed history row \(index). This row is generated for large-list scrolling and search validation. Keyword seedneedle\(index % 97).",
                createdAt: createdAt,
                transcriptionEngine: "Seed Engine",
                transcriptionModel: "Seed Model",
                enhancementMode: "Seed Mode",
                enhancementModel: "Seed Enhancement",
                kind: kind,
                isTranslation: kind == .translation,
                audioDurationSeconds: Double(index % 240),
                transcriptionProcessingDurationSeconds: Double(index % 10) / 10.0,
                llmDurationSeconds: Double(index % 7) / 10.0,
                focusedAppName: appName,
                focusedAppBundleID: "com.voxt.seed.\(appName.lowercased().replacingOccurrences(of: " ", with: "-"))",
                matchedGroupID: index.isMultiple(of: 6) ? deterministicUUID(prefix: 9, index: index % 20) : nil,
                matchedGroupName: index.isMultiple(of: 6) ? "Seed Group \(index % 20)" : nil,
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
                displayTitle: "Seed History \(index)",
                dictionaryHitTerms: [dictionaryTerm],
                dictionaryCorrectedTerms: index.isMultiple(of: 8) ? ["Seed Alias \(index % 20_000)"] : [],
                dictionarySuggestedTerms: [
                    DictionarySuggestionSnapshot(
                        term: "Seed Suggestion \(index % 500)",
                        normalizedTerm: "seed suggestion \(index % 500)",
                        groupID: nil,
                        groupNameSnapshot: nil
                    )
                ]
            )
        }
    }

    private static func historyKind(for index: Int) -> TranscriptionHistoryKind {
        if index.isMultiple(of: 9) {
            return .rewrite
        }
        if index.isMultiple(of: 5) {
            return .translation
        }
        return .normal
    }

    private static func deterministicUUID(prefix: Int, index: Int) -> UUID {
        let suffix = String(format: "%012llX", UInt64(index))
        return UUID(uuidString: String(format: "00000000-%04X-4000-8000-%@", prefix, suffix)) ?? UUID()
    }
}
#endif
