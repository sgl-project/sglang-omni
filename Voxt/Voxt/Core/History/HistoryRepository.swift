// HistoryRepository.swift
// Provides History Repository for core app behavior.

import Foundation
import GRDB

protocol HistoryRepositoryProtocol: AnyObject, Sendable {
    func entries(
        kind: TranscriptionHistoryKind?,
        query: String,
        limit: Int?,
        offset: Int
    ) throws -> [TranscriptionHistoryEntry]
    func listEntries(
        kind: TranscriptionHistoryKind?,
        query: String,
        limit: Int?,
        offset: Int
    ) throws -> [TranscriptionHistoryListEntry]
    func entry(id: UUID) throws -> TranscriptionHistoryEntry?
    func latestEntryText() throws -> String?
    func audioRelativePaths() throws -> [String]
    func entryCount(kind: TranscriptionHistoryKind?, query: String) throws -> Int
    func pendingNormalEntryCount(after checkpoint: DictionaryHistoryScanCheckpoint?) throws -> Int
    func pendingNormalEntries(after checkpoint: DictionaryHistoryScanCheckpoint?) throws -> [TranscriptionHistoryEntry]
    func reportMetrics(dayStarts: [Date], branchStartDate: Date?) throws -> HistoryReportMetrics
    func upsert(_ entry: TranscriptionHistoryEntry) throws
    func delete(id: UUID) throws -> TranscriptionHistoryEntry?
    func clearAll() throws
    func deleteEntries(kind: TranscriptionHistoryKind) throws -> [TranscriptionHistoryEntry]
    func deleteEntries(
        olderThan cutoff: Date,
        kinds: Set<TranscriptionHistoryKind>
    ) throws -> [TranscriptionHistoryEntry]
    func deleteEntries(
        keepingNewest count: Int,
        kind: TranscriptionHistoryKind
    ) throws -> [TranscriptionHistoryEntry]
}

final class HistoryRepository: HistoryRepositoryProtocol, @unchecked Sendable {
    private let database: VoxtDatabase
    private let fileManager: FileManager
    private let legacyJSONURL: URL?

    init(
        database: VoxtDatabase = .shared,
        fileManager: FileManager = .default,
        legacyJSONURL: URL? = nil,
        migrateLegacyJSON: Bool = true
    ) {
        self.database = database
        self.fileManager = fileManager
        self.legacyJSONURL = legacyJSONURL ?? Self.defaultLegacyJSONURL(fileManager: fileManager)

        if migrateLegacyJSON {
            migrateLegacyJSONIfNeeded()
        }
    }

    func allEntries() throws -> [TranscriptionHistoryEntry] {
        try entries(kind: nil, query: "", limit: nil, offset: 0)
    }

    func entries(
        kind: TranscriptionHistoryKind?,
        query: String = "",
        limit: Int?,
        offset: Int
    ) throws -> [TranscriptionHistoryEntry] {
        var whereClauses: [String] = []
        var arguments: StatementArguments = []
        var joinSQL = ""

        if let kind {
            whereClauses.append("h.kind = ?")
            arguments += [kind.rawValue]
        }

        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if let ftsQuery = VoxtFTSQueryBuilder.query(from: trimmedQuery) {
            joinSQL = "JOIN history_search ON history_search.historyID = h.id"
            whereClauses.append("history_search MATCH ?")
            arguments += [ftsQuery]
        }

        let whereSQL = whereClauses.isEmpty ? "" : "WHERE \(whereClauses.joined(separator: " AND "))"
        let limitSQL: String
        if let limit {
            limitSQL = "LIMIT ? OFFSET ?"
            arguments += [limit, offset]
        } else {
            limitSQL = ""
        }

        return try database.dbQueue.read { db in
            let jsonEntries = try String.fetchAll(
                db,
                sql: """
                    SELECT h.entryJSON FROM history_entries h
                    \(joinSQL)
                    \(whereSQL)
                    ORDER BY h.createdAt DESC
                    \(limitSQL)
                    """,
                arguments: arguments
            )
            return try jsonEntries.map {
                try VoxtPersistenceCoding.decodeJSON(TranscriptionHistoryEntry.self, from: $0)
            }
        }
    }

    func listEntries(
        kind: TranscriptionHistoryKind?,
        query: String = "",
        limit: Int?,
        offset: Int
    ) throws -> [TranscriptionHistoryListEntry] {
        var whereClauses: [String] = []
        var arguments: StatementArguments = []
        var joinSQL = ""

        if let kind {
            whereClauses.append("h.kind = ?")
            arguments += [kind.rawValue]
        }

        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if let ftsQuery = VoxtFTSQueryBuilder.query(from: trimmedQuery) {
            joinSQL = "JOIN history_search ON history_search.historyID = h.id"
            whereClauses.append("history_search MATCH ?")
            arguments += [ftsQuery]
        }

        let whereSQL = whereClauses.isEmpty ? "" : "WHERE \(whereClauses.joined(separator: " AND "))"
        let limitSQL: String
        if let limit {
            limitSQL = "LIMIT ? OFFSET ?"
            arguments += [limit, offset]
        } else {
            limitSQL = ""
        }

        return try database.dbQueue.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: """
                    SELECT
                        h.id,
                        substr(h.text, 1, ?) AS previewText,
                        length(h.text) AS textLength,
                        h.createdAt,
                        h.kind,
                        h.displayTitle
                    FROM history_entries h
                    \(joinSQL)
                    \(whereSQL)
                    ORDER BY h.createdAt DESC
                    \(limitSQL)
                    """,
                arguments: StatementArguments([TranscriptionHistoryListEntry.previewCharacterLimit]) + arguments
            )

            return rows.compactMap { row -> TranscriptionHistoryListEntry? in
                guard let idString: String = row["id"],
                      let id = UUID(uuidString: idString),
                      let kindRawValue: String = row["kind"],
                      let kind = TranscriptionHistoryKind(rawValue: kindRawValue),
                      let createdAt: Double = row["createdAt"]
                else {
                    return nil
                }

                return TranscriptionHistoryListEntry(
                    id: id,
                    previewText: row["previewText"] ?? "",
                    textLength: row["textLength"] ?? 0,
                    createdAt: Date(timeIntervalSince1970: createdAt),
                    kind: kind,
                    displayTitle: row["displayTitle"]
                )
            }
        }
    }

    func entry(id: UUID) throws -> TranscriptionHistoryEntry? {
        try database.dbQueue.read { db in
            guard let json = try String.fetchOne(
                db,
                sql: "SELECT entryJSON FROM history_entries WHERE id = ?",
                arguments: [id.uuidString]
            ) else {
                return nil
            }
            return try VoxtPersistenceCoding.decodeJSON(TranscriptionHistoryEntry.self, from: json)
        }
    }

    func latestEntryText() throws -> String? {
        try database.dbQueue.read { db in
            try String.fetchOne(
                db,
                sql: """
                    SELECT text
                    FROM history_entries
                    WHERE kind != ?
                    ORDER BY createdAt DESC
                    LIMIT 1
                    """,
                arguments: [TranscriptionHistoryKind.transcript.rawValue]
            )
        }
    }

    func audioRelativePaths() throws -> [String] {
        try database.dbQueue.read { db in
            try String.fetchAll(
                db,
                sql: """
                    SELECT path FROM (
                        SELECT COALESCE(NULLIF(audioRelativePath, ''), NULLIF(transcriptAudioRelativePath, '')) AS path
                        FROM history_entries
                    )
                    WHERE path IS NOT NULL AND path != ''
                    """
            )
        }
    }

    func entryCount(kind: TranscriptionHistoryKind? = nil, query: String = "") throws -> Int {
        var whereClauses: [String] = []
        var arguments: StatementArguments = []
        var joinSQL = ""

        if let kind {
            whereClauses.append("h.kind = ?")
            arguments += [kind.rawValue]
        }

        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if let ftsQuery = VoxtFTSQueryBuilder.query(from: trimmedQuery) {
            joinSQL = "JOIN history_search ON history_search.historyID = h.id"
            whereClauses.append("history_search MATCH ?")
            arguments += [ftsQuery]
        }

        let whereSQL = whereClauses.isEmpty ? "" : "WHERE \(whereClauses.joined(separator: " AND "))"
        return try database.dbQueue.read { db in
            try Int.fetchOne(
                db,
                sql: """
                    SELECT COUNT(*) FROM history_entries h
                    \(joinSQL)
                    \(whereSQL)
                    """,
                arguments: arguments
            ) ?? 0
        }
    }

    func pendingNormalEntryCount(after checkpoint: DictionaryHistoryScanCheckpoint?) throws -> Int {
        try database.dbQueue.read { db in
            try Int.fetchOne(
                db,
                sql: """
                    SELECT COUNT(*)
                    FROM history_entries
                    WHERE kind = ?
                    \(pendingCheckpointWhereSQL(checkpoint))
                    """,
                arguments: pendingCheckpointArguments(checkpoint, prefix: StatementArguments([TranscriptionHistoryKind.normal.rawValue]))
            ) ?? 0
        }
    }

    func pendingNormalEntries(after checkpoint: DictionaryHistoryScanCheckpoint?) throws -> [TranscriptionHistoryEntry] {
        try database.dbQueue.read { db in
            let jsonEntries = try String.fetchAll(
                db,
                sql: """
                    SELECT entryJSON
                    FROM history_entries
                    WHERE kind = ?
                    \(pendingCheckpointWhereSQL(checkpoint))
                    ORDER BY createdAt ASC, id ASC
                    """,
                arguments: pendingCheckpointArguments(checkpoint, prefix: StatementArguments([TranscriptionHistoryKind.normal.rawValue]))
            )
            return try jsonEntries.map {
                try VoxtPersistenceCoding.decodeJSON(TranscriptionHistoryEntry.self, from: $0)
            }
        }
    }

    func reportMetrics(dayStarts: [Date], branchStartDate: Date? = nil) throws -> HistoryReportMetrics {
        try database.dbQueue.read { db in
            let totals = try Row.fetchOne(
                db,
                sql: """
                    SELECT
                        COALESCE(SUM(COALESCE(audioDurationSeconds, 0)), 0) AS totalDuration,
                        COALESCE(SUM(LENGTH(text)), 0) AS totalCharacters,
                        COALESCE(SUM(CASE WHEN kind = ? THEN LENGTH(text) ELSE 0 END), 0) AS totalTranslationCharacters
                    FROM history_entries
                    """,
                arguments: [TranscriptionHistoryKind.translation.rawValue]
            )

            var dailyCharacters: [Date: Int] = [:]
            for dayStart in dayStarts {
                let nextDay = Calendar.current.date(byAdding: .day, value: 1, to: dayStart) ?? dayStart
                let count = try Int.fetchOne(
                    db,
                    sql: """
                        SELECT COALESCE(SUM(LENGTH(text)), 0)
                        FROM history_entries
                        WHERE createdAt >= ? AND createdAt < ?
                        """,
                    arguments: [
                        dayStart.timeIntervalSince1970,
                        nextDay.timeIntervalSince1970
                    ]
                ) ?? 0
                dailyCharacters[dayStart] = count
            }

            let branchItems = try fetchBranchItems(db: db, startDate: branchStartDate)

            return HistoryReportMetrics(
                totalDictationSeconds: totals?["totalDuration"] ?? 0,
                totalCharacters: totals?["totalCharacters"] ?? 0,
                totalTranslationCharacters: totals?["totalTranslationCharacters"] ?? 0,
                dailyCharacters: dailyCharacters,
                branchItems: branchItems
            )
        }
    }

    private func fetchBranchItems(db: Database, startDate: Date?) throws -> [HistoryBranchMetricItem] {
        var arguments: StatementArguments = [TranscriptionHistoryKind.transcript.rawValue]
        var conditions = ["kind != ?"]
        if let startDate {
            conditions.append("createdAt >= ?")
            arguments += [startDate.timeIntervalSince1970]
        }
        let branchFilter = "WHERE " + conditions.joined(separator: " AND ")

        let appRows = try Row.fetchAll(
            db,
            sql: """
                SELECT
                    COALESCE(NULLIF(focusedAppName, ''), NULLIF(focusedAppBundleID, ''), 'Unknown App') AS title,
                    NULLIF(focusedAppBundleID, '') AS bundleID,
                    COALESCE(SUM(LENGTH(text)), 0) AS characterCount
                FROM history_entries
                \(branchFilter)
                    AND (NULLIF(focusedAppName, '') IS NOT NULL OR NULLIF(focusedAppBundleID, '') IS NOT NULL)
                GROUP BY COALESCE(NULLIF(focusedAppBundleID, ''), NULLIF(focusedAppName, ''), 'Unknown App')
                HAVING characterCount > 0
                """,
            arguments: arguments
        )
        var items = appRows.map { row in
            let title: String = row["title"] ?? "Unknown App"
            let bundleID: String? = row["bundleID"]
            return HistoryBranchMetricItem(
                kind: .app,
                title: title,
                subtitle: bundleID,
                bundleID: bundleID,
                urlHost: nil,
                urlOrigin: nil,
                characterCount: row["characterCount"] ?? 0
            )
        }

        let urlRows = try Row.fetchAll(
            db,
            sql: """
                SELECT
                    browserURLHost AS host,
                    MIN(browserURLOrigin) AS origin,
                    COALESCE(SUM(LENGTH(text)), 0) AS characterCount
                FROM history_entries
                \(branchFilter)
                    AND browserURLHost IS NOT NULL AND browserURLHost != ''
                GROUP BY browserURLHost
                HAVING characterCount > 0
                """,
            arguments: arguments
        )
        items += urlRows.map { row in
            let host: String = row["host"] ?? ""
            let origin: String? = row["origin"]
            return HistoryBranchMetricItem(
                kind: .url,
                title: host,
                subtitle: origin,
                bundleID: nil,
                urlHost: host,
                urlOrigin: origin,
                characterCount: row["characterCount"] ?? 0
            )
        }

        return Array(
            items
                .sorted {
                    if $0.characterCount == $1.characterCount {
                        return $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending
                    }
                    return $0.characterCount > $1.characterCount
                }
                .prefix(10)
        )
    }

    func upsert(_ entry: TranscriptionHistoryEntry) throws {
        try database.dbQueue.write { db in
            try upsert(entry, db: db)
        }
    }

    func replaceAll(_ entries: [TranscriptionHistoryEntry]) throws {
        try database.dbQueue.write { db in
            try db.execute(sql: "DELETE FROM history_search")
            try db.execute(sql: "DELETE FROM history_entries")
            for entry in entries {
                try upsert(entry, db: db)
            }
        }
    }

    func upsertAll(_ entries: [TranscriptionHistoryEntry]) throws {
        try database.dbQueue.write { db in
            for entry in entries {
                try upsert(entry, db: db)
            }
        }
    }

    @discardableResult
    func delete(id: UUID) throws -> TranscriptionHistoryEntry? {
        let removed = try entry(id: id)
        try database.dbQueue.write { db in
            try db.execute(sql: "DELETE FROM history_search WHERE historyID = ?", arguments: [id.uuidString])
            try db.execute(sql: "DELETE FROM history_entries WHERE id = ?", arguments: [id.uuidString])
        }
        return removed
    }

    func clearAll() throws {
        try replaceAll([])
    }

    @discardableResult
    func deleteEntries(kind: TranscriptionHistoryKind) throws -> [TranscriptionHistoryEntry] {
        try deleteEntries(
            whereSQL: "kind = ?",
            arguments: [kind.rawValue]
        )
    }

    @discardableResult
    func deleteEntries(
        olderThan cutoff: Date,
        kinds: Set<TranscriptionHistoryKind>
    ) throws -> [TranscriptionHistoryEntry] {
        let rawKinds = kinds.map(\.rawValue).sorted()
        guard !rawKinds.isEmpty else { return [] }

        let placeholders = Array(repeating: "?", count: rawKinds.count).joined(separator: ", ")
        var arguments: StatementArguments = [cutoff.timeIntervalSince1970]
        arguments += StatementArguments(rawKinds)
        return try deleteEntries(
            whereSQL: "createdAt < ? AND kind IN (\(placeholders))",
            arguments: arguments
        )
    }

    @discardableResult
    func deleteEntries(
        keepingNewest count: Int,
        kind: TranscriptionHistoryKind
    ) throws -> [TranscriptionHistoryEntry] {
        guard count >= 0 else { return [] }
        return try deleteEntries(
            whereSQL: """
                id IN (
                    SELECT id FROM history_entries
                    WHERE kind = ?
                    ORDER BY createdAt DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
            arguments: [kind.rawValue, count]
        )
    }

    private func deleteEntries(
        whereSQL: String,
        arguments: StatementArguments
    ) throws -> [TranscriptionHistoryEntry] {
        let removed = try database.dbQueue.read { db in
            let jsonEntries = try String.fetchAll(
                db,
                sql: "SELECT entryJSON FROM history_entries WHERE \(whereSQL) ORDER BY createdAt DESC",
                arguments: arguments
            )
            return try jsonEntries.map {
                try VoxtPersistenceCoding.decodeJSON(TranscriptionHistoryEntry.self, from: $0)
            }
        }

        guard !removed.isEmpty else { return [] }
        try database.dbQueue.write { db in
            for entry in removed {
                try db.execute(sql: "DELETE FROM history_search WHERE historyID = ?", arguments: [entry.id.uuidString])
                try db.execute(sql: "DELETE FROM history_entries WHERE id = ?", arguments: [entry.id.uuidString])
            }
        }
        return removed
    }

    private func upsert(_ entry: TranscriptionHistoryEntry, db: Database) throws {
        let entryJSON = try VoxtPersistenceCoding.encodeJSONString(entry)
        let dictionaryTermsText = dictionaryTerms(for: entry).joined(separator: " ")

        try db.execute(
            sql: """
                INSERT INTO history_entries (
                    id, text, createdAt, kind, transcriptionEngine, transcriptionModel,
                    enhancementMode, enhancementModel, isTranslation, audioDurationSeconds,
                    transcriptionProcessingDurationSeconds, llmDurationSeconds, focusedAppName,
                    focusedAppBundleID, browserURLHost, browserURLOrigin, matchedGroupID,
                    matchedGroupName, matchedAppGroupName, matchedURLGroupName,
                    remoteASRProvider, remoteASRModel, remoteASREndpoint,
                    remoteLLMProvider, remoteLLMModel, remoteLLMEndpoint, audioRelativePath,
                    transcriptAudioRelativePath, displayTitle, dictionaryTermsText, entryJSON
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text = excluded.text,
                    createdAt = excluded.createdAt,
                    kind = excluded.kind,
                    transcriptionEngine = excluded.transcriptionEngine,
                    transcriptionModel = excluded.transcriptionModel,
                    enhancementMode = excluded.enhancementMode,
                    enhancementModel = excluded.enhancementModel,
                    isTranslation = excluded.isTranslation,
                    audioDurationSeconds = excluded.audioDurationSeconds,
                    transcriptionProcessingDurationSeconds = excluded.transcriptionProcessingDurationSeconds,
                    llmDurationSeconds = excluded.llmDurationSeconds,
                    focusedAppName = excluded.focusedAppName,
                    focusedAppBundleID = excluded.focusedAppBundleID,
                    browserURLHost = excluded.browserURLHost,
                    browserURLOrigin = excluded.browserURLOrigin,
                    matchedGroupID = excluded.matchedGroupID,
                    matchedGroupName = excluded.matchedGroupName,
                    matchedAppGroupName = excluded.matchedAppGroupName,
                    matchedURLGroupName = excluded.matchedURLGroupName,
                    remoteASRProvider = excluded.remoteASRProvider,
                    remoteASRModel = excluded.remoteASRModel,
                    remoteASREndpoint = excluded.remoteASREndpoint,
                    remoteLLMProvider = excluded.remoteLLMProvider,
                    remoteLLMModel = excluded.remoteLLMModel,
                    remoteLLMEndpoint = excluded.remoteLLMEndpoint,
                    audioRelativePath = excluded.audioRelativePath,
                    transcriptAudioRelativePath = excluded.transcriptAudioRelativePath,
                    displayTitle = excluded.displayTitle,
                    dictionaryTermsText = excluded.dictionaryTermsText,
                    entryJSON = excluded.entryJSON
                """,
            arguments: VoxtDatabaseArguments.make([
                entry.id.uuidString,
                entry.text,
                entry.createdAt.timeIntervalSince1970,
                entry.kind.rawValue,
                entry.transcriptionEngine,
                entry.transcriptionModel,
                entry.enhancementMode,
                entry.enhancementModel,
                entry.isTranslation ? 1 : 0,
                entry.audioDurationSeconds,
                entry.transcriptionProcessingDurationSeconds,
                entry.llmDurationSeconds,
                entry.focusedAppName,
                entry.focusedAppBundleID,
                entry.browserURLHost,
                entry.browserURLOrigin,
                entry.matchedGroupID?.uuidString,
                entry.matchedGroupName,
                entry.matchedAppGroupName,
                entry.matchedURLGroupName,
                entry.remoteASRProvider,
                entry.remoteASRModel,
                entry.remoteASREndpoint,
                entry.remoteLLMProvider,
                entry.remoteLLMModel,
                entry.remoteLLMEndpoint,
                entry.audioRelativePath,
                entry.transcriptAudioRelativePath,
                entry.displayTitle,
                dictionaryTermsText,
                entryJSON
            ])
        )

        try db.execute(sql: "DELETE FROM history_search WHERE historyID = ?", arguments: [entry.id.uuidString])
        try db.execute(
            sql: """
                INSERT INTO history_search (
                    historyID, text, displayTitle, focusedAppName, matchedGroupName, dictionaryTerms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
            arguments: VoxtDatabaseArguments.make([
                entry.id.uuidString,
                entry.text,
                entry.displayTitle ?? "",
                entry.focusedAppName ?? "",
                [
                    entry.matchedGroupName,
                    entry.matchedAppGroupName,
                    entry.matchedURLGroupName
                ].compactMap { $0 }.joined(separator: " "),
                dictionaryTermsText
            ])
        )
    }

    private func dictionaryTerms(for entry: TranscriptionHistoryEntry) -> [String] {
        var terms = entry.dictionaryHitTerms + entry.dictionaryCorrectedTerms
        terms.append(contentsOf: entry.dictionarySuggestedTerms.map(\.term))
        terms.append(contentsOf: entry.dictionaryCorrectionSnapshots.flatMap { [$0.originalText, $0.correctedText] })
        return terms
    }

    private func pendingCheckpointWhereSQL(_ checkpoint: DictionaryHistoryScanCheckpoint?) -> String {
        guard checkpoint != nil else { return "" }
        return "AND (createdAt > ? OR (createdAt = ? AND id > ?))"
    }

    private func pendingCheckpointArguments(
        _ checkpoint: DictionaryHistoryScanCheckpoint?,
        prefix: StatementArguments
    ) -> StatementArguments {
        var arguments = prefix
        if let checkpoint {
            arguments += [
                checkpoint.lastProcessedAt.timeIntervalSince1970,
                checkpoint.lastProcessedAt.timeIntervalSince1970,
                checkpoint.lastHistoryEntryID.uuidString
            ]
        }
        return arguments
    }

    private func migrateLegacyJSONIfNeeded() {
        do {
            let existingCount = try database.dbQueue.read { db in
                try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM history_entries") ?? 0
            }
            guard existingCount == 0,
                  let legacyJSONURL,
                  fileManager.fileExists(atPath: legacyJSONURL.path)
            else {
                return
            }

            let data = try Data(contentsOf: legacyJSONURL)
            let entries = try VoxtPersistenceCoding.decoder.decode([TranscriptionHistoryEntry].self, from: data)
            try replaceAll(entries)
            try backupLegacyJSON(at: legacyJSONURL)
        } catch {
            VoxtLog.warning("History SQLite migration skipped or failed: \(error.localizedDescription)")
        }
    }

    private func backupLegacyJSON(at url: URL) throws {
        let backupURL = url.deletingLastPathComponent()
            .appendingPathComponent("\(url.lastPathComponent).migrated-backup")
        if fileManager.fileExists(atPath: backupURL.path) {
            try fileManager.removeItem(at: backupURL)
        }
        try fileManager.moveItem(at: url, to: backupURL)
    }

    private static func defaultLegacyJSONURL(fileManager: FileManager) -> URL? {
        guard let appSupport = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) else {
            return nil
        }
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("transcription-history.json")
    }
}
