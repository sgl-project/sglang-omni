// DictionaryRepository.swift
// Provides Dictionary Repository for core app behavior.

import Foundation
import GRDB

protocol DictionaryRepositoryProtocol: AnyObject, Sendable {
    func allCategories() throws -> [DictionaryCategory]
    func allEntries() throws -> [DictionaryEntry]
    func allTerms(limit: Int?) throws -> [String]
    func entries(filter: DictionaryFilter, query: String, limit: Int, offset: Int) throws -> [DictionaryEntry]
    func entryCount(filter: DictionaryFilter, query: String) throws -> Int
    func entries(requiringReplacementTerms: Bool, query: String, limit: Int, offset: Int) throws -> [DictionaryEntry]
    func entryCount(requiringReplacementTerms: Bool, query: String) throws -> Int
    func matchingEntries(sourceText: String, activeGroupID: UUID?, limit: Int) throws -> [DictionaryEntry]
    func activeEntriesForRemoteRequest(activeGroupID: UUID?, limit: Int) throws -> [DictionaryEntry]
    func upsert(_ entry: DictionaryEntry) throws
    func upsertCategory(_ category: DictionaryCategory) throws
    func replaceAll(_ entries: [DictionaryEntry]) throws
    func replaceAll(entries: [DictionaryEntry], categories: [DictionaryCategory]) throws
    func delete(id: UUID) throws
    func deleteCategory(id: UUID, moveEntriesTo category: DictionaryCategory?) throws
    func clearAll() throws
    func hasEntry(normalizedTerm: String, activeGroupID: UUID?) throws -> Bool
}

final class DictionaryRepository: DictionaryRepositoryProtocol, @unchecked Sendable {
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

    func allCategories() throws -> [DictionaryCategory] {
        try database.dbQueue.read { db in
            let rows = try Row.fetchAll(
                db,
                sql: """
                    SELECT * FROM dictionary_categories
                    ORDER BY isDefault DESC, sortOrder ASC, name COLLATE NOCASE ASC
                    """
            )
            var categories = rows.compactMap(category(from:))
            if !categories.contains(where: \.isDefault) {
                categories.insert(DictionaryCategory.defaultCategory, at: 0)
            }
            return categories
        }
    }

    func allEntries() throws -> [DictionaryEntry] {
        try fetchEntries(
            sql: "SELECT id FROM dictionary_entries ORDER BY updatedAt DESC, term COLLATE NOCASE ASC",
            arguments: []
        )
    }

    func allTerms(limit: Int? = nil) throws -> [String] {
        try database.dbQueue.read { db in
            var sql = "SELECT term FROM dictionary_entries ORDER BY term COLLATE NOCASE ASC"
            var arguments: StatementArguments = []
            if let limit {
                sql += " LIMIT ?"
                arguments += [limit]
            }
            return try String.fetchAll(
                db,
                sql: sql,
                arguments: arguments
            )
        }
    }

    func entries(filter: DictionaryFilter, query: String = "", limit: Int, offset: Int) throws -> [DictionaryEntry] {
        var whereClauses: [String] = []
        var arguments: StatementArguments = []

        switch filter {
        case .all:
            break
        case .autoAdded:
            whereClauses.append("source = ?")
            arguments += [DictionaryEntrySource.auto.rawValue]
        case .manualAdded:
            whereClauses.append("source = ?")
            arguments += [DictionaryEntrySource.manual.rawValue]
        }

        if let searchIDs = try searchEntryIDs(query: query) {
            if searchIDs.isEmpty {
                return []
            }
            whereClauses.append("id IN \(sqlPlaceholders(count: searchIDs.count))")
            arguments += StatementArguments(searchIDs)
        }

        let whereSQL = whereClauses.isEmpty ? "" : "WHERE \(whereClauses.joined(separator: " AND "))"
        arguments += [limit, offset]

        return try fetchEntries(
            sql: """
                SELECT id FROM dictionary_entries
                \(whereSQL)
                ORDER BY updatedAt DESC, term COLLATE NOCASE ASC
                LIMIT ? OFFSET ?
                """,
            arguments: arguments
        )
    }

    func entryCount(filter: DictionaryFilter = .all, query: String = "") throws -> Int {
        var whereClauses: [String] = []
        var arguments: StatementArguments = []

        switch filter {
        case .all:
            break
        case .autoAdded:
            whereClauses.append("source = ?")
            arguments += [DictionaryEntrySource.auto.rawValue]
        case .manualAdded:
            whereClauses.append("source = ?")
            arguments += [DictionaryEntrySource.manual.rawValue]
        }

        if let searchIDs = try searchEntryIDs(query: query) {
            if searchIDs.isEmpty {
                return 0
            }
            whereClauses.append("id IN \(sqlPlaceholders(count: searchIDs.count))")
            arguments += StatementArguments(searchIDs)
        }

        let whereSQL = whereClauses.isEmpty ? "" : "WHERE \(whereClauses.joined(separator: " AND "))"
        return try database.dbQueue.read { db in
            try Int.fetchOne(
                db,
                sql: "SELECT COUNT(*) FROM dictionary_entries \(whereSQL)",
                arguments: arguments
            ) ?? 0
        }
    }

    func entries(requiringReplacementTerms: Bool, query: String = "", limit: Int, offset: Int) throws -> [DictionaryEntry] {
        var whereClauses: [String] = [
            requiringReplacementTerms
                ? "EXISTS (SELECT 1 FROM dictionary_replacement_terms r WHERE r.entryID = dictionary_entries.id)"
                : "NOT EXISTS (SELECT 1 FROM dictionary_replacement_terms r WHERE r.entryID = dictionary_entries.id)"
        ]
        var arguments: StatementArguments = []

        if let searchIDs = try searchEntryIDs(query: query) {
            if searchIDs.isEmpty {
                return []
            }
            whereClauses.append("id IN \(sqlPlaceholders(count: searchIDs.count))")
            arguments += StatementArguments(searchIDs)
        }

        let whereSQL = "WHERE \(whereClauses.joined(separator: " AND "))"
        arguments += [limit, offset]

        return try fetchEntries(
            sql: """
                SELECT id FROM dictionary_entries
                \(whereSQL)
                ORDER BY updatedAt DESC, term COLLATE NOCASE ASC
                LIMIT ? OFFSET ?
                """,
            arguments: arguments
        )
    }

    func entryCount(requiringReplacementTerms: Bool, query: String = "") throws -> Int {
        var whereClauses: [String] = [
            requiringReplacementTerms
                ? "EXISTS (SELECT 1 FROM dictionary_replacement_terms r WHERE r.entryID = dictionary_entries.id)"
                : "NOT EXISTS (SELECT 1 FROM dictionary_replacement_terms r WHERE r.entryID = dictionary_entries.id)"
        ]
        var arguments: StatementArguments = []

        if let searchIDs = try searchEntryIDs(query: query) {
            if searchIDs.isEmpty {
                return 0
            }
            whereClauses.append("id IN \(sqlPlaceholders(count: searchIDs.count))")
            arguments += StatementArguments(searchIDs)
        }

        let whereSQL = "WHERE \(whereClauses.joined(separator: " AND "))"
        return try database.dbQueue.read { db in
            try Int.fetchOne(
                db,
                sql: "SELECT COUNT(*) FROM dictionary_entries \(whereSQL)",
                arguments: arguments
            ) ?? 0
        }
    }

    func matchingEntries(sourceText: String, activeGroupID: UUID?, limit: Int = 200) throws -> [DictionaryEntry] {
        let normalizedSource = DictionaryStore.normalizeTerm(sourceText)
        guard !normalizedSource.isEmpty, limit > 0 else { return [] }

        let queryLimit = min(max(limit * 8, 200), 1_000)
        let candidates = try fetchEntries(
            sql: """
                WITH matched_rows AS (
                    SELECT e.id, length(e.normalizedTerm) AS matchLength, e.matchCount, e.updatedAt, e.term
                    FROM dictionary_entries e
                    WHERE e.status = ?
                      AND \(scopeSQL(activeGroupID: activeGroupID))
                      AND length(e.normalizedTerm) > 0
                      AND instr(?, e.normalizedTerm) > 0

                    UNION

                    SELECT e.id, length(r.normalizedText) AS matchLength, e.matchCount, e.updatedAt, e.term
                    FROM dictionary_replacement_terms r
                    JOIN dictionary_entries e ON e.id = r.entryID
                    WHERE e.status = ?
                      AND \(scopeSQL(activeGroupID: activeGroupID))
                      AND length(r.normalizedText) > 0
                      AND instr(?, r.normalizedText) > 0

                    UNION

                    SELECT e.id, length(v.normalizedText) AS matchLength, e.matchCount, e.updatedAt, e.term
                    FROM dictionary_observed_variants v
                    JOIN dictionary_entries e ON e.id = v.entryID
                    WHERE e.status = ?
                      AND \(scopeSQL(activeGroupID: activeGroupID))
                      AND length(v.normalizedText) > 0
                      AND instr(?, v.normalizedText) > 0
                ),
                matched_ids AS (
                    SELECT
                        id,
                        MAX(matchLength) AS matchLength,
                        MAX(matchCount) AS matchCount,
                        MAX(updatedAt) AS updatedAt,
                        MIN(term) AS term
                    FROM matched_rows
                    GROUP BY id
                )
                SELECT id FROM matched_ids
                ORDER BY matchLength DESC, matchCount DESC, updatedAt DESC, term COLLATE NOCASE ASC
                LIMIT ?
                """,
            arguments: matchingEntriesArguments(
                normalizedSource: normalizedSource,
                activeGroupID: activeGroupID,
                limit: queryLimit
            )
        )
        return Array(candidates
            .filter { entryMatchesSource($0, normalizedSource: normalizedSource) }
            .prefix(limit))
    }

    func activeEntriesForRemoteRequest(activeGroupID: UUID?, limit: Int) throws -> [DictionaryEntry] {
        guard limit > 0 else { return [] }
        var arguments: StatementArguments = [DictionaryEntryStatus.active.rawValue]
        let scopeSQL: String
        let scopeOrderSQL: String
        if let activeGroupID {
            scopeSQL = "(groupID IS NULL OR groupID = ?)"
            scopeOrderSQL = "CASE WHEN groupID = ? THEN 0 ELSE 1 END,"
            arguments += [activeGroupID.uuidString, activeGroupID.uuidString]
        } else {
            scopeSQL = "groupID IS NULL"
            scopeOrderSQL = ""
        }
        arguments += [limit]

        return try fetchEntries(
            sql: """
                SELECT id FROM dictionary_entries
                WHERE status = ? AND \(scopeSQL)
                ORDER BY
                    \(scopeOrderSQL)
                    matchCount DESC,
                    COALESCE(lastMatchedAt, 0) DESC,
                    updatedAt DESC,
                    term COLLATE NOCASE ASC
                LIMIT ?
                """,
            arguments: arguments
        )
    }

    func upsert(_ entry: DictionaryEntry) throws {
        try database.dbQueue.write { db in
            try upsert(entry, db: db)
        }
    }

    func upsertCategory(_ category: DictionaryCategory) throws {
        try database.dbQueue.write { db in
            try upsert(category, db: db)
        }
    }

    func replaceAll(_ entries: [DictionaryEntry]) throws {
        try database.dbQueue.write { db in
            try db.execute(sql: "DELETE FROM dictionary_search")
            try db.execute(sql: "DELETE FROM dictionary_observed_variants")
            try db.execute(sql: "DELETE FROM dictionary_replacement_terms")
            try db.execute(sql: "DELETE FROM dictionary_entries")
            for entry in entries {
                try upsert(entry, db: db)
            }
        }
    }

    func replaceAll(entries: [DictionaryEntry], categories: [DictionaryCategory]) throws {
        try database.dbQueue.write { db in
            try db.execute(sql: "DELETE FROM dictionary_search")
            try db.execute(sql: "DELETE FROM dictionary_observed_variants")
            try db.execute(sql: "DELETE FROM dictionary_replacement_terms")
            try db.execute(sql: "DELETE FROM dictionary_entries")
            try db.execute(sql: "DELETE FROM dictionary_categories WHERE isDefault = 0")

            var categoriesByID = Dictionary(uniqueKeysWithValues: categories.map { ($0.id, $0) })
            categoriesByID[DictionaryCategory.defaultID] = categoriesByID[DictionaryCategory.defaultID]
                ?? DictionaryCategory.defaultCategory
            for entry in entries where categoriesByID[entry.categoryID] == nil {
                categoriesByID[entry.categoryID] = DictionaryCategory(
                    id: entry.categoryID,
                    name: entry.categoryNameSnapshot ?? DictionaryCategory.defaultName,
                    isDefault: entry.categoryID == DictionaryCategory.defaultID,
                    isExpanded: true,
                    sortOrder: categoriesByID.count
                )
            }

            for category in categoriesByID.values {
                try upsert(category, db: db)
            }
            for entry in entries {
                try upsert(entry, db: db)
            }
        }
    }

    func upsertAll(_ entries: [DictionaryEntry]) throws {
        try database.dbQueue.write { db in
            for entry in entries {
                try upsert(entry, db: db)
            }
        }
    }

    func delete(id: UUID) throws {
        try database.dbQueue.write { db in
            try db.execute(sql: "DELETE FROM dictionary_search WHERE entryID = ?", arguments: [id.uuidString])
            try db.execute(sql: "DELETE FROM dictionary_entries WHERE id = ?", arguments: [id.uuidString])
        }
    }

    func deleteCategory(id: UUID, moveEntriesTo category: DictionaryCategory?) throws {
        try database.dbQueue.write { db in
            if let category {
                let ids = try String.fetchAll(
                    db,
                    sql: "SELECT id FROM dictionary_entries WHERE categoryID = ?",
                    arguments: [id.uuidString]
                )
                try db.execute(
                    sql: """
                        UPDATE dictionary_entries
                        SET categoryID = ?, categoryNameSnapshot = ?, updatedAt = ?
                        WHERE categoryID = ?
                        """,
                    arguments: [
                        category.id.uuidString,
                        category.name,
                        Date().timeIntervalSince1970,
                        id.uuidString
                    ]
                )
                for entry in try entries(for: ids, db: db) {
                    try refreshSearchContent(for: entry, db: db)
                }
            } else {
                let ids = try String.fetchAll(
                    db,
                    sql: "SELECT id FROM dictionary_entries WHERE categoryID = ?",
                    arguments: [id.uuidString]
                )
                if !ids.isEmpty {
                    try db.execute(
                        sql: "DELETE FROM dictionary_search WHERE entryID IN \(sqlPlaceholders(count: ids.count))",
                        arguments: StatementArguments(ids)
                    )
                    try db.execute(
                        sql: "DELETE FROM dictionary_entries WHERE id IN \(sqlPlaceholders(count: ids.count))",
                        arguments: StatementArguments(ids)
                    )
                }
            }
            try db.execute(sql: "DELETE FROM dictionary_categories WHERE id = ?", arguments: [id.uuidString])
        }
    }

    func clearAll() throws {
        try replaceAll([])
    }

    func hasEntry(normalizedTerm: String, activeGroupID: UUID?) throws -> Bool {
        let group = activeGroupID?.uuidString ?? ""
        return try database.dbQueue.read { db in
            let entryCount = try Int.fetchOne(
                db,
                sql: """
                    SELECT COUNT(*) FROM dictionary_entries
                    WHERE COALESCE(groupID, '') = ? AND normalizedTerm = ?
                    """,
                arguments: [group, normalizedTerm]
            ) ?? 0
            if entryCount > 0 {
                return true
            }
            let replacementCount = try Int.fetchOne(
                db,
                sql: """
                    SELECT COUNT(*) FROM dictionary_replacement_terms r
                    JOIN dictionary_entries e ON e.id = r.entryID
                    WHERE COALESCE(e.groupID, '') = ? AND r.normalizedText = ?
                    """,
                arguments: [group, normalizedTerm]
            ) ?? 0
            return replacementCount > 0
        }
    }

    private func searchEntryIDs(query: String) throws -> [String]? {
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let ftsQuery = VoxtFTSQueryBuilder.query(from: trimmedQuery) else {
            return nil
        }
        return try database.dbQueue.read { db in
            try String.fetchAll(
                db,
                sql: "SELECT entryID FROM dictionary_search WHERE dictionary_search MATCH ?",
                arguments: [ftsQuery]
            )
        }
    }

    private func fetchEntries(sql: String, arguments: StatementArguments) throws -> [DictionaryEntry] {
        try database.dbQueue.read { db in
            let ids = try String.fetchAll(db, sql: sql, arguments: arguments)
            guard !ids.isEmpty else { return [] }
            return try entries(for: ids, db: db)
        }
    }

    private func entries(for ids: [String], db: Database) throws -> [DictionaryEntry] {
        guard !ids.isEmpty else { return [] }

        var entryRowsByID: [String: Row] = [:]
        var replacementsByEntryID: [String: [DictionaryReplacementTerm]] = [:]
        var variantsByEntryID: [String: [ObservedVariant]] = [:]

        for chunk in chunks(ids, size: 500) {
            let entryRows = try Row.fetchAll(
                db,
                sql: "SELECT * FROM dictionary_entries WHERE id IN \(sqlPlaceholders(count: chunk.count))",
                arguments: StatementArguments(chunk)
            )
            for row in entryRows {
                let id: String = row["id"]
                entryRowsByID[id] = row
            }

            let replacementRows = try Row.fetchAll(
                db,
                sql: """
                    SELECT * FROM dictionary_replacement_terms
                    WHERE entryID IN \(sqlPlaceholders(count: chunk.count))
                    ORDER BY entryID, text COLLATE NOCASE ASC
                    """,
                arguments: StatementArguments(chunk)
            )
            for row in replacementRows {
                let entryID: String = row["entryID"]
                guard let replacement = replacement(from: row) else { continue }
                replacementsByEntryID[entryID, default: []].append(replacement)
            }

            let variantRows = try Row.fetchAll(
                db,
                sql: """
                    SELECT * FROM dictionary_observed_variants
                    WHERE entryID IN \(sqlPlaceholders(count: chunk.count))
                    ORDER BY entryID, count DESC, lastSeenAt DESC
                    """,
                arguments: StatementArguments(chunk)
            )
            for row in variantRows {
                let entryID: String = row["entryID"]
                guard let variant = observedVariant(from: row) else { continue }
                variantsByEntryID[entryID, default: []].append(variant)
            }
        }

        var entries: [DictionaryEntry] = []
        entries.reserveCapacity(ids.count)
        for id in ids {
            guard let row = entryRowsByID[id] else { continue }
            entries.append(
                entry(
                    from: row,
                    replacementTerms: replacementsByEntryID[id] ?? [],
                    observedVariants: variantsByEntryID[id] ?? []
                )
            )
        }

        return entries
    }

    private func upsert(_ entry: DictionaryEntry, db: Database) throws {
        try db.execute(
            sql: """
                INSERT INTO dictionary_entries (
                    id, term, normalizedTerm, categoryID, categoryNameSnapshot,
                    groupID, groupNameSnapshot, source, status,
                    createdAt, updatedAt, lastMatchedAt, matchCount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    term = excluded.term,
                    normalizedTerm = excluded.normalizedTerm,
                    categoryID = excluded.categoryID,
                    categoryNameSnapshot = excluded.categoryNameSnapshot,
                    groupID = excluded.groupID,
                    groupNameSnapshot = excluded.groupNameSnapshot,
                    source = excluded.source,
                    status = excluded.status,
                    createdAt = excluded.createdAt,
                    updatedAt = excluded.updatedAt,
                    lastMatchedAt = excluded.lastMatchedAt,
                    matchCount = excluded.matchCount
                """,
            arguments: VoxtDatabaseArguments.make([
                entry.id.uuidString,
                entry.term,
                entry.normalizedTerm,
                entry.categoryID.uuidString,
                entry.categoryNameSnapshot,
                entry.groupID?.uuidString,
                entry.groupNameSnapshot,
                entry.source.rawValue,
                entry.status.rawValue,
                entry.createdAt.timeIntervalSince1970,
                entry.updatedAt.timeIntervalSince1970,
                entry.lastMatchedAt?.timeIntervalSince1970,
                entry.matchCount
            ])
        )

        try db.execute(sql: "DELETE FROM dictionary_replacement_terms WHERE entryID = ?", arguments: [entry.id.uuidString])
        for replacement in entry.replacementTerms {
            try db.execute(
                sql: """
                    INSERT INTO dictionary_replacement_terms
                    (id, entryID, groupID, text, normalizedText) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        entryID = excluded.entryID,
                        groupID = excluded.groupID,
                        text = excluded.text,
                        normalizedText = excluded.normalizedText
                    """,
                arguments: VoxtDatabaseArguments.make([
                    replacement.id.uuidString,
                    entry.id.uuidString,
                    entry.groupID?.uuidString,
                    replacement.text,
                    replacement.normalizedText
                ])
            )
        }

        try db.execute(sql: "DELETE FROM dictionary_observed_variants WHERE entryID = ?", arguments: [entry.id.uuidString])
        for variant in entry.observedVariants {
            try db.execute(
                sql: """
                    INSERT INTO dictionary_observed_variants
                    (id, entryID, text, normalizedText, count, lastSeenAt, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        entryID = excluded.entryID,
                        text = excluded.text,
                        normalizedText = excluded.normalizedText,
                        count = excluded.count,
                        lastSeenAt = excluded.lastSeenAt,
                        confidence = excluded.confidence
                    """,
                arguments: VoxtDatabaseArguments.make([
                    variant.id.uuidString,
                    entry.id.uuidString,
                    variant.text,
                    variant.normalizedText,
                    variant.count,
                    variant.lastSeenAt.timeIntervalSince1970,
                    variant.confidence.rawValue
                ])
            )
        }

        try refreshSearchContent(for: entry, db: db)
    }

    private func refreshSearchContent(for entry: DictionaryEntry, db: Database) throws {
        try db.execute(sql: "DELETE FROM dictionary_search WHERE entryID = ?", arguments: [entry.id.uuidString])
        try db.execute(
            sql: """
                INSERT INTO dictionary_search (entryID, term, aliases, variants, groupName)
                VALUES (?, ?, ?, ?, ?)
                """,
            arguments: VoxtDatabaseArguments.make([
                entry.id.uuidString,
                entry.term,
                entry.replacementTerms.map(\.text).joined(separator: " "),
                entry.observedVariants.map(\.text).joined(separator: " "),
                [entry.categoryNameSnapshot, entry.groupNameSnapshot]
                    .compactMap { $0 }
                    .joined(separator: " ")
                ])
        )
    }

    private func upsert(_ category: DictionaryCategory, db: Database) throws {
        try db.execute(
            sql: """
                INSERT INTO dictionary_categories (
                    id, name, normalizedName, isDefault, isExpanded, sortOrder, createdAt, updatedAt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    normalizedName = excluded.normalizedName,
                    isDefault = excluded.isDefault,
                    isExpanded = excluded.isExpanded,
                    sortOrder = excluded.sortOrder,
                    createdAt = excluded.createdAt,
                    updatedAt = excluded.updatedAt
                """,
            arguments: VoxtDatabaseArguments.make([
                category.id.uuidString,
                category.name,
                category.normalizedName,
                category.isDefault ? 1 : 0,
                category.isExpanded ? 1 : 0,
                category.sortOrder,
                category.createdAt.timeIntervalSince1970,
                category.updatedAt.timeIntervalSince1970
            ])
        )
    }

    private func entry(
        from row: Row,
        replacementTerms: [DictionaryReplacementTerm],
        observedVariants: [ObservedVariant]
    ) -> DictionaryEntry {
        let id: String = row["id"]
        let term: String = row["term"]
        let normalizedTerm: String = row["normalizedTerm"]
        let source: String = row["source"]
        let createdAt: Double = row["createdAt"]
        let updatedAt: Double = row["updatedAt"]
        let matchCount: Int = row["matchCount"]
        let status: String = row["status"]
        return DictionaryEntry(
            id: UUID(uuidString: id) ?? UUID(),
            term: term,
            normalizedTerm: normalizedTerm,
            categoryID: (row["categoryID"] as String?).flatMap(UUID.init(uuidString:)) ?? DictionaryCategory.defaultID,
            categoryNameSnapshot: (row["categoryNameSnapshot"] as String?) ?? DictionaryCategory.defaultName,
            groupID: (row["groupID"] as String?).flatMap(UUID.init(uuidString:)),
            groupNameSnapshot: row["groupNameSnapshot"],
            source: DictionaryEntrySource(rawValue: source) ?? .manual,
            createdAt: Date(timeIntervalSince1970: createdAt),
            updatedAt: Date(timeIntervalSince1970: updatedAt),
            lastMatchedAt: (row["lastMatchedAt"] as Double?).map(Date.init(timeIntervalSince1970:)),
            matchCount: matchCount,
            status: DictionaryEntryStatus(rawValue: status) ?? .active,
            observedVariants: observedVariants,
            replacementTerms: replacementTerms
        )
    }

    private func category(from row: Row) -> DictionaryCategory? {
        let idText: String = row["id"]
        guard let id = UUID(uuidString: idText) else { return nil }
        let name: String = row["name"]
        let normalizedName: String = row["normalizedName"]
        let isDefault: Int = row["isDefault"]
        let isExpanded: Int = row["isExpanded"]
        let sortOrder: Int = row["sortOrder"]
        let createdAt: Double = row["createdAt"]
        let updatedAt: Double = row["updatedAt"]
        return DictionaryCategory(
            id: id,
            name: name,
            normalizedName: normalizedName,
            isDefault: isDefault != 0,
            isExpanded: isExpanded != 0,
            sortOrder: sortOrder,
            createdAt: Date(timeIntervalSince1970: createdAt),
            updatedAt: Date(timeIntervalSince1970: updatedAt)
        )
    }

    private func replacement(from row: Row) -> DictionaryReplacementTerm? {
        let idText: String = row["id"]
        guard let id = UUID(uuidString: idText) else { return nil }
        let text: String = row["text"]
        let normalizedText: String = row["normalizedText"]
        return DictionaryReplacementTerm(
            id: id,
            text: text,
            normalizedText: normalizedText
        )
    }

    private func observedVariant(from row: Row) -> ObservedVariant? {
        let idText: String = row["id"]
        guard let id = UUID(uuidString: idText) else { return nil }
        let text: String = row["text"]
        let normalizedText: String = row["normalizedText"]
        let count: Int = row["count"]
        let lastSeenAt: Double = row["lastSeenAt"]
        let confidence: String = row["confidence"]
        return ObservedVariant(
            id: id,
            text: text,
            normalizedText: normalizedText,
            count: count,
            lastSeenAt: Date(timeIntervalSince1970: lastSeenAt),
            confidence: DictionaryVariantConfidence(rawValue: confidence) ?? .medium
        )
    }

    private func migrateLegacyJSONIfNeeded() {
        do {
            let existingCount = try database.dbQueue.read { db in
                try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM dictionary_entries") ?? 0
            }
            guard existingCount == 0,
                  let legacyJSONURL,
                  fileManager.fileExists(atPath: legacyJSONURL.path)
            else {
                return
            }

            let data = try Data(contentsOf: legacyJSONURL)
            let legacyEntries = try VoxtPersistenceCoding.decoder.decode([DictionaryEntry].self, from: data)
            let (categories, entries) = categoriesAndEntriesForLegacyJSONMigration(legacyEntries)
            for category in categories {
                try upsertCategory(category)
            }
            try replaceAll(entries)
            try backupLegacyJSON(at: legacyJSONURL)
        } catch {
            VoxtLog.warning("Dictionary SQLite migration skipped or failed: \(error.localizedDescription)")
        }
    }

    private func categoriesAndEntriesForLegacyJSONMigration(
        _ entries: [DictionaryEntry]
    ) -> (categories: [DictionaryCategory], entries: [DictionaryEntry]) {
        var categoriesByID: [UUID: DictionaryCategory] = [
            DictionaryCategory.defaultID: DictionaryCategory.defaultCategory
        ]
        var migratedEntries: [DictionaryEntry] = []
        migratedEntries.reserveCapacity(entries.count)

        for entry in entries {
            guard entry.categoryID == DictionaryCategory.defaultID,
                  let groupID = entry.groupID
            else {
                categoriesByID[entry.categoryID] = categoriesByID[entry.categoryID] ?? DictionaryCategory(
                    id: entry.categoryID,
                    name: entry.categoryNameSnapshot ?? DictionaryCategory.defaultName,
                    isDefault: entry.categoryID == DictionaryCategory.defaultID
                )
                migratedEntries.append(entry)
                continue
            }

            let groupName = entry.groupNameSnapshot?.trimmingCharacters(in: .whitespacesAndNewlines)
            let categoryName = (groupName?.isEmpty == false ? groupName : nil)
                ?? AppLocalization.localizedString("Imported Category")
            if categoriesByID[groupID] == nil {
                categoriesByID[groupID] = DictionaryCategory(
                    id: groupID,
                    name: categoryName,
                    normalizedName: DictionaryStore.normalizeTerm(categoryName),
                    isDefault: false,
                    isExpanded: true,
                    sortOrder: categoriesByID.count
                )
            }

            var migratedEntry = entry
            migratedEntry.categoryID = groupID
            migratedEntry.categoryNameSnapshot = categoryName
            migratedEntries.append(migratedEntry)
        }

        let categories = categoriesByID.values.sorted {
            if $0.isDefault != $1.isDefault {
                return $0.isDefault && !$1.isDefault
            }
            if $0.sortOrder != $1.sortOrder {
                return $0.sortOrder < $1.sortOrder
            }
            return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
        return (categories, migratedEntries)
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
            .appendingPathComponent("dictionary.json")
    }

    private func sqlPlaceholders(count: Int) -> String {
        "(\(Array(repeating: "?", count: count).joined(separator: ",")))"
    }

    private func chunks<Value>(_ values: [Value], size: Int) -> [[Value]] {
        guard size > 0 else { return [values] }
        var result: [[Value]] = []
        result.reserveCapacity((values.count + size - 1) / size)
        var startIndex = values.startIndex
        while startIndex < values.endIndex {
            let endIndex = values.index(startIndex, offsetBy: size, limitedBy: values.endIndex) ?? values.endIndex
            result.append(Array(values[startIndex..<endIndex]))
            startIndex = endIndex
        }
        return result
    }

    private func scopeSQL(activeGroupID: UUID?) -> String {
        activeGroupID == nil ? "e.groupID IS NULL" : "(e.groupID IS NULL OR e.groupID = ?)"
    }

    private func matchingEntriesArguments(
        normalizedSource: String,
        activeGroupID: UUID?,
        limit: Int
    ) -> StatementArguments {
        var arguments: StatementArguments = []
        for _ in 0..<3 {
            arguments += [DictionaryEntryStatus.active.rawValue]
            if let activeGroupID {
                arguments += [activeGroupID.uuidString]
            }
            arguments += [normalizedSource]
        }
        arguments += [limit]
        return arguments
    }

    private func entryMatchesSource(_ entry: DictionaryEntry, normalizedSource: String) -> Bool {
        normalizedNeedles(for: entry).contains { sourceContainsNeedle($0, normalizedSource: normalizedSource) }
    }

    private func normalizedNeedles(for entry: DictionaryEntry) -> [String] {
        var values = [entry.normalizedTerm]
        values.append(contentsOf: entry.replacementTerms.map(\.normalizedText))
        values.append(contentsOf: entry.observedVariants.map(\.normalizedText))
        return values
            .map(DictionaryStore.normalizeTerm)
            .filter { !$0.isEmpty }
    }

    private func sourceContainsNeedle(_ needle: String, normalizedSource: String) -> Bool {
        var searchRange: Range<String.Index>? = normalizedSource.startIndex..<normalizedSource.endIndex
        while let range = normalizedSource.range(of: needle, options: [], range: searchRange) {
            if hasValidBoundary(before: range.lowerBound, after: range.upperBound, needle: needle, source: normalizedSource) {
                return true
            }
            searchRange = range.upperBound..<normalizedSource.endIndex
        }
        return false
    }

    private func hasValidBoundary(
        before lowerBound: String.Index,
        after upperBound: String.Index,
        needle: String,
        source: String
    ) -> Bool {
        let needsLeadingBoundary = needle.unicodeScalars.first.map(isASCIIAlphaNumeric) ?? false
        let needsTrailingBoundary = needle.unicodeScalars.last.map(isASCIIAlphaNumeric) ?? false

        if needsLeadingBoundary,
           lowerBound > source.startIndex,
           let previous = source[..<lowerBound].unicodeScalars.last,
           isASCIIAlphaNumeric(previous) {
            return false
        }

        if needsTrailingBoundary,
           upperBound < source.endIndex,
           let next = source[upperBound...].unicodeScalars.first,
           isASCIIAlphaNumeric(next) {
            return false
        }

        return true
    }

    private func isASCIIAlphaNumeric(_ scalar: UnicodeScalar) -> Bool {
        (65...90).contains(Int(scalar.value))
            || (97...122).contains(Int(scalar.value))
            || (48...57).contains(Int(scalar.value))
    }
}
