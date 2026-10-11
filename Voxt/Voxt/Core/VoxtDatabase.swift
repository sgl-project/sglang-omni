// VoxtDatabase.swift
// Provides Voxt Database for core app behavior.

import Foundation
import GRDB

final class VoxtDatabase: @unchecked Sendable {
    static let shared = VoxtDatabase()

    let dbQueue: DatabaseQueue

    init(fileManager: FileManager = .default, databaseURL: URL? = nil) {
        do {
            let resolvedURL = try databaseURL ?? Self.defaultDatabaseURL(fileManager: fileManager)
            try fileManager.createDirectory(
                at: resolvedURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            var configuration = Configuration()
            configuration.journalMode = .wal
            configuration.prepareDatabase { db in
                try db.execute(sql: "PRAGMA foreign_keys = ON")
            }

            dbQueue = try DatabaseQueue(path: resolvedURL.path, configuration: configuration)
            try Self.migrator.migrate(dbQueue)
        } catch {
            fatalError("Failed to initialize Voxt database: \(error)")
        }
    }

    static func defaultDatabaseURL(fileManager: FileManager = .default) throws -> URL {
        let appSupport = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("voxt.sqlite")
    }

    #if DEBUG
    func debugSQLiteObjectNames(type: String) throws -> [String] {
        try dbQueue.read { db in
            try String.fetchAll(
                db,
                sql: "SELECT name FROM sqlite_master WHERE type = ?",
                arguments: [type]
            )
        }
    }
    #endif

    private static var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1_dictionary") { db in
            try db.execute(sql: """
                CREATE TABLE IF NOT EXISTS dictionary_entries (
                    id TEXT PRIMARY KEY NOT NULL,
                    term TEXT NOT NULL,
                    normalizedTerm TEXT NOT NULL,
                    groupID TEXT,
                    groupNameSnapshot TEXT,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    createdAt REAL NOT NULL,
                    updatedAt REAL NOT NULL,
                    lastMatchedAt REAL,
                    matchCount INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS dictionary_replacement_terms (
                    id TEXT PRIMARY KEY NOT NULL,
                    entryID TEXT NOT NULL REFERENCES dictionary_entries(id) ON DELETE CASCADE,
                    groupID TEXT,
                    text TEXT NOT NULL,
                    normalizedText TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dictionary_observed_variants (
                    id TEXT PRIMARY KEY NOT NULL,
                    entryID TEXT NOT NULL REFERENCES dictionary_entries(id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    normalizedText TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    lastSeenAt REAL NOT NULL,
                    confidence TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dictionary_updated
                    ON dictionary_entries(updatedAt DESC, term COLLATE NOCASE ASC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_dictionary_normalized_scope
                    ON dictionary_entries(COALESCE(groupID, ''), normalizedTerm);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_dictionary_replacement_scope
                    ON dictionary_replacement_terms(COALESCE(groupID, ''), normalizedText);
                CREATE INDEX IF NOT EXISTS idx_dictionary_replacement_entry
                    ON dictionary_replacement_terms(entryID);
                CREATE INDEX IF NOT EXISTS idx_dictionary_variant_entry
                    ON dictionary_observed_variants(entryID);
                """)
        }

        migrator.registerMigration("v2_history") { db in
            try db.execute(sql: """
                CREATE TABLE IF NOT EXISTS history_entries (
                    id TEXT PRIMARY KEY NOT NULL,
                    text TEXT NOT NULL,
                    createdAt REAL NOT NULL,
                    kind TEXT NOT NULL,
                    transcriptionEngine TEXT NOT NULL,
                    transcriptionModel TEXT NOT NULL,
                    enhancementMode TEXT NOT NULL,
                    enhancementModel TEXT NOT NULL,
                    isTranslation INTEGER NOT NULL,
                    audioDurationSeconds REAL,
                    transcriptionProcessingDurationSeconds REAL,
                    llmDurationSeconds REAL,
                    focusedAppName TEXT,
                    focusedAppBundleID TEXT,
                    matchedGroupID TEXT,
                    matchedGroupName TEXT,
                    matchedAppGroupName TEXT,
                    matchedURLGroupName TEXT,
                    remoteASRProvider TEXT,
                    remoteASRModel TEXT,
                    remoteASREndpoint TEXT,
                    remoteLLMProvider TEXT,
                    remoteLLMModel TEXT,
                    remoteLLMEndpoint TEXT,
                    audioRelativePath TEXT,
                    transcriptAudioRelativePath TEXT,
                    displayTitle TEXT,
                    dictionaryTermsText TEXT,
                    entryJSON TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_history_created
                    ON history_entries(createdAt DESC);
                CREATE INDEX IF NOT EXISTS idx_history_kind_created
                    ON history_entries(kind, createdAt DESC);
                CREATE INDEX IF NOT EXISTS idx_history_focused_app
                    ON history_entries(focusedAppBundleID, focusedAppName);
                CREATE INDEX IF NOT EXISTS idx_history_group
                    ON history_entries(matchedGroupID);
                """)
        }

        migrator.registerMigration("v3_fts") { db in
            try db.execute(sql: """
                CREATE VIRTUAL TABLE IF NOT EXISTS dictionary_search
                    USING fts5(entryID UNINDEXED, term, aliases, variants, groupName);
                CREATE VIRTUAL TABLE IF NOT EXISTS history_search
                    USING fts5(historyID UNINDEXED, text, displayTitle, focusedAppName, matchedGroupName, dictionaryTerms);
                """)
        }

        migrator.registerMigration("v4_dictionary_active_scope_indexes") { db in
            try db.execute(sql: """
                CREATE INDEX IF NOT EXISTS idx_dictionary_active_scope_rank
                    ON dictionary_entries(
                        status,
                        groupID,
                        matchCount DESC,
                        lastMatchedAt DESC,
                        updatedAt DESC,
                        term COLLATE NOCASE ASC
                    );
                """)
        }

        migrator.registerMigration("v5_history_browser_context") { db in
            try db.execute(sql: """
                ALTER TABLE history_entries ADD COLUMN browserURLHost TEXT;
                ALTER TABLE history_entries ADD COLUMN browserURLOrigin TEXT;
                CREATE INDEX IF NOT EXISTS idx_history_browser_host
                    ON history_entries(browserURLHost);
                """)
        }

        migrator.registerMigration("v6_dictionary_categories") { db in
            try db.execute(sql: """
                CREATE TABLE IF NOT EXISTS dictionary_categories (
                    id TEXT PRIMARY KEY NOT NULL,
                    name TEXT NOT NULL,
                    normalizedName TEXT NOT NULL,
                    isDefault INTEGER NOT NULL,
                    isExpanded INTEGER NOT NULL,
                    sortOrder INTEGER NOT NULL,
                    createdAt REAL NOT NULL,
                    updatedAt REAL NOT NULL
                );

                INSERT OR IGNORE INTO dictionary_categories (
                    id, name, normalizedName, isDefault, isExpanded, sortOrder, createdAt, updatedAt
                ) VALUES (
                    '00000000-0000-0000-0000-000000000001',
                    'Default',
                    'default',
                    1,
                    1,
                    0,
                    strftime('%s', 'now'),
                    strftime('%s', 'now')
                );

                ALTER TABLE dictionary_entries
                    ADD COLUMN categoryID TEXT DEFAULT '00000000-0000-0000-0000-000000000001';
                ALTER TABLE dictionary_entries
                    ADD COLUMN categoryNameSnapshot TEXT DEFAULT 'Default';

                INSERT OR IGNORE INTO dictionary_categories (
                    id, name, normalizedName, isDefault, isExpanded, sortOrder, createdAt, updatedAt
                )
                SELECT
                    groupID,
                    COALESCE(NULLIF(groupNameSnapshot, ''), 'Imported Category'),
                    lower(COALESCE(NULLIF(groupNameSnapshot, ''), 'Imported Category')),
                    0,
                    1,
                    row_number() OVER (ORDER BY COALESCE(groupNameSnapshot, groupID)),
                    strftime('%s', 'now'),
                    strftime('%s', 'now')
                FROM dictionary_entries
                WHERE groupID IS NOT NULL
                GROUP BY groupID;

                UPDATE dictionary_entries
                SET
                    categoryID = groupID,
                    categoryNameSnapshot = COALESCE(NULLIF(groupNameSnapshot, ''), 'Imported Category')
                WHERE groupID IS NOT NULL;

                UPDATE dictionary_entries
                SET
                    categoryID = '00000000-0000-0000-0000-000000000001',
                    categoryNameSnapshot = 'Default'
                WHERE categoryID IS NULL OR categoryID = '';

                CREATE INDEX IF NOT EXISTS idx_dictionary_category_order
                    ON dictionary_categories(isDefault DESC, sortOrder ASC, name COLLATE NOCASE ASC);
                CREATE INDEX IF NOT EXISTS idx_dictionary_entries_category
                    ON dictionary_entries(categoryID, updatedAt DESC);
                """)
        }

        return migrator
    }
}

enum VoxtPersistenceCoding {
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        return encoder
    }()

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        return decoder
    }()

    static func encodeJSONString<Value: Encodable>(_ value: Value) throws -> String {
        let data = try encoder.encode(value)
        guard let text = String(data: data, encoding: .utf8) else {
            throw CocoaError(.fileWriteUnknown)
        }
        return text
    }

    static func decodeJSON<Value: Decodable>(_ type: Value.Type, from text: String) throws -> Value {
        try decoder.decode(type, from: Data(text.utf8))
    }
}

enum VoxtDatabaseArguments {
    static func make(_ values: [(any DatabaseValueConvertible)?]) -> StatementArguments {
        StatementArguments(values)
    }
}

enum VoxtFTSQueryBuilder {
    static func query(from input: String) -> String? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let tokens = trimmed
            .components(separatedBy: CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let resolvedTokens = tokens.isEmpty ? [trimmed] : tokens
        let escaped = resolvedTokens.map {
            "\"\($0.replacingOccurrences(of: "\"", with: "\"\""))\"*"
        }
        return escaped.joined(separator: " AND ")
    }
}
