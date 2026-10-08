// DictionaryTransferManager.swift
// Provides Dictionary Transfer Manager for dictionary matching and learning.

import Foundation
import UniformTypeIdentifiers
import SwiftUI

struct DictionaryTransferDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }

    var text: String

    init(text: String) {
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents,
              let text = String(data: data, encoding: .utf8)
        else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.text = text
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

enum DictionaryTransferManager {
    struct Payload: Codable {
        var version: Int
        var exportedAt: String
        var categories: [DictionaryCategory]
        var entries: [Entry]

        enum CodingKeys: String, CodingKey {
            case version
            case exportedAt
            case categories
            case entries
        }

        init(
            version: Int,
            exportedAt: String,
            categories: [DictionaryCategory],
            entries: [Entry]
        ) {
            self.version = version
            self.exportedAt = exportedAt
            self.categories = categories
            self.entries = entries
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            version = try container.decodeIfPresent(Int.self, forKey: .version) ?? 1
            exportedAt = try container.decodeIfPresent(String.self, forKey: .exportedAt) ?? ""
            categories = try container.decodeIfPresent([DictionaryCategory].self, forKey: .categories) ?? []
            entries = try container.decodeIfPresent([Entry].self, forKey: .entries) ?? []
        }
    }

    struct Entry: Codable {
        var term: String
        var categoryID: UUID?
        var categoryNameSnapshot: String?
        var groupID: UUID?
        var groupNameSnapshot: String?
        var replacementTerms: [String]

        enum CodingKeys: String, CodingKey {
            case term
            case categoryID
            case categoryNameSnapshot
            case groupID
            case groupNameSnapshot
            case replacementTerms
        }

        init(
            term: String,
            categoryID: UUID?,
            categoryNameSnapshot: String?,
            groupID: UUID?,
            groupNameSnapshot: String?,
            replacementTerms: [String] = []
        ) {
            self.term = term
            self.categoryID = categoryID
            self.categoryNameSnapshot = categoryNameSnapshot
            self.groupID = groupID
            self.groupNameSnapshot = groupNameSnapshot
            self.replacementTerms = replacementTerms
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            term = try container.decode(String.self, forKey: .term)
            categoryID = try container.decodeIfPresent(UUID.self, forKey: .categoryID)
            categoryNameSnapshot = try container.decodeIfPresent(String.self, forKey: .categoryNameSnapshot)
            groupID = try container.decodeIfPresent(UUID.self, forKey: .groupID)
            groupNameSnapshot = try container.decodeIfPresent(String.self, forKey: .groupNameSnapshot)
            replacementTerms = try container.decodeIfPresent([String].self, forKey: .replacementTerms) ?? []
        }
    }

    static func exportJSONString(entries: [DictionaryEntry], categories: [DictionaryCategory]) throws -> String {
        let payload = Payload(
            version: 3,
            exportedAt: iso8601Formatter.string(from: Date()),
            categories: categories,
            entries: entries.map {
                Entry(
                    term: $0.term,
                    categoryID: $0.categoryID,
                    categoryNameSnapshot: $0.categoryNameSnapshot,
                    groupID: $0.groupID,
                    groupNameSnapshot: $0.groupNameSnapshot,
                    replacementTerms: $0.replacementTerms.map(\.text)
                )
            }
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(payload)
        guard let text = String(data: data, encoding: .utf8) else {
            throw CocoaError(.fileWriteUnknown)
        }
        return text
    }

    static func importPayload(from json: String) throws -> Payload {
        let data = Data(json.utf8)
        return try JSONDecoder().decode(Payload.self, from: data)
    }

    private static let iso8601Formatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()
}
