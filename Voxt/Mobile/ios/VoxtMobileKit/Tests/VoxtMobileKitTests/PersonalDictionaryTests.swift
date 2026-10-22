import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct PersonalDictionaryTests {
    @Test func savedTermsSurviveRestartAndCanBeEditedSearchedAndDeleted() throws {
        let name = "dictionary.test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let dictionary = PersonalDictionary(defaults: defaults)
        let entry = try dictionary.add("  SGLang  ")
        let reopened = PersonalDictionary(defaults: defaults)
        #expect(reopened.entries.map(\.term) == ["SGLang"])
        #expect(reopened.search("sgl").map(\.id) == [entry.id])
        try reopened.rename(entry.id, to: "SGLang Omni")
        #expect(reopened.vocabulary == ["SGLang Omni"])
        reopened.delete(entry.id)
        #expect(PersonalDictionary(defaults: defaults).entries.isEmpty)
    }
    @Test func invalidAndDuplicateTermsDoNotChangeSavedVocabulary() throws {
        let name = "dictionary.test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let dictionary = PersonalDictionary(defaults: defaults)
        let original = try dictionary.add("Voxt")
        #expect(throws: (any Error).self) { try dictionary.add("  ") }
        #expect(throws: (any Error).self) { try dictionary.add("VOXT") }
        #expect(throws: (any Error).self) { try dictionary.rename(original.id, to: String(repeating: "a", count: 101)) }
        #expect(dictionary.vocabulary == ["Voxt"])
        for index in 1..<200 { try dictionary.add("term \(index)") }
        #expect(throws: (any Error).self) { try dictionary.add("overflow") }
        #expect(dictionary.entries.count == 200)
    }

}
