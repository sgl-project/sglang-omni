import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct TranscriptHistoryTests {
    @Test func legacyHistorySurvivesMigrationAndSupportsSearchAndDeletion() throws {
        let name = "history.test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let id = UUID()
        // Existing app schema stores Date as seconds since 2001, without mode.
        let data = try JSONSerialization.data(withJSONObject: [["id": id.uuidString, "text": "Meet SGLang tomorrow", "date": 123456.0]])
        defaults.set(data, forKey: "history.items")
        let history = TranscriptHistory(defaults: defaults)
        #expect(history.items.first?.id == id)
        #expect(history.items.first?.elapsedSeconds == nil)
        #expect(history.search("sglang").count == 1)
        #expect(history.search("absent").isEmpty)
        history.append(text: "こんにちは", mode: .translate, elapsedSeconds: 3.5)
        #expect(TranscriptHistory(defaults: defaults).items.first?.mode == .translate)
        #expect(TranscriptHistory(defaults: defaults).items.first?.elapsedSeconds == 3.5)
        history.delete(ids: [id])
        #expect(history.items.map(\.text) == ["こんにちは"])
    }
    @Test func retentionPrunesOldEntriesAndDisabledHistoryDoesNotSave() {
        let name = "history.test.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let history = TranscriptHistory(defaults: defaults)
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        history.append(text: "old", now: now.addingTimeInterval(-8 * 86400))
        history.append(text: "recent", now: now.addingTimeInterval(-86400))
        history.setRetention(.sevenDays, now: now)
        #expect(history.items.map(\.text) == ["recent"])
        history.setRetention(.off, now: now)
        history.append(text: "private", now: now)
        #expect(history.items.isEmpty)
        #expect(TranscriptHistory(defaults: defaults).retention == .off)
        history.setRetention(.forever, now: now)
        history.append(text: "saved", now: now)
        #expect(history.items.map(\.text) == ["saved"])
        history.clear()
        #expect(history.items.isEmpty)
    }

}
