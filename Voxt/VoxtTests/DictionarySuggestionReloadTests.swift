import Foundation
import XCTest
@testable import Voxt

@MainActor
final class DictionarySuggestionReloadTests: XCTestCase {
    func testSynchronousReloadSupersedesBlockedAsyncRead() async throws {
        let fixture = try Fixture()
        let reader = LegacyReader([item("old")])
        let store = fixture.store(reader: reader)
        reader.blockNext()
        let old = store.reloadAsync()
        defer { reader.release() }
        try reader.waitUntilBlocked()
        reader.replace([item("new")])
        store.reload()
        reader.release()
        await old.value
        XCTAssertEqual(store.suggestions.map(\.term), ["new"])
    }

    func testNewAsyncReadWinsEvenIfOlderReadFinishesLast() async throws {
        let fixture = try Fixture()
        let reader = LegacyReader([item("old")])
        let store = fixture.store(reader: reader)
        reader.blockNext()
        let old = store.reloadAsync()
        defer { reader.release() }
        try reader.waitUntilBlocked()
        reader.replace([item("new")])
        await store.reloadAsync().value
        reader.release()
        await old.value
        XCTAssertEqual(store.suggestions.map(\.term), ["new"])
    }

    func testCancelledReadWaitsForIOButCannotPublish() async throws {
        let fixture = try Fixture()
        let reader = LegacyReader([item("current")])
        let store = fixture.store(reader: reader)
        reader.replace([item("cancelled")])
        reader.blockNext()
        let read = store.reloadAsync()
        defer { reader.release() }
        try reader.waitUntilBlocked()
        read.cancel()
        let entered = ManualTaskBarrier()
        var completed = false
        let waiter = Task { entered.release(); await read.value; completed = true }
        await entered.wait()
        XCTAssertFalse(completed)
        reader.release()
        await waiter.value
        XCTAssertEqual(store.suggestions.map(\.term), ["current"])
    }

    func testCorruptFilePreservesSnapshotAndDoesNotRewriteFile() async throws {
        let fixture = try Fixture()
        try JSONEncoder().encode([item("current")]).write(to: fixture.url)
        let store = fixture.store()
        let invalid = Data("invalid JSON".utf8)
        try invalid.write(to: fixture.url)
        store.reload()
        await store.reloadAsync().value
        XCTAssertEqual(store.suggestions.map(\.term), ["current"])
        XCTAssertEqual(try Data(contentsOf: fixture.url), invalid)
    }

    func testMissingFileStillMeansEmptySnapshot() async throws {
        let fixture = try Fixture()
        try JSONEncoder().encode([item("current")]).write(to: fixture.url)
        let store = fixture.store()
        try FileManager.default.removeItem(at: fixture.url)
        await store.reloadAsync().value
        XCTAssertTrue(store.suggestions.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.url.path))
    }

    func testStaleDuplicateReadCannotWriteBackOverNewerFile() async throws {
        let fixture = try Fixture()
        let duplicate = item("old")
        let reader = LegacyReader([duplicate])
        let store = fixture.store(reader: reader)
        reader.replace([duplicate, duplicate])
        reader.blockNext()
        let old = store.reloadAsync()
        defer { reader.release() }
        try reader.waitUntilBlocked()
        let latest = [item("new")]
        let data = try JSONEncoder().encode(latest)
        try data.write(to: fixture.url)
        reader.replace(latest)
        store.reload()
        reader.release()
        await old.value
        XCTAssertEqual(try Data(contentsOf: fixture.url), data)
        XCTAssertEqual(store.suggestions.map(\.term), ["new"])
    }

    private func item(_ term: String) -> DictionarySuggestion {
        DictionarySuggestion(
            term: term, normalizedTerm: term, sourceContext: .history,
            firstSeenAt: Date(timeIntervalSinceReferenceDate: 1),
            lastSeenAt: Date(timeIntervalSinceReferenceDate: 2)
        )
    }
}

@MainActor
private final class Fixture {
    let directory: TemporaryDirectory
    let defaults: UserDefaults
    let name = UUID().uuidString
    let url: URL

    init() throws {
        directory = try TemporaryDirectory()
        url = directory.url.appendingPathComponent("suggestions.json")
        defaults = TestDoubles.makeUserDefaults(testName: name)
    }

    func store(reader: LegacyReader? = nil) -> DictionarySuggestionStore {
        if let reader {
            return DictionarySuggestionStore(defaults: defaults, legacySuggestionsURL: url, readLegacySuggestions: reader.read)
        }
        return DictionarySuggestionStore(defaults: defaults, legacySuggestionsURL: url)
    }

    isolated deinit { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
}

nonisolated private final class LegacyReader: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [DictionarySuggestion]
    private var shouldBlock = false
    private let entered = DispatchSemaphore(value: 0)
    private let released = DispatchSemaphore(value: 0)

    init(_ items: [DictionarySuggestion]) { self.items = items }
    func replace(_ items: [DictionarySuggestion]) { lock.withLock { self.items = items } }
    func blockNext() { lock.withLock { shouldBlock = true } }
    func release() { released.signal() }
    func waitUntilBlocked() throws {
        guard entered.wait(timeout: .now() + 3) == .success else { throw URLError(.timedOut) }
    }
    func read(_ url: URL) throws -> [DictionarySuggestion] {
        let snapshot = lock.withLock { () -> ([DictionarySuggestion], Bool) in
            let block = shouldBlock
            shouldBlock = false
            return (items, block)
        }
        if snapshot.1 { entered.signal(); released.wait() }
        return snapshot.0
    }
}
