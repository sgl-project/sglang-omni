import XCTest
@testable import VoxtNewLLMKit

final class PromptCacheTests: XCTestCase {
    /// A cache holding Strings in place of a model's state, on a grid of 4.
    func makeCache(entries: Int = 4, bytes: Int = 100) -> PromptCache<String> {
        PromptCache(maximumEntries: entries, maximumBytes: bytes, minimumTokens: 3, gridTokens: 4)
    }

    func testMissThenHit() {
        var c = makeCache()
        let system = [1, 2, 3, 4, 5, 6]
        let first = c.plan(tokens: system + [9, 9], sharedPrefix: 6)
        XCTAssertNil(first.entry)
        XCTAssertEqual(first.prefixEnd, 6)
        XCTAssertEqual(first.keepAt, [6])
        c.insert(tokens: system, snapshot: "system", bytes: 10)

        let second = c.plan(tokens: system + [7, 8, 9], sharedPrefix: 6)
        XCTAssertEqual(second.entry?.snapshot, "system")
        XCTAssertEqual(second.reusedTokens, 6)
        XCTAssertNil(second.prefixEnd, "the prefix is kept already")
        XCTAssertEqual(second.keepAt, [])
    }

    /// A prompt that parts from a kept one inside the shared prefix keeps
    /// the grid point before that too, and the next such prompt starts
    /// there.
    func testBranchPoint() {
        var c = makeCache()
        c.insert(tokens: [1, 2, 3, 4, 5, 6, 7], snapshot: "glossary a", bytes: 1)
        let p = c.plan(tokens: [1, 2, 3, 4, 5, 8, 8, 0], sharedPrefix: 7)
        XCTAssertNil(p.entry)
        XCTAssertEqual(p.keepAt, [4, 7])
        c.insert(tokens: [1, 2, 3, 4], snapshot: "branch", bytes: 1)
        c.insert(tokens: [1, 2, 3, 4, 5, 8, 8], snapshot: "glossary b", bytes: 1)
        let q = c.plan(tokens: [1, 2, 3, 4, 9, 9, 9, 0], sharedPrefix: 7)
        XCTAssertEqual(q.entry?.snapshot, "branch")
        XCTAssertEqual(q.prefixEnd, 7)
        XCTAssertEqual(q.keepAt, [7])
        XCTAssertEqual(c.stops(from: 4, to: 7), [7])
        XCTAssertEqual(c.stops(from: 0, to: 9), [4, 8, 9])
        XCTAssertEqual(c.stops(from: 5, to: 8), [8])
    }

    /// An entry is reused whole, leaves a token to evaluate, and ends at
    /// a grid point or at the prompt's shared prefix (where a cold
    /// evaluation stops too).
    func testWhichEntriesServe() {
        var c = makeCache()
        c.insert(tokens: [1, 2, 3, 4], snapshot: "a", bytes: 1)
        XCTAssertNil(c.plan(tokens: [1, 2, 3, 4], sharedPrefix: 0).entry)
        XCTAssertNil(c.plan(tokens: [1, 2, 3], sharedPrefix: 0).entry)
        XCTAssertEqual(c.plan(tokens: [1, 2, 3, 4, 5], sharedPrefix: 0).entry?.snapshot, "a")
        c.insert(tokens: [1, 2, 3, 4, 5], snapshot: "b", bytes: 1)
        XCTAssertEqual(c.plan(tokens: [1, 2, 3, 4, 5, 6, 7, 8], sharedPrefix: 7).entry?.snapshot, "a", "b ends off the grid and before the shared prefix")
        XCTAssertEqual(c.plan(tokens: [1, 2, 3, 4, 5, 6, 7, 8], sharedPrefix: 5).entry?.snapshot, "b")
    }

    func testShortOrWholePrefixesAreNotKept() {
        var c = makeCache()
        XCTAssertEqual(c.plan(tokens: [1, 2, 3, 4], sharedPrefix: 2).keepAt, [], "under minimumTokens")
        XCTAssertEqual(c.plan(tokens: [1, 2, 3, 4], sharedPrefix: 2).prefixEnd, 2, "but split there")
        XCTAssertNil(c.plan(tokens: [1, 2, 3, 4], sharedPrefix: 4).prefixEnd, "nothing would be left")
        XCTAssertNil(c.plan(tokens: [1, 2, 3, 4], sharedPrefix: 0).prefixEnd)
    }

    /// Turned off, it keeps nothing but plans the same evaluation.
    func testOff() {
        var c = PromptCache<String>(maximumEntries: 0, maximumBytes: 100, minimumTokens: 3, gridTokens: 4)
        let p = c.plan(tokens: [1, 2, 3, 4, 5, 6], sharedPrefix: 5)
        XCTAssertEqual(p.prefixEnd, 5)
        XCTAssertEqual(p.keepAt, [])
        c.insert(tokens: [1, 2, 3, 4, 5], snapshot: "a", bytes: 1)
        XCTAssertTrue(c.entries.isEmpty)
    }

    func testLeastRecentlyUsedGoesFirst() {
        var c = makeCache(entries: 2, bytes: 25)
        c.insert(tokens: [1, 1, 1, 1], snapshot: "a", bytes: 10)
        c.insert(tokens: [2, 2, 2, 2], snapshot: "b", bytes: 10)
        _ = c.plan(tokens: [1, 1, 1, 1, 0], sharedPrefix: 0) // a is used
        c.insert(tokens: [3, 3, 3, 3], snapshot: "c", bytes: 10)
        XCTAssertEqual(c.entries.map(\.snapshot), ["c", "a"], "count bound")
        c.insert(tokens: [4, 4, 4, 4], snapshot: "d", bytes: 20)
        XCTAssertEqual(c.entries.map(\.snapshot), ["d"], "byte bound")
        c.insert(tokens: [5, 5, 5, 5], snapshot: "huge", bytes: 26)
        XCTAssertEqual(c.entries.map(\.snapshot), ["d"], "larger than the bound: not kept")
        c.insert(tokens: [4, 4, 4, 4], snapshot: "d2", bytes: 5)
        XCTAssertEqual(c.entries.map(\.snapshot), ["d2"], "same tokens: replaced")
        XCTAssertEqual(c.bytes, 5)
        c.removeAll()
        XCTAssertTrue(c.entries.isEmpty)
    }

    func testSharedPrefixLength() {
        XCTAssertEqual(sharedPrefixLength([1, 2, 3], [1, 2, 4]), 2)
        XCTAssertEqual(sharedPrefixLength([1, 2], [1, 2, 3]), 2)
        XCTAssertEqual(sharedPrefixLength([], [1]), 0)
    }
}
