import XCTest
@testable import VoxtNewHelperKit

/// textio.read's rules: the range a read covers (clamped, widened to
/// whole characters), when it reads the whole value, and when it refuses.
final class TextReadTests: XCTestCase {
    /// A read of r from value through AXStringForRange, as TextIO does it.
    private func read(_ value: String, _ location: Int, _ length: Int) -> ReadResult? {
        let ns = value as NSString
        return TextRead.read(TextRange(location: location, length: length), length: ns.length,
                             byRange: { ns.substring(with: NSRange(location: $0.location, length: $0.length)) },
                             value: { XCTFail("the whole value read"); return nil })
    }

    func testClamp() {
        let twenty = String(repeating: "x", count: 20)
        // A negative start: the part inside the value.
        XCTAssertEqual(read(twenty, -300, 310)?.range, TextRange(location: 0, length: 10))
        // Past the end.
        XCTAssertEqual(read(twenty, 15, 10)?.range, TextRange(location: 15, length: 5))
        XCTAssertEqual(read(twenty, 25, 5), ReadResult(range: TextRange(location: 20, length: 0), length: 20))
        // A negative length reads nothing.
        XCTAssertEqual(read(twenty, 5, -3)?.range, TextRange(location: 5, length: 0))
        // At most maxRead (4000).
        let long = read(String(repeating: "y", count: 10_000), 100, 5000)
        XCTAssertEqual(long?.range, TextRange(location: 100, length: TextRead.maxRead))
        XCTAssertEqual(long?.text?.utf16.count, 4000)
        XCTAssertEqual(long?.length, 10_000)
    }

    func testWholeCharacters() {
        // "a👍b": the emoji is the units 1 and 2.
        XCTAssertEqual(read("a👍b", 2, 1), ReadResult(text: "👍", range: TextRange(location: 1, length: 2), length: 4), "starts inside the emoji")
        XCTAssertEqual(read("a👍b", 0, 2)?.text, "a👍", "ends inside it")
        XCTAssertEqual(read("a👍b", 2, 0)?.range, TextRange(location: 1, length: 0), "an empty range inside it")
        // A family (one character of 11 units) and a flag (4 units).
        XCTAssertEqual(read("x👨‍👩‍👧‍👦y", 5, 1)?.range, TextRange(location: 1, length: 11))
        XCTAssertEqual(read("🇨🇳!", 2, 1)?.text, "🇨🇳")
        // The cap cuts into an emoji: widened past it.
        let capped = read(String(repeating: "z", count: 3999) + "👍", 0, 5000)
        XCTAssertEqual(capped?.range, TextRange(location: 0, length: 4001))
        XCTAssertEqual(capped?.text?.hasSuffix("👍"), true)
    }

    func testFetchesAMarginOnly() {
        var asked: [VoxtNewHelperKit.TextRange] = [] // XCTest brings Carbon's TextRange
        let r = TextRead.read(TextRange(location: 5000, length: 10), length: 10_000,
                              byRange: { asked.append($0); return String(repeating: "x", count: $0.length) },
                              value: { XCTFail("the whole value read"); return nil })
        XCTAssertEqual(asked, [TextRange(location: 5000 - TextRead.margin, length: 10 + 2 * TextRead.margin)])
        XCTAssertEqual(r?.range, TextRange(location: 5000, length: 10))
    }

    func testWholeValue() {
        // No AXStringForRange: the value, cut the same way.
        let small = TextRead.read(TextRange(location: 1, length: 2), length: 4, byRange: { _ in nil }, value: { "a👍b" })
        XCTAssertEqual(small, ReadResult(text: "👍", range: TextRange(location: 1, length: 2), length: 4))
        // Unknown length (no AXNumberOfCharacters): the value, as the paste check reads it.
        XCTAssertEqual(TextRead.read(TextRange(location: 0, length: 9), length: nil, byRange: { _ in XCTFail("no length, no range"); return nil },
                                     value: { "Hi" })?.text, "Hi")
        // Past 200 000 characters without AXStringForRange: the value is not read.
        XCTAssertNil(TextRead.read(TextRange(location: 0, length: 9), length: TextRead.maxValue + 1, byRange: { _ in nil },
                                   value: { XCTFail("a large value read"); return "" }))
        XCTAssertNotNil(TextRead.read(TextRange(location: 0, length: 9), length: TextRead.maxValue, byRange: { _ in nil },
                                      value: { String(repeating: "x", count: TextRead.maxValue) }))
        // No value at all.
        XCTAssertNil(TextRead.read(TextRange(location: 0, length: 9), length: 0, byRange: { _ in nil }, value: { nil }))
    }

    func testRefuses() {
        XCTAssertTrue(TextRead.refuses(subrole: "AXSecureTextField", secureInputOwner: nil, target: 9286), "a password field")
        XCTAssertTrue(TextRead.refuses(subrole: nil, secureInputOwner: 9286, target: 9286), "secure input of the target")
        XCTAssertTrue(TextRead.refuses(subrole: nil, secureInputOwner: 0, target: 9286), "secure input of an app nobody can tell")
        XCTAssertFalse(TextRead.refuses(subrole: nil, secureInputOwner: 1702, target: 9286), "another app's Secure Keyboard Entry")
        XCTAssertFalse(TextRead.refuses(subrole: "AXSearchField", secureInputOwner: nil, target: 9286))
    }
}
