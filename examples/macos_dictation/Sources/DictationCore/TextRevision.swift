import Foundation

/// One contiguous edit, computed on grapheme boundaries and addressed in AX UTF-16 units.
public struct TextRevision {
    public let original: String
    public let corrected: String
    public let range: NSRange
    public let replacement: String
    public var isEmpty: Bool { range.length == 0 && replacement.isEmpty }

    public init(original: String, corrected: String) {
        self.original = original
        self.corrected = corrected
        let before = Array(original), after = Array(corrected)
        // Swift Character equality normalizes Unicode. Compare code units so an explicit
        // change between composed/decomposed spellings still produces the exact output.
        func equal(_ a: Character, _ b: Character) -> Bool {
            String(a).utf16.elementsEqual(String(b).utf16)
        }
        var prefix = 0
        while prefix < min(before.count, after.count), equal(before[prefix], after[prefix]) { prefix += 1 }
        var suffix = 0
        while suffix < min(before.count, after.count) - prefix,
              equal(before[before.count - suffix - 1], after[after.count - suffix - 1]) { suffix += 1 }
        let start = String(before.prefix(prefix)).utf16.count
        let length = String(before[prefix..<(before.count - suffix)]).utf16.count
        range = NSRange(location: start, length: length)
        replacement = String(after[prefix..<(after.count - suffix)])
    }
}

/// Retains only the last inserted span, without searching for similar text elsewhere.
public struct RevisionLocation {
    public private(set) var text: String
    public private(set) var range: NSRange

    public init(text: String, start: Int) {
        self.text = text
        range = NSRange(location: start, length: text.utf16.count)
    }

    public func validate(in value: String) throws {
        guard range.location >= 0, range.length >= 0, range.location <= value.utf16.count,
              range.length <= value.utf16.count - range.location,
              let span = Range(range, in: value),
              String(value[span]).utf16.elementsEqual(text.utf16) else {
            throw DictationError("上一段文字已变化或无法定位；更正结果仅供复制。")
        }
    }

    public func selection(for revision: TextRevision, in value: String) throws -> NSRange {
        try validate(in: value)
        guard text.utf16.elementsEqual(revision.original.utf16) else {
            throw DictationError("上一段的版本已变化；没有自动更正。")
        }
        return NSRange(location: range.location + revision.range.location, length: revision.range.length)
    }

    public mutating func accept(_ revision: TextRevision) {
        text = revision.corrected
        range.length = text.utf16.count
    }
}
