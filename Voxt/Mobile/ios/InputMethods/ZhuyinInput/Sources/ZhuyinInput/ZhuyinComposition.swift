import Foundation

/// Completed syllables plus one editable syllable. A tone key completes the
/// current syllable; space supplies first tone. No network or dictionary here.
public struct ZhuyinComposition: Sendable {
    private var readings: [String] = []
    private var parts: [Int: String] = [:]
    private static let initials = "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ"
    private static let medials = "ㄧㄨㄩ"
    private static let finals = "ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ"
    private static let tones = "ˊˇˋ˙"

    public init() {}
    private var current: String { parts.keys.sorted().compactMap { parts[$0] }.joined() }
    public var display: String { (readings + (current.isEmpty ? [] : [current])).joined(separator: " ") }
    public var readingsForConversion: [String]? { parts.isEmpty && !readings.isEmpty ? readings : nil }

    @discardableResult public mutating func press(_ key: String) -> Bool {
        guard key.count == 1 else { return false }
        let previous = self
        if key == " " || Self.tones.contains(key) {
            guard !parts.isEmpty,
                  parts[1] != nil || parts[2] != nil || parts[0].map({ "ㄓㄔㄕㄖㄗㄘㄙ".contains($0) }) == true else { return false }
            readings.append(current + (key == " " ? "" : key))
            parts.removeAll()
        } else {
            guard readings.count < 32 else { return false }
            if Self.initials.contains(key) { parts[0] = key }
            else if Self.medials.contains(key) { parts[1] = key }
            else if Self.finals.contains(key) { parts[2] = key }
            else { return false }
        }
        guard display.count <= 128 else { self = previous; return false }
        return true
    }

    public mutating func deleteBackward() {
        if parts.isEmpty, let reading = readings.popLast() {
            var symbols = Array(reading).map(String.init)
            // Reopen a completed syllable and remove its last typed key.
            // First tone has no visible mark, so delete removes its final symbol.
            if let last = symbols.last, Self.tones.contains(last) { symbols.removeLast() }
            else if !symbols.isEmpty { symbols.removeLast() }
            for symbol in symbols { _ = press(symbol) }
        } else if let last = parts.keys.max() { parts[last] = nil }
    }

    public mutating func reset() { readings.removeAll(); parts.removeAll() }
}
