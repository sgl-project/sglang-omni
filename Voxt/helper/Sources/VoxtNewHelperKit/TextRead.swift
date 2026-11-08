import Foundation

// textio.read's rules: when the helper refuses to read, which range a
// read covers (clamped to the value and to maxRead, widened to whole
// characters), and when it reads the whole value. The reading itself is
// Accessibility's (the helper's TextIO); the rules are here, pure, so
// that they are tested.

public enum TextRead {
    /// The most a read covers before widening, in UTF-16 units (proto.MaxRead).
    public static let maxRead = 4000
    /// Past this many characters the whole value is not read (the paste check's threshold).
    public static let maxValue = 200_000
    /// Units fetched beyond the range on each side, to widen it to whole characters.
    public static let margin = 16

    /// A password field, or secure event input on for the target or for
    /// an app nobody can tell (owner pid 0; nil: secure input is off).
    public static func refuses(subrole: String?, secureInputOwner: Int?, target: Int) -> Bool {
        subrole == "AXSecureTextField" || secureInputOwner == 0 || secureInputOwner == target
    }

    /// What a read of r gives: through byRange (AXStringForRange) when the
    /// value's length is known, else out of the whole value, which is read
    /// only up to maxValue characters; nil: nothing readable. Element is
    /// left to the caller.
    public static func read(_ r: TextRange, length: Int?, byRange: (TextRange) -> String?, value: () -> String?) -> ReadResult? {
        if let length, let got = cut(r, length: length, fetch: byRange) { return got }
        guard (length ?? 0) <= maxValue, let v = value() else { return nil }
        let ns = v as NSString
        return cut(r, length: ns.length) { ns.substring(with: NSRange(location: $0.location, length: $0.length)) }
    }

    /// r clamped to a value of length units and to maxRead, fetched with
    /// margin units more on each side, cut back to the whole characters
    /// that cover it.
    static func cut(_ r: TextRange, length: Int, fetch: (TextRange) -> String?) -> ReadResult? {
        let start = min(max(0, r.location), length)
        let end = min(max(start, r.location + r.length), length, start + maxRead)
        let from = max(0, start - margin)
        // An app may answer "" or a short string here: that is no read, so the value is tried.
        guard let fetched = fetch(TextRange(location: from, length: min(length, end + margin) - from)),
              (fetched as NSString).length >= start - from else { return nil }
        let s = fetched as NSString
        let a = min(start - from, s.length)
        var w = NSRange(location: a, length: min(end - from, s.length) - a)
        if w.length > 0 {
            w = s.rangeOfComposedCharacterSequences(for: w)
        } else if a < s.length {
            w.location = s.rangeOfComposedCharacterSequence(at: a).location
        }
        return ReadResult(text: s.substring(with: w), range: TextRange(location: from + w.location, length: w.length), length: length)
    }
}
