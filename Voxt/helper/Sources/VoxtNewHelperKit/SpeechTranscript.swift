import Foundation

/// SpeechTranscript is a speech.recognize job's whole text. Some macOS
/// versions restart the recognizer's transcription after a pause: the
/// next result holds only what follows, the final one too, so what was
/// said before the pause would be lost. A result much shorter than the
/// last that does not start like it is such a restart: the last text is
/// kept, and the new one is appended to it.
public struct SpeechTranscript: Sendable {
    private var kept: [String] = []
    private var last = ""

    public init() {}

    /// The whole text so far.
    public var text: String { Self.join(kept + [last]) }

    /// The whole text with the recognizer's latest result (partial or final).
    public mutating func update(_ text: String) -> String {
        if Self.restarted(from: last, to: text) {
            kept.append(last)
        }
        last = text
        let whole = Self.join(kept + [text])
        // A final that repeats the kept text is the whole text already.
        return text.lowercased().hasPrefix(Self.join(kept).lowercased()) ? text : whole
    }

    // shortcut: a new utterance that starts with the last one's first
    // words reads as a revision (its start is lost), and a revision that
    // drops the start ("嗯我想" to "我想") as a restart (doubled); upgrade
    // to segment timestamps if either shows up.
    static func restarted(from old: String, to new: String) -> Bool {
        let a = Array(old.lowercased()), b = Array(new.lowercased())
        guard !a.isEmpty, !b.isEmpty, b.count < a.count else { return false }
        let common = zip(a, b).prefix { $0 == $1 }.count
        return common * 2 <= b.count
    }

    /// Joins texts with a space, but none next to Chinese, Japanese or
    /// Korean text or full-width punctuation.
    static func join(_ parts: [String]) -> String {
        var out = ""
        for p in parts where !p.isEmpty {
            if let l = out.unicodeScalars.last, let f = p.unicodeScalars.first,
               !wide(l), !wide(f), !CharacterSet.whitespaces.contains(l) {
                out += " "
            }
            out += p
        }
        return out
    }

    static func wide(_ s: Unicode.Scalar) -> Bool {
        switch s.value {
        case 0x2E80...0x9FFF, 0xAC00...0xD7AF, 0xF900...0xFAFF, 0xFF00...0xFFEF: return true
        default: return false
        }
    }
}
