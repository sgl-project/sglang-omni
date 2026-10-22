import Foundation

public enum JapaneseFlickDirection: Equatable, Sendable {
    case center, left, up, right, down
}

public enum JapaneseFlickKey: String, CaseIterable, Equatable, Sendable {
    case a, ka, sa, ta, na, ha, ma, ya, ra, wa, punctuation
}

/// Hiragana flick cells. This does not convert kana to kanji.
public enum JapaneseFlick {
    /// Positive x points right and positive y points down. A tie, or any non-finite value, is center.
    public static func direction(dx: Double, dy: Double, threshold: Double = 18) -> JapaneseFlickDirection {
        guard dx.isFinite, dy.isFinite, threshold.isFinite else { return .center }
        let horizontal = abs(dx)
        let vertical = abs(dy)
        guard max(horizontal, vertical) >= threshold else { return .center }
        if horizontal > vertical { return dx > 0 ? .right : .left }
        if vertical > horizontal { return dy > 0 ? .down : .up }
        return .center
    }

    public static func character(for key: JapaneseFlickKey, direction: JapaneseFlickDirection) -> String? {
        switch key {
        case .a:
            switch direction {
            case .center: "あ"
            case .left: "い"
            case .up: "う"
            case .right: "え"
            case .down: "お"
            }
        case .ka:
            switch direction {
            case .center: "か"
            case .left: "き"
            case .up: "く"
            case .right: "け"
            case .down: "こ"
            }
        case .sa:
            switch direction {
            case .center: "さ"
            case .left: "し"
            case .up: "す"
            case .right: "せ"
            case .down: "そ"
            }
        case .ta:
            switch direction {
            case .center: "た"
            case .left: "ち"
            case .up: "つ"
            case .right: "て"
            case .down: "と"
            }
        case .na:
            switch direction {
            case .center: "な"
            case .left: "に"
            case .up: "ぬ"
            case .right: "ね"
            case .down: "の"
            }
        case .ha:
            switch direction {
            case .center: "は"
            case .left: "ひ"
            case .up: "ふ"
            case .right: "へ"
            case .down: "ほ"
            }
        case .ma:
            switch direction {
            case .center: "ま"
            case .left: "み"
            case .up: "む"
            case .right: "め"
            case .down: "も"
            }
        case .ya:
            switch direction {
            case .center: "や"
            case .left, .right: nil
            case .up: "ゆ"
            case .down: "よ"
            }
        case .ra:
            switch direction {
            case .center: "ら"
            case .left: "り"
            case .up: "る"
            case .right: "れ"
            case .down: "ろ"
            }
        case .wa:
            switch direction {
            case .center: "わ"
            case .left: "を"
            case .up: "ん"
            case .right: "ー"
            case .down: nil
            }
        case .punctuation:
            switch direction {
            case .center: "、"
            case .left: "。"
            case .up: "？"
            case .right: "！"
            case .down: "…"
            }
        }
    }

    /// Cycles one precomposed hiragana through voiced or small forms. Anything else is returned unchanged.
    public static func cycleModifier(_ kana: String) -> String {
        guard kana.unicodeScalars.count == 1 else { return kana }
        return nextModifier[kana] ?? kana
    }

    private static let nextModifier: [String: String] = {
        let cycles = [
            ["か", "が"], ["き", "ぎ"], ["く", "ぐ"], ["け", "げ"], ["こ", "ご"],
            ["さ", "ざ"], ["し", "じ"], ["す", "ず"], ["せ", "ぜ"], ["そ", "ぞ"],
            ["た", "だ"], ["ち", "ぢ"], ["て", "で"], ["と", "ど"],
            ["は", "ば", "ぱ"], ["ひ", "び", "ぴ"], ["ふ", "ぶ", "ぷ"], ["へ", "べ", "ぺ"], ["ほ", "ぼ", "ぽ"],
            ["あ", "ぁ"], ["い", "ぃ"], ["う", "ゔ", "ぅ"], ["え", "ぇ"], ["お", "ぉ"],
            ["つ", "づ", "っ"], ["や", "ゃ"], ["ゆ", "ゅ"], ["よ", "ょ"],
        ]
        var next: [String: String] = [:]
        for cycle in cycles {
            for (index, kana) in cycle.enumerated() {
                next[kana] = cycle[(index + 1) % cycle.count]
            }
        }
        return next
    }()
}
