import Foundation
import Testing
@testable import JapaneseInput

@Suite struct JapaneseFlickTests {
    @Test func keyRawValuesAreTheKeyIDs() {
        #expect(JapaneseFlickKey.a.rawValue == "a")
        #expect(JapaneseFlickKey.ka.rawValue == "ka")
        #expect(JapaneseFlickKey.sa.rawValue == "sa")
        #expect(JapaneseFlickKey.ta.rawValue == "ta")
        #expect(JapaneseFlickKey.na.rawValue == "na")
        #expect(JapaneseFlickKey.ha.rawValue == "ha")
        #expect(JapaneseFlickKey.ma.rawValue == "ma")
        #expect(JapaneseFlickKey.ya.rawValue == "ya")
        #expect(JapaneseFlickKey.ra.rawValue == "ra")
        #expect(JapaneseFlickKey.wa.rawValue == "wa")
        #expect(JapaneseFlickKey.punctuation.rawValue == "punctuation")
        #expect(JapaneseFlickKey.allCases.map(\.rawValue) == [
            "a", "ka", "sa", "ta", "na", "ha", "ma", "ya", "ra", "wa", "punctuation",
        ])
    }

    @Test func gojuonRowsUseCenterLeftUpRightDown() {
        let rows: [(JapaneseFlickKey, [String?])] = [
            (.a, ["あ", "い", "う", "え", "お"]),
            (.ka, ["か", "き", "く", "け", "こ"]),
            (.sa, ["さ", "し", "す", "せ", "そ"]),
            (.ta, ["た", "ち", "つ", "て", "と"]),
            (.na, ["な", "に", "ぬ", "ね", "の"]),
            (.ha, ["は", "ひ", "ふ", "へ", "ほ"]),
            (.ma, ["ま", "み", "む", "め", "も"]),
            (.ya, ["や", nil, "ゆ", nil, "よ"]),
            (.ra, ["ら", "り", "る", "れ", "ろ"]),
            (.wa, ["わ", "を", "ん", "ー", nil]),
            (.punctuation, ["、", "。", "？", "！", "…"]),
        ]
        let directions: [JapaneseFlickDirection] = [.center, .left, .up, .right, .down]
        for (key, expected) in rows {
            for (direction, character) in zip(directions, expected) {
                #expect(JapaneseFlick.character(for: key, direction: direction) == character)
            }
        }
        #expect(JapaneseFlick.character(for: .punctuation, direction: .up) == "\u{FF1F}")
        #expect(JapaneseFlick.character(for: .punctuation, direction: .right) == "\u{FF01}")
        #expect(JapaneseFlick.character(for: .punctuation, direction: .down) == "\u{2026}")
        #expect(JapaneseFlick.character(for: .punctuation, direction: .down)?.count == 1)
        #expect(JapaneseFlick.character(for: .wa, direction: .right) == "\u{30FC}")
    }

    @Test func yaAndWaOmitTheEmptyFlickCells() {
        #expect(JapaneseFlick.character(for: .ya, direction: .left) == nil)
        #expect(JapaneseFlick.character(for: .ya, direction: .right) == nil)
        #expect(JapaneseFlick.character(for: .wa, direction: .down) == nil)
    }

    @Test func dominantAxisChoosesTheFlickAndPositiveYIsDown() {
        #expect(JapaneseFlick.direction(dx: 30, dy: 10) == .right)
        #expect(JapaneseFlick.direction(dx: -30, dy: 10) == .left)
        #expect(JapaneseFlick.direction(dx: 10, dy: -30) == .up)
        #expect(JapaneseFlick.direction(dx: 10, dy: 30) == .down)
        #expect(JapaneseFlick.direction(dx: 40, dy: 19) == .right)
        #expect(JapaneseFlick.direction(dx: 19, dy: 40) == .down)
        #expect(JapaneseFlick.direction(dx: 30, dy: 30) == .center)
        #expect(JapaneseFlick.direction(dx: -18, dy: 18) == .center)
    }

    @Test func movementUnderTheThresholdStaysCenter() {
        #expect(JapaneseFlick.direction(dx: 0, dy: 0) == .center)
        #expect(JapaneseFlick.direction(dx: 17.9, dy: 0) == .center)
        #expect(JapaneseFlick.direction(dx: 0, dy: -17.9) == .center)
        #expect(JapaneseFlick.direction(dx: 18, dy: 0) == .right)
        #expect(JapaneseFlick.direction(dx: -18, dy: 0) == .left)
        #expect(JapaneseFlick.direction(dx: 0, dy: -18) == .up)
        #expect(JapaneseFlick.direction(dx: 0, dy: 18) == .down)
        #expect(JapaneseFlick.direction(dx: 5, dy: 1, threshold: 5) == .right)
        #expect(JapaneseFlick.direction(dx: 4.9, dy: 0, threshold: 5) == .center)
        #expect(JapaneseFlick.direction(dx: 0, dy: -9, threshold: 9) == .up)
    }

    @Test func nonFiniteMovementOrThresholdIsCenter() {
        #expect(JapaneseFlick.direction(dx: .nan, dy: 40) == .center)
        #expect(JapaneseFlick.direction(dx: 40, dy: .nan) == .center)
        #expect(JapaneseFlick.direction(dx: .infinity, dy: 0) == .center)
        #expect(JapaneseFlick.direction(dx: 0, dy: -.infinity) == .center)
        #expect(JapaneseFlick.direction(dx: .nan, dy: .nan) == .center)
        #expect(JapaneseFlick.direction(dx: 40, dy: 0, threshold: .nan) == .center)
        #expect(JapaneseFlick.direction(dx: 40, dy: 0, threshold: .infinity) == .center)
        #expect(JapaneseFlick.direction(dx: -40, dy: 0, threshold: -.infinity) == .center)
    }

    @Test func dakutenCyclesPlainVoicedAndSemivoiced() {
        let pairs = [
            ("か", "が"), ("が", "か"),
            ("き", "ぎ"), ("ぎ", "き"),
            ("く", "ぐ"), ("ぐ", "く"),
            ("け", "げ"), ("げ", "け"),
            ("こ", "ご"), ("ご", "こ"),
            ("さ", "ざ"), ("ざ", "さ"),
            ("し", "じ"), ("じ", "し"),
            ("す", "ず"), ("ず", "す"),
            ("せ", "ぜ"), ("ぜ", "せ"),
            ("そ", "ぞ"), ("ぞ", "そ"),
            ("た", "だ"), ("だ", "た"),
            ("ち", "ぢ"), ("ぢ", "ち"),
            ("て", "で"), ("で", "て"),
            ("と", "ど"), ("ど", "と"),
        ]
        let triples = [
            ("は", "ば"), ("ば", "ぱ"), ("ぱ", "は"),
            ("ひ", "び"), ("び", "ぴ"), ("ぴ", "ひ"),
            ("ふ", "ぶ"), ("ぶ", "ぷ"), ("ぷ", "ふ"),
            ("へ", "べ"), ("べ", "ぺ"), ("ぺ", "へ"),
            ("ほ", "ぼ"), ("ぼ", "ぽ"), ("ぽ", "ほ"),
        ]
        for (kana, next) in pairs + triples {
            #expect(JapaneseFlick.cycleModifier(kana) == next)
        }
    }

    @Test func smallKanaCyclesVowelsTsuAndYaRow() {
        let pairs = [
            ("あ", "ぁ"), ("ぁ", "あ"),
            ("い", "ぃ"), ("ぃ", "い"),
            ("え", "ぇ"), ("ぇ", "え"),
            ("お", "ぉ"), ("ぉ", "お"),
            ("や", "ゃ"), ("ゃ", "や"),
            ("ゆ", "ゅ"), ("ゅ", "ゆ"),
            ("よ", "ょ"), ("ょ", "よ"),
        ]
        for (kana, next) in pairs {
            #expect(JapaneseFlick.cycleModifier(kana) == next)
        }
    }

    @Test func tsuAndUCycleThroughVoicedThenSmall() {
        #expect(JapaneseFlick.cycleModifier("つ") == "づ")
        #expect(JapaneseFlick.cycleModifier("づ") == "っ")
        #expect(JapaneseFlick.cycleModifier("っ") == "つ")
        #expect(JapaneseFlick.cycleModifier("う") == "ゔ")
        #expect(JapaneseFlick.cycleModifier("ゔ") == "ぅ")
        #expect(JapaneseFlick.cycleModifier("ぅ") == "う")
    }

    @Test func unsupportedOrMultiCharacterModifierInputStaysUnchanged() {
        let unchanged = [
            "", " ", "ん", "を", "ー", "わ", "ゎ",
            "カ", "ガ", "、", "a", "はは", "つっ", "かい",
            "は\u{3099}",
        ]
        for kana in unchanged {
            #expect(JapaneseFlick.cycleModifier(kana) == kana)
        }
    }
}
