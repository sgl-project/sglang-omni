import Foundation
import Testing
@testable import VoxtMobileKit

/// Standard iOS key geometry, measured from a 3x screenshot of a 440 pt wide iPhone
/// keyboard: letter keys 36.7 pt wide
/// with 6.6 pt gaps and 6.7 pt side margins; a–l centered; shift and delete wider;
/// 123 and Return a quarter of the row each.
@Suite struct TextKeyboardGeometryTests {
    private let letters = TextKeyboardLayout.make(.englishQWERTY).rows(.letters).map { $0.map(\.kind) }
    private let width: Double = 440

    private func near(_ value: Double, _ expected: Double, _ tolerance: Double = 1) -> Bool {
        abs(value - expected) <= tolerance
    }

    @Test func topRowMatchesTheMeasuredKeys() {
        let rows = TextKeyboardGeometry.layout(letters, width: width)
        let top = rows[0]
        #expect(top.count == 10)
        #expect(near(top[0].x, 6.7))
        #expect(near(top[0].width, 36.7))
        #expect(near(top[1].x - (top[0].x + top[0].width), 6.6, 0.5))
        #expect(near(top[9].x + top[9].width, width - 6.7))
    }

    @Test func secondRowIsNineLetterKeysCentered() {
        let rows = TextKeyboardGeometry.layout(letters, width: width)
        let top = rows[0], middle = rows[1]
        #expect(middle.count == 9)
        #expect(middle.allSatisfy { near($0.width, top[0].width, 0.01) })
        let left = middle[0].x, right = width - (middle[8].x + middle[8].width)
        #expect(near(left, right, 0.01))
        // Half a key plus half a gap in from the top row (measured: a starts at 31 pt).
        #expect(near(left, 6.7 + (36.7 + 6.6) / 2))
    }

    @Test func thirdRowHasLetterWidthKeysAndWiderShiftAndDelete() {
        let rows = TextKeyboardGeometry.layout(letters, width: width)
        let top = rows[0], third = rows[2]
        #expect(letters[2].first == .shift && letters[2].last == .delete)
        let keys = third.dropFirst().dropLast()
        #expect(keys.allSatisfy { near($0.width, top[0].width, 0.01) })
        // Measured: shift 54 pt, delete 54 pt, z starts under s.
        #expect(near(third[0].width, 54))
        #expect(near(third[third.count - 1].width, 54))
        #expect(near(third[0].x, 6.7))
        #expect(near(third[1].x, rows[1][1].x, 2))
    }

    @Test func bottomRowGivesSpaceWhatTheQuarterKeysLeave() {
        let rows = TextKeyboardGeometry.layout(letters, width: width)
        let bottom = rows[3]
        #expect(letters[3] == [.numbers, .space, .return])
        // Measured: 123 and Return 106.7 pt each, space 200 pt.
        #expect(near(bottom[0].width, 106.7))
        #expect(near(bottom[2].width, 106.7))
        #expect(near(bottom[1].width, 200, 1.5))
        #expect(near(bottom[2].x + bottom[2].width, width - 6.7))
    }

    @Test func aShortSymbolRowStretchesItsKeysInsteadOfItsSideKeys() {
        // Numbers plane, third row: ABC, five symbols, delete.
        let numbers = TextKeyboardLayout.make(.englishQWERTY).rows(.numbers).map { $0.map(\.kind) }
        let rows = TextKeyboardGeometry.layout(numbers, width: width)
        let third = rows[2]
        #expect(third.count == 7)
        #expect(third[0].width <= 56)
        #expect(third[1].width > rows[0][0].width * 1.3)
        #expect(near(third[6].x + third[6].width, width - 6.7))
    }

    @Test func everyRowStaysInsideTheKeyboard() {
        for layout in TextKeyboardLayoutID.allCases {
            for plane in [TextKeyboardPlane.letters, .numbers] {
                let kinds = TextKeyboardLayout.make(layout).rows(plane).map { $0.map(\.kind) }
                for row in TextKeyboardGeometry.layout(kinds, width: 375) where !row.isEmpty {
                    #expect(row[0].x >= 0)
                    #expect(row[row.count - 1].x + row[row.count - 1].width <= 375 + 0.01)
                    for (a, b) in zip(row, row.dropFirst()) { #expect(b.x >= a.x + a.width) }
                }
            }
        }
    }
}
