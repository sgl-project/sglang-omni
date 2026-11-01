import Foundation

/// Where each key of a text keyboard sits, in standard iOS QWERTY proportions
/// (measured on a 440 pt wide iPhone). Every letter key has the
/// same width: the widest letter row fills the keyboard, shorter rows are
/// centered (a–l), shift and delete take the space beside z–m, and in the row
/// with the space bar the other keys get a quarter of the row each.
public enum TextKeyboardGeometry {
    public struct Frame: Equatable, Sendable {
        public var x: Double
        public var width: Double
    }

    public static let sideMargin = 20.0 / 3
    public static let gap = 6.6
    public static let keyHeight = 42.0
    public static let rowGap = 12.0
    public static let cornerRadius = 8.5
    /// Shift and delete beside seven letters: 54 pt next to 36.7 pt letters.
    static let sideKeyRatio = 54.0 / 36.7

    public static func layout(_ rows: [[TextKeyboardKey.Kind]], width: Double) -> [[Frame]] {
        let available = width - 2 * sideMargin
        let columns = max(1, rows.filter { !$0.contains(.space) && !isControlRow($0) }.map(\.count).max() ?? 10)
        let key = (available - Double(columns - 1) * gap) / Double(columns)
        return rows.map { row in
            if row.contains(.space) { return bottomRow(row, available: available) }
            if isControlRow(row) { return controlRow(row, key: key, available: available) }
            return centered(Array(repeating: key, count: row.count), available: available)
        }
    }

    private static func isControlRow(_ row: [TextKeyboardKey.Kind]) -> Bool {
        guard let first = row.first, let last = row.last, row.count > 2 else { return false }
        return first != .character && last != .character
    }

    private static func bottomRow(_ row: [TextKeyboardKey.Kind], available: Double) -> [Frame] {
        let others = row.filter { $0 != .space }.count
        let side = others == 0 ? 0 : available * 0.5 / Double(others)
        let spaces = Double(row.count - others)
        let space = (available - Double(others) * side - Double(row.count - 1) * gap) / max(1, spaces)
        return place(row.map { $0 == .space ? space : side }, gaps: Array(repeating: gap, count: max(0, row.count - 1)))
    }

    /// Shift, letters, delete (or ABC, symbols, delete on the numbers plane).
    private static func controlRow(_ row: [TextKeyboardKey.Kind], key: Double, available: Double) -> [Frame] {
        let inner = row.count - 2
        let block = Double(inner) * key + Double(inner - 1) * gap
        let beside = (available - block) / 2
        var side = min(key * sideKeyRatio, beside - gap)
        var letter = key
        // Few keys between (five symbols): the symbols grow, the side keys don't.
        if beside - side > 2 * gap {
            side = key * sideKeyRatio
            letter = (available - 2 * side - 2 * gap - Double(inner - 1) * gap) / Double(inner)
        }
        let edge = (available - Double(inner) * letter - Double(inner - 1) * gap) / 2 - side
        let widths = [side] + Array(repeating: letter, count: inner) + [side]
        let gaps = [edge] + Array(repeating: gap, count: max(0, inner - 1)) + [edge]
        return place(widths, gaps: gaps)
    }

    private static func centered(_ widths: [Double], available: Double) -> [Frame] {
        let used = widths.reduce(0, +) + Double(max(0, widths.count - 1)) * gap
        var frames = place(widths, gaps: Array(repeating: gap, count: max(0, widths.count - 1)))
        let inset = (available - used) / 2
        for index in frames.indices { frames[index].x += inset }
        return frames
    }

    private static func place(_ widths: [Double], gaps: [Double]) -> [Frame] {
        var x = sideMargin
        var frames: [Frame] = []
        for (index, width) in widths.enumerated() {
            frames.append(Frame(x: x, width: width))
            x += width + (index < gaps.count ? gaps[index] : 0)
        }
        return frames
    }
}
