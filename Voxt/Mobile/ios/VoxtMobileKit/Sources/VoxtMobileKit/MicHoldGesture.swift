import Foundation

/// The keyboard mic's touch: a tap dictates; holding fans out up to three
/// translation pills above the mic, and letting go on one translates into that
/// language. Letting go anywhere else after a hold does nothing. With no targets
/// one pill offers to choose them. Offsets are points from the mic's center,
/// y growing downward like UIKit.
public struct MicHoldGesture: Equatable, Sendable {
    public struct Offset: Equatable, Sendable {
        public var dx: Double
        public var dy: Double
        public init(dx: Double, dy: Double) { self.dx = dx; self.dy = dy }
        public static let zero = Offset(dx: 0, dy: 0)
    }

    public enum Pill: Equatable, Sendable {
        case language(String)
        case chooseLanguages
    }

    public enum Outcome: Equatable, Sendable {
        case dictate
        case translate(String)
        case chooseLanguages
        case nothing
    }

    /// For haptics: the pills just appeared, or the finger just moved onto a pill.
    public enum Feedback: Equatable, Sendable { case pillsShown, pillEntered }

    public static let holdDelay: TimeInterval = 0.3
    /// From the mic's center to each pill's center.
    public static let pillDistance = 82.0
    /// A pill counts as entered within this distance of its center.
    public static let pillRadius = 34.0

    public private(set) var pills: [Pill] = []
    public private(set) var entered: Int?
    private let targets: [String]
    private var downAt: Date?

    public init(targets: [String]) {
        self.targets = Array(targets.prefix(3))
    }

    /// Pill centers for `count` pills: the first straight above, then up-left, then up-right.
    public static func pillCenters(count: Int) -> [Offset] {
        let angles = [90.0, 135.0, 45.0].prefix(max(0, min(count, 3)))
        return angles.map { degrees in
            let radians = degrees * .pi / 180
            return Offset(dx: (pillDistance * cos(radians) * 1000).rounded() / 1000, dy: -pillDistance * sin(radians))
        }
    }

    public mutating func down(at date: Date) {
        downAt = date
        pills = []
        entered = nil
    }

    /// Call on every move and on a short timer while the finger is down.
    public mutating func update(now: Date, offset: Offset) -> [Feedback] {
        guard let downAt else { return [] }
        var feedback: [Feedback] = []
        if pills.isEmpty, now.timeIntervalSince(downAt) >= Self.holdDelay {
            pills = targets.isEmpty ? [.chooseLanguages] : targets.map(Pill.language)
            feedback.append(.pillsShown)
        }
        guard !pills.isEmpty else { return feedback }
        let hit = Self.pillCenters(count: pills.count).firstIndex { center in
            hypot(offset.dx - center.dx, offset.dy - center.dy) <= Self.pillRadius
        }
        if hit != entered {
            entered = hit
            if hit != nil { feedback.append(.pillEntered) }
        }
        return feedback
    }

    public mutating func up(now: Date, offset: Offset) -> Outcome {
        _ = update(now: now, offset: offset)
        defer { downAt = nil; pills = []; entered = nil }
        guard downAt != nil else { return .nothing }
        if pills.isEmpty { return .dictate }
        guard let entered else { return .nothing }
        switch pills[entered] {
        case let .language(code): return .translate(code)
        case .chooseLanguages: return .chooseLanguages
        }
    }
}
