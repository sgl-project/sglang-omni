import Foundation

/// Two complete short taps of the same Option key. No key text is stored.
public struct DoubleOptionGesture {
    private var pressed: (key: UInt16, time: Double)?
    private var released: (key: UInt16, time: Double)?
    private let maximumHold: Double
    private let maximumGap: Double

    public init(maximumHold: Double = 0.25, maximumGap: Double = 0.35) {
        self.maximumHold = maximumHold
        self.maximumGap = maximumGap
    }

    public mutating func reset() { pressed = nil; released = nil }

    public mutating func option(key: UInt16, down: Bool, otherModifiers: Bool, time: Double) -> Bool {
        guard [58, 61].contains(key), !otherModifiers, time.isFinite else { reset(); return false }
        if down {
            guard pressed == nil else { reset(); return false }
            if let released, released.key != key || time < released.time || time - released.time > maximumGap {
                self.released = nil
            }
            pressed = (key, time)
            return false
        }
        guard let pressed, pressed.key == key, time >= pressed.time,
              time - pressed.time <= maximumHold else { reset(); return false }
        self.pressed = nil
        if let released, released.key == key, pressed.time >= released.time,
           pressed.time - released.time <= maximumGap {
            reset()
            return true
        }
        released = (key, time)
        return false
    }
}
