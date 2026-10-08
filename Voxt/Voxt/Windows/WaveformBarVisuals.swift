// WaveformBarVisuals.swift
// Provides Waveform Bar Visuals for window and overlay UI.

import SwiftUI

enum WaveformBarVisuals {
    static let barGradient = LinearGradient(
        colors: [Color.white.opacity(0.98), Color.white.opacity(0.8)],
        startPoint: .top,
        endPoint: .bottom
    )

    static func emphasizedLevel(_ rawLevel: Float) -> CGFloat {
        let clamped = max(0, min(rawLevel, 1))
        let lifted = min(1.0, pow(Double(clamped), 0.9) * 1.02)
        return CGFloat(lifted)
    }

    static func barHeight(level: Float, minHeight: CGFloat, maxHeight: CGFloat) -> CGFloat {
        let emphasized = emphasizedLevel(level)
        return max(minHeight, min(maxHeight, minHeight + (maxHeight - minHeight) * emphasized))
    }

    static func glowOpacity(level: Float, base: Double = 0.08, gain: Double = 0.3, cap: Double = 0.36) -> Double {
        let emphasized = Double(emphasizedLevel(level))
        let activity = max(0, emphasized - 0.05)
        return min(cap, base + activity * gain)
    }
}
