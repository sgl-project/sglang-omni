import SwiftUI

/// Shared microphone states: idle capsule, RMS recording bars, timed processing bars,
/// and a failed circle. Processing ignores activation without dimming the control.
public struct DictationMicButton: View {
    public enum Size {
        case keyboard, app

        var capsule: CGSize { self == .keyboard ? CGSize(width: 148, height: 56) : CGSize(width: 216, height: 84) }
        var circle: CGFloat { self == .keyboard ? 72 : 120 }
        var glyph: CGFloat { self == .keyboard ? 24 : 38 }
        var halo: CGFloat { circle + (self == .keyboard ? 18 : 28) }
    }

    private let level: Float
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private let phase: DictationChannel.Phase
    private let sessionActive: Bool
    private let size: Size
    private let startLabel: String
    private let stopLabel: String
    private let action: () -> Void

    public init(
        phase: DictationChannel.Phase,
        sessionActive: Bool,
        level: Float = 0,
        size: Size,
        startLabel: String,
        stopLabel: String,
        action: @escaping () -> Void
    ) {
        self.level = level
        self.phase = phase
        self.sessionActive = sessionActive
        self.size = size
        self.startLabel = startLabel
        self.stopLabel = stopLabel
        self.action = action
    }

    private var isRound: Bool { phase == .recording || phase == .processing || phase == .failed }

    public var body: some View {
        Button { if phase != .processing { action() } } label: {
            ZStack {
                if phase == .recording {
                    Circle()
                        .fill(VoxtTheme.Colors.keyGray)
                        .frame(width: size.halo, height: size.halo)
                        .scaleEffect(1 + CGFloat(min(1, level * 5)) * 0.12)
                        .animation(.linear(duration: 0.1), value: level)
                }
                ZStack {
                    if phase == .processing || phase == .recording {
                        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: phase != .processing || reduceMotion)) { context in
                            HStack(spacing: size.glyph * 0.12) {
                                ForEach(0..<5) { index in
                                    Capsule()
                                        .fill(VoxtTheme.Colors.onInk)
                                        .frame(width: size.glyph * 0.12,
                                               height: barHeight(index, time: context.date.timeIntervalSinceReferenceDate))
                                }
                            }
                        }
                        .accessibilityHidden(true)
                    } else {
                        Image(systemName: glyphName)
                            .font(.system(size: size.glyph, weight: .semibold))
                            .foregroundStyle(VoxtTheme.Colors.onInk)
                            .contentTransition(.symbolEffect(.replace))
                    }
                }
                .frame(
                    width: isRound ? size.circle : size.capsule.width,
                    height: isRound ? size.circle : size.capsule.height
                )
                .background(VoxtTheme.Colors.ink)
                .clipShape(RoundedRectangle(
                    cornerRadius: (isRound ? size.circle : size.capsule.height) / 2,
                    style: .continuous
                ))
            }
            .animation(.spring(duration: 0.35, bounce: 0.15), value: isRound)
            .animation(.spring(duration: 0.3), value: phase)
        }
        .buttonStyle(.plain)
        .allowsHitTesting(phase != .processing)
        .accessibilityValue(phase == .recording ? String(format: "recording %.2f", level) : phase.rawValue)
        .accessibilityLabel(phase == .recording ? stopLabel : startLabel)
    }

    private func barHeight(_ index: Int, time: Double) -> CGFloat {
        let amplitude: Double
        if phase == .processing {
            amplitude = reduceMotion ? 0.5 : (sin(time * 4 + Double(index) * 0.8) + 1) / 2
        } else {
            amplitude = Double(min(1, level * 5)) * [0.5, 0.8, 1, 0.8, 0.5][index]
        }
        return size.glyph * (0.16 + 0.84 * amplitude)
    }

    private var glyphName: String {
        switch phase {
        case .recording, .processing: "waveform"
        case .failed: "exclamationmark"
        case .idle, .done: sessionActive ? "mic.fill" : "mic"
        }
    }
}
