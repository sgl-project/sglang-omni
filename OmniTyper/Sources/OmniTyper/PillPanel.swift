// SPDX-License-Identifier: Apache-2.0
import SwiftUI

// Note (Yifei Leng): The compact popup says only "the microphone hears you" or "working on it".
// Text, buttons and modes stay in the detailed popup, so this one never competes with the field being typed into.
struct PillWaveform {
    static let barCount = 10
    // Note (Yifei Leng): Proportions follow a 10-bar capsule: bar width and gap are height / 14,
    // a silent bar is a dot of that width, and the loudest bar reaches 58% of the height.
    static let capsuleSize = CGSize(width: 84, height: 36)
    static let barWidth: CGFloat = 2.6
    static let maximumHeight: CGFloat = 21
    // Note (Yifei Leng): The bulge leans right of centre, the way a voice reads, not a symmetric meter.
    private static let profile: [Double] = [0.30, 0.38, 0.62, 0.86, 0.72, 1.0, 0.86, 0.62, 0.56, 0.32]

    static let greetingDuration: TimeInterval = 0.45

    private var levels = [Double](repeating: 0, count: barCount)
    private var gain = 0.15
    private var listeningSince: TimeInterval?
    private(set) var heights = [CGFloat](repeating: barWidth, count: barCount)

    static var idle: [CGFloat] { [CGFloat](repeating: barWidth, count: barCount) }

    // Note (Yifei Leng): The moment the microphone is live the bars rise once and settle, so "you can speak now"
    // is visible even with the sounds off and nothing said yet.
    mutating func startListening(at time: TimeInterval) {
        listeningSince = time
    }

    mutating func stopListening() {
        listeningSince = nil
    }

    // Note (Yifei Leng): Quiet speakers still fill the capsule: the loudest recent level sets the scale.
    // Bars rise fast and fall a little slower, so the shape stays full between syllables without lagging.
    mutating func advance(by dt: TimeInterval, level: Double, time: TimeInterval) {
        let meter = min(max(level, 0), 1)
        gain = max(0.15, meter, gain * pow(0.5, dt / 1.5))
        let drive = min(1, meter / gain)
        let sinceListening = listeningSince.map { time - $0 }
        let greeting = sinceListening.map { $0 < Self.greetingDuration ? sin(.pi * $0 / Self.greetingDuration) * 0.85 : 0 } ?? 0
        // Note (Yifei Leng): A live but silent microphone breathes; a waiting one holds still.
        let breath = sinceListening == nil ? 0 : 0.05 + 0.05 * sin(time * 5.5)
        step(by: dt, time: time) { index, sway in max(drive * Self.profile[index] * sway, greeting * Self.profile[index], breath) }
    }

    mutating func rest(by dt: TimeInterval, time: TimeInterval = 0) {
        listeningSince = nil
        step(by: dt, time: time) { _, _ in 0 }
    }

    private mutating func step(by dt: TimeInterval, time: TimeInterval, target: (Int, Double) -> Double) {
        let attack = 1 - pow(0.5, dt / 0.03)
        let release = 1 - pow(0.5, dt / 0.16)
        heights = (0..<Self.barCount).map { index in
            // Note (Yifei Leng): Each bar pulses about twice a second at its own phase, with a slower drift on top.
            let sway = (0.62 + 0.38 * sin(time * 13 + Double(index) * 2.1)) * (0.9 + 0.1 * sin(time * 2.9 + Double(index) * 0.9))
            let goal = target(index, sway)
            levels[index] += (goal - levels[index]) * (goal > levels[index] ? attack : release)
            return Self.barWidth + (Self.maximumHeight - Self.barWidth) * CGFloat(levels[index])
        }
    }
}

// Note (Yifei Leng): The window owner decides when the popup shows, so the capsule can animate in and out
// even when the popup is dismissed before the dictation is over.
@MainActor
final class PopupState: ObservableObject {
    @Published var visible = false
}

struct RecordingPopup: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: AppStore
    @ObservedObject var popup: PopupState

    var body: some View {
        if store.preferences.compactPopup == true { PillPanel(model: model, recorder: model.recorder, popup: popup) }
        else { VoicePanel(model: model, recorder: model.recorder, worker: model.worker) }
    }
}

struct PillCapsule: View {
    let heights: [CGFloat]
    let listening: Bool

    var body: some View {
        HStack(alignment: .center, spacing: PillWaveform.barWidth) {
            ForEach(0..<PillWaveform.barCount, id: \.self) { index in
                Capsule().fill(.white)
                    .frame(width: PillWaveform.barWidth, height: heights[index])
            }
        }
        // Note (Yifei Leng): Bars glow while listening; resting dots dim so "waiting" and "hearing" look different.
        .opacity(listening ? 1 : 0.55)
        .animation(.easeInOut(duration: 0.35), value: listening)
        .frame(width: PillWaveform.capsuleSize.width, height: PillWaveform.capsuleSize.height)
        .background(.black.opacity(0.9), in: Capsule())
        .overlay(Capsule().stroke(.white.opacity(0.16), lineWidth: 1))
        .shadow(color: .black.opacity(0.28), radius: 8, y: 3)
    }
}

// Note (Yifei Leng): Frame-to-frame smoothing needs memory, which a TimelineView closure cannot keep on its own.
@MainActor
private final class WaveformClock {
    var waveform = PillWaveform()
    var lastTime: TimeInterval?

    private var listening = false

    func heights(at time: TimeInterval, level: Double?) -> [CGFloat] {
        let dt = min(0.1, max(0, time - (lastTime ?? time)))
        lastTime = time
        if let level {
            if !listening { waveform.startListening(at: time); listening = true }
            waveform.advance(by: dt, level: level, time: time)
        } else {
            listening = false
            waveform.rest(by: dt, time: time)
        }
        return waveform.heights
    }
}

struct PillPanel: View {
    // Note (Yifei Leng): The window is larger than the capsule so the scale-in and the shadow are not clipped.
    static let windowSize = CGSize(width: 124, height: 64)

    @ObservedObject var model: AppModel
    @ObservedObject var recorder: AudioRecorder
    @ObservedObject var popup: PopupState
    @ViewState private var clock = WaveformClock()

    var body: some View {
        let active = popup.visible
        let listening = model.phase == .recording
        TimelineView(.animation(paused: !active)) { timeline in
            let time = timeline.date.timeIntervalSinceReferenceDate
            PillCapsule(heights: clock.heights(at: time, level: listening ? recorder.level : nil), listening: listening)
        }
        .contentShape(Capsule())
        .onTapGesture { if model.phase == .recording { model.finish() } }
        .scaleEffect(active ? 1 : 0.7)
        .opacity(active ? 1 : 0)
        .animation(.spring(response: 0.22, dampingFraction: 0.82), value: active)
        .frame(width: Self.windowSize.width, height: Self.windowSize.height)
        .accessibilityElement()
        .accessibilityLabel(model.phase == .recording ? L("panel.listening", model.mode.title)
                            : model.phase == .starting ? L("status.loadingModel") : model.liveStatus)
        .accessibilityAddTraits(.isButton)
    }
}
