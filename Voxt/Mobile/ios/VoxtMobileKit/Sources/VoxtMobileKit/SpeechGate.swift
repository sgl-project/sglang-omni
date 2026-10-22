import Foundation

/// Decides on the phone whether a recording holds any speech, so an empty tap costs no
/// upload or model pass. Thresholds follow OpenWhispr's keyboard gate (MIT;
/// openwhispr-mobile AppGroupStorageModule.swift:435-526): 20 ms RMS frames, a noise
/// floor at the 20th percentile, and a speech threshold of floor + 10 dB clamped to
/// [-48, -30] dBFS. It errs toward "speech": a loud click still uploads.
public enum SpeechGate {
    /// Loudness numbers for the stage timings: peak level and time above the speech threshold.
    public struct Measurement: Equatable, Sendable {
        public var peakDb: Int
        public var speechMs: Int
    }

    public static func measure(_ samples: [Float], sampleRate: Double = 16_000) -> Measurement {
        let analysis = analyze(samples, sampleRate: sampleRate)
        return Measurement(peakDb: Int(analysis.peak.rounded()), speechMs: analysis.speechMs)
    }

    public static func containsSpeech(_ samples: [Float], sampleRate: Double = 16_000) -> Bool {
        let frameLength = max(1, Int(sampleRate * 0.02))
        guard Double(samples.count) / sampleRate >= 0.25, samples.count >= frameLength else { return false }
        let analysis = analyze(samples, sampleRate: sampleRate)
        guard analysis.peak >= -48 else { return false }
        return !(analysis.speechMs < 160 && analysis.ratio < 0.08 && analysis.peak < -34 && analysis.average < -45)
    }

    private static func decibels(_ value: Float) -> Float { 20 * log10(max(value, 1e-9)) }

    private static func analyze(_ samples: [Float], sampleRate: Double) -> (peak: Float, speechMs: Int, ratio: Float, average: Float) {
        let frameLength = max(1, Int(sampleRate * 0.02))
        let peak = decibels(samples.reduce(0) { max($0, abs($1)) })
        var frames: [Float] = []
        var start = 0
        while start + frameLength <= samples.count {
            let frame = samples[start..<start + frameLength]
            frames.append(decibels(sqrt(frame.reduce(0) { $0 + $1 * $1 } / Float(frameLength))))
            start += frameLength
        }
        guard !frames.isEmpty else { return (peak, 0, 0, peak) }
        let floor = frames.sorted()[frames.count / 5]
        let threshold = max(-48, min(-30, floor + 10))
        let speechFrames = frames.filter { $0 >= threshold }.count
        let average = decibels(sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(max(1, samples.count))))
        return (peak, speechFrames * 20, Float(speechFrames) / Float(frames.count), average)
    }
}
