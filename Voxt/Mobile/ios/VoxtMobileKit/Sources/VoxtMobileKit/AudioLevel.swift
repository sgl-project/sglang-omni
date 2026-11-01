import Foundation

public enum AudioLevel {
    /// Linear RMS of mono PCM; invalid samples are silence and clipping is bounded.
    public static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let energy = samples.reduce(0.0) { sum, value in
            sum + (value.isFinite ? Double(value) * Double(value) : 0)
        }
        return Float(min(1, sqrt(energy / Double(samples.count))))
    }
}
