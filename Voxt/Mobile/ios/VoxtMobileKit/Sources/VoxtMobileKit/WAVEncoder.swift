import Foundation

/// Encodes mono Float32 samples as a 16-bit PCM WAV file.
public enum WAVEncoder {
    public static func encode(samples: [Float], sampleRate: Int) -> Data {
        wav(pcm16: pcm16(samples), sampleRate: sampleRate)
    }

    /// The samples alone, as little-endian 16-bit PCM: what a recording appends while it grows.
    public static func pcm16(_ samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1, min(1, sample))
            withUnsafeBytes(of: Int16(clamped * Float(Int16.max)).littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    /// A WAV header in front of 16-bit mono PCM.
    public static func wav(pcm16 pcm: Data, sampleRate: Int) -> Data {
        let dataSize = UInt32(pcm.count)
        var data = Data(capacity: 44 + Int(dataSize))

        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36) + dataSize)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        append(UInt32(16))
        append(UInt16(1))
        append(UInt16(1))
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))
        append(UInt16(2))
        append(UInt16(16))
        data.append(contentsOf: Array("data".utf8))
        append(dataSize)
        data.append(pcm)
        return data
    }

    public static func duration(sampleCount: Int, sampleRate: Int) -> TimeInterval {
        TimeInterval(sampleCount) / TimeInterval(sampleRate)
    }
}
