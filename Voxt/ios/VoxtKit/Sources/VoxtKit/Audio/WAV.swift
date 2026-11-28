import Foundation

/// A finished recording: 16 kHz mono samples, written as a WAV file for
/// the speech engines.
public struct Recording: Sendable, Equatable {
    public static let sampleRate = 16_000

    public var samples: [Float]

    public init(samples: [Float]) {
        self.samples = samples
    }

    public var duration: TimeInterval { Double(samples.count) / Double(Self.sampleRate) }

    /// The recording as 16-bit PCM WAV.
    public func wav() -> Data {
        WAV.encode(samples, sampleRate: Self.sampleRate)
    }

    /// Writes the WAV into directory and returns its URL.
    public func write(to directory: URL, name: String = UUID().uuidString) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(name).appendingPathExtension("wav")
        try wav().write(to: url, options: .atomic)
        return url
    }
}

public enum WAV {
    /// Mono float samples (-1...1, clipped) as a 16-bit PCM WAV file.
    public static func encode(_ samples: [Float], sampleRate: Int) -> Data {
        let dataBytes = samples.count * 2
        var out = Data(capacity: 44 + dataBytes)
        out.append(contentsOf: Array("RIFF".utf8))
        out.append(le32(UInt32(36 + dataBytes)))
        out.append(contentsOf: Array("WAVEfmt ".utf8))
        out.append(le32(16))                       // fmt chunk size
        out.append(le16(1))                        // PCM
        out.append(le16(1))                        // mono
        out.append(le32(UInt32(sampleRate)))
        out.append(le32(UInt32(sampleRate * 2)))   // byte rate
        out.append(le16(2))                        // block align
        out.append(le16(16))                       // bits per sample
        out.append(contentsOf: Array("data".utf8))
        out.append(le32(UInt32(dataBytes)))
        let pcm = samples.map { Int16((max(-1, min(1, $0)) * Float(Int16.max)).rounded()).littleEndian }
        pcm.withUnsafeBytes { out.append(contentsOf: $0) }
        return out
    }

    private static func le16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    private static func le32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
}
