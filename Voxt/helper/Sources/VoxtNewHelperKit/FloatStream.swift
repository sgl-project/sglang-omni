import Foundation

/// Float32 samples from bytes that come in pieces of any size (a pipe):
/// a sample split between two pieces waits for the rest of its bytes.
public struct FloatStream: Sendable {
    private var pending = Data()

    public init() {}

    /// The whole samples so far, in order; a partial one's bytes stay.
    public mutating func push(_ bytes: Data) -> [Float] {
        pending.append(bytes)
        let n = pending.count / MemoryLayout<Float>.size
        guard n > 0 else { return [] }
        var samples = [Float](repeating: 0, count: n)
        _ = samples.withUnsafeMutableBufferPointer { pending.copyBytes(to: $0) } // n floats: pending holds them all
        pending = Data(pending.dropFirst(n * MemoryLayout<Float>.size))
        return samples
    }
}
