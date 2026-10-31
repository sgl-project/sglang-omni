import AVFoundation

/// Converts microphone buffers (any rate, any channel count) to 16 kHz mono Float32,
/// the format Voxt's ASR models expect.
public final class PCMDownsampler {
    public static let targetSampleRate: Double = 16_000

    public enum Error: Swift.Error, LocalizedError {
        case unsupportedFormat
        case conversionFailed(String)

        public var errorDescription: String? {
            switch self {
            case .unsupportedFormat: "This audio format is not supported."
            case let .conversionFailed(message): message
            }
        }
    }

    private let converter: AVAudioConverter
    private let outputFormat: AVAudioFormat

    public init(inputFormat: AVAudioFormat) throws {
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.targetSampleRate,
            channels: 1,
            interleaved: false
        ), let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw Error.unsupportedFormat
        }
        // Downmix all input channels instead of dropping all but the first.
        converter.downmix = true
        self.converter = converter
        self.outputFormat = outputFormat
    }

    public func convert(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        var delivered = false
        return try run { _, status in
            if delivered {
                status.pointee = .noDataNow
                return nil
            }
            delivered = true
            status.pointee = .haveData
            return buffer
        }
    }

    /// Finishes this recording and prepares the converter for the next one.
    public func flush() throws -> [Float] {
        // Without reset, convert after endOfStream returns zero samples.
        defer { converter.reset() }
        return try run { _, status in
            status.pointee = .endOfStream
            return nil
        }
    }

    private func run(_ input: @escaping AVAudioConverterInputBlock) throws -> [Float] {
        var samples: [Float] = []
        while true {
            guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 4_096) else {
                throw Error.conversionFailed("Could not allocate output buffer")
            }
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError, withInputFrom: input)
            if let conversionError { throw Error.conversionFailed(conversionError.localizedDescription) }
            if let data = output.floatChannelData?[0], output.frameLength > 0 {
                samples.append(contentsOf: UnsafeBufferPointer(start: data, count: Int(output.frameLength)))
            }
            switch status {
            case .haveData:
                continue
            case .inputRanDry, .endOfStream:
                return samples
            case .error:
                throw Error.conversionFailed("AVAudioConverter returned an error")
            @unknown default:
                return samples
            }
        }
    }
}
