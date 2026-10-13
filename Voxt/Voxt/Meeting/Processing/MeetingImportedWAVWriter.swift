import CoreMedia
import Foundation

nonisolated enum MeetingImportedWAVFormat {
    static let headerByteCount = 44
    static let riffSizeOverhead: Int64 = 36
    static let maximumDataByteCount = Int64(UInt32.max) - riffSizeOverhead

    static func validatedIncomingSampleCount(
        floatByteCount: Int,
        currentSampleCount: Int,
        maximumSampleCount: Int
    ) throws -> Int {
        guard floatByteCount > 0,
              floatByteCount <= MeetingFilePreparationLimits.maximumDecoderBufferBytes,
              floatByteCount.isMultiple(of: MemoryLayout<Float32>.size) else {
            throw MeetingImportedAudioFileError.unableToDecode
        }
        let incoming = floatByteCount / MemoryLayout<Float32>.size
        guard currentSampleCount >= 0, incoming <= maximumSampleCount,
              currentSampleCount <= maximumSampleCount - incoming else {
            throw MeetingImportedAudioFileError.fileTooLarge
        }
        _ = try dataByteCount(sampleCount: currentSampleCount + incoming)
        return incoming
    }

    static func dataByteCount(sampleCount: Int) throws -> UInt32 {
        let bytesPerSample = Int64(MemoryLayout<Int16>.size)
        guard sampleCount >= 0,
              Int64(sampleCount) <= maximumDataByteCount / bytesPerSample
        else {
            throw MeetingImportedAudioFileError.fileTooLarge
        }
        let dataByteCount = Int64(sampleCount) * bytesPerSample
        return UInt32(dataByteCount)
    }
}

nonisolated final class MeetingImportedWAVWriter {
    static let headerByteCount = MeetingImportedWAVFormat.headerByteCount

    private let handle: FileHandle
    private let sampleRate: Int
    private let maximumSampleCount: Int
    private(set) var sampleCount = 0
    private(set) var peakDecoderBufferByteCount = 0
    private var isFinished = false

    init(destinationURL: URL, sampleRate: Int, maximumSampleCount: Int) throws {
        self.handle = try FileHandle(forWritingTo: destinationURL)
        self.sampleRate = sampleRate
        self.maximumSampleCount = maximumSampleCount
        try handle.seek(toOffset: UInt64(Self.headerByteCount))
    }

    deinit { close() }

    func close() { try? handle.close() }

    func append(sampleBuffer: CMSampleBuffer) throws {
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
            throw MeetingImportedAudioFileError.unableToDecode
        }
        let byteCount = CMBlockBufferGetDataLength(blockBuffer)
        let incomingSampleCount = try MeetingImportedWAVFormat.validatedIncomingSampleCount(
            floatByteCount: byteCount, currentSampleCount: sampleCount, maximumSampleCount: maximumSampleCount
        )
        peakDecoderBufferByteCount = max(peakDecoderBufferByteCount, byteCount)

        var offset = 0
        while offset < byteCount {
            try Task.checkCancellation()
            let count = min(MeetingFilePreparationLimits.conversionBufferBytes, byteCount - offset)
            var floatData = Data(count: count)
            let copyStatus = floatData.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) in
                CMBlockBufferCopyDataBytes(blockBuffer, atOffset: offset, dataLength: count, destination: bytes.baseAddress!)
            }
            guard copyStatus == kCMBlockBufferNoErr else { throw MeetingImportedAudioFileError.unableToDecode }
            var pcmData = Data(count: count / 2)
            try floatData.withUnsafeBytes { (input: UnsafeRawBufferPointer) in
                try pcmData.withUnsafeMutableBytes { (output: UnsafeMutableRawBufferPointer) in
                    for index in 0..<(count / MemoryLayout<Float32>.size) {
                        let sample = input.loadUnaligned(fromByteOffset: index * 4, as: Float32.self)
                        guard sample.isFinite else { throw MeetingImportedAudioFileError.unableToDecode }
                        let clamped = max(-1, min(1, sample))
                        let pcm = UInt16(bitPattern: Int16((clamped * Float32(Int16.max)).rounded()))
                        output[index * 2] = UInt8(truncatingIfNeeded: pcm)
                        output[index * 2 + 1] = UInt8(truncatingIfNeeded: pcm >> 8)
                    }
                }
            }
            try handle.write(contentsOf: pcmData)
            offset += count
        }
        sampleCount += incomingSampleCount
    }

    func finish() throws {
        guard !isFinished else { return }
        isFinished = true
        let dataByteCount = try MeetingImportedWAVFormat.dataByteCount(sampleCount: sampleCount)
        let header = Self.wavHeader(sampleRate: sampleRate, dataByteCount: dataByteCount)
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: header)
        try handle.synchronize()
        try handle.close()
    }

    static func wavHeader(sampleRate: Int, dataByteCount: UInt32) -> Data {
        var data = Data()
        data.append("RIFF".data(using: .ascii)!)
        data.append(littleEndianData(36 + dataByteCount))
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        data.append(littleEndianData(UInt32(16)))
        data.append(littleEndianData(UInt16(1)))
        data.append(littleEndianData(UInt16(1)))
        data.append(littleEndianData(UInt32(sampleRate)))
        data.append(littleEndianData(UInt32(sampleRate * MemoryLayout<Int16>.size)))
        data.append(littleEndianData(UInt16(MemoryLayout<Int16>.size)))
        data.append(littleEndianData(UInt16(16)))
        data.append("data".data(using: .ascii)!)
        data.append(littleEndianData(dataByteCount))
        return data
    }

    private static func littleEndianData<Value: FixedWidthInteger>(_ value: Value) -> Data {
        var littleEndianValue = value.littleEndian
        return Data(bytes: &littleEndianValue, count: MemoryLayout<Value>.size)
    }
}
