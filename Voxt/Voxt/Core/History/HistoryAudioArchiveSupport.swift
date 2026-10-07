// HistoryAudioArchiveSupport.swift
// Provides History Audio Archive Support for history storage and display support.

import Foundation

enum HistoryAudioArchiveSupport {
    nonisolated static let targetSampleRate: Double = 16_000
    nonisolated static let rewriteJoinGapSeconds: Double = 0.3

    nonisolated static func exportWAV(
        samples: [Float],
        sampleRate: Double,
        to destinationURL: URL
    ) throws -> Bool {
        guard !samples.isEmpty else { return false }
        let preparedSamples = resample(samples: samples, from: sampleRate, to: targetSampleRate)
        guard !preparedSamples.isEmpty else { return false }

        let data = wavData(for: preparedSamples, sampleRate: Int(targetSampleRate))
        let directory = destinationURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: destinationURL, options: .atomic)
        return true
    }

    nonisolated static func mergedRewriteArchive(
        existingArchiveURL: URL?,
        appendedArchiveURL: URL
    ) throws -> URL {
        let appendedSamples = try readWAVSamples(from: appendedArchiveURL)
        guard !appendedSamples.isEmpty else {
            throw NSError(
                domain: "Voxt.HistoryAudio",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "The appended rewrite audio archive was empty."]
            )
        }

        var mergedSamples: [Float] = []
        if let existingArchiveURL {
            mergedSamples = try readWAVSamples(from: existingArchiveURL)
            if !mergedSamples.isEmpty {
                mergedSamples.append(contentsOf: silenceSamples(durationSeconds: rewriteJoinGapSeconds))
            }
        }
        mergedSamples.append(contentsOf: appendedSamples)

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-history-rewrite-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        _ = try exportWAV(samples: mergedSamples, sampleRate: targetSampleRate, to: tempURL)
        return tempURL
    }

    nonisolated static func readWAVSamples(from fileURL: URL) throws -> [Float] {
        let data = try Data(contentsOf: fileURL)
        guard data.count >= 44 else {
            throw NSError(
                domain: "Voxt.HistoryAudio",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "The WAV file was too small to parse."]
            )
        }

        guard String(data: data[0..<4], encoding: .ascii) == "RIFF",
              String(data: data[8..<12], encoding: .ascii) == "WAVE" else {
            throw NSError(
                domain: "Voxt.HistoryAudio",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "The audio archive was not a WAV file."]
            )
        }

        var offset = 12
        var bitsPerSample: UInt16?
        var channelCount: UInt16?
        var dataChunkRange: Range<Int>?

        while offset + 8 <= data.count {
            let chunkIDData = data[offset..<(offset + 4)]
            guard let chunkID = String(data: chunkIDData, encoding: .ascii) else { break }
            let chunkSize = littleEndianUInt32(from: data, at: offset + 4)
            let chunkBodyStart = offset + 8
            let chunkBodyEnd = chunkBodyStart + Int(chunkSize)
            guard chunkBodyEnd <= data.count else { break }

            if chunkID == "fmt ", chunkSize >= 16 {
                channelCount = littleEndianUInt16(from: data, at: chunkBodyStart + 2)
                bitsPerSample = littleEndianUInt16(from: data, at: chunkBodyStart + 14)
            } else if chunkID == "data" {
                dataChunkRange = chunkBodyStart..<chunkBodyEnd
            }

            let paddedSize = Int(chunkSize) + (Int(chunkSize) % 2)
            offset = chunkBodyStart + paddedSize
        }

        guard let resolvedChannelCount = channelCount,
              resolvedChannelCount == 1,
              let resolvedBitsPerSample = bitsPerSample,
              resolvedBitsPerSample == 16,
              let dataChunkRange
        else {
            throw NSError(
                domain: "Voxt.HistoryAudio",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Only mono 16-bit WAV archives are supported."]
            )
        }

        let audioData = data[dataChunkRange]
        return audioData.withUnsafeBytes { rawBuffer in
            let int16Samples = rawBuffer.bindMemory(to: Int16.self)
            return int16Samples.map { Float($0) / Float(Int16.max) }
        }
    }

    nonisolated static func silenceSamples(durationSeconds: Double) -> [Float] {
        let count = max(Int((durationSeconds * targetSampleRate).rounded()), 0)
        return [Float](repeating: 0, count: count)
    }

    nonisolated static func temporaryArchiveURL(prefix: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)")
            .appendingPathExtension("wav")
    }

    private nonisolated static func wavData(for samples: [Float], sampleRate: Int) -> Data {
        let channelCount: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate) * UInt32(channelCount) * UInt32(bitsPerSample / 8)
        let blockAlign = channelCount * (bitsPerSample / 8)

        var pcmData = Data(capacity: samples.count * MemoryLayout<Int16>.size)
        for sample in samples {
            let clamped = max(-1, min(1, sample))
            var value = Int16((clamped * Float(Int16.max)).rounded())
            pcmData.append(Data(bytes: &value, count: MemoryLayout<Int16>.size))
        }

        let riffChunkSize = UInt32(36 + pcmData.count)
        let dataChunkSize = UInt32(pcmData.count)

        var data = Data()
        data.append("RIFF".data(using: .ascii)!)
        data.append(bytes(of: riffChunkSize))
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        data.append(bytes(of: UInt32(16)))
        data.append(bytes(of: UInt16(1)))
        data.append(bytes(of: channelCount))
        data.append(bytes(of: UInt32(sampleRate)))
        data.append(bytes(of: byteRate))
        data.append(bytes(of: blockAlign))
        data.append(bytes(of: bitsPerSample))
        data.append("data".data(using: .ascii)!)
        data.append(bytes(of: dataChunkSize))
        data.append(pcmData)
        return data
    }

    private nonisolated static func bytes<T>(of value: T) -> Data {
        var mutableValue = value
        return withUnsafeBytes(of: &mutableValue) { Data($0) }
    }

    private nonisolated static func resample(samples: [Float], from inputRate: Double, to outputRate: Double) -> [Float] {
        guard !samples.isEmpty, inputRate > 0, outputRate > 0 else { return samples }
        if abs(inputRate - outputRate) <= 1 {
            return samples
        }

        let ratio = outputRate / inputRate
        let outputCount = max(Int(Double(samples.count) * ratio), 1)
        var output = [Float](repeating: 0, count: outputCount)

        for index in 0..<outputCount {
            let position = Double(index) / ratio
            let lowerIndex = Int(position)
            let upperIndex = min(lowerIndex + 1, samples.count - 1)
            let fraction = Float(position - Double(lowerIndex))
            let lower = samples[min(lowerIndex, samples.count - 1)]
            let upper = samples[upperIndex]
            output[index] = lower + (upper - lower) * fraction
        }

        return output
    }

    private nonisolated static func littleEndianUInt16(from data: Data, at offset: Int) -> UInt16 {
        data.subdata(in: offset..<(offset + 2)).withUnsafeBytes { rawBuffer in
            rawBuffer.load(as: UInt16.self).littleEndian
        }
    }

    private nonisolated static func littleEndianUInt32(from data: Data, at offset: Int) -> UInt32 {
        data.subdata(in: offset..<(offset + 4)).withUnsafeBytes { rawBuffer in
            rawBuffer.load(as: UInt32.self).littleEndian
        }
    }
}
