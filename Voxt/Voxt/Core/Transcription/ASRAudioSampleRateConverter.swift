// MIT License
// Copyright (c) 2025 Prince Canuma
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import AVFoundation
import Foundation

nonisolated enum ASRAudioSampleRateConverter {
    enum ConversionError: LocalizedError {
        case resamplingFailed

        var errorDescription: String? { "Audio resampling failed." }
    }

    static func resample(
        samples: [Float],
        from sourceSampleRate: Int,
        to targetSampleRate: Int
    ) throws -> [Float] {
        if samples.isEmpty || sourceSampleRate == targetSampleRate {
            return samples
        }
        guard sourceSampleRate > 0, targetSampleRate > 0 else {
            throw ConversionError.resamplingFailed
        }
        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sourceSampleRate),
            channels: 1,
            interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw ConversionError.resamplingFailed
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw ConversionError.resamplingFailed
        }

        let inputFrameCount = AVAudioFrameCount(samples.count)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrameCount) else {
            throw ConversionError.resamplingFailed
        }
        inputBuffer.frameLength = inputFrameCount
        samples.withUnsafeBufferPointer { samplePointer in
            guard let sampleAddress = samplePointer.baseAddress else { return }
            memcpy(inputBuffer.floatChannelData![0], sampleAddress, samples.count * MemoryLayout<Float>.size)
        }

        let sampleRateRatio = Double(targetSampleRate) / Double(sourceSampleRate)
        let outputPaddingFrameCount = 64
        let estimatedOutputFrameCount = max(1, Int(ceil(Double(samples.count) * sampleRateRatio)) + outputPaddingFrameCount)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(estimatedOutputFrameCount)
        ) else {
            throw ConversionError.resamplingFailed
        }

        let provider = AudioConverterInputProvider(inputBuffer: inputBuffer)
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outputStatus in
            if provider.consumedInput {
                outputStatus.pointee = .endOfStream
                return nil
            } else {
                provider.consumedInput = true
                outputStatus.pointee = .haveData
                return provider.inputBuffer
            }
        }
        guard conversionError == nil,
              status == .haveData || status == .endOfStream || status == .inputRanDry else {
            throw ConversionError.resamplingFailed
        }

        let outputFrameCount = Int(outputBuffer.frameLength)
        guard let outputSamples = outputBuffer.floatChannelData?[0], outputFrameCount > 0 else {
            return []
        }
        return Array(UnsafeBufferPointer(start: outputSamples, count: outputFrameCount))
    }
}

private nonisolated final class AudioConverterInputProvider: @unchecked Sendable {
    let inputBuffer: AVAudioPCMBuffer
    var consumedInput = false

    init(inputBuffer: AVAudioPCMBuffer) {
        self.inputBuffer = inputBuffer
    }
}
