import AVFoundation
import Testing
@testable import VoxtMobileKit

@Suite struct PCMDownsamplerTests {
    private func sineBuffer(sampleRate: Double, channels: AVAudioChannelCount, frequency: Double, seconds: Double) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: channels)!
        let frames = AVAudioFrameCount(sampleRate * seconds)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for channel in 0..<Int(channels) {
            let data = buffer.floatChannelData![channel]
            for i in 0..<Int(frames) {
                data[i] = Float(0.5 * sin(2 * .pi * frequency * Double(i) / sampleRate))
            }
        }
        return buffer
    }

    /// Counts sign changes; a sine of f Hz crosses zero 2f times per second.
    private func zeroCrossings(_ samples: [Float]) -> Int {
        zip(samples, samples.dropFirst()).filter { ($0 < 0) != ($1 < 0) }.count
    }

    @Test func converts48kStereoTo16kMonoKeepingPitch() throws {
        let converter = try PCMDownsampler(inputFormat: sineBuffer(sampleRate: 48_000, channels: 2, frequency: 440, seconds: 0).format)
        var output: [Float] = []
        // Feed in 10 chunks like an input tap does.
        for _ in 0..<10 {
            output += try converter.convert(sineBuffer(sampleRate: 48_000, channels: 2, frequency: 440, seconds: 0.1))
        }
        output += try converter.flush()

        #expect(abs(output.count - 16_000) < 200)
        let crossings = zeroCrossings(output)
        #expect(abs(crossings - 880) < 30)
        #expect(output.map(abs).max()! > 0.4)
    }

    @Test func passes16kMonoThrough() throws {
        let input = sineBuffer(sampleRate: 16_000, channels: 1, frequency: 200, seconds: 0.5)
        let converter = try PCMDownsampler(inputFormat: input.format)
        let output = try converter.convert(input) + converter.flush()
        #expect(abs(output.count - 8_000) < 100)
    }
    @Test func consecutiveRecordingsKeepAudioAfterFlush() throws {
        let first = sineBuffer(sampleRate: 48_000, channels: 1, frequency: 200, seconds: 0.5)
        let converter = try PCMDownsampler(inputFormat: first.format)
        let firstOutput = try converter.convert(first) + converter.flush()
        #expect(abs(firstOutput.count - 8_000) < 100)
        let second = sineBuffer(sampleRate: 48_000, channels: 1, frequency: 600, seconds: 0.5)
        let secondOutput = try converter.convert(second) + converter.flush()
        #expect(abs(secondOutput.count - 8_000) < 100)
        #expect(abs(zeroCrossings(secondOutput) - 600) < 20)
        #expect((secondOutput.map(abs).max() ?? 0) > 0.4)
    }
}
