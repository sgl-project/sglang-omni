import Foundation
import XCTest
@testable import Voxt

final class ASRAudioSampleRateConverterTests: XCTestCase {
    func testEmptyAndEqualRateInputsPreserveEarlyReturnBeforeValidation() throws {
        for (sourceSampleRate, targetSampleRate) in [(48_000, 16_000), (0, -1), (-1, 0)] {
            XCTAssertEqual(
                try ASRAudioSampleRateConverter.resample(samples: [], from: sourceSampleRate, to: targetSampleRate),
                []
            )
        }
        let samples: [Float] = [0, -0.0, .nan, .infinity, -.infinity, 0.5]
        for sampleRate in [16_000, 0, -1] {
            let output = try ASRAudioSampleRateConverter.resample(samples: samples, from: sampleRate, to: sampleRate)
            XCTAssertEqual(output.map(\.bitPattern), samples.map(\.bitPattern))
            samples.withUnsafeBufferPointer { inputPointer in
                output.withUnsafeBufferPointer { outputPointer in
                    XCTAssertEqual(inputPointer.baseAddress, outputPointer.baseAddress)
                }
            }
        }
    }

    func testInvalidUnequalRatesKeepFailureDescription() {
        for (sourceSampleRate, targetSampleRate) in [(0, 16_000), (-1, 16_000), (16_000, 0), (16_000, -1)] {
            XCTAssertThrowsError(
                try ASRAudioSampleRateConverter.resample(samples: [0, 1], from: sourceSampleRate, to: targetSampleRate)
            ) { error in
                XCTAssertEqual(error as? ASRAudioSampleRateConverter.ConversionError, .resamplingFailed)
                XCTAssertEqual(error.localizedDescription, "Audio resampling failed.")
            }
        }
    }

    func testConversionMatchesAudioCoreAcrossRatesAndFrameBoundaries() throws {
        for (sourceSampleRate, targetSampleRate) in [
            (48_000, 16_000), (44_100, 16_000), (22_050, 16_000), (8_000, 16_000), (16_000, 48_000)
        ] {
            for sampleCount in [1, 2, 63, 64, 127, 1_024, 4_097] {
                let samples: [Float] = (0..<sampleCount).map { sampleIndex in
                    let timeSeconds = Double(sampleIndex) / Double(sourceSampleRate)
                    return Float(0.5 * sin(2 * .pi * 440 * timeSeconds) + 0.2 * cos(2 * .pi * 3_100 * timeSeconds))
                }
                let expected = try baseline(signal: .tone, sourceSampleRate: sourceSampleRate,
                                            targetSampleRate: targetSampleRate, inputSampleCount: sampleCount)
                let actual = try ASRAudioSampleRateConverter.resample(
                    samples: samples, from: sourceSampleRate, to: targetSampleRate
                )
                XCTAssertEqual(actual.map(\.bitPattern), expected,
                               "\(sampleCount) frames, \(sourceSampleRate) -> \(targetSampleRate)")
            }
        }
    }

    func testSilenceConstantAndImpulseMatchAudioCore() throws {
        var impulseSamples = Array(repeating: Float.zero, count: 4_099)
        impulseSamples[2_049] = 1
        let signals: [(Signal, [Float])] = [(.silence, Array(repeating: Float.zero, count: 4_099)),
                                            (.constant, Array(repeating: Float(0.25), count: 4_099)), (.impulse, impulseSamples)]
        for (signal, samples) in signals {
            let expected = try baseline(signal: signal, sourceSampleRate: 44_100, targetSampleRate: 16_000, inputSampleCount: samples.count)
            let actual = try ASRAudioSampleRateConverter.resample(samples: samples, from: 44_100, to: 16_000)
            XCTAssertEqual(actual.map(\.bitPattern), expected)
        }
    }

    func testDetachedPreparationPreservesOneHertzToleranceAndSharedStorage() throws {
        let samples: [Float] = [0, -0.0, .nan, .infinity, 0.5]
        for sampleRate in [15_999.0, 15_999.5, 16_000.0, 16_000.5, 16_001.0] {
            let output = try MLXTranscriber.prepareInputSamplesDetached(samples, sampleRate: sampleRate, targetSampleRate: 16_000)
            XCTAssertEqual(output.map(\.bitPattern), samples.map(\.bitPattern))
            samples.withUnsafeBufferPointer { inputPointer in
                output.withUnsafeBufferPointer { outputPointer in
                    XCTAssertEqual(inputPointer.baseAddress, outputPointer.baseAddress)
                }
            }
        }
    }

    func testDetachedPreparationKeepsIntegerRateTruncation() throws {
        let samples = (0..<4_097).map { Float($0 % 17) / 17 }
        for sampleRate in [44_100.9, 48_000.75, 16_001.01, 15_998.99] {
            let expected = try baseline(signal: .modulo17, sourceSampleRate: Int(sampleRate),
                                        targetSampleRate: 16_000, inputSampleCount: samples.count)
            let actual = try MLXTranscriber.prepareInputSamplesDetached(samples, sampleRate: sampleRate, targetSampleRate: 16_000)
            XCTAssertEqual(actual.map(\.bitPattern), expected)
        }
    }

    private enum Signal: UInt32 {
        case tone = 1, silence, constant, impulse, modulo17
    }

    private enum BaselineFailure: Error { case invalidFixture }

    private func baseline(signal: Signal, sourceSampleRate: Int, targetSampleRate: Int, inputSampleCount: Int) throws -> [UInt32] {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let bytes = try Data(contentsOf: directory.appendingPathComponent("Fixtures/Audio/audio-core-resampler/baseline.bin"))
        guard bytes.prefix(8) == Data("VOXTAC01".utf8) else { throw BaselineFailure.invalidFixture }
        var byteOffset = 8
        func readInteger() throws -> UInt32 {
            guard byteOffset + 4 <= bytes.count else { throw BaselineFailure.invalidFixture }
            defer { byteOffset += 4 }
            return bytes.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self)) }
        }
        guard try readInteger() == 42 else { throw BaselineFailure.invalidFixture }
        var matchingSamples: [UInt32]?
        for _ in 0..<42 {
            let capturedSignal = try readInteger()
            let capturedSourceSampleRate = try readInteger()
            let capturedTargetSampleRate = try readInteger()
            let capturedInputSampleCount = try readInteger()
            let capturedOutputSampleCount = try readInteger()
            guard Signal(rawValue: capturedSignal) != nil, capturedInputSampleCount <= 4_099,
                  capturedOutputSampleCount <= 16_000 else { throw BaselineFailure.invalidFixture }
            let outputSampleBits = try (0..<capturedOutputSampleCount).map { _ in try readInteger() }
            if capturedSignal == signal.rawValue, capturedSourceSampleRate == sourceSampleRate,
               capturedTargetSampleRate == targetSampleRate, capturedInputSampleCount == inputSampleCount {
                guard matchingSamples == nil else { throw BaselineFailure.invalidFixture }
                matchingSamples = outputSampleBits
            }
        }
        guard byteOffset == bytes.count, let matchingSamples else { throw BaselineFailure.invalidFixture }
        return matchingSamples
    }
}
