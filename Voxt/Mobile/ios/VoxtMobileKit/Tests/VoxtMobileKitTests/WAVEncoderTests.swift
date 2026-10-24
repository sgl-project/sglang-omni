import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct WAVEncoderTests {
    private func readUInt32(_ data: Data, _ offset: Int) -> UInt32 {
        data.subdata(in: offset..<offset + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian
    }

    private func readUInt16(_ data: Data, _ offset: Int) -> UInt16 {
        data.subdata(in: offset..<offset + 2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }.littleEndian
    }

    @Test func writesCanonicalPCM16Header() {
        let data = WAVEncoder.encode(samples: [0, 0.5, -0.5, 1], sampleRate: 16_000)

        #expect(data.count == 44 + 8)
        #expect(String(decoding: data.prefix(4), as: UTF8.self) == "RIFF")
        #expect(readUInt32(data, 4) == UInt32(36 + 8))
        #expect(String(decoding: data[8..<12], as: UTF8.self) == "WAVE")
        #expect(String(decoding: data[12..<16], as: UTF8.self) == "fmt ")
        #expect(readUInt32(data, 16) == 16)
        #expect(readUInt16(data, 20) == 1) // PCM
        #expect(readUInt16(data, 22) == 1) // mono
        #expect(readUInt32(data, 24) == 16_000)
        #expect(readUInt32(data, 28) == 32_000) // byte rate
        #expect(readUInt16(data, 32) == 2) // block align
        #expect(readUInt16(data, 34) == 16)
        #expect(String(decoding: data[36..<40], as: UTF8.self) == "data")
        #expect(readUInt32(data, 40) == 8)
    }

    @Test func convertsAndClampsSamples() {
        let data = WAVEncoder.encode(samples: [0, 1, -1, 2, -2], sampleRate: 16_000)
        let pcm = (0..<5).map { Int16(bitPattern: readUInt16(data, 44 + $0 * 2)) }
        #expect(pcm == [0, 32_767, -32_767, 32_767, -32_767])
    }

    @Test func durationFromSampleCount() {
        #expect(WAVEncoder.duration(sampleCount: 24_000, sampleRate: 16_000) == 1.5)
    }
}
