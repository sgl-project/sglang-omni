import Foundation
import Testing
@testable import VoxtMobileKit

/// A tap with no speech should not cost an upload and two model passes.
@Suite struct SpeechGateTests {
    private func tone(seconds: Double, amplitude: Float) -> [Float] {
        (0..<Int(seconds * 16_000)).map { amplitude * Float(sin(Double($0) * 2 * .pi * 220 / 16_000)) }
    }

    @Test func silenceAndRoomNoiseAreNotSpeech() {
        #expect(SpeechGate.containsSpeech([Float](repeating: 0, count: 16_000)) == false)
        let hiss = (0..<32_000).map { Float(($0 * 7919) % 101 - 50) / 50 * 0.002 }
        #expect(SpeechGate.containsSpeech(hiss) == false)
        #expect(SpeechGate.containsSpeech(tone(seconds: 0.2, amplitude: 0.3)) == false, "under 250 ms")
    }

    @Test func voicedAudioIsSpeech() {
        let quietLead = [Float](repeating: 0, count: 8_000)
        #expect(SpeechGate.containsSpeech(quietLead + tone(seconds: 0.6, amplitude: 0.2) + quietLead))
        #expect(SpeechGate.containsSpeech([Float](repeating: 0.1, count: 16_000)))
        #expect(SpeechGate.containsSpeech(tone(seconds: 1, amplitude: 0.02)), "quiet but present (-37 dBFS)")
    }
}
