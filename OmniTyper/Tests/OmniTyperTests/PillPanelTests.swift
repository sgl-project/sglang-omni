// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import OmniTyper

struct PillPanelTests {
    private static let frame = 1.0 / 60

    private static func speak(_ waveform: inout PillWaveform, level: Double, seconds: Double) {
        var time = 0.0
        while time < seconds {
            waveform.advance(by: frame, level: level, time: time)
            time += frame
        }
    }

    @Test func waitingIsARowOfStillDotsAndListeningBreathes() {
        var waveform = PillWaveform()
        #expect(waveform.heights == PillWaveform.idle)
        Self.speak(&waveform, level: 0, seconds: 1)
        #expect(waveform.heights.allSatisfy { $0 == PillWaveform.barWidth }, "Waiting never moves")
        #expect(PillWaveform.maximumHeight < PillWaveform.capsuleSize.height)

        waveform.startListening(at: 0)
        var tallest: CGFloat = 0
        for step in 0..<12 {
            waveform.advance(by: Self.frame, level: 0, time: Double(step) * Self.frame)
            tallest = max(tallest, waveform.heights.max()!)
        }
        #expect(tallest > PillWaveform.maximumHeight * 0.5, "Going live raises the bars once, even in silence")
        var time = 1.0
        var seen: Set<Int> = []
        while time < 3 {
            waveform.advance(by: Self.frame, level: 0, time: time); time += Self.frame
            seen.insert(Int(waveform.heights[0] * 10))
        }
        #expect(waveform.heights.allSatisfy { $0 > PillWaveform.barWidth && $0 < PillWaveform.barWidth + 3 }, "A live microphone breathes just above the dots")
        #expect(seen.count > 2, "The breathing moves")
        waveform.rest(by: 1)
        #expect(waveform.heights.allSatisfy { $0 < PillWaveform.barWidth + 0.2 }, "Stopping settles back to dots")
    }

    @Test func speechFillsTheCapsuleAndFallsBackSlowly() {
        var waveform = PillWaveform()
        Self.speak(&waveform, level: 0.6, seconds: 0.3)
        var tallest: CGFloat = 0
        for step in 0..<30 {
            waveform.advance(by: Self.frame, level: 0.6, time: 0.3 + Double(step) * Self.frame)
            tallest = max(tallest, waveform.heights.max()!)
        }
        let speaking = waveform.heights
        #expect(speaking.count == PillWaveform.barCount)
        #expect(tallest > PillWaveform.maximumHeight * 0.85, "Steady speech reaches near the top")
        #expect(speaking.allSatisfy { $0 > PillWaveform.barWidth && $0 <= PillWaveform.maximumHeight })
        #expect(Set(speaking.map { Int($0 * 10) }).count > 4, "Bars form a shape, not a flat block")

        waveform.advance(by: Self.frame, level: 0, time: 0.5)
        let justAfter = waveform.heights
        #expect(zip(justAfter, speaking).allSatisfy { $0 > $1 * 0.9 }, "One silent frame barely moves the bars")
        for step in 0..<90 { waveform.advance(by: Self.frame, level: 0, time: 0.5 + Double(step) * Self.frame) }
        #expect(waveform.heights.allSatisfy { $0 < PillWaveform.barWidth + 3 }, "Sustained silence settles to the breathing floor")
    }

    @Test func aQuietMicrophoneStillReadsAsSpeech() {
        var loud = PillWaveform(), quiet = PillWaveform()
        Self.speak(&loud, level: 0.9, seconds: 1)
        Self.speak(&quiet, level: 0.15, seconds: 1)
        #expect(quiet.heights.max()! > loud.heights.max()! * 0.7, "The scale follows the speaker, not the microphone")
    }

    @Test func librariesSavedBeforeTheSettingKeepTheDetailedPopup() throws {
        var preferences = Preferences()
        #expect(preferences.compactPopup == nil)
        let saved = try JSONEncoder().encode(preferences)
        #expect(try JSONDecoder().decode(Preferences.self, from: saved).compactPopup == nil)
        preferences.compactPopup = true
        let updated = try JSONEncoder().encode(preferences)
        #expect(try JSONDecoder().decode(Preferences.self, from: updated).compactPopup == true)
    }
}
