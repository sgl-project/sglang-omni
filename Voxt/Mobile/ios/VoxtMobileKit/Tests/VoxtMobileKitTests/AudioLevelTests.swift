import Testing
@testable import VoxtMobileKit

struct AudioLevelTests {
    @Test func silenceAndEmptyAreZero() {
        #expect(AudioLevel.rms([]) == 0)
        #expect(AudioLevel.rms([0, 0, 0]) == 0)
    }
    @Test func measuresEnergyWithoutCancellingOppositeSamples() {
        #expect(abs(AudioLevel.rms([0.5, -0.5]) - 0.5) < 0.001)
        #expect(abs(AudioLevel.rms([0, 1]) - 0.7071) < 0.001)
    }
    @Test func clampsInvalidAndClippedAudio() {
        #expect(AudioLevel.rms([2, -2]) == 1)
        #expect(AudioLevel.rms([.nan, .infinity]) == 0)
    }
}
