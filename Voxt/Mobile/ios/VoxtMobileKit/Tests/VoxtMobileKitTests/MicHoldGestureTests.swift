import Foundation
import Testing
@testable import VoxtMobileKit

/// Tap the mic to dictate; hold it, slide onto a language pill and let go to translate.
@Suite struct MicHoldGestureTests {
    private let t0 = Date(timeIntervalSince1970: 1_000)
    private func at(_ seconds: Double) -> Date { t0.addingTimeInterval(seconds) }
    private func center(_ index: Int, of count: Int) -> MicHoldGesture.Offset { MicHoldGesture.pillCenters(count: count)[index] }

    @Test func aTapIsPlainDictation() {
        var gesture = MicHoldGesture(targets: ["es", "ja"])
        gesture.down(at: t0)
        #expect(gesture.update(now: at(0.1), offset: .zero).isEmpty)
        #expect(gesture.pills.isEmpty, "no pills before the hold delay")
        #expect(gesture.up(now: at(0.15), offset: .zero) == .dictate)
    }

    @Test func holdingFansOutThePillsInTheSavedOrder() {
        var gesture = MicHoldGesture(targets: ["es", "ja", "ko"])
        gesture.down(at: t0)
        #expect(gesture.update(now: at(0.31), offset: .zero) == [.pillsShown])
        #expect(gesture.pills == [.language("es"), .language("ja"), .language("ko")])
        #expect(gesture.update(now: at(0.4), offset: .zero).isEmpty, "shown once")
        let centers = MicHoldGesture.pillCenters(count: 3)
        #expect(centers[0].dx == 0 && centers[0].dy < 0, "the first sits straight above the mic")
        #expect(centers[1].dx < 0 && centers[2].dx > 0, "second to the left, third to the right")
    }

    @Test func releasingOnAPillTranslatesIntoIt() {
        var gesture = MicHoldGesture(targets: ["es", "ja"])
        gesture.down(at: t0)
        _ = gesture.update(now: at(0.35), offset: .zero)
        #expect(gesture.update(now: at(0.5), offset: center(1, of: 2)) == [.pillEntered])
        #expect(gesture.entered == 1)
        #expect(gesture.update(now: at(0.55), offset: center(1, of: 2)).isEmpty, "entering the same pill once")
        #expect(gesture.up(now: at(0.6), offset: center(1, of: 2)) == .translate("ja"))
    }

    @Test func releasingAnywhereElseDoesNothing() {
        var gesture = MicHoldGesture(targets: ["es"])
        gesture.down(at: t0)
        _ = gesture.update(now: at(0.35), offset: .zero)
        _ = gesture.update(now: at(0.5), offset: center(0, of: 1))
        #expect(gesture.update(now: at(0.6), offset: .zero).isEmpty)
        #expect(gesture.entered == nil, "sliding back down leaves the pill")
        #expect(gesture.up(now: at(0.7), offset: .zero) == .nothing)
        var held = MicHoldGesture(targets: ["es"])
        held.down(at: t0)
        #expect(held.up(now: at(0.6), offset: MicHoldGesture.Offset(dx: 140, dy: 40)) == .nothing,
                "a hold released away from every pill is not a tap")
    }

    @Test func withNoTargetsOnePillOpensTheTranslationScreen() {
        var gesture = MicHoldGesture(targets: [])
        gesture.down(at: t0)
        _ = gesture.update(now: at(0.35), offset: .zero)
        #expect(gesture.pills == [.chooseLanguages])
        _ = gesture.update(now: at(0.5), offset: center(0, of: 1))
        #expect(gesture.up(now: at(0.6), offset: center(0, of: 1)) == .chooseLanguages)
    }

    @Test func atMostThreePills() {
        var gesture = MicHoldGesture(targets: ["es", "ja", "ko", "fr"])
        gesture.down(at: t0)
        _ = gesture.update(now: at(0.35), offset: .zero)
        #expect(gesture.pills.count == 3)
    }
}
