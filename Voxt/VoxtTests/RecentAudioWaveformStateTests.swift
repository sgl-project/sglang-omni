// RecentAudioWaveformStateTests.swift
// Provides Recent Audio Waveform State Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RecentAudioWaveformStateTests: XCTestCase {
    @MainActor
    func testAudioLevelDeliveryCoalescesPendingLevels() async {
        let delivery = MLXAudioLevelDelivery()
        var deliveredLevels: [Float] = []
        let delivered = expectation(description: "latest audio level delivered")

        delivery.submit(0.1) {
            deliveredLevels.append($0)
            delivered.fulfill()
        }
        delivery.submit(0.4) {
            deliveredLevels.append($0)
            delivered.fulfill()
        }
        delivery.submit(0.9) {
            deliveredLevels.append($0)
            delivered.fulfill()
        }

        await fulfillment(of: [delivered], timeout: 1)

        XCTAssertEqual(deliveredLevels, [0.9])
    }

    @MainActor
    func testAudioLevelDeliveryCanScheduleAgainAfterDelivery() async {
        let delivery = MLXAudioLevelDelivery()
        var deliveredLevels: [Float] = []
        let firstDelivery = expectation(description: "first audio level delivered")
        let secondDelivery = expectation(description: "second audio level delivered")

        delivery.submit(0.2) {
            deliveredLevels.append($0)
            firstDelivery.fulfill()
        }
        await fulfillment(of: [firstDelivery], timeout: 1)
        delivery.submit(0.7) {
            deliveredLevels.append($0)
            secondDelivery.fulfill()
        }
        await fulfillment(of: [secondDelivery], timeout: 1)

        XCTAssertEqual(deliveredLevels, [0.2, 0.7])
    }

    func testWaveformMaintainsConfiguredBarCount() {
        var waveform = RecentAudioWaveformModel(barCount: 16, historyDuration: 2.0, framesPerSecond: 18)

        waveform.ingest(level: 0.9)
        waveform.advanceFrame()

        XCTAssertEqual(waveform.barLevels.count, 16)
        XCTAssertTrue(waveform.barLevels.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    func testWaveformRisesForLoudInputAndDecaysAfterSilence() {
        var waveform = RecentAudioWaveformModel(barCount: 16, historyDuration: 2.0, framesPerSecond: 18)

        waveform.ingest(level: 0.95)
        for _ in 0..<6 {
            waveform.advanceFrame()
        }
        let loudPeak = waveform.barLevels.max() ?? 0

        for _ in 0..<36 {
            waveform.advanceFrame()
        }
        let decayedPeak = waveform.barLevels.max() ?? 0
        let recentDecayedPeak = waveform.barLevels.suffix(4).max() ?? 0

        XCTAssertGreaterThan(loudPeak, 0.55)
        XCTAssertLessThan(decayedPeak, loudPeak)
        XCTAssertLessThan(recentDecayedPeak, 0.18)
    }

    func testSilenceDoesNotProduceFlatFullWaveform() {
        var waveform = RecentAudioWaveformModel(barCount: 16, historyDuration: 2.0, framesPerSecond: 18)

        for _ in 0..<18 {
            waveform.advanceFrame()
        }

        let peak = waveform.barLevels.max() ?? 0
        XCTAssertLessThan(peak, 0.08)
    }

    func testLoudInputCreatesHigherPeakThanMediumInput() {
        var mediumWaveform = RecentAudioWaveformModel(barCount: 16, historyDuration: 2.0, framesPerSecond: 18)
        var loudWaveform = RecentAudioWaveformModel(barCount: 16, historyDuration: 2.0, framesPerSecond: 18)

        for _ in 0..<6 {
            mediumWaveform.ingest(level: 0.35)
            mediumWaveform.advanceFrame()
            loudWaveform.ingest(level: 0.85)
            loudWaveform.advanceFrame()
        }

        XCTAssertGreaterThan(loudWaveform.barLevels.max() ?? 0, mediumWaveform.barLevels.max() ?? 0)
    }
}
