// AudioLevelMeterTests.swift
// Provides Audio Level Meter Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class AudioLevelMeterTests: XCTestCase {
    func testSilenceAndAmbientNoiseStayNearZero() {
        XCTAssertEqual(AudioLevelMeter.normalizedLevel(fromLinearRMS: 0), 0, accuracy: 0.0001)
        XCTAssertEqual(AudioLevelMeter.normalizedLevel(fromLinearRMS: 0.009), 0, accuracy: 0.0001)
        XCTAssertLessThan(AudioLevelMeter.normalizedLevel(fromLinearRMS: 0.012), 0.05)
    }

    func testMeterSeparatesLowMediumAndHighInput() {
        let low = AudioLevelMeter.normalizedLevel(fromLinearRMS: 0.02)
        let medium = AudioLevelMeter.normalizedLevel(fromLinearRMS: 0.05)
        let high = AudioLevelMeter.normalizedLevel(fromLinearRMS: 0.12)

        XCTAssertGreaterThan(low, 0)
        XCTAssertGreaterThan(medium, low)
        XCTAssertGreaterThan(high, medium)
        XCTAssertLessThan(low, 0.15)
        XCTAssertGreaterThan(high, 0.75)
    }

    func testPCM16MeteringMatchesLinearExpectation() {
        let samples: [Int16] = [0, 0, 1200, -1200, 2400, -2400, 0, 0]
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        XCTAssertGreaterThan(AudioLevelMeter.normalizedLevel(fromPCM16: data), 0)
    }
}
