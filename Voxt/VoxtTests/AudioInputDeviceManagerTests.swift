// AudioInputDeviceManagerTests.swift
// Provides Audio Input Device Manager Tests for Voxt test coverage.

import XCTest
import AVFoundation
@testable import Voxt

final class AudioInputDeviceManagerTests: XCTestCase {
    func testSnapshotFilteringExcludesCoreAudioAggregateDevices() {
        XCTAssertFalse(
            AudioInputDeviceManager.shouldIncludeInSnapshot(
                uid: "CADefaultDeviceAggregate-72904-0",
                name: "CADefaultDeviceAggregate-72904-0"
            )
        )
    }

    func testSnapshotFilteringKeepsRegularMicrophones() {
        XCTAssertTrue(
            AudioInputDeviceManager.shouldIncludeInSnapshot(
                uid: "BuiltInMicrophoneDevice",
                name: "MacBook Pro Mic"
            )
        )
    }

    func testPreferredDeviceWinsWhenAvailable() {
        let devices = [
            AudioInputDevice(id: 10, uid: "mic-a", name: "Mic A"),
            AudioInputDevice(id: 20, uid: "mic-b", name: "Mic B")
        ]

        let resolved = AudioInputDeviceManager.resolvedInputDeviceID(
            from: devices,
            preferredID: 20,
            defaultDeviceID: 10
        )

        XCTAssertEqual(resolved, 20)
    }

    func testDefaultDeviceFallbackIsUsedWhenPreferredMissing() {
        let devices = [
            AudioInputDevice(id: 10, uid: "mic-a", name: "Mic A"),
            AudioInputDevice(id: 20, uid: "mic-b", name: "Mic B")
        ]

        let resolved = AudioInputDeviceManager.resolvedInputDeviceID(
            from: devices,
            preferredID: 99,
            defaultDeviceID: 20
        )

        XCTAssertEqual(resolved, 20)
    }

    func testFirstDeviceFallbackIsUsedWhenNothingElseMatches() {
        let devices = [
            AudioInputDevice(id: 10, uid: "mic-a", name: "Mic A"),
            AudioInputDevice(id: 20, uid: "mic-b", name: "Mic B")
        ]

        let resolved = AudioInputDeviceManager.resolvedInputDeviceID(
            from: devices,
            preferredID: nil,
            defaultDeviceID: 99
        )

        XCTAssertEqual(resolved, 10)
    }

    func testSnapshotFilterExcludesVoxtProcessTapDevice() {
        XCTAssertFalse(
            AudioInputDeviceManager.shouldIncludeInSnapshot(
                uid: "voxt-process-tap-2B3C106D-E5A2-4C0C-B910-23275522F843",
                name: "VoxtProcessTap"
            )
        )
    }

    func testSnapshotFilterExcludesAnonymousCoreAudioAggregateDevice() {
        XCTAssertFalse(
            AudioInputDeviceManager.shouldIncludeInSnapshot(
                uid: "CADefaultDeviceAggregate-8135-0",
                name: "CADefaultDeviceAggregate-8135-0"
            )
        )
    }

    func testSnapshotFilterKeepsRealMicrophoneDevices() {
        XCTAssertTrue(
            AudioInputDeviceManager.shouldIncludeInSnapshot(
                uid: "BuiltInMicrophoneDevice",
                name: "MacBook Air麦克风"
            )
        )
    }

    func testCaptureTapFormatUsesHardwareSampleRateWhenNodeFormatDiffers() throws {
        let nodeFormat = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 48_000,
                channels: 1,
                interleaved: false
            )
        )

        let tapFormat = AudioInputDeviceManager.captureTapFormat(
            nodeOutputFormat: nodeFormat,
            hardwareSampleRate: 44_100
        )

        XCTAssertEqual(tapFormat.sampleRate, 44_100, accuracy: 0.1)
        XCTAssertEqual(tapFormat.channelCount, nodeFormat.channelCount)
        XCTAssertEqual(tapFormat.commonFormat, nodeFormat.commonFormat)
        XCTAssertEqual(tapFormat.isInterleaved, nodeFormat.isInterleaved)
    }

    func testCaptureTapFormatUsesHardwareSampleRateWhenNodeReportsLowerRate() throws {
        let nodeFormat = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 44_100,
                channels: 1,
                interleaved: false
            )
        )

        let tapFormat = AudioInputDeviceManager.captureTapFormat(
            nodeOutputFormat: nodeFormat,
            hardwareSampleRate: 48_000
        )

        XCTAssertEqual(tapFormat.sampleRate, 48_000, accuracy: 0.1)
        XCTAssertEqual(tapFormat.channelCount, nodeFormat.channelCount)
        XCTAssertEqual(tapFormat.commonFormat, nodeFormat.commonFormat)
        XCTAssertEqual(tapFormat.isInterleaved, nodeFormat.isInterleaved)
    }

    func testCaptureTapFormatKeepsNodeFormatWhenHardwareSampleRateIsUnavailable() throws {
        let nodeFormat = try XCTUnwrap(
            AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 48_000,
                channels: 1,
                interleaved: false
            )
        )

        let tapFormat = AudioInputDeviceManager.captureTapFormat(
            nodeOutputFormat: nodeFormat,
            hardwareSampleRate: nil
        )

        XCTAssertEqual(tapFormat.sampleRate, nodeFormat.sampleRate, accuracy: 0.1)
    }
}
