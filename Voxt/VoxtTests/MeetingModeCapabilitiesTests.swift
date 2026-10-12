// MeetingModeCapabilitiesTests.swift
// Provides Meeting Mode Capabilities Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MeetingModeCapabilitiesTests: XCTestCase {
    func testMeetingModeCapabilitiesMatchRoutingPolicy() {
        let meeting = MeetingCaptureMode.meeting.capabilities
        let subtitles = MeetingCaptureMode.subtitles.capabilities
        let recording = MeetingCaptureMode.recording.capabilities

        XCTAssertTrue(meeting.allowsSpeakerFeatures)
        XCTAssertTrue(meeting.realtimeDiarizationSources.isEmpty)
        XCTAssertEqual(meeting.finalDiarizationSources, [.systemAudio])
        XCTAssertEqual(meeting.defaultSpeaker(for: .microphone), .me)
        XCTAssertEqual(meeting.defaultSpeaker(for: .systemAudio), .them)

        XCTAssertFalse(subtitles.allowsSpeakerFeatures)
        XCTAssertTrue(subtitles.realtimeDiarizationSources.isEmpty)
        XCTAssertTrue(subtitles.finalDiarizationSources.isEmpty)
        XCTAssertEqual(subtitles.defaultSpeaker(for: .systemAudio), .them)

        XCTAssertFalse(recording.allowsSpeakerFeatures)
        XCTAssertTrue(recording.realtimeDiarizationSources.isEmpty)
        XCTAssertTrue(recording.finalDiarizationSources.isEmpty)
        XCTAssertEqual(recording.defaultSpeaker(for: .microphone), .me)
    }

    func testCaptureModeTransitionsPreserveSharedAudioSources() {
        XCTAssertEqual(
            MeetingCaptureMode.meeting.sourceTransition(to: .subtitles),
            MeetingCaptureSourceTransition(
                stopsMicrophone: true,
                stopsSystemAudio: false,
                startsMicrophone: false,
                startsSystemAudio: false
            )
        )
        XCTAssertEqual(
            MeetingCaptureMode.subtitles.sourceTransition(to: .meeting),
            MeetingCaptureSourceTransition(
                stopsMicrophone: false,
                stopsSystemAudio: false,
                startsMicrophone: true,
                startsSystemAudio: false
            )
        )
        XCTAssertEqual(
            MeetingCaptureMode.subtitles.sourceTransition(to: .recording),
            MeetingCaptureSourceTransition(
                stopsMicrophone: false,
                stopsSystemAudio: true,
                startsMicrophone: true,
                startsSystemAudio: false
            )
        )
        XCTAssertEqual(
            MeetingCaptureMode.recording.sourceTransition(to: .subtitles),
            MeetingCaptureSourceTransition(
                stopsMicrophone: true,
                stopsSystemAudio: false,
                startsMicrophone: false,
                startsSystemAudio: true
            )
        )
    }
}
