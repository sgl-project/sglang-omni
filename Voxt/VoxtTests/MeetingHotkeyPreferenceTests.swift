// MeetingHotkeyPreferenceTests.swift
// Provides Meeting Hotkey Preference Tests for Voxt test coverage.

import XCTest
import AppKit
import Carbon
@testable import Voxt

final class MeetingHotkeyPreferenceTests: XCTestCase {
    func testDefaultMeetingHotkeyIsFnOption() {
        let hotkey = HotkeyPreference.loadMeeting()

        XCTAssertEqual(hotkey.keyCode, HotkeyPreference.defaultMeetingKeyCode)
        XCTAssertEqual(hotkey.modifiers, [.function, .option])
        XCTAssertEqual(hotkey.sidedModifiers, [])
    }

    func testCommandPresetUsesRightCommandPlusLForMeeting() {
        let preset = HotkeyPreference.presetHotkeys(for: .commandCombo)

        XCTAssertEqual(preset?.meeting.keyCode, UInt16(kVK_ANSI_L))
        XCTAssertEqual(preset?.meeting.modifiers, [.command])
        XCTAssertEqual(preset?.meeting.sidedModifiers, [.rightCommand])
    }

    func testFnPresetKeepsCustomPasteAtControlCommandV() {
        let preset = HotkeyPreference.presetHotkeys(for: .fnCombo)

        XCTAssertEqual(preset?.customPaste.keyCode, UInt16(kVK_ANSI_V))
        XCTAssertEqual(preset?.customPaste.modifiers, [.control, .command])
        XCTAssertEqual(preset?.customPaste.sidedModifiers, [])
    }

    func testCommandPresetKeepsCustomPasteAtControlCommandV() {
        let preset = HotkeyPreference.presetHotkeys(for: .commandCombo)

        XCTAssertEqual(preset?.customPaste.keyCode, UInt16(kVK_ANSI_V))
        XCTAssertEqual(preset?.customPaste.modifiers, [.control, .command])
        XCTAssertEqual(preset?.customPaste.sidedModifiers, [])
    }
}
