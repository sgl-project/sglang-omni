// HotkeySupportTests.swift
// Provides Hotkey Support Tests for Voxt test coverage.

import XCTest
import AppKit
import Carbon
import IOKit.hidsystem
@testable import Voxt

@MainActor
final class HotkeySupportTests: XCTestCase {
    func testNoteShortcutsAlwaysLoadAsMultipleTapBindings() {
        let suiteName = "HotkeySupportTests.note.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: UInt16(kVK_ANSI_N),
            modifiers: [.command],
            sidedModifiers: []
        )

        HotkeyPreference.saveNoteBindings([
            .init(hotkey: hotkey, behavior: .doubleTap),
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_ANSI_B),
                    modifiers: [],
                    sidedModifiers: []
                ),
                behavior: .longPress
            )
        ], defaults: defaults)

        let loaded = HotkeyPreference.loadNoteBindings(defaults: defaults)
        XCTAssertEqual(loaded.count, 2)
        XCTAssertEqual(loaded.first?.hotkey, hotkey)
        XCTAssertTrue(loaded.allSatisfy { $0.behavior == .tap })
        XCTAssertEqual(loaded.last?.hotkey.keyCode, UInt16(kVK_ANSI_B))
    }

    func testFnPresetUsesRightCommandForNotes() {
        let preset = HotkeyPreference.presetHotkeys(for: .fnCombo)

        XCTAssertTrue(preset?.distinguishSides == true)
        XCTAssertEqual(preset?.note.keyCode, HotkeyPreference.modifierOnlyKeyCode)
        XCTAssertEqual(preset?.note.modifiers, [.command])
        XCTAssertEqual(preset?.note.sidedModifiers, [.rightCommand])
        XCTAssertEqual(preset?.noteBindings.first?.behavior, .tap)
    }

    func testCommandPresetUsesRightOptionForNotes() {
        let preset = HotkeyPreference.presetHotkeys(for: .commandCombo)

        XCTAssertTrue(preset?.distinguishSides == true)
        XCTAssertEqual(preset?.note.keyCode, HotkeyPreference.modifierOnlyKeyCode)
        XCTAssertEqual(preset?.note.modifiers, [.option])
        XCTAssertEqual(preset?.note.sidedModifiers, [.rightOption])
        XCTAssertEqual(preset?.noteBindings.first?.behavior, .tap)
    }

    func testMouseMiddlePresetUsesRightCommandForNotes() {
        let preset = HotkeyPreference.presetHotkeys(for: .mouseMiddleFnShift)

        XCTAssertTrue(preset?.distinguishSides == true)
        XCTAssertEqual(preset?.note.keyCode, HotkeyPreference.modifierOnlyKeyCode)
        XCTAssertEqual(preset?.note.modifiers, [.command])
        XCTAssertEqual(preset?.note.sidedModifiers, [.rightCommand])
        XCTAssertEqual(preset?.noteBindings.first?.behavior, .tap)
    }

    func testGlobalShortcutValidationAllowsBareKeyboardKey() {
        XCTAssertTrue(
            HotkeyPreference.isAllowedGlobalShortcut(
                HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_ANSI_F),
                    modifiers: [],
                    sidedModifiers: []
                )
            )
        )
    }

    func testGlobalShortcutValidationAllowsRightCommandModifierOnly() {
        XCTAssertTrue(
            HotkeyPreference.isAllowedGlobalShortcut(
                HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: [.command],
                    sidedModifiers: [.rightCommand]
                )
            )
        )
    }

    func testGlobalShortcutValidationRejectsUnsupportedKeyboardKeyCode() {
        XCTAssertFalse(
            HotkeyPreference.isAllowedGlobalShortcut(
                HotkeyPreference.Hotkey(
                    keyCode: 179,
                    modifiers: [],
                    sidedModifiers: []
                )
            )
        )
    }

    func testRecorderIgnoresUnsupportedKeyboardKeyCode() async {
        let recorder = KeyCaptureView()
        var capturedHotkey: HotkeyPreference.Hotkey?
        recorder.onHotkeyCaptured = { capturedHotkey = $0 }

        recorder.debugCaptureKeyDownForTests(keyCode: 179)
        await Task.yield()

        XCTAssertNil(capturedHotkey)
    }

    func testLoadCanonicalizesStoredFunctionModifierHotkey() {
        let defaults = UserDefaults.standard
        let presetKey = AppPreferenceKey.hotkeyPreset
        let keyCodeKey = AppPreferenceKey.hotkeyKeyCode
        let modifiersKey = AppPreferenceKey.hotkeyModifiers
        let sidedModifiersKey = AppPreferenceKey.hotkeySidedModifiers
        let savedPreset = defaults.object(forKey: presetKey)
        let savedKeyCode = defaults.object(forKey: keyCodeKey)
        let savedModifiers = defaults.object(forKey: modifiersKey)
        let savedSidedModifiers = defaults.object(forKey: sidedModifiersKey)
        defer {
            restoreDefaultsValue(savedPreset, forKey: presetKey)
            restoreDefaultsValue(savedKeyCode, forKey: keyCodeKey)
            restoreDefaultsValue(savedModifiers, forKey: modifiersKey)
            restoreDefaultsValue(savedSidedModifiers, forKey: sidedModifiersKey)
        }

        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: presetKey)
        defaults.set(Int(UInt16(kVK_Function)), forKey: keyCodeKey)
        defaults.set(Int(NSEvent.ModifierFlags.function.rawValue), forKey: modifiersKey)
        defaults.set(0, forKey: sidedModifiersKey)

        XCTAssertEqual(
            HotkeyPreference.load(),
            HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.modifierOnlyKeyCode,
                modifiers: [.function],
                sidedModifiers: []
            )
        )
    }

    func testLoadCanonicalizesStoredRightCommandModifierHotkey() {
        let defaults = UserDefaults.standard
        let presetKey = AppPreferenceKey.hotkeyPreset
        let keyCodeKey = AppPreferenceKey.hotkeyKeyCode
        let modifiersKey = AppPreferenceKey.hotkeyModifiers
        let sidedModifiersKey = AppPreferenceKey.hotkeySidedModifiers
        let savedPreset = defaults.object(forKey: presetKey)
        let savedKeyCode = defaults.object(forKey: keyCodeKey)
        let savedModifiers = defaults.object(forKey: modifiersKey)
        let savedSidedModifiers = defaults.object(forKey: sidedModifiersKey)
        defer {
            restoreDefaultsValue(savedPreset, forKey: presetKey)
            restoreDefaultsValue(savedKeyCode, forKey: keyCodeKey)
            restoreDefaultsValue(savedModifiers, forKey: modifiersKey)
            restoreDefaultsValue(savedSidedModifiers, forKey: sidedModifiersKey)
        }

        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: presetKey)
        defaults.set(Int(UInt16(kVK_RightCommand)), forKey: keyCodeKey)
        defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: modifiersKey)
        defaults.set(0, forKey: sidedModifiersKey)

        XCTAssertEqual(
            HotkeyPreference.load(),
            HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.modifierOnlyKeyCode,
                modifiers: [.command],
                sidedModifiers: [.rightCommand]
            )
        )
    }

    func testLoadCustomPasteResolvesToPresetValueWhenPresetIsCommandCombo() {
        let defaults = UserDefaults.standard
        let presetKey = AppPreferenceKey.hotkeyPreset
        let keyCodeKey = AppPreferenceKey.customPasteHotkeyKeyCode
        let modifiersKey = AppPreferenceKey.customPasteHotkeyModifiers
        let sidedModifiersKey = AppPreferenceKey.customPasteHotkeySidedModifiers
        let distinguishSidesKey = AppPreferenceKey.hotkeyDistinguishModifierSides
        let savedPreset = defaults.object(forKey: presetKey)
        let savedKeyCode = defaults.object(forKey: keyCodeKey)
        let savedModifiers = defaults.object(forKey: modifiersKey)
        let savedSidedModifiers = defaults.object(forKey: sidedModifiersKey)
        let savedDistinguishSides = defaults.object(forKey: distinguishSidesKey)
        defer {
            restoreDefaultsValue(savedPreset, forKey: presetKey)
            restoreDefaultsValue(savedKeyCode, forKey: keyCodeKey)
            restoreDefaultsValue(savedModifiers, forKey: modifiersKey)
            restoreDefaultsValue(savedSidedModifiers, forKey: sidedModifiersKey)
            restoreDefaultsValue(savedDistinguishSides, forKey: distinguishSidesKey)
        }

        defaults.set(HotkeyPreference.Preset.commandCombo.rawValue, forKey: presetKey)
        defaults.set(Int(UInt16(kVK_ANSI_V)), forKey: keyCodeKey)
        defaults.set(Int(NSEvent.ModifierFlags([.control, .command]).rawValue), forKey: modifiersKey)
        defaults.set(0, forKey: sidedModifiersKey)
        defaults.set(false, forKey: distinguishSidesKey)

        XCTAssertEqual(
            HotkeyPreference.loadCustomPaste(),
            HotkeyPreference.Hotkey(
                keyCode: UInt16(kVK_ANSI_V),
                modifiers: [.control, .command],
                sidedModifiers: []
            )
        )
        XCTAssertTrue(HotkeyPreference.loadDistinguishModifierSides())
    }

    func testMigrateDefaultsSyncsStoredCommandPresetHotkeys() {
        let defaults = UserDefaults.standard
        let presetKey = AppPreferenceKey.hotkeyPreset
        let keyCodeKey = AppPreferenceKey.customPasteHotkeyKeyCode
        let modifiersKey = AppPreferenceKey.customPasteHotkeyModifiers
        let sidedModifiersKey = AppPreferenceKey.customPasteHotkeySidedModifiers
        let distinguishSidesKey = AppPreferenceKey.hotkeyDistinguishModifierSides
        let transcriptionKeyCodeKey = AppPreferenceKey.hotkeyKeyCode
        let transcriptionModifiersKey = AppPreferenceKey.hotkeyModifiers
        let transcriptionSidedKey = AppPreferenceKey.hotkeySidedModifiers
        let savedPreset = defaults.object(forKey: presetKey)
        let savedKeyCode = defaults.object(forKey: keyCodeKey)
        let savedModifiers = defaults.object(forKey: modifiersKey)
        let savedSidedModifiers = defaults.object(forKey: sidedModifiersKey)
        let savedDistinguishSides = defaults.object(forKey: distinguishSidesKey)
        let savedTranscriptionKeyCode = defaults.object(forKey: transcriptionKeyCodeKey)
        let savedTranscriptionModifiers = defaults.object(forKey: transcriptionModifiersKey)
        let savedTranscriptionSided = defaults.object(forKey: transcriptionSidedKey)
        defer {
            restoreDefaultsValue(savedPreset, forKey: presetKey)
            restoreDefaultsValue(savedKeyCode, forKey: keyCodeKey)
            restoreDefaultsValue(savedModifiers, forKey: modifiersKey)
            restoreDefaultsValue(savedSidedModifiers, forKey: sidedModifiersKey)
            restoreDefaultsValue(savedDistinguishSides, forKey: distinguishSidesKey)
            restoreDefaultsValue(savedTranscriptionKeyCode, forKey: transcriptionKeyCodeKey)
            restoreDefaultsValue(savedTranscriptionModifiers, forKey: transcriptionModifiersKey)
            restoreDefaultsValue(savedTranscriptionSided, forKey: transcriptionSidedKey)
        }

        defaults.set(HotkeyPreference.Preset.commandCombo.rawValue, forKey: presetKey)
        defaults.set(Int(UInt16(kVK_ANSI_V)), forKey: keyCodeKey)
        defaults.set(Int(NSEvent.ModifierFlags([.control, .command]).rawValue), forKey: modifiersKey)
        defaults.set(0, forKey: sidedModifiersKey)
        defaults.set(false, forKey: distinguishSidesKey)
        defaults.set(Int(HotkeyPreference.modifierOnlyKeyCode), forKey: transcriptionKeyCodeKey)
        defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: transcriptionModifiersKey)
        defaults.set(0, forKey: transcriptionSidedKey)

        HotkeyPreference.migrateDefaultsIfNeeded()

        XCTAssertEqual(defaults.integer(forKey: keyCodeKey), Int(UInt16(kVK_ANSI_V)))
        XCTAssertEqual(
            defaults.integer(forKey: modifiersKey),
            Int(NSEvent.ModifierFlags([.control, .command]).rawValue)
        )
        XCTAssertEqual(defaults.integer(forKey: sidedModifiersKey), 0)
        XCTAssertTrue(defaults.bool(forKey: distinguishSidesKey))
        XCTAssertEqual(defaults.integer(forKey: transcriptionSidedKey), SidedModifierFlags.rightCommand.rawValue)
    }

    func testLoadCustomPasteClearsSidedModifiersForChordHotkeys() {
        let defaults = UserDefaults.standard
        let presetKey = AppPreferenceKey.hotkeyPreset
        let keyCodeKey = AppPreferenceKey.customPasteHotkeyKeyCode
        let modifiersKey = AppPreferenceKey.customPasteHotkeyModifiers
        let sidedModifiersKey = AppPreferenceKey.customPasteHotkeySidedModifiers
        let savedPreset = defaults.object(forKey: presetKey)
        let savedKeyCode = defaults.object(forKey: keyCodeKey)
        let savedModifiers = defaults.object(forKey: modifiersKey)
        let savedSidedModifiers = defaults.object(forKey: sidedModifiersKey)
        defer {
            restoreDefaultsValue(savedPreset, forKey: presetKey)
            restoreDefaultsValue(savedKeyCode, forKey: keyCodeKey)
            restoreDefaultsValue(savedModifiers, forKey: modifiersKey)
            restoreDefaultsValue(savedSidedModifiers, forKey: sidedModifiersKey)
        }

        defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: presetKey)
        defaults.set(Int(UInt16(kVK_ANSI_L)), forKey: keyCodeKey)
        defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: modifiersKey)
        defaults.set(SidedModifierFlags.leftCommand.rawValue, forKey: sidedModifiersKey)

        XCTAssertEqual(
            HotkeyPreference.loadCustomPaste(),
            HotkeyPreference.Hotkey(
                keyCode: UInt16(kVK_ANSI_L),
                modifiers: [.command],
                sidedModifiers: []
            )
        )
    }

    func testApplyPresetPersistsCommandPresetHotkeys() {
        let defaults = UserDefaults.standard
        let presetKey = AppPreferenceKey.hotkeyPreset
        let distinguishSidesKey = AppPreferenceKey.hotkeyDistinguishModifierSides
        let transcriptionKeyCodeKey = AppPreferenceKey.hotkeyKeyCode
        let transcriptionModifiersKey = AppPreferenceKey.hotkeyModifiers
        let transcriptionSidedKey = AppPreferenceKey.hotkeySidedModifiers
        let customPasteKeyCodeKey = AppPreferenceKey.customPasteHotkeyKeyCode
        let customPasteModifiersKey = AppPreferenceKey.customPasteHotkeyModifiers
        let customPasteSidedKey = AppPreferenceKey.customPasteHotkeySidedModifiers
        let savedPreset = defaults.object(forKey: presetKey)
        let savedDistinguishSides = defaults.object(forKey: distinguishSidesKey)
        let savedTranscriptionKeyCode = defaults.object(forKey: transcriptionKeyCodeKey)
        let savedTranscriptionModifiers = defaults.object(forKey: transcriptionModifiersKey)
        let savedTranscriptionSided = defaults.object(forKey: transcriptionSidedKey)
        let savedCustomPasteKeyCode = defaults.object(forKey: customPasteKeyCodeKey)
        let savedCustomPasteModifiers = defaults.object(forKey: customPasteModifiersKey)
        let savedCustomPasteSided = defaults.object(forKey: customPasteSidedKey)
        defer {
            restoreDefaultsValue(savedPreset, forKey: presetKey)
            restoreDefaultsValue(savedDistinguishSides, forKey: distinguishSidesKey)
            restoreDefaultsValue(savedTranscriptionKeyCode, forKey: transcriptionKeyCodeKey)
            restoreDefaultsValue(savedTranscriptionModifiers, forKey: transcriptionModifiersKey)
            restoreDefaultsValue(savedTranscriptionSided, forKey: transcriptionSidedKey)
            restoreDefaultsValue(savedCustomPasteKeyCode, forKey: customPasteKeyCodeKey)
            restoreDefaultsValue(savedCustomPasteModifiers, forKey: customPasteModifiersKey)
            restoreDefaultsValue(savedCustomPasteSided, forKey: customPasteSidedKey)
        }

        let values = HotkeyPreference.applyPreset(.commandCombo)

        XCTAssertEqual(values?.distinguishSides, true)
        XCTAssertEqual(defaults.string(forKey: presetKey), HotkeyPreference.Preset.commandCombo.rawValue)
        XCTAssertTrue(defaults.bool(forKey: distinguishSidesKey))
        XCTAssertEqual(defaults.integer(forKey: transcriptionKeyCodeKey), Int(HotkeyPreference.modifierOnlyKeyCode))
        XCTAssertEqual(defaults.integer(forKey: transcriptionModifiersKey), Int(NSEvent.ModifierFlags.command.rawValue))
        XCTAssertEqual(defaults.integer(forKey: transcriptionSidedKey), SidedModifierFlags.rightCommand.rawValue)
        XCTAssertEqual(defaults.integer(forKey: customPasteKeyCodeKey), Int(UInt16(kVK_ANSI_V)))
        XCTAssertEqual(defaults.integer(forKey: customPasteModifiersKey), Int(NSEvent.ModifierFlags([.control, .command]).rawValue))
        XCTAssertEqual(defaults.integer(forKey: customPasteSidedKey), 0)
    }

    func testUpdatingKeepsModifierSetWhenDuplicatePressEventArrives() {
        let afterFirstPress = SidedModifierFlags.updating(
            from: [],
            keyCode: UInt16(kVK_RightShift),
            isPressed: true
        )
        let afterDuplicatePress = SidedModifierFlags.updating(
            from: afterFirstPress,
            keyCode: UInt16(kVK_RightShift),
            isPressed: true
        )

        XCTAssertEqual(afterDuplicatePress, [.rightShift])
    }

    func testUpdatingClearsModifierWhenDuplicateReleaseEventArrives() {
        let afterRelease = SidedModifierFlags.updating(
            from: [.rightShift],
            keyCode: UInt16(kVK_RightShift),
            isPressed: false
        )
        let afterDuplicateRelease = SidedModifierFlags.updating(
            from: afterRelease,
            keyCode: UInt16(kVK_RightShift),
            isPressed: false
        )

        XCTAssertEqual(afterDuplicateRelease, [])
    }

    func testUpdatingClearsOnlyReleasedSideWhenOppositeSideStaysDown() {
        let updated = SidedModifierFlags.updating(
            from: [.leftShift, .rightShift],
            keyCode: UInt16(kVK_Shift),
            isPressed: false
        )

        XCTAssertEqual(updated, [.rightShift])
    }

    func testFilteredDropsSidedModifiersWhenBaseModifierIsMissing() {
        let filtered = SidedModifierFlags([.rightShift, .rightCommand]).filtered(by: [.command])

        XCTAssertEqual(filtered, [.rightCommand])
    }

    func testHotkeyMatchesRequiresMatchingSideWhenDistinguishingEnabled() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.shift],
            sidedModifiers: [.rightShift]
        )

        XCTAssertTrue(
            HotkeyPreference.hotkeyMatches(
                hotkey,
                eventFlags: [.maskShift],
                sidedModifiers: [.rightShift],
                distinguishModifierSides: true
            )
        )
        XCTAssertFalse(
            HotkeyPreference.hotkeyMatches(
                hotkey,
                eventFlags: [.maskShift],
                sidedModifiers: [.leftShift],
                distinguishModifierSides: true
            )
        )
    }

    func testHotkeyMatchesIgnoresSideWhenDistinguishingDisabled() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.shift],
            sidedModifiers: [.rightShift]
        )

        XCTAssertTrue(
            HotkeyPreference.hotkeyMatches(
                hotkey,
                eventFlags: [.maskShift],
                sidedModifiers: [.leftShift],
                distinguishModifierSides: false
            )
        )
    }

    func testHotkeyMatchesAllowsAdditionalSidedModifiersWhenRequiredSideIsPresent() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.command, .shift],
            sidedModifiers: [.rightCommand, .rightShift]
        )

        XCTAssertTrue(
            HotkeyPreference.hotkeyMatches(
                hotkey,
                eventFlags: [.maskCommand, .maskShift],
                sidedModifiers: [.leftShift, .rightShift, .rightCommand],
                distinguishModifierSides: true
            )
        )
    }

    func testHotkeyMatchesRequiresMatchingSideForCustomNonModifierHotkey() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: UInt16(kVK_ANSI_L),
            modifiers: [.option],
            sidedModifiers: [.rightOption]
        )

        XCTAssertTrue(
            HotkeyPreference.hotkeyMatches(
                hotkey,
                eventFlags: [.maskAlternate],
                sidedModifiers: [.rightOption],
                distinguishModifierSides: true
            )
        )
        XCTAssertFalse(
            HotkeyPreference.hotkeyMatches(
                hotkey,
                eventFlags: [.maskAlternate],
                sidedModifiers: [.leftOption],
                distinguishModifierSides: true
            )
        )
    }

    func testSidedModifierFlagsFromEventFlagsParsesDeviceSpecificBits() {
        let flags = CGEventFlags(rawValue: UInt64(NX_COMMANDMASK | NX_DEVICERCMDKEYMASK | NX_DEVICERSHIFTKEYMASK))

        XCTAssertEqual(
            SidedModifierFlags.from(eventFlags: flags),
            [.rightCommand, .rightShift]
        )
    }

    func testDisplayStringShowsRightShiftWhenSideIsAvailable() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.shift],
            sidedModifiers: [.rightShift]
        )

        XCTAssertEqual(
            HotkeyPreference.displayString(for: hotkey, distinguishModifierSides: true),
            AppLocalization.format("Right %@", AppLocalization.localizedString("Shift"))
        )
    }

    func testDisplayStringFallsBackToGenericShiftWhenBothSidesArePresent() {
        let hotkey = HotkeyPreference.Hotkey(
            keyCode: HotkeyPreference.modifierOnlyKeyCode,
            modifiers: [.shift],
            sidedModifiers: [.leftShift, .rightShift]
        )

        XCTAssertEqual(
            HotkeyPreference.displayString(for: hotkey, distinguishModifierSides: true),
            "Shift"
        )
    }

    func testMouseHotkeyPersistsAndLoadsWithModifiers() {
        withRestoredDefaults([
            AppPreferenceKey.hotkeyPreset,
            AppPreferenceKey.hotkeyInputType,
            AppPreferenceKey.hotkeyKeyCode,
            AppPreferenceKey.hotkeyMouseButtonNumber,
            AppPreferenceKey.hotkeyModifiers,
            AppPreferenceKey.hotkeySidedModifiers
        ]) {
            UserDefaults.standard.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)

            HotkeyPreference.save(
                HotkeyPreference.Hotkey(
                    mouseButtonNumber: 4,
                    modifiers: [.command],
                    sidedModifiers: [.rightCommand]
                )
            )

            XCTAssertEqual(
                HotkeyPreference.load(),
                HotkeyPreference.Hotkey(
                    mouseButtonNumber: 4,
                    modifiers: [.command],
                    sidedModifiers: [.rightCommand]
                )
            )
            XCTAssertEqual(
                UserDefaults.standard.string(forKey: AppPreferenceKey.hotkeyInputType),
                HotkeyPreference.Hotkey.Input.Kind.mouseButton.rawValue
            )
            XCTAssertEqual(UserDefaults.standard.integer(forKey: AppPreferenceKey.hotkeyMouseButtonNumber), 4)
        }
    }

    func testMouseHotkeyDisplayStringsUseMouseLabels() {
        XCTAssertEqual(
            HotkeyPreference.displayString(
                for: HotkeyPreference.Hotkey(mouseButtonNumber: 2),
                distinguishModifierSides: false
            ),
            AppLocalization.localizedString("Mouse Middle Button")
        )
        XCTAssertEqual(
            HotkeyPreference.displayString(
                for: HotkeyPreference.Hotkey(
                    mouseButtonNumber: 5,
                    modifiers: [.command],
                    sidedModifiers: []
                ),
                distinguishModifierSides: false
            ),
            "⌘ \(AppLocalization.format("Mouse Button %d", 5))"
        )
    }

    func testMouseMiddleFnShiftPresetPersistsUnifiedHotkeys() {
        withRestoredDefaults([
            AppPreferenceKey.hotkeyPreset,
            AppPreferenceKey.hotkeyInputType,
            AppPreferenceKey.hotkeyKeyCode,
            AppPreferenceKey.hotkeyMouseButtonNumber,
            AppPreferenceKey.hotkeyModifiers,
            AppPreferenceKey.hotkeySidedModifiers,
            AppPreferenceKey.translationHotkeyInputType,
            AppPreferenceKey.translationHotkeyKeyCode,
            AppPreferenceKey.translationHotkeyMouseButtonNumber,
            AppPreferenceKey.translationHotkeyModifiers,
            AppPreferenceKey.translationHotkeySidedModifiers,
            AppPreferenceKey.rewriteHotkeyInputType,
            AppPreferenceKey.rewriteHotkeyKeyCode,
            AppPreferenceKey.rewriteHotkeyMouseButtonNumber,
            AppPreferenceKey.rewriteHotkeyModifiers,
            AppPreferenceKey.rewriteHotkeySidedModifiers,
            AppPreferenceKey.customPasteHotkeyInputType,
            AppPreferenceKey.customPasteHotkeyKeyCode,
            AppPreferenceKey.customPasteHotkeyMouseButtonNumber,
            AppPreferenceKey.customPasteHotkeyModifiers,
            AppPreferenceKey.customPasteHotkeySidedModifiers,
            AppPreferenceKey.hotkeyTriggerMode,
            AppPreferenceKey.rewriteHotkeyActivationMode,
            AppPreferenceKey.transcriptionHotkeyBindings,
            AppPreferenceKey.translationHotkeyBindings,
            AppPreferenceKey.meetingHotkeyBindings,
            AppPreferenceKey.rewriteHotkeyBindings
        ]) {
            let values = HotkeyPreference.applyPreset(.mouseMiddleFnShift)

            XCTAssertEqual(values?.transcription, HotkeyPreference.Hotkey(mouseButtonNumber: 2))
            XCTAssertEqual(values?.rewrite, HotkeyPreference.Hotkey(mouseButtonNumber: 2))
            XCTAssertEqual(values?.translation, HotkeyPreference.Hotkey(keyCode: HotkeyPreference.defaultTranslationKeyCode, modifiers: HotkeyPreference.defaultTranslationModifiers, sidedModifiers: []))
            XCTAssertEqual(values?.triggerMode, .tap)
            XCTAssertEqual(values?.rewriteActivationMode, .doubleTapTranscriptionHotkey)
            XCTAssertEqual(HotkeyPreference.load(), HotkeyPreference.Hotkey(mouseButtonNumber: 2))
            XCTAssertEqual(HotkeyPreference.loadRewriteActivationMode(), .doubleTapTranscriptionHotkey)
            XCTAssertEqual(HotkeyPreference.loadTriggerMode(), .tap)
            XCTAssertEqual(HotkeyPreference.loadTranscriptionBindings().first?.behavior, .tap)
            XCTAssertEqual(HotkeyPreference.loadRewriteBindings().first?.behavior, .doubleTap)
            XCTAssertEqual(HotkeyPreference.loadRewriteBindings().first?.hotkey, HotkeyPreference.Hotkey(mouseButtonNumber: 2))
        }
    }

    func testLegacyScalarHotkeyMigratesToBindingListWithTriggerBehavior() {
        withRestoredDefaults(hotkeyMigrationKeys) {
            let defaults = UserDefaults.standard
            defaults.set(Int(HotkeyPreference.modifierOnlyKeyCode), forKey: AppPreferenceKey.hotkeyKeyCode)
            defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: AppPreferenceKey.hotkeyModifiers)
            defaults.set(0, forKey: AppPreferenceKey.hotkeySidedModifiers)
            defaults.set(HotkeyPreference.TriggerMode.longPress.rawValue, forKey: AppPreferenceKey.hotkeyTriggerMode)
            defaults.removeObject(forKey: AppPreferenceKey.transcriptionHotkeyBindings)

            HotkeyPreference.migrateHotkeyBindingsIfNeeded(defaults: defaults)

            let bindings = HotkeyPreference.loadTranscriptionBindings(defaults: defaults)
            XCTAssertEqual(bindings.count, 1)
            XCTAssertEqual(bindings.first?.behavior, .longPress)
            XCTAssertEqual(bindings.first?.hotkey.sidedModifiers, [.leftCommand])
        }
    }

    func testLegacyRewriteDoubleTapWakeMigratesToTranscriptionDoubleTapBinding() {
        withRestoredDefaults(hotkeyMigrationKeys) {
            let defaults = UserDefaults.standard
            defaults.set(Int(HotkeyPreference.modifierOnlyKeyCode), forKey: AppPreferenceKey.hotkeyKeyCode)
            defaults.set(Int(NSEvent.ModifierFlags([.function, .shift]).rawValue), forKey: AppPreferenceKey.hotkeyModifiers)
            defaults.set(0, forKey: AppPreferenceKey.hotkeySidedModifiers)
            defaults.set(
                HotkeyPreference.RewriteActivationMode.doubleTapTranscriptionHotkey.rawValue,
                forKey: AppPreferenceKey.rewriteHotkeyActivationMode
            )
            defaults.removeObject(forKey: AppPreferenceKey.rewriteHotkeyBindings)

            HotkeyPreference.migrateHotkeyBindingsIfNeeded(defaults: defaults)

            let binding = HotkeyPreference.loadRewriteBindings(defaults: defaults).first
            XCTAssertEqual(binding?.behavior, .doubleTap)
            XCTAssertEqual(binding?.hotkey.modifiers, [.function, .shift])
            XCTAssertEqual(binding?.hotkey.sidedModifiers, [.leftShift])
        }
    }

    func testLegacyModifierWithoutSideMigratesToLeftSide() {
        withRestoredDefaults(hotkeyMigrationKeys) {
            let defaults = UserDefaults.standard
            defaults.set(Int(HotkeyPreference.modifierOnlyKeyCode), forKey: AppPreferenceKey.translationHotkeyKeyCode)
            defaults.set(Int(NSEvent.ModifierFlags([.command, .shift]).rawValue), forKey: AppPreferenceKey.translationHotkeyModifiers)
            defaults.set(0, forKey: AppPreferenceKey.translationHotkeySidedModifiers)
            defaults.removeObject(forKey: AppPreferenceKey.translationHotkeyBindings)

            HotkeyPreference.migrateHotkeyBindingsIfNeeded(defaults: defaults)

            XCTAssertEqual(
                HotkeyPreference.loadTranslationBindings(defaults: defaults).first?.hotkey.sidedModifiers,
                [.leftShift, .leftCommand]
            )
        }
    }

    func testLegacyKeyboardHotkeyDefaultsWhenInputTypeIsMissing() {
        withRestoredDefaults([
            AppPreferenceKey.hotkeyPreset,
            AppPreferenceKey.hotkeyInputType,
            AppPreferenceKey.hotkeyKeyCode,
            AppPreferenceKey.hotkeyMouseButtonNumber,
            AppPreferenceKey.hotkeyModifiers,
            AppPreferenceKey.hotkeySidedModifiers
        ]) {
            let defaults = UserDefaults.standard
            defaults.set(HotkeyPreference.Preset.custom.rawValue, forKey: AppPreferenceKey.hotkeyPreset)
            defaults.removeObject(forKey: AppPreferenceKey.hotkeyInputType)
            defaults.set(Int(UInt16(kVK_ANSI_L)), forKey: AppPreferenceKey.hotkeyKeyCode)
            defaults.set(9, forKey: AppPreferenceKey.hotkeyMouseButtonNumber)
            defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: AppPreferenceKey.hotkeyModifiers)
            defaults.set(0, forKey: AppPreferenceKey.hotkeySidedModifiers)

            XCTAssertEqual(
                HotkeyPreference.load(),
                HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_ANSI_L),
                    modifiers: [.command],
                    sidedModifiers: []
                )
            )
        }
    }

    func testKeyCaptureViewCapturesMouseButtonsWithModifiers() async {
        let view = KeyCaptureView()
        var captured: HotkeyPreference.Hotkey?
        view.onHotkeyCaptured = { captured = $0 }

        view.debugCaptureMouseButtonDownForTests(
            buttonNumber: 4,
            modifiers: [.command, .shift],
            sidedModifiers: [.rightCommand, .leftShift]
        )
        try? await Task.sleep(for: .milliseconds(20))

        XCTAssertEqual(
            captured,
            HotkeyPreference.Hotkey(
                mouseButtonNumber: 4,
                modifiers: [.command, .shift],
                sidedModifiers: [.rightCommand, .leftShift]
            )
        )
    }

    private func restoreDefaultsValue(_ value: Any?, forKey key: String) {
        let defaults = UserDefaults.standard
        if let value {
            defaults.set(value, forKey: key)
        } else {
            defaults.removeObject(forKey: key)
        }
    }

    private func withRestoredDefaults(_ keys: [String], _ body: () -> Void) {
        let defaults = UserDefaults.standard
        let savedValues = Dictionary(uniqueKeysWithValues: keys.map { ($0, defaults.object(forKey: $0)) })
        defer {
            for key in keys {
                restoreDefaultsValue(savedValues[key] ?? nil, forKey: key)
            }
        }
        body()
    }

    private var hotkeyMigrationKeys: [String] {
        [
            AppPreferenceKey.hotkeyInputType,
            AppPreferenceKey.hotkeyKeyCode,
            AppPreferenceKey.hotkeyMouseButtonNumber,
            AppPreferenceKey.hotkeyModifiers,
            AppPreferenceKey.hotkeySidedModifiers,
            AppPreferenceKey.translationHotkeyInputType,
            AppPreferenceKey.translationHotkeyKeyCode,
            AppPreferenceKey.translationHotkeyMouseButtonNumber,
            AppPreferenceKey.translationHotkeyModifiers,
            AppPreferenceKey.translationHotkeySidedModifiers,
            AppPreferenceKey.rewriteHotkeyInputType,
            AppPreferenceKey.rewriteHotkeyKeyCode,
            AppPreferenceKey.rewriteHotkeyMouseButtonNumber,
            AppPreferenceKey.rewriteHotkeyModifiers,
            AppPreferenceKey.rewriteHotkeySidedModifiers,
            AppPreferenceKey.meetingHotkeyInputType,
            AppPreferenceKey.meetingHotkeyKeyCode,
            AppPreferenceKey.meetingHotkeyMouseButtonNumber,
            AppPreferenceKey.meetingHotkeyModifiers,
            AppPreferenceKey.meetingHotkeySidedModifiers,
            AppPreferenceKey.hotkeyTriggerMode,
            AppPreferenceKey.rewriteHotkeyActivationMode,
            AppPreferenceKey.transcriptionHotkeyBindings,
            AppPreferenceKey.translationHotkeyBindings,
            AppPreferenceKey.meetingHotkeyBindings,
            AppPreferenceKey.rewriteHotkeyBindings
        ]
    }
}
