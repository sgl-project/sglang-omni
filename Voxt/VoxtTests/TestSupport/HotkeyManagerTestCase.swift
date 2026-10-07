// Shared defaults restoration and event construction for hotkey behavior suites.
import XCTest
import AppKit
import Carbon
import ApplicationServices
import IOKit.hidsystem
@testable import Voxt

@MainActor
class HotkeyManagerTestCase: XCTestCase {
    private static var retainedManagers: [HotkeyManager] = []
    private let managedDefaultKeys = [
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
        AppPreferenceKey.rewriteHotkeyActivationMode,
        AppPreferenceKey.meetingHotkeyInputType,
        AppPreferenceKey.meetingHotkeyKeyCode,
        AppPreferenceKey.meetingHotkeyMouseButtonNumber,
        AppPreferenceKey.meetingHotkeyModifiers,
        AppPreferenceKey.meetingHotkeySidedModifiers,
        AppPreferenceKey.customPasteHotkeyEnabled,
        AppPreferenceKey.customPasteHotkeyInputType,
        AppPreferenceKey.customPasteHotkeyKeyCode,
        AppPreferenceKey.customPasteHotkeyMouseButtonNumber,
        AppPreferenceKey.customPasteHotkeyModifiers,
        AppPreferenceKey.customPasteHotkeySidedModifiers,
        AppPreferenceKey.transcriptionHotkeyBindings,
        AppPreferenceKey.translationHotkeyBindings,
        AppPreferenceKey.meetingHotkeyBindings,
        AppPreferenceKey.rewriteHotkeyBindings,
        AppPreferenceKey.noteHotkeyBindings,
        AppPreferenceKey.hotkeyTriggerMode,
        AppPreferenceKey.hotkeyDistinguishModifierSides,
        AppPreferenceKey.hotkeyPreset,
        AppPreferenceKey.hotkeyCaptureInProgress
    ]

    private var savedDefaults: [String: Any] = [:]
    private var missingDefaultKeys = Set<String>()

    override func setUp() {
        super.setUp()

        let defaults = UserDefaults.standard
        savedDefaults = [:]
        missingDefaultKeys = []

        for key in managedDefaultKeys {
            if let value = defaults.object(forKey: key) {
                savedDefaults[key] = value
            } else {
                missingDefaultKeys.insert(key)
            }
        }

        managedDefaultKeys.forEach { defaults.removeObject(forKey: $0) }
        HotkeyPreference.registerDefaults()
        defaults.set(HotkeyPreference.TriggerMode.tap.rawValue, forKey: AppPreferenceKey.hotkeyTriggerMode)
        defaults.set(false, forKey: AppPreferenceKey.hotkeyCaptureInProgress)
        HotkeyPreference.saveNoteBindings([
            .init(
                hotkey: HotkeyPreference.Hotkey(
                    keyCode: UInt16(kVK_F20),
                    modifiers: [],
                    sidedModifiers: []
                ),
                behavior: .tap
            )
        ])
    }

    override func tearDown() {
        let defaults = UserDefaults.standard

        for key in managedDefaultKeys {
            if let value = savedDefaults[key] {
                defaults.set(value, forKey: key)
            } else if missingDefaultKeys.contains(key) {
                defaults.removeObject(forKey: key)
            }
        }

        savedDefaults = [:]
        missingDefaultKeys = []
        super.tearDown()
    }

    func makeManager() -> HotkeyManager {
        let manager = HotkeyManager()
        Self.retainedManagers.append(manager)
        return manager
    }

    func voxtInjectedEventSourceUserData() -> Int64 {
        guard let source = CGEventSource(stateID: .hidSystemState),
              let event = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_Return), keyDown: true)
        else {
            XCTFail("Unable to create CGEvent for injected event marker test.")
            return 0
        }
        HotkeyEventSupport.markAsVoxtInjected(event)
        return event.getIntegerValueField(.eventSourceUserData)
    }

    func combinedFlags(_ flags: CGEventFlags...) -> CGEventFlags {
        flags.reduce([]) { partialResult, next in
            partialResult.union(next)
        }
    }

    func commandFlags(for side: SidedModifierFlags) -> CGEventFlags {
        switch side {
        case .leftCommand:
            return CGEventFlags(rawValue: UInt64(NX_COMMANDMASK | NX_DEVICELCMDKEYMASK))
        case .rightCommand:
            return CGEventFlags(rawValue: UInt64(NX_COMMANDMASK | NX_DEVICERCMDKEYMASK))
        default:
            XCTFail("Unsupported command side \(side)")
            return []
        }
    }

    func shiftFlags(for side: SidedModifierFlags) -> CGEventFlags {
        switch side {
        case .leftShift:
            return CGEventFlags(rawValue: UInt64(NX_SHIFTMASK | NX_DEVICELSHIFTKEYMASK))
        case .rightShift:
            return CGEventFlags(rawValue: UInt64(NX_SHIFTMASK | NX_DEVICERSHIFTKEYMASK))
        default:
            XCTFail("Unsupported shift side \(side)")
            return []
        }
    }
}
