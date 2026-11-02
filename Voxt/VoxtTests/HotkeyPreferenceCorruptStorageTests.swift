import AppKit
import XCTest
@testable import Voxt

final class HotkeyPreferenceCorruptStorageTests: XCTestCase {
    func testOutOfRangeLegacyKeyCodesFallBackWithoutTrapping() throws {
        for code in [-1, 65_536, Int.max, Int.min] {
            try withDefaults { defaults in
                defaults.set(code, forKey: AppPreferenceKey.hotkeyKeyCode)
                defaults.set(Int(HotkeyPreference.defaultModifiers.rawValue), forKey: AppPreferenceKey.hotkeyModifiers)
                let binding = try migrate(defaults)
                XCTAssertEqual(binding.hotkey.keyCode, HotkeyPreference.defaultKeyCode)
                XCTAssertEqual(binding.hotkey.modifiers, HotkeyPreference.defaultModifiers)
            }
        }
    }

    func testNegativeLegacyModifierBitsFallBackWithoutTrapping() throws {
        try withDefaults { defaults in
            defaults.set(Int(HotkeyPreference.modifierOnlyKeyCode), forKey: AppPreferenceKey.hotkeyKeyCode)
            defaults.set(-1, forKey: AppPreferenceKey.hotkeyModifiers)
            XCTAssertEqual(try migrate(defaults).hotkey.modifiers, HotkeyPreference.defaultModifiers)
        }
    }

    func testValidLegacyKeyAndSidedModifiersArePreserved() throws {
        try withDefaults { defaults in
            defaults.set(0, forKey: AppPreferenceKey.hotkeyKeyCode)
            defaults.set(Int(NSEvent.ModifierFlags.command.rawValue), forKey: AppPreferenceKey.hotkeyModifiers)
            defaults.set(SidedModifierFlags.rightCommand.rawValue, forKey: AppPreferenceKey.hotkeySidedModifiers)
            let binding = try migrate(defaults)
            XCTAssertEqual(binding.hotkey.keyCode, 0)
            XCTAssertEqual(binding.hotkey.modifiers, [.command])
            XCTAssertEqual(binding.hotkey.sidedModifiers, [.rightCommand])
        }
    }

    private func migrate(_ defaults: UserDefaults) throws -> HotkeyPreference.HotkeyBinding {
        HotkeyPreference.migrateHotkeyBindingsIfNeeded(defaults: defaults)
        let data = try XCTUnwrap(defaults.data(forKey: AppPreferenceKey.transcriptionHotkeyBindings))
        return try XCTUnwrap(JSONDecoder().decode([HotkeyPreference.HotkeyBinding].self, from: data).first)
    }

    private func withDefaults(_ body: (UserDefaults) throws -> Void) rethrows {
        let name = UUID().uuidString
        let defaults = TestDoubles.makeUserDefaults(testName: name)
        defer { defaults.removePersistentDomain(forName: "VoxtTests.\(name)") }
        try body(defaults)
    }
}
