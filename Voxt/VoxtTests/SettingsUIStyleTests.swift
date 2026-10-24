// SettingsUIStyleTests.swift
// Provides Settings UIStyle Tests for Voxt test coverage.

import AppKit
import XCTest
@testable import Voxt

final class SettingsUIStyleTests: XCTestCase {
    func testResolvedSelectWidthShrinksConfiguredWidth() {
        XCTAssertEqual(SettingsUIStyle.resolvedSelectWidth(220), 204)
        XCTAssertEqual(SettingsUIStyle.resolvedSelectWidth(160), 144)
    }

    func testResolvedSelectWidthHasMinimumFloor() {
        XCTAssertEqual(SettingsUIStyle.resolvedSelectWidth(120), 120)
        XCTAssertEqual(SettingsUIStyle.resolvedSelectWidth(80), 120)
        XCTAssertEqual(SettingsUIStyle.resolvedSelectWidth(0), 120)
    }

    func testSettingsMenuInteractionPerformsMenuItemAction() {
        let target = MenuActionTarget()
        let menu = NSMenu()
        let item = NSMenuItem(title: "Soft", action: #selector(MenuActionTarget.handleSelection(_:)), keyEquivalent: "")
        item.target = target
        item.tag = 7
        menu.addItem(item)

        XCTAssertTrue(SettingsMenuInteraction.performSelection(for: item))
        XCTAssertEqual(target.selectedTag, 7)
    }

    func testSettingsMenuInteractionReturnsFalseWithoutMenu() {
        let item = NSMenuItem(title: "Soft", action: nil, keyEquivalent: "")
        XCTAssertFalse(SettingsMenuInteraction.performSelection(for: item))
    }
}

private final class MenuActionTarget: NSObject {
    private(set) var selectedTag: Int?

    @objc
    func handleSelection(_ sender: NSMenuItem) {
        selectedTag = sender.tag
    }
}
