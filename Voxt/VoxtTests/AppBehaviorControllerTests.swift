// AppBehaviorControllerTests.swift
// Provides App Behavior Controller Tests for Voxt test coverage.

import XCTest
import AppKit
@testable import Voxt

final class AppBehaviorControllerTests: XCTestCase {
    func testResolvedActivationPolicyUsesRegularWhenDockIconIsVisible() {
        XCTAssertEqual(
            AppBehaviorController.resolvedActivationPolicy(
                showInDock: true,
                mainWindowVisible: false
            ),
            .regular
        )
        XCTAssertEqual(
            AppBehaviorController.resolvedActivationPolicy(
                showInDock: true,
                mainWindowVisible: true
            ),
            .regular
        )
    }

    func testResolvedActivationPolicyKeepsVisibleMainWindowRegularWhenDockIconHidden() {
        XCTAssertEqual(
            AppBehaviorController.resolvedActivationPolicy(
                showInDock: false,
                mainWindowVisible: true
            ),
            .regular
        )
    }

    func testResolvedActivationPolicyUsesAccessoryWhenDockIconHiddenAndMainWindowHidden() {
        XCTAssertEqual(
            AppBehaviorController.resolvedActivationPolicy(
                showInDock: false,
                mainWindowVisible: false
            ),
            .accessory
        )
    }
}
