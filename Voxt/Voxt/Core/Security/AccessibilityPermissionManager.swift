// AccessibilityPermissionManager.swift
// Provides Accessibility Permission Manager for permissions and secure storage.

import Foundation
import AppKit
import ApplicationServices

nonisolated enum AccessibilityPermissionManager {
    static func isTrusted() -> Bool {
        AXIsProcessTrusted()
    }

    @discardableResult
    static func request(prompt: Bool) -> Bool {
        if prompt {
            let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
            _ = AXIsProcessTrustedWithOptions(options)
        } else {
            _ = AXIsProcessTrusted()
        }

        // Perform a benign AX query so macOS registers the app against the
        // accessibility service immediately instead of waiting for a later AX read.
        primeAccessibilityRegistration()
        return AXIsProcessTrusted()
    }

    private static func primeAccessibilityRegistration() {
        let systemWide = AXUIElementCreateSystemWide()
        var focusedApp: CFTypeRef?
        let focusedAppStatus = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedApplicationAttribute as CFString,
            &focusedApp
        )

        guard
            focusedAppStatus == .success,
            let runningApp = NSWorkspace.shared.frontmostApplication
        else {
            return
        }

        let appElement = AXUIElementCreateApplication(runningApp.processIdentifier)
        var focusedWindow: CFTypeRef?
        _ = AXUIElementCopyAttributeValue(
            appElement,
            kAXFocusedWindowAttribute as CFString,
            &focusedWindow
        )
    }
}

nonisolated enum EventListeningPermissionManager {
    struct Status {
        let accessibilityGranted: Bool

        var description: String {
            let accessibility = accessibilityGranted ? "on" : "off"
            return "permissions: accessibility=\(accessibility)"
        }
    }

    static func status() -> Status {
        Status(accessibilityGranted: AccessibilityPermissionManager.isTrusted())
    }
}

nonisolated extension CGEventTapLocation {
    var debugName: String {
        switch self {
        case .cghidEventTap:
            return "cghidEventTap"
        case .cgSessionEventTap:
            return "cgSessionEventTap"
        case .cgAnnotatedSessionEventTap:
            return "cgAnnotatedSessionEventTap"
        @unknown default:
            return "unknown"
        }
    }
}
