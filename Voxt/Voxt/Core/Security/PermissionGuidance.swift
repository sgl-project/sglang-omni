// PermissionGuidance.swift
// Provides Permission Guidance for permissions and secure storage.

import AppKit
import PermissionFlow
import SystemSettingsKit

@MainActor
enum PermissionGuidance {
    private static let controller = PermissionFlow.makeController(
        configuration: .init(
            requiredAppURLs: [Bundle.main.bundleURL],
            promptForAccessibilityTrust: false
        )
    )

    static func openSettings(for permission: SettingsPermissionKind) {
        openSettings(for: target(for: permission))
    }

    static func openSettings(for permission: OnboardingContextualPermission) {
        openSettings(for: target(for: permission))
    }

    static func openBrowserAutomationSettings() {
        _ = SystemSettings.open(.privacy(anchor: .privacyAutomation))
    }

    static func openSystemAudioSettings() {
        _ = SystemSettings.open(.privacy(anchor: .privacyAudioCapture))
    }

    private static func openSettings(for target: Target) {
        switch target {
        case .pane(let pane):
            controller.authorize(
                pane: pane,
                suggestedAppURLs: [Bundle.main.bundleURL],
                sourceFrameInScreen: clickSourceFrameInScreen()
            )
        case .destination(let destination):
            _ = SystemSettings.open(destination)
        }
    }

    private static func target(for permission: SettingsPermissionKind) -> Target {
        switch permission {
        case .microphone:
            return .pane(.microphone)
        case .speechRecognition:
            return .destination(.privacy(anchor: .privacySpeechRecognition))
        case .accessibility:
            return .pane(.accessibility)
        case .reminders:
            return .destination(.privacy(anchor: .privacyReminders))
        }
    }

    private static func target(for permission: OnboardingContextualPermission) -> Target {
        switch permission {
        case .microphone:
            return .pane(.microphone)
        case .speechRecognition:
            return .destination(.privacy(anchor: .privacySpeechRecognition))
        case .accessibility:
            return .pane(.accessibility)
        }
    }

    private static func clickSourceFrameInScreen() -> CGRect {
        let mouseLocation = NSEvent.mouseLocation
        return CGRect(x: mouseLocation.x - 16, y: mouseLocation.y - 16, width: 32, height: 32)
    }

    private enum Target {
        case pane(PermissionFlowPane)
        case destination(SystemSettingsDestination)
    }
}
