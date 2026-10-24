// LaunchPresentationPolicy.swift
// Provides Launch Presentation Policy for app lifecycle and routing.

import AppKit
import Carbon

enum LaunchPresentationPolicy {
    @MainActor
    static func shouldPresentMainWindowOnLaunch() -> Bool {
        return !LaunchSourceDetector.isLaunchedAsLoginItem
    }
}

enum LaunchSourceDetector {
    @MainActor
    static var isLaunchedAsLoginItem: Bool {
        guard let appleEvent = NSAppleEventManager.shared().currentAppleEvent else {
            return false
        }

        return appleEvent.paramDescriptor(forKeyword: AEKeyword(keyAELaunchedAsLogInItem)) != nil
    }
}
