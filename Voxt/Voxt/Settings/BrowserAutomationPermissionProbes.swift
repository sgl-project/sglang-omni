import Foundation
import SwiftUI
import AppKit
import ApplicationServices
import Carbon
import Security

// Browser probe inputs/results and nonisolated native checks. The view retains
// permission state, task cancellation and the explicit Request/Test entry points.
extension PermissionsSettingsView {
    enum PermissionState: Equatable, Sendable {
        case enabled
        case disabled
        case onDemand

        var titleKey: LocalizedStringKey {
            switch self {
            case .enabled: return "Enabled"
            case .disabled: return "Disabled"
            case .onDemand: return "On Demand"
            }
        }

        var tint: Color {
            switch self {
            case .enabled: return .green
            case .disabled: return .orange
            case .onDemand: return .secondary
            }
        }
    }

    struct BrowserAutomationTarget: Identifiable, Hashable, Sendable {
        let bundleID: String
        let displayName: String
        let scripts: [String]
        let isCustom: Bool

        var id: String { bundleID }
    }

    struct StoredCustomBrowser: Codable, Hashable, Sendable {
        let bundleID: String
        let displayName: String
    }

    struct ScriptProbeResult: Sendable {
        let success: Bool
        let permissionDenied: Bool
        let appNotRunning: Bool
        let lastErrorCode: Int?
    }

    struct BrowserTargetPreflight: Sendable {
        let appPath: String?
        let appNotFoundError: String?
        let isRunning: Bool
    }


    struct BrowserAutomationRequestResult: Sendable {
        let integrityError: String?
        let enabled: Bool
        let permissionGranted: Bool
        let scriptProbe: ScriptProbeResult?
        let failureMessage: String?
    }


    struct BrowserAutomationTestResult: Sendable {
        let integrityError: String?
        let scriptProbe: ScriptProbeResult?
    }

    nonisolated static func nonPromptingBrowserAutomationState(
        _ target: BrowserAutomationTarget,
        knownAuthorizedBrowserBundleIDs: Set<String>
    ) -> PermissionState {
        if target.isCustom {
            // Custom browsers are less consistent than Safari / Chrome: some do
            // not answer the low-level permission API reliably, especially when
            // the app is not running. For them we combine three signals:
            // installation, running state, and remembered authorization.
            let isRememberedAuthorized = knownAuthorizedBrowserBundleIDs.contains(target.bundleID)
            guard isApplicationInstalled(bundleID: target.bundleID) else {
                return .disabled
            }
            guard isApplicationRunning(bundleID: target.bundleID) else {
                return isRememberedAuthorized ? .enabled : .disabled
            }

            let permissionProbe = runAutomationPermissionProbe(bundleID: target.bundleID)
            if permissionProbe.success {
                return .enabled
            }
            if permissionProbe.permissionDenied {
                return .disabled
            }
            return isRememberedAuthorized ? .enabled : .disabled
        }

        let status = automationPermissionStatus(for: target.bundleID, askUserIfNeeded: false)
        if status == noErr {
            return .enabled
        }
        if status == errAEEventNotPermitted || status == errAEPrivilegeError {
            return .disabled
        }
        if knownAuthorizedBrowserBundleIDs.contains(target.bundleID) {
            return .enabled
        }
        return .disabled
    }

    nonisolated static func runAutomationPermissionProbe(bundleID: String) -> ScriptProbeResult {
        runAppleScriptCandidates([
            "tell application id \"\(bundleID)\" to get name"
        ])
    }

    nonisolated private static func automationPermissionStatus(for bundleID: String, askUserIfNeeded: Bool) -> OSStatus {
        let descriptor = NSAppleEventDescriptor(bundleIdentifier: bundleID)
        guard let aeDesc = descriptor.aeDesc else {
            return OSStatus(errAEEventNotPermitted)
        }

        return AEDeterminePermissionToAutomateTarget(
            aeDesc,
            AEEventClass(typeWildCard),
            AEEventID(typeWildCard),
            askUserIfNeeded
        )
    }


    nonisolated static func runAppleScriptCandidates(_ scripts: [String]) -> ScriptProbeResult {
        var sawPermissionDenied = false
        var sawAppNotRunning = false
        var lastErrorCode: Int?

        for source in scripts {
            if Task.isCancelled {
                return ScriptProbeResult(
                    success: false,
                    permissionDenied: sawPermissionDenied,
                    appNotRunning: sawAppNotRunning,
                    lastErrorCode: lastErrorCode
                )
            }
            var error: NSDictionary?
            let wrapped = """
            with timeout of 1 seconds
            \(source)
            end timeout
            """
            let script = NSAppleScript(source: wrapped)
            let result = script?.executeAndReturnError(&error)
            if let output = result?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines), !output.isEmpty {
                return ScriptProbeResult(success: true, permissionDenied: false, appNotRunning: false, lastErrorCode: nil)
            }

            let code = error?[NSAppleScript.errorNumber] as? Int
            lastErrorCode = code
            if code == -1743 || code == -10004 {
                sawPermissionDenied = true
            }
            if code == -600 {
                sawAppNotRunning = true
            }
        }

        return ScriptProbeResult(
            success: false,
            permissionDenied: sawPermissionDenied,
            appNotRunning: sawAppNotRunning,
            lastErrorCode: lastErrorCode
        )
    }


    nonisolated static func isApplicationRunning(bundleID: String) -> Bool {
        NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)
            .contains(where: { !$0.isTerminated })
    }

    nonisolated private static func isApplicationInstalled(bundleID: String) -> Bool {
        NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) != nil
    }

    nonisolated static func browserTargetSignatureIntegrityError(appPath: String?) -> String? {
        guard let appPath else { return nil }

        var staticCode: SecStaticCode?
        let appURL = URL(fileURLWithPath: appPath)
        let createStatus = SecStaticCodeCreateWithPath(appURL as CFURL, [], &staticCode)
        guard createStatus == errSecSuccess, let staticCode else {
            return MainActorSync.run {
                AppLocalization.localizedString("Browser app signature could not be verified. Reinstall or update the browser, then request authorization again.")
            }
        }

        let checkStatus = SecStaticCodeCheckValidity(staticCode, SecCSFlags(rawValue: kSecCSStrictValidate), nil)
        guard checkStatus == errSecSuccess else {
            return MainActorSync.run {
                AppLocalization.localizedString("Browser app signature is invalid. Reinstall or update the browser, then request authorization again.")
            }
        }

        return nil
    }
}
