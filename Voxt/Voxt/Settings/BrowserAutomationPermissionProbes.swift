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

    typealias ScriptProbeResult = BrowserAutomationAccess.ProbeResult

    struct BrowserTargetPreflight: Sendable {
        let appPath: String?
        let appNotFoundError: String?
        let isRunning: Bool
    }

    struct BrowserAutomationRequestResult: Sendable {
        let integrityError: String?
        let status: OSStatus
    }

    struct BrowserAutomationTestResult: Sendable {
        let integrityError: String?
        let scriptProbe: ScriptProbeResult?
    }

    nonisolated static func nonPromptingBrowserAutomationState(
        _ target: BrowserAutomationTarget,
        knownAuthorizedBrowserBundleIDs: Set<String>
    ) -> PermissionState {
        guard isApplicationInstalled(bundleID: target.bundleID) else { return .disabled }
        let status = BrowserAutomationAccess.shared.permissionStatus(bundleID: target.bundleID)
        switch status {
        case OSStatus(noErr):
            return .enabled
        case OSStatus(errAEEventNotPermitted), OSStatus(errAEPrivilegeError), OSStatus(errAEEventWouldRequireUserConsent):
            return .disabled
        case OSStatus(procNotFound):
            return knownAuthorizedBrowserBundleIDs.contains(target.bundleID) ? .enabled : .disabled
        default:
            return .onDemand
        }
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
