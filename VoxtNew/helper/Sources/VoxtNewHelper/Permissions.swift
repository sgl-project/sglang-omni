import ApplicationServices
import VoxtNewHelperKit
@preconcurrency import AVFoundation
import Foundation

/// The permissions the helper's work needs, as the helper sees them. Inside
/// VoxtNew.app macOS checks them against the app (the responsible
/// process): the prompts and the lists in System Settings name VoxtNew.
enum Permissions {
    static func microphone() -> String {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: "granted"
        case .denied: "denied"
        case .restricted: "restricted"
        case .notDetermined: "notDetermined"
        @unknown default: "denied"
        }
    }

    /// The Microphone and Accessibility. (CGPreflightPostEventAccess is not
    /// asked: it stays false in a process started before the grant, while
    /// posting keys works; AXIsProcessTrusted is the truth for both.)
    static func status() -> PermissionsResult {
        PermissionsResult(microphone: microphone(), accessibility: AXIsProcessTrusted())
    }

    /// Shows macOS's prompt, then reports: the microphone's waits for the
    /// user's answer; Accessibility's adds VoxtNew to the list in System
    /// Settings and answers at once (the user grants it there).
    static func request(_ permission: String) async throws -> PermissionsResult {
        switch permission {
        case PermissionName.microphone:
            if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
                _ = await AVCaptureDevice.requestAccess(for: .audio)
            }
        case PermissionName.accessibility:
            let key = "AXTrustedCheckOptionPrompt" // kAXTrustedCheckOptionPrompt
            _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
        default:
            throw RPCError(Code.badRequest, "Unknown permission \(permission)")
        }
        return status()
    }
}
