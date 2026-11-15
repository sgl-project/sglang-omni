import ApplicationServices
import VoxtNewHelperKit
@preconcurrency import AVFoundation
import Foundation
@preconcurrency import Speech

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

    static func speechRecognition() -> String {
        switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized: "granted"
        case .denied: "denied"
        case .restricted: "restricted"
        case .notDetermined: "notDetermined"
        @unknown default: "denied"
        }
    }

    /// Shows macOS's Speech Recognition prompt when it has not asked yet
    /// (and waits for the answer); whether it is allowed.
    static func askSpeechRecognition() async -> Bool {
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            _ = await withCheckedContinuation { done in SFSpeechRecognizer.requestAuthorization { done.resume(returning: $0) } }
        }
        return SFSpeechRecognizer.authorizationStatus() == .authorized
    }

    /// The Microphone, Accessibility, Speech Recognition and Screen Recording.
    /// (CGPreflightPostEventAccess is not asked: it stays false in a
    /// process started before the grant, while posting keys works;
    /// AXIsProcessTrusted is the truth for both.)
    static func status() -> PermissionsResult {
        PermissionsResult(microphone: microphone(), accessibility: AXIsProcessTrusted(), speechRecognition: speechRecognition(),
                          screenRecording: CGPreflightScreenCaptureAccess())
    }

    /// Shows macOS's prompt, then reports: the microphone's and Speech
    /// Recognition's wait for the user's answer; Accessibility's adds
    /// VoxtNew to the list in System Settings and answers at once (the
    /// user grants it there).
    static func request(_ permission: String) async throws -> PermissionsResult {
        switch permission {
        case PermissionName.microphone:
            if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
                _ = await AVCaptureDevice.requestAccess(for: .audio)
            }
        case PermissionName.accessibility:
            let key = "AXTrustedCheckOptionPrompt" // kAXTrustedCheckOptionPrompt
            _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
        case PermissionName.speechRecognition:
            _ = await askSpeechRecognition()
        case PermissionName.screenRecording:
            // Asks once and adds Voxt to the list in System Settings; the
            // user grants it there.
            _ = CGRequestScreenCaptureAccess()
        default:
            throw RPCError(Code.badRequest, "Unknown permission \(permission)")
        }
        return status()
    }
}
