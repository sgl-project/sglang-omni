import AppKit
import Carbon
import Foundation

nonisolated final class BrowserAutomationAccess: @unchecked Sendable {
    struct ScriptResult: Sendable {
        let output: String?
        let errorCode: Int?
    }

    struct ProbeResult: Sendable {
        let success: Bool
        let permissionDenied: Bool
        let appNotRunning: Bool
        let lastErrorCode: Int?
    }

    static let shared = BrowserAutomationAccess()
    private let operationLock = NSLock()
    private let queryPermission: (String, Bool) -> OSStatus
    private let executeScript: (String) -> ScriptResult

    init(
        queryPermission: @escaping (String, Bool) -> OSStatus = BrowserAutomationAccess.systemPermissionStatus,
        executeScript: @escaping (String) -> ScriptResult = BrowserAutomationAccess.systemExecuteScript
    ) {
        self.queryPermission = queryPermission
        self.executeScript = executeScript
    }

    func permissionStatus(bundleID: String, askUserIfNeeded: Bool = false) -> OSStatus {
        guard !Task.isCancelled else { return OSStatus(userCanceledErr) }
        // A native consent call can outlive task cancellation. Do not queue another operation behind it.
        guard operationLock.try() else { return OSStatus(errAETimeout) }
        defer { operationLock.unlock() }
        guard !Task.isCancelled else { return OSStatus(userCanceledErr) }
        return queryPermission(bundleID, askUserIfNeeded)
    }

    func testURLRead(bundleID: String, scripts: [String]) -> ProbeResult {
        guard !Task.isCancelled else {
            return ProbeResult(success: false, permissionDenied: false, appNotRunning: false, lastErrorCode: Int(userCanceledErr))
        }
        guard operationLock.try() else {
            return ProbeResult(success: false, permissionDenied: false, appNotRunning: false, lastErrorCode: Int(errAETimeout))
        }
        defer { operationLock.unlock() }
        guard !Task.isCancelled else {
            return ProbeResult(success: false, permissionDenied: false, appNotRunning: false, lastErrorCode: Int(userCanceledErr))
        }
        let status = queryPermission(bundleID, false)
        guard status == noErr else {
            return ProbeResult(
                success: false,
                permissionDenied: status == errAEEventNotPermitted || status == errAEPrivilegeError,
                appNotRunning: status == procNotFound,
                lastErrorCode: Int(status)
            )
        }

        var lastErrorCode: Int?
        for source in scripts {
            guard !Task.isCancelled else { break }
            let result = executeScript(source)
            if let output = result.output?.trimmingCharacters(in: .whitespacesAndNewlines), !output.isEmpty {
                return ProbeResult(success: true, permissionDenied: false, appNotRunning: false, lastErrorCode: nil)
            }
            lastErrorCode = result.errorCode
            switch result.errorCode {
            case Int(errAEEventNotPermitted), Int(errAEPrivilegeError), Int(errAETimeout), Int(procNotFound), Int(userCanceledErr):
                return ProbeResult(
                    success: false,
                    permissionDenied: result.errorCode == Int(errAEEventNotPermitted) || result.errorCode == Int(errAEPrivilegeError),
                    appNotRunning: result.errorCode == Int(procNotFound),
                    lastErrorCode: result.errorCode
                )
            default:
                continue
            }
        }
        return ProbeResult(success: false, permissionDenied: false, appNotRunning: false, lastErrorCode: lastErrorCode)
    }

    private static func systemPermissionStatus(bundleID: String, askUserIfNeeded: Bool) -> OSStatus {
        // Apple documents this API as thread safe and potentially unbounded; callers run off the main thread.
        let descriptor = NSAppleEventDescriptor(bundleIdentifier: bundleID)
        guard let aeDesc = descriptor.aeDesc else { return OSStatus(errAEEventNotPermitted) }
        return AEDeterminePermissionToAutomateTarget(aeDesc, AEEventClass(kAECoreSuite), AEEventID(kAEGetData), askUserIfNeeded)
    }

    private static func systemExecuteScript(_ source: String) -> ScriptResult {
        // Authorization is checked before the short URL-read timeout is applied.
        let wrapped = """
        with timeout of 1 seconds
        \(source)
        end timeout
        """
        var error: NSDictionary?
        let output = NSAppleScript(source: wrapped)?.executeAndReturnError(&error).stringValue
        return ScriptResult(output: output, errorCode: error?[NSAppleScript.errorNumber] as? Int)
    }
}
