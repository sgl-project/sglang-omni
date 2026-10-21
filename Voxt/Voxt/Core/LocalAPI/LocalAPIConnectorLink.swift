// LocalAPIConnectorLink.swift
// Hands the Local API port and Access token to the signed Voxt Connector over a Unix
// socket, after one "Allow" (or none once the user chose Always Allow).

import Foundation
import Security

final class LocalAPIConnectorLink: @unchecked Sendable {
    enum Decision: Sendable { case allowAlways, allowOnce, deny }

    static let defaultConnectorBundleIdentifier = "com.voxt.VoxtConnector"

    /// The Connector's bundle ID: VoxtConnectorBundleIdentifier in Voxt's Info.plist names a renamed
    /// Connector. Anything that is not a bundle ID is ignored, since the value goes into a code
    /// requirement, where a quote would end the identifier string.
    static func connectorBundleIdentifier(info: [String: Any]? = Bundle.main.infoDictionary) -> String {
        guard let identifier = info?["VoxtConnectorBundleIdentifier"] as? String, !identifier.isEmpty,
              identifier.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "-") })
        else { return defaultConnectorBundleIdentifier }
        return identifier
    }

    /// The Connector must be signed by Voxt's own team (read at runtime, so no team ID is
    /// built in), with its bundle ID and the hardened runtime.
    static func connectorRequirement(team: String, identifier: String = connectorBundleIdentifier()) -> String {
        #"anchor apple generic and certificate leaf[subject.OU] = ""# + team + #"" and identifier ""# + identifier + #"""#
    }
    static var defaultPath: String { socketPath() }

    /// Where Voxt Connector finds the socket. In the sandbox the home folder is Voxt's container,
    /// ~/Library/Containers/<bundle ID>/Data, and the socket sits at its top: a socket path holds
    /// at most 103 bytes, too few for Application Support inside the container. The Connector
    /// derives the same path from Voxt's bundle ID, so the link needs no extra entitlement.
    static func socketPath(environment: [String: String] = ProcessInfo.processInfo.environment,
                           home: URL = FileManager.default.homeDirectoryForCurrentUser) -> String {
        environment["APP_SANDBOX_CONTAINER_ID"] != nil
            ? home.appending(path: "connector.sock").path
            : home.appending(path: "Library/Application Support/Voxt/connector.sock").path
    }

    private let path: String
    private let verifyPeer: (Int32) -> Bool
    private let isRemembered: () -> Bool
    private let remember: () -> Void
    private let ask: () async -> Decision
    private let credentials: () -> (port: Int, token: String)?
    private let lock = NSLock()
    private var listener: Int32 = -1

    /// `team` is Voxt's own signing team; nil (ad-hoc or unsigned build) rejects every peer, and
    /// the app doesn't start the link at all, so the Connector falls back to manual entry.
    init(team: String? = LocalAPIConnectorLink.ownTeamIdentifier(), path: String = LocalAPIConnectorLink.defaultPath,
         verifyPeer: ((Int32) -> Bool)? = nil,
         isRemembered: @escaping () -> Bool, remember: @escaping () -> Void,
         ask: @escaping () async -> Decision, credentials: @escaping () -> (port: Int, token: String)?) {
        self.path = path
        if let verifyPeer {
            self.verifyPeer = verifyPeer
        } else if let team {
            let identifier = LocalAPIConnectorLink.connectorBundleIdentifier()
            let requirement = LocalAPIConnectorLink.connectorRequirement(team: team, identifier: identifier)
            self.verifyPeer = { LocalAPIConnectorLink.verifyConnector($0, requirement: requirement) }
        } else {
            self.verifyPeer = { _ in false }
        }
        self.isRemembered = isRemembered
        self.remember = remember
        self.ask = ask
        self.credentials = credentials
    }

    /// Serves one Connector request on an accepted connection, then closes it.
    func handle(_ fd: Int32) async {
        defer { close(fd) }
        var timeout = timeval(tv_sec: 5, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        guard let request = readLine(fd), request.contains(#""op":"connect""#) else { return }
        // No dialog and no token for a process that isn't the signed Connector.
        guard verifyPeer(fd) else { return send(["status": "rejected", "reason": "signature"], to: fd) }
        guard let credentials = credentials() else { return send(["status": "unavailable"], to: fd) }
        func approved(alwaysAllowed: Bool) -> [String: Any] {
            // The Connector connects silently at launch only after Always Allow.
            ["status": "approved", "port": credentials.port, "token": credentials.token, "alwaysAllowed": alwaysAllowed]
        }
        if isRemembered() { return send(approved(alwaysAllowed: true), to: fd) }
        send(["status": "pending"], to: fd)
        switch await ask() {
        case .allowAlways:
            remember()
            send(approved(alwaysAllowed: true), to: fd)
        case .allowOnce:
            send(approved(alwaysAllowed: false), to: fd)
        case .deny:
            send(["status": "denied"], to: fd)
        }
    }

    func start() throws {
        let directory = (path as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        chmod(directory, 0o700)
        unlink(path)
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        guard fd >= 0, bytes.count < MemoryLayout.size(ofValue: address.sun_path) else {
            if fd >= 0 { close(fd) }
            throw POSIXError(.ENAMETOOLONG)
        }
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: bytes + [0]) }
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard bound == 0, chmod(path, 0o600) == 0, listen(fd, 4) == 0 else {
            let code = errno
            close(fd)
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        lock.withLock { listener = fd }
        Thread.detachNewThread { [weak self] in
            while true {
                let client = accept(fd, nil, nil)
                guard client >= 0 else { return }
                guard let self else { close(client); return }
                Task { await self.handle(client) }
            }
        }
    }

    func stop() {
        let fd = lock.withLock { () -> Int32 in let fd = listener; listener = -1; return fd }
        guard fd >= 0 else { return }
        shutdown(fd, SHUT_RDWR)
        close(fd)
        unlink(path)
    }

    private func readLine(_ fd: Int32) -> String? {
        var data = Data()
        var byte: UInt8 = 0
        while data.count < 1024, read(fd, &byte, 1) == 1 {
            if byte == 0x0A { return String(decoding: data, as: UTF8.self) }
            data.append(byte)
        }
        return nil
    }

    private func send(_ fields: [String: Any], to fd: Int32) {
        var message = fields
        message["v"] = 1
        guard var data = try? JSONSerialization.data(withJSONObject: message, options: [.sortedKeys]) else { return }
        data.append(0x0A)
        _ = data.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
    }

    /// The peer must be the Connector signed by our team with the hardened runtime; without
    /// the runtime an injected library passed the signature check in the fixture.
    static func ownTeamIdentifier() -> String? {
        var code: SecCode?
        var staticCode: SecStaticCode?
        var info: CFDictionary?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let code,
              SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode,
              SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess
        else { return nil }
        return (info as? [String: Any])?[kSecCodeInfoTeamIdentifier as String] as? String
    }

    static func verifyConnector(_ fd: Int32, requirement requirementText: String) -> Bool {
        var token = audit_token_t()
        var size = socklen_t(MemoryLayout<audit_token_t>.size)
        guard getsockopt(fd, SOL_LOCAL, LOCAL_PEERTOKEN, &token, &size) == 0 else { return false }
        var requirement: SecRequirement?
        guard SecRequirementCreateWithString(requirementText as CFString, [], &requirement) == errSecSuccess,
              let requirement else { return false }
        let tokenData = withUnsafeBytes(of: token) { Data($0) }
        var code: SecCode?
        var staticCode: SecStaticCode?
        let lookup = SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributeAudit: tokenData] as CFDictionary, [], &code)
        guard lookup == errSecSuccess, let code,
              SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode
        else { return false }
        return acceptsPeer(
            runningCodeValid: SecCodeCheckValidity(code, [], requirement) == errSecSuccess,
            bundleOnDiskValid: SecStaticCodeCheckValidity(staticCode, SecCSFlags(rawValue: kSecCSStrictValidate), requirement) == errSecSuccess,
            hardenedRuntime: hasHardenedRuntime(staticCode))
    }

    /// Every check must pass. The running code alone accepted a Connector whose bundle had a file
    /// added after signing, so its bundle on disk is checked too. In the App Sandbox the look-up of
    /// the peer's code is refused (EPERM) unless that code is somewhere the sandbox may read, which
    /// for an app is /Applications; such a Connector falls back to manual entry.
    static func acceptsPeer(runningCodeValid: Bool, bundleOnDiskValid: Bool, hardenedRuntime: Bool) -> Bool {
        runningCodeValid && bundleOnDiskValid && hardenedRuntime
    }

    /// Without the hardened runtime an injected library passed the signature check in the fixture.
    private static func hasHardenedRuntime(_ staticCode: SecStaticCode) -> Bool {
        var info: CFDictionary?
        guard SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess,
              let flags = (info as? [String: Any])?[kSecCodeInfoFlags as String] as? UInt32 else { return false }
        return flags & 0x10000 != 0
    }
}
