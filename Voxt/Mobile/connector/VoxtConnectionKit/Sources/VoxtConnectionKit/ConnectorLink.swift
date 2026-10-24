import Foundation
import Security

/// What ConnectorAppModel needs to connect without the user typing a port or token.
public protocol ConnectorAuthorizing: Sendable {
    func authorize() async throws -> ConnectorGrant
}

/// Voxt's answer. `alwaysAllowed` is true when the user chose Always Allow, so the Connector
/// may connect silently at launch; Allow Once (or an older Voxt) gives false.
public struct ConnectorGrant: Sendable, Equatable {
    public var port: Int
    public var token: String
    public var alwaysAllowed: Bool
    public init(port: Int, token: String, alwaysAllowed: Bool) {
        self.port = port
        self.token = token
        self.alwaysAllowed = alwaysAllowed
    }
}

/// Asks Voxt for its Local API port and Access token over a Unix socket. Each side checks
/// the other's code signature first; Voxt asks the user once ("Allow Voxt Connector to use
/// Voxt?").
public struct ConnectorLink: ConnectorAuthorizing {
    public enum Failure: Error, Equatable { case voxtNotRunning, untrustedVoxt, denied, rejectedBySignature, invalidReply }

    /// Voxt must be signed by the same team as this Connector (read at runtime, so no team ID
    /// is built in), with one of its bundle IDs (release and development by default) and the
    /// hardened runtime.
    public static func voxtRequirement(team: String,
                                       identifiers: [String] = ConnectorIdentity.voxtBundleIdentifiers()) -> String {
        let names = identifiers.map { #"identifier ""# + $0 + #"""# }.joined(separator: " or ")
        return #"anchor apple generic and certificate leaf[subject.OU] = ""# + team + #"" and ("# + names + ")"
    }
    /// Where Voxt's socket may be: at the top of each trusted Voxt's sandbox container, then where
    /// an unsandboxed Voxt puts it. A sandboxed Voxt can only write inside its container, and a
    /// socket path holds at most 103 bytes, too few for Application Support inside it.
    public static func socketPaths(identifiers: [String] = ConnectorIdentity.voxtBundleIdentifiers(),
                                   home: URL = FileManager.default.homeDirectoryForCurrentUser) -> [String] {
        identifiers.map { home.appending(path: "Library/Containers/\($0)/Data/connector.sock").path }
            + [home.appending(path: "Library/Application Support/Voxt/connector.sock").path]
    }

    private let connect: @Sendable () throws -> Int32
    private let verifyPeer: (@Sendable (Int32) -> Bool)?
    private let replyTimeout: Int

    /// `team` is this Connector's own signing team; nil (ad-hoc or unsigned build) means
    /// automatic connect can't verify Voxt, so the user enters the port and token by hand.
    public init(team: String? = PeerSignature.ownTeamIdentifier(),
                connect: @escaping @Sendable () throws -> Int32 = { try ConnectorLink.connectToVoxt() },
                verifyPeer: (@Sendable (Int32) -> Bool)? = nil,
                replyTimeout: Int = 130) {
        self.connect = connect
        if let verifyPeer {
            self.verifyPeer = verifyPeer
        } else if let team {
            let requirement = ConnectorLink.voxtRequirement(team: team)
            self.verifyPeer = { @Sendable fd in PeerSignature.verify(fd, requirement: requirement) }
        } else {
            self.verifyPeer = nil
        }
        self.replyTimeout = replyTimeout
    }

    /// Cancelling closes the socket at once.
    public func authorize() async throws -> ConnectorGrant {
        guard let verifyPeer else { throw Failure.rejectedBySignature }
        let replyTimeout = replyTimeout
        let fd = try connect()
        let work = Task.detached { () throws -> ConnectorGrant in
            defer { close(fd) }
            // Nothing is sent to a process that isn't the signed Voxt.
            guard verifyPeer(fd) else { throw Failure.untrustedVoxt }
            // Voxt may wait for the user's click; give up after the dialog's own expiry.
            var timeout = timeval(tv_sec: replyTimeout, tv_usec: 0)
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            let request = Data(#"{"v":1,"op":"connect"}"#.utf8) + Data([0x0A])
            guard request.withUnsafeBytes({ write(fd, $0.baseAddress, $0.count) }) == request.count else { throw Failure.voxtNotRunning }
            var buffer = Data()
            while true {
                if let newline = buffer.firstIndex(of: 0x0A) {
                    let line = buffer[buffer.startIndex..<newline]
                    buffer = Data(buffer[buffer.index(after: newline)...])
                    struct Reply: Decodable { var status: String; var port: Int?; var token: String?; var alwaysAllowed: Bool? }
                    guard let reply = try? JSONDecoder().decode(Reply.self, from: Data(line)) else { throw Failure.invalidReply }
                    switch reply.status {
                    case "pending": continue
                    case "approved":
                        guard let port = reply.port, (1024...65535).contains(port), let token = reply.token, !token.isEmpty else {
                            throw Failure.invalidReply
                        }
                        return ConnectorGrant(port: port, token: token, alwaysAllowed: reply.alwaysAllowed ?? false)
                    case "denied": throw Failure.denied
                    case "rejected": throw Failure.rejectedBySignature
                    default: throw Failure.invalidReply
                    }
                }
                var chunk = [UInt8](repeating: 0, count: 4096)
                let count = read(fd, &chunk, chunk.count)
                guard count > 0 else { throw Failure.invalidReply }
                buffer.append(contentsOf: chunk[0..<count])
            }
        }
        return try await withTaskCancellationHandler {
            do { return try await work.value } catch {
                if Task.isCancelled { throw CancellationError() }
                throw error
            }
        } onCancel: {
            // Unblocks the read; the work task then closes the socket.
            shutdown(fd, SHUT_RDWR)
        }
    }

    /// Connects to the first path where a Voxt answers; a quit Voxt can leave its socket file behind.
    public static func connectToVoxt(paths: [String] = socketPaths()) throws -> Int32 {
        for path in paths {
            if let fd = try? connectToVoxt(path: path) { return fd }
        }
        throw Failure.voxtNotRunning
    }

    public static func connectToVoxt(path: String) throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw Failure.voxtNotRunning }
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        guard bytes.count < MemoryLayout.size(ofValue: address.sun_path) else { close(fd); throw Failure.voxtNotRunning }
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: bytes + [0]) }
        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard connected == 0 else { close(fd); throw Failure.voxtNotRunning }
        return fd
    }
}

/// Checks the process at the other end of a Unix socket against a code requirement, and
/// requires the hardened runtime: without it an injected library passes the signature
/// check.
public enum PeerSignature {
    /// This process's signing team, or nil for ad-hoc and unsigned builds.
    public static func ownTeamIdentifier() -> String? {
        var code: SecCode?
        var staticCode: SecStaticCode?
        var info: CFDictionary?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let code,
              SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode,
              SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess
        else { return nil }
        return (info as? [String: Any])?[kSecCodeInfoTeamIdentifier as String] as? String
    }

    public static func verify(_ fd: Int32, requirement text: String) -> Bool {
        var token = audit_token_t()
        var size = socklen_t(MemoryLayout<audit_token_t>.size)
        guard getsockopt(fd, SOL_LOCAL, LOCAL_PEERTOKEN, &token, &size) == 0 else { return false }
        let tokenData = withUnsafeBytes(of: token) { Data($0) }
        var code: SecCode?
        var requirement: SecRequirement?
        guard SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributeAudit: tokenData] as CFDictionary, [], &code) == errSecSuccess,
              let code,
              SecRequirementCreateWithString(text as CFString, [], &requirement) == errSecSuccess, let requirement,
              SecCodeCheckValidity(code, [], requirement) == errSecSuccess else { return false }
        var staticCode: SecStaticCode?
        var info: CFDictionary?
        guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode,
              SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess,
              let flags = (info as? [String: Any])?[kSecCodeInfoFlags as String] as? UInt32 else { return false }
        return flags & 0x10000 != 0
    }
}
