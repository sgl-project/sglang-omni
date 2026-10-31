import Foundation

/// Host app supplies app-private Keychain storage. Read/write failures propagate;
/// silently replacing a stored identity would break the Mac's allowlist.
@MainActor public final class TailcatIdentityStore {
    private let read: () throws -> String?
    private let write: (String) throws -> Void
    private let generate: () throws -> TailcatIdentity

    public init(read: @escaping () throws -> String?, write: @escaping (String) throws -> Void,
                generate: @escaping () throws -> TailcatIdentity) {
        self.read = read
        self.write = write
        self.generate = generate
    }

    public func identity() throws -> TailcatIdentity {
        if let stored = try read() {
            return try NativeResponse.identity(from: Data(stored.utf8))
        }
        let identity = try generate()
        let data = try JSONEncoder().encode(identity)
        _ = try NativeResponse.identity(from: data)
        try write(String(decoding: data, as: UTF8.self))
        return identity
    }
}
