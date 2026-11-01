import Darwin
import Foundation

/// Stores transport identity and authorized public keys, never the Voxt API token.
/// Access is deferred until Connect; constructing the app does not read the disk.
@MainActor public final class FileConnectorConfigurationStore: ConnectorConfigurationStore {
    private let url: URL
    public init(url: URL) { self.url = url }

    public func load() throws -> ConnectorConfiguration {
        let fd = open(url.path, O_RDONLY | O_NOFOLLOW)
        guard fd >= 0 else {
            if errno == ENOENT { return .init(allowedClients: [], state: nil) }
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        defer { try? handle.close() }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_uid == getuid(),
              info.st_mode & S_IFMT == S_IFREG, info.st_mode & 0o777 == 0o600 else {
            throw CocoaError(.fileReadNoPermission)
        }
        return try JSONDecoder().decode(ConnectorConfiguration.self, from: handle.readToEnd() ?? Data())
    }

    public func save(_ configuration: ConnectorConfiguration) throws {
        let folder = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        let temporary = folder.appending(path: ".identity-" + UUID().uuidString)
        let fd = open(temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        defer { try? handle.close(); try? FileManager.default.removeItem(at: temporary) }
        try handle.write(contentsOf: JSONEncoder().encode(configuration))
        try handle.synchronize()
        try handle.close()
        guard rename(temporary.path, url.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }
}
