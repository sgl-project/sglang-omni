import CryptoKit
import Foundation

/// A data file a layout ships, checked against the size and SHA256 recorded
/// when it was generated (each layout's manifest). Check once, before the
/// engine maps the file.
public struct VerifiedDictionary: Sendable {
    public enum Failure: Error, Equatable {
        case missing
        /// Not a readable regular file, or not the recorded bytes.
        case invalid
    }

    public let url: URL

    public init(url: URL, byteCount: Int, sha256: String) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw Failure.missing }
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey, .isReadableKey]),
              values.isRegularFile == true, values.isReadable == true, values.fileSize == byteCount,
              let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sha256.lowercased() else {
            throw Failure.invalid
        }
        self.url = url
    }
}
