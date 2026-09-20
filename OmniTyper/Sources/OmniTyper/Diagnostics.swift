// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Bounded JSON events containing failure codes and metadata, never user content.
enum Diagnostics {
    private static let byteLimit = 128 * 1024
    private static let lock = NSLock()
    private final class Anchor {}

    static var directory: URL? {
        // Note (Jiaxin Deng): The code's bundle isolates parallel tests from the user's real log.
        guard !Bundle(for: Anchor.self).bundlePath.hasSuffix(".xctest") else {
            return FileManager.default.temporaryDirectory
                .appendingPathComponent("OmniTyperTests/Logs", isDirectory: true)
        }
        return FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Logs/OmniTyper", isDirectory: true)
    }

    static var fileURL: URL? { directory?.appendingPathComponent("diagnostics.log") }

    static func code(of error: Error) -> String {
        (error as? Failure)?.code ?? "unknown"
    }

    static func record(_ event: String, _ details: [String: String] = [:]) {
        var fields = details
        fields["at"] = ISO8601DateFormatter().string(from: Date())
        fields["event"] = event
        guard var line = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]),
              line.count < byteLimit, let directory, let fileURL else { return }
        line.append(0x0A)
        lock.lock()
        defer { lock.unlock() }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            // ponytail: rewrite at most 128 KiB; use rotation if diagnostic volume grows.
            var data = (try? Data(contentsOf: fileURL)) ?? Data()
            data.append(line)
            if data.count > byteLimit {
                let tail = data.suffix(byteLimit)
                let newline = tail.firstIndex(of: 0x0A)!
                data = Data(tail.suffix(from: tail.index(after: newline)))
            }
            try data.write(to: fileURL, options: .atomic)
        } catch {
            // Note (Jiaxin Deng): Diagnostics must never interrupt dictation.
        }
    }
}
