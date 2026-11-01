import Foundation

/// Words the user picked, by dictionary reading, kept on the phone. The
/// converter puts them first next time. One tab-separated line per pick
/// (reading, text, count, last use); other instances reload when the file
/// changes or disappears, so clearing from the app reaches the keyboard.
public final class PinyinLearning: @unchecked Sendable {
    public struct Entry: Equatable, Sendable {
        public var reading: String
        public var text: String
        public var count: Int
        public var lastUsed: Date
    }

    private let fileURL: URL?
    private let capacity: Int
    private let lock = NSLock()
    private var stored: [Entry] = []
    private var loadedStamp: Date?

    /// `fileURL` nil keeps picks in memory only (no shared container).
    public init(fileURL: URL?, capacity: Int = 1000) {
        self.fileURL = fileURL
        self.capacity = capacity
    }

    /// Most recent first.
    public var entries: [Entry] {
        lock.withLock {
            reloadIfChanged()
            return stored
        }
    }

    public func record(reading: String, text: String, now: Date = Date()) {
        guard !reading.isEmpty, !text.isEmpty, !reading.contains("\t"), !text.contains("\t"), !text.contains("\n") else { return }
        lock.withLock {
            reloadIfChanged()
            var entry = Entry(reading: reading, text: text, count: 1, lastUsed: now)
            if let index = stored.firstIndex(where: { $0.reading == reading && $0.text == text }) {
                entry.count = stored[index].count + 1
                stored.remove(at: index)
            }
            stored.insert(entry, at: 0)
            if stored.count > capacity { stored.removeLast(stored.count - capacity) }
            save()
        }
    }

    public func clear() {
        lock.withLock {
            stored = []
            loadedStamp = nil
            if let fileURL { Self.clear(fileURL: fileURL) }
        }
    }

    /// Forgets every pick saved at `fileURL` (Settings › Clear learned words).
    public static func clear(fileURL: URL) {
        try? FileManager.default.removeItem(at: fileURL)
    }

    private func stamp() -> Date? {
        guard let fileURL else { return nil }
        // Not URL.resourceValues: those are cached per URL and miss another process's write.
        return (try? FileManager.default.attributesOfItem(atPath: fileURL.path))?[.modificationDate] as? Date
    }

    private func reloadIfChanged() {
        guard let fileURL else { return }
        let current = stamp()
        guard current != loadedStamp else { return }
        loadedStamp = current
        guard current != nil, let text = try? String(contentsOf: fileURL, encoding: .utf8) else {
            stored = []
            return
        }
        stored = text.split(separator: "\n").compactMap { line in
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard fields.count == 4, let count = Int(fields[2]), let seconds = Double(fields[3]) else { return nil }
            return Entry(reading: String(fields[0]), text: String(fields[1]), count: count, lastUsed: Date(timeIntervalSince1970: seconds))
        }
        .sorted { $0.lastUsed > $1.lastUsed }
    }

    private func save() {
        guard let fileURL else { return }
        let text = stored.map { "\($0.reading)\t\($0.text)\t\($0.count)\t\($0.lastUsed.timeIntervalSince1970)" }.joined(separator: "\n")
        try? FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? Data(text.utf8).write(to: fileURL, options: .atomic)
        loadedStamp = stamp()
    }
}
