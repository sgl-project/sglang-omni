import Foundation

struct ReplayError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

enum JSONLine {
    static func encode(_ object: [String: Any]) -> Data {
        // note (luojiaxuan): Every value written is a string, number, bool, NSNull, array or dictionary.
        var data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
        data.append(0x0A)
        return data
    }

    static func decode(_ line: Data) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: line)) as? [String: Any]
    }

    /// note (luojiaxuan): The JSON string literal for `text` (used in single-line socket replies).
    static func quote(_ text: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: [text], options: [.withoutEscapingSlashes])
        let array = String(decoding: data, as: UTF8.self)
        return String(array.dropFirst().dropLast())
    }
}

/// note (luojiaxuan): Appends one JSON object per line. Each line goes out in a single write(2) on an
/// O_APPEND descriptor, so a concurrent reader sees lines in order and at most one
/// incomplete trailing line.
final class LineAppender {
    private let fd: Int32
    private let lock = NSLock()

    init(path: String) throws {
        fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { throw ReplayError("cannot open \(path): \(String(cString: strerror(errno)))") }
    }

    deinit { close(fd) }

    func append(_ object: [String: Any]) {
        let data = JSONLine.encode(object)
        lock.lock()
        defer { lock.unlock() }
        data.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                let n = write(fd, buffer.baseAddress! + offset, buffer.count - offset)
                if n < 0 {
                    if errno == EINTR { continue }
                    fatalError("write failed: \(String(cString: strerror(errno)))")
                }
                offset += n
            }
        }
    }
}

/// note (luojiaxuan): Reads the JSON lines appended to a file since the last call, keeping an
/// incomplete trailing line for the next call.
final class LineTail {
    private let handle: FileHandle
    private var pending = Data()

    /// note (luojiaxuan): Starts at the current end of the file, so lines written before this
    /// object existed are never returned.
    init(path: String) throws {
        guard let handle = FileHandle(forReadingAtPath: path) else { throw ReplayError("cannot read \(path)") }
        self.handle = handle
        handle.seekToEndOfFile()
    }

    func readNew() -> [[String: Any]] {
        pending.append(handle.readDataToEndOfFile())
        var objects: [[String: Any]] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            let line = pending[pending.startIndex..<newline]
            pending.removeSubrange(pending.startIndex...newline)
            if let object = JSONLine.decode(Data(line)) { objects.append(object) }
        }
        return objects
    }
}

func readJSONLines(path: String) throws -> [[String: Any]] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    var rows: [[String: Any]] = []
    for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }
        guard let row = JSONLine.decode(Data(trimmed.utf8)) else {
            throw ReplayError("\(path):\(index + 1): not a JSON object")
        }
        rows.append(row)
    }
    return rows
}
