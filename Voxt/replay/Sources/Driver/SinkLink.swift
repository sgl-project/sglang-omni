import Foundation

/// note (luojiaxuan): One line of the sink's log.
struct SinkEvent {
    enum Kind: String { case start, text, clear, mark, focus }

    let kind: Kind
    let t: UInt64
    let text: String?
    let label: String?
    let key: Bool?

    init?(_ object: [String: Any]) {
        guard let name = object["event"] as? String, let kind = Kind(rawValue: name),
              let t = (object["t_mach_continuous_ns"] as? NSNumber)?.uint64Value else { return nil }
        self.kind = kind
        self.t = t
        text = object["text"] as? String
        label = object["label"] as? String
        key = object["key"] as? Bool
    }
}

/// note (luojiaxuan): The driver's connection to replay-sink: commands over its socket, events from its log.
final class SinkLink {
    private let fd: Int32
    private let reader: LineReader
    private let tail: LineTail
    /// note (luojiaxuan): Log events read but not yet consumed by a segment.
    private(set) var events: [SinkEvent] = []

    init(socketPath: String, logPath: String) throws {
        var address = try unixAddress(socketPath)
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw ReplayError("socket: \(String(cString: strerror(errno)))") }
        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard connected == 0 else {
            close(fd)
            throw ReplayError("cannot connect to replay-sink at \(socketPath): \(String(cString: strerror(errno)))")
        }
        self.fd = fd
        reader = LineReader(fd: fd)
        tail = try LineTail(path: logPath)
    }

    deinit { close(fd) }

    /// note (luojiaxuan): Sends one command and returns the reply without its "ok " prefix.
    @discardableResult
    func send(_ command: String) throws -> String {
        guard writeAll(fd: fd, command + "\n") else { throw ReplayError("replay-sink closed the connection") }
        guard let reply = reader.next() else { throw ReplayError("replay-sink closed the connection") }
        if reply == "ok" { return "" }
        guard reply.hasPrefix("ok ") else { throw ReplayError("replay-sink: \(reply) (command: \(command))") }
        return String(reply.dropFirst(3))
    }

    /// note (luojiaxuan): Reads every event the sink has logged since the last call.
    func poll() {
        events.append(contentsOf: tail.readNew().compactMap(SinkEvent.init))
    }

    /// note (luojiaxuan): Drops the consumed events: those at or before `index`.
    func consume(through index: Int) {
        events.removeFirst(min(index + 1, events.count))
    }

    func consume(upToTime t: UInt64) {
        events.removeAll { $0.t <= t }
    }
}
