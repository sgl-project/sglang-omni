import Darwin

/// note (luojiaxuan): sockaddr_un for `path`; sun_path holds at most 103 bytes plus the terminator,
/// so long absolute paths must be replaced by a shorter or relative one.
func unixAddress(_ path: String) throws -> sockaddr_un {
    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)
    let bytes = Array(path.utf8)
    let capacity = MemoryLayout.size(ofValue: address.sun_path)
    guard bytes.count < capacity else {
        throw ReplayError("socket path is \(bytes.count) bytes, the limit is \(capacity - 1): use a shorter or relative path")
    }
    withUnsafeMutableBytes(of: &address.sun_path) { raw in
        raw.copyBytes(from: bytes)
        raw[bytes.count] = 0
    }
    address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
    return address
}

/// note (luojiaxuan): Reads newline-terminated lines from a socket descriptor.
final class LineReader {
    private let fd: Int32
    private var buffer: [UInt8] = []

    init(fd: Int32) { self.fd = fd }

    /// note (luojiaxuan): The next line without its newline, or nil at end of stream.
    func next() -> String? {
        while true {
            if let newline = buffer.firstIndex(of: 0x0A) {
                let line = String(decoding: buffer[..<newline], as: UTF8.self)
                buffer.removeSubrange(...newline)
                return line
            }
            var chunk = [UInt8](repeating: 0, count: 4096)
            let n = read(fd, &chunk, chunk.count)
            if n < 0 && errno == EINTR { continue }
            if n <= 0 { return nil }
            buffer.append(contentsOf: chunk[..<n])
        }
    }
}

func writeAll(fd: Int32, _ text: String) -> Bool {
    let bytes = Array(text.utf8)
    var offset = 0
    while offset < bytes.count {
        let n = bytes[offset...].withUnsafeBytes { write(fd, $0.baseAddress!, $0.count) }
        if n < 0 && errno == EINTR { continue }
        if n <= 0 { return false }
        offset += n
    }
    return true
}
