// TextInputCDP.swift
// Provides Text Input CDP for app lifecycle and routing.

import Foundation
import AppKit
import ApplicationServices

private enum FocusedInputCDPSupport {
    nonisolated static let missingRemoteDebuggingPortCacheTTL: TimeInterval = 30
    nonisolated static let remoteDebuggingPortCacheTTL: TimeInterval = 5
    nonisolated static let remoteDebuggingPortCache = RemoteDebuggingPortCache()

    nonisolated final class RemoteDebuggingPortCache: @unchecked Sendable {
        private struct Entry {
            let port: Int?
            let expiresAt: Date
        }

        private let lock = NSLock()
        private var entries: [pid_t: Entry] = [:]

        func lookup(processIdentifier: pid_t, resolver: () -> Int?) -> Int? {
            let now = Date()
            lock.lock()
            if let entry = entries[processIdentifier], entry.expiresAt > now {
                let port = entry.port
                lock.unlock()
                return port
            }
            lock.unlock()

            let port = resolver()
            let ttl = port == nil
                ? FocusedInputCDPSupport.missingRemoteDebuggingPortCacheTTL
                : FocusedInputCDPSupport.remoteDebuggingPortCacheTTL

            lock.lock()
            entries[processIdentifier] = Entry(
                port: port,
                expiresAt: now.addingTimeInterval(ttl)
            )
            lock.unlock()

            return port
        }
    }

    static let inputSnapshotJavaScript = """
    (() => {
        const deepActiveElement = (root) => {
            let current = root && root.activeElement ? root.activeElement : null;
            while (current && current.shadowRoot && current.shadowRoot.activeElement) {
                current = current.shadowRoot.activeElement;
            }
            return current;
        };

        const elementSnapshot = (element, source) => {
            if (!element) {
                return { text: "", selectionStart: 0, selectionEnd: 0, tag: null, source };
            }

            if (typeof element.value === "string") {
                return {
                    text: element.value,
                    selectionStart: typeof element.selectionStart === "number" ? element.selectionStart : 0,
                    selectionEnd: typeof element.selectionEnd === "number" ? element.selectionEnd : 0,
                    tag: element.tagName || null,
                    source
                };
            }

            if (element.isContentEditable) {
                return {
                    text: element.innerText || element.textContent || "",
                    selectionStart: 0,
                    selectionEnd: 0,
                    tag: element.tagName || null,
                    source
                };
            }

            if (element.classList && element.classList.contains("cm-content")) {
                return {
                    text: element.innerText || element.textContent || "",
                    selectionStart: 0,
                    selectionEnd: 0,
                    tag: element.tagName || null,
                    source
                };
            }

            return {
                text: "",
                selectionStart: 0,
                selectionEnd: 0,
                tag: element.tagName || null,
                source
            };
        };

        const active = deepActiveElement(document);
        const candidates = [elementSnapshot(active, "active-element")];
        const querySelectors = [
            "textarea",
            "input",
            "[contenteditable='true']",
            "[role='textbox']",
            ".cm-content",
            "[data-lexical-editor='true']"
        ];

        for (const selector of querySelectors) {
            const match = document.querySelector(selector);
            if (match) {
                candidates.push(elementSnapshot(match, `query:${selector}`));
            }
        }

        const resolved = candidates.find((item) => typeof item.text === "string" && item.text.trim().length > 0)
            || candidates[0]
            || { text: "", selectionStart: 0, selectionEnd: 0, tag: null, source: "none" };

        return JSON.stringify(resolved);
    })()
    """

    static let escapedInputSnapshotJavaScript = inputSnapshotJavaScript
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
        .replacingOccurrences(of: "\n", with: " ")

    struct Target: Decodable {
        let id: String
        let type: String
        let title: String?
        let url: String?
        let webSocketDebuggerUrl: String?
    }

    struct EvaluationResponse: Decodable {
        struct ResultContainer: Decodable {
            struct RemoteObject: Decodable {
                let value: String?
            }

            let result: RemoteObject?
        }

        struct CDPError: Decodable {
            let message: String
        }

        let id: Int?
        let result: ResultContainer?
        let error: CDPError?
    }

    struct InputSnapshotPayload: Decodable {
        let text: String
        let selectionStart: Int
        let selectionEnd: Int
        let tag: String?
        let source: String?
    }

    enum ClientError: LocalizedError {
        case socketCreateFailed
        case socketConnectFailed
        case socketWriteFailed
        case socketReadFailed
        case invalidHTTPResponse
        case invalidWebSocketHandshake
        case invalidWebSocketFrame

        var errorDescription: String? {
            switch self {
            case .socketCreateFailed:
                return "socket create failed"
            case .socketConnectFailed:
                return "socket connect failed"
            case .socketWriteFailed:
                return "socket write failed"
            case .socketReadFailed:
                return "socket read failed"
            case .invalidHTTPResponse:
                return "invalid HTTP response"
            case .invalidWebSocketHandshake:
                return "invalid websocket handshake"
            case .invalidWebSocketFrame:
                return "invalid websocket frame"
            }
        }
    }
}

extension AppDelegate {
    func electronCDPFocusedInputTextSnapshot(
        bundleIdentifier: String?,
        processIdentifier: pid_t?
    ) async -> FocusedInputTextSnapshot? {
        guard let processIdentifier else { return nil }
        let portLookupStartedAt = Date()
        let port = await Self.cachedCommandLineRemoteDebuggingPort(for: processIdentifier)
        let portLookupElapsed = Date().timeIntervalSince(portLookupStartedAt)
        if portLookupElapsed >= 0.04 {
            VoxtLog.input(
                "Focused input CDP port lookup was slow. elapsedMs=\(Int(portLookupElapsed * 1000)), bundleID=\(bundleIdentifier ?? "unknown"), pid=\(processIdentifier)"
            )
        }

        guard let port else {
            VoxtLog.input(
                "Focused input CDP fallback unavailable: remote debugging port missing. bundleID=\(bundleIdentifier ?? "unknown"), pid=\(processIdentifier)"
            )
            return nil
        }

        do {
            guard let target = try await electronCDPPageTarget(port: port) else {
                VoxtLog.input(
                    "Focused input CDP fallback unavailable: no page target. bundleID=\(bundleIdentifier ?? "unknown"), pid=\(processIdentifier), port=\(port)"
                )
                return nil
            }
            guard let payload = try await electronCDPInputPayload(
                webSocketDebuggerURL: target.webSocketDebuggerUrl,
                port: port
            ) else {
                VoxtLog.input(
                    "Focused input CDP fallback unavailable: active element payload empty. bundleID=\(bundleIdentifier ?? "unknown"), pid=\(processIdentifier), port=\(port), targetTitle=\(target.title ?? "nil")"
                )
                return nil
            }

            let trimmedText = payload.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedText.isEmpty else {
                VoxtLog.input(
                    "Focused input CDP fallback unavailable: extracted text empty. bundleID=\(bundleIdentifier ?? "unknown"), pid=\(processIdentifier), port=\(port), tag=\(payload.tag ?? "nil"), source=\(payload.source ?? "nil")"
                )
                return nil
            }

            return FocusedInputTextSnapshot(
                text: trimmedText,
                bundleIdentifier: bundleIdentifier,
                processIdentifier: processIdentifier,
                role: payload.tag ?? "CDP",
                isEditable: true,
                isFocusedTarget: true,
                selectedRange: nil,
                failureReason: nil,
                textSource: "electron-cdp:\(payload.source ?? "unknown")"
            )
        } catch {
            VoxtLog.input(
                "Focused input CDP fallback failed. bundleID=\(bundleIdentifier ?? "unknown"), pid=\(processIdentifier), error=\(error.localizedDescription)"
            )
            return nil
        }
    }

    private nonisolated static func cachedCommandLineRemoteDebuggingPort(for processIdentifier: pid_t) async -> Int? {
        await Task.detached(priority: .utility) {
            FocusedInputCDPSupport.remoteDebuggingPortCache.lookup(processIdentifier: processIdentifier) {
                commandLineRemoteDebuggingPort(for: processIdentifier)
            }
        }.value
    }

    private nonisolated static func commandLineRemoteDebuggingPort(for processIdentifier: pid_t) -> Int? {
        guard let commandLine = processCommandLine(for: processIdentifier) else { return nil }
        let patterns = [
            #"--remote-debugging-port=(\d+)"#,
            #"--remote-debugging-port\s+(\d+)"#
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let range = NSRange(location: 0, length: commandLine.utf16.count)
            guard let match = regex.firstMatch(in: commandLine, range: range),
                  match.numberOfRanges >= 2,
                  let captureRange = Range(match.range(at: 1), in: commandLine) else {
                continue
            }
            return Int(commandLine[captureRange])
        }
        return nil
    }

    private nonisolated static func processCommandLine(for processIdentifier: pid_t) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-ww", "-o", "command=", "-p", String(processIdentifier)]

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = Pipe()

        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return nil }
            let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return output?.isEmpty == false ? output : nil
        } catch {
            return nil
        }
    }

    private func electronCDPPageTarget(port: Int) async throws -> FocusedInputCDPSupport.Target? {
        let data = try electronCDPHTTPGet(path: "/json/list", port: port)
        let targets = try JSONDecoder().decode([FocusedInputCDPSupport.Target].self, from: data)
        return targets.first {
            $0.type == "page"
                && ($0.webSocketDebuggerUrl?.isEmpty == false)
        }
    }

    private func electronCDPInputPayload(
        webSocketDebuggerURL: String?,
        port: Int
    ) async throws -> FocusedInputCDPSupport.InputSnapshotPayload? {
        guard let webSocketDebuggerURL,
              let components = URLComponents(string: webSocketDebuggerURL),
              let host = components.host else {
            return nil
        }

        let requestID = 1
        let requestPayload = """
        {"id":\(requestID),"method":"Runtime.evaluate","params":{"expression":"\(FocusedInputCDPSupport.escapedInputSnapshotJavaScript)","returnByValue":true,"awaitPromise":true}}
        """

        let socket = try electronCDPOpenSocket(host: host, port: components.port ?? port)
        defer { close(socket) }

        let path = (components.path.isEmpty ? "/" : components.path)
            + (components.percentEncodedQuery.map { "?\($0)" } ?? "")
        try electronCDPPerformWebSocketHandshake(
            socket: socket,
            host: host,
            port: components.port ?? port,
            path: path
        )
        try electronCDPSendWebSocketTextFrame(socket: socket, text: requestPayload)

        while true {
            let text = try electronCDPReceiveWebSocketTextFrame(socket: socket)
            guard let responseData = text.data(using: .utf8),
                  let response = try? JSONDecoder().decode(FocusedInputCDPSupport.EvaluationResponse.self, from: responseData),
                  response.id == requestID else {
                continue
            }

            if let error = response.error {
                throw NSError(
                    domain: "Voxt.ElectronCDP",
                    code: port,
                    userInfo: [NSLocalizedDescriptionKey: error.message]
                )
            }

            guard let value = response.result?.result?.value,
                  let payloadData = value.data(using: .utf8) else {
                return nil
            }
            return try JSONDecoder().decode(FocusedInputCDPSupport.InputSnapshotPayload.self, from: payloadData)
        }
    }

    private func electronCDPOpenSocket(host: String, port: Int) throws -> Int32 {
        let socketDescriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard socketDescriptor >= 0 else {
            throw FocusedInputCDPSupport.ClientError.socketCreateFailed
        }

        var timeout = timeval(tv_sec: 2, tv_usec: 0)
        setsockopt(socketDescriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(socketDescriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        let conversion = host.withCString { inet_pton(AF_INET, $0, &address.sin_addr) }
        guard conversion == 1 else {
            close(socketDescriptor)
            throw FocusedInputCDPSupport.ClientError.socketConnectFailed
        }

        let connectResult = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { pointer in
                connect(socketDescriptor, pointer, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connectResult == 0 else {
            close(socketDescriptor)
            throw FocusedInputCDPSupport.ClientError.socketConnectFailed
        }

        return socketDescriptor
    }

    private func electronCDPHTTPGet(path: String, port: Int) throws -> Data {
        let socket = try electronCDPOpenSocket(host: "127.0.0.1", port: port)
        defer { close(socket) }

        let request = """
        GET \(path) HTTP/1.1\r
        Host: 127.0.0.1:\(port)\r
        Connection: close\r
        \r
        """
        try electronCDPWriteAll(socket: socket, data: Data(request.utf8))
        let responseData = try electronCDPReadUntilClose(socket: socket)

        guard let separatorRange = responseData.range(of: Data("\r\n\r\n".utf8)) else {
            throw FocusedInputCDPSupport.ClientError.invalidHTTPResponse
        }
        let headerData = responseData.subdata(in: 0..<separatorRange.lowerBound)
        guard let headerText = String(data: headerData, encoding: .utf8),
              headerText.hasPrefix("HTTP/1.1 200") || headerText.hasPrefix("HTTP/1.0 200") else {
            throw FocusedInputCDPSupport.ClientError.invalidHTTPResponse
        }
        return responseData.subdata(in: separatorRange.upperBound..<responseData.count)
    }

    private func electronCDPPerformWebSocketHandshake(
        socket: Int32,
        host: String,
        port: Int,
        path: String
    ) throws {
        let websocketKey = Data(UUID().uuidString.utf8).base64EncodedString()
        let request = """
        GET \(path) HTTP/1.1\r
        Host: \(host):\(port)\r
        Upgrade: websocket\r
        Connection: Upgrade\r
        Sec-WebSocket-Key: \(websocketKey)\r
        Sec-WebSocket-Version: 13\r
        \r
        """
        try electronCDPWriteAll(socket: socket, data: Data(request.utf8))
        let handshake = try electronCDPReadUntilHeaderTerminator(socket: socket)
        guard let handshakeText = String(data: handshake, encoding: .utf8),
              handshakeText.hasPrefix("HTTP/1.1 101") || handshakeText.hasPrefix("HTTP/1.0 101") else {
            throw FocusedInputCDPSupport.ClientError.invalidWebSocketHandshake
        }
    }

    private func electronCDPSendWebSocketTextFrame(socket: Int32, text: String) throws {
        let payload = Data(text.utf8)
        var frame = Data()
        frame.append(0x81)

        let maskKey = UInt32.random(in: UInt32.min...UInt32.max)
        let maskBytes: [UInt8] = [
            UInt8((maskKey >> 24) & 0xff),
            UInt8((maskKey >> 16) & 0xff),
            UInt8((maskKey >> 8) & 0xff),
            UInt8(maskKey & 0xff)
        ]

        if payload.count < 126 {
            frame.append(UInt8(payload.count) | 0x80)
        } else if payload.count <= UInt16.max {
            frame.append(126 | 0x80)
            var length = UInt16(payload.count).bigEndian
            withUnsafeBytes(of: &length) { frame.append(contentsOf: $0) }
        } else {
            frame.append(127 | 0x80)
            var length = UInt64(payload.count).bigEndian
            withUnsafeBytes(of: &length) { frame.append(contentsOf: $0) }
        }

        frame.append(contentsOf: maskBytes)
        for (index, byte) in payload.enumerated() {
            frame.append(byte ^ maskBytes[index % maskBytes.count])
        }

        try electronCDPWriteAll(socket: socket, data: frame)
    }

    private func electronCDPReceiveWebSocketTextFrame(socket: Int32) throws -> String {
        while true {
            let header = try electronCDPReadExactly(socket: socket, count: 2)
            guard header.count == 2 else { throw FocusedInputCDPSupport.ClientError.invalidWebSocketFrame }

            let first = header[header.startIndex]
            let second = header[header.startIndex + 1]
            let opcode = first & 0x0f
            let masked = (second & 0x80) != 0

            var payloadLength = Int(second & 0x7f)
            if payloadLength == 126 {
                let extended = try electronCDPReadExactly(socket: socket, count: 2)
                payloadLength = Int(extended.withUnsafeBytes { $0.load(as: UInt16.self).bigEndian })
            } else if payloadLength == 127 {
                let extended = try electronCDPReadExactly(socket: socket, count: 8)
                payloadLength = Int(extended.withUnsafeBytes { $0.load(as: UInt64.self).bigEndian })
            }

            let maskData = masked ? try electronCDPReadExactly(socket: socket, count: 4) : Data()
            var payload = try electronCDPReadExactly(socket: socket, count: payloadLength)
            if masked {
                for index in payload.indices {
                    payload[index] ^= maskData[maskData.startIndex + (payload.distance(from: payload.startIndex, to: index) % 4)]
                }
            }

            switch opcode {
            case 0x1:
                guard let text = String(data: payload, encoding: .utf8) else {
                    throw FocusedInputCDPSupport.ClientError.invalidWebSocketFrame
                }
                return text
            case 0x8:
                throw FocusedInputCDPSupport.ClientError.invalidWebSocketFrame
            case 0x9:
                try electronCDPSendWebSocketControlFrame(socket: socket, opcode: 0xA, payload: payload)
            default:
                continue
            }
        }
    }

    private func electronCDPSendWebSocketControlFrame(
        socket: Int32,
        opcode: UInt8,
        payload: Data
    ) throws {
        guard payload.count < 126 else {
            throw FocusedInputCDPSupport.ClientError.invalidWebSocketFrame
        }
        var frame = Data()
        frame.append(0x80 | opcode)
        frame.append(UInt8(payload.count))
        frame.append(payload)
        try electronCDPWriteAll(socket: socket, data: frame)
    }

    private func electronCDPWriteAll(socket: Int32, data: Data) throws {
        try data.withUnsafeBytes { buffer in
            guard let baseAddress = buffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                throw FocusedInputCDPSupport.ClientError.socketWriteFailed
            }
            var totalSent = 0
            while totalSent < data.count {
                let sent = send(socket, baseAddress.advanced(by: totalSent), data.count - totalSent, 0)
                guard sent > 0 else {
                    throw FocusedInputCDPSupport.ClientError.socketWriteFailed
                }
                totalSent += sent
            }
        }
    }

    private func electronCDPReadUntilClose(socket: Int32) throws -> Data {
        var collected = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let received = recv(socket, &buffer, buffer.count, 0)
            if received == 0 {
                return collected
            }
            guard received > 0 else {
                throw FocusedInputCDPSupport.ClientError.socketReadFailed
            }
            collected.append(buffer, count: received)
        }
    }

    private func electronCDPReadUntilHeaderTerminator(socket: Int32) throws -> Data {
        var collected = Data()
        var buffer = [UInt8](repeating: 0, count: 512)
        while collected.range(of: Data("\r\n\r\n".utf8)) == nil {
            let received = recv(socket, &buffer, buffer.count, 0)
            guard received > 0 else {
                throw FocusedInputCDPSupport.ClientError.socketReadFailed
            }
            collected.append(buffer, count: received)
            guard collected.count < 16_384 else {
                throw FocusedInputCDPSupport.ClientError.invalidWebSocketHandshake
            }
        }
        return collected
    }

    private func electronCDPReadExactly(socket: Int32, count: Int) throws -> Data {
        var collected = Data(count: count)
        try collected.withUnsafeMutableBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                throw FocusedInputCDPSupport.ClientError.socketReadFailed
            }
            var totalRead = 0
            while totalRead < count {
                let received = recv(socket, baseAddress.advanced(by: totalRead), count - totalRead, 0)
                guard received > 0 else {
                    throw FocusedInputCDPSupport.ClientError.socketReadFailed
                }
                totalRead += received
            }
        }
        return collected
    }
}
