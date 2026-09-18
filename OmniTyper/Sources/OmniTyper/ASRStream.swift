// SPDX-License-Identifier: Apache-2.0
import Foundation

// Note (Codex): Segment hypotheses replace earlier words; only transcription.completed may be inserted.
struct TranscriptionPreview {
    private var segments: [Int: String] = [:]
    private var finalized: Set<Int> = []
    private var eventIndex = -1
    var text: String { segments.keys.sorted().compactMap { segments[$0] }.joined(separator: "\n") }

    mutating func receive(_ event: [String: Any]) throws -> String? {
        if event["type"] as? String == "error" {
            throw Failure("asr.error.stream")
        }
        guard let type = event["type"] as? String,
              type == "transcription.segment" || type == "transcription.completed" else { return nil }
        guard let index = event["event_index"] as? Int, index >= 0,
              let value = event["text"] as? String,
              value.unicodeScalars.count <= 12_000, !value.contains("\0") else {
            throw Failure("asr.error.transcript")
        }
        guard index > eventIndex else { return nil }
        eventIndex = index
        if type == "transcription.completed" { return value }
        guard let id = event["segment_id"] as? Int, (0..<300).contains(id),
              let isFinal = event["is_final"] as? Bool else {
            throw Failure("asr.error.segment")
        }
        guard !finalized.contains(id) else { return nil }
        segments[id] = value
        if isFinal { finalized.insert(id) }
        guard text.unicodeScalars.count <= 12_000 else {
            throw Failure("asr.error.tooLong")
        }
        return nil
    }
}

@MainActor
final class ASRStream {
    private let session: URLSession
    private let socket: URLSessionWebSocketTask
    private let audio = AsyncThrowingStream<Data, Error>.makeStream(bufferingPolicy: .bufferingOldest(32))
    private var sender: Task<Void, Error>?
    private var receiver: Task<String, Error>?
    private var failure: Error?
    private var cancelled = false
    private let onPartial: (String) -> Void
    private let onFailure: () -> Void

    init(url: String, onPartial: @escaping (String) -> Void, onFailure: @escaping () -> Void) throws {
        guard let address = URL(string: url), address.scheme == "ws", address.host == "127.0.0.1",
              let port = address.port, (1...65535).contains(port), address.user == nil,
              address.password == nil, address.path == "/v1/realtime",
              address.query == "intent=transcription", address.fragment == nil else {
            throw Failure("asr.error.address")
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 1200
        session = URLSession(configuration: configuration)
        socket = session.webSocketTask(with: address)
        socket.maximumMessageSize = 256 * 1024
        self.onPartial = onPartial
        self.onFailure = onFailure
    }

    var audioInput: @Sendable (Data) -> Void {
        let continuation = audio.continuation
        return { data in
            guard !data.isEmpty else { return }
            guard data.count <= 32_768, data.count.isMultiple(of: 2) else {
                continuation.finish(throwing: Failure("asr.error.packet"))
                return
            }
            if case .dropped = continuation.yield(data) {
                // ponytail: bounded queue; keep the WAV and retry batch ASR on a slow connection.
                continuation.finish(throwing: Failure("asr.error.behind"))
            }
        }
    }

    func connect(language: String) async throws {
        socket.resume()
        let timeout = deadline(seconds: 30)
        defer { timeout.cancel() }
        do {
            try await send(["type": "session.update", "session": [
                "language": language.isEmpty ? NSNull() : language as Any,
                "turn_detection": NSNull()
            ]])
            while true {
                let event = try await receive()
                if event["type"] as? String == "error" {
                    throw Failure("asr.error.start")
                }
                if event["type"] as? String == "session.updated" { break }
            }
            try Task.checkCancellation()
            receiver = Task { [self] in
                do {
                    var preview = TranscriptionPreview()
                    while true {
                        let event = try await receive()
                        if let final = try preview.receive(event) { return final }
                        if event["type"] as? String == "transcription.segment" { onPartial(preview.text) }
                    }
                } catch { fail(error); throw error }
            }
            sender = Task { [self] in
                do {
                    for try await packet in audio.stream {
                        try Task.checkCancellation()
                        try await send(["type": "input_audio_buffer.append", "audio": packet.base64EncodedString()])
                    }
                } catch { fail(error); throw error }
            }
        } catch { cancel(); throw error }
    }

    func finish() async throws -> String {
        if let failure { throw failure }
        guard let sender, let receiver, !cancelled else { throw CancellationError() }
        let timeout = deadline(seconds: 180)
        defer { timeout.cancel(); cancel() }
        audio.continuation.finish()
        try await sender.value // Drain every captured sample before ending the turn.
        try await send(["type": "transcription.done"])
        let text = try await receiver.value
        try Task.checkCancellation()
        return text
    }

    func cancel() {
        cancelled = true
        audio.continuation.finish()
        sender?.cancel(); receiver?.cancel()
        sender = nil; receiver = nil
        socket.cancel(with: .goingAway, reason: nil)
        session.invalidateAndCancel()
    }

    private func fail(_ error: Error) {
        guard failure == nil, !cancelled else { return }
        failure = error
        onFailure()
        cancel()
    }

    private func deadline(seconds: UInt64) -> Task<Void, Never> {
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
            if !Task.isCancelled { self?.fail(Failure("asr.error.timeout")) }
        }
    }

    private func send(_ event: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: event)
        try await socket.send(.string(String(decoding: data, as: UTF8.self)))
    }

    private func receive() async throws -> [String: Any] {
        let data: Data
        switch try await socket.receive() {
        case .data(let bytes): data = bytes
        case .string(let string): data = Data(string.utf8)
        @unknown default: throw Failure("asr.error.unsupported")
        }
        guard let event = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Failure("asr.error.response")
        }
        return event
    }
}
