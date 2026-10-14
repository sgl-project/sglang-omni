import Foundation
import os

/// Live preview events from the local Omni realtime transcription socket.
nonisolated enum OmniLiveEvent: Sendable, Equatable {
    case display(confirmedText: String, provisionalText: String)
    case ended(text: String, segments: [OmniSpeakerSegment])
    case failed(message: String)
}

/// Rebuilds the visible transcript from realtime segment events.
///
/// Each segment's latest hypothesis replaces the previous one for the same
/// segment id, and a replayed or reordered event index is ignored.
nonisolated struct OmniLiveTranscriptAssembler: Equatable {
    /// How final segments join into the confirmed text.
    enum Joining: Sendable {
        /// A space only between scripts that separate words with spaces (Qwen3-ASR).
        case scriptAware
        /// One line per segment, as MLXAudio's MOSS session joined its windows;
        /// Voxt's MOSS rendering then merges the lines.
        case lines
    }

    let joining: Joining
    private(set) var lastEventIndex = 0
    private var finalTextBySegment: [Int: String] = [:]
    private var provisional: (segment: Int, text: String)?

    init(joining: Joining = .scriptAware) {
        self.joining = joining
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.joining == rhs.joining
            && lhs.lastEventIndex == rhs.lastEventIndex
            && lhs.finalTextBySegment == rhs.finalTextBySegment
            && lhs.provisional?.segment == rhs.provisional?.segment
            && lhs.provisional?.text == rhs.provisional?.text
    }

    /// Returns the new display state, or nil when the event changes nothing.
    mutating func apply(eventIndex: Int?, segmentID: Int, text: String, isFinal: Bool) -> OmniLiveEvent? {
        if let eventIndex {
            guard eventIndex > lastEventIndex else { return nil }
            lastEventIndex = eventIndex
        }
        if isFinal {
            finalTextBySegment[segmentID] = text
            if provisional?.segment == segmentID { provisional = nil }
        } else if finalTextBySegment[segmentID] == nil {
            provisional = (segmentID, text)
        } else {
            return nil
        }
        let provisionalText = provisional?.text ?? ""
        var confirmed = confirmedText
        // MLXAudio's MOSS session put the pending window on its own line.
        if joining == .lines, !confirmed.isEmpty, !provisionalText.isEmpty {
            confirmed += "\n"
        }
        return .display(confirmedText: confirmed, provisionalText: provisionalText)
    }

    var confirmedText: String {
        let finals = finalTextBySegment.keys.sorted().compactMap { finalTextBySegment[$0] }
        switch joining {
        case .scriptAware:
            return OmniTranscriptJoining.join(finals)
        case .lines:
            return finals
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
        }
    }
}

/// One `/v1/realtime?intent=transcription` session against the local server.
nonisolated final class OmniRealtimeTranscriptionSession: @unchecked Sendable {
    /// Unsent audio beyond this many samples means the server cannot keep up.
    static let maximumQueuedSamples = 16_000 * 30
    static let sampleRate = 16_000

    let events: AsyncStream<OmniLiveEvent>
    private let continuation: AsyncStream<OmniLiveEvent>.Continuation
    private let socket: URLSessionWebSocketTask
    private let session: URLSession
    private struct Shared {
        var assembler: OmniLiveTranscriptAssembler
        var queuedSamples = 0
        var closed = false
    }
    private let shared: OSAllocatedUnfairLock<Shared>
    private struct Outbound: Sendable {
        let text: String
        let sampleCount: Int
    }
    private let outbound: AsyncStream<Outbound>.Continuation
    private let sender: Task<Void, Never>
    private let receiver: Task<Void, Never>

    init(
        endpoint: OmniServerEndpoint,
        language: String?,
        prompt: String? = nil,
        chunkMilliseconds: Int? = nil,
        joining: OmniLiveTranscriptAssembler.Joining = .scriptAware
    ) {
        shared = OSAllocatedUnfairLock(initialState: Shared(assembler: OmniLiveTranscriptAssembler(joining: joining)))
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        session = URLSession(configuration: configuration)
        var components = URLComponents()
        components.scheme = "ws"
        components.host = endpoint.host
        components.port = endpoint.port
        components.path = "/v1/realtime"
        components.queryItems = [
            URLQueryItem(name: "intent", value: "transcription"),
            URLQueryItem(name: "model", value: endpoint.modelName),
        ]
        socket = session.webSocketTask(with: components.url!)
        (events, continuation) = AsyncStream.makeStream(of: OmniLiveEvent.self)
        let (outboundStream, outbound) = AsyncStream.makeStream(of: Outbound.self)
        self.outbound = outbound
        socket.resume()

        let update = Self.sessionUpdate(language: language, prompt: prompt, chunkMilliseconds: chunkMilliseconds)
        let socket = socket
        let shared = shared
        sender = Task.detached {
            do {
                try await socket.send(.string(update))
                for await message in outboundStream {
                    try await socket.send(.string(message.text))
                    shared.withLock { $0.queuedSamples -= message.sampleCount }
                }
            } catch {
                return
            }
        }
        let continuation = continuation
        let urlSession = session
        receiver = Task.detached {
            while true {
                let message: URLSessionWebSocketTask.Message
                do {
                    message = try await socket.receive()
                } catch {
                    let wasClosed = shared.withLock { $0.closed }
                    if !wasClosed {
                        continuation.yield(.failed(message: "Live transcription connection closed."))
                    }
                    continuation.finish()
                    return
                }
                guard case .string(let text) = message,
                      let data = text.data(using: .utf8),
                      let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let type = event["type"] as? String
                else { continue }
                switch type {
                case "transcription.segment":
                    guard let segmentID = event["segment_id"] as? Int,
                          let segmentText = event["text"] as? String
                    else { continue }
                    let display = shared.withLock {
                        $0.assembler.apply(
                            eventIndex: event["event_index"] as? Int,
                            segmentID: segmentID,
                            text: segmentText,
                            isFinal: event["is_final"] as? Bool ?? false
                        )
                    }
                    if let display { continuation.yield(display) }
                case "transcription.completed":
                    continuation.yield(.ended(
                        text: event["text"] as? String ?? "",
                        segments: OmniSpeakerSegment.parse(event["segments"])
                    ))
                    shared.withLock { $0.closed = true }
                    socket.cancel(with: .normalClosure, reason: nil)
                    urlSession.invalidateAndCancel()
                    continuation.finish()
                    return
                case "error":
                    // Not terminal on the server, but the session cannot recover:
                    // end it so its runtime lease is released. Only the error code
                    // is kept; messages can echo request content such as audio.
                    let error = event["error"] as? [String: Any]
                    let code = error?["code"] as? String ?? error?["type"] as? String ?? "error"
                    continuation.yield(.failed(message: "Live transcription failed (\(code))."))
                    shared.withLock { $0.closed = true }
                    socket.cancel(with: .normalClosure, reason: nil)
                    urlSession.invalidateAndCancel()
                    continuation.finish()
                    return
                default:
                    continue
                }
            }
        }
    }

    /// Queues 16 kHz mono samples; overflowing the bounded queue fails the session.
    func feedAudio(samples: [Float]) {
        guard !samples.isEmpty else { return }
        enum Admission { case accepted, closed, overflow }
        let admission = shared.withLock { state -> Admission in
            if state.closed { return .closed }
            if state.queuedSamples + samples.count > Self.maximumQueuedSamples { return .overflow }
            state.queuedSamples += samples.count
            return .accepted
        }
        switch admission {
        case .closed:
            return
        case .overflow:
            continuation.yield(.failed(message: "Live transcription fell behind the microphone."))
            cancel()
        case .accepted:
            let audio = OmniWAVEncoding.pcm16(samples: samples[...]).base64EncodedString()
            outbound.yield(Outbound(
                text: Self.json(["type": "input_audio_buffer.append", "audio": audio]),
                sampleCount: samples.count
            ))
        }
    }

    /// Commits buffered audio and asks for the final transcript; `.ended` follows.
    func stop() {
        guard !shared.withLock({ $0.closed }) else { return }
        outbound.yield(Outbound(text: Self.json(["type": "input_audio_buffer.commit"]), sampleCount: 0))
        outbound.yield(Outbound(text: Self.json(["type": "transcription.done"]), sampleCount: 0))
    }

    /// Closes the socket without asking for more inference.
    func cancel() {
        let alreadyClosed = shared.withLock { state -> Bool in
            defer { state.closed = true }
            return state.closed
        }
        outbound.finish()
        socket.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
        if !alreadyClosed { continuation.finish() }
    }

    /// The session settings Voxt's Swift live session implies: continuous
    /// decoding with no voice-activity onset to wait for. The prompt is MOSS's
    /// task instruction.
    static func sessionUpdate(language: String?, prompt: String? = nil, chunkMilliseconds: Int? = nil) -> String {
        var sessionConfig: [String: Any] = [
            "input_audio_format": "pcm16",
            "turn_detection": NSNull(),
        ]
        if let language { sessionConfig["language"] = language }
        if let prompt { sessionConfig["prompt"] = prompt }
        if let chunkMilliseconds { sessionConfig["chunk_ms"] = chunkMilliseconds }
        return json(["type": "session.update", "session": sessionConfig])
    }

    private static func json(_ object: [String: Any]) -> String {
        let data = try! JSONSerialization.data(withJSONObject: object)
        return String(decoding: data, as: UTF8.self)
    }
}
