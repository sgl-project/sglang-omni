import Foundation

/// Sortformer on the native runtime (`--model-kind sortformer`): every meeting
/// speaker analysis shares one server.
nonisolated enum OmniSortformerRuntime {
    static let shared = OmniSharedModelRuntime(kind: .sortformer)

    /// Whether Sortformer runs on the native runtime in this process.
    static var isEnabled: Bool { OmniASRBackend.launchSettings != nil }
}

nonisolated enum OmniSpeakerDiarizationError: LocalizedError, Equatable {
    case server(String)
    case malformedResponse

    var errorDescription: String? {
        switch self {
        case .server(let message):
            return "The local speaker analysis server refused the audio: \(message)"
        case .malformedResponse:
            return "The local speaker analysis server sent a malformed response."
        }
    }
}

/// Swift `SortformerModel.feed` arguments; the defaults are Voxt's.
nonisolated struct OmniDiarizationOptions: Sendable, Equatable {
    var threshold: Float = 0.5
    var minDuration: Float = 0
    var mergeGap: Float = 0.18
    var spkcacheMax = 188
    var fifoMax = 188
}

/// One feed's results, as Swift `SortformerModel.feed` returns them.
nonisolated struct OmniDiarizationFeed: Sendable, Equatable {
    nonisolated struct Segment: Sendable, Equatable {
        /// Seconds from the start of the stream.
        let start: Float
        let end: Float
        let speaker: Int
    }

    /// Per frame, one probability per speaker.
    let probabilities: [[Float]]
    let segments: [Segment]
    let fifoLength: Int
    let spkcacheLength: Int
    let framesProcessed: Int
}

/// One `/v1/diarization/stream` socket: the server keeps this stream's
/// Sortformer state (speaker cache, FIFO, silence profile) across feeds.
actor OmniDiarizationStream {
    private let session: URLSession
    private let socket: URLSessionWebSocketTask
    private var tail: Task<Void, Never>?

    init(endpoint: OmniServerEndpoint, options: OmniDiarizationOptions = .init()) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        session = URLSession(configuration: configuration)
        socket = session.webSocketTask(with: Self.url(endpoint: endpoint, options: options))
        socket.resume()
    }

    /// Diarizes one feed of 16 kHz samples and advances the stream's state.
    func feed(samples16k: [Float]) async throws -> OmniDiarizationFeed {
        let previous = tail
        let socket = socket
        // Note (Jiaxin Deng): replies arrive in order, so exchanges run one at a time per stream.
        let exchange = Task { () throws -> OmniDiarizationFeed in
            await previous?.value
            try await socket.send(.data(OmniVoiceActivityStream.float32LittleEndian(samples16k)))
            guard case .string(let text) = try await socket.receive() else {
                throw OmniSpeakerDiarizationError.malformedResponse
            }
            return try Self.feed(fromReply: text)
        }
        tail = Task { _ = try? await exchange.value }
        return try await exchange.value
    }

    func close() {
        socket.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
    }

    nonisolated static func url(endpoint: OmniServerEndpoint, options: OmniDiarizationOptions) -> URL {
        var components = URLComponents()
        components.scheme = "ws"
        components.host = endpoint.host
        components.port = endpoint.port
        components.path = "/v1/diarization/stream"
        components.queryItems = [
            URLQueryItem(name: "threshold", value: String(options.threshold)),
            URLQueryItem(name: "min_duration", value: String(options.minDuration)),
            URLQueryItem(name: "merge_gap", value: String(options.mergeGap)),
            URLQueryItem(name: "spkcache_max", value: String(options.spkcacheMax)),
            URLQueryItem(name: "fifo_max", value: String(options.fifoMax)),
        ]
        return components.url!
    }

    nonisolated static func feed(fromReply text: String) throws -> OmniDiarizationFeed {
        guard let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else {
            throw OmniSpeakerDiarizationError.malformedResponse
        }
        if let error = object["error"] as? String {
            throw OmniSpeakerDiarizationError.server(error)
        }
        guard let frames = object["frames"] as? Int,
              let speakers = object["speakers"] as? Int,
              let rows = object["probabilities"] as? [[NSNumber]],
              let segments = object["segments"] as? [[String: Any]],
              let state = object["state"] as? [String: Any],
              let fifoLength = state["fifo_length"] as? Int,
              let spkcacheLength = state["spkcache_length"] as? Int,
              let framesProcessed = state["frames_processed"] as? Int,
              rows.count == frames,
              rows.allSatisfy({ $0.count == speakers })
        else { throw OmniSpeakerDiarizationError.malformedResponse }
        let parsedSegments = try segments.map { segment in
            guard let start = segment["start"] as? NSNumber,
                  let end = segment["end"] as? NSNumber,
                  let speaker = segment["speaker"] as? Int,
                  start.floatValue <= end.floatValue,
                  (0 ..< speakers).contains(speaker)
            else { throw OmniSpeakerDiarizationError.malformedResponse }
            return OmniDiarizationFeed.Segment(start: start.floatValue, end: end.floatValue, speaker: speaker)
        }
        return OmniDiarizationFeed(
            probabilities: rows.map { $0.map(\.floatValue) },
            segments: parsedSegments,
            fifoLength: fifoLength,
            spkcacheLength: spkcacheLength,
            framesProcessed: framesProcessed
        )
    }
}
