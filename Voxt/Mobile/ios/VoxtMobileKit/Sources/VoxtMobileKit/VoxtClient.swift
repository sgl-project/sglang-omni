import Foundation

/// Talks to Voxt's Local API server on the Mac.
public struct VoxtClient: Sendable {
    public enum Error: Swift.Error, Equatable, LocalizedError {
        case unauthorized
        case transportUnavailable
        case server(status: Int, message: String)
        case invalidResponse
        case editNotApplied
        case cloudModelSelected
        case modelLocationUnknown
        case notOnThisMac

        public var errorDescription: String? {
            switch self {
            case .cloudModelSelected: VoxtL10n.shared.string("Your Mac is set to a cloud model")
            case .modelLocationUnknown: VoxtL10n.shared.string("Could not verify that your Mac uses local models. Update Voxt on your Mac and try again.")
            case .transportUnavailable: "The connection to your Mac is not ready. Reconnect and try again."
            case .unauthorized: "The Mac rejected the pairing token. Pair again from Voxt Settings."
            case let .server(status, message): message.isEmpty ? VoxtL10n.shared.format("Mac returned HTTP %d.", status) : message
            case .editNotApplied: VoxtL10n.shared.string("Couldn't apply that edit")
            case .invalidResponse: "The Mac sent a response this app doesn't understand."
            case .notOnThisMac: VoxtL10n.shared.string("Update Voxt on your Mac to use this.")
            }
        }
    }

    public struct Dictation: Decodable, Equatable, Sendable {
        public var id: String
        public var text: String
        public var rawText: String
        public var asrMs: Int?
        public var enhanceMs: Int?
        /// Measured on the phone, not sent by the Mac.
        public var preflightMs: Int?
        public var requestMs: Int?
    }

    public struct Health: Decodable, Equatable, Sendable {
        public var ok: Bool
        public var app: String
        public var version: String
    }

    /// Local API endpoints a Mac may leave out; capabilities name the ones it serves.
    public enum Feature: String, CaseIterable, Sendable {
        case compose, composePrompts = "compose/prompts", corrections, historySync = "history/sync"
    }

    /// `GET /v1/capabilities`: where the selected models run (`locations`) and which endpoints
    /// the Mac serves. Older Voxt sends only the two locations, at the top level, and serves all.
    public struct Capabilities: Decodable, Equatable, Sendable {
        public var transcription: String
        public var text: String
        public var endpoints: Set<String>?
        public var isLocal: Bool { transcription == "local" && text == "local" }
        public var hasRemote: Bool { transcription == "remote" || text == "remote" }
        public func supports(_ feature: Feature) -> Bool { endpoints?.contains(feature.rawValue) ?? true }

        private struct Locations: Decodable { var transcription: String; var text: String }
        private enum CodingKeys: String, CodingKey { case locations, endpoints, transcription, text }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let locations = try container.decodeIfPresent(Locations.self, forKey: .locations) {
                transcription = locations.transcription
                text = locations.text
            } else {
                transcription = try container.decode(String.self, forKey: .transcription)
                text = try container.decode(String.self, forKey: .text)
            }
            endpoints = try container.decodeIfPresent([String].self, forKey: .endpoints).map(Set.init)
        }
    }

    public let server: ServerConfig
    private let allowCloudModels: @MainActor @Sendable () -> Bool
    private let customPrompt: @MainActor @Sendable (CompositionOptions.Mode) -> String?
    private let session: URLSession
    private let resolveBaseURL: (@Sendable () async throws -> URL)?

    private let onTransportFailure: (@Sendable (URL) async -> Void)?
    private let preflight: PreflightCache?
    /// Pause between tunnel checks before audio is sent: in the background the tunnel can take
    /// longer than two back-to-back checks to come back.
    private let tunnelRetryDelay: Duration
    static let tunnelChecks = 4
    public var endpointTimeout: TimeInterval = 10

    public init(server: ServerConfig, session: URLSession = .shared, resolveBaseURL: (@Sendable () async throws -> URL)? = nil,
                onTransportFailure: (@Sendable (URL) async -> Void)? = nil,
                allowCloudModels: @escaping @MainActor @Sendable () -> Bool = { false },
                customPrompt: @escaping @MainActor @Sendable (CompositionOptions.Mode) -> String? = { _ in nil },
                preflight: PreflightCache? = nil, tunnelRetryDelay: Duration = .seconds(2)) {
        self.tunnelRetryDelay = tunnelRetryDelay
        self.allowCloudModels = allowCloudModels
        self.customPrompt = customPrompt
        self.server = server
        self.session = session
        self.resolveBaseURL = resolveBaseURL
        self.onTransportFailure = onTransportFailure
        self.preflight = preflight
    }

    public func health() async throws -> Health {
        try await health(attempts: 2, delay: .zero)
    }

    /// Health is safe to repeat: after a failure the tunnel is reset (onTransportFailure),
    /// then resolved again on the next attempt.
    private func health(attempts: Int, delay: Duration) async throws -> Health {
        for attempt in 1...attempts {
            do {
                var request = URLRequest(url: try await endpoint(path: "v1/health"))
                request.timeoutInterval = 10
                return try await send(request)
            } catch Error.transportUnavailable where attempt < attempts && onTransportFailure != nil {
                if delay > .zero { try await Task.sleep(for: delay) }
            }
        }
        throw Error.transportUnavailable
    }

    public func syncHistory(_ payload: HistorySyncRequest) async throws -> HistorySyncResponse {
        try payload.validate()
        try await require(.historySync)
        var request = URLRequest(url: try await endpoint(path: "v1/history/sync"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(payload)
        request.timeoutInterval = 30
        let response: HistorySyncResponse = try await send(request)
        try response.validate(for: payload)
        return response
    }

    public func dictate(audio: Data, contentType: String, enhance: Bool = true) async throws -> Dictation {
        try await withPreflight { try await dictateRequest(audio: audio, contentType: contentType, enhance: enhance) }
    }

    private func dictateRequest(audio: Data, contentType: String, enhance: Bool) async throws -> Dictation {
        let preflightStart = Date()
        try await prepareAudioTransport()
        let requestStart = Date()
        var url = try await audioEndpoint(path: "v1/dictations")
        if !enhance { url.append(queryItems: [URLQueryItem(name: "enhance", value: "0")]) }
        // Without it the Mac applies its own language settings; the phone has none to send.
        url.append(queryItems: [URLQueryItem(name: "language", value: "auto")])
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        request.httpBody = audio
        // Long dictations plus a local LLM pass can take a while on the Mac.
        request.timeoutInterval = 120
        var dictation: Dictation = try await send(request, reportsNeverSent: true)
        dictation.preflightMs = Self.ms(since: preflightStart, until: requestStart)
        dictation.requestMs = Self.ms(since: requestStart, until: Date())
        return dictation
    }

    private static func ms(since start: Date, until end: Date) -> Int {
        Int((end.timeIntervalSince(start) * 1000).rounded())
    }

    public func compose(audio: Data, contentType: String, options: CompositionOptions) async throws -> Dictation {
        try await withPreflight { try await composeRequest(audio: audio, contentType: contentType, options: options) }
    }

    private func composeRequest(audio: Data, contentType: String, options: CompositionOptions) async throws -> Dictation {
        var options = options
        if options.customPrompt == nil { options.customPrompt = await customPrompt(options.mode) }
        try options.validate()
        // compose always runs the text model (an identical English WAV took enhanceMs 3352
        // there, 0 with enhance=0). Literal dictation keeps the server's text.
        if options.mode == .dictate && options.literalDictation {
            return try await dictate(audio: audio, contentType: contentType, enhance: false)
        }
        struct Body: Encodable {
            var audio: Data
            var contentType: String
            var language = "auto"
            var options: CompositionOptions
        }
        struct Response: Decodable {
            var id: String
            var text: String
            var rawText: String
            var mode: CompositionOptions.Mode
            var status: String?
            var asrMs: Int?
            var enhanceMs: Int?
        }
        let preflightStart = Date()
        try await prepareAudioTransport()
        let requestStart = Date()
        var request = URLRequest(url: try await audioEndpoint(path: "v1/compose"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(Body(audio: audio, contentType: contentType, options: options))
        request.timeoutInterval = 120
        let response: Response = try await send(request, reportsNeverSent: true)
        guard response.mode == options.mode else { throw Error.invalidResponse }
        if response.status == "not_applied", options.mode == .edit { throw Error.editNotApplied }
        guard response.status == nil || response.status == "applied" else { throw Error.invalidResponse }
        return Dictation(id: response.id, text: response.text, rawText: response.rawText, asrMs: response.asrMs, enhanceMs: response.enhanceMs,
                         preflightMs: Self.ms(since: preflightStart, until: requestStart), requestMs: Self.ms(since: requestStart, until: Date()))
    }

    public func correction(for passage: CorrectionLearningRequest) async throws -> LearnedCorrection? {
        guard passage.isValid else { throw Error.invalidResponse }
        try await requireModelConsent(for: .corrections)
        struct Response: Decodable {
            let correction: LearnedCorrection?
            enum CodingKeys: String, CodingKey { case correction }
            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                // Missing is malformed; explicit null means no vocabulary correction.
                correction = try container.decode(LearnedCorrection?.self, forKey: .correction)
            }
        }
        var request = URLRequest(url: try await endpoint(path: "v1/corrections"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(passage)
        request.timeoutInterval = 45
        let response: Response = try await send(request)
        if response.correction != nil, passage.validated(response.correction) == nil { throw Error.invalidResponse }
        return response.correction
    }

    /// The Mac's built-in instruction per mode, to prefill the Developer prompt editor.
    public func builtInPrompts() async throws -> [CompositionOptions.Mode: String] {
        struct Response: Decodable { var prompts: [String: String] }
        try await require(.composePrompts)
        var request = URLRequest(url: try await endpoint(path: "v1/compose/prompts"))
        request.timeoutInterval = 10
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let response: Response = try await send(request)
        return Dictionary(uniqueKeysWithValues: response.prompts.compactMap { key, value in
            CompositionOptions.Mode(rawValue: key).map { ($0, value) }
        })
    }

    public func capabilities() async throws -> Capabilities {
        var request = URLRequest(url: try await endpoint(path: "v1/capabilities"))
        request.timeoutInterval = 10
        request.cachePolicy = .reloadIgnoringLocalCacheData
        return try await send(request)
    }

    /// Fails with `notOnThisMac` before calling an endpoint the Mac's capabilities leave out.
    private func require(_ feature: Feature) async throws {
        guard try await capabilities().supports(feature) else { throw Error.notOnThisMac }
    }

    /// With cloud models off, the Mac must report local models; one capabilities read also
    /// checks that it serves `feature`.
    private func requireModelConsent(for feature: Feature? = nil) async throws {
        let allowCloud = await allowCloudModels()
        if allowCloud {
            if let feature { try await require(feature) }
            return
        }
        let capabilities: Capabilities
        do { capabilities = try await self.capabilities() }
        catch Error.unauthorized { throw Error.unauthorized }
        catch Error.transportUnavailable { throw Error.transportUnavailable }
        catch { throw Error.modelLocationUnknown }
        if let feature, !capabilities.supports(feature) { throw Error.notOnThisMac }
        guard capabilities.isLocal else {
            throw capabilities.hasRemote ? Error.cloudModelSelected : Error.modelLocationUnknown
        }
    }

    private func prepareAudioTransport() async throws {
        let allowCloud = await allowCloudModels()
        if preflight?.isFresh(allowCloud: allowCloud) == true { return }
        // Check the native tunnel before sending audio. Never blindly replay an audio POST,
        // which may already have been accepted when the response connection is lost.
        if server.tailcat != nil, onTransportFailure != nil {
            _ = try await health(attempts: Self.tunnelChecks, delay: tunnelRetryDelay)
        }
        try await requireModelConsent()
        preflight?.markVerified(allowCloud: allowCloud)
    }

    /// The audio request provably never reached the Mac (no connection, or no endpoint).
    private struct NeverSent: Swift.Error {}

    /// Runs an audio request; success keeps the pre-checks fresh, any failure drops them.
    /// When the cached path fails before anything was sent, retry once through the full
    /// checks; a request that may have been accepted is never replayed.
    private func withPreflight<T>(_ body: () async throws -> T) async throws -> T {
        let usedCache = preflight?.isFresh(allowCloud: await allowCloudModels()) == true
        do {
            let value = try await body()
            preflight?.markVerified(allowCloud: await allowCloudModels())
            return value
        } catch is NeverSent {
            preflight?.invalidate()
            guard usedCache else { throw Error.transportUnavailable }
            do {
                let value = try await body()
                preflight?.markVerified(allowCloud: await allowCloudModels())
                return value
            } catch is NeverSent {
                throw Error.transportUnavailable
            } catch {
                preflight?.invalidate()
                throw error
            }
        } catch {
            preflight?.invalidate()
            throw error
        }
    }

    /// Endpoint resolution happens before any byte is sent.
    private func audioEndpoint(path: String) async throws -> URL {
        do { return try await endpoint(path: path) } catch Error.transportUnavailable { throw NeverSent() }
    }

    /// Resolve on every request so a rebuilt native bridge can use a new loopback port.
    private func endpoint(path: String) async throws -> URL {
        try Task.checkCancellation()
        guard server.tailcat == nil || resolveBaseURL != nil else { throw Error.transportUnavailable }
        let base: URL
        if let resolveBaseURL { base = try await Self.resolve(resolveBaseURL, timeout: endpointTimeout) }
        else { base = server.baseURL }
        try Task.checkCancellation()
        return base.appending(path: path)
    }

    /// Resolves the native bridge endpoint, giving up after `timeout`. The bridge may not
    /// honor cancellation, so the caller is released without waiting for it to return.
    private static func resolve(_ resolver: @escaping @Sendable () async throws -> URL, timeout: TimeInterval) async throws -> URL {
        let gate = ResumeOnce()
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URL, Swift.Error>) in
            let resolving = Task {
                do {
                    let url = try await resolver()
                    // An adapter can finish its native hop after the request was cancelled.
                    if Task.isCancelled { throw CancellationError() }
                    gate.run { continuation.resume(returning: url) }
                } catch {
                    gate.run { continuation.resume(throwing: error) }
                }
            }
            Task {
                try? await Task.sleep(for: .seconds(timeout))
                gate.run {
                    resolving.cancel()
                    continuation.resume(throwing: Error.transportUnavailable)
                }
            }
        }
    }

    private func send<T: Decodable>(_ request: URLRequest, reportsNeverSent: Bool = false) async throws -> T {
        var request = request
        request.setValue("Bearer \(server.token)", forHTTPHeaderField: "Authorization")
        request.setValue(await allowCloudModels() ? "cloud-allowed" : "local-only", forHTTPHeaderField: "X-Voxt-Model-Policy")
        let data: Data
        let response: URLResponse
        do { (data, response) = try await session.data(for: request) }
        catch let error as URLError where [.cannotConnectToHost, .cannotFindHost, .dnsLookupFailed,
                                          .networkConnectionLost, .notConnectedToInternet, .timedOut].contains(error.code) {
            if let url = request.url { await onTransportFailure?(url) }
            // Lost connections and timeouts may follow an accepted request; these may not.
            if reportsNeverSent, [.cannotConnectToHost, .cannotFindHost, .dnsLookupFailed, .notConnectedToInternet].contains(error.code) {
                throw NeverSent()
            }
            throw Error.transportUnavailable
        }
        guard let http = response as? HTTPURLResponse else { throw Error.invalidResponse }
        switch http.statusCode {
        case 200..<300:
            do { return try JSONDecoder().decode(T.self, from: data) } catch { throw Error.invalidResponse }
        case 401:
            throw Error.unauthorized
        default:
            let message = (try? JSONDecoder().decode([String: String].self, from: data))?["error"] ?? ""
            throw Error.server(status: http.statusCode, message: message)
        }
    }
}

/// Lets exactly one of several racing tasks resume a continuation.
private final class ResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func run(_ body: () -> Void) {
        lock.lock()
        defer { lock.unlock() }
        guard !done else { return }
        done = true
        body()
    }
}

/// Remembers that the tunnel answered and the Mac reported local models, so consecutive
/// dictations skip the health and capabilities round trips. The Mac re-checks the
/// X-Voxt-Model-Policy header on every request, so a stale "local" can't send audio to a
/// cloud model; any failure or a policy change drops the cache.
public final class PreflightCache: @unchecked Sendable {
    private let lock = NSLock()
    private let ttl: TimeInterval
    private let now: @Sendable () -> Date
    private var verified: (at: Date, allowCloud: Bool)?

    public init(ttl: TimeInterval = 60, now: @escaping @Sendable () -> Date = Date.init) {
        self.ttl = ttl
        self.now = now
    }

    func isFresh(allowCloud: Bool) -> Bool {
        lock.withLock {
            guard let verified, verified.allowCloud == allowCloud else { return false }
            return now().timeIntervalSince(verified.at) < ttl
        }
    }

    func markVerified(allowCloud: Bool) {
        lock.withLock { verified = (now(), allowCloud) }
    }

    public func invalidate() {
        lock.withLock { verified = nil }
    }
}
