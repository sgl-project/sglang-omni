import Foundation

/// Short-lived App Group messages. The extension never receives the pairing token.
public final class CorrectionLearningChannel {
    public static let requestKey = "dictionary.learningRequest"
    public static let resultKey = "dictionary.learningResult"
    public struct Request: Codable, Sendable {
        public var id: UUID
        public var passage: CorrectionLearningRequest
        public var date: Date
    }
    public struct Result: Codable, Sendable {
        public var id: UUID
        public var correction: LearnedCorrection?
        public var error: String?
        public var date: Date
    }
    private let defaults: UserDefaults
    private let notifier: DictationChannel.Notifier

    public init(defaults: UserDefaults, notifier: DictationChannel.Notifier = .darwin) {
        self.defaults = defaults
        self.notifier = notifier
    }

    @discardableResult
    public func submit(_ passage: CorrectionLearningRequest, now: Date = Date()) -> UUID {
        let request = Request(id: UUID(), passage: passage, date: now)
        defaults.set(try? JSONEncoder().encode(request), forKey: Self.requestKey)
        notifier.post(DictationChannel.commandNotification)
        return request.id
    }

    public func takeRequest(now: Date = Date()) -> Request? {
        defer { defaults.removeObject(forKey: Self.requestKey) }
        guard let data = defaults.data(forKey: Self.requestKey),
              let request = try? JSONDecoder().decode(Request.self, from: data),
              now.timeIntervalSince(request.date) >= 0, now.timeIntervalSince(request.date) < 120,
              request.passage.isValid else { return nil }
        return request
    }

    public func publish(id: UUID, correction: LearnedCorrection?, error: String? = nil, now: Date = Date()) {
        defaults.set(try? JSONEncoder().encode(Result(id: id, correction: correction, error: error, date: now)), forKey: Self.resultKey)
        notifier.post(DictationChannel.stateNotification)
    }

    public func takeResult(for id: UUID, now: Date = Date()) -> Result? {
        guard let data = defaults.data(forKey: Self.resultKey), let result = try? JSONDecoder().decode(Result.self, from: data) else { return nil }
        defaults.removeObject(forKey: Self.resultKey)
        guard result.id == id, now.timeIntervalSince(result.date) >= 0, now.timeIntervalSince(result.date) < 120 else { return nil }
        return result
    }

    public func clear() {
        defaults.removeObject(forKey: Self.requestKey)
        defaults.removeObject(forKey: Self.resultKey)
    }
}

@MainActor
public final class CorrectionLearningProcessor {
    private let channel: CorrectionLearningChannel
    private let store: CorrectionLearningStore
    private let extract: @MainActor (CorrectionLearningRequest) async throws -> LearnedCorrection?
    private var running: UUID?
    private var generation = UUID()

    public init(channel: CorrectionLearningChannel, store: CorrectionLearningStore, extract: @escaping @MainActor (CorrectionLearningRequest) async throws -> LearnedCorrection?) {
        self.channel = channel
        self.store = store
        self.extract = extract
    }

    public func invalidate() {
        generation = UUID()
        running = nil
        channel.clear()
    }

    public func process() async {
        guard running == nil else { return }
        guard store.isEnabled else { channel.clear(); return }
        guard let request = channel.takeRequest() else { return }
        let consentRevision = store.consentRevision
        let operation = UUID(), requestGeneration = generation
        running = operation
        defer { if running == operation { running = nil } }
        do {
            let correction = try await extract(request.passage)
            guard generation == requestGeneration, store.isEnabled, store.consentRevision == consentRevision, Date().timeIntervalSince(request.date) < 120 else { return }
            if correction != nil, request.passage.validated(correction) == nil { throw VoxtClient.Error.invalidResponse }
            channel.publish(id: request.id, correction: correction)
        } catch {
            guard generation == requestGeneration, store.isEnabled, store.consentRevision == consentRevision else { return }
            channel.publish(id: request.id, correction: nil, error: error.localizedDescription)
        }
    }
}
