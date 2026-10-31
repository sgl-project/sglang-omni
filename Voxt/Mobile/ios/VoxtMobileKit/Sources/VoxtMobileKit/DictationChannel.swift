import Foundation

/// The keyboard extension can't use the microphone, so it drives the main app through
/// the shared App Group: the keyboard sends commands, the app publishes state and results.
/// Both sides post a Darwin notification after writing so the other side re-reads.
public final class DictationChannel: @unchecked Sendable {
    /// The App Group the app and keyboard share. App Group IDs are unique across Apple
    /// teams, so each signer's build has its own (`VOXT_MOBILE_ID_PREFIX`); the build writes
    /// it into both Info.plists as `VoxtAppGroup`.
    public static let appGroup = appGroup(from: Bundle.main.infoDictionary)
    public static let defaultAppGroup = "group.com.voxt.VoxtMobile.shared"

    static func appGroup(from info: [String: Any]?) -> String {
        if let group = info?["VoxtAppGroup"] as? String, group.hasPrefix("group."), !group.contains("$(") { return group }
        // AltStore renames groups per team and lists the result here.
        if let group = (info?["ALTAppGroups"] as? [String])?.first, group.hasPrefix("group.") { return group }
        return defaultAppGroup
    }
    /// Darwin notification names are system-wide, so they carry the App Group: two builds
    /// signed by different teams on one phone then never wake each other.
    static func notification(_ name: String, appGroup: String) -> String { appGroup + "." + name }
    public static let commandNotification = notification("command", appGroup: appGroup)
    public static let stateNotification = notification("state", appGroup: appGroup)
    /// The app writes a heartbeat every second while its audio session is running.
    public static let heartbeatTimeout: TimeInterval = 3

    public enum Phase: String, Codable, Sendable {
        case idle, recording, processing, done, failed
    }

    public struct State: Codable, Equatable, Sendable {
        public var phase: Phase
        public var text: String?
        public var resultID: String?
        public var errorMessage: String?
        public var requestID: String?
        public var canRetry: Bool?
        /// The Mac couldn't be reached and the recording never left (the tunnel can be down
        /// while the app is in the background); in front, the app reconnects and resends it.
        public var reconnectInApp: Bool?

        public init(phase: Phase, text: String? = nil, resultID: String? = nil, errorMessage: String? = nil, requestID: String? = nil, canRetry: Bool? = nil) {
            self.phase = phase
            self.text = text
            self.resultID = resultID
            self.errorMessage = errorMessage
            self.requestID = requestID
            self.canRetry = canRetry
        }

        /// Whether the keyboard's Retry should open Voxt (voxtmobile://reconnect) instead of
        /// sending a command: a suspended app never reads it, and in front Voxt resends.
        public func retryOpensApp(sessionAlive: Bool) -> Bool {
            reconnectInApp == true || !sessionAlive
        }

        /// Whether the keyboard that made `ownRequestID` should open Voxt (voxtmobile://reconnect)
        /// for this failure; once per request.
        public func opensAppToReconnect(ownRequestID: String?, alreadyOpened: String?) -> Bool {
            guard phase == .failed, reconnectInApp == true, canRetry == true,
                  let requestID, requestID == ownRequestID else { return false }
            return requestID != alreadyOpened
        }
    }

    public struct Command: Codable, Equatable, Sendable {
        public enum Kind: String, Codable, Sendable { case start, stop, cancel, retry }
        public var kind: Kind
        public var sequence: Int
        public var options: CompositionOptions?
        public var requestID: String?
    }

    public struct Insertion: Equatable, Sendable {
        public var text: String
        public var resultID: String
        public var requestID: String?
    }

    public enum Notifier: Sendable {
        case darwin
        case none

        func post(_ name: String) {
            guard case .darwin = self else { return }
            CFNotificationCenterPostNotification(
                CFNotificationCenterGetDarwinNotifyCenter(),
                CFNotificationName(name as CFString), nil, nil, true
            )
        }
    }

    #if DEBUG
    /// Keeps only the latest response on this development device to diagnose language changes.
    /// Never compiled into release builds; no audio or credentials are stored.
    func recordDebugResponse(_ result: TranscriptionResult, mode: CompositionOptions.Mode) {
        struct Record: Encodable {
            let date: Date
            let mode: CompositionOptions.Mode
            let rawText: String
            let text: String
            let asrMs: Int?
            let enhanceMs: Int?
        }
        let record = Record(date: Date(), mode: mode, rawText: result.rawText, text: result.text,
                            asrMs: result.asrMs, enhanceMs: result.enhanceMs)
        defaults.set(try? JSONEncoder().encode(record), forKey: "dictation.debugLastResponse")
    }
    #endif

    private enum Key {
        static let state = "dictation.state"
        static let command = "dictation.command"
        static let heartbeat = "dictation.heartbeat"
        static let timings = "dictation.timings"
        static let recentTimings = "dictation.timings.recent"
    }

    private let defaults: UserDefaults
    private let notifier: Notifier

    public init(defaults: UserDefaults, notifier: Notifier = .darwin) {
        self.defaults = defaults
        self.notifier = notifier
    }

    /// Channel on the shared App Group. Nil when the container isn't available
    /// (keyboard without Full Access).
    public static func shared() -> DictationChannel? {
        UserDefaults(suiteName: appGroup).map { DictationChannel(defaults: $0) }
    }

    public var audioLevel: Float { defaults.float(forKey: "dictation.audioLevel") }
    public func publishAudioLevel(_ level: Float) { defaults.set(level, forKey: "dictation.audioLevel") }

    // MARK: State (app → keyboard)

    public var state: State {
        guard let data = defaults.data(forKey: Key.state),
              let state = try? JSONDecoder().decode(State.self, from: data) else { return State(phase: .idle) }
        return state
    }

    public func publish(_ state: State) {
        defaults.set(try? JSONEncoder().encode(state), forKey: Key.state)
        notifier.post(Self.stateNotification)
    }

    public func textToInsert(lastInsertedID: String?) -> Insertion? {
        let state = state
        guard state.phase == .done, let text = state.text, !text.isEmpty,
              let id = state.resultID, id != lastInsertedID else { return nil }
        return Insertion(text: text, resultID: id, requestID: state.requestID)
    }

    // MARK: Per-stage timing

    public struct Timings: Codable, Equatable, Sendable {
        public var requestID: String?
        public var mode: String
        public var audioMs: Int
        public var stopAt: Date
        public var encodeMs: Int
        public var preflightMs: Int?
        public var requestMs: Int?
        public var hostAsrMs: Int?
        public var hostEnhanceMs: Int?
        public var stopToResultMs: Int
        public var stopToInsertMs: Int?
        /// Recording loudness (dBFS peak, ms above the speech threshold) and the failure
        /// message when the request did not produce text.
        public var peakDb: Int?
        public var speechMs: Int?
        public var error: String?
    }

    /// Numbers only (no text or audio), so it is safe in release builds.
    public var lastTimings: Timings? {
        defaults.data(forKey: Key.timings).flatMap { try? JSONDecoder().decode(Timings.self, from: $0) }
    }

    public func publishTimings(_ timings: Timings) {
        defaults.set(try? JSONEncoder().encode(timings), forKey: Key.timings)
        var recent = recentTimings.filter { $0.requestID == nil || $0.requestID != timings.requestID }
        recent.append(timings)
        defaults.set(try? JSONEncoder().encode(Array(recent.suffix(20))), forKey: Key.recentTimings)
    }

    /// The last 20 requests' timings (numbers only), for reading back after a device run.
    public var recentTimings: [Timings] {
        defaults.data(forKey: Key.recentTimings).flatMap { try? JSONDecoder().decode([Timings].self, from: $0) } ?? []
    }

    /// Called by the keyboard when it inserts a result: completes stop → insert for that request.
    public func markInserted(requestID: String?, at date: Date = Date()) {
        guard var timings = lastTimings, requestID != nil, timings.requestID == requestID, timings.stopToInsertMs == nil else { return }
        timings.stopToInsertMs = Int((date.timeIntervalSince(timings.stopAt) * 1000).rounded())
        publishTimings(timings)
    }

    // MARK: Session liveness (app → keyboard)

    public func heartbeat(now: Date = Date()) {
        defaults.set(now.timeIntervalSince1970, forKey: Key.heartbeat)
    }

    public func endSession() {
        defaults.removeObject(forKey: Key.heartbeat)
        notifier.post(Self.stateNotification)
    }

    /// How long the keyboard trusts a recording or processing phase without a heartbeat.
    /// Longer than `heartbeatTimeout` so a busy main thread is not mistaken for a stopped app.
    public static let stalledAfter: TimeInterval = 10

    /// The app published recording or processing but stopped running, for example when iOS
    /// suspends it after a handoff; without this the keyboard would wait forever.
    public func isStalled(now: Date = Date()) -> Bool {
        guard [.recording, .processing].contains(state.phase) else { return false }
        let last = defaults.double(forKey: Key.heartbeat)
        return last <= 0 || now.timeIntervalSince1970 - last >= Self.stalledAfter
    }

    public func isSessionAlive(now: Date = Date()) -> Bool {
        let last = defaults.double(forKey: Key.heartbeat)
        guard last > 0 else { return false }
        return now.timeIntervalSince1970 - last < Self.heartbeatTimeout
    }

    // MARK: Commands (keyboard → app)

    public func send(_ kind: Command.Kind, options: CompositionOptions? = nil, requestID: String? = nil) {
        let next = (lastCommand?.sequence ?? 0) + 1
        defaults.set(try? JSONEncoder().encode(Command(kind: kind, sequence: next, options: options, requestID: requestID)), forKey: Key.command)
        notifier.post(Self.commandNotification)
    }

    public func pendingCommand(after sequence: Int) -> Command? {
        guard let command = lastCommand, command.sequence > sequence else { return nil }
        return command
    }

    private var lastCommand: Command? {
        defaults.data(forKey: Key.command).flatMap { try? JSONDecoder().decode(Command.self, from: $0) }
    }
}
