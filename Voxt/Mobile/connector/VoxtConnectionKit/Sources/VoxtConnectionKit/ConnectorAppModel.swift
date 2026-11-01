import Foundation
import Observation

public struct ConnectorConfiguration: Codable, Equatable, Sendable {
    public var allowedClients: [String]
    public var state: String?
    public init(allowedClients: [String], state: String?) {
        self.allowedClients = allowedClients; self.state = state
    }
}

@MainActor public protocol ConnectorConfigurationStore {
    func load() throws -> ConnectorConfiguration
    func save(_ configuration: ConnectorConfiguration) throws
}

/// Same-Wi-Fi pairing that runs while the Add iPhone QR code is shown.
@MainActor public protocol ConnectorLocalPairing: AnyObject {
    func stop()
}

/// A phone on the same Wi-Fi asking to pair; the user compares the code with the phone's.
public struct ConnectorPhoneRequest: Identifiable, Equatable, Sendable {
    public let id = UUID()
    public let code: String
    public let deviceName: String
}

@MainActor @Observable public final class ConnectorAppModel {
    public var port = "47911"
    public var token = ""
    public private(set) var session: ConnectorSession?
    public private(set) var configurationError: String?
    public private(set) var pairingCode: ConnectorPairingCode?
    public private(set) var pairingError: String?
    public private(set) var isPreparingPairing = false
    public private(set) var phoneRequest: ConnectorPhoneRequest?
    /// Called when a phone starts asking; the app brings Settings, which shows the Allow sheet, forward.
    @ObservationIgnored public var onPhoneRequest: () -> Void = {}
    /// Voxt version from the last successful check; kept while pairing restarts the runtime.
    private var connectedVersion: String?
    /// True while Voxt shows its Allow dialog for this Connector.
    public private(set) var isAuthorizing = false
    private var pairingStatus: ConnectorSession.Status?
    @ObservationIgnored private var pairingRuntime: (any ConnectorPairingRuntime)?
    @ObservationIgnored private var pairingExpiry: Task<Void, Never>?
    @ObservationIgnored private var pairingGeneration = UUID()
    @ObservationIgnored private var authenticated: (port: Int, token: String)?
    @ObservationIgnored private let makePairingRuntime: () -> any ConnectorPairingRuntime
    @ObservationIgnored private let runtimeURL: URL
    @ObservationIgnored private let store: any ConnectorConfigurationStore
    @ObservationIgnored private let health: (Int, String) async throws -> LocalInferenceClient.Health
    @ObservationIgnored private let authorizer: (any ConnectorAuthorizing)?
    @ObservationIgnored private var authorizationGeneration = UUID()
    @ObservationIgnored private var authorizationTask: Task<ConnectorGrant, Error>?
    @ObservationIgnored private let makeLocalPairing: @MainActor (URL, Date, @escaping @Sendable (String, String) async -> Bool) throws -> any ConnectorLocalPairing
    @ObservationIgnored private var localPairing: (any ConnectorLocalPairing)?
    @ObservationIgnored private var phoneAnswer: CheckedContinuation<Bool, Never>?
    @ObservationIgnored private var launchGeneration = UUID()
    @ObservationIgnored private var lastAuthorizationFailure: ConnectorLink.Failure?
    @ObservationIgnored private let preferences: any ConnectorPreferences
    @ObservationIgnored private let loginItem: any ConnectorLoginItem
    @ObservationIgnored private let sleep: @Sendable (Duration) async -> Void
    /// Last status read from the system; Settings refreshes it when it appears.
    public private(set) var loginItemStatus: ConnectorLoginItemStatus = .disabled
    private var loginItemError: String?

    public init(runtimeURL: URL, store: any ConnectorConfigurationStore,
                health: @escaping (Int, String) async throws -> LocalInferenceClient.Health = { port, token in
                    let url = URL(string: "http://127.0.0.1:\(port)")!
                    return try await LocalInferenceClient(baseURL: url, token: token).health()
                }, makePairingRuntime: (() -> any ConnectorPairingRuntime)? = nil,
                authorizer: (any ConnectorAuthorizing)? = ConnectorLink(),
                preferences: any ConnectorPreferences = UserDefaultsConnectorPreferences(),
                loginItem: (any ConnectorLoginItem)? = nil,
                sleep: @escaping @Sendable (Duration) async -> Void = { try? await Task.sleep(for: $0) },
                makeLocalPairing: (@MainActor (URL, Date, @escaping @Sendable (String, String) async -> Bool) throws -> any ConnectorLocalPairing)? = nil) {
        self.runtimeURL = runtimeURL; self.store = store; self.health = health; self.authorizer = authorizer
        self.preferences = preferences; self.loginItem = loginItem ?? MainAppLoginItem(); self.sleep = sleep
        self.makeLocalPairing = makeLocalPairing ?? { try AdvertisedLocalPairing(invitation: $0, expiresAt: $1, onRequest: $2) }
        self.makePairingRuntime = makePairingRuntime ?? {
            ConnectorPairingProcess(executableURL: runtimeURL, persist: { try store.save($0) })
        }
    }

    public var status: ConnectorSession.Status { pairingStatus ?? session?.status ?? .disconnected }
    public var canPair: Bool { !isPreparingPairing && isConnected }
    public var isConnected: Bool {
        switch status { case .localReady, .ready: return true; default: return false }
    }
    public var isConnecting: Bool { isAuthorizing || status == .checking }

    public enum ConnectFeedback: Equatable, Sendable { case idle, connecting, waitingForApproval, connected(String), failed(String) }
    /// Shown next to the Connect button.
    public var connectFeedback: ConnectFeedback {
        if let configurationError { return .failed(configurationError) }
        if isAuthorizing { return .waitingForApproval }
        switch status {
        case .disconnected: return .idle
        case .checking:
            // Pairing restarts the runtime after a verified connect; that isn't a new connect.
            return connectedVersion.map { .connected(Self.connectedText($0)) } ?? .connecting
        case let .localReady(version), let .ready(version, _): return .connected(Self.connectedText(version))
        case .failed: return .failed(statusText)
        }
    }
    /// Before Voxt has approved this Connector once, launch opens Settings and waits for a click,
    /// so Voxt's dialog never appears on its own.
    public var opensSettingsAtLaunch: Bool { !isConnected && !preferences.wasApproved }
    private static func authorizationMessage(_ error: any Error) -> String {
        switch error as? ConnectorLink.Failure {
        case .voxtNotRunning: return "Voxt didn't answer. Open Voxt, or enter its port and Access token."
        case .denied: return "Voxt didn't allow this Connector. Connect again, or enter the port and Access token."
        case .rejectedBySignature: return "Voxt doesn't accept this Connector build. Enter the port and Access token instead."
        case .untrustedVoxt: return "The app answering as Voxt isn't signed by Voxt. Nothing was sent."
        case .invalidReply, nil: return "Voxt sent an unexpected reply. Enter the port and Access token instead."
        }
    }
    private static func connectedText(_ version: String) -> String { "Connected to Voxt \(version)" }
    public var statusText: String {
        if let configurationError { return configurationError }
        if isAuthorizing { return "Waiting for Voxt to allow this Connector" }
        switch status {
        case .disconnected: return "Connect to Voxt on this Mac"
        case .checking: return "Connecting…"
        case let .localReady(version): return "Voxt \(version) is available"
        case let .ready(version, _): return "Ready for dictation · Voxt \(version)"
        case .failed(.unauthorized): return "The access token was rejected. Copy it from Voxt Settings."
        case .failed(.unavailable): return "Could not reach Voxt. Open it and, in Settings › General › Local API, turn on “Allow other apps to dictate through Voxt”."
        case .failed(.invalidResponse): return "This address did not return a valid Voxt response."
        case .failed(.runtime): return "The phone connection stopped. Connect again."
        }
    }

    /// With no Access token typed, asks Voxt for its port and token; Voxt shows one Allow dialog.
    /// A typed token keeps the manual path, which also covers builds Voxt won't accept.
    public func connect() async {
        launchGeneration = UUID()
        await performConnect()
    }

    /// After an earlier approval, connects without a window or dialog, waiting up to two minutes
    /// for Voxt to start (both may open at login). Returns whether the Connector is connected.
    public func connectAtLaunch() async -> Bool {
        guard preferences.wasApproved, !isConnected else { return isConnected }
        let attempt = UUID()
        launchGeneration = attempt
        for remaining in stride(from: Self.launchAttempts - 1, through: 0, by: -1) {
            token = ""
            await performConnect()
            guard launchGeneration == attempt else { return isConnected }
            if isConnected { return true }
            guard remaining > 0, lastAuthorizationFailure == .voxtNotRunning else { return false }
            await sleep(.seconds(5))
            guard launchGeneration == attempt else { return isConnected }
        }
        return false
    }
    private static let launchAttempts = 25

    private func performConnect() async {
        configurationError = nil; lastAuthorizationFailure = nil
        var remember = false
        cancelAuthorization()
        stopPairingRuntime()
        authenticated = nil; connectedVersion = nil
        session?.disconnect()
        if token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, let authorizer {
            let attempt = UUID()
            authorizationGeneration = attempt; isAuthorizing = true
            let result: Result<ConnectorGrant, Error>
            let request = Task { try await authorizer.authorize() }
            authorizationTask = request
            result = await request.result
            guard authorizationGeneration == attempt else { return }
            isAuthorizing = false
            switch result {
            case let .success(granted):
                port = String(granted.port); token = granted.token
                // Only Always Allow keeps the next launch free of Voxt's dialog.
                remember = granted.alwaysAllowed
                if !granted.alwaysAllowed { preferences.wasApproved = false }
            case let .failure(error):
                let failure = error as? ConnectorLink.Failure
                lastAuthorizationFailure = failure
                if failure == .denied || failure == .rejectedBySignature || failure == .untrustedVoxt {
                    preferences.wasApproved = false
                }
                configurationError = Self.authorizationMessage(error); return
            }
        }
        guard let number = Int(port), (1024...65534).contains(number) else {
            configurationError = "Enter the port shown in Voxt Settings."; return
        }
        let credential = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !credential.isEmpty else {
            configurationError = "Paste the access token from Voxt Settings."; return
        }
        let configuration: ConnectorConfiguration
        do {
            configuration = try store.load()
            guard Set(configuration.allowedClients).count == configuration.allowedClients.count,
                  configuration.allowedClients.allSatisfy(TailcatConfiguration.isPublicKey) else {
                throw TailcatError.invalidConfiguration
            }
        } catch {
            configurationError = "Could not read the saved phone connections."; return
        }
        let health = self.health, runtimeURL = self.runtimeURL, store = self.store
        let next = ConnectorSession(health: { try await health(number, credential) }, makeRuntime: {
            guard !configuration.allowedClients.isEmpty else { return nil }
            return TailcatRuntime(executableURL: runtimeURL, localPort: number,
                allowedClients: configuration.allowedClients, state: configuration.state,
                persistState: { state in
                    try store.save(.init(allowedClients: configuration.allowedClients, state: state))
                })
        })
        session = next
        await next.connect()
        guard session === next else { return }
        switch next.status {
        case let .ready(version, _), let .localReady(version):
            authenticated = (number, credential); connectedVersion = version
            if remember { preferences.wasApproved = true }
        default: break
        }
    }

    public var opensAtLogin: Bool { loginItemStatus == .enabled }
    public var loginItemMessage: String? {
        if loginItemStatus == .requiresApproval { return "Allow Voxt Connector in System Settings → General → Login Items." }
        return loginItemError
    }
    public func setOpensAtLogin(_ enabled: Bool) {
        do {
            if enabled { try loginItem.register() } else { try loginItem.unregister() }
            loginItemError = nil
        } catch {
            loginItemError = enabled ? "macOS didn't add Voxt Connector to Login Items."
                                     : "macOS didn't remove Voxt Connector from Login Items."
        }
        // Right after the user's own click, read it directly so the switch reflects the result.
        loginItemStatus = loginItem.status
    }
    /// The user can change Login Items in System Settings; re-read when Settings appears.
    public func refreshLoginItem() async { loginItemStatus = await loginItem.currentStatus() }

    /// Stops waiting for Voxt's Allow dialog; a late answer is ignored.
    public func cancelAuthorization() {
        authorizationGeneration = UUID(); isAuthorizing = false
        authorizationTask?.cancel(); authorizationTask = nil
    }

    public func disconnect() {
        cancelAuthorization()
        stopPairingRuntime(); authenticated = nil; connectedVersion = nil
        session?.disconnect(); token = ""; configurationError = nil
    }

    public func beginPairing() async {
        guard canPair, let authenticated else { return }
        pairingError = nil; isPreparingPairing = true
        pairingExpiry?.cancel(); pairingExpiry = nil; pairingCode = nil
        let attempt = pairingGeneration
        do {
            if pairingRuntime == nil {
                let configuration = try store.load()
                let version: String
                switch status {
                case let .localReady(v), let .ready(v, _): version = v
                default: return
                }
                session?.disconnect()
                pairingStatus = .checking
                let runtime = makePairingRuntime()
                pairingRuntime = runtime
                runtime.onEvent = { [weak self] event in
                    guard let self, self.pairingGeneration == attempt else { return }
                    self.receive(event, version: version)
                }
                try await runtime.start(localPort: authenticated.port, token: authenticated.token, configuration: configuration)
            }
            guard pairingGeneration == attempt else { return }
            try pairingRuntime?.beginPairing()
        } catch {
            guard pairingGeneration == attempt else { return }
            pairingError = "Could not start pairing. Connect again."
            pairingStatus = .failed(.runtime)
        }
        if pairingGeneration == attempt { isPreparingPairing = false }
    }

    /// The user's answer in the "Allow this iPhone?" sheet.
    public func answerPhoneRequest(_ allow: Bool) {
        guard let request = phoneRequest else { return }
        resolvePhoneRequest(request.id, allow)
    }

    /// A phone on the same Wi-Fi asks to pair: shows the Allow sheet and waits for the answer.
    public func askAboutPhone(code: String, deviceName: String) async -> Bool {
        // One phone at a time; the coordinator also enforces this.
        guard phoneRequest == nil else { return false }
        let digits = code.filter(\.isNumber)
        let shown = digits.count == 6 ? digits.prefix(3) + " " + digits.suffix(3) : Substring(code)
        let request = ConnectorPhoneRequest(code: String(shown), deviceName: deviceName)
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                phoneRequest = request; phoneAnswer = continuation
                onPhoneRequest()
            }
        } onCancel: {
            // The coordinator gave up (expiry or Cancel): close the sheet as Don't Allow.
            Task { @MainActor [weak self] in self?.resolvePhoneRequest(request.id, false) }
        }
    }

    private func resolvePhoneRequest(_ id: UUID, _ allow: Bool) {
        guard phoneRequest?.id == id else { return }
        let answer = phoneAnswer
        phoneRequest = nil; phoneAnswer = nil
        answer?.resume(returning: allow)
    }

    private func startLocalPairing(_ code: ConnectorPairingCode) {
        stopLocalPairing()
        // The QR code keeps working if the local network listener can't start.
        localPairing = try? makeLocalPairing(code.url, code.expiresAt) { [weak self] code, deviceName in
            await self?.askAboutPhone(code: code, deviceName: deviceName) ?? false
        }
    }

    private func stopLocalPairing() {
        localPairing?.stop(); localPairing = nil
        if let request = phoneRequest { resolvePhoneRequest(request.id, false) }
    }

    public func cancelPairing() {
        pairingExpiry?.cancel(); pairingExpiry = nil; pairingCode = nil
        stopLocalPairing()
        do { try pairingRuntime?.cancelPairing() }
        catch { pairingError = "Could not close pairing. Connect again." }
    }

    private func receive(_ event: ConnectorPairingProcess.Event, version: String) {
        switch event {
        case let .ready(address):
            pairingStatus = address.isEmpty ? .localReady(version: version) : .ready(version: version, address: address)
        case let .pairing(code):
            pairingCode = code; pairingExpiry?.cancel()
            startLocalPairing(code)
            let generation = pairingGeneration
            pairingExpiry = Task { @MainActor [weak self] in
                do { try await Task.sleep(for: .seconds(max(0, code.expiresAt.timeIntervalSinceNow))) }
                catch { return }
                guard let self, self.pairingGeneration == generation else { return }
                self.cancelPairing()
            }
        case let .paired(address, _):
            pairingStatus = .ready(version: version, address: address)
            // Keep the short-lived session open for lost-response retries until
            // explicit cancellation or expiry. Do not close before HTTP is sent.
            pairingCode = nil
            stopLocalPairing()
        case .pairingClosed: pairingCode = nil; stopLocalPairing()
        case .failed:
            pairingCode = nil; stopLocalPairing()
            pairingError = "Pairing failed. Connect again."
            pairingStatus = .failed(.runtime)
        }
    }

    private func stopPairingRuntime() {
        pairingGeneration = UUID()
        pairingExpiry?.cancel(); pairingExpiry = nil
        pairingRuntime?.stop(); pairingRuntime = nil
        stopLocalPairing()
        pairingCode = nil; pairingError = nil; pairingStatus = nil; isPreparingPairing = false
    }

}
