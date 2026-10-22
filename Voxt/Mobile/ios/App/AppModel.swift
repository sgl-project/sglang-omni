import Foundation
import UIKit
import Observation
import VoxtMobileKit
import VoxtTailcat

typealias HistoryItem = TranscriptRecord

@MainActor
@Observable
final class AppModel {
    enum ConnectionState: Equatable {
        case unpaired
        case checking
        case connected(version: String)
        /// Reachable, but Voxt there uses a cloud model, so dictation is refused on this phone.
        case cloudModel(version: String)
        case failed(String)

        /// Folds a dictation outcome into the connection shown on the status card.
        static func after(phase: DictationChannel.Phase, failure: (any Error)?, current: ConnectionState) -> ConnectionState {
            switch (phase, current) {
            case let (.failed, .connected(version)) where (failure as? VoxtClient.Error) == .cloudModelSelected:
                return .cloudModel(version: version)
            case let (.done, .cloudModel(version)):
                return .connected(version: version)
            default:
                return current
            }
        }
    }

    static func statusCardState(for connection: ConnectionState) -> MacStatusCard.State? {
        switch connection {
        case .unpaired: nil
        case .checking: .connecting
        case let .connected(version): .connected(version: version)
        case let .cloudModel(version): .cloudModel(version: version)
        case let .failed(message): .unreachable(reason: message)
        }
    }

    private(set) var server: ServerConfig?
    private(set) var connection: ConnectionState = .unpaired
    private(set) var audioLevel: Float = 0
    private var levelTicker: Timer?
    var isLevelTimerRunning: Bool { levelTicker?.isValid == true }
    var isSessionTimerRunning: Bool { ticker?.isValid == true }
    private let schedulesTimers: Bool
    private var levelTick = 0
    private(set) var phase: DictationChannel.Phase = .idle
    private(set) var errorMessage: String?
    private(set) var history: [HistoryItem] = []
    /// Recordings the Mac never turned into text, newest first; History offers to send them again.
    private(set) var unsentRecordings: [SavedRecording] = []
    private(set) var historyRetention: TranscriptHistory.Retention = .forever
    private(set) var microphoneReadiness: MicrophoneReadiness = .fiveMinutes
    private(set) var latestText: String?
    private let historyStore: TranscriptHistory
    private let pairingStore: PairingStore
    private let tailcatIdentityStore: TailcatIdentityStore
    private(set) var sessionOpen = false
    /// Increments each time a dictation started from the app's own mic button produces text,
    /// so RootView can switch to the History tab to show it. Keyboard dictations don't count.
    private(set) var appResultCount = 0
    var showsKeyboardHandoff = false
    /// The keyboard's "Choose translation languages" pill opens voxtmobile://translation.
    var showsTranslation = false
    /// voxtmobile://settings, from the Voxt name at the keyboard's top left.
    var showsSettings = false
    /// When Voxt last became the active app; nil while it isn't.
    private(set) var activeSince: Date?

    func sceneBecameActive(at date: Date = Date()) { activeSince = date }
    func sceneLeftActive() {
        activeSince = nil
        // iOS may stop the app once it's in the background; save what's been said so far.
        controller.saveProgress()
    }
    private var appDictationActive = false
    var showsAppRecording: Bool { appDictationActive && (phase == .recording || phase == .processing) }
    private var recordingStartedAt: Date?
    private let now: () -> Date
    private(set) var preferences: CompositionPreferences
    private(set) var keyboardSettings = KeyboardSettings()
    private(set) var onboarding = OnboardingProgress()
    private(set) var canRetry = false
    private(set) var historySyncEnabled = false
    private(set) var isSyncingHistory = false
    private(set) var historySyncMessage: String?
    private let historySyncStore: HistorySyncStore
    @ObservationIgnored private var historySyncCoordinator: HistorySyncCoordinator?
    private var historySyncGeneration = UUID()
    private var nextHistorySync = Date.distantPast
    @ObservationIgnored private(set) var floatingWindow: FloatingDictationWindow?

    let controller: DictationController
    private let recordings: RecordingStore?
    /// Starts a background task and returns what ends it; tests inject one.
    @ObservationIgnored private let beginBackgroundTask: (@MainActor (String) -> @MainActor () -> Void)?
    private let usesMicrophone: Bool
    private let channel: DictationChannel
    private let defaults: UserDefaults
    private let compositionStore: CompositionStore
    private let developerPrompts: DeveloperPromptStore
    private var ticker: Timer?
    @ObservationIgnored private var learningProcessor: CorrectionLearningProcessor?
    /// Injected test models skip system interfaces; timer tests can opt into scheduling.
    /// the observer holds self unretained and is never removed, so a deallocated
    /// test model must not stay reachable from system callbacks.
    private let ownsSystemInterfaces: Bool
    private var connectionGeneration = UUID()
    @ObservationIgnored private var pairingConnection: PairingConnection?
    private let connectionFactory: @MainActor (ServerConfig) -> PairingConnection
    private let healthProbe: (@MainActor (ServerConfig) async throws -> VoxtClient.Health)?
    private let capabilitiesProbe: (@MainActor (ServerConfig) async throws -> VoxtClient.Capabilities)?
    /// Endpoints the Mac listed; nil until it answers, or when it lists none (older Voxt).
    private(set) var macEndpoints: Set<String>?

    private enum Key {
        static let server = "server.config"
        static let history = "history.items"
    }

    /// Tests inject a capture and a private defaults suite so no credential,
    /// app-group state, or microphone is touched.
    init(injectedCapture: AudioCapturing? = nil, injectedDefaults: UserDefaults? = nil, injectedTokens: (any TokenStoring)? = nil, injectedNow: (() -> Date)? = nil,
         injectedHistoryExchange: (@MainActor (HistorySyncRequest) async throws -> HistorySyncResponse)? = nil,
         injectedHealthProbe: (@MainActor (ServerConfig) async throws -> VoxtClient.Health)? = nil,
         injectedCapabilitiesProbe: (@MainActor (ServerConfig) async throws -> VoxtClient.Capabilities)? = nil,
         injectedConnectionFactory: (@MainActor (ServerConfig) -> PairingConnection)? = nil,
         injectedIdentityTokens: (any TokenStoring)? = nil,
         injectedCorrectionExchange: (@MainActor (CorrectionLearningRequest) async throws -> LearnedCorrection?)? = nil,
         injectedTimerScheduling: Bool? = nil,
         injectedPreferredLanguages: [String]? = nil,
         injectedRecordings: RecordingStore? = nil,
         injectedBackgroundTask: (@MainActor (String) -> @MainActor () -> Void)? = nil) {
        beginBackgroundTask = injectedBackgroundTask
        connectionFactory = injectedConnectionFactory ?? Self.makeConnection
        healthProbe = injectedHealthProbe
        capabilitiesProbe = injectedCapabilitiesProbe
        let defaults = injectedDefaults ?? UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard
        let isInjected = injectedCapture != nil || injectedDefaults != nil || injectedTokens != nil || injectedNow != nil || injectedHistoryExchange != nil || injectedHealthProbe != nil || injectedConnectionFactory != nil || injectedIdentityTokens != nil || injectedCorrectionExchange != nil
        now = injectedNow ?? Date.init
        let tokens: any TokenStoring = injectedTokens ?? (isInjected ? TransientTokenStore() : KeychainTokenStore())
        let pairingStore = PairingStore(defaults: defaults, tokens: tokens)
        self.pairingStore = pairingStore
        let identityTokens: any TokenStoring = injectedIdentityTokens ?? (isInjected ? TransientTokenStore() : KeychainTokenStore(service: KeychainTokenStore.service("tailcat-identity")))
        self.tailcatIdentityStore = TailcatIdentityStore(read: { try identityTokens.read() },
                                                       write: { try identityTokens.write($0) },
                                                       generate: TailcatIdentity.generate)
        var capture: AudioCapturing = injectedCapture ?? MicrophoneCapture()
        #if DEBUG
        if ProcessInfo.processInfo.environment["VOXT_TEST_RESET"] == "1" {
            defaults.set("en", forKey: VoxtL10n.preferenceKey)
            Self.resetPinyinForTests(defaults: defaults,
                                     container: FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: DictationChannel.appGroup))
            try? pairingStore.clear()
            defaults.removeObject(forKey: Key.history)
            defaults.removeObject(forKey: "typingWordsPerMinute")
            HistorySyncStore(defaults: defaults).resetPairing()
            let developerPrompts = DeveloperPromptStore(defaults: defaults)
            developerPrompts.isEnabled = false
            developerPrompts.resetAll()
            let personalStyle = PersonalStyleStore(defaults: defaults)
            personalStyle.isEnabled = false
            personalStyle.clear()
            for key in ["history.retention", "dictionary.entries", "composition.preferences", "composition.gate", "appearance", "language.variants", "microphone.disableAfter", TextKeyboardState.storageKey, KeyboardSettings.storageKey, KeyboardStatus.storageKey, OnboardingProgress.storageKey, CorrectionLearningStore.enabledKey, CorrectionLearningStore.revisionKey, CorrectionLearningStore.recordsKey, CorrectionLearningChannel.requestKey, CorrectionLearningChannel.resultKey] {
                defaults.removeObject(forKey: key)
            }
        }
        Self.applyTestLanguage(ProcessInfo.processInfo.environment, to: defaults)
        // UI tests: "text|minutes ago" per line, fake dictations for Home's Recent list.
        if let fixture = ProcessInfo.processInfo.environment["VOXT_TEST_HISTORY"] {
            let store = TranscriptHistory(defaults: defaults)
            if store.items.isEmpty {
                for line in fixture.split(separator: "\n") {
                    let parts = line.split(separator: "|")
                    let minutes = parts.count > 1 ? Double(parts[1]) ?? 0 : 0
                    store.append(text: String(parts[0]), now: Date().addingTimeInterval(-minutes * 60))
                }
            }
        }
        #if targetEnvironment(simulator)
        if let fixture = ProcessInfo.processInfo.environment["VOXT_TEST_SERVER"],
           let data = fixture.data(using: .utf8),
           let config = try? JSONDecoder().decode(ServerConfig.self, from: data),
           config.baseURL.host == "127.0.0.1", config.tailcat == nil {
            try? pairingStore.save(config)
        }
        #endif
        if injectedCapture == nil, let fileCapture = FileAudioCapture.fromEnvironment() {
            capture = fileCapture
        }
        #endif
        let channel = DictationChannel(defaults: defaults)
        let compositionStore = CompositionStore(defaults: defaults)
        self.defaults = defaults
        self.historyStore = TranscriptHistory(defaults: defaults)
        self.historySyncStore = HistorySyncStore(defaults: defaults)
        self.historySyncEnabled = historySyncStore.isEnabled
        macEndpoints = MacFeatures.endpoints(in: defaults)
        self.historyRetention = historyStore.retention
        self.compositionStore = compositionStore
        let developerPrompts = DeveloperPromptStore(defaults: defaults)
        self.developerPrompts = developerPrompts
        developerModeEnabled = developerPrompts.isEnabled
        customPrompts = Dictionary(uniqueKeysWithValues: CompositionOptions.Mode.allCases.compactMap { mode in
            developerPrompts.prompt(for: mode).map { (mode, $0) }
        })
        self.preferences = compositionStore.loadPreferences()
        var preferredLanguages = injectedPreferredLanguages ?? Locale.preferredLanguages
        #if DEBUG
        if injectedPreferredLanguages == nil, let languages = ProcessInfo.processInfo.environment["VOXT_TEST_PREFERRED_LANGUAGES"] {
            preferredLanguages = languages.split(separator: ",").map(String.init)
        }
        #endif
        keyboardSettings = KeyboardSettings.load(from: defaults, preferredLanguages: preferredLanguages)
        self.channel = channel
        ownsSystemInterfaces = !isInjected
        schedulesTimers = injectedTimerScheduling ?? !isInjected
        usesMicrophone = capture is MicrophoneCapture
        let recordings = injectedRecordings ?? (isInjected ? nil : RecordingStore.appDefault())
        #if DEBUG
        if ProcessInfo.processInfo.environment["VOXT_TEST_RESET"] == "1" {
            for saved in recordings?.all ?? [] { recordings?.remove(saved.id) }
        }
        #endif
        recordings?.recoverInterrupted()
        self.recordings = recordings
        controller = DictationController(channel: channel, capture: capture, transcriber: nil, recordings: recordings)
        #if DEBUG
        if let delay = ProcessInfo.processInfo.environment["VOXT_TEST_RESULT_DELAY_MS"].flatMap(Double.init) {
            controller.resultDelay = delay / 1000
        }
        #endif
        microphoneReadiness = MicrophoneReadiness.load(from: defaults)
        controller.idleTimeout = microphoneReadiness.timeout

        do {
            if let server = try pairingStore.load() {
                self.server = server
                // Keep the already-published result available on Home while the new
                // controller clears stale keyboard state. No history entry is created.
                if channel.state.phase == .done { latestText = channel.state.text }
                let connection = connectionFactory(server)
                pairingConnection = connection
                controller.setTranscriber(connection.client(allowCloudModels: { [compositionStore] in compositionStore.loadPreferences().allowCloudModels },
                                                        customPrompt: { [developerPrompts] mode in developerPrompts.override(for: mode) }))
            }
        } catch {
            connection = .failed(Self.describe(error, tailcat: self.server?.tailcat != nil))
        }
        history = historyStore.items
        onboarding = OnboardingProgress.load(from: defaults, paired: server != nil)
        #if DEBUG
        if let progress = ProcessInfo.processInfo.environment["VOXT_TEST_ONBOARDING"].flatMap(OnboardingProgress.init(testLaunchSetting:)) {
            onboarding = progress
        }
        #endif
        if server != nil, latestText == nil {
            latestText = history.first?.text
        }

        controller.onChange = { [weak self] in self?.controllerChanged() }
        // A recording cut off by a kill or a lost connection: the keyboard's Retry sends it.
        if server != nil { controller.restoreUnsentRecording() }
        unsentRecordings = recordings?.unprocessed ?? []
        historySyncCoordinator = HistorySyncCoordinator(store: historySyncStore, exchange: injectedHistoryExchange ?? { [weak self] request in
            guard let connection = self?.pairingConnection else { throw VoxtClient.Error.unauthorized }
            return try await connection.client(allowCloudModels: { [compositionStore] in compositionStore.loadPreferences().allowCloudModels }).syncHistory(request)
        })
        learningProcessor = CorrectionLearningProcessor(
            channel: CorrectionLearningChannel(defaults: defaults), store: learning,
            extract: injectedCorrectionExchange ?? { [weak self] passage in
            guard let connection = self?.pairingConnection else { throw VoxtClient.Error.unauthorized }
            return try await connection.client(allowCloudModels: { [compositionStore] in compositionStore.loadPreferences().allowCloudModels }).correction(for: passage)
        })
        if ownsSystemInterfaces {
            let window = FloatingDictationWindow()
            floatingWindow = window
            window.onDidStart = { [weak self] in self?.controller.setArmed(true) }
            window.onDidStop = { [weak self] in self?.controller.setArmed(false) }
            window.onFailure = { [weak self] message in
                self?.controller.setArmed(false)
                self?.errorMessage = message
            }
            observeKeyboardCommands()
        }
        updateTimers()
    }

    /// Meter work exists only during capture. PiP readiness still needs a heartbeat
    /// when the microphone is closed; history sync also needs periodic polling.
    private func updateTimers() {
        guard schedulesTimers else { return }
        if phase == .recording {
            if levelTicker == nil {
                levelTicker = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] timer in
                    guard self != nil else { timer.invalidate(); return }
                    MainActor.assumeIsolated {
                        guard let self else { return }
                        guard self.phase == .recording else { return }
                        let level = self.controller.audioLevel
                        if self.audioLevel != level { self.audioLevel = level }
                        self.levelTick += 1
                        if self.levelTick % 3 == 0, self.phase == .recording || self.channel.audioLevel != 0 {
                            self.channel.publishAudioLevel(self.audioLevel)
                        }
                    }
                }
            }
        } else {
            levelTicker?.invalidate()
            levelTicker = nil
        }
        if controller.isSessionOpen || controller.isArmed || historySyncEnabled || phase == .processing {
            if ticker == nil {
                ticker = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] timer in
                    guard self != nil else { timer.invalidate(); return }
                    MainActor.assumeIsolated {
                        guard let self else { return }
                        self.controller.tick()
                        Task { await self.processPendingCorrectionLearning() }
                        self.sessionOpen = self.controller.isSessionOpen
                        if self.historySyncEnabled, Date() >= self.nextHistorySync {
                            self.nextHistorySync = Date().addingTimeInterval(30)
                            Task { await self.syncHistory() }
                        }
                    }
                }
            }
        } else {
            ticker?.invalidate()
            ticker = nil
        }
    }

    func processPendingCorrectionLearning() async { await learningProcessor?.process() }

    var learning: CorrectionLearningStore { CorrectionLearningStore(defaults: defaults) }

    var dictionary: PersonalDictionary { PersonalDictionary(defaults: defaults) }

    var languageVariants: [String] { defaults.stringArray(forKey: "language.variants") ?? [] }

    func setLanguageVariant(_ value: String?, for base: String) {
        var updated = languageVariants.filter { $0.split(separator: "-").first.map(String.init) != base }
        if let value { updated.append(value) }
        guard (try? CompositionOptions(languageVariants: updated).validate()) != nil else { return }
        defaults.set(updated, forKey: "language.variants")
    }

    private static func makeConnection(_ server: ServerConfig) -> PairingConnection {
        guard let tailcat = server.tailcat else { return PairingConnection(server: server) }
        let bridge = TailcatBridge()
        let options = TailcatOptions(address: tailcat.address, clientKey: tailcat.clientKey, remotePort: tailcat.remotePort)
        return PairingConnection(server: server, resolve: { try await bridge.endpoint(for: options) },
                                 reset: { await bridge.close() }, close: { await bridge.invalidate() })
    }


    // MARK: Pairing

    // Same-Wi-Fi pairing without the QR.
    private(set) var localPairingSearching = false
    private(set) var localPairingCode: String?
    private(set) var localPairingError: String?
    /// Finds the Connector, shows the code, returns the invitation. Real Bonjour runs only from
    /// the Find my Mac button; tests inject a fake so no Local Network prompt can appear.
    @ObservationIgnored var localPairingSearch: @MainActor (@escaping @Sendable (String) -> Void) async throws -> URL = { showCode in
        let browser = LocalPairingBrowser()
        guard let endpoint = await browser.firstConnector() else { throw LocalPairingBrowserFailure.notFound }
        let channel = browser.channel(to: endpoint)
        defer { channel.close() }
        return try await LocalPairingClient().pair(channel: channel, deviceName: UIDevice.current.name, showCode: showCode)
    }
    @ObservationIgnored var openPairingLink: ((URL) -> Void)?

    func findMacOnWiFi() async {
        guard !localPairingSearching else { return }
        localPairingSearching = true
        localPairingError = nil
        localPairingCode = nil
        defer { localPairingSearching = false; localPairingCode = nil }
        do {
            let invitation = try await localPairingSearch { [weak self] code in
                if Thread.isMainThread { MainActor.assumeIsolated { self?.localPairingCode = code } }
                else { Task { @MainActor in self?.localPairingCode = code } }
            }
            if let openPairingLink { openPairingLink(invitation) } else { handle(url: invitation) }
        } catch LocalPairingBrowserFailure.notFound {
            localPairingError = "No Mac found on this Wi-Fi. Use the QR instead."
        } catch LocalPairingClient.Failure.denied {
            localPairingError = "Your Mac said no. Scan the QR code in Voxt Connector instead."
        } catch LocalPairingClient.Failure.timedOut {
            localPairingError = "No answer from your Mac. Scan the QR code in Voxt Connector instead."
        } catch {
            localPairingError = "Couldn't reach your Mac. Use the QR instead."
        }
    }

    /// The pairing screen's QR scanner.
    private(set) var pairingScan = PairingScan()

    func cameraAccessChanged(_ access: PairingScan.Camera) {
        pairingScan.camera(access)
    }

    /// Text decoded by the in-app scanner; a Voxt link pairs exactly as a Camera-app scan does.
    func scanned(_ text: String) {
        guard let link = pairingScan.decoded(text) else { return }
        localPairingError = nil
        if let openPairingLink { openPairingLink(link) } else { handle(url: link) }
    }
    @ObservationIgnored private var pairingTask: Task<Void, Never>?
    @ObservationIgnored private var pairingAttempt = UUID()

    func handle(url: URL) {
        guard url.scheme == "voxtmobile" else { return }
        switch url.host {
        case "connect":
            pairingTask?.cancel()
            pairingAttempt = UUID()
            let attempt = pairingAttempt
            pairingTask = Task { await connect(url: url, attempt: attempt) }
        case "session":
            Task { await startFromKeyboard() }
        case "reconnect":
            Task { await keepingAppAlive("voxt.resend") { await reconnectAndRetry() } }
        case "translation":
            showsTranslation = true
        case "settings":
            showsSettings = true
        default:
            break
        }
    }

    private func connect(url: URL, attempt: UUID) async {
        let bridge = TailcatBridge()
        do {
            let bootstrap = try PairingBootstrap(url: url)
            connection = .checking
            let identity = try tailcatIdentityStore.identity()
            let endpoint = try await bridge.endpoint(for: .init(address: bootstrap.address, clientKey: bootstrap.temporaryKey, remotePort: bootstrap.port))
            let config = try await PairingExchange.exchange(endpoint: endpoint, proof: bootstrap.proof,
                                                            publicKey: identity.publicKey, privateKey: identity.privateKey)
            try Task.checkCancellation()
            guard pairingAttempt == attempt else { await bridge.invalidate(); return }
            try save(config)
            pairingScan.pairingFinished(error: nil)
        } catch {
            if pairingAttempt == attempt && !Task.isCancelled {
                #if DEBUG
                NSLog("Connector pairing failed: %@, code=%ld, credentialStatus=%d", String(describing: type(of: error)), (error as NSError).code, (error as? KeychainTokenStore.Failure)?.status ?? 0)
                #endif
                let message = "Couldn't pair. Scan a new QR in Voxt Connector."
                errorMessage = message
                connection = .failed(message)
                pairingScan.pairingFinished(error: message)
            }
        }
        await bridge.invalidate()
    }

    func unpair() {
        pairingTask?.cancel()
        pairingAttempt = UUID()
        #if DEBUG
        defaults.removeObject(forKey: "dictation.debugLastResponse")
        #endif
        do { try pairingStore.clear() } catch {
            connection = .failed(Self.describe(error, tailcat: self.server?.tailcat != nil))
            return
        }
        connectionGeneration = UUID()
        learningProcessor?.invalidate()
        pairingConnection?.invalidate()
        pairingConnection = nil
        server = nil
        MacFeatures.clear(in: defaults)
        macEndpoints = nil
        resetHistorySyncPairing()
        latestText = nil
        floatingWindow?.stop()
        controller.setTranscriber(nil)
        connection = .unpaired
    }

    /// The keyboard opens voxtmobile://reconnect when the Mac couldn't be reached from the
    /// background, where the tunnel can stay down. In front the tunnel comes back; then the
    /// recording that never left is sent again, and the keyboard inserts it on return.
    func reconnectAndRetry(attempts: Int = 4) async {
        await checkConnection(attempts: attempts)
        guard case .connected = connection, controller.canRetry else { return }
        await controller.retry()
    }

    /// Network errors are retried a few times: Tailscale on the phone often needs a moment
    /// to reconnect when the app comes to the foreground.
    func checkConnection(attempts: Int = 4) async {
        guard let server else { connection = .unpaired; return }
        let generation = UUID()
        connectionGeneration = generation
        connection = .checking
        for attempt in 1...attempts {
            guard generation == connectionGeneration, !Task.isCancelled else { return }
            do {
                let health: VoxtClient.Health
                if let healthProbe { health = try await healthProbe(server) }
                else {
                    guard let pairingConnection else { throw VoxtClient.Error.unauthorized }
                    health = try await pairingConnection.client().health()
                }
                guard generation == connectionGeneration, !Task.isCancelled else { return }
                connection = .connected(version: health.version)
                await refreshMacFeatures(server, generation: generation)
                return
            } catch let error as URLError where attempt < attempts {
                guard generation == connectionGeneration, !Task.isCancelled else { return }
                _ = error
                try? await Task.sleep(for: .seconds(2))
            } catch {
                guard generation == connectionGeneration, !Task.isCancelled else { return }
                connection = .failed(Self.describe(error, tailcat: self.server?.tailcat != nil))
                return
            }
        }
    }

    /// Settings shows a feature only when the Mac serves its endpoint.
    func macServes(_ feature: VoxtClient.Feature) -> Bool { macEndpoints?.contains(feature.rawValue) ?? true }

    /// Reads which endpoints the Mac serves and shares them with the keyboard. A failed read
    /// keeps what was known; the client still refuses an endpoint the Mac doesn't list.
    private func refreshMacFeatures(_ server: ServerConfig, generation: UUID) async {
        let capabilities: VoxtClient.Capabilities
        do {
            if let capabilitiesProbe { capabilities = try await capabilitiesProbe(server) }
            else if healthProbe == nil, let pairingConnection { capabilities = try await pairingConnection.client().capabilities() }
            else { return }
        } catch { return }
        guard generation == connectionGeneration else { return }
        MacFeatures.save(capabilities, to: defaults)
        macEndpoints = capabilities.endpoints
    }

    // MARK: Dictation

    func setLiteralDictation(_ enabled: Bool) { updatePreferences { $0.literalDictation = enabled } }
    #if DEBUG
    /// VOXT_TEST_RESET: pinyin as a new install has it (fuzzy off, nothing learned), so a test
    /// that stopped early can't change the next one.
    nonisolated static func resetPinyinForTests(defaults: UserDefaults, container: URL?) {
        PinyinSettings.setFuzzy(false, in: defaults)
        PinyinSettings.clearLearnedWords(in: container)
    }

    /// Screenshot runs: VOXT_TEST_LANGUAGE=<shipped code> sets the interface language in the App
    /// Group, which the keyboard reads too.
    nonisolated static func applyTestLanguage(_ environment: [String: String], to defaults: UserDefaults) {
        guard let code = environment["VOXT_TEST_LANGUAGE"], VoxtL10n.shippedLanguages.contains(code) else { return }
        defaults.set(code, forKey: VoxtL10n.preferenceKey)
    }
    #endif

    /// Settings › Writing › Clean up my words: on is today's literal dictation off.
    var cleansUpWords: Bool { !preferences.literalDictation }
    func setCleansUpWords(_ enabled: Bool) { setLiteralDictation(!enabled) }

    /// The first-open screen to show, or nil for Home.
    var onboardingScreen: OnboardingProgress.Step? { onboarding.screen(paired: server != nil) }

    func advanceOnboarding() {
        onboarding.advance(paired: server != nil)
        onboarding.save(to: defaults)
    }

    func finishOnboarding() {
        onboarding.finish()
        onboarding.save(to: defaults)
    }

    func setUpAgain() {
        onboarding.setUpAgain()
        onboarding.save(to: defaults)
    }

    func addTranslationTarget(_ code: String) {
        updateKeyboardSettings { try? $0.addTarget(code) }
    }

    func removeTranslationTarget(_ code: String) {
        updateKeyboardSettings { $0.removeTarget(code) }
    }

    func moveTranslationTargets(fromOffsets source: IndexSet, toOffset destination: Int) {
        updateKeyboardSettings { $0.moveTargets(fromOffsets: source, toOffset: destination) }
    }

    func setHapticFeedback(_ enabled: Bool) {
        updateKeyboardSettings { $0.hapticFeedback = enabled }
    }

    private func updateKeyboardSettings(_ change: (inout KeyboardSettings) -> Void) {
        change(&keyboardSettings)
        keyboardSettings.save(to: defaults)
    }
    func setMode(_ mode: CompositionOptions.Mode) { updatePreferences { $0.mode = mode } }
    func setStyle(_ style: CompositionOptions.Style) { updatePreferences { $0.style = style } }

    func toggleRecording(selectedText: String = "", contextBefore: String = "", contextAfter: String = "") async {
        if phase == .recording {
            do {
                try await controller.toggle(options: preferences.makeOptions(selectedText: selectedText, contextBefore: contextBefore, contextAfter: contextAfter))
            } catch {
                appDictationActive = false
                errorMessage = Self.describe(error, tailcat: self.server?.tailcat != nil)
            }
            return
        }
        let options = preferences.makeOptions(selectedText: selectedText, contextBefore: contextBefore, contextAfter: contextAfter)
        await startAppRecording(options: options)
    }

    private func startAppRecording(options: CompositionOptions) async {
        var options = options
        options.vocabulary = dictionary.vocabulary
        options.languageVariants = languageVariants
        options.styleExamples = PersonalStyleStore(defaults: defaults).examples(for: options.mode, style: options.style)
        do {
            try options.validate()
        } catch {
            errorMessage = Self.describe(error, tailcat: self.server?.tailcat != nil)
            return
        }
        guard await microphoneAllowed() else {
            errorMessage = MicrophoneCapture.CaptureError.microphoneDenied.errorDescription
            return
        }
        appDictationActive = true
        errorMessage = nil
        compositionStore.noteSubmittedMode(options.mode)
        do {
            try await controller.toggle(options: options)
        } catch {
            // The start threw before any state change was published, so nothing
            // will clear the flag later: a keyboard result must not inherit it.
            appDictationActive = false
            errorMessage = Self.describe(error, tailcat: self.server?.tailcat != nil)
        }
    }

    func editLatestResult() async {
        guard phase != .recording, phase != .processing,
              let latestText, !latestText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        var options = preferences.makeOptions(selectedText: latestText, contextBefore: "", contextAfter: "")
        options.mode = .edit
        options.targetLanguage = nil
        await startAppRecording(options: options)
    }

    func cancelComposition() {
        appDictationActive = false
        controller.cancel()
    }

    func retryComposition() async {
        appDictationActive = true
        await controller.retry()
    }

    func endSession() {
        floatingWindow?.stop()
        controller.endSession()
        sessionOpen = false
    }

    /// Sends a recording from History's unsent list again; its text lands in History and on Home.
    func sendAgain(_ id: UUID) async {
        await keepingAppAlive("voxt.resend") { await controller.reprocess(id) }
        unsentRecordings = recordings?.unprocessed ?? []
    }

    /// People switch back as soon as Voxt opens to resend; without a background task iOS
    /// suspends the app, which has no audio session then, before the Mac's text arrives.
    private func keepingAppAlive(_ name: String, _ work: () async -> Void) async {
        if let beginBackgroundTask {
            let end = beginBackgroundTask(name)
            await work()
            end()
            return
        }
        guard ownsSystemInterfaces else { await work(); return }
        let task = UIApplication.shared.beginBackgroundTask(withName: name)
        await work()
        if task != .invalid { UIApplication.shared.endBackgroundTask(task) }
    }

    func deleteUnsentRecording(_ id: UUID) {
        recordings?.remove(id)
        unsentRecordings = recordings?.unprocessed ?? []
    }

    func deleteHistory(at offsets: IndexSet) {
        historyStore.delete(ids: Set(offsets.compactMap { history.indices.contains($0) ? history[$0].id : nil }))
        history = historyStore.items
        scheduleHistorySync()
    }

    func setHistorySyncEnabled(_ enabled: Bool) {
        guard !enabled || server != nil else { return }
        historySyncGeneration = UUID()
        historySyncStore.setEnabled(enabled)
        historySyncEnabled = historySyncStore.isEnabled
        historySyncMessage = nil
        scheduleHistorySync()
    }

    func syncHistory() async {
        guard historySyncEnabled, !isSyncingHistory else { return }
        let generation = historySyncGeneration
        isSyncingHistory = true
        defer { isSyncingHistory = false }
        await historySyncCoordinator?.sync()
        guard generation == historySyncGeneration, historySyncEnabled else { return }
        history = historyStore.items
        historySyncMessage = historySyncCoordinator?.lastError
            ?? (historySyncCoordinator?.lastSyncedAt == nil ? nil : "Up to date")
    }

    private func scheduleHistorySync() {
        updateTimers()
        if ownsSystemInterfaces, historySyncEnabled { Task { await syncHistory() } }
    }

    private func resetHistorySyncPairing() {
        historySyncGeneration = UUID()
        historySyncStore.resetPairing()
        historySyncEnabled = false
        updateTimers()
        historySyncMessage = nil
    }

    private func microphoneAllowed() async -> Bool {
        guard usesMicrophone else { return true }
        return await MicrophoneCapture.requestPermission()
    }

    /// Cold launch marks the keyboard's start command handled inside `DictationController.init`,
    /// so the deep link has to read that command itself or the selected text is lost.
    func startFromKeyboard() async {
        guard let staged = channel.pendingCommand(after: 0), staged.kind == .start else {
            errorMessage = "Return to your text field and tap the Voxt microphone again."
            return
        }
        guard await microphoneAllowed() else {
            errorMessage = MicrophoneCapture.CaptureError.microphoneDenied.errorDescription
            return
        }
        let options = staged.options ?? CompositionOptions()
        let requestID = staged.requestID
        compositionStore.noteSubmittedMode(options.mode)
        do {
            try controller.start(options: options, requestID: requestID)
            showsKeyboardHandoff = KeyboardHandoffCover.shows(activeSince: activeSince, now: now())
        } catch {
            errorMessage = Self.describe(error, tailcat: self.server?.tailcat != nil)
        }
    }

    private func controllerChanged() {
        let previousPhase = phase
        phase = controller.phase
        if previousPhase == .recording, phase != .recording {
            audioLevel = 0
            levelTick = 0
            channel.publishAudioLevel(0)
        }
        if phase == .recording, previousPhase != .recording { recordingStartedAt = now() }
        if phase == .idle { recordingStartedAt = nil }
        sessionOpen = controller.isSessionOpen
        updateTimers()
        canRetry = controller.canRetry
        unsentRecordings = recordings?.unprocessed ?? []
        errorMessage = phase == .failed ? controller.lastError : nil
        if phase == .failed, let failure = controller.lastFailure as? VoxtClient.Error,
           failure == .transportUnavailable || failure == .unauthorized {
            connectionGeneration = UUID()
            connection = .failed(Self.describe(failure, tailcat: server?.tailcat != nil))
        }
        if phase == .done, case .failed = connection {
            Task { await checkConnection(attempts: 1) }
        }
        connection = ConnectionState.after(phase: phase, failure: controller.lastFailure, current: connection)
        switch phase {
        case .recording: floatingWindow?.show(.listening)
        case .processing: floatingWindow?.show(.processing)
        case .failed: floatingWindow?.show(.failed)
        case .idle, .done: floatingWindow?.show(.ready)
        }
        if phase == .done, previousPhase != .done, let text = controller.lastResult {
            latestText = text
            let elapsed = recordingStartedAt.map { max(0, now().timeIntervalSince($0)) }
            historyStore.append(id: controller.lastResultID ?? UUID(), text: text, mode: compositionStore.submittedMode(), elapsedSeconds: elapsed)
            recordingStartedAt = nil
            history = historyStore.items
            if appDictationActive, historyRetention != .off { appResultCount += 1 }
            scheduleHistorySync()
        }
        if phase != .recording, phase != .processing {
            showsKeyboardHandoff = false
            appDictationActive = false
        }
    }

    func save(_ config: ServerConfig) throws {
        guard !config.token.isEmpty else { throw ServerConfig.Error.missingToken }
        try pairingStore.save(config)
        connectionGeneration = UUID()
        if server != config { resetHistorySyncPairing() }
        floatingWindow?.stop()
        server = config
        learningProcessor?.invalidate()
        pairingConnection?.invalidate()
        let connection = connectionFactory(config)
        pairingConnection = connection
        controller.setTranscriber(connection.client(allowCloudModels: { [compositionStore] in compositionStore.loadPreferences().allowCloudModels },
                                                        customPrompt: { [developerPrompts] mode in developerPrompts.override(for: mode) }))
        self.connection = .checking
        Task { await checkConnection() }
    }

    func setAllowCloudModels(_ enabled: Bool) { updatePreferences { $0.allowCloudModels = enabled } }

    // MARK: Developer mode (Settings > Developer > Prompts)

    private(set) var developerModeEnabled: Bool
    private(set) var customPrompts: [CompositionOptions.Mode: String]
    private(set) var builtInPrompts: [CompositionOptions.Mode: String] = [:]
    private(set) var builtInPromptsError: String?
    /// Tests replace the Mac fetch.
    @ObservationIgnored var builtInPromptsLoader: (@MainActor () async throws -> [CompositionOptions.Mode: String])?

    /// Off keeps the saved prompts; they are sent again when Developer mode is turned back on.
    func setDeveloperMode(_ enabled: Bool) {
        developerPrompts.isEnabled = enabled
        developerModeEnabled = enabled
    }

    func loadBuiltInPrompts() async {
        do {
            if let builtInPromptsLoader {
                builtInPrompts = try await builtInPromptsLoader()
            } else if let pairingConnection {
                builtInPrompts = try await pairingConnection.client().builtInPrompts()
            } else {
                throw VoxtClient.Error.transportUnavailable
            }
            builtInPromptsError = nil
        } catch VoxtClient.Error.server(status: 404, message: _) {
            builtInPromptsError = "Update Voxt on your Mac to see its built-in prompts."
        } catch {
            builtInPromptsError = "Connect to your Mac to load its built-in prompts."
        }
    }

    /// What the editor shows: the custom prompt, else the Mac's built-in one.
    func promptText(for mode: CompositionOptions.Mode) -> String {
        customPrompts[mode] ?? builtInPrompts[mode] ?? ""
    }

    func hasCustomPrompt(for mode: CompositionOptions.Mode) -> Bool { customPrompts[mode] != nil }

    func setCustomPrompt(_ text: String, for mode: CompositionOptions.Mode) {
        developerPrompts.setPrompt(text, for: mode, builtIn: builtInPrompts[mode] ?? "")
        customPrompts[mode] = developerPrompts.prompt(for: mode)
    }

    func resetCustomPrompt(for mode: CompositionOptions.Mode) {
        developerPrompts.reset(mode)
        customPrompts[mode] = nil
    }

    func resetAllCustomPrompts() {
        developerPrompts.resetAll()
        customPrompts = [:]
    }

    func customPromptOverride(for mode: CompositionOptions.Mode) -> String? { developerPrompts.override(for: mode) }

    private func updatePreferences(_ body: (inout CompositionPreferences) -> Void) {
        var updated = preferences
        body(&updated)
        preferences = updated
        compositionStore.save(preferences)
    }

    func setMicrophoneReadiness(_ readiness: MicrophoneReadiness) {
        readiness.save(to: defaults)
        microphoneReadiness = readiness
        controller.idleTimeout = readiness.timeout
    }

    func setHistoryRetention(_ retention: TranscriptHistory.Retention) {
        historyStore.setRetention(retention)
        if retention == .off {
            historySyncGeneration = UUID()
            historySyncEnabled = false
            historySyncMessage = nil
        }
        historyRetention = retention
        history = historyStore.items
    }

    func clearHistory() {
        historyStore.clear()
        history = []
        latestText = nil
        scheduleHistorySync()
    }

    func refreshStoredState() {
        preferences = compositionStore.loadPreferences()
        keyboardSettings = KeyboardSettings.load(from: defaults)
        historyStore.prune()
        history = historyStore.items
        historySyncEnabled = historySyncStore.isEnabled
        scheduleHistorySync()
    }

    private func observeKeyboardCommands() {
        let observer = Unmanaged.passUnretained(self).toOpaque()
        CFNotificationCenterAddObserver(
            CFNotificationCenterGetDarwinNotifyCenter(),
            observer,
            { _, observer, _, _, _ in
                guard let observer else { return }
                let model = Unmanaged<AppModel>.fromOpaque(observer).takeUnretainedValue()
                Task { @MainActor in
                    await model.controller.handlePendingCommand()
                    await model.learningProcessor?.process()
                }
            },
            DictationChannel.commandNotification as CFString,
            nil,
            .deliverImmediately
        )
    }

    static func describe(_ error: Error, tailcat: Bool = false) -> String {
        if let urlError = error as? URLError {
            switch urlError.code {
            case .cannotFindHost, .cannotConnectToHost, .timedOut, .notConnectedToInternet, .networkConnectionLost:
                if tailcat {
                    return "Can't reach your Mac. Check your internet connection and make sure Voxt is running on the Mac."
                }
                return "Can't reach your Mac. Check that Tailscale is connected on this phone and Voxt is running on the Mac."
            default:
                return urlError.localizedDescription
            }
        }
        return (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}

/// Used only when a test injects dependencies; never touches the device Keychain.
private final class TransientTokenStore: TokenStoring {
    private var token: String?
    func read() throws -> String? { token }
    func write(_ value: String) throws { token = value }
    func delete() throws { token = nil }
}
