import Foundation
import SwiftUI
import VoxtKit

/// The app's state the screens share: the settings (saved as they
/// change), the services the settings offer, the history and the voice
/// session.
@MainActor
public final class AppModel: ObservableObject {
    @Published public var settings: AppSettings {
        didSet {
            guard settings != oldValue else { return }
            scheduleSave()
        }
    }
    @Published public private(set) var presets: [Preset] = []
    @Published public private(set) var languages: [Language] = []
    @Published public private(set) var history: [HistoryEntry] = []

    public let session: VoiceSession
    let core: CoreService
    let keys: KeyStore
    private let settingsStore: SettingsStore
    private let historyStore: HistoryStore
    private var pendingSave: Task<Void, Never>?
    private let saveDelay: Duration

    public init(core: CoreService, keys: KeyStore, recorder: AudioRecorder, directory: URL,
                saveDelay: Duration = .milliseconds(500), microphoneAllowed: @escaping @Sendable () async -> Bool) {
        self.saveDelay = saveDelay
        self.core = core
        self.keys = keys
        settingsStore = SettingsStore(directory: directory)
        historyStore = HistoryStore(directory: directory)
        let loaded = settingsStore.load()
        settings = loaded
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("recordings")
        let env = SessionEnvironment(core: core, recorder: recorder, keys: keys, history: historyStore,
                                     speech: SessionEnvironment.speechEngine(core: core, scratch: scratch),
                                     microphoneAllowed: microphoneAllowed)
        session = VoiceSession(environment: env, feature: loaded.lastFeature, settings: { loaded })
        history = historyStore.entries()
        session.settings = { [weak self] in self?.settings ?? loaded }
    }

    /// The app on this device: the Go core, the Keychain, the microphone
    /// and Application Support.
    public static func live() -> AppModel {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Voxt", isDirectory: true)
        return AppModel(core: VoxtCore(), keys: KeychainStore(), recorder: MicRecorder(), directory: support,
                        microphoneAllowed: { await MicRecorder.requestPermission() })
    }

    /// Writes the settings now (the app leaving the screen) instead of
    /// when typing pauses.
    public func saveNow() {
        pendingSave?.cancel()
        pendingSave = nil
        _ = try? settingsStore.save(settings) // kept in memory when the disk refuses; saved again on the next change
    }

    /// The text fields change the settings on every keystroke: the document
    /// is written once typing pauses (saveDelay), and at once when the app
    /// leaves the screen (VoxtApp). A change in the last moment before the
    /// process is killed outright can be lost.
    private func scheduleSave() {
        pendingSave?.cancel()
        let delay = saveDelay
        pendingSave = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.saveNow()
        }
    }

    /// Waits for a scheduled save (tests).
    func waitForSave() async {
        await pendingSave?.value
    }

    /// Loads what the core offers (presets, languages) once.
    func loadCatalogues() async {
        guard presets.isEmpty || languages.isEmpty else { return }
        async let presetList = try? core.presets()
        async let languageList = try? core.languages()
        if let list = await presetList {
            presets = list
        }
        if let list = await languageList {
            languages = list.sorted { $0.displayName(chinese: showsChinese) < $1.displayName(chinese: showsChinese) }
        }
    }

    func reloadHistory() {
        history = historyStore.entries()
    }

    func deleteHistory(_ ids: Set<UUID>) {
        history = (try? historyStore.delete(ids)) ?? historyStore.entries()
    }

    func clearHistory() {
        history = (try? historyStore.clear()) ?? historyStore.entries()
    }

    var preset: Preset? { presets.first { $0.id == settings.languageModel.preset } }

    func languageName(_ code: String) -> String {
        languages.first { $0.code == code }?.displayName(chinese: showsChinese) ?? code
    }

    /// Saves an API key in the Keychain ("" removes it): the Keychain's
    /// status when it refuses the key, nil when it is kept.
    func saveKey(_ key: String, for account: String) -> OSStatus? {
        do {
            try keys.setKey(key, for: account)
            return nil
        } catch {
            return (error as? KeychainError)?.status ?? errSecParam
        }
    }
}
