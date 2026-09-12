import Foundation

/// Explicit preferences only. Audio, transcripts and prompt responses never enter this store.
@MainActor
final class ClientPreferences {
    static let maximumBackgroundLength = 2000
    let defaults: UserDefaults
    init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    var recordingShortcut: RecordingShortcut {
        get {
            guard let data = defaults.data(forKey: "recordingShortcut"),
                  let key = try? JSONDecoder().decode(RecordingShortcut.self, from: data),
                  key.validationMessage == nil else { return .default }
            return key
        }
        set { defaults.set(try? JSONEncoder().encode(newValue), forKey: "recordingShortcut") }
    }

    var polishEnabled: Bool {
        get { defaults.bool(forKey: "polishEnabled") }
        set { defaults.set(newValue, forKey: "polishEnabled") }
    }

    var personalBackgroundEnabled: Bool {
        get { defaults.bool(forKey: "personalBackgroundEnabled") }
        set { defaults.set(newValue, forKey: "personalBackgroundEnabled") }
    }

    var personalBackground: String {
        get { defaults.string(forKey: "personalBackground") ?? "" }
        set {
            if newValue.isEmpty { defaults.removeObject(forKey: "personalBackground") }
            else { defaults.set(newValue, forKey: "personalBackground") }
        }
    }
}
