import Foundation

/// Developer mode's per-mode prompts, kept on the phone. While Developer mode is off the
/// prompts are kept but not sent. A prompt equal to the built-in one, or empty, is no prompt.
public final class DeveloperPromptStore: @unchecked Sendable {
    private enum Key {
        static let enabled = "developerMode.enabled"
        static func prompt(_ mode: CompositionOptions.Mode) -> String { "developerMode.prompt.\(mode.rawValue)" }
    }

    private let defaults: UserDefaults

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public var isEnabled: Bool {
        get { defaults.bool(forKey: Key.enabled) }
        set { defaults.set(newValue, forKey: Key.enabled) }
    }

    /// The saved custom prompt, whether or not Developer mode is on.
    public func prompt(for mode: CompositionOptions.Mode) -> String? {
        defaults.string(forKey: Key.prompt(mode))
    }

    public func setPrompt(_ text: String, for mode: CompositionOptions.Mode, builtIn: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == builtIn.trimmingCharacters(in: .whitespacesAndNewlines) {
            defaults.removeObject(forKey: Key.prompt(mode))
        } else {
            defaults.set(trimmed, forKey: Key.prompt(mode))
        }
    }

    public func reset(_ mode: CompositionOptions.Mode) {
        defaults.removeObject(forKey: Key.prompt(mode))
    }

    public func resetAll() {
        CompositionOptions.Mode.allCases.forEach(reset)
    }

    /// The prompt to send with a request in this mode: only while Developer mode is on.
    public func override(for mode: CompositionOptions.Mode) -> String? {
        isEnabled ? prompt(for: mode) : nil
    }
}
