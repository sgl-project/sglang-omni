import Foundation

/// Which feedback a keyboard press gives. The keyboard plays it on touch-down: a light
/// haptic, and the system input click for typing keys (iOS applies the user's Keyboard
/// Clicks setting to that click). A keyboard extension can do neither without Full Access.
public enum KeyFeedback {
    public enum Event: Equatable, Sendable {
        case key(TextKeyboardKey.Kind)
        /// The dictation mic button.
        case mic
        /// Writing mode and layout choices.
        case mode
        /// Edit, Undo, Retry, Cancel, Copy and similar buttons.
        case action
    }

    public enum Output: Hashable, Sendable {
        case haptic
        case click
    }

    public static func outputs(for event: Event, hasFullAccess: Bool, hapticFeedback: Bool = true) -> Set<Output> {
        guard hasFullAccess else { return [] }
        let outputs: Set<Output>
        switch event {
        case .key: outputs = [.haptic, .click]
        case .mic, .mode, .action: outputs = [.haptic]
        }
        // Settings › Keyboard › Haptic feedback. The click follows iOS's own Keyboard Clicks.
        return hapticFeedback ? outputs : outputs.subtracting([.haptic])
    }
}
