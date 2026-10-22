import Testing
@testable import VoxtMobileKit

/// With Keyboard Feedback > Haptic on, every Voxt key vibrates like the system keyboard's.
@Suite struct KeyFeedbackTests {
    @Test func everyTypingKeyTicksAndClicks() {
        for kind: TextKeyboardKey.Kind in [.character, .space, .delete, .return, .shift, .numbers, .letters, .voice] {
            #expect(KeyFeedback.outputs(for: .key(kind), hasFullAccess: true) == [.haptic, .click], "\(kind)")
        }
    }

    @Test func micAndModeControlsTickWithoutAClick() {
        for control: KeyFeedback.Event in [.mic, .mode, .action] {
            #expect(KeyFeedback.outputs(for: control, hasFullAccess: true) == [.haptic], "\(control)")
        }
    }

    @Test func nothingWithoutFullAccess() {
        // Keyboard extensions can't play haptics or input clicks without Full Access.
        #expect(KeyFeedback.outputs(for: .key(.character), hasFullAccess: false).isEmpty)
        #expect(KeyFeedback.outputs(for: .mic, hasFullAccess: false).isEmpty)
    }

    @Test func hapticFeedbackOffKeepsOnlyTheClick() {
        // Settings › Keyboard › Haptic feedback. The click follows iOS's own setting.
        #expect(KeyFeedback.outputs(for: .key(.character), hasFullAccess: true, hapticFeedback: false) == [.click])
        #expect(KeyFeedback.outputs(for: .mic, hasFullAccess: true, hapticFeedback: false).isEmpty)
    }
}
