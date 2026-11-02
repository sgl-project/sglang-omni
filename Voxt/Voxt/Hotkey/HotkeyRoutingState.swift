import Foundation

// One state record per business; priority and trigger semantics are unchanged.
nonisolated enum RoutedHotkeyBusiness: CaseIterable, Hashable {
    case translation
    case rewrite
    case meeting
    case customPaste
    case note
    case transcription

    var priority: Int {
        switch self {
        case .translation:
            return 0
        case .rewrite:
            return 1
        case .meeting:
            return 2
        case .customPaste:
            return 3
        case .note:
            return 4
        case .transcription:
            return 5
        }
    }
}

nonisolated struct HotkeyBusinessState {
    var isDown = false
    var behavior: HotkeyPreference.TriggerBehavior?
    var keyCode: UInt16?
    var mouseButton: Int?
    var bindingID: UUID?
    var modifierTapCandidate = false
}
