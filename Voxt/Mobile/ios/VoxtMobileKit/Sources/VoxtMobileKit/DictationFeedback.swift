import SwiftUI

public enum DictationFeedback: Equatable {
    case started, stopped, success, failure

    public static func event(from old: DictationChannel.Phase, to new: DictationChannel.Phase,
                             includeSuccess: Bool = true) -> Self? {
        guard old != new else { return nil }
        if new == .recording { return .started }
        if new == .failed { return .failure }
        if new == .done { return includeSuccess ? .success : nil }
        if old == .recording { return .stopped }
        return nil
    }

    var sensory: SensoryFeedback {
        switch self {
        case .started: .impact(weight: .medium)
        case .stopped: .impact(weight: .light)
        case .success: .success
        case .failure: .error
        }
    }
}

extension View {
    public func dictationFeedback(phase: DictationChannel.Phase, includeSuccess: Bool = true) -> some View {
        sensoryFeedback(trigger: phase) { old, new in
            DictationFeedback.event(from: old, to: new, includeSuccess: includeSuccess)?.sensory
        }
    }
}
