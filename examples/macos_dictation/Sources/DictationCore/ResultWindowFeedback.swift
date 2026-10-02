import Combine
import Foundation

/// A completed round dismisses existing results; explicitly reopening them keeps them readable.
@MainActor
public final class ResultWindowFeedback: ObservableObject {
    @Published public private(set) var opacity: Double = 1
    private let session: DictationSession
    private let insertion: DictationInsertion
    private let feedback: DictationFeedback
    private let correction: CorrectionSession?
    private var correctionFeedback: DictationFeedback?
    private var correctionPhaseObserver: AnyCancellable?
    private var correctionOpacityObserver: AnyCancellable?
    private var automaticallyDismisses = false
    private var phaseObserver: AnyCancellable?
    private var opacityObserver: AnyCancellable?

    public init(session: DictationSession, insertion: DictationInsertion, correction: CorrectionSession? = nil,
                holdDuration: Double = 3, fadeDuration: Double = 0.3,
                attentionHoldDuration: Double = 5,
                reduceMotion: @escaping () -> Bool = { false }) {
        self.session = session
        self.insertion = insertion
        self.correction = correction
        // Separate timing from the floating bar: hovering that bar must not pin this window.
        feedback = DictationFeedback(session: session, insertion: insertion,
                                     holdDuration: holdDuration, fadeDuration: fadeDuration,
                                     attentionHoldDuration: attentionHoldDuration, reduceMotion: reduceMotion)
        phaseObserver = session.$phase.removeDuplicates().sink { [weak self] phase in
            guard let self else { return }
            // Use the emitted phase because @Published sends before storing the new value.
            if [.authorizing, .recording, .recognizing, .polishing, .failed].contains(phase) {
                self.automaticallyDismisses = true
                self.opacity = 1
            }
            self.refreshOpacity()
        }
        opacityObserver = feedback.$opacity.removeDuplicates().sink { [weak self] _ in
            self?.refreshOpacity()
        }
        if let correction {
            correctionPhaseObserver = correction.$phase.removeDuplicates().sink { [weak self] phase in
                guard let self else { return }
                if phase.isBusy || phase == .failed {
                    self.automaticallyDismisses = true
                    self.opacity = 1
                }
                self.refreshOpacity()
            }
            let feedback = DictationFeedback(correction: correction, holdDuration: holdDuration,
                                              fadeDuration: fadeDuration, attentionHoldDuration: attentionHoldDuration,
                                              reduceMotion: reduceMotion)
            correctionFeedback = feedback
            correctionOpacityObserver = feedback.$opacity.removeDuplicates().sink { [weak self] _ in
                self?.refreshOpacity()
            }
        }
    }

    /// Called by an explicit "show results" action, never by a completion notification.
    public func show() {
        automaticallyDismisses = session.isBusy || insertion.isDelivering || correction?.isBusy == true
        opacity = 1
    }

    private func refreshOpacity() {
        // Combine can deliver nested @Published willSet notifications in either
        // subscriber order. Read the settled state on the next main-actor turn,
        // never a timer value captured while another round was being cleared.
        Task { [weak self] in
            guard let self, self.automaticallyDismisses else { return }
            if self.session.isBusy || self.insertion.isDelivering || self.correction?.isBusy == true {
                self.opacity = 1
            } else if let correction = self.correction, correction.phase != .idle {
                self.opacity = self.correctionFeedback?.opacity ?? 0
            } else {
                self.opacity = self.feedback.opacity
            }
        }
    }
}
