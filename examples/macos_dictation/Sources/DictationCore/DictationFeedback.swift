import Combine
import Foundation

/// Presentation only: hiding feedback never clears the transcript or cancels delivery.
@MainActor
public final class DictationFeedback: ObservableObject {
    @Published public private(set) var opacity: Double = 0
    private enum Mode: Equatable {
        case hidden
        case active(DictationPhase)
        case completed
        case attention
    }
    private var observer: AnyCancellable?
    private var task: Task<Void, Never>?
    private var generation = UUID()
    private var mode: Mode = .hidden
    private var isHovered = false
    private let holdDuration: Double
    private let attentionHoldDuration: Double
    private let fadeDuration: Double
    private let reduceMotion: () -> Bool

    public init(session: DictationSession, insertion: DictationInsertion,
                holdDuration: Double = 1, fadeDuration: Double = 0.3,
                attentionHoldDuration: Double = 3,
                reduceMotion: @escaping () -> Bool = { false }) {
        self.holdDuration = holdDuration
        self.attentionHoldDuration = attentionHoldDuration
        self.fadeDuration = fadeDuration
        self.reduceMotion = reduceMotion
        // Consume published values directly: @Published emits before the property is stored.
        observer = session.$phase
            .combineLatest(insertion.$didInsert.combineLatest(insertion.$isDelivering, insertion.$hasWarning,
                                                             insertion.$canDismissAfterPaste))
            .map { phase, insertion -> Mode in
                let (didInsert, isDelivering, hasWarning, canDismissAfterPaste) = insertion
                if phase == .idle { return .hidden }
                if phase == .failed { return .attention }
                let completed = (didInsert && !hasWarning) || canDismissAfterPaste
                // A polish fallback can still deliver the original successfully.
                // Keep its notice in the results, without pinning completed feedback.
                if phase == .ready && !isDelivering {
                    return completed ? .completed : .attention
                }
                return .active(phase)
            }
            .removeDuplicates()
            .sink { [weak self] in self?.update($0) }
    }

    public func dismiss() {
        mode = .hidden
        isHovered = false
        cancelFade()
        opacity = 0
    }

    public func setHovered(_ hovered: Bool) {
        // A queued mouse event must not revive an already hidden or dismissed panel.
        guard mode != .hidden, isHovered != hovered else { return }
        isHovered = hovered
        guard mode == .completed || mode == .attention else { return }
        if hovered {
            cancelFade()
            opacity = 1
        } else {
            fade(after: 0)
        }
    }

    private func cancelFade() {
        generation = UUID()
        task?.cancel()
        task = nil
    }

    private func update(_ mode: Mode) {
        self.mode = mode
        cancelFade()
        if mode == .hidden { isHovered = false }
        opacity = mode == .hidden ? 0 : 1
        guard !isHovered else { return }
        if mode == .completed { fade(after: holdDuration) }
        else if mode == .attention { fade(after: attentionHoldDuration) }
    }

    private func fade(after hold: Double) {
        cancelFade()
        let id = generation
        let fade = fadeDuration
        task = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: UInt64(max(0, hold) * 1_000_000_000))
                guard let self, id == self.generation, !Task.isCancelled else { return }
                if !self.reduceMotion(), fade > 0 {
                    let started = ProcessInfo.processInfo.systemUptime
                    while true {
                        try await Task.sleep(nanoseconds: 16_000_000)
                        guard id == self.generation, !Task.isCancelled else { return }
                        let progress = min(1, (ProcessInfo.processInfo.systemUptime - started) / fade)
                        // Ease out using one cancellable task, with no detached window animation.
                        if progress >= 1 { self.dismiss(); return }
                        self.opacity = (1 - progress) * (1 - progress)
                    }
                } else { self.dismiss() }
            } catch { /* Cancellation belongs to a newer presentation or explicit dismissal. */ }
        }
    }

    deinit { task?.cancel() }
}
