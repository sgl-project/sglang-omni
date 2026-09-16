import Combine
import Foundation

@MainActor
public protocol TextRevisionTarget: AnyObject {
    var confirmationFailure: String? { get }
    func confirmInsertion() async -> Bool
    func prepare(original: String) throws
    func apply(_ revision: TextRevision) async throws -> Bool
    func cancel()
}

public extension TextRevisionTarget {
    var confirmationFailure: String? { nil }
}

public enum CorrectionPhase: String {
    case idle = "待更正", authorizing = "准备更正录音", recording = "请说修改意见"
    case recognizing = "转写修改意见", correcting = "正在更正", applying = "正在替换"
    case ready = "更正结果", failed = "更正未完成"

    public var isBusy: Bool { ![.idle, .ready, .failed].contains(self) }
}

/// Retains one committed dictation and isolates correction recording from ordinary dictation.
@MainActor
public final class CorrectionSession: ObservableObject {
    @Published public private(set) var phase: CorrectionPhase = .idle
    @Published public private(set) var lastText: String?
    @Published public private(set) var correctedText: String?
    @Published public private(set) var instruction = ""
    @Published public private(set) var notice = ""
    @Published public private(set) var canReplace = false
    @Published public private(set) var isTrackingInsertion = false
    @Published public private(set) var didReplace = false
    public let recording: DictationSession
    public var isBusy: Bool { phase.isBusy }
    public var hasResult: Bool { correctedText != nil || !notice.isEmpty }
    private let makeCorrector: () -> TextCorrecting
    private var corrector: TextCorrecting?
    private var target: TextRevisionTarget?
    private var runTarget: TextRevisionTarget?
    private var original = ""
    private var background = ""
    private var targetUnavailableReason = ""
    private var generation = UUID()
    private var task: Task<Void, Never>?
    private var insertionTask: Task<Void, Never>?
    private var phaseObserver: AnyCancellable?

    public init(recorder: AudioRecording, service: SpeechServing,
                makeCorrector: @escaping () -> TextCorrecting) {
        recording = DictationSession(recorder: recorder, service: service)
        self.makeCorrector = makeCorrector
        phaseObserver = recording.$phase.removeDuplicates().sink { [weak self] phase in
            guard let self, self.isBusy else { return }
            switch phase {
            case .authorizing: self.phase = .authorizing
            case .recording: self.phase = .recording
            case .recognizing: self.phase = .recognizing
            case .failed:
                self.notice = self.recording.notice
                self.phase = .failed
                self.runTarget?.cancel()
                self.runTarget = nil
            default: break
            }
        }
        recording.onResult = { [weak self] in self?.correct($0) }
    }

    public func remember(_ text: String) {
        clear()
        if !text.isEmpty { lastText = text }
    }

    public func trackInsertion(_ target: TextRevisionTarget?) {
        guard let target, lastText != nil else { target?.cancel(); return }
        insertionTask?.cancel()
        self.target?.cancel()
        self.target = target
        canReplace = false
        isTrackingInsertion = true
        let id = generation
        insertionTask = Task { [weak self] in
            let confirmed = await target.confirmInsertion()
            guard let self, id == self.generation, !Task.isCancelled else { target.cancel(); return }
            self.isTrackingInsertion = false
            self.canReplace = confirmed
            if !confirmed {
                target.cancel()
                self.target = nil
                self.targetUnavailableReason = target.confirmationFailure ?? "未能在原输入位置确认上一段文字。"
            }
            self.insertionTask = nil
        }
    }

    public func noteUnavailableTarget(_ reason: String) {
        targetUnavailableReason = reason
    }

    public func toggleRecording(personalBackground: String) {
        if phase == .recording { recording.toggleRecording(); return }
        guard !isBusy else { return }
        guard !isTrackingInsertion else {
            notice = "正在确认上一段的位置，请稍后再更正。"
            phase = .failed
            return
        }
        guard let lastText, !lastText.isEmpty else {
            notice = "没有可更正的上一段，请先完成一次听写。"
            phase = .failed
            return
        }
        generation = UUID()
        correctedText = nil
        instruction = ""
        notice = ""
        didReplace = false
        original = lastText
        background = personalBackground
        corrector = makeCorrector()
        runTarget = nil
        if canReplace, let target {
            do { try target.prepare(original: lastText); runTarget = target }
            catch {
                targetUnavailableReason = error.localizedDescription
                target.cancel()
                self.target = nil
                canReplace = false
            }
        }
        if runTarget == nil { notice = "无法确认上一段输入位置；本次更正结果将供你复制。" }
        phase = .authorizing
        recording.polishEnabled = false
        recording.toggleRecording()
    }

    public func cancel() {
        generation = UUID()
        task?.cancel()
        task = nil
        insertionTask?.cancel()
        insertionTask = nil
        if isTrackingInsertion || phase == .applying {
            target?.cancel()
            target = nil
            canReplace = false
        }
        isTrackingInsertion = false
        runTarget?.cancel()
        runTarget = nil
        corrector = nil
        phase = .idle
        recording.cancel()
        correctedText = nil
        instruction = ""
        notice = ""
        didReplace = false
    }

    public func clear() {
        cancel()
        target?.cancel()
        target = nil
        canReplace = false
        lastText = nil
        original = ""
        background = ""
        targetUnavailableReason = ""
    }

    private func correct(_ instruction: String) {
        guard isBusy, let corrector else { return }
        self.instruction = instruction
        guard !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            notice = "没有听清修改意见，上一段保持不变。"
            phase = .failed
            runTarget?.cancel()
            runTarget = nil
            return
        }
        let id = generation, original = original, background = background, target = runTarget
        phase = .correcting
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await corrector.correct(original: original, instruction: instruction,
                                                         personalBackground: background)
                guard id == self.generation, !Task.isCancelled else { return }
                self.correctedText = result.correctedText
                if result.status == .noChange {
                    self.notice = "原文保持不变，未修改输入框。如需更正，请明确要改的词和正确写法。"
                } else if TextRevision(original: original, corrected: result.correctedText).isEmpty {
                    self.notice = "更正结果与原文相同，未修改输入框；请更明确地说明要改哪个词。"
                } else if let target {
                    self.phase = .applying
                    do {
                        let confirmed = try await target.apply(TextRevision(original: original, corrected: result.correctedText))
                        guard id == self.generation, !Task.isCancelled else { return }
                        if confirmed {
                            self.lastText = result.correctedText
                            self.didReplace = true
                            self.notice = "已更正上一段，未发送。"
                            if result.correctedText.isEmpty { self.canReplace = false; self.target = nil }
                        } else {
                            self.canReplace = false
                            self.target = nil
                            self.notice = "替换结果未确认，请检查原输入框；不会重复替换。"
                        }
                    } catch {
                        guard id == self.generation, !Task.isCancelled else { return }
                        self.canReplace = false
                        self.target = nil
                        self.notice = "自动更正未确认：\(error.localizedDescription) 请检查原输入框；更正结果可复制。"
                    }
                } else {
                    let reason = self.targetUnavailableReason.isEmpty ? "无法定位上一段的输入位置。" : self.targetUnavailableReason
                    self.notice = "未自动替换：\(reason) 更正文本已生成，请复制到原输入框。"
                }
                target?.cancel()
                self.runTarget = nil
                self.corrector = nil
                self.phase = .ready
            } catch {
                guard id == self.generation, !Task.isCancelled else { return }
                target?.cancel()
                self.runTarget = nil
                self.corrector = nil
                self.notice = "更正未完成，上一段保持不变。\(error.localizedDescription)"
                self.phase = .failed
            }
            if id == self.generation { self.task = nil }
        }
    }
}
