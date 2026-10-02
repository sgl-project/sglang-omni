import Combine
import Foundation

@MainActor
public protocol AudioRecording: AnyObject {
    func start() async throws
    func stop() throws -> Data
    func cancel()
    var level: Double { get }
    var recordingError: String? { get }
}

public extension AudioRecording {
    var level: Double { 0 }
    var recordingError: String? { nil }
}

@MainActor
public protocol AudioTranscribing: AnyObject {
    func transcribe(wav: Data) async throws -> String
}

@MainActor
public protocol TextPolishing: AnyObject {
    func polish(text: String, personalBackground: String) async throws -> String
}

@MainActor
public protocol SpeechServing: AudioTranscribing, TextPolishing {
    func polish(text: String) async throws -> String
    func snapshot() -> SpeechServing
}

public extension SpeechServing {
    func snapshot() -> SpeechServing { self }
    // Preserve the original interface for services without personalization support.
    func polish(text: String, personalBackground: String) async throws -> String {
        guard personalBackground.isEmpty else { throw DictationError("当前整理服务不支持个人背景。") }
        return try await polish(text: text)
    }
}

public enum DictationPhase: String {
    case idle = "待录音"
    case authorizing = "准备麦克风"
    case recording = "录音中"
    case recognizing = "正在转写"
    case polishing = "正在整理"
    case ready = "已完成"
    case failed = "未完成"
}

public struct DictationError: LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

@MainActor
public final class DictationSession: ObservableObject {
    public nonisolated static let maximumAudioSeconds: Double = 60
    @Published public private(set) var phase: DictationPhase = .idle
    @Published public private(set) var rawText = ""
    @Published public private(set) var resultText = ""
    @Published public private(set) var hasPolishedResult = false
    @Published public private(set) var notice = ""
    @Published public private(set) var recordingStartedAt: Date?
    @Published public private(set) var elapsed: Double = 0
    @Published public private(set) var level: Double = 0
    /// Client-observed processing durations, excluding recording, warmup and insertion.
    @Published public private(set) var asrSeconds: Double?
    @Published public private(set) var polishSeconds: Double?
    @Published public private(set) var totalSeconds: Double?
    @Published public private(set) var polishWasRequested = false
    @Published public var polishEnabled = false
    public var personalBackground = ""
    /// Delivered once, only for the current successful run, after optional polishing.
    public var onResult: ((String) -> Void)?

    private let recorder: AudioRecording
    private let service: SpeechServing
    private var runService: SpeechServing
    public let recordingLimit: Double
    private let processingTime: () -> Double
    private var runID = UUID()
    private var task: Task<Void, Never>?
    private var timer: Task<Void, Never>?
    private var runBackground = ""

    public init(recorder: AudioRecording, service: SpeechServing,
                recordingLimit: Double = DictationSession.maximumAudioSeconds,
                processingTime: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }) {
        self.recorder = recorder
        self.service = service
        runService = service
        self.recordingLimit = recordingLimit
        self.processingTime = processingTime
    }

    public var isBusy: Bool {
        [.authorizing, .recording, .recognizing, .polishing].contains(phase)
    }

    public func toggleRecording() {
        if phase == .recording { finishRecording(); return }
        guard !isBusy else { return }
        let id = begin()
        phase = .authorizing
        task = Task { [weak self] in
            guard let self, self.isCurrent(id) else { return }
            do {
                try await recorder.start()
                guard isCurrent(id) else { return }
                recordingStartedAt = Date()
                phase = .recording
                let started = ProcessInfo.processInfo.systemUptime
                timer = Task { [weak self] in
                    while !Task.isCancelled {
                        do { try await Task.sleep(nanoseconds: 50_000_000) } catch { return }
                        guard let self, self.isCurrent(id), self.phase == .recording else { return }
                        self.elapsed = min(ProcessInfo.processInfo.systemUptime - started, self.recordingLimit)
                        self.level = self.recorder.level
                        if let error = self.recorder.recordingError {
                            self.recorder.cancel()
                            self.fail(error)
                            return
                        }
                        if self.elapsed >= self.recordingLimit {
                            self.finishRecording()
                            return
                        }
                    }
                }
            } catch {
                guard isCurrent(id) else { return }
                recorder.cancel()
                fail(error.localizedDescription)
            }
        }
    }

    public func submitAudio(_ wav: Data) {
        guard !isBusy else { return }
        let id = begin()
        guard !wav.isEmpty else { fail("没有录到音频，请重试。"); return }
        process(wav, id: id)
    }

    public func reportInputFailure(_ message: String) {
        guard !isBusy else { return }
        _ = begin()
        fail(message)
    }

    public func cancel() {
        runID = UUID()
        task?.cancel()
        timer?.cancel()
        recorder.cancel()
        clear()
        phase = .idle
    }

    private func begin() -> UUID {
        task?.cancel()
        timer?.cancel()
        runID = UUID()
        clear()
        polishWasRequested = polishEnabled
        runBackground = polishEnabled ? personalBackground : ""
        runService = service.snapshot()
        return runID
    }

    private func clear() {
        rawText = ""
        resultText = ""
        hasPolishedResult = false
        notice = ""
        elapsed = 0
        level = 0
        recordingStartedAt = nil
        asrSeconds = nil
        polishSeconds = nil
        totalSeconds = nil
        polishWasRequested = false
    }

    private func isCurrent(_ id: UUID) -> Bool { id == runID && !Task.isCancelled }

    private func finishRecording() {
        timer?.cancel()
        recordingStartedAt = nil
        level = 0
        do {
            let wav = try recorder.stop()
            guard !wav.isEmpty else { throw DictationError("没有录到音频，请检查麦克风。") }
            process(wav, id: runID)
        } catch {
            recorder.cancel()
            fail(error.localizedDescription)
        }
    }

    private func process(_ wav: Data, id: UUID) {
        phase = .recognizing
        let service = runService
        task = Task { [weak self] in
            guard let self, isCurrent(id) else { return }
            let started = processingTime()
            do {
                let raw = try await service.transcribe(wav: wav).trimmingCharacters(in: .whitespacesAndNewlines)
                guard isCurrent(id) else { return }
                asrSeconds = processingTime() - started
                rawText = raw
                resultText = raw
                if raw.isEmpty {
                    notice = "没有识别到文字，请检查音量后重新录音。"
                } else if polishWasRequested {
                    phase = .polishing
                    let polishStart = processingTime()
                    do {
                        let result = try await service.polish(text: raw, personalBackground: runBackground)
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        guard isCurrent(id) else { return }
                        guard !result.isEmpty else { throw DictationError("整理返回了空文本") }
                        resultText = result
                        hasPolishedResult = true
                    } catch {
                        guard isCurrent(id) else { return }
                        notice = "整理未完成，已保留原文。\(error.localizedDescription)"
                    }
                    polishSeconds = processingTime() - polishStart
                }
                totalSeconds = processingTime() - started
                phase = .ready
                guard isCurrent(id) else { return }
                onResult?(resultText)
            } catch {
                guard isCurrent(id) else { return }
                asrSeconds = processingTime() - started
                totalSeconds = asrSeconds
                fail("转写未完成：\(error.localizedDescription)")
            }
        }
    }

    private func fail(_ message: String) {
        timer?.cancel()
        recordingStartedAt = nil
        level = 0
        notice = message
        phase = .failed
    }
}
