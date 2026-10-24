import Foundation

/// Microphone access as the controller needs it. The real implementation keeps an
/// AVAudioEngine input tap running for the whole session so dictation can start while
/// the app is in the background; `beginCapture`/`endCapture` only mark which samples to keep.
@MainActor
public protocol AudioCapturing: AnyObject {
    var level: Float { get }
    func startSession() throws
    func endSession()
    func beginCapture()
    /// Returns the captured audio as 16 kHz mono samples.
    func endCapture() -> [Float]
    /// The samples captured since `beginCapture` from `offset` on, without ending the capture;
    /// the controller saves them to disk every few seconds.
    func capturedSamples(from offset: Int) -> [Float]
}

extension AudioCapturing {
    public var level: Float { 0 }
    public func capturedSamples(from offset: Int) -> [Float] { [] }
}

@MainActor
public protocol Transcribing {
    func transcribe(wav: Data) async throws -> (text: String, rawText: String)
    func transcribe(wav: Data, options: CompositionOptions) async throws -> (text: String, rawText: String)
    func transcribeResult(wav: Data, options: CompositionOptions) async throws -> TranscriptionResult
}

public struct TranscriptionResult: Sendable {
    public var id: UUID?
    public var text: String
    public var rawText: String
    public var asrMs: Int?
    public var enhanceMs: Int?
    /// Phone-measured: health/capabilities pre-checks, and the compose request itself.
    public var preflightMs: Int?
    public var requestMs: Int?
    public init(id: UUID? = nil, text: String, rawText: String, asrMs: Int? = nil, enhanceMs: Int? = nil,
                preflightMs: Int? = nil, requestMs: Int? = nil) {
        self.id = id; self.text = text; self.rawText = rawText
        self.asrMs = asrMs; self.enhanceMs = enhanceMs
        self.preflightMs = preflightMs; self.requestMs = requestMs
    }
}

extension Transcribing {
    public func transcribeResult(wav: Data, options: CompositionOptions) async throws -> TranscriptionResult {
        let result = try await transcribe(wav: wav, options: options)
        return TranscriptionResult(text: result.text, rawText: result.rawText)
    }
    public func transcribe(wav: Data, options: CompositionOptions) async throws -> (text: String, rawText: String) {
        guard options.isPlainDictation else {
            throw CompositionOptions.ValidationError(message: "This connection does not support writing actions. Update the Mac app.")
        }
        return try await transcribe(wav: wav)
    }
}

extension VoxtClient: Transcribing {}

extension VoxtClient {
    public func transcribeResult(wav: Data, options: CompositionOptions) async throws -> TranscriptionResult {
        let result = try await compose(audio: wav, contentType: "audio/wav", options: options)
        guard let id = UUID(uuidString: result.id) else { throw Error.invalidResponse }
        return TranscriptionResult(id: id, text: result.text, rawText: result.rawText, asrMs: result.asrMs, enhanceMs: result.enhanceMs,
                                   preflightMs: result.preflightMs, requestMs: result.requestMs)
    }
    public func transcribe(wav: Data) async throws -> (text: String, rawText: String) {
        let result = try await dictate(audio: wav, contentType: "audio/wav")
        return (result.text, result.rawText)
    }

    public func transcribe(wav: Data, options: CompositionOptions) async throws -> (text: String, rawText: String) {
        let result = try await compose(audio: wav, contentType: "audio/wav", options: options)
        return (result.text, result.rawText)
    }
}

/// Runs dictations in the app on behalf of the keyboard (through `DictationChannel`)
/// and of the app's own record button.
@MainActor
public final class DictationController {
    /// The default time the audio session stays open without a dictation.
    public nonisolated static let idleSessionTimeout: TimeInterval = 5 * 60
    /// The user's "Disable after" choice; nil keeps the session open.
    public var idleTimeout: TimeInterval? = idleSessionTimeout
    /// Shorter recordings are treated as accidental taps.
    public static let minimumDuration: TimeInterval = 0.3
    /// A cancelled recording at least this long stays in History; shorter ones are deleted.
    public static let keepCancelledAfter: TimeInterval = 10
    /// A recording cut off this recently is offered to the keyboard's Retry after a relaunch;
    /// older ones wait in History.
    public static let keyboardRetryWindow: TimeInterval = 10 * 60
    public static let savedRecordingMessage = "Your last recording was saved. Tap Retry to send it."
    /// Seconds between saves of the audio recorded so far.
    public var autosaveInterval: TimeInterval = 2

    public private(set) var phase: DictationChannel.Phase = .idle
    public private(set) var lastResult: String?
    public private(set) var lastResultID: UUID?
    public private(set) var lastError: String?
    /// True while a PiP session keeps the host reachable with the microphone off.
    /// The keyboard sees a fresh heartbeat (filled mic) even though no audio session runs.
    public private(set) var isArmed = false
    public var onChange: (() -> Void)?
    public var now: () -> Date = Date.init

    private let channel: DictationChannel
    private let capture: AudioCapturing
    private var transcriber: Transcribing?
    private let recordings: RecordingStore?
    /// The recording on disk for the current dictation, and how many of its samples are saved.
    private var recordingID: UUID?
    private var savedSampleCount = 0
    private var lastAutosave = Date()
    private var sessionOpen = false
    private var lastActivity = Date()
    private var handledCommandSequence = 0
    private var currentOptions = CompositionOptions()
    private var activeRequestID: String?
    private var generation = UUID()
    private var uploadTask: Task<TranscriptionResult, Error>?
    private struct PendingUpload {
        var wav: Data
        var recordingID: UUID?
        var options: CompositionOptions
        var requestID: String?
        var stopAt = Date()
        var audioMs = 0
        var encodeMs = 0
        var level = SpeechGate.Measurement(peakDb: 0, speechMs: 0)
    }
    private var pendingUpload: PendingUpload?
    /// Seconds to hold a finished result before publishing it. Debug device tests only.
    public var resultDelay: TimeInterval = 0

    public var audioLevel: Float { phase == .recording ? capture.level : 0 }

    public private(set) var lastFailure: (any Error)?

    public var canRetry: Bool { phase == .failed && pendingUpload != nil }

    public init(channel: DictationChannel, capture: AudioCapturing, transcriber: Transcribing?, recordings: RecordingStore? = nil) {
        self.channel = channel
        self.capture = capture
        self.transcriber = transcriber
        self.recordings = recordings
        handledCommandSequence = channel.pendingCommand(after: 0)?.sequence ?? 0
    }

    public func setTranscriber(_ transcriber: Transcribing?) {
        // A new pairing must never receive text from the previous connection.
        endSession()
        activeRequestID = nil
        lastResult = nil
        lastResultID = nil
        lastError = nil
        lastFailure = nil
        self.transcriber = transcriber
        update(.init(phase: .idle))
    }

    public var isSessionOpen: Bool { sessionOpen }

    public func start(options: CompositionOptions = .init(), requestID: String? = nil) throws {
        guard phase != .recording, phase != .processing else { return }
        try options.validate()
        if !sessionOpen {
            try capture.startSession()
            sessionOpen = true
        }
        currentOptions = options
        activeRequestID = requestID
        generation = UUID()
        // A failed recording that was never retried stays on disk for History.
        pendingUpload = nil
        lastResult = nil
        lastResultID = nil
        lastError = nil
        lastFailure = nil
        capture.beginCapture()
        // Dictation still works when the disk can't be written; only the safety copy is missing.
        recordingID = try? recordings?.begin(options: options, requestID: requestID, now: now())
        savedSampleCount = 0
        lastAutosave = now()
        lastActivity = Date()
        channel.heartbeat()
        update(.init(phase: .recording))
    }

    public func stop() async {
        guard phase == .recording else { return }
        let stopAt = now()
        let samples = capture.endCapture()
        let recordingID = finishRecording(samples)
        lastActivity = Date()
        if isArmed { closeMicrophone() }
        let duration = WAVEncoder.duration(sampleCount: samples.count, sampleRate: Int(PCMDownsampler.targetSampleRate))
        guard duration >= Self.minimumDuration else {
            discard(recordingID)
            update(.init(phase: .idle))
            return
        }
        let audioMs = Int((duration * 1000).rounded())
        let level = SpeechGate.measure(samples, sampleRate: PCMDownsampler.targetSampleRate)
        guard SpeechGate.containsSpeech(samples, sampleRate: PCMDownsampler.targetSampleRate) else {
            discard(recordingID)
            let message = "No speech heard. Try again."
            channel.publishTimings(.init(requestID: activeRequestID, mode: currentOptions.mode.rawValue, audioMs: audioMs,
                                         stopAt: stopAt, encodeMs: 0, stopToResultMs: 0,
                                         peakDb: level.peakDb, speechMs: level.speechMs, error: message))
            update(.init(phase: .failed, errorMessage: message))
            return
        }
        let wav = WAVEncoder.encode(samples: samples, sampleRate: Int(PCMDownsampler.targetSampleRate))
        await submit(PendingUpload(wav: wav, recordingID: recordingID, options: currentOptions, requestID: activeRequestID, stopAt: stopAt,
                                   audioMs: audioMs, encodeMs: Self.ms(from: stopAt, to: now()), level: level))
    }

    /// Saves the audio recorded so far; called every `autosaveInterval` while recording, and by
    /// the app when it leaves the foreground.
    public func saveProgress() {
        guard phase == .recording, let recordingID, let recordings else { return }
        let samples = capture.capturedSamples(from: savedSampleCount)
        guard (try? recordings.append(samples, to: recordingID)) != nil else { return }
        savedSampleCount += samples.count
        lastAutosave = now()
    }

    /// Writes the samples not saved yet and hands over the recording's id.
    private func finishRecording(_ samples: [Float]) -> UUID? {
        defer { recordingID = nil; savedSampleCount = 0 }
        guard let recordingID, let recordings else { return nil }
        if samples.count > savedSampleCount {
            try? recordings.append(Array(samples[savedSampleCount...]), to: recordingID)
        }
        return recordingID
    }

    private func discard(_ recordingID: UUID?) {
        if let recordingID { recordings?.remove(recordingID) }
    }

    /// After a relaunch, offers the latest keyboard recording the Mac never answered to the
    /// keyboard's Retry, with its original request so the text can still land in that field.
    /// Recordings made in the app wait in History: no keyboard field is waiting for them.
    public func restoreUnsentRecording() {
        guard phase == .idle, let recordings else { return }
        recordings.recoverInterrupted()
        guard let saved = recordings.unprocessed.first, saved.status != .cancelled, saved.requestID != nil,
              now().timeIntervalSince(saved.startedAt) <= Self.keyboardRetryWindow,
              let wav = recordings.wav(saved.id) else { return }
        pendingUpload = PendingUpload(wav: wav, recordingID: saved.id, options: saved.options, requestID: saved.requestID,
                                      stopAt: now(), audioMs: Int((saved.duration * 1000).rounded()))
        activeRequestID = saved.requestID
        // A heartbeat left by the previous process would tell the keyboard this app can take a
        // Retry command in the background; with no session open, iOS suspends it instead.
        if !sessionOpen, !isArmed { channel.endSession() }
        fail(Self.savedRecordingMessage)
    }

    /// Sends a saved recording again from History. No keyboard request owns it, so its text
    /// goes to History and Home, never into a field.
    public func reprocess(_ id: UUID) async {
        guard phase != .recording, phase != .processing, let recordings,
              let saved = recordings.recording(id), let wav = recordings.wav(id) else { return }
        lastResult = nil
        lastResultID = nil
        lastError = nil
        lastFailure = nil
        await submit(PendingUpload(wav: wav, recordingID: id, options: saved.options, requestID: nil,
                                   stopAt: now(), audioMs: Int((saved.duration * 1000).rounded())))
    }

    private static func ms(from start: Date, to end: Date) -> Int {
        Int((end.timeIntervalSince(start) * 1000).rounded())
    }

    public func retry() async {
        guard canRetry, let pendingUpload else { return }
        await submit(pendingUpload)
    }

    private func submit(_ upload: PendingUpload) async {
        pendingUpload = upload
        activeRequestID = upload.requestID
        guard let transcriber else {
            if let id = upload.recordingID { recordings?.mark(id, .failed, errorMessage: "Pair with your Mac first.") }
            fail("Pair with your Mac first.")
            return
        }
        if let id = upload.recordingID { recordings?.mark(id, .processing) }
        // The keyboard reads "processing" with no heartbeat as a stopped app and cancels the
        // request; a resend with the microphone closed has had no heartbeat until now.
        channel.heartbeat(now: now())
        let requestGeneration = UUID()
        generation = requestGeneration
        update(.init(phase: .processing))
        let task = Task { try await transcriber.transcribeResult(wav: upload.wav, options: upload.options) }
        uploadTask = task
        do {
            let result = try await task.value
            if resultDelay > 0 { try? await Task.sleep(for: .seconds(resultDelay)) }
            guard generation == requestGeneration else { return }
            uploadTask = nil
            pendingUpload = nil
            discard(upload.recordingID)
            #if DEBUG
            channel.recordDebugResponse(result, mode: upload.options.mode)
            #endif
            channel.publishTimings(.init(
                requestID: upload.requestID, mode: upload.options.mode.rawValue, audioMs: upload.audioMs,
                stopAt: upload.stopAt, encodeMs: upload.encodeMs, preflightMs: result.preflightMs,
                requestMs: result.requestMs, hostAsrMs: result.asrMs, hostEnhanceMs: result.enhanceMs,
                stopToResultMs: Self.ms(from: upload.stopAt, to: now()),
                peakDb: upload.level.peakDb, speechMs: upload.level.speechMs
            ))
            lastResult = result.text
            lastResultID = result.id
            lastError = nil
        lastFailure = nil
            lastActivity = Date()
            update(.init(phase: .done, text: result.text, resultID: UUID().uuidString))
        } catch {
            guard generation == requestGeneration else { return }
            uploadTask = nil
            let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            channel.publishTimings(.init(
                requestID: upload.requestID, mode: upload.options.mode.rawValue, audioMs: upload.audioMs,
                stopAt: upload.stopAt, encodeMs: upload.encodeMs, stopToResultMs: Self.ms(from: upload.stopAt, to: now()),
                peakDb: upload.level.peakDb, speechMs: upload.level.speechMs, error: message
            ))
            if let id = upload.recordingID { recordings?.mark(id, .failed, errorMessage: message) }
            fail(message, error: error)
        }
    }

    public func cancel() {
        abandon(captured: nil, keepingAudio: { $0 >= Self.keepCancelledAfter ? .cancelled : nil })
    }

    /// Stops the current dictation. `keepingAudio` decides, from the audio's length, whether a
    /// recording or upload in progress stays on disk; a failed one always stays for History.
    private func abandon(captured: [Float]?, keepingAudio: (TimeInterval) -> SavedRecording.Status?) {
        guard phase == .recording || phase == .processing || phase == .failed else { return }
        let abandoned: UUID? = switch phase {
        case .recording: finishRecording(captured ?? capture.endCapture())
        case .processing: pendingUpload?.recordingID
        default: nil
        }
        if let abandoned, let recordings {
            let duration = recordings.recording(abandoned)?.duration ?? 0
            if duration >= Self.minimumDuration, let status = keepingAudio(duration) {
                recordings.mark(abandoned, status)
            } else {
                recordings.remove(abandoned)
            }
        }
        if isArmed { closeMicrophone() }
        generation = UUID()
        uploadTask?.cancel()
        uploadTask = nil
        pendingUpload = nil
        lastResult = nil
        lastResultID = nil
        lastError = nil
        lastFailure = nil
        update(.init(phase: .idle))
    }

    public func toggle(options: CompositionOptions = .init()) async throws {
        if phase == .recording { await stop() } else { try start(options: options) }
    }

    public func endSession() {
        isArmed = false
        // Close the microphone before any publish so observers never see a torn-down
        // session with the mic still on; cancel() publishes nothing at idle/done, so
        // those phases need an explicit final notification.
        let cancelPublishes = phase == .recording || phase == .processing || phase == .failed
        // Take what was said before the microphone closes, which clears the capture.
        let captured = phase == .recording ? capture.endCapture() : nil
        closeMicrophone()
        abandon(captured: captured, keepingAudio: { _ in .interrupted })
        channel.endSession()
        if !cancelPublishes { onChange?() }
    }

    /// Arms or disarms PiP readiness. Arming publishes a heartbeat without touching the
    /// microphone; disarming is a full teardown (cancel, mic off, heartbeat cleared).
    public func setArmed(_ armed: Bool) {
        guard armed != isArmed else { return }
        if armed {
            isArmed = true
            lastActivity = Date()
            channel.heartbeat()
            // A legacy idle session keeps the mic open; PiP readiness means mic-off idle.
            // An active recording is never interrupted — its stop closes the mic instead.
            if phase != .recording { closeMicrophone() }
            onChange?()
        } else {
            endSession()
        }
    }

    /// Stops the engine and deactivates the audio session without touching readiness.
    private func closeMicrophone() {
        guard sessionOpen else { return }
        capture.endSession()
        sessionOpen = false
    }

    /// Reads the keyboard's latest command, if it hasn't been handled yet.
    public func handlePendingCommand() async {
        guard let command = channel.pendingCommand(after: handledCommandSequence) else { return }
        handledCommandSequence = command.sequence
        switch command.kind {
        case .start:
            do { try start(options: command.options ?? .init(), requestID: command.requestID) } catch { fail(error.localizedDescription) }
        case .stop:
            await stop()
        case .cancel:
            cancel()
        case .retry:
            await retry()
        }
    }

    /// Called every second while the app runs: keeps the heartbeat fresh and closes an idle session.
    public func tick(now: Date = Date()) {
        if phase == .recording, now.timeIntervalSince(lastAutosave) >= autosaveInterval {
            saveProgress()
            lastAutosave = now
        }
        if isArmed {
            // PiP holds readiness; the idle timeout only governs the legacy session.
            channel.heartbeat(now: now)
            return
        }
        guard sessionOpen else {
            // A resend with the microphone closed (after a relaunch, or from History): without
            // a heartbeat the keyboard would take the request for a stopped app and cancel it.
            if phase == .processing { channel.heartbeat(now: now) }
            return
        }
        if phase != .recording, phase != .processing, let idleTimeout,
           now.timeIntervalSince(lastActivity) > idleTimeout {
            endSession()
            return
        }
        channel.heartbeat(now: now)
    }

    private func fail(_ message: String, error: (any Error)? = nil) {
        lastFailure = error
        lastError = message
        var state = DictationChannel.State(phase: .failed, errorMessage: message)
        if (error as? VoxtClient.Error) == .transportUnavailable { state.reconnectInApp = true }
        update(state)
    }

    private func update(_ state: DictationChannel.State) {
        var state = state
        phase = state.phase
        state.requestID = activeRequestID
        state.canRetry = canRetry
        channel.publish(state)
        onChange?()
    }
}
