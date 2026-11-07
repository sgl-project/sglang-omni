@preconcurrency import AVFoundation
import Foundation
@preconcurrency import Speech
import VoxtNewHelperKit

/// speech.recognize: macOS's own speech recognition, on this Mac only
/// (requiresOnDeviceRecognition), of the audio speech.audio brings. A job
/// answers itself once its recognizer gives the final text or fails, or
/// when it is cancelled.
final class SpeechJobs: @unchecked Sendable {
    let out: Writer
    private let lock = NSLock()
    private var jobs: [String: SpeechJob] = [:]
    /// The recognizers' callbacks, in order and off the main thread.
    private let queue = OperationQueue()

    init(out: Writer) {
        self.out = out
        queue.maxConcurrentOperationCount = 1
    }

    /// Starts the job; it answers when its recognizer ends.
    func start(job id: String, _ p: SpeechParams) async throws {
        guard await Permissions.askSpeechRecognition() else {
            throw RPCError.denied(PermissionName.speechRecognition, "Speech Recognition is not allowed")
        }
        let recognizer = try Self.recognizer(p.lang ?? "")
        recognizer.queue = queue
        let rate = Double(p.sampleRate ?? 16000)
        guard rate > 0, let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false) else {
            throw RPCError(Code.badRequest, "Bad sample rate \(rate)")
        }
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.requiresOnDeviceRecognition = true // the audio never leaves the Mac
        request.shouldReportPartialResults = true
        request.taskHint = .dictation
        request.addsPunctuation = true
        request.contextualStrings = p.hints ?? []
        let job = SpeechJob(request: request, format: format)
        lock.withLock { jobs[id] = job }
        out.notify(Method.speechStarted, SpeechStartedParams(job: id, locale: recognizer.locale.identifier(.bcp47)))
        let task = recognizer.recognitionTask(with: request) { [self] result, error in
            if let result, result.isFinal {
                finish(id) { out.result(id, SpeechResult(text: result.bestTranscription.formattedString)) }
            } else if let result {
                if lock.withLock({ jobs[id] }) != nil {
                    out.notify(Method.speechPartial, SpeechPartialParams(job: id, text: result.bestTranscription.formattedString))
                }
            } else if let error {
                let e = error as NSError
                if e.domain == "kAFAssistantErrorDomain" && e.code == 1110 { // no speech detected: nothing was said
                    finish(id) { out.result(id, SpeechResult(text: "")) }
                } else {
                    finish(id) { out.fail(id, RPCError(Code.internal, errorMessage(error))) }
                }
            }
        }
        lock.withLock {
            if jobs[id] === job { job.task = task } else { task.cancel() } // it ended already
        }
    }

    /// Appends speech.audio's PCM to its job; the reader thread calls it,
    /// so that the audio stays in order.
    func audio(_ p: SpeechAudioParams) throws {
        guard let job = lock.withLock({ jobs[p.job] }) else {
            throw RPCError(Code.badRequest, "No speech.recognize job \(p.job)")
        }
        if let data = p.data, data.count >= 2 {
            let frames = data.count / 2
            guard let buffer = AVAudioPCMBuffer(pcmFormat: job.format, frameCapacity: AVAudioFrameCount(frames)),
                  let samples = buffer.floatChannelData?[0]
            else { throw RPCError(Code.internal, "No audio buffer for \(frames) frames") }
            buffer.frameLength = AVAudioFrameCount(frames)
            data.withUnsafeBytes { raw in
                for i in 0..<frames {
                    samples[i] = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: 2 * i, as: Int16.self))) / 32768
                }
            }
            job.request.append(buffer)
        }
        if p.end == true {
            job.request.endAudio()
        }
    }

    /// Discards the job (it answers cancelled first); false when not running.
    func cancel(job id: String) -> Bool {
        finish(id) { out.fail(id, RPCError(Code.cancelled, "Cancelled")) }
    }

    func cancelAll() {
        for id in lock.withLock({ Array(jobs.keys) }) {
            _ = cancel(job: id)
        }
    }

    /// Ends job id once: stops its recognizer, then answer answers it.
    @discardableResult
    private func finish(_ id: String, answer: () -> Void) -> Bool {
        guard let job = lock.withLock({ jobs.removeValue(forKey: id) }) else { return false }
        job.task?.cancel()
        answer()
        return true
    }

    /// The recognizer for lang ("": the Mac's own dictation language);
    /// unsupported when macOS has no on-device model for it.
    static func recognizer(_ lang: String) throws -> SFSpeechRecognizer {
        var recognizer: SFSpeechRecognizer?
        if lang.isEmpty {
            recognizer = SFSpeechRecognizer()
        } else if let locale = speechLocale(lang, supported: SFSpeechRecognizer.supportedLocales().map { $0.identifier(.bcp47) },
                                            userRegion: Locale.current.region?.identifier) {
            recognizer = SFSpeechRecognizer(locale: Locale(identifier: locale))
        }
        guard let recognizer, recognizer.supportsOnDeviceRecognition else {
            throw RPCError(Code.unsupported, "macOS has no on-device speech recognition for \(lang.isEmpty ? "the Mac's language" : lang)",
                           data: .object(["capability": .string("onDeviceSpeech")]))
        }
        return recognizer
    }
}

private final class SpeechJob: @unchecked Sendable {
    let request: SFSpeechAudioBufferRecognitionRequest
    let format: AVAudioFormat
    /// Set under SpeechJobs' lock.
    var task: SFSpeechRecognitionTask?

    init(request: SFSpeechAudioBufferRecognitionRequest, format: AVAudioFormat) {
        self.request = request
        self.format = format
    }
}
