import Foundation
import Darwin

@MainActor
private final class FileRecorder: AudioRecording {
    let wav: Data
    init(_ wav: Data) { self.wav = wav }
    func start() async throws { }
    func stop() throws -> Data { wav }
    func cancel() { }
}

/// Runs the actual revision math, but never accesses a real editor or pasteboard.
@MainActor
private final class DraftTarget: TextRevisionTarget {
    var text: String
    var writes = 0
    init(_ text: String) { self.text = text }
    func confirmInsertion() async -> Bool { true }
    func prepare(original: String) throws {
        guard text == original else { throw DictationError("Test draft changed") }
    }
    func apply(_ revision: TextRevision) async throws -> Bool {
        text = try TextDraft(value: text, selection: revision.range).inserting(revision.replacement)
        writes += 1
        print("PATCH UTF16: \(revision.range.location), \(revision.range.length); replacement: \(revision.replacement)")
        return text.utf16.elementsEqual(revision.corrected.utf16)
    }
    func cancel() { }
}

@MainActor
private func wait(_ predicate: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 180
    while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 10_000_000)
    }
    guard predicate() else { throw DictationError("Audio correction test timed out") }
}

@main
private enum AudioCorrectionTests {
    @MainActor
    static func main() async {
        setbuf(stdout, nil)
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        guard ProcessInfo.processInfo.environment["OMNI_CORRECTION_AUDIO_TEST"] == "1" else {
            throw DictationError("Set OMNI_CORRECTION_AUDIO_TEST=1 to send the supplied files to local models.")
        }
        let useSaved = CommandLine.arguments.contains("--saved-settings")
        let args = CommandLine.arguments.dropFirst().filter { $0 != "--saved-settings" }
        guard (2...3).contains(args.count) else {
            throw DictationError("Usage: audio_correction_test.sh [--saved-settings] original-audio correction-audio [expected-text]")
        }
        let preferences = useSaved ? UserDefaults.standard.persistentDomain(forName: "local.omni.dictation") ?? [:] : [:]
        var configuration = try (preferences["serviceConfiguration"] as? Data)
            .map { try JSONDecoder().decode(ServiceConfiguration.self, from: $0) } ?? ServiceConfiguration()
        if let model = ProcessInfo.processInfo.environment["OMNI_CORRECTION_MODEL"] {
            configuration.polish = try LocalModelConfiguration(baseURL: configuration.polish.baseURL.absoluteString, model: model)
        }
        let background = preferences["personalBackgroundEnabled"] as? Bool == true
            ? preferences["personalBackground"] as? String ?? "" : ""
        let service = LocalSpeechService(configuration: configuration)
        let ordinary = DictationSession(recorder: FileRecorder(Data()), service: service)
        ordinary.polishEnabled = preferences["polishEnabled"] as? Bool == true
        ordinary.personalBackground = background
        print("ASR: \(configuration.asr.model); Ollama: \(configuration.polish.model)")
        print("Saved settings: \(useSaved); polishing: \(ordinary.polishEnabled); background enabled: \(!background.isEmpty)")
        ordinary.submitAudio(try AudioEncoder.load(URL(fileURLWithPath: args[0])))
        try await wait { !ordinary.isBusy }
        guard ordinary.phase == .ready, !ordinary.resultText.isEmpty else { throw DictationError(ordinary.notice) }
        print("ORIGINAL ASR: \(ordinary.rawText)")
        print("DELIVERED: \(ordinary.resultText)")
        if !ordinary.notice.isEmpty { print("ORDINARY NOTICE: \(ordinary.notice)") }
        let target = DraftTarget(ordinary.resultText)
        let correction = CorrectionSession(recorder: FileRecorder(try AudioEncoder.load(URL(fileURLWithPath: args[1]))),
                                           service: service, makeCorrector: { service.makeCorrector() })
        defer { ordinary.cancel(); correction.clear() }
        correction.remember(ordinary.resultText)
        correction.trackInsertion(target)
        try await wait { correction.canReplace }
        correction.toggleRecording(personalBackground: background)
        try await wait { correction.phase == .recording || correction.phase == .failed }
        if correction.phase == .recording { correction.toggleRecording(personalBackground: background) }
        try await wait { !correction.isBusy }
        print("INSTRUCTION ASR: \(correction.instruction)")
        print("CORRECTED: \(correction.correctedText ?? "<none>")")
        print("NOTICE: \(correction.notice)")
        print("DRAFT: \(target.text); writes: \(target.writes)")
        guard correction.didReplace, target.writes == 1 else {
            throw DictationError("No confirmed correction was produced for these audio files")
        }
        if args.count == 3, target.text.caseInsensitiveCompare(args[2]) != .orderedSame {
            throw DictationError("Correction differs from expected text (ignoring letter case)")
        }
        print("PASS: real file decoder, ASR, optional polishing, correction session and draft patch. No real microphone/editor events.")
    }
}
