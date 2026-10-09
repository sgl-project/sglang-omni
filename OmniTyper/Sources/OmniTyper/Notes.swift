// SPDX-License-Identifier: Apache-2.0
import AppKit
import Combine
import Foundation

struct Note: Codable, Identifiable, Equatable {
    enum Status: String, Codable { case recording, transcribing, writing, ready, interrupted, failed }

    var id = UUID()
    var date = Date()
    var title: String
    var duration: Double = 0
    var segments: [String] = []
    var notes = ""
    var status: Status = .recording
    var warning: String?

    var transcript: String { segments.filter { !$0.isEmpty }.joined(separator: "\n") }
    var isUnfinished: Bool { [.recording, .transcribing, .writing].contains(status) }

    var markdown: String {
        var text = notes.isEmpty ? "# \(title)" : notes
        if !transcript.isEmpty { text += "\n\n## \(L("notes.transcript"))\n\n" + transcript }
        return text + "\n"
    }

    static func defaultTitle(for date: Date) -> String {
        L("notes.defaultTitle", date.formatted(date: .abbreviated, time: .shortened))
    }

    static func title(fromMarkdown markdown: String) -> String? {
        guard let line = markdown.split(separator: "\n").first(where: { $0.hasPrefix("# ") }) else { return nil }
        let title = line.dropFirst(2).trimmingCharacters(in: .whitespaces)
        return title.isEmpty ? nil : String(title.prefix(120))
    }
}

@MainActor
final class NoteStore: ObservableObject {
    private struct Saved: Codable {
        var version = 1
        var notes: [Note]
    }

    @Published private(set) var notes: [Note] = []
    @Published private(set) var storageError = ""
    private let file: URL
    private var canSave = true

    init(directory: URL) {
        file = directory.appendingPathComponent("notes.json")
        do {
            if FileManager.default.fileExists(atPath: file.path) {
                let saved = try JSONDecoder().decode(Saved.self, from: Data(contentsOf: file))
                guard saved.version == 1 else { throw CocoaError(.coderReadCorrupt) }
                notes = saved.notes
            }
        } catch {
            canSave = false
            storageError = L("error.libraryLoad", error.localizedDescription)
        }
        for index in notes.indices where notes[index].isUnfinished {
            notes[index].status = .interrupted
            notes[index].warning = L("notes.warning.interrupted")
        }
    }

    func note(_ id: UUID) -> Note? { notes.first { $0.id == id } }

    func insert(_ note: Note) {
        notes.insert(note, at: 0)
        persist()
    }

    func update(_ id: UUID, _ change: (inout Note) -> Void) {
        guard let index = notes.firstIndex(where: { $0.id == id }) else { return }
        change(&notes[index])
        persist()
    }

    func delete(_ id: UUID) {
        notes.removeAll { $0.id == id }
        persist()
    }

    private func persist() {
        guard canSave else { return }
        do {
            try JSONEncoder().encode(Saved(notes: notes)).write(to: file, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
            storageError = ""
        } catch { storageError = L("error.librarySave", error.localizedDescription) }
    }
}

/// Splits 16 kHz mono PCM into chunks for batch transcription, cutting at the quietest window
/// between the minimum and maximum length so words are rarely split.
struct SpeechChunker {
    var minimumSamples = 15 * 16_000
    var maximumSamples = 30 * 16_000
    var windowSamples = 4_000
    private(set) var pending: [Int16] = []

    mutating func append(_ samples: [Int16]) -> [[Int16]] {
        pending += samples
        var chunks: [[Int16]] = []
        while pending.count >= maximumSamples {
            let cut = quietestCut()
            chunks.append(Array(pending[..<cut]))
            pending.removeFirst(cut)
        }
        return chunks
    }

    mutating func flush() -> [Int16]? {
        defer { pending = [] }
        return pending.isEmpty ? nil : pending
    }

    private func quietestCut() -> Int {
        var best = maximumSamples
        var bestEnergy = Int64.max
        var start = minimumSamples
        while start + windowSamples <= maximumSamples {
            let energy = pending[start..<start + windowSamples].reduce(Int64(0)) { $0 + Int64($1) * Int64($1) }
            if energy < bestEnergy { bestEnergy = energy; best = start + windowSamples / 2 }
            start += windowSamples
        }
        return best
    }

    static func wav(_ samples: [Int16], rate: Int = 16_000) -> Data {
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = samples.count * 2
        data.append(contentsOf: Array("RIFF".utf8)); append(UInt32(36 + bytes))
        data.append(contentsOf: Array("WAVEfmt ".utf8)); append(UInt32(16)); append(UInt16(1)); append(UInt16(1))
        append(UInt32(rate)); append(UInt32(rate * 2)); append(UInt16(2)); append(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); append(UInt32(bytes))
        samples.withUnsafeBufferPointer { buffer in
            for sample in buffer { append(sample) }
        }
        return data
    }
}

private final class ChunkFeed: @unchecked Sendable {
    private let lock = NSLock()
    private var chunker = SpeechChunker()
    private let deliver: @MainActor ([Int16]) -> Void

    init(deliver: @escaping @MainActor ([Int16]) -> Void) { self.deliver = deliver }

    func receive(_ data: Data) {
        let samples = data.withUnsafeBytes { Array($0.bindMemory(to: Int16.self)).map { Int16(littleEndian: $0) } }
        lock.lock()
        let chunks = chunker.append(samples)
        lock.unlock()
        for chunk in chunks { post(chunk) }
    }

    /// Posts the remainder behind every chunk already queued, then calls `done`.
    func finish(_ done: @escaping @MainActor () -> Void) {
        lock.lock()
        let rest = chunker.flush()
        lock.unlock()
        let deliver = deliver
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                if let rest { deliver(rest) }
                done()
            }
        }
    }

    private func post(_ chunk: [Int16]) {
        let deliver = deliver
        DispatchQueue.main.async { MainActor.assumeIsolated { deliver(chunk) } }
    }
}

@MainActor
final class NoteTaker: ObservableObject {
    enum Phase { case idle, starting, recording, finishing }
    static let maximumSeconds = 3.0 * 60 * 60

    @Published private(set) var phase: Phase = .idle { didSet { model.objectWillChange.send() } }
    @Published private(set) var activeID: UUID?
    @Published private(set) var status = ""
    @Published var error = ""
    let store: NoteStore
    private unowned let model: AppModel
    private var feed: ChunkFeed?
    private var transcription: Task<Void, Never>?
    private var task: Task<Void, Never>?
    private var timer: Timer?

    var isActive: Bool { phase != .idle }

    init(model: AppModel) {
        self.model = model
        store = NoteStore(directory: model.store.directory)
    }

    func start() {
        guard phase == .idle, model.phase == .idle else { return }
        let preferences = model.store.preferences
        let note = Note(title: Note.defaultTitle(for: Date()))
        error = ""
        store.insert(note)
        activeID = note.id
        phase = .starting
        status = L("notes.status.loading")
        task = Task {
            await model.finishCancelledLoad()
            do {
                _ = try await model.worker.request(["op": "prepare", "asr_model": preferences.asrModel],
                                                   python: preferences.pythonExecutable)
                guard phase == .starting else { return }
                let feed = ChunkFeed { [weak self] samples in self?.transcribe(samples, into: note.id) }
                self.feed = feed
                try await model.recorder.start(deviceUID: preferences.microphoneUID, onPCM: { feed.receive($0) },
                                               keepsFile: false, maximumSeconds: Self.maximumSeconds)
                phase = .recording
                status = L("notes.status.listening")
                timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
                    MainActor.assumeIsolated { self?.tick() }
                }
            } catch {
                self.feed = nil
                store.update(note.id) { $0.status = .failed; $0.warning = error.localizedDescription }
                self.error = error.localizedDescription
                finishSession()
            }
        }
    }

    func stop(warning: String? = nil) {
        guard phase == .recording, let id = activeID, let feed else { return }
        timer?.invalidate(); timer = nil
        let duration = model.recorder.elapsed
        try? model.recorder.stopStream()
        self.feed = nil
        phase = .finishing
        status = L("notes.status.transcribing")
        store.update(id) {
            $0.duration = duration
            $0.status = .transcribing
            if let warning { $0.warning = warning }
        }
        feed.finish { [weak self] in
            guard let self else { return }
            let transcription = self.transcription
            self.task = Task {
                await transcription?.value
                await self.writeNotes(id)
                self.finishSession()
            }
        }
    }

    func rewrite(_ id: UUID) {
        guard phase == .idle, model.phase == .idle, store.note(id) != nil else { return }
        error = ""
        activeID = id
        phase = .finishing
        task = Task {
            await model.finishCancelledLoad()
            await writeNotes(id)
            finishSession()
        }
    }

    func shutdown() {
        timer?.invalidate(); timer = nil
        task?.cancel(); transcription?.cancel()
        if phase == .recording { try? model.recorder.stopStream() }
        if let id = activeID, store.note(id)?.isUnfinished == true {
            store.update(id) {
                if $0.status == .recording { $0.duration = model.recorder.elapsed }
                $0.status = .interrupted
                $0.warning = L("notes.warning.interrupted")
            }
        }
        feed = nil
        phase = .idle
        activeID = nil
    }

    private func tick() {
        guard phase == .recording, let id = activeID else { return }
        let elapsed = model.recorder.elapsed
        store.update(id) { $0.duration = elapsed }
        if model.recorder.captureError != nil { stop(warning: L("notes.warning.micStopped")) }
        else if elapsed >= Self.maximumSeconds { stop(warning: L("notes.warning.limit")) }
    }

    private func transcribe(_ samples: [Int16], into id: UUID) {
        let previous = transcription
        let preferences = model.store.preferences
        let dictionary = model.store.dictionary.filter(\.isValid).prefix(200)
            .map { ["spoken": $0.spoken, "written": $0.written] }
        transcription = Task {
            await previous?.value
            guard !Task.isCancelled else { return }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("OmniTyper-note-\(UUID().uuidString).wav")
            defer { try? FileManager.default.removeItem(at: url) }
            do {
                try SpeechChunker.wav(samples).write(to: url, options: .atomic)
                let response = try await model.worker.request([
                    "op": "transcribe", "asr_model": preferences.asrModel, "mode": "dictate", "style": "verbatim",
                    "language": preferences.language, "dictionary": Array(dictionary), "audio_path": url.path,
                ], python: preferences.pythonExecutable)
                let text = (response["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { store.update(id) { $0.segments.append(text) } }
            } catch {
                guard !Task.isCancelled else { return }
                store.update(id) { $0.warning = L("notes.warning.chunk") }
                Diagnostics.record("notes.chunkFailed", ["reason": Diagnostics.code(of: error)])
            }
        }
    }

    private func writeNotes(_ id: UUID) async {
        guard let note = store.note(id) else { return }
        guard !note.transcript.isEmpty else {
            store.update(id) { $0.status = .ready; $0.warning = $0.warning ?? L("notes.warning.noSpeech") }
            return
        }
        let preferences = model.store.preferences
        guard !preferences.textSettings.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            store.update(id) { $0.status = .ready; $0.warning = L("notes.warning.noModel") }
            return
        }
        status = L("notes.status.writing")
        store.update(id) { $0.status = .writing }
        do {
            var request = try preferences.textSettings.payload(apiKey: model.textAPIKey)
            request["op"] = "notes"
            request["asr_model"] = preferences.asrModel
            request["language"] = preferences.language
            request["text"] = note.transcript
            let response = try await model.worker.request(request, python: preferences.pythonExecutable)
            let markdown = (response["text"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            store.update(id) {
                $0.notes = markdown
                $0.title = Note.title(fromMarkdown: markdown) ?? $0.title
                $0.status = .ready
                if $0.warning == L("notes.warning.noModel") { $0.warning = nil }
            }
        } catch {
            store.update(id) { $0.status = .failed; $0.warning = error.localizedDescription }
            self.error = error.localizedDescription
        }
    }

    private func finishSession() {
        timer?.invalidate(); timer = nil
        task = nil
        transcription = nil
        activeID = nil
        status = ""
        phase = .idle
    }
}
