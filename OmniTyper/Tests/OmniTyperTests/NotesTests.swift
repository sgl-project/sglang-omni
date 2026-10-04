// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import OmniTyper

struct NotesTests {
    @Test func chunkerCutsAtTheQuietestWindowAndKeepsEverySample() {
        var chunker = SpeechChunker(minimumSamples: 100, maximumSamples: 200, windowSamples: 10)
        var audio = [Int16](repeating: 1_000, count: 450)
        for index in 150..<160 { audio[index] = 0 }
        let chunks = chunker.append(Array(audio[..<250])) + chunker.append(Array(audio[250...]))
        #expect(chunks.first?.count == 155)
        #expect(chunks.allSatisfy { $0.count >= 100 && $0.count <= 200 })
        let rest = chunker.flush() ?? []
        #expect(chunks.flatMap { $0 } + rest == audio)
        #expect(chunker.flush() == nil)
    }

    @Test func chunkWavIsSixteenKilohertzMonoPCM() {
        let data = SpeechChunker.wav([0, 1, -1, .max])
        #expect(data.count == 44 + 8)
        #expect(String(decoding: data[0..<4], as: UTF8.self) == "RIFF")
        #expect(String(decoding: data[8..<16], as: UTF8.self) == "WAVEfmt ")
        func uint32(_ offset: Int) -> UInt32 { data[offset..<offset + 4].reversed().reduce(0) { $0 << 8 | UInt32($1) } }
        #expect(uint32(24) == 16_000)
        #expect(uint32(40) == 8)
        #expect(data[46] == 1 && data[47] == 0 && data[48] == 0xFF && data[49] == 0xFF)
    }

    @Test func titleComesFromTheFirstHeading() {
        #expect(Note.title(fromMarkdown: "intro\n# Weekly sync \n## Summary") == "Weekly sync")
        #expect(Note.title(fromMarkdown: "## Summary\n- point") == nil)
        #expect(Note.title(fromMarkdown: "# " + String(repeating: "x", count: 300))?.count == 120)
    }

    @Test func markdownExportAppendsTheTranscript() {
        var note = Note(title: "Sync", segments: ["first", "", "second"])
        #expect(note.transcript == "first\nsecond")
        #expect(note.markdown.hasPrefix("# Sync\n\n## "))
        note.notes = "# Sync\n## Summary\nok"
        #expect(note.markdown.hasPrefix("# Sync\n## Summary\nok\n\n## "))
        #expect(note.markdown.hasSuffix("first\nsecond\n"))
    }

    @Test @MainActor func storePersistsAndMarksUnfinishedNotesInterrupted() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("OmniTyperNotes-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = NoteStore(directory: directory)
        let recording = Note(title: "Live")
        var done = Note(title: "Done")
        done.status = .ready
        store.insert(done)
        store.insert(recording)
        store.update(recording.id) { $0.segments.append("hello") }

        let reloaded = NoteStore(directory: directory)
        #expect(reloaded.notes.map(\.title) == ["Live", "Done"])
        #expect(reloaded.note(recording.id)?.status == .interrupted)
        #expect(reloaded.note(recording.id)?.transcript == "hello")
        #expect(reloaded.note(done.id)?.status == .ready)
        reloaded.delete(done.id)
        #expect(NoteStore(directory: directory).notes.count == 1)
        let attributes = try FileManager.default.attributesOfItem(atPath: directory.appendingPathComponent("notes.json").path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    }
}
