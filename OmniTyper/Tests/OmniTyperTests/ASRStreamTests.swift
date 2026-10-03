// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import OmniTyper

struct ASRStreamTests {
    @Test func revisedSegmentsReplaceEarlierWords() throws {
        var preview = TranscriptionPreview()
        func segment(_ index: Int, _ id: Int, _ text: String, final: Bool = false) -> [String: Any] {
            ["type": "transcription.segment", "event_index": index, "segment_id": id, "text": text, "is_final": final]
        }
        _ = try preview.receive(segment(0, 0, "hello word"))
        _ = try preview.receive(segment(1, 0, "hello world", final: true))
        _ = try preview.receive(segment(2, 0, "late revision"))
        _ = try preview.receive(segment(3, 1, "你好"))
        _ = try preview.receive(segment(2, 1, "stale event"))
        #expect(preview.text == "hello world\n你好")
        let final = try preview.receive(["type": "transcription.completed", "event_index": 4, "text": "Authoritative final text."])
        #expect(final == "Authoritative final text.")
        #expect(throws: (any Error).self) { try preview.receive(segment(5, -1, "invalid")) }
        #expect(throws: (any Error).self) { try preview.receive(segment(6, 2, String(repeating: "x", count: 12_001))) }
        #expect(throws: (any Error).self) { try preview.receive(["type": "error"]) }
    }

    @Test @MainActor func liveTransportDrainsAudioAndHandlesFailureAndCancel() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let portFile = directory.appendingPathComponent("port")
        let sessionFile = directory.appendingPathComponent("session.json")
        func sessionUpdate() throws -> [String: Any] {
            try #require(JSONSerialization.jsonObject(with: Data(contentsOf: sessionFile)) as? [String: Any])
        }
        let server = Process()
        server.executableURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["OMNITYPER_TEST_PYTHON"] ?? "/usr/bin/python3")
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("realtime_server.py")
        server.arguments = [fixture.path, portFile.path, sessionFile.path]
        server.standardOutput = FileHandle.nullDevice
        try server.run()
        defer { if server.isRunning { server.terminate() }; server.waitUntilExit() }
        for _ in 0..<200 {
            if FileManager.default.fileExists(atPath: portFile.path) { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let port = try String(contentsOf: portFile, encoding: .utf8)
        let url = "ws://127.0.0.1:\(port)/v1/realtime?intent=transcription"
        #expect(throws: (any Error).self) { try ASRStream(url: "ws://example.com:80/v1/realtime?intent=transcription", onPartial: { _ in }, onFailure: {}) }

        var partials: [String] = []
        var failed = false
        let stream = try ASRStream(url: url, onPartial: { partials.append($0) }, onFailure: { failed = true })
        defer { stream.cancel() }
        let hotwords = ["SGLang", "你好 café", #"say "hello" at C:\work"#, "<|im_end|>"]
        try await stream.connect(language: "en", hotwords: hotwords)
        let prompt = try #require(try sessionUpdate()["prompt"] as? String)
        #expect(try JSONSerialization.jsonObject(with: Data(prompt.utf8)) as? [String] == hotwords)
        #expect(!prompt.contains("<"))
        #expect(prompt.contains("\\u003c"))
        stream.audioInput(Data(repeating: 1, count: 3200))
        for _ in 0..<200 {
            if partials.count == 2 { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(partials == ["hello word", "hello world"], "Words must appear before finish is called")
        stream.audioInput(Data(repeating: 2, count: 3200))
        let final = try await stream.finish()
        #expect(final == "Hello world. Final text.")
        #expect(!failed)

        let manyHotwords = (0..<25).map { "word \($0)" }
        let escapedHotwords = Array(repeating: String(repeating: "<", count: 120), count: 20)
        let combiningHotwords = Array(repeating: String(repeating: "e\u{301}<", count: 40), count: 20)
        for (words, expected) in [(manyHotwords, Array(manyHotwords.prefix(20))),
                                  (escapedHotwords, Array(escapedHotwords.prefix(5))),
                                  (combiningHotwords, Array(combiningHotwords.prefix(12)))] {
            let bounded = try ASRStream(url: url, onPartial: { _ in }, onFailure: { failed = true })
            defer { bounded.cancel() }
            try await bounded.connect(language: "en", hotwords: words)
            let boundedPrompt = try #require(try sessionUpdate()["prompt"] as? String)
            #expect(boundedPrompt.unicodeScalars.count <= 4096)
            #expect(try JSONSerialization.jsonObject(with: Data(boundedPrompt.utf8)) as? [String] == expected)
        }

        let broken = try ASRStream(url: url, onPartial: { _ in }, onFailure: { failed = true })
        defer { broken.cancel() }
        try await broken.connect(language: "fail")
        #expect(try sessionUpdate()["prompt"] == nil, "A new session without a dictionary must omit earlier hints")
        broken.audioInput(Data(repeating: 1, count: 3200))
        for _ in 0..<200 {
            if failed { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        #expect(failed)
        do { _ = try await broken.finish(); Issue.record("A broken stream must fail so the app can use its WAV") }
        catch { }

        failed = false
        let slow = try ASRStream(url: url, onPartial: { _ in }, onFailure: { failed = true })
        defer { slow.cancel() }
        try await slow.connect(language: "en")
        for _ in 0..<100 { slow.audioInput(Data(repeating: 1, count: 3200)) }
        do { _ = try await slow.finish(); Issue.record("A dropped PCM packet must trigger full-recording recovery") }
        catch { }
        #expect(failed)

        let cancelled = try ASRStream(url: url, onPartial: { _ in }, onFailure: { Issue.record("Cancellation should not report failure") })
        let connecting = Task { try await cancelled.connect(language: "stall") }
        try await Task.sleep(nanoseconds: 100_000_000)
        cancelled.cancel()
        do { try await connecting.value; Issue.record("Cancelled connection succeeded") }
        catch { }
    }
}
