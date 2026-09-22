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
        let server = Process()
        server.executableURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["OMNITYPER_TEST_PYTHON"] ?? "/usr/bin/python3")
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("realtime_server.py")
        server.arguments = [fixture.path, portFile.path]
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
        try await stream.connect(language: "en")
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

        let broken = try ASRStream(url: url, onPartial: { _ in }, onFailure: { failed = true })
        defer { broken.cancel() }
        try await broken.connect(language: "fail")
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
