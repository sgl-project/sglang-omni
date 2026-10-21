// LocalAPIDictationPipelineTests.swift
// Provides Local API dictation pipeline tests for Voxt test coverage.

import XCTest
@testable import Voxt

@MainActor
final class LocalAPIDictationPipelineTests: XCTestCase {
    private var directory: URL!
    private var transcribedFiles: [(URL, Data)] = []
    private var transcribedLanguages: [ASRLanguageSelection] = []
    private var enhancedInputs: [String] = []
    private var recorded: [LocalAPIDictationOutcome] = []

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LocalAPIDictationPipelineTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        transcribedFiles = []
        transcribedLanguages = []
        enhancedInputs = []
        recorded = []
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makePipeline(
        busyReason: String? = nil,
        transcript: Result<String, Error> = .success("嗯 那个 明天下午三点开会"),
        enhanced: Result<String, Error> = .success("明天下午三点开会。"),
        transcribeDelay: Duration = .zero,
        checksAudioWithAVFoundation: Bool = false
    ) -> LocalAPIDictationPipeline {
        LocalAPIDictationPipeline(
            temporaryDirectory: directory,
            busyReason: { busyReason },
            isReadableAudio: checksAudioWithAVFoundation ? LocalAPIDictationPipeline.isReadableAudioFile : { _ in true },
            transcribe: { [unowned self] url, language in
                transcribedFiles.append((url, (try? Data(contentsOf: url)) ?? Data()))
                transcribedLanguages.append(language)
                if transcribeDelay != .zero { try await Task.sleep(for: transcribeDelay) }
                return try transcript.get()
            },
            enhance: { [unowned self] text in
                enhancedInputs.append(text)
                return try enhanced.get()
            },
            finalize: { text in
                LocalAPIFinalText(
                    text: text.replacingOccurrences(of: "开会", with: "开会议"),
                    dictionaryHitTerms: text.contains("开会") ? ["开会"] : [],
                    dictionaryCorrectedTerms: text.contains("开会") ? ["开会议"] : []
                )
            },
            record: { [unowned self] in recorded.append($0) }
        )
    }

    private func json(_ response: LocalAPIHTTPResponse) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    }

    private let request = LocalAPIDictationRequest(audio: Data([9, 8, 7]), fileExtension: "m4a", enhance: true)

    func testTranscriptionGetsTheRequestedLanguage() async {
        let pipeline = makePipeline()
        var chinese = request
        chinese.language = .request("Chinese")
        _ = await pipeline.run(request)
        _ = await pipeline.run(chinese)
        XCTAssertEqual(transcribedLanguages, [.userSettings, .request("Chinese")])
    }

    func testSuccessfulDictationTranscribesEnhancesFinalizesAndRecords() async throws {
        let pipeline = makePipeline()

        let response = await pipeline.run(request)

        XCTAssertEqual(response.status, 200)
        let body = try json(response)
        XCTAssertEqual(body["text"] as? String, "明天下午三点开会议。")
        XCTAssertEqual(body["rawText"] as? String, "嗯 那个 明天下午三点开会")
        XCTAssertNotNil(body["id"] as? String)
        XCTAssertNotNil(body["asrMs"] as? Int)
        XCTAssertNotNil(body["enhanceMs"] as? Int)
        XCTAssertEqual(body["enhanced"] as? Bool, true)

        XCTAssertEqual(transcribedFiles.count, 1)
        XCTAssertEqual(transcribedFiles[0].0.pathExtension, "m4a")
        XCTAssertEqual(transcribedFiles[0].1, Data([9, 8, 7]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: transcribedFiles[0].0.path), "audio file should be removed")
        XCTAssertEqual(enhancedInputs, ["嗯 那个 明天下午三点开会"])
        XCTAssertEqual(recorded.count, 1)
        XCTAssertEqual(recorded.first?.text, "明天下午三点开会议。")
        XCTAssertEqual(recorded.first?.usedEnhancement, true)
        XCTAssertEqual(recorded.first?.dictionaryHitTerms, ["开会"])
        XCTAssertEqual(recorded.first?.dictionaryCorrectedTerms, ["开会议"])
    }

    func testSkippingEnhancementFinalizesRawTranscript() async throws {
        let pipeline = makePipeline()
        var noEnhance = request
        noEnhance.enhance = false

        let response = await pipeline.run(noEnhance)

        XCTAssertEqual(try json(response)["text"] as? String, "嗯 那个 明天下午三点开会议")
        XCTAssertEqual(try json(response)["enhanced"] as? Bool, false)
        XCTAssertTrue(enhancedInputs.isEmpty)
        XCTAssertEqual(recorded.first?.usedEnhancement, false)
    }

    func testEnhancementFailureFallsBackToRawTranscript() async throws {
        struct Boom: Error {}
        let pipeline = makePipeline(enhanced: .failure(Boom()))

        let response = await pipeline.run(request)

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(try json(response)["text"] as? String, "嗯 那个 明天下午三点开会议")
        XCTAssertEqual(try json(response)["enhanced"] as? Bool, false)
        XCTAssertEqual(recorded.first?.usedEnhancement, false)
    }

    func testBusyMacRejectsWithoutTranscribing() async throws {
        let pipeline = makePipeline(busyReason: "Voxt is recording on the Mac.")

        let response = await pipeline.run(request)

        XCTAssertEqual(response.status, 503)
        XCTAssertEqual(try json(response)["error"] as? String, "Voxt is recording on the Mac.")
        XCTAssertTrue(transcribedFiles.isEmpty)
    }

    func testTranscriptionFailureReturns500AndCleansUp() async throws {
        struct ModelMissing: LocalizedError { var errorDescription: String? { "ASR model is not installed." } }
        let pipeline = makePipeline(transcript: .failure(ModelMissing()))

        let response = await pipeline.run(request)

        XCTAssertEqual(response.status, 500)
        XCTAssertEqual(try json(response)["error"] as? String, "ASR model is not installed.")
        XCTAssertFalse(FileManager.default.fileExists(atPath: transcribedFiles[0].0.path))
        XCTAssertTrue(recorded.isEmpty)
    }

    func testSilenceReturns422AndRecordsNothing() async throws {
        let pipeline = makePipeline(transcript: .success("  \n "))

        let response = await pipeline.run(request)

        XCTAssertEqual(response.status, 422)
        XCTAssertTrue(enhancedInputs.isEmpty)
        XCTAssertTrue(recorded.isEmpty)
    }

    func testSecondRequestWhileRunningIsRejected() async throws {
        let pipeline = makePipeline(transcribeDelay: .milliseconds(200))

        async let first = pipeline.run(request)
        try await Task.sleep(for: .milliseconds(50))
        let second = await pipeline.run(request)
        let firstResponse = await first

        XCTAssertEqual(second.status, 503)
        XCTAssertEqual(firstResponse.status, 200)
    }

    func testUnreadableAudioIsRejectedAs400WithoutTranscribing() async throws {
        let pipeline = makePipeline(checksAudioWithAVFoundation: true)
        let corrupt = LocalAPIDictationRequest(audio: Data("not audio".utf8), fileExtension: "wav", enhance: true)

        let response = await pipeline.run(corrupt)

        XCTAssertEqual(response.status, 400)
        XCTAssertTrue(transcribedFiles.isEmpty)
        XCTAssertTrue(recorded.isEmpty)
    }

    func testReadableWAVPassesTheAudioCheck() async throws {
        let pipeline = makePipeline(checksAudioWithAVFoundation: true)
        let silence = LocalAPIDictationRequest(audio: Self.pcm16WAV(sampleCount: 1_600), fileExtension: "wav", enhance: false)

        let response = await pipeline.run(silence)

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(transcribedFiles.count, 1)
    }

    /// A 16 kHz mono 16-bit WAV of silence.
    private static func pcm16WAV(sampleCount: Int) -> Data {
        func littleEndian<T: FixedWidthInteger>(_ value: T) -> Data { withUnsafeBytes(of: value.littleEndian) { Data($0) } }
        let dataBytes = UInt32(sampleCount * 2)
        var wav = Data("RIFF".utf8) + littleEndian(UInt32(36) + dataBytes) + Data("WAVEfmt ".utf8)
        wav += littleEndian(UInt32(16)) + littleEndian(UInt16(1)) + littleEndian(UInt16(1))
        wav += littleEndian(UInt32(16_000)) + littleEndian(UInt32(32_000)) + littleEndian(UInt16(2)) + littleEndian(UInt16(16))
        wav += Data("data".utf8) + littleEndian(dataBytes) + Data(count: Int(dataBytes))
        return wav
    }
}
