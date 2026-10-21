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
        enhanced: Result<String?, Error> = .success("明天下午三点开会。"),
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

    func testNoTextModelMeansNotEnhanced() async throws {
        // A Mac whose text model is off returned the transcript as "enhanced" in 3 ms.
        let pipeline = makePipeline(enhanced: .success(nil))

        let response = await pipeline.run(request)

        XCTAssertEqual(response.status, 200)
        let body = try json(response)
        XCTAssertEqual(body["enhanced"] as? Bool, false)
        XCTAssertEqual(body["text"] as? String, "嗯 那个 明天下午三点开会议")
        XCTAssertEqual(recorded.first?.usedEnhancement, false)
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

    // MARK: Compose

    private var composeOptions: LocalAPIComposeOptions {
        LocalAPIComposeOptions(
            mode: .translate,
            targetLanguage: "en",
            selectedText: "",
            contextBefore: "",
            contextAfter: "",
            style: .automatic,
            vocabulary: []
        )
    }

    private func makeComposeRequest(_ options: LocalAPIComposeOptions? = nil) -> LocalAPIComposeRequest {
        LocalAPIComposeRequest(
            audio: Data([9, 8, 7]),
            fileExtension: "m4a",
            options: options ?? composeOptions
        )
    }

    func testComposeTranslateRunsModelAndReportsMode() async throws {
        var composed: [(String, LocalAPIComposeOptions)] = []
        let pipeline = makePipeline()
        pipeline.composeHandler = { text, options in
            composed.append((text, options))
            return "Tomorrow at three, meeting about the release."
        }

        let response = await pipeline.runCompose(makeComposeRequest())

        XCTAssertEqual(response.status, 200)
        let body = try json(response)
        XCTAssertEqual(body["text"] as? String, "Tomorrow at three, meeting about the release.")
        XCTAssertEqual(body["rawText"] as? String, "嗯 那个 明天下午三点开会")
        XCTAssertEqual(body["mode"] as? String, "translate")
        XCTAssertEqual(composed.count, 1)
        XCTAssertEqual(composed.first?.1.mode, .translate)
        XCTAssertEqual(composed.first?.0, "嗯 那个 明天下午三点开会")
        XCTAssertEqual(recorded.count, 1)
        XCTAssertEqual(recorded.first?.mode, .translate)
        XCTAssertTrue(enhancedInputs.isEmpty, "compose must not run the plain enhancement path")
    }

    func testRejectedEditReturnsNotAppliedWithoutFinalizingOrRecordingSelection() async throws {
        let selection = "The server has a cash limit."
        let pipeline = makePipeline()
        pipeline.composeHandler = { _, options in
            try await LocalAPIComposePrompt.compose(rawTranscription: "Fix cash to cash.", options: options) { _ in
                "The server has a cash to cash limit."
            }
        }
        var options = composeOptions
        options.mode = .edit
        options.selectedText = selection
        let response = await pipeline.runCompose(makeComposeRequest(options))
        XCTAssertEqual(response.status, 200)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        XCTAssertEqual(json["status"] as? String, "not_applied")
        XCTAssertEqual(json["message"] as? String, "Couldn't apply that edit")
        XCTAssertEqual(json["text"] as? String, selection)
        XCTAssertEqual(json["mode"] as? String, "edit")
        XCTAssertTrue(recorded.isEmpty)
    }

    func testComposeModelFailureReturns503WithoutFakeSuccess() async throws {
        struct Unavailable: LocalizedError { var errorDescription: String? { "No text model is selected." } }
        let pipeline = makePipeline()
        pipeline.composeHandler = { _, _ in throw Unavailable() }

        let response = await pipeline.runCompose(makeComposeRequest())

        XCTAssertEqual(response.status, 503)
        XCTAssertEqual(try json(response)["error"] as? String, "No text model is selected.")
        XCTAssertTrue(recorded.isEmpty, "a failed compose must not be recorded as a success")
    }

    func testComposeWithoutModelHandlerFailsClosedForNonDictateModes() async throws {
        let pipeline = makePipeline()

        let response = await pipeline.runCompose(makeComposeRequest())

        XCTAssertEqual(response.status, 503)
        XCTAssertTrue(recorded.isEmpty)
    }

    func testComposeEmptyModelOutputFailsClosed() async throws {
        let pipeline = makePipeline()
        pipeline.composeHandler = { _, _ in "  \n " }

        let response = await pipeline.runCompose(makeComposeRequest())

        XCTAssertEqual(response.status, 503)
        XCTAssertTrue(recorded.isEmpty)
    }

    func testDictateDiscardsCleanupThatChangesTranscriptScript() async throws {
        // Qwen3.5-4B translated these Chinese ASR results.
        let examples = [
            ("第一，保存报告；第二，发送邮件；第三，关闭窗口。", "First, save the report. Second, send the email. Third, close the window."),
            ("第一，检查门锁。第二，关掉灯。", "First, check the door lock. Second, turn off the lights."),
            ("Check the door lock.", "检查门锁。")
        ]
        for (raw, cleaned) in examples {
            let pipeline = makePipeline(transcript: .success(raw))
            pipeline.composeHandler = { _, _ in cleaned }
            let response = await pipeline.runCompose(makeComposeRequest(.init()))
            XCTAssertEqual(response.status, 200)
            XCTAssertEqual(try json(response)["text"] as? String, raw)
            XCTAssertEqual(recorded.last?.text, raw)
        }
    }

    func testScriptGuardPreservesSameScriptMixedTextAndExplicitTranslation() async throws {
        for (raw, cleaned) in [("今天用 Voxt 写报告。", "今天用 Voxt 写报告。"), ("um call Alex", "Call Alex."), ("下午三点。", "下午三点！")] {
            let pipeline = makePipeline(transcript: .success(raw))
            pipeline.composeHandler = { _, _ in cleaned }
            let response = await pipeline.runCompose(makeComposeRequest(.init()))
            XCTAssertEqual(try json(response)["text"] as? String, cleaned)
        }
        let pipeline = makePipeline(transcript: .success("检查门锁。"))
        pipeline.composeHandler = { _, _ in "Check the door lock." }
        let response = await pipeline.runCompose(makeComposeRequest(composeOptions))
        XCTAssertEqual(try json(response)["text"] as? String, "Check the door lock.")
    }

    func testComposeDictateAlwaysUsesTheModelPath() async throws {
        var composed: [(String, LocalAPIComposeOptions)] = []
        let pipeline = makePipeline(transcript: .success("um clean up this text"))
        pipeline.composeHandler = { text, options in
            composed.append((text, options))
            return "Cleaned up text."
        }

        var dictateOptions = composeOptions
        dictateOptions.mode = .dictate
        dictateOptions.targetLanguage = nil
        let response = await pipeline.runCompose(makeComposeRequest(dictateOptions))

        XCTAssertEqual(response.status, 200)
        let body = try json(response)
        XCTAssertEqual(body["text"] as? String, "Cleaned up text.")
        XCTAssertEqual(body["mode"] as? String, "dictate")
        XCTAssertEqual(composed.count, 1, "even a plain compose dictate runs the model path")
        XCTAssertTrue(enhancedInputs.isEmpty, "compose never takes the legacy enhancement path")
    }

    func testComposeDictateWithOptionsUsesTheModelPath() async throws {
        var composed: [(String, LocalAPIComposeOptions)] = []
        let pipeline = makePipeline()
        pipeline.composeHandler = { text, options in
            composed.append((text, options))
            return "Cleaned up text."
        }

        var dictateOptions = composeOptions
        dictateOptions.mode = .dictate
        dictateOptions.targetLanguage = nil
        dictateOptions.style = .formal
        dictateOptions.vocabulary = ["SGLang"]
        let response = await pipeline.runCompose(makeComposeRequest(dictateOptions))

        XCTAssertEqual(response.status, 200)
        XCTAssertEqual(try json(response)["mode"] as? String, "dictate")
        XCTAssertEqual(composed.count, 1, "non-plain dictate must run the compose path")
        XCTAssertEqual(composed.first?.1.vocabulary, ["SGLang"])
        XCTAssertTrue(enhancedInputs.isEmpty, "the plain enhancement prompt must not run for option-carrying dictate")
    }

    func testComposeDictateFailsClosedWhenTheModelThrows() async throws {
        struct Boom: Error {}
        let pipeline = makePipeline(enhanced: .failure(Boom()))

        var dictateOptions = composeOptions
        dictateOptions.mode = .dictate
        dictateOptions.targetLanguage = nil
        let response = await pipeline.runCompose(makeComposeRequest(dictateOptions))

        XCTAssertEqual(response.status, 503, "compose dictate must not fall back to the raw transcript")
        XCTAssertTrue(recorded.isEmpty)
    }

    func testComposeSharesTheBusyGuardWithDictations() async throws {
        let pipeline = makePipeline(busyReason: "Voxt is recording on the Mac.")

        let response = await pipeline.runCompose(makeComposeRequest())

        XCTAssertEqual(response.status, 503)
        XCTAssertTrue(transcribedFiles.isEmpty)
    }

    // MARK: Corrections

    private func correctionRequest(original: String, edited: String) -> LocalAPICorrectionRequest {
        LocalAPICorrectionRequest(original: original, edited: edited)
    }

    func testCorrectionReturnsTheValidatedPairAndWritesNoHistory() async throws {
        var prompts: [String] = []
        let pipeline = makePipeline()
        pipeline.correctHandler = { _, instruction in
            prompts.append(instruction)
            return #"{"correction":{"original":"星浪","replacement":"星朗"}}"#
        }

        let response = await pipeline.runCorrection(correctionRequest(original: "请用星浪处理。", edited: "请用星朗处理。"))

        XCTAssertEqual(response.status, 200)
        let body = try json(response)
        let correction = try XCTUnwrap(body["correction"] as? [String: String])
        XCTAssertEqual(correction["original"], "星浪")
        XCTAssertEqual(correction["replacement"], "星朗")
        XCTAssertEqual(prompts.count, 1)
        XCTAssertTrue(prompts[0].contains("请用星浪处理。"))
        XCTAssertTrue(prompts[0].contains("星朗"))
        XCTAssertTrue(recorded.isEmpty)
        XCTAssertTrue(transcribedFiles.isEmpty)
        XCTAssertTrue(enhancedInputs.isEmpty)
    }

    func testIgnorableCorrectionSkipsTheModelAndReturnsNull() async throws {
        let pipeline = makePipeline()
        pipeline.correctHandler = { _, _ in
            XCTFail("an append, deletion, number, punctuation, or rewrite must not call the model")
            return #"{"correction":{"original":"Voxt","replacement":"Voice"}}"#
        }
        let samples = [
            ("Use Voxt.", "Use Voxt. Thanks!"),
            ("Use Voxt today.", "Use today."),
            ("Use Voxt.", "Use Voxt!"),
            ("Pay $50.", "Pay $60."),
            ("Use 50 today.", "Use 60 today."),
            ("See Alice about the invoice", "See Bob about the bill"),
        ]
        for sample in samples {
            let response = await pipeline.runCorrection(correctionRequest(original: sample.0, edited: sample.1))
            XCTAssertEqual(response.status, 200, sample.0)
            XCTAssertTrue(try json(response)["correction"] is NSNull, sample.0)
        }
        XCTAssertTrue(recorded.isEmpty)
    }

    func testLetterInsertAndDeleteInsideAWordCallTheModel() async throws {
        let pipeline = makePipeline()
        pipeline.correctHandler = { request, _ in
            if request.original == "Use oxt." {
                return #"{"correction":{"original":"oxt","replacement":"Voxt"}}"#
            }
            return #"{"correction":{"original":"Vooxt","replacement":"Voxt"}}"#
        }
        let inserted = await pipeline.runCorrection(correctionRequest(original: "Use oxt.", edited: "Use Voxt."))
        let deleted = await pipeline.runCorrection(correctionRequest(original: "Use Vooxt.", edited: "Use Voxt."))
        XCTAssertEqual(inserted.status, 200)
        XCTAssertEqual(deleted.status, 200)
        let insertedPair = try XCTUnwrap(try json(inserted)["correction"] as? [String: String])
        let deletedPair = try XCTUnwrap(try json(deleted)["correction"] as? [String: String])
        XCTAssertEqual(insertedPair["original"], "oxt")
        XCTAssertEqual(insertedPair["replacement"], "Voxt")
        XCTAssertEqual(deletedPair["original"], "Vooxt")
        XCTAssertEqual(deletedPair["replacement"], "Voxt")
        XCTAssertTrue(recorded.isEmpty)
        XCTAssertTrue(transcribedFiles.isEmpty)
    }

    func testModelNullAndInvalidOutput() async throws {
        let pipeline = makePipeline()
        pipeline.correctHandler = { _, _ in #"{"correction":null}"# }
        let absent = await pipeline.runCorrection(correctionRequest(original: "Use Voice today.", edited: "Use Voxt today."))
        XCTAssertEqual(absent.status, 200)
        XCTAssertTrue(try json(absent)["correction"] is NSNull)

        pipeline.correctHandler = { _, _ in "not json" }
        let invalid = await pipeline.runCorrection(correctionRequest(original: "Use Voice today.", edited: "Use Voxt today."))
        XCTAssertEqual(invalid.status, 503)
        XCTAssertFalse((try json(invalid)["error"] as? String ?? "").isEmpty)

        pipeline.correctHandler = { _, _ in #"{"correction":{"original":"浪","replacement":"朗"}}"# }
        let fragment = await pipeline.runCorrection(correctionRequest(original: "请用星浪处理。", edited: "请用星朗处理。"))
        XCTAssertEqual(fragment.status, 503)
        XCTAssertTrue(recorded.isEmpty)
        XCTAssertTrue(transcribedFiles.isEmpty)
    }

    func testCorrectionSharesTheBusyGuard() async throws {
        let busy = makePipeline(busyReason: "Voxt is recording on the Mac.")
        busy.correctHandler = { _, _ in #"{"correction":null}"# }
        let blocked = await busy.runCorrection(correctionRequest(original: "Use Voice today.", edited: "Use Voxt today."))
        XCTAssertEqual(blocked.status, 503)
        XCTAssertEqual(try json(blocked)["error"] as? String, "Voxt is recording on the Mac.")

        let pipeline = makePipeline()
        pipeline.correctHandler = { _, _ in
            try await Task.sleep(for: .milliseconds(200))
            return #"{"correction":{"original":"星浪","replacement":"星朗"}}"#
        }
        async let first = pipeline.runCorrection(correctionRequest(original: "请用星浪处理。", edited: "请用星朗处理。"))
        try await Task.sleep(for: .milliseconds(40))
        let second = await pipeline.runCompose(makeComposeRequest())
        let firstResponse = await first
        XCTAssertEqual(second.status, 503)
        XCTAssertEqual(firstResponse.status, 200)
        XCTAssertTrue(recorded.isEmpty)
    }
}
