// SPDX-License-Identifier: Apache-2.0
import Testing
import AVFoundation
@testable import OmniTyper

@Suite(.serialized)
struct WorkerClientTests {
    @Test func testAudioResamplingAndRecordingLimit() throws {
        final class Packets: @unchecked Sendable {
            let lock = NSLock()
            var bytes = 0
            var first = Data()
            func receive(_ data: Data) {
                lock.withLock {
                    if first.isEmpty { first = data }
                    bytes += data.count
                }
            }
        }
        let packets = Packets()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let inputFormat = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                                    sampleRate: 48_000, channels: 2, interleaved: false))
        let input = try #require(AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: 48_000))
        input.frameLength = 48_000
        let channels = try #require(input.floatChannelData)
        for frame in 0..<48_000 {
            channels[0][frame] = 0.25
            channels[1][frame] = 0.25
        }
        let sink = try AudioCaptureSink(input: inputFormat, url: url, onPCM: { packets.receive($0) })
        for _ in 0..<302 { sink.consume(input) }
        try sink.close()
        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.sampleRate == 16_000)
        #expect(file.fileFormat.channelCount == 1)
        #expect(file.fileFormat.commonFormat == .pcmFormatInt16)
        #expect(file.length == 300 * 16_000, "The WAV must never exceed the worker's 300-second limit")
        #expect(packets.bytes == Int(file.length) * 2, "Streaming and WAV must contain the same number of samples")
        let pcmSample = Int(packets.first[1024]) | Int(packets.first[1025]) << 8
        #expect(abs(pcmSample - 8192) < 328, "Streaming packets must be little-endian PCM16 at the resampled amplitude")
        let decoded = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1_024))
        try file.read(into: decoded)
        let sample = try #require(decoded.floatChannelData)[0][512]
        #expect(abs(sample - 0.25) < 0.01)
    }

    @Test func audioLevelTracksVolumeAndResetsWhenCaptureEnds() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                               sampleRate: 16_000, channels: 1, interleaved: false))
        let input = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1_600))
        input.frameLength = 1_600
        let samples = try #require(input.floatChannelData)[0]
        let sink = try AudioCaptureSink(input: format, url: url)
        for (amplitude, expected) in [(Float(0), 0.0), (0.01, 0.25), (-0.1, 0.75), (1, 1), (0, 0)] {
            for index in 0..<1_600 { samples[index] = amplitude }
            sink.consume(input)
            #expect(abs(sink.level() - expected) < 0.01)
        }
        for index in 0..<1_600 { samples[index] = 1 }
        sink.consume(input)
        #expect(sink.level() == 1)
        try sink.close()
        #expect(sink.level() == 0)
    }

    @Test @MainActor
    func testFramingLifecycleAndCancellation() async throws {
        try await withWorker { _, python in
            let client = WorkerClient()
            defer { client.stop() }
            do {
                _ = try await client.request(["op": "echo", "text": String(repeating: "汉", count: 90_000)], python: python)
                Issue.record("Requests over the worker's 256 KiB limit must fail before sending")
            } catch { #expect(error.localizedDescription.contains("256 KiB")) }
            #expect(!client.isRunning)

            let first = try await client.request(["op": "echo"], python: python)
            #expect(first["text"] as? String == "你好 café")
            #expect(client.isRunning)
            let second = try await client.request(["op": "echo"], python: python)
            #expect(first["worker_pid"] as? Int == second["worker_pid"] as? Int)
            #expect(second["serial"] as? Int == 2)
            #expect(client.status == "Ready")

            do {
                _ = try await client.request(["op": "failure"], python: python)
                Issue.record("Worker error responses must fail the request")
            } catch { #expect(error.localizedDescription == "Model unavailable") }
            #expect(client.isRunning, "An ordinary model failure keeps the protocol usable")
            _ = try await client.request(["op": "echo"], python: python)

            for operation in ["invalid", "oversized", "crash"] {
                do {
                    _ = try await client.request(["op": operation], python: python)
                    Issue.record("\(operation) must fail the request")
                } catch { #expect(!(error is CancellationError)) }
                #expect(!client.isRunning)
                _ = try await client.request(["op": "echo"], python: python)
            }

            let waiting = Task { try await client.request(["op": "sleep"], python: python) }
            for _ in 0..<100 {
                if client.status == "waiting" { break }
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            #expect(client.status == "waiting")
            do {
                _ = try await client.request(["op": "echo"], python: python)
                Issue.record("Concurrent requests must be rejected")
            } catch { #expect(error.localizedDescription.contains("busy")) }
            waiting.cancel()
            do {
                _ = try await waiting.value
                Issue.record("Cancellation must resume the outstanding request")
            } catch { #expect(error is CancellationError) }
            #expect(!client.isRunning)
            let restarted = try await client.request(["op": "echo"], python: python)
            #expect(restarted["serial"] as? Int == 1)

            let final = try await client.request(["op": "final_exit"], python: python)
            #expect(final["text"] as? String == "你好 café", "A final response must survive an immediate worker exit")
            for _ in 0..<100 {
                if !client.isRunning { break }
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            #expect(!client.isRunning)

            let log = try #require(Diagnostics.fileURL)
            #expect(log.path.hasPrefix(FileManager.default.temporaryDirectory.path),
                    "a test run must not append to the real log")
            let diagnostics = try String(contentsOf: log, encoding: .utf8)
            #expect(!diagnostics.contains("PRIVATE_TRANSCRIPT_DO_NOT_LOG"))
            #expect(diagnostics.utf8.count <= 128 * 1024)
            #expect(diagnostics.contains("worker.stderr"), "the worker's own events belong in the same log")
        }
    }

    @Test @MainActor
    func modelRetentionReusesPreparationAndReleasesOnDemand() async throws {
        try await withWorker { directory, python in
            let store = AppStore(directory: directory.appendingPathComponent("retention-library"))
            store.preferences.pythonExecutable = python
            let model = AppModel(store: store)
            defer { model.shutdown() }
            #expect(!model.isPreloading && !model.worker.isRunning)
            model.setKeepModelLoaded(true)
            #expect(model.isPreloading && model.phase == .idle)
            model.cancel()
            #expect(model.isPreloading, "Dismissing the popup preserves an opted-in preload")
            _ = try await model.preloadTask?.value
            let warm = try await model.worker.request(["op": "echo"], python: python)
            for phase in [AppModel.Phase.starting, .recording] {
                model.phase = phase
                model.cancel()
                #expect(!model.isPreloading, "Cancelling with a ready worker must not reload the model")
                if let preload = model.preloadTask { _ = try await preload.value }
                let stillWarm = try await model.worker.request(["op": "echo"], python: python)
                #expect(warm["worker_pid"] as? Int == stillWarm["worker_pid"] as? Int)
            }
            model.setKeepModelLoaded(false)
            #expect(!model.worker.isRunning)
            model.setKeepModelLoaded(true)
            let cancelledPreload = try #require(model.preloadTask)
            model.setKeepModelLoaded(false)
            model.setKeepModelLoaded(true)
            let replacementPreload = try #require(model.preloadTask)
            do {
                _ = try await cancelledPreload.value
                Issue.record("Disabling retention must cancel its pending preload")
            } catch { #expect(error is CancellationError) }
            #expect(model.isPreloading, "A cancelled preload must not clear its replacement")
            _ = try await replacementPreload.value
            #expect(model.worker.isRunning && model.error.isEmpty)
            model.setKeepModelLoaded(false)
            await Task.yield()
            #expect(!model.isPreloading && !model.worker.isRunning)
            store.preferences.style = "verbatim"
            store.preferences.keepAudio = true
            let recording = directory.appendingPathComponent("retained.wav")
            try Data([1, 2, 3]).write(to: recording)
            store.add(HistoryEntry(mode: .dictate, appName: "Test editor", rawText: "Earlier text",
                                   text: "Earlier text", duration: 1), recording: recording)
            model.prepareModels()
            model.retry(try #require(store.history.first))
            await model.task?.value
            #expect(model.error.isEmpty && model.resultText == "Original transcript")
            #expect(!model.worker.isRunning, "Retention off releases the model after a retry waits for preload")
            model.setKeepModelLoaded(true)
            _ = try await model.preloadTask?.value
            model.loadTextModels()
            let processing = try #require(model.task)
            for _ in 0..<100 where model.worker.status != "waiting" { try await Task.sleep(nanoseconds: 10_000_000) }
            #expect(model.worker.status == "waiting")
            model.cancel()
            await processing.value
            _ = try await model.preloadTask?.value
            #expect(model.worker.isRunning)
            model.shutdown()
            #expect(!model.worker.isRunning)
            let relaunched = AppModel(store: AppStore(directory: store.directory))
            #expect(relaunched.isPreloading)
            relaunched.shutdown()
            await Task.yield()
            #expect(!relaunched.worker.isRunning)
        }
    }

    @Test(arguments: ["unavailable-model", "incomplete-response"]) @MainActor
    func textProcessingFailurePreservesAudioAndAllowsVerbatimRecovery(textModel: String) async throws {
        try await withWorker { directory, python in
            let store = AppStore(directory: directory.appendingPathComponent("library"))
            store.preferences.pythonExecutable = python
            store.preferences.textSettings.model = textModel
            store.preferences.keepAudio = true
            let audio = directory.appendingPathComponent("recording.wav")
            try Data([1, 2, 3]).write(to: audio)
            store.add(HistoryEntry(mode: .translate, appName: "Test editor", rawText: "Earlier transcript",
                                   text: "Earlier translation", duration: 1), recording: audio)
            let model = AppModel(store: store, captureTarget: { throw Failure("sys.focusField") })
            defer { model.shutdown() }
            model.retry(try #require(store.history.first))
            await model.task?.value
            try #require(model.canRetry)
            #expect(model.error == (textModel == "unavailable-model" ? "Text API unavailable" : L("worker.incomplete")))
            #expect(FileManager.default.fileExists(atPath: try #require(model.retryRecording).url.path))
            #expect(model.rawText == "Original transcript")
            #expect(store.history.count == 1)
            model.useVerbatimDictation()
            await model.task?.value
            #expect(model.phase == .idle)
            #expect(model.error.isEmpty)
            #expect(!model.canRetry)
            #expect(model.mode == .dictate)
            #expect(model.resultText == "Original transcript")
            #expect(store.history.first?.mode == .dictate)
            #expect(store.history.count == 2)
        }
    }

    @Test(arguments: RecordingPresentation.allCases) @MainActor
    func processingRespectsInsertionPolicy(presentation: RecordingPresentation) async throws {
        try await withWorker { directory, python in
            let store = AppStore(directory: directory.appendingPathComponent("library"))
            store.preferences.pythonExecutable = python
            store.preferences.recordingPresentation = presentation
            var inserted: [String] = []
            var restoredFocus: [Bool] = []
            let model = AppModel(store: store, insertText: { text, _, restore in
                inserted.append(text)
                restoredFocus.append(restore)
            })
            defer { model.shutdown() }
            model.target = AppModelTests.target()
            model.sessionPreferences = store.preferences
            let audio = directory.appendingPathComponent("recording.wav")
            try Data([1, 2, 3]).write(to: audio)
            model.run(audio: audio, duration: 1, allowInsertion: true)
            await model.task?.value
            #expect(model.error.isEmpty && model.resultText == "Original transcript")
            if presentation == .edit {
                #expect(inserted.isEmpty && model.resultDraft == "Original transcript")
                model.resultDraft = "Reviewed and corrected"
                model.insertReviewedResult()
                await model.task?.value
                #expect(inserted == ["Reviewed and corrected"] && restoredFocus == [true])
                #expect(store.history.first?.text == "Reviewed and corrected")
                #expect(store.history.first?.rawText == "Original transcript")
            } else {
                #expect(inserted == ["Original transcript"] && restoredFocus == [false])
            }
        }
    }

    @Test @MainActor
    func draftProcessingAppliesSelectionAndPreservesConcurrentEdits() async throws {
        try await withWorker { directory, python in
            let store = AppStore(directory: directory.appendingPathComponent("library"))
            store.preferences.pythonExecutable = python
            store.preferences.textSettings.model = "test-model"
            let model = AppModel(store: store)
            defer { model.shutdown() }
            model.sessionPreferences = store.preferences
            model.isReviewingResult = true
            let audio = directory.appendingPathComponent("recording.wav")
            for (mode, length, expected) in [
                (VoiceMode.dictate, 0, "Before Original transcript🌏 after"),
                (.translate, 2, "Before Translated text after"),
                (.edit, 2, "Before Revised text after")
            ] {
                model.mode = mode
                model.resultDraft = "Before 🌏 after"
                model.draftOperation = DraftSelection(text: model.resultDraft, range: NSRange(location: 7, length: length))
                try Data([1, 2, 3]).write(to: audio)
                model.run(audio: audio, duration: 1, allowInsertion: false)
                await model.task?.value
                #expect(model.error.isEmpty && model.resultDraft == expected)
                #expect(store.history.first?.text == expected)
            }
            model.draftOperation = DraftSelection(text: model.resultDraft, range: NSRange(location: 7, length: 12))
            model.resultDraft = "Manually changed while processing"
            try Data([1, 2, 3]).write(to: audio)
            model.run(audio: audio, duration: 1, allowInsertion: false)
            await model.task?.value
            #expect(model.resultDraft == "Manually changed while processing")
            #expect(model.unappliedResult == "Revised text" && !model.error.isEmpty)
            #expect(store.history.count == 3)
        }
    }

    @Test @MainActor
    func askKeepsDocumentAndHistorySeparateUntilExplicitInsertion() async throws {
        try await withWorker { directory, python in
            let store = AppStore(directory: directory.appendingPathComponent("library"))
            store.preferences.pythonExecutable = python
            store.preferences.textSettings.model = "test-model"
            var inserted: [String] = []
            let model = AppModel(store: store, insertText: { text, _, _ in inserted.append(text) })
            defer { model.shutdown() }
            model.sessionPreferences = store.preferences
            let audio = directory.appendingPathComponent("recording.wav")
            try Data([1, 2, 3]).write(to: audio)
            model.run(audio: audio, duration: 1, allowInsertion: false)
            await model.task?.value
            model.mode = .ask
            model.target = AppModelTests.target()
            model.resultDraft = "My document"
            model.draftSelection = NSRange(location: 3, length: 8)
            try Data([1, 2, 3]).write(to: audio)
            model.run(audio: audio, duration: 1, allowInsertion: true)
            await model.task?.value
            #expect(model.isReviewingResult && inserted.isEmpty)
            #expect(model.questionText == "Spoken request" && model.answerText == "An answer")
            #expect(model.resultDraft == "My document")
            model.saveReviewEdits()
            #expect(store.history.first(where: { $0.mode == .dictate })?.text == "Original transcript")
            model.start()
            #expect(model.phase == .idle && !model.canResumeVoice)
            model.newQuestion()
            #expect(model.questionText.isEmpty && model.answerText.isEmpty && model.canResumeVoice)
            #expect(model.resultDraft == "My document" && model.draftSelection == NSRange(location: 3, length: 8))
            model.draftOperation = DraftSelection(text: model.resultDraft, range: model.draftSelection)
            try Data([1, 2, 3]).write(to: audio)
            model.run(audio: audio, duration: 1, allowInsertion: true)
            await model.task?.value
            model.insertReviewedResult()
            await model.task?.value
            #expect(inserted == ["An answer"] && !model.canInsertReview)
            model.selectMode(.edit)
            #expect(model.reviewContent == "My document")
        }
    }

    @MainActor
    private func withWorker(_ body: (URL, String) async throws -> Void) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = directory.appendingPathComponent("worker.py")
        let oldWorker = ProcessInfo.processInfo.environment["OMNITYPER_WORKER"]
        let python = ProcessInfo.processInfo.environment["OMNITYPER_TEST_PYTHON"] ?? "/usr/bin/python3"
        guard FileManager.default.isExecutableFile(atPath: python) else {
            Issue.record("Set OMNITYPER_TEST_PYTHON to a Python 3 executable.")
            return
        }
        try #"""
        import json, os, sys, time
        assert sys.dont_write_bytecode, "The worker must not modify the signed app bundle"
        serial = 0
        for line in sys.stdin:
            request = json.loads(line)
            serial += 1
            op = request['op']
            if op == 'transcribe':
                if request['mode'] in ('ask', 'edit'):
                    response = dict(ok=True, text='An answer' if request['mode'] == 'ask' else 'Revised text', raw_text='Spoken request')
                elif request.get('text_model') == 'incomplete-response':
                    response = dict(ok=True, raw_text='Original transcript')
                elif request['mode'] == 'translate' and request.get('text_model') == 'unavailable-model':
                    response = dict(ok=False, error='Text API unavailable', raw_text='Original transcript')
                elif request['mode'] == 'translate':
                    response = dict(ok=True, text='Translated text', raw_text='Spoken request')
                else:
                    assert request['mode'] == 'dictate'
                    assert request['style'] == 'verbatim'
                    assert 'text_model' not in request
                    response = dict(ok=True, text='Original transcript', raw_text='Original transcript')
                print(json.dumps(dict(id=request['id'], **response)), flush=True)
                continue
            if op == 'crash':
                os._exit(17)
            if op == 'invalid':
                print('{bad json', flush=True)
                continue
            if op == 'oversized':
                os.write(1, b'x' * (2 * 1024 * 1024))
                continue
            if op in ('sleep', 'models'):
                print(json.dumps(dict(id=request['id'], event='progress', message='waiting')), flush=True)
                time.sleep(30)
            if op == 'failure':
                print(json.dumps(dict(id=request['id'], ok=False, error='Model unavailable')), flush=True)
                continue
            os.write(2, b'PRIVATE_TRANSCRIPT_DO_NOT_LOG\n')
            progress = json.dumps(dict(id=request['id'], event='progress', message='loading')).encode() + b'\n'
            final = json.dumps(dict(id=request['id'], ok=True, text='你好 café',
                                    serial=serial, worker_pid=os.getpid()), ensure_ascii=False).encode() + b'\n'
            cut = final.index('你'.encode()) + 1
            os.write(1, progress + final[:cut])
            time.sleep(0.02)
            os.write(1, final[cut:])
            if op == 'final_exit':
                sys.exit(0)
        """#.write(to: script, atomically: true, encoding: .utf8)
        setenv("OMNITYPER_WORKER", script.path, 1)
        defer {
            if let oldWorker { setenv("OMNITYPER_WORKER", oldWorker, 1) }
            else { unsetenv("OMNITYPER_WORKER") }
            try? FileManager.default.removeItem(at: directory)
        }
        try await body(directory, python)
    }
}
