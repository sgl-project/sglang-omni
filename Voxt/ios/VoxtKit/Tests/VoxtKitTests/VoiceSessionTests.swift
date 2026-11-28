import Foundation
import Testing
@testable import VoxtKit

@MainActor
@Suite struct VoiceSessionTests {
    @Test func transcribeRecognisesThenCleansUpAndKeepsTheEntry() async throws {
        let h = Harness(speech: "um hello there")
        h.core.answer = ProcessResult(text: "Hello there.", profile: "transcribe:cleanup", latencyMillis: 3)
        var settings = AppSettings()
        settings.general.mainLanguage = "en"
        settings.dictionary.terms = [Term(term: "Voxt", aliases: ["vox"]), Term(term: " ")]
        settings.snippets = [Snippet(trigger: "my email", text: "me@example.com")]
        let session = h.session(.transcribe, settings: settings)
        await session.start()
        #expect(session.state == .recording && h.recorder.started == 1)
        session.stop()
        await session.settled()
        #expect(session.state == .done)
        #expect(session.speech == "um hello there" && session.result == "Hello there.")
        let request = try #require(h.core.requests.first)
        #expect(request.task == "transcribe" && request.text == "um hello there" && request.mainLanguage == "en")
        #expect(request.terms?.map(\.term) == ["Voxt"], "blank terms are not sent")
        #expect(request.snippets?.first?.trigger == "my email")
        #expect(request.provider == CoreProvider(preset: "openai", key: "sk-test"))
        #expect(h.speech.inputs.first?.language == "en" && h.speech.inputs.first?.terms.count == 1)
        let entry = try #require(h.history.entries().first)
        #expect(entry.feature == .transcribe && entry.result == "Hello there." && entry.speech == "um hello there")
    }

    @Test func aDegradedCleanupShowsTheWarning() async {
        let h = Harness()
        h.core.answer = ProcessResult(text: "hello there", degraded: true, warning: "piece 1 of 1 kept as recognised", latencyMillis: 1)
        let session = h.session(.transcribe)
        await session.start()
        session.stop()
        await session.settled()
        #expect(session.state == .done && session.warning.contains("kept as recognised"))
    }

    @Test func rewriteFollowUpsCarryTheEarlierTurns() async throws {
        let h = Harness(speech: "make it polite")
        h.core.answer = ProcessResult(text: "Could you send it?", latencyMillis: 1)
        let session = h.session(.rewrite)
        session.source = "send it"
        await session.start()
        session.stop()
        await session.settled()
        #expect(session.state == .done && session.turns.count == 1 && session.source.isEmpty)
        let first = try #require(h.core.requests.first)
        #expect(first.instruction == "make it polite" && first.text == "send it" && first.previous == nil)

        h.speech.text = "shorter"
        await session.start()
        session.stop()
        await session.settled()
        let second = try #require(h.core.requests.last)
        #expect(second.instruction == "shorter" && second.text.isEmpty)
        #expect(second.previous == [CoreTurn(instruction: "make it polite", source: "send it", title: nil, answer: "Could you send it?")])
        #expect(session.turns.count == 2)
        session.clearConversation()
        #expect(session.turns.isEmpty && session.result.isEmpty)
    }

    @Test func translateTypedTextSkipsSpeech() async throws {
        let h = Harness()
        h.core.answer = ProcessResult(text: "Good morning", latencyMillis: 1)
        var settings = AppSettings()
        settings.translate.targetLanguage = "en"
        let session = h.session(.translate, settings: settings)
        session.translateTyped()
        #expect(session.state == .failed(.noSource))
        session.source = "  早上好 "
        session.translateTyped()
        await session.settled()
        #expect(session.state == .done && session.result == "Good morning" && h.speech.inputs.isEmpty)
        let request = try #require(h.core.requests.first)
        #expect(request.text == "早上好" && request.typed == true && request.targetLanguage == "en")
        #expect(h.history.entries().first?.source == "早上好")
    }

    @Test func problemsEndTheSessionWithTheirKind() async {
        let cases: [(Error?, Error?, SessionProblem)] = [
            (SpeechError.noSpeech, nil, .noSpeech),
            (SpeechError.notAuthorized, nil, .speechNotAuthorized),
            (SpeechError.notOffline("ko-KR"), nil, .notOffline("ko-KR")),
            (CoreError(code: CoreError.speech, message: "500"), nil, .speech("500")),
            (CoreError(code: CoreError.config, message: "401"), nil, .speechConfig("401")),
            (nil, CoreError(code: CoreError.config, message: "bad address"), .config("bad address")),
            (nil, CoreError(code: CoreError.config, message: "OpenAI needs an API key", field: "key"), .missing("key")),
            (nil, CoreError(code: CoreError.untranslated, message: "x"), .notTranslated),
            (nil, CoreError(code: CoreError.empty, message: "x"), .emptyAnswer),
            (nil, CoreError(code: CoreError.languageModel, message: "429"), .languageModel("429")),
        ]
        for (speechError, coreError, want) in cases {
            let h = Harness()
            h.speech.error = speechError
            h.core.error = coreError
            let session = h.session(.transcribe)
            await session.start()
            session.stop()
            await session.settled()
            #expect(session.state == .failed(want), "\(String(describing: speechError ?? coreError))")
            #expect(h.history.entries().isEmpty)
        }
    }

    @Test func aTooShortRecordingIsNoSpeech() async {
        let h = Harness()
        h.recorder.seconds = 0.1
        let session = h.session(.transcribe)
        await session.start()
        session.stop()
        #expect(session.state == .failed(.noSpeech) && h.speech.inputs.isEmpty)
    }

    @Test func theMicrophoneMustBeAllowed() async {
        var h = Harness()
        h.allowed = false
        let session = h.session(.transcribe)
        await session.start()
        #expect(session.state == .failed(.microphoneDenied) && h.recorder.started == 0)
        let denied = Harness()
        denied.recorder.startError = RecorderError.microphoneDenied
        let other = denied.session(.transcribe)
        await other.start()
        #expect(other.state == .failed(.microphoneDenied))
    }

    @Test func aSlowStartIgnoresASecondTapAndHonoursACancel() async {
        let h = Harness()
        h.recorder.holds = true
        let session = h.session(.transcribe)
        let first = Task { await session.start() }
        for _ in 0..<10_000 where !h.recorder.isHolding {
            await Task.yield()
        }
        await session.start() // a second tap while the audio opens
        #expect(h.recorder.started == 1 && session.state == .idle)
        session.cancel()
        h.recorder.release()
        await first.value
        #expect(session.state == .idle && h.recorder.cancelled == 1, "the recording opened after the cancel is closed")

        h.recorder.holds = false
        await session.start()
        #expect(session.state == .recording && h.recorder.started == 2)
    }

    @Test func cancelDropsTheRecordingOrTheRequest() async {
        let h = Harness()
        let session = h.session(.transcribe)
        await session.start()
        session.cancel()
        #expect(session.state == .idle && h.recorder.cancelled == 1)

        h.core.hang = true
        await session.start()
        session.stop()
        for _ in 0..<10_000 where h.core.requests.isEmpty {
            await Task.yield()
        }
        #expect(session.state == .composing)
        session.cancel()
        for _ in 0..<1_000 {
            await Task.yield()
        }
        #expect(session.state == .idle && h.history.entries().isEmpty)
    }
}
