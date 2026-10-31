import Foundation
import Testing
@testable import VoxtMobileKit

/// Talks to a real Voxt Local API server. Runs only when these are set:
///   VOXT_E2E_URL    e.g. https://your-mac.tailnet.ts.net:47911 or http://127.0.0.1:47910
///   VOXT_E2E_TOKEN  pairing token from Voxt Settings
///   VOXT_E2E_WAV    path to a 16 kHz mono WAV with speech
///   VOXT_E2E_EXPECT optional substring the returned text must contain
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["VOXT_E2E_URL"] != nil))
struct LiveServerTests {
    private let env = ProcessInfo.processInfo.environment

    private var client: VoxtClient {
        get throws {
            let url = try ServerConfig.normalizeBaseURL(try #require(env["VOXT_E2E_URL"]))
            return VoxtClient(server: ServerConfig(baseURL: url, token: try #require(env["VOXT_E2E_TOKEN"])))
        }
    }

    @Test func healthAnswers() async throws {
        let health = try await client.health()
        #expect(health.ok)
        #expect(health.app == "Voxt")
    }

    @Test func wrongTokenIsRejected() async throws {
        let url = try ServerConfig.normalizeBaseURL(try #require(env["VOXT_E2E_URL"]))
        let bad = VoxtClient(server: ServerConfig(baseURL: url, token: "wrong"))
        await #expect(throws: VoxtClient.Error.unauthorized) { try await bad.health() }
    }

    @Test func correctionLearnsWholeChineseNameAndRejectsNonVocabularyChanges() async throws {
        // Local NLTokenizer produces 浪→朗 for this unknown name.
        // The actual Mac text model must select the complete name, not just accept JSON.
        let chinese = try await client.correction(for: .init(original: "请用星浪处理。", edited: "请用星朗处理。"))
        #expect(chinese == .init(original: "星浪", replacement: "星朗"))
        let english = try await client.correction(for: .init(original: "The car is gray.", edited: "The car is grey."))
        #expect(english == .init(original: "gray", replacement: "grey"))
        let secondName = try await client.correction(for: .init(original: "请联系张伟确认。", edited: "请联系张玮确认。"))
        #expect(secondName == .init(original: "张伟", replacement: "张玮"))
        for (original, edited, old, new) in [
            ("Use oxt.", "Use Voxt.", "oxt", "Voxt"),
            ("Use Vooxt.", "Use Voxt.", "Vooxt", "Voxt"),
            ("Use Qwen2 today.", "Use Qwen3 today.", "Qwen2", "Qwen3")
        ] {
            let correction = try await client.correction(for: .init(original: original, edited: edited))
            #expect(correction == .init(original: old, replacement: new))
        }
        for pair in [("Use Voxt.", "Use Voxt. Thanks!"), ("Use Voxt.", "Use Voxt!"), ("Pay $50.", "Pay $60.")] {
            #expect(try await client.correction(for: .init(original: pair.0, edited: pair.1)) == nil)
        }
    }

    @Test func learnedVocabularyActuallyChangesTheNextComposeResult() async throws {
        let path = try #require(env["VOXT_E2E_VARIANTS_WAV"])
        let wav = try Data(contentsOf: URL(fileURLWithPath: path))
        let name = "voxt-live-learning.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let store = CorrectionLearningStore(defaults: defaults)
        store.isEnabled = true
        let before = try await client.compose(audio: wav, contentType: "audio/wav", options: .init())
        // The same spoken color can initially produce gray or grey.
        // Learn the opposite spelling so an ignored vocabulary option still fails.
        let baseline = before.text.lowercased()
        try #require(baseline.contains("gray") || baseline.contains("grey"))
        let original = baseline.contains("gray") ? "gray" : "grey"
        let replacement = original == "gray" ? "grey" : "gray"
        let correction = try #require(try await client.correction(for: .init(
            original: "The car is \(original).", edited: "The car is \(replacement)."
        )))
        let record = try #require(try store.learn(correction))
        let words = PersonalDictionary(defaults: defaults).vocabulary
        #expect(words == [replacement])
        let after = try await client.compose(audio: wav, contentType: "audio/wav", options: .init(vocabulary: words))
        print("Vocabulary contrast before=\(before.text) after=\(after.text)")
        #expect(after.text.lowercased().contains(replacement))
        #expect(!after.text.lowercased().contains(original))
        store.undo(record.id)
        #expect(PersonalDictionary(defaults: defaults).vocabulary.isEmpty)
    }

    @Test func dictationReturnsText() async throws {
        let wav = try Data(contentsOf: URL(fileURLWithPath: try #require(env["VOXT_E2E_WAV"])))
        let started = Date()
        let result = try await client.dictate(audio: wav, contentType: "audio/wav")
        print("E2E text=\(result.text) raw=\(result.rawText) asrMs=\(result.asrMs ?? -1) totalMs=\(Int(Date().timeIntervalSince(started) * 1000))")
        #expect(!result.text.isEmpty)
        if let expected = env["VOXT_E2E_EXPECT"] {
            #expect(result.text.contains(expected))
        }
    }
}
