import Testing
@testable import VoxtKit

/// The Swift side of the bridge against the real Go core
/// (build/macos/libvoxtcore.a): no network, no models.
@Suite struct VoxtCoreTests {
    @Test func presetsAndLanguagesComeFromTheGoCore() async throws {
        let core = VoxtCore()
        let presets = try await core.presets()
        #expect(presets.contains { $0.id == "openai" })
        #expect(!presets.contains { $0.id == "codex" })
        let languages = try await core.languages()
        #expect(languages.first { $0.code == "zh" }?.chinese == "中文")
    }

    @Test func processWithoutCleanupExpandsSnippetsInGo() async throws {
        let result = try await VoxtCore().process(ProcessRequest(
            task: "transcribe", text: "write to my email address", mainLanguage: "en",
            feature: CoreFeature(cleanup: false),
            snippets: [Snippet(trigger: "my email address", text: "me@example.com")],
            provider: CoreProvider(preset: "custom")
        ))
        #expect(result.text.contains("me@example.com"), "\(result.text)")
    }

    @Test func errorsKeepTheirCode() async {
        await #expect(throws: CoreError(code: CoreError.noText, message: "voxtcore: nothing to process")) {
            try await VoxtCore().process(ProcessRequest(
                task: "transcribe", text: " ", mainLanguage: "en",
                feature: CoreFeature(cleanup: true), provider: CoreProvider(preset: "custom")
            ))
        }
    }
}
