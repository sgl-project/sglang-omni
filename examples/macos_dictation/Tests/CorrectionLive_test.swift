import Foundation

@main
private enum CorrectionLiveTests {
    @MainActor
    static func main() async throws {
        guard ProcessInfo.processInfo.environment["OMNI_CORRECTION_LIVE_TEST"] == "1" else {
            throw DictationError("Set OMNI_CORRECTION_LIVE_TEST=1 to contact the configured local Ollama server.")
        }
        let environment = ProcessInfo.processInfo.environment
        let model = try LocalModelConfiguration(baseURL: environment["OMNI_CORRECTION_URL"] ?? "http://127.0.0.1:11434",
                                                model: environment["OMNI_CORRECTION_MODEL"] ?? LocalModelConfiguration.ollama.model)
        let corrector = OllamaCorrector(configuration: model, transport: LocalHTTPTransport())
        for (name, original, instruction, expected) in [
            ("name", "明天下午两点和张三开会。", "把张三改成张珊，珊瑚的珊，其他不变。", "明天下午两点和张珊开会。"),
            ("number", "费用是300元。", "把300改成500，其他不变。", "费用是500元。"),
            ("negation", "不要推送这个改动。", "把不要改成可以，其他不变。", "可以推送这个改动。"),
        ] {
            let result = try await corrector.correct(original: original, instruction: instruction, personalBackground: "")
            guard result.status == .ok, result.correctedText.utf16.elementsEqual(expected.utf16) else {
                throw DictationError("Live \(name) check failed: \(result.status.rawValue), \(result.correctedText)")
            }
            print("PASS: live \(name) correction")
        }
        print("Synthetic text smoke checks only; no microphone/editor access, quality benchmark or latency claim.")
    }
}
