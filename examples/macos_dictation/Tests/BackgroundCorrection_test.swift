import Darwin
import Foundation

@main
private enum BackgroundCorrectionTests {
    @MainActor
    static func main() async {
        setbuf(stdout, nil)
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        let original = "大家好，我们是S.G.浪组合。"
        let instruction = "S G lang是S G L A N G。"
        let background = "项目是 SGLang-Omni。"
        let stubbed = OllamaCorrector(transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self]))
        HTTPStub.reset([:])
        let result = try await stubbed.correct(original: original, instruction: instruction, personalBackground: background)
        precondition(result.correctedText == "大家好，我们是SGLang组合。", "Explicit spelling must not acquire a background suffix")
        precondition(HTTPStub.requests.isEmpty, "Explicit spelling should not wait for the model")
        let request = try OllamaCorrector.request(original: "项目待定。", instruction: "按背景确定项目", personalBackground: background)
        let payload = try JSONSerialization.jsonObject(with: request.httpBody!) as! [String: Any]
        let messages = payload["messages"] as! [[String: String]]
        let input = messages.last!["content"]!
        precondition(input.contains(background) && input.contains("项目待定。") && input.hasSuffix("按背景确定项目"))

        // Automatic polishing retains its narrower contract. A bare project name
        // must not silently authorize expanding another valid project name.
        do {
            try PolishPolicy.validate("使用 SGLang-Omni。", original: "使用 SGLang。", personalBackground: background)
            preconditionFailure("Bare background must not expand a valid name during automatic polishing")
        } catch { }
        try PolishPolicy.validate("使用 SGLang-Omni。", original: "使用 S.G.浪。",
                                  personalBackground: "S.G.浪 → SGLang-Omni")
        print("PASS: background transmission, explicit spelling precedence and automatic-polishing boundaries")

        guard ProcessInfo.processInfo.environment["OMNI_CORRECTION_LIVE_TEST"] == "1" else { return }
        let environment = ProcessInfo.processInfo.environment
        let model = try LocalModelConfiguration(baseURL: environment["OMNI_CORRECTION_URL"] ?? "http://127.0.0.1:11434",
                                                model: environment["OMNI_CORRECTION_MODEL"] ?? LocalModelConfiguration.ollama.model)
        let corrector = OllamaCorrector(configuration: model, transport: LocalHTTPTransport())
        var failures: [String] = []
        for (name, source, command, context, expected) in [
            ("spoken spelling", original, instruction, background, "大家好，我们是SGLang组合。"),
            ("background project", original, "按背景里的项目名改，其他不变。", background, "大家好，我们是SGLang-Omni组合。"),
            ("unchanged project", "我们使用SGLang。", "不改了，保持原样。", background, "我们使用SGLang。"),
        ] {
            let reply = try await corrector.correct(original: source, instruction: command, personalBackground: context)
            print("LIVE \(name): \(reply.status.rawValue), \(reply.correctedText)")
            if !reply.correctedText.utf16.elementsEqual(expected.utf16) { failures.append(name) }
        }
        guard failures.isEmpty else { throw DictationError("Live checks failed: \(failures.joined(separator: ", "))") }
        print("PASS: local-model background smoke checks; single observations, not an accuracy estimate")
    }
}
