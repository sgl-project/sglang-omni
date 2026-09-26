import Foundation

private func response(_ content: [String: Any], done: Bool = true, reason: String = "stop") throws -> Data {
    let inner = try JSONSerialization.data(withJSONObject: content)
    return try JSONSerialization.data(withJSONObject: ["done": done, "done_reason": reason,
        "message": ["content": String(decoding: inner, as: UTF8.self)]])
}

private func rejected(_ action: () throws -> Void) {
    do { try action(); preconditionFailure("Invalid correction must be rejected") }
    catch { }
}

@main
private enum OllamaCorrectionTests {
    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        guard predicate() else { throw DictationError("Timed out waiting for intercepted request") }
    }

    @MainActor
    static func main() async throws {
        let original = "明天下午两点见。"
        let valid = ["scope": "local", "text": "明天下午三点见。"]
        let reply = try response(valid)
        let result = try OllamaCorrector.decode(reply, original: original, instruction: "两点改三点")
        precondition(result.correctedText == "明天下午三点见。")
        for reason in ["length", "", "error"] {
            rejected { _ = try OllamaCorrector.decode(response(valid, reason: reason), original: original, instruction: "修改") }
        }
        rejected { _ = try OllamaCorrector.decode(response(valid, done: false), original: original, instruction: "修改") }
        rejected { _ = try OllamaCorrector.decode(reply, original: original, instruction: " \n") }
        rejected { _ = try OllamaCorrector.decode(Data("not JSON".utf8), original: original, instruction: "修改") }
        for bad in [
            ["scope": "unknown", "text": original],
            ["scope": "local", "text": original, "extra": "unexpected"],
            ["scope": "clarify", "text": ""],
            ["target": "s0", "text": "三点"],
            ["text": "明天下午三点见。"],
        ] {
            rejected { _ = try OllamaCorrector.decode(response(bad), original: original, instruction: "修改") }
        }
        let unchanged = try OllamaCorrector.decode(response(["scope": "local", "text": original]),
                                                   original: original, instruction: "不改了")
        precondition(unchanged.status == .noChange)
        let deletion = try response(["scope": "local", "text": ""])
        let transport = LocalHTTPTransport(protocolClasses: [HTTPStub.self])
        let model = try LocalModelConfiguration(baseURL: "http://127.0.0.1:11434", model: "correction-model")
        let corrector = OllamaCorrector(configuration: model, transport: transport)
        HTTPStub.reset([:])
        for instruction in ["删除全部", "把上一段全部删掉", "delete all"] {
            let result = try await corrector.correct(original: original, instruction: instruction, personalBackground: "")
            precondition(result.correctedText.isEmpty)
        }
        precondition(HTTPStub.requests.isEmpty)
        for instruction in ["改一下人名", "不要删除全部", "don't delete all", "删除全部标点", "delete all commas"] {
            rejected { _ = try OllamaCorrector.decode(deletion, original: original, instruction: instruction) }
        }
        rejected { _ = try OllamaCorrector.request(original: "", instruction: "改一下") }
        rejected { _ = try OllamaCorrector.request(original: String(repeating: "长", count: 10_000), instruction: "改一下") }

        let instruction = "时间往后推一个小时，其他不变。"
        HTTPStub.reset(["/api/chat": .http(200, String(decoding: reply, as: UTF8.self))])
        let corrected = try await corrector.correct(original: original, instruction: instruction, personalBackground: "术语")
        precondition(corrected.correctedText == result.correctedText && HTTPStub.requests.count == 1)
        let body = try JSONSerialization.jsonObject(with: Data(HTTPStub.requests[0].body.utf8)) as! [String: Any]
        precondition(body["model"] as? String == "correction-model" && body["think"] as? Bool == false)
        let schema = body["format"] as! [String: Any]
        precondition(Set(schema["required"] as! [String]) == Set(["scope", "text"]))
        let messages = body["messages"] as! [[String: String]]
        let input = messages.last!["content"]!
        precondition(input.contains("参考背景：术语") && input.contains("原文：" + original)
                     && input.hasSuffix("修改意见：" + instruction))
        // Fixed teaching examples are separate role pairs; only the final user
        // message carries this round's input, never another recording's history.
        precondition(messages.map { $0["role"]! } == ["system", "user", "assistant", "user", "assistant",
                                                       "user", "assistant", "user", "assistant", "user"])
        for index in stride(from: 2, through: 8, by: 2) {
            let example = try JSONSerialization.jsonObject(with: Data(messages[index]["content"]!.utf8)) as! [String: String]
            precondition(Set(example.keys) == Set(["scope", "text"]))
            precondition(!messages[index - 1]["content"]!.contains(original))
        }

        // Regenerate once with concrete feedback, still using the same original.
        let single = "林珊负责通知。", command = "姓名最后一个字是高山的山"
        let wrong = try response(["scope": "local", "text": single])
        let right = try response(["scope": "local", "text": "林山负责通知。"])
        HTTPStub.reset(["/api/chat": .held])
        let retry = Task { try await corrector.correct(original: single, instruction: command, personalBackground: "") }
        try await wait { HTTPStub.heldCount == 1 }
        HTTPStub.set(.http(200, String(decoding: right, as: UTF8.self)), for: "/api/chat")
        HTTPStub.releaseHeld(.http(200, String(decoding: wrong, as: UTF8.self)))
        let retried = try await retry.value
        precondition(retried.correctedText == "林山负责通知。" && HTTPStub.requests.count == 2)
        let retriedBody = try JSONSerialization.jsonObject(with: Data(HTTPStub.requests[1].body.utf8)) as! [String: Any]
        let retryMessages = retriedBody["messages"] as! [[String: String]]
        precondition(Array(retryMessages.dropLast()) == Array(messages.dropLast()))
        let retryInput = retryMessages.last!["content"]!
        precondition(retryInput.contains("上次结果未通过检查：") && retryInput.contains("原文：" + single)
                     && retryInput.hasSuffix("修改意见：" + command))
        HTTPStub.reset(["/api/chat": .http(200, String(decoding: wrong, as: UTF8.self))])
        do { _ = try await corrector.correct(original: single, instruction: command, personalBackground: ""); preconditionFailure("Invalid retry accepted") }
        catch is CorrectionPlan.ValidationError { }
        precondition(HTTPStub.requests.count == 2, "Never retry more than once")

        let question = "你希望修改哪处内容？"
        let clarification = try response(["scope": "clarify", "text": question])
        HTTPStub.reset(["/api/chat": .http(200, String(decoding: clarification, as: UTF8.self))])
        do { _ = try await corrector.correct(original: original, instruction: "改一下", personalBackground: ""); preconditionFailure("Question became text") }
        catch { precondition(error.localizedDescription.contains(question)) }
        precondition(HTTPStub.requests.count == 1, "Clarification is shown, not retried")
        for fault in [HTTPStub.Reply.http(200, "not JSON"), .failure(.timedOut)] {
            HTTPStub.reset(["/api/chat": fault])
            do { _ = try await corrector.correct(original: original, instruction: instruction, personalBackground: ""); preconditionFailure("Transport/schema failure accepted") }
            catch { }
            precondition(HTTPStub.requests.count == 1)
        }
        for cancelRetry in [false, true] {
            HTTPStub.reset(["/api/chat": .held])
            let task = Task { try await corrector.correct(original: single, instruction: command, personalBackground: "") }
            try await wait { HTTPStub.heldCount == 1 }
            if cancelRetry {
                HTTPStub.releaseHeld(.http(200, String(decoding: wrong, as: UTF8.self)))
                try await wait { HTTPStub.requests.count == 2 && HTTPStub.heldCount == 1 }
            }
            task.cancel()
            do { _ = try await task.value; preconditionFailure("Cancellation must propagate") }
            catch is CancellationError { }
            try await wait { HTTPStub.cancelledCount == 1 }
        }
        print("PASS: scope schema, teaching examples, bounded retry, clarification, deletion, input limits and cancellation")
    }
}
