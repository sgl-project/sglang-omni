import Foundation

@main
private enum CorrectionPlanTests {
    static func rejects(_ action: () throws -> Void) {
        do { try action(); preconditionFailure("Invalid edit accepted") }
        catch { }
    }

    @MainActor
    static func main() async throws {
        let corrector = OllamaCorrector(transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self]))
        HTTPStub.reset([:])
        for (original, instruction, background, expected) in [
            ("备注：待定。", "把“待定”改成“把两点改成三点”。", "", "备注：把两点改成三点。"),
            ("修改意见：待定。", "把“修改意见”改成“建议”。", "", "建议：待定。"),
            ("备注：第一个张三。", "把“第一个张三”改成“李四”。", "", "备注：李四。"),
            ("S G浪组合。", "S G浪的浪是L A N G。", "", "S GLANG组合。"),
            ("S.G.浪组合。", "S G lang是S G L A N G。", "SGLang-Omni", "SGLang组合。"),
            ("SGLang。", "不改了，保持原样。", "SGLang-Omni", "SGLang。"),
            ("我们使用SGLang-Omni。", "按背景里的项目名改。", "SGLang-Omni", "我们使用SGLang-Omni。"),
            ("保留\n格式。", "删除全部", "", ""),
        ] {
            let result = try await corrector.correct(original: original, instruction: instruction, personalBackground: background)
            precondition(result.correctedText.utf16.elementsEqual(expected.utf16), "Unexpected direct result: \(result.correctedText)")
        }
        precondition(HTTPStub.requests.isEmpty, "Narrow explicit rules must not call a model")
        for (original, command) in [("张三和张三", "把张三改成李四。"), ("张三", "把第二个张三改成李四。"),
                                    ("杭州。", "把“南京”改成“北京”。"), ("杭州和苏州。", "把南京改成北京。"),
                                    ("1300元。", "把300改成500。"), ("请杨帆发通知。", "把手机号改成67890"),
                                    ("请联系李梅。", "把邮箱改成a@example.com"),
                                    ("S G浪与S G浪", "S G浪的浪是L A N G。") ] {
            rejects { _ = try CorrectionPlan(original: original, instruction: command, background: "") }
        }
        // Model interpretations retain exact context around a recognized local target.
        for (original, instruction, good, bad) in [
            ("费用300，预算1300。", "把300改成500。", "费用500，预算1300。", "费用500，预算1500。"),
            ("👩🏽‍💻 张三与张三 e\u{301}", "把第二个张三改成李四。", "👩🏽‍💻 张三与李四 e\u{301}", "👩🏽‍💻 李四与李四 é"),
            ("备注： 待定  。", "把备注改成已确认。", "备注： 已确认  。", "备注：已确认。"),
            ("负责人：张三。", "把负责人改成李四。", "负责人：李四。", "李四：张三。"),
            ("foo foobar foo_bar /foo/path", "把foo改成bar。", "bar foobar foo_bar /foo/path", "bar barbar foo_bar /foo/path"),
            ("备注：待定。", "把备注改成“已确认”。", "备注：已确认。", "备注：已完成。"),
        ] {
            let plan = try CorrectionPlan(original: original, instruction: instruction, background: "")
            precondition(plan.direct == nil)
            let result = try plan.applying(scope: "local", text: good)
            precondition(result.utf16.elementsEqual(good.utf16))
            rejects { _ = try plan.applying(scope: "local", text: bad) }
            rejects { _ = try plan.applying(scope: "rewrite", text: good) }
        }
        let original = "👩🏽‍💻 明天和张三开会。e\u{301}保持。"
        let homophone = try CorrectionPlan(original: "章三来开会。", instruction: "把张三改成李四。", background: "")
        precondition(homophone.direct == nil)
        let disambiguated = try homophone.applying(scope: "local", text: "李四来开会。")
        precondition(disambiguated == "李四来开会。")
        rejects { _ = try homophone.applying(scope: "local", text: "李四来聚餐。") }
        for (source, command) in [("电话待定。", "把电话改成12345"), ("请联系12345。", "把手机号改成67890"),
                                  ("邮箱待定。", "把邮箱改成a@example.com"), ("请联系a@example.com。", "把邮箱改成b@example.com")] {
            let contact = try CorrectionPlan(original: source, instruction: command, background: "")
            precondition(contact.direct == nil, "Contact evidence permits model interpretation, not a guessed rule edit")
        }
        let instruction = "人名最后一个字是珊瑚的珊，其他不变。"
        let plan = try CorrectionPlan(original: original, instruction: instruction, background: "")
        let good = "👩🏽‍💻 明天和张珊开会。e\u{301}保持。"
        let corrected = try plan.applying(scope: "local", text: good)
        precondition(corrected.utf16.elementsEqual(good.utf16))
        for bad in [original, "👩🏽‍💻 明天和王珊开会。e\u{301}保持。", good + instruction, "", "```" + good] {
            rejects { _ = try plan.applying(scope: "local", text: bad) }
        }
        rejects { _ = try plan.applying(scope: "unknown", text: good) }
        rejects { _ = try plan.applying(scope: "clarify", text: "") }
        rejects { _ = try plan.applying(scope: "local", text: String(repeating: "字", count: 8001)) }
        let rewrite = try CorrectionPlan(original: "明天开会。下午三点。", instruction: "整段合成一句话", background: "")
        let rewritten = try rewrite.applying(scope: "rewrite", text: "明天下午三点开会。")
        precondition(rewritten == "明天下午三点开会。")
        rejects { _ = try rewrite.applying(scope: "rewrite", text: "") }
        let natural = try CorrectionPlan(original: "明天和张三开会。", instruction: "不要把张三改成李四，改成王五", background: "")
        precondition(natural.direct == nil, "Compound instructions need model interpretation")
        let naturalResult = try natural.applying(scope: "local", text: "明天和王五开会。")
        precondition(naturalResult == "明天和王五开会。")
        let cancelled = Task { @MainActor in
            try await corrector.correct(original: "张三", instruction: "把“张三”改成“李四”", personalBackground: "")
        }
        cancelled.cancel()
        do { _ = try await cancelled.value; preconditionFailure("Cancellation was ignored by local edit") }
        catch is CancellationError { }
        print("PASS: direct spelling/literals, local bounds, ordinals, field values, Unicode, rewrite and cancellation")
    }
}
