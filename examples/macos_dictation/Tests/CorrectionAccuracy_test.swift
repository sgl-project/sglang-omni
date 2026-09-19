import Darwin
import Foundation

/// Fixed functional cases, shared by the baseline and candidate implementations.
/// No microphone, external editor, user files, or model downloads.
@main
private enum CorrectionAccuracyTests {
    @MainActor
    static func main() async throws {
        setbuf(stdout, nil)
        guard ProcessInfo.processInfo.environment["OMNI_CORRECTION_LIVE_TEST"] == "1" else {
            throw DictationError("Set OMNI_CORRECTION_LIVE_TEST=1 to use the local model.")
        }
        let environment = ProcessInfo.processInfo.environment
        let configuration = try LocalModelConfiguration(
            baseURL: environment["OMNI_CORRECTION_URL"] ?? "http://127.0.0.1:11434",
            model: environment["OMNI_CORRECTION_MODEL"] ?? LocalModelConfiguration.ollama.model)
        let corrector = OllamaCorrector(configuration: configuration, transport: LocalHTTPTransport())
        print("MODEL \(configuration.model)")
        let cases: [(String, String, String, String, String)] = [
            ("spelling", "大家好，我们是S.G.浪组合。", "S G lang是S G L A N G。", "SGLang-Omni", "大家好，我们是SGLang组合。"),
            ("scoped-spelling", "大家好，我们是S G浪组合。", "S.G. Lang的Lang是L.A.N.G.", "SGLang-Omni", "大家好，我们是SGLang组合。"),
            ("partial-spelling", "大家好，我们是S G浪组合。", "S G浪的浪是L A N G。", "", "大家好，我们是S GLANG组合。"),
            ("number", "费用是300元。", "把300改成500，其他不变。", "", "费用是500元。"),
            ("name", "明天下午两点和张三开会。", "把张三改成张珊，珊瑚的珊，其他不变。", "", "明天下午两点和张珊开会。"),
            ("negation-edit", "不要推送这个改动。", "把不要改成可以，其他不变。", "", "可以推送这个改动。"),
            ("ordinal", "张三通知张三。", "把第二个张三改成李四。", "", "张三通知李四。"),
            ("ambiguous", "张三通知张三。", "把张三改成李四。", "", "张三通知张三。"),
            ("number-boundary", "1300和300元。", "把300改成500。", "", "1300和500元。"),
            ("literal", "备注：待定。", "把“待定”改成“把两点改成三点”。", "", "备注：把两点改成三点。"),
            ("cancel", "我们使用SGLang。", "不改了，保持原样。", "项目是SGLang-Omni。", "我们使用SGLang。"),
            ("negative", "明天和张三开会。", "不要把张三改成李四。", "", "明天和张三开会。"),
            ("background", "大家好，我们是S.G.浪组合。", "按背景里的项目名改，其他不变。", "项目是SGLang-Omni。", "大家好，我们是SGLang-Omni组合。"),
            ("semantic-name", "明天和张三开会。", "人名最后一个字是珊瑚的珊，其他不变。", "", "明天和张珊开会。"),
            ("unicode", "👩🏽‍💻 费用是300元。e\u{301}不变。", "把300改成500。", "", "👩🏽‍💻 费用是500元。e\u{301}不变。"),
            ("delete", "明天见，真的真的。", "删除第二个真的。", "", "明天见，真的。"),
        ]
        let rounds = Int(ProcessInfo.processInfo.environment["OMNI_CORRECTION_ROUNDS"] ?? "3") ?? 3
        let extraCases: [(String, String, String, String, String)] = [
            ("new-name", "张三去接李四。", "把张三改成章三。", "", "章三去接李四。"),
            ("new-number", "地址是杭州，预算300元。", "把300改成350。", "", "地址是杭州，预算350元。"),
            ("first", "甲说同意，乙说同意。", "把第一个同意改成反对。", "", "甲说反对，乙说同意。"),
            ("literal-space", "备注：待定。", "把“待定”改成“ A B ”。", "", "备注： A B 。"),
            ("homophone", "我们使用S.G.郎。", "S G lang是S G L A N G。", "SGLang-Omni", "我们使用SGLang。"),
            ("single-character", "张三负责这件事。", "把三改成珊。", "", "张珊负责这件事。"),
            ("relative-time", "下午两点开会。", "时间往后推一个小时，其他不变。", "", "下午三点开会。"),
            ("field-label", "备注：待定。", "把备注改成已确认。", "", "备注：已确认。"),
            ("semantic-number", "预算是300元。", "金额改成500，其他不变。", "", "预算是500元。"),
            ("absent-target", "杭州和苏州。", "把南京改成北京。", "", "杭州和苏州。"),
            ("duplicate-scope", "S G浪和S G浪。", "S G浪的浪是L A N G。", "", "S G浪和S G浪。"),
            ("literal-arrow", "文字：A→B。", "把“A→B”改成“B→C”。", "", "文字：B→C。"),
            ("ordinary-word", "猫坐在椅子上。", "把猫换成狗，其他不变。", "", "狗坐在椅子上。"),
            ("existing-name", "SGLang很好。", "S G lang是S G L A N G。", "SGLang-Omni", "SGLang很好。"),
            ("duplicate-spelling", "我使用S G浪和S.G.郎。", "S G lang是S G L A N G。", "SGLang", "我使用S G浪和S.G.郎。"),
            ("negative-name", "请使用LightGBM。", "别把LightGBM改成XGBoost。", "", "请使用LightGBM。"),
        ]
        let activeCases = ProcessInfo.processInfo.environment["OMNI_CORRECTION_EXTRA_CASES"] == "1" ? extraCases : cases
        // Warm the identical configured model before either variant's measured rounds.
        _ = try? await corrector.correct(original: "明天见。", instruction: "时间改为后天，其他不变。", personalBackground: "")
        var allTimes: [Double] = [], roundScores: [Double] = []
        for round in 1...max(1, rounds) {
            var passed = 0
            for (id, original, instruction, background, expected) in activeCases {
                let start = ProcessInfo.processInfo.systemUptime
                let actual: String
                var failure: String?
                do { actual = try await corrector.correct(original: original, instruction: instruction, personalBackground: background).correctedText }
                catch { actual = original; failure = error.localizedDescription; print("ERROR \(id): \(error.localizedDescription)") }
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                let expectsClarification = ["ambiguous", "absent-target", "duplicate-scope", "duplicate-spelling"].contains(id)
                let clarified = failure.map { message in
                    ["需要确认：", "请说明第几个", "无法唯一定位", "请明确", "请确认要修改"].contains(where: message.contains)
                } ?? false
                let ok = expectsClarification ? clarified : (failure == nil && actual.utf16.elementsEqual(expected.utf16))
                if ok { passed += 1 }
                allTimes.append(elapsed)
                print(String(format: "CASE round=%d id=%@ pass=%@ seconds=%.6f output=%@", round, id, String(ok), elapsed, actual))
            }
            roundScores.append(Double(passed) / Double(activeCases.count))
            print("ROUND \(round): \(passed)/\(activeCases.count)")
        }
        func summarize(_ values: [Double]) -> String {
            let mean = values.reduce(0, +) / Double(values.count)
            guard values.count > 1 else { return String(format: "%.6f (single observation; std unavailable)", mean) }
            let std = sqrt(values.reduce(0) { $0 + pow($1 - mean, 2) } / Double(values.count - 1))
            return String(format: "%.6f ± %.6f", mean, std)
        }
        print("SUMMARY round exact-match fraction mean ± sample std: \(summarize(roundScores))")
        print("SUMMARY request seconds mean ± sample std: \(summarize(allTimes)); n=\(allTimes.count)")
        print("Fixed regression set, not a population accuracy estimate. Only expected clarification counts as success; inspect ERROR lines for failed edits.")
    }
}
