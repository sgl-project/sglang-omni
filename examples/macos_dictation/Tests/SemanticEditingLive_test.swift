import Darwin
import Foundation

/// Fixed functional criteria, not a benchmark of general writing quality.
@main
private enum SemanticEditingLiveTests {
    struct Case {
        let name: String
        let original: String
        let instruction: String
        var background = ""
        var expectsClarification = false
        let accepts: (String) -> Bool
    }

    @MainActor
    static func main() async throws {
        setbuf(stdout, nil)
        let env = ProcessInfo.processInfo.environment
        guard env["OMNI_CORRECTION_LIVE_TEST"] == "1" else { throw DictationError("Opt in with OMNI_CORRECTION_LIVE_TEST=1") }
        let environment = ProcessInfo.processInfo.environment
        let configuration = try LocalModelConfiguration(
            baseURL: environment["OMNI_CORRECTION_URL"] ?? "http://127.0.0.1:11434",
            model: environment["OMNI_CORRECTION_MODEL"] ?? LocalModelConfiguration.ollama.model)
        let corrector = OllamaCorrector(configuration: configuration, transport: LocalHTTPTransport())
        print("MODEL \(configuration.model)")
        let cases: [Case] = [
            Case(name: "polite-particle", original: "明天和张三开会。", instruction: "把张三改成李四吧", accepts: { $0 == "明天和李四开会。" }),
            Case(name: "negative-then-positive", original: "明天和张三开会。", instruction: "不要把张三改成李四，改成王五", accepts: { $0 == "明天和王五开会。" }),
            Case(name: "semantic-last-character", original: "明天和张三开会。", instruction: "人名的最后一个字是珊瑚的珊", accepts: { $0 == "明天和张珊开会。" }),
            Case(name: "later-duplicate", original: "张三通知张三。", instruction: "后面那个张三换成李四，前面的保留", accepts: { $0 == "张三通知李四。" }),
            Case(name: "ambiguous", original: "张三通知张三。", instruction: "把张三改成李四", expectsClarification: true, accepts: { $0 == "张三通知张三。" }),
            Case(name: "semantic-amount", original: "预算是300元。", instruction: "这个金额有误，应该是五百，其他不动", accepts: { ["预算是500元。", "预算是五百元。"].contains($0) }),
            Case(name: "background-name", original: "大家好，我们是S.G.浪组合。", instruction: "按背景里的项目名改，其他不变", background: "项目是SGLang-Omni。", accepts: { $0 == "大家好，我们是SGLang-Omni组合。" }),
            Case(name: "local-unchanged-context", original: "👩🏽‍💻 明天下午两点见。e\u{301}不变。", instruction: "见面的时间往后推一个小时，其余别动", accepts: { $0.utf16.elementsEqual("👩🏽‍💻 明天下午三点见。e\u{301}不变。".utf16) }),
            Case(name: "rewrite-merge", original: "明天开会。下午三点。", instruction: "整段合成一句话", accepts: { $0 == "明天下午三点开会。" }),
            Case(name: "rewrite-polite", original: "把报告给我。明天下午三点之前。", instruction: "整段改得礼貌一点，时间保留", accepts: { $0.contains("明天下午三点") && $0.contains("之前") && $0.contains("报告") && ($0.contains("请") || $0.contains("麻烦")) }),
            Case(name: "rewrite-translate", original: "会议明天下午三点开始。地点在一号楼。", instruction: "整段翻译成英文", accepts: {
                let lower = $0.lowercased()
                return lower.contains("tomorrow") && lower.contains("meeting") && (lower.contains("3") || lower.contains("three"))
                    && (lower.contains("building 1") || lower.contains("building one")) && !$0.contains("会议")
            }),
            Case(name: "rewrite-shorten", original: "我们明天下午三点在一号楼开会，请大家准时来。", instruction: "整段压缩成一句简短的会议通知，保留时间和地点", accepts: {
                $0.contains("明天下午三点") && $0.contains("一号楼") && $0.contains("开会") && $0.count < 24
            }),
            Case(name: "rewrite-exact", original: "原来的介绍不合适。", instruction: "整段重写为：大家好，我在做一个本地语音输入工具。", accepts: { $0 == "大家好，我在做一个本地语音输入工具。" }),
            Case(name: "negative-rewrite", original: "明天和张三开会。", instruction: "不要整段重写，只把张三改成李四", accepts: { $0 == "明天和李四开会。" }),
            Case(name: "vague", original: "明天和张三开会。", instruction: "改一下", expectsClarification: true, accepts: { $0 == "明天和张三开会。" }),
            Case(name: "cancel", original: "明天和张三开会。", instruction: "不改了，保持原样", accepts: { $0 == "明天和张三开会。" }),
        ]
        let rounds = max(1, Int(env["OMNI_CORRECTION_ROUNDS"] ?? "3") ?? 3)
        _ = try? await corrector.correct(original: "明天见。", instruction: "时间改成后天", personalBackground: "")
        var scores: [Double] = [], times: [Double] = []
        for round in 1...rounds {
            var passed = 0
            for item in cases {
                let start = ProcessInfo.processInfo.systemUptime
                let actual: String
                var failure: String?
                do { actual = try await corrector.correct(original: item.original, instruction: item.instruction, personalBackground: item.background).correctedText }
                catch { actual = item.original; failure = error.localizedDescription; print("ERROR \(item.name): \(error.localizedDescription)") }
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                let clarified = failure.map { message in
                    ["需要确认：", "请说明第几个", "请明确", "请确认要修改"].contains(where: message.contains)
                } ?? false
                let ok = item.expectsClarification ? clarified : (failure == nil && item.accepts(actual))
                if ok { passed += 1 }
                times.append(elapsed)
                print(String(format: "CASE round=%d id=%@ pass=%@ seconds=%.6f output=%@", round, item.name, String(ok), elapsed, actual))
            }
            scores.append(Double(passed) / Double(cases.count))
            print("ROUND \(round): \(passed)/\(cases.count)")
        }
        func summary(_ values: [Double]) -> String {
            let mean = values.reduce(0, +) / Double(values.count)
            guard values.count > 1 else { return String(format: "%.6f (single observation; std unavailable)", mean) }
            let std = sqrt(values.reduce(0) { $0 + pow($1 - mean, 2) } / Double(values.count - 1))
            return String(format: "%.6f ± %.6f", mean, std)
        }
        print("SUMMARY fixed-criteria pass fraction mean ± sample std: \(summary(scores))")
        print("SUMMARY request seconds mean ± sample std: \(summary(times)); n=\(times.count)")
        print("Synthetic development cases. Only expected clarification counts as success; infrastructure errors fail the case. Rewrite criteria check facts/format only; review outputs for writing quality. No ASR or real-editor delivery timing.")
    }
}
