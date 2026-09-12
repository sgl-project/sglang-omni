import Foundation

@main
private enum PolishFidelityTests {
    @MainActor
    static func main() throws {
        // Meaningful punctuation is not interchangeable with prose formatting.
        for (original, output) in [
            ("人人平等", "人平等"),
            ("请看看这个问题。", "请看这个问题。"),
            ("这个PR先先不要合并", "这个 PR 先不要合并。"),
            ("不要合并。", "不要不要合并。"),
            ("do not", "donot"),
            ("nowhere", "now here"),
            ("保留 batch size", "保留 batchsize"),
            ("do \t\n not", "donot"),
            ("可以合并。", "可以合并？"),
            ("Ready.", "Ready?"),
            ("可以合并？", "可以合并。"),
            ("条件是 x != y。", "条件是 x = y。"),
            ("文件在 src/api.py。", "文件在 srcapi.py。"),
            ("名称是 foo_bar。", "名称是 foobar。"),
            ("执行 git checkout --detach。", "执行 git checkout detach。"),
            ("版本 v1.2。", "版本 v12。"),
            ("条件 x <= y。", "条件 x < y。"),
            ("调用 f(a, b)。", "调用 f(ab)。"),
            ("选项 --no-cache。", "选项 no-cache。"),
            ("地址 https://example.test/a?b=c。", "地址 https://example.test/ab=c。"),
            ("保留 don't。", "保留 dont。"),
            ("值为 -5。", "值为 5。"),
        ] {
            do {
                try LocalSpeechService.validatePolish(output, original: original)
            } catch { continue }
            throw DictationError("不应接受：\(original) → \(output)")
        }
        for (original, output) in [
            ("这个PR先先不要合并等CI通过以后再说", "这个 PR 先先不要合并，等 CI 通过以后再说。"),
            ("人人平等", "人人平等。"),
            ("请看看这个问题", "请看看这个问题。"),
            ("do  \t\n not", "do not."),
            ("使用MLX跑模型", "使用 MLX 跑模型。"),
            ("可以合并？", "可以合并?"),
            ("条件是 x != y", "条件是 x != y。"),
            ("文件在 src/api.py", "文件在 src/api.py。"),
            ("可以换快捷键吗？", "可以换快捷键吗？"),
            ("Hello world", "Hello world."),
        ] {
            try LocalSpeechService.validatePolish(output, original: original)
        }
        try LocalSpeechService.validatePolish("我们用 MLX。", original: "我们用 em el ex。",
                                              personalBackground: "术语：em el ex → MLX。")
        // Every few-shot output must obey the same policy used for real responses.
        let messages = PolishPrompt.messages(text: "测试原文", personalBackground: "术语：em el ex → MLX。")
        for index in messages.indices where messages[index]["role"] == "assistant" {
            let input = try JSONSerialization.jsonObject(with: Data(messages[index - 1]["content"]!.utf8)) as! [String: String]
            try LocalSpeechService.validatePolish(messages[index]["content"]!, original: input["transcript"]!,
                                                  personalBackground: input["personal_background"] ?? "")
        }
        print("PASS: repeated words, word boundaries, question marks, code syntax, formatting and explicit spelling corrections")
    }
}
