import Foundation

@main
private enum PolishFidelityTests {
    @MainActor
    static func main() throws {
        // Meaningful punctuation is not interchangeable with prose formatting.
        for (original, output) in [
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
            ("这个PR先先不要合并等CI通过以后再说", "这个 PR 先不要合并，等 CI 通过以后再说。"),
            ("条件是 x != y", "条件是 x != y。"),
            ("文件在 src/api.py", "文件在 src/api.py。"),
            ("可以换快捷键吗？", "可以换快捷键吗？"),
            ("Hello world", "Hello world."),
        ] {
            try LocalSpeechService.validatePolish(output, original: original)
        }
        try LocalSpeechService.validatePolish("我们用 MLX。", original: "我们用 em el ex。",
                                              personalBackground: "术语：em el ex → MLX。")
        print("PASS: code/path/operator preservation, ordinary punctuation and explicit spelling corrections")
    }
}
