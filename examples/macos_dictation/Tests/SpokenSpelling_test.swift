import Foundation

@main
private enum SpokenSpellingTests {
    static func main() {
        let original = "大家好，我们是S.G.浪组合。"
        let instruction = "S G lang是S G L A N G。"
        let references = SpokenSpelling.references(in: instruction)
        precondition(references.count == 1 && references[0].joined == "SGLANG")
        precondition(SpokenSpelling.normalizeInstruction(instruction, terminology: "喜欢SGLang-Omni") == "S G lang是SGLang。")
        precondition(SpokenSpelling.normalizeInstruction("不是 A B C。", terminology: "") == "不是 A B C。")
        precondition(SpokenSpelling.normalizeInstruction("把300改成500。", terminology: "") == "把300改成500。")
        precondition(SpokenSpelling.normalizeInstruction("拼作 P Y T O R C H。", terminology: "PyTorch-Lightning") == "拼作 PyTorch。")
        let scoped = "S.G. Lang的Lang是L.A.N.G."
        precondition(SpokenSpelling.normalizeInstruction(scoped, terminology: "") == "S.G. Lang的Lang是LANG.")
        precondition(SpokenSpelling.explicitRevision(original: "大家好，我们是S G浪组合。", instruction: scoped,
                                                     terminology: "SGLang-Omni")?.corrected == "大家好，我们是SGLang组合。")
        for command in ["SGLang的Other是O T H E R。", "SGLang的Lang不是L A N G。",
                        scoped + "然后删除全部", "SGLang的Lang是L O N G。"] {
            precondition(SpokenSpelling.explicitRevision(original: original, instruction: command, terminology: "SGLang") == nil)
        }
        precondition(SpokenSpelling.explicitRevision(original: "S G浪和S.G.郎", instruction: scoped, terminology: "SGLang") == nil)
        precondition(SpokenSpelling.explicitRevision(original: original, instruction: instruction,
                                                     terminology: "SGLang-Omni")?.corrected == "大家好，我们是SGLang组合。")
        precondition(SpokenSpelling.explicitRevision(original: original, instruction: "SGLANG是SGLANG。",
                                                     terminology: "SGLang-Omni")?.corrected == "大家好，我们是SGLang组合。")
        precondition(SpokenSpelling.normalize("大家好，我们是S.G.Lang组合。", original: original,
                                              instruction: "SGLANG是SGLANG。", terminology: "SGLang-Omni") == "大家好，我们是SGLang组合。")
        precondition(SpokenSpelling.explicitRevision(original: "我用M.L.叉训练。", instruction: "M L cha是M L C H A。",
                                                     terminology: "MLCha")?.corrected == "我用MLCha训练。")
        for (text, command) in [("S.G.浪和S.G.郎", instruction), ("路径 /S.G.浪", instruction),
                                (original, instruction + "再删除后面内容"), (original, "S G lang不是S G L A N G。"),
                                ("已经使用SGLang。", instruction), ("不要修改数字300。", instruction)] {
            precondition(SpokenSpelling.explicitRevision(original: text, instruction: command, terminology: "SGLang") == nil,
                         "Ambiguous, unrelated and compound instructions must not trigger the spelling fallback")
        }
        let revised = SpokenSpelling.normalize("大家好，我们是S.G.L.A.N.G组合。", original: original,
                                               instruction: instruction, terminology: "喜欢SGLang-Omni")
        precondition(revised == "大家好，我们是SGLang组合。")
        precondition(SpokenSpelling.normalize("大家好，我们是S G L A N G组合。", original: original,
                                              instruction: instruction, terminology: "") == "大家好，我们是SGLANG组合。")
        precondition(SpokenSpelling.references(in: "这是 A B C 三个选项，不要合并。").isEmpty)
        precondition(SpokenSpelling.references(in: "把300改成500。").isEmpty)
        precondition(SpokenSpelling.normalize(original, original: original,
                                              instruction: instruction, terminology: "SGLang") == original,
                     "Spelling formatting must not guess a replacement for an unchanged model result")
        let existing = "现有S.G.L.A.N.G。费用300。"
        let unrelated = "现有S.G.L.A.N.G。费用500。"
        precondition(SpokenSpelling.normalize(unrelated, original: existing,
                                              instruction: instruction, terminology: "SGLang") == unrelated,
                     "Do not change a matching abbreviation outside the model's edit")
        let duplicate = "S.G.L.A.N.G和S.G.L.A.N.G。"
        precondition(SpokenSpelling.normalize(duplicate, original: "旧词和旧词。",
                                              instruction: instruction, terminology: "SGLang") == duplicate,
                     "Do not pick one of multiple spelling matches")
        precondition(SpokenSpelling.normalize("S.G.L.A.N.Gx", original: "旧词",
                                              instruction: instruction, terminology: "SGLang") == "S.G.L.A.N.Gx",
                     "Do not rewrite part of a longer ASCII word")
        print("PASS: explicit spoken spelling, changed-span formatting, terminology case, ambiguity and unrelated-text retention")
    }
}
