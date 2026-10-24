import Foundation

/// Prepared output shared by delivery and post-delivery history/dictionary updates.
/// History IDs and suggestions are produced after delivery, not stored in this snapshot.
struct SessionFinalizeContext {
    let outputText: String
    let llmDurationSeconds: TimeInterval?
    let dictionaryMatches: [DictionaryMatchCandidate]
    let dictionaryCorrectedTerms: [String]
    let dictionaryCorrectionSnapshots: [DictionaryCorrectionSnapshot]
    let rewriteAnswerPayload: RewriteAnswerPayload?
}
