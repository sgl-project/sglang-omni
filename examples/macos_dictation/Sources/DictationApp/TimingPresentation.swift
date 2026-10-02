#if canImport(DictationCore)
import DictationCore
#endif
import Foundation

/// One completed round, presented identically in the floating bar and result window.
@MainActor
struct TimingPresentation {
    let summary: String
    let asrDuration: String
    let polishDuration: String
    let totalDuration: String
    let asrStatus: String
    let polishStatus: String

    init?(session: DictationSession) {
        guard [.ready, .failed].contains(session.phase), session.totalSeconds != nil else { return nil }
        asrDuration = Self.duration(session.asrSeconds)
        polishDuration = Self.duration(session.polishSeconds)
        totalDuration = Self.duration(session.totalSeconds)
        var stages = ["ASR \(asrDuration)"]
        if session.polishSeconds != nil { stages.append("LLM \(polishDuration)") }
        summary = stages.joined(separator: " · ")
        asrStatus = session.phase == .failed ? "失败" : (session.rawText.isEmpty ? "未识别到文字" : "完成")
        if session.polishSeconds != nil {
            polishStatus = session.hasPolishedResult ? "完成" : "已回退原文"
        } else {
            polishStatus = session.polishWasRequested ? "未执行" : "已关闭"
        }
    }

    private static func duration(_ seconds: Double?) -> String {
        guard let seconds else { return "—" }
        // A fast or immediately failed request must not appear as zero cost.
        return seconds < 0.1 ? "<0.1s" : String(format: "%.1fs", seconds)
    }
}
