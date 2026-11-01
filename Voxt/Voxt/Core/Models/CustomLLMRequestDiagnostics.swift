import Foundation
import MLXLMCommon

@MainActor
enum CustomLLMRequestDiagnostics {
    static func make(
        for request: CustomLLMRequestPlan,
        containerLoadSource: CustomLLMContainerLoadSource,
        containerLoadMilliseconds: Int,
        setupMilliseconds: Int,
        modelElapsedMilliseconds: Int,
        totalElapsedMilliseconds: Int,
        firstChunkMilliseconds: Int?,
        completionInfo: GenerateCompletionInfo?
    ) -> CustomLLMRunDiagnostics {
        var diagnostics = CustomLLMRunDiagnostics(
            repo: request.repo,
            taskLabel: request.kind.logLabel,
            containerLoadSource: containerLoadSource,
            containerLoadMs: containerLoadMilliseconds,
            setupMs: max(0, setupMilliseconds),
            modelElapsedMs: modelElapsedMilliseconds,
            totalElapsedMs: totalElapsedMilliseconds,
            firstChunkMs: firstChunkMilliseconds,
            overallFirstChunkMs: firstChunkMilliseconds.map { max(0, containerLoadMilliseconds + max(0, setupMilliseconds) + $0) },
            promptTokens: nil,
            completionTokens: nil,
            prefillMs: nil,
            generationMs: nil,
            modelOverheadMs: nil,
            totalOverheadMs: nil
        )
        if let completionInfo {
            let firstChunkText = firstChunkMilliseconds.map(String.init) ?? "n/a"
            let promptTPS = String(format: "%.1f", completionInfo.promptTokensPerSecond)
            let generationTPS = String(format: "%.1f", completionInfo.tokensPerSecond)
            let prefillMs = Int((completionInfo.promptTime * 1000).rounded())
            let generationMs = Int((completionInfo.generateTime * 1000).rounded())
            let modelOverheadMs = max(0, modelElapsedMilliseconds - prefillMs - generationMs)
            let totalOverheadMs = max(0, totalElapsedMilliseconds - prefillMs - generationMs)
            diagnostics = CustomLLMRunDiagnostics(
                repo: request.repo,
                taskLabel: request.kind.logLabel,
                containerLoadSource: containerLoadSource,
                containerLoadMs: containerLoadMilliseconds,
                setupMs: max(0, setupMilliseconds),
                modelElapsedMs: modelElapsedMilliseconds,
                totalElapsedMs: totalElapsedMilliseconds,
                firstChunkMs: firstChunkMilliseconds,
                overallFirstChunkMs: firstChunkMilliseconds.map { max(0, containerLoadMilliseconds + max(0, setupMilliseconds) + $0) },
                promptTokens: completionInfo.promptTokenCount,
                completionTokens: completionInfo.generationTokenCount,
                prefillMs: prefillMs,
                generationMs: generationMs,
                modelOverheadMs: modelOverheadMs,
                totalOverheadMs: totalOverheadMs
            )
            VoxtLog.llmDebug(
                "Custom LLM \(request.kind.logLabel) metrics. repo=\(request.repo), containerSource=\(containerLoadSource.rawValue), containerLoadMs=\(containerLoadMilliseconds), setupMs=\(max(0, setupMilliseconds)), firstChunkMs=\(firstChunkText), overallFirstChunkMs=\(diagnostics.overallFirstChunkMs.map(String.init) ?? "n/a"), promptTokens=\(completionInfo.promptTokenCount), generationTokens=\(completionInfo.generationTokenCount), prefillMs=\(prefillMs), generationMs=\(generationMs), modelOverheadMs=\(modelOverheadMs), totalOverheadMs=\(totalOverheadMs), promptTPS=\(promptTPS), generationTPS=\(generationTPS), stopReason=\(completionInfo.stopReason)"
            )
        }
        return diagnostics
    }
}
