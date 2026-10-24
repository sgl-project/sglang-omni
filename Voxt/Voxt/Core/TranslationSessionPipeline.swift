// TranslationSessionPipeline.swift
// Provides Translation Session Pipeline for core app behavior.

import Foundation

struct TranslationSessionTranslateStage: SessionPipelineStage {
    let targetLanguage: TranslationTargetLanguage
    let transform: @MainActor (String, TranslationTargetLanguage) async throws -> String

    var name: String { "translate" }

    func run(context: SessionPipelineContext) async throws -> SessionPipelineContext {
        var next = context
        next.workingText = try await transform(context.workingText, targetLanguage)
        return next
    }
}

struct TranslationSessionRewriteStage: SessionPipelineStage {
    let sourceText: String
    let transform: @MainActor (String, String) async throws -> String

    var name: String { "rewrite" }

    func run(context: SessionPipelineContext) async throws -> SessionPipelineContext {
        var next = context
        next.workingText = try await transform(context.workingText, sourceText)
        return next
    }
}

struct TranslationSessionStrictRetryTranslateStage: SessionPipelineStage {
    let targetLanguage: TranslationTargetLanguage
    let shouldRetry: @MainActor (String, String) -> Bool
    let strictTranslate: @MainActor (String, TranslationTargetLanguage) async throws -> String

    var name: String { "strictRetryTranslate" }

    func run(context: SessionPipelineContext) async throws -> SessionPipelineContext {
        guard shouldRetry(context.originalText, context.workingText) else { return context }
        var next = context
        next.workingText = try await strictTranslate(context.originalText, targetLanguage)
        return next
    }
}

enum TranslationSessionPipelineBuilder {
    static func makeTranslationStages(
        targetLanguage: TranslationTargetLanguage,
        allowStrictRetry: Bool,
        translate: @escaping @MainActor (String, TranslationTargetLanguage) async throws -> String,
        shouldRetry: @escaping @MainActor (String, String) -> Bool,
        strictTranslate: @escaping @MainActor (String, TranslationTargetLanguage) async throws -> String
    ) -> [any SessionPipelineStage] {
        var stages: [any SessionPipelineStage] = [
            TranslationSessionTranslateStage(
                targetLanguage: targetLanguage,
                transform: translate
            )
        ]

        if allowStrictRetry {
            stages.append(
                TranslationSessionStrictRetryTranslateStage(
                    targetLanguage: targetLanguage,
                    shouldRetry: shouldRetry,
                    strictTranslate: strictTranslate
                )
            )
        }

        return stages
    }

    static func makeRewriteStages(
        sourceText: String,
        rewrite: @escaping @MainActor (String, String) async throws -> String
    ) -> [any SessionPipelineStage] {
        [
            TranslationSessionRewriteStage(
                sourceText: sourceText,
                transform: rewrite
            )
        ]
    }
}
