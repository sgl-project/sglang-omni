// SessionPipeline.swift
// Provides Session Pipeline for core app behavior.

import Foundation

@MainActor
struct SessionPipelineContext {
    let originalText: String
    var workingText: String
}

@MainActor
protocol SessionPipelineStage {
    var name: String { get }
    func run(context: SessionPipelineContext) async throws -> SessionPipelineContext
}

@MainActor
struct SessionPipelineRunner {
    let stages: [any SessionPipelineStage]

    func run(initial: SessionPipelineContext) async throws -> SessionPipelineContext {
        var context = initial
        for stage in stages {
            context = try await stage.run(context: context)
        }
        return context
    }
}
