// ModelDownloadStateRouting.swift
// Provides Model Download State Routing for model settings.

import Foundation

enum ModelDownloadStateRouting {
    private enum OperationPhase {
        case idle
        case downloading
        case paused
    }

    static func isCustomLLMOperationTarget(
        repo: String,
        managerRepo: String
    ) -> Bool {
        repo == managerRepo
    }

    static func isCustomLLMDownloading(
        repo: String,
        managerRepo: String,
        state: CustomLLMModelManager.ModelState
    ) -> Bool {
        isOperationTargetActive(
            isTarget: isCustomLLMOperationTarget(repo: repo, managerRepo: managerRepo),
            phase: operationPhase(for: state),
            expected: .downloading
        )
    }

    static func isCustomLLMPaused(
        repo: String,
        managerRepo: String,
        state: CustomLLMModelManager.ModelState
    ) -> Bool {
        isOperationTargetActive(
            isTarget: isCustomLLMOperationTarget(repo: repo, managerRepo: managerRepo),
            phase: operationPhase(for: state),
            expected: .paused
        )
    }

    static func isAnotherCustomLLMDownloadActive(
        repo: String,
        managerRepo: String,
        state: CustomLLMModelManager.ModelState
    ) -> Bool {
        isAnotherOperationActive(
            isTarget: isCustomLLMOperationTarget(repo: repo, managerRepo: managerRepo),
            phase: operationPhase(for: state)
        )
    }

    private static func operationPhase(for state: CustomLLMModelManager.ModelState) -> OperationPhase {
        switch state {
        case .downloading:
            return .downloading
        case .paused:
            return .paused
        default:
            return .idle
        }
    }

    private static func isOperationTargetActive(
        isTarget: Bool,
        phase: OperationPhase,
        expected: OperationPhase
    ) -> Bool {
        isTarget && phase == expected
    }

    private static func isAnotherOperationActive(
        isTarget: Bool,
        phase: OperationPhase
    ) -> Bool {
        phase == .downloading && !isTarget
    }
}
