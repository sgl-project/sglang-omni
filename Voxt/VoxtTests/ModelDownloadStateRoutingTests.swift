// ModelDownloadStateRoutingTests.swift
// Provides Model Download State Routing Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class ModelDownloadStateRoutingTests: XCTestCase {
    func testCustomLLMDownloadRoutingTracksManagerCurrentRepoInsteadOfSelectedRepo() {
        let targetRepo = "mlx-community/Qwen3-4B-4bit"
        let otherRepo = "mlx-community/Qwen3-8B-4bit"
        let state = CustomLLMModelManager.ModelState.downloading(
            progress: 0.25,
            completed: 25,
            total: 100,
            currentFile: "model.safetensors",
            completedFiles: 1,
            totalFiles: 4
        )

        XCTAssertTrue(
            ModelDownloadStateRouting.isCustomLLMDownloading(
                repo: targetRepo,
                managerRepo: targetRepo,
                state: state
            )
        )
        XCTAssertTrue(
            ModelDownloadStateRouting.isAnotherCustomLLMDownloadActive(
                repo: otherRepo,
                managerRepo: targetRepo,
                state: state
            )
        )
        XCTAssertFalse(
            ModelDownloadStateRouting.isAnotherCustomLLMDownloadActive(
                repo: targetRepo,
                managerRepo: targetRepo,
                state: state
            )
        )
    }

    func testPausedRoutingMatchesManagerOperationTarget() {
        let llmState = CustomLLMModelManager.ModelState.paused(
            progress: 0.8,
            completed: 80,
            total: 100,
            currentFile: "weights",
            completedFiles: 3,
            totalFiles: 4
        )

        XCTAssertTrue(
            ModelDownloadStateRouting.isCustomLLMPaused(
                repo: "mlx-community/Qwen3-4B-4bit",
                managerRepo: "mlx-community/Qwen3-4B-4bit",
                state: llmState
            )
        )
    }
}
