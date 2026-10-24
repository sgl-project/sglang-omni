// ModelSettingsManagerRefreshSupportTests.swift
// Provides Model Settings Manager Refresh Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class ModelSettingsManagerRefreshSupportTests: XCTestCase {
    func testMLXPhaseIgnoresProgressPayloadChanges() {
        let phaseA = ModelSettingsManagerRefreshSupport.phase(
            for: MLXModelManager.ModelState.downloading(
                progress: 0.1,
                completed: 10,
                total: 100,
                currentFile: "a",
                completedFiles: 0,
                totalFiles: 2
            )
        )
        let phaseB = ModelSettingsManagerRefreshSupport.phase(
            for: MLXModelManager.ModelState.downloading(
                progress: 0.9,
                completed: 90,
                total: 100,
                currentFile: "b",
                completedFiles: 1,
                totalFiles: 2
            )
        )

        XCTAssertEqual(phaseA, ModelSettingsManagerActivityPhase.downloading)
        XCTAssertEqual(phaseA, phaseB)
    }

    func testCustomLLMPhaseMapsPausedState() {
        let phase = ModelSettingsManagerRefreshSupport.phase(
            for: CustomLLMModelManager.ModelState.paused(
                progress: 0.4,
                completed: 40,
                total: 100,
                currentFile: "weights.safetensors",
                completedFiles: 1,
                totalFiles: 4
            )
        )

        XCTAssertEqual(phase, ModelSettingsManagerActivityPhase.paused)
    }

}
