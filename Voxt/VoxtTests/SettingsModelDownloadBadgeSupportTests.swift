// SettingsModelDownloadBadgeSupportTests.swift
// Provides Settings Model Download Badge Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class SettingsModelDownloadBadgeSupportTests: XCTestCase {
    func testActiveDownloadCountTracksConcurrentMLXDownloads() {
        let count = SettingsModelDownloadBadgeSupport.activeDownloadCount(
            mlxActiveDownloadRepos: [
                MLXModelManager.canonicalModelRepo("openai/whisper-tiny"),
                MLXModelManager.canonicalModelRepo("mlx-community/FireRedASR")
            ],
            customLLMActiveDownloadRepos: [],
            ggufActiveDownloadModelID: nil
        )

        XCTAssertEqual(count, 2)
    }

    func testActiveDownloadCountKeepsRemainingMLXDownloadAfterCancelingAnother() {
        let count = SettingsModelDownloadBadgeSupport.activeDownloadCount(
            mlxActiveDownloadRepos: [
                MLXModelManager.canonicalModelRepo("mlx-community/FireRedASR")
            ],
            customLLMActiveDownloadRepos: [],
            ggufActiveDownloadModelID: nil
        )

        XCTAssertEqual(count, 1)
    }

    func testActiveDownloadCountTracksConcurrentCustomLLMDownloads() {
        let count = SettingsModelDownloadBadgeSupport.activeDownloadCount(
            mlxActiveDownloadRepos: [],
            customLLMActiveDownloadRepos: [
                CustomLLMModelManager.canonicalModelRepo("mlx-community/Qwen3.5-4B-4bit"),
                CustomLLMModelManager.canonicalModelRepo("mlx-community/LFM2-1.2B-4bit")
            ],
            ggufActiveDownloadModelID: nil
        )

        XCTAssertEqual(count, 2)
    }
}
