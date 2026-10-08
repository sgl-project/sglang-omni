// ModelSettingsProgressRefreshSupportTests.swift
// Provides Model Settings Progress Refresh Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class ModelSettingsProgressRefreshSupportTests: XCTestCase {
    func testShouldRefreshCatalogForLifecycleChangeOnlyWhenActiveAndVisible() {
        XCTAssertTrue(
            ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForLifecycleChange(
                isActive: true,
                isWindowVisible: true
            )
        )
        XCTAssertFalse(
            ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForLifecycleChange(
                isActive: false,
                isWindowVisible: true
            )
        )
        XCTAssertFalse(
            ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForLifecycleChange(
                isActive: true,
                isWindowVisible: false
            )
        )
    }

    func testShouldRefreshCatalogForMetadataChangeOnlyWhenActiveVisibleAndPolling() {
        XCTAssertTrue(
            ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForMetadataChange(
                isActive: true,
                isWindowVisible: true,
                shouldPollModelState: true
            )
        )
        XCTAssertFalse(
            ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForMetadataChange(
                isActive: false,
                isWindowVisible: true,
                shouldPollModelState: true
            )
        )
        XCTAssertFalse(
            ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForMetadataChange(
                isActive: true,
                isWindowVisible: false,
                shouldPollModelState: true
            )
        )
        XCTAssertFalse(
            ModelSettingsProgressRefreshSupport.shouldRefreshCatalogForMetadataChange(
                isActive: true,
                isWindowVisible: true,
                shouldPollModelState: false
            )
        )
    }

    func testShouldPollModelStateWhenNonCurrentMLXDownloadIsActive() {
        let shouldPoll = shouldPollModelState(
            mlxState: .notDownloaded,
            mlxHasActiveDownloadingRepos: true,
            customLLMState: .notDownloaded
        )

        XCTAssertTrue(shouldPoll)
    }



    func testShouldNotPollModelStateWithoutActiveDownloads() {
        let shouldPoll = shouldPollModelState(
            mlxState: .downloaded,
            mlxHasActiveDownloadingRepos: false,
            customLLMState: .downloaded
        )

        XCTAssertFalse(shouldPoll)
    }

    func testShouldNotPollModelStateForLoadingWithoutActiveDownloads() {
        let shouldPoll = shouldPollModelState(
            mlxState: .loading,
            mlxHasActiveDownloadingRepos: false,
            customLLMState: .notDownloaded
        )

        XCTAssertFalse(shouldPoll)
    }

    func testShouldNotPollModelStateForPausedMLXWhileCancellationStillCleansUp() {
        let shouldPoll = shouldPollModelState(
            mlxState: .paused(
                progress: 0.5,
                completed: 50,
                total: 100,
                currentFile: "weights.bin",
                completedFiles: 1,
                totalFiles: 2
            ),
            mlxHasActiveDownloadingRepos: false,
            customLLMState: .notDownloaded
        )

        XCTAssertFalse(shouldPoll)
    }

    func testShouldPollModelStateWhenCustomLLMDownloadIsActive() {
        let shouldPoll = shouldPollModelState(
            mlxState: .notDownloaded,
            mlxHasActiveDownloadingRepos: false,
            customLLMState: .notDownloaded,
            customLLMHasActiveDownloadingRepos: true
        )

        XCTAssertTrue(shouldPoll)
    }

    private func shouldPollModelState(
        mlxState: MLXModelManager.ModelState,
        mlxHasActiveDownloadingRepos: Bool,
        customLLMState: CustomLLMModelManager.ModelState,
        customLLMStateByRepo: [String: CustomLLMModelManager.ModelState] = [:],
        customLLMHasActiveDownloadingRepos: Bool = false,
        ggufStateByID: [GGUFTranslationModelID: GGUFTranslationModelManager.ModelState] = [:],
        ggufActiveDownloadModelID: GGUFTranslationModelID? = nil
    ) -> Bool {
        ModelSettingsProgressRefreshSupport.shouldPollModelState(
            mlxState: mlxState,
            mlxHasActiveDownloadingRepos: mlxHasActiveDownloadingRepos,
            customLLMState: customLLMState,
            customLLMStateByRepo: customLLMStateByRepo,
            customLLMHasActiveDownloadingRepos: customLLMHasActiveDownloadingRepos,
            ggufStateByID: ggufStateByID,
            ggufActiveDownloadModelID: ggufActiveDownloadModelID
        )
    }
}
