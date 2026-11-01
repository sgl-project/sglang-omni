import AppKit
import SwiftUI
import XCTest
@testable import Voxt

@MainActor
final class ModelSettingsViewRefreshTests: MLXModelManagerTestCase {
    private final class CatalogReadCountingMLXModelManager: MLXModelManager {
        private(set) var catalogReads = 0

        override func catalogSnapshot(for repo: String) -> CatalogSnapshot {
            catalogReads += 1
            return super.catalogSnapshot(for: repo)
        }
    }

    /// An idle, visible Model page rebuilt its catalog several times a second and
    /// kept every replaced catalog reachable, growing memory without bound (#3397).
    func testIdleModelPageSettlesAndInstallationChangesDoNotAccumulateCatalogs() async throws {
        try await withIsolatedModelStorageRoot { _ in
            let mlx = CatalogReadCountingMLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            let customLLM = CustomLLMModelManager(modelRepo: CustomLLMModelManager.defaultModelRepo)
            let gguf = GGUFTranslationModelManager(modelID: GGUFTranslationModelCatalog.defaultModelID)
            let windowState = MainWindowVisibilityState()
            windowState.isVisible = true
            let window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 820, height: 560),
                styleMask: [.titled],
                backing: .buffered,
                defer: false
            )
            window.isReleasedWhenClosed = false
            window.contentView = NSHostingView(rootView: ModelSettingsView(
                mlxModelManager: mlx,
                customLLMManager: customLLM,
                ggufTranslationModelManager: gguf,
                mainWindowState: windowState,
                missingConfigurationIssues: [],
                navigationRequest: nil,
                isActive: true
            ))
            window.orderFrontRegardless()

            let settledWhileIdle = await waitForCatalogToSettle(mlx)
            XCTAssertTrue(settledWhileIdle, "The catalog kept rebuilding with no installation change")
            // Every retained catalog keeps view copies that reference the window state.
            let settledRetainCount = CFGetRetainCount(windowState)

            for _ in 0..<3 {
                let readsBeforeChange = mlx.catalogReads
                mlx.checkExistingModel(refresh: true)
                let rebuilt = await waitUntil(timeout: .seconds(5)) { mlx.catalogReads > readsBeforeChange }
                XCTAssertTrue(rebuilt, "An installation change did not rebuild the catalog")
                let settledAfterChange = await waitForCatalogToSettle(mlx)
                XCTAssertTrue(settledAfterChange, "The catalog kept rebuilding after an installation change")
            }
            let retainCountAfterChanges = CFGetRetainCount(windowState)
            XCTAssertEqual(retainCountAfterChanges, settledRetainCount, "Replaced catalogs are still reachable")

            window.contentView = nil
            window.close()
            await mlx.shutdownForApplicationTermination()
            await customLLM.shutdownForApplicationTermination()
            await gguf.shutdownForApplicationTermination()
        }
    }

    private func waitForCatalogToSettle(
        _ mlx: CatalogReadCountingMLXModelManager,
        quietPeriod: Duration = .seconds(1),
        timeout: Duration = .seconds(10)
    ) async -> Bool {
        let clock = ContinuousClock()
        var lastReads = mlx.catalogReads
        var quietSince = clock.now
        return await waitUntil(timeout: timeout) {
            if mlx.catalogReads != lastReads {
                lastReads = mlx.catalogReads
                quietSince = clock.now
                return false
            }
            return clock.now - quietSince >= quietPeriod
        }
    }

    // Suspends instead of spinning a nested run loop, so main-queue work keeps running.
    private func waitUntil(timeout: Duration, _ condition: () -> Bool) async -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now + timeout
        while clock.now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return condition()
    }
}
