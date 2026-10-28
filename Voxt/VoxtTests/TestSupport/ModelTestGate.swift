// ModelTestGate.swift
// Provides Model Test Gate for test support.

import XCTest
@testable import Voxt

enum ModelTestGate {
    static let environmentVariable = "VOXT_RUN_MODEL_TESTS"

    @MainActor
    static func configureStorageRoot(for testCase: XCTestCase) {
        guard let path = ProcessInfo.processInfo.environment["VOXT_MODEL_STORAGE_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty else { return }
        let defaults = UserDefaults.standard
        let previousPath = defaults.string(forKey: AppPreferenceKey.modelStorageRootPath)
        let previousBookmark = defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark)
        defaults.set(path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(URL(fileURLWithPath: path, isDirectory: true))
        testCase.addTeardownBlock {
            await MainActor.run {
                let restored = UserDefaults.standard
                if let previousPath { restored.set(previousPath, forKey: AppPreferenceKey.modelStorageRootPath) }
                else { restored.removeObject(forKey: AppPreferenceKey.modelStorageRootPath) }
                if let previousBookmark { restored.set(previousBookmark, forKey: AppPreferenceKey.modelStorageRootBookmark) }
                else { restored.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark) }
                ModelStorageDirectoryManager.resetForTesting()
            }
        }
    }

    @MainActor
    static func waitForASRInstallations(_ manager: MLXModelManager) async throws {
        // Installation reads are asynchronous. Do not mistake an initial unknown
        // state for a missing checkpoint and silently skip all model regressions.
        for model in MLXModelManager.availableModels {
            _ = try await manager.refreshInstallation(repo: model.id)
        }
    }

    static func requireEnabled(_ testDescription: String) throws {
        let rawValue = ProcessInfo.processInfo.environment[environmentVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let enabledValues: Set<String> = ["1", "true", "yes", "on"]

        guard let rawValue, enabledValues.contains(rawValue) else {
            throw XCTSkip(
                "\(testDescription) skipped by default. Set \(environmentVariable)=1 when running model tests explicitly."
            )
        }
    }
}
