import XCTest
@testable import Voxt
import HuggingFace
import MLX
import MLXAudioSTT

@MainActor
class MLXModelManagerTestCase: XCTestCase {
    func withIsolatedModelStorageRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        let defaults = UserDefaults.standard
        let previousPath = defaults.string(forKey: AppPreferenceKey.modelStorageRootPath)
        let previousBookmark = defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark)
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defaults.set(root.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(root)
        defer {
            if let previousPath {
                defaults.set(previousPath, forKey: AppPreferenceKey.modelStorageRootPath)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
            }
            if let previousBookmark {
                defaults.set(previousBookmark, forKey: AppPreferenceKey.modelStorageRootBookmark)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
            }
            ModelStorageDirectoryManager.resetForTesting()
            try? FileManager.default.removeItem(at: root)
        }
        return try await body(root)
    }

    func seedValidMLXModelDirectory(repo: String, root: URL) throws {
        let modelSubdir = repo.replacingOccurrences(of: "/", with: "_")
        let modelDir = root
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: modelDir.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: modelDir.appendingPathComponent("weights.safetensors"))
    }
}
