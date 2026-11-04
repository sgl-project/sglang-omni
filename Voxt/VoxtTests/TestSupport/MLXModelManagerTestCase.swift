import XCTest
@testable import Voxt
import HuggingFace
import MLX

@MainActor
class MLXModelManagerTestCase: XCTestCase {
    func withIsolatedModelStorageRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        let defaults = UserDefaults.standard
        let previousPath = defaults.string(forKey: AppPreferenceKey.modelStorageRootPath)
        let previousBookmark = defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark)
        guard let temporaryDirectory = try? TestDoubles.canonicalTemporaryDirectory() else {
            preconditionFailure("The temporary test directory must exist and resolve canonically.")
        }
        let isolatedRoot = temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defaults.set(isolatedRoot.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(isolatedRoot)
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
            try? FileManager.default.removeItem(at: isolatedRoot)
        }
        return try await body(isolatedRoot)
    }

    func seedValidMLXModelDirectory(repo: String, root: URL) throws {
        let modelSubdir = repo.replacingOccurrences(of: "/", with: "_")
        let modelDir = root
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: modelDir.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: modelDir.appendingPathComponent("weights.safetensors"))
        if repo.lowercased().contains("sensevoice") {
            try Data("[]".utf8).write(to: modelDir.appendingPathComponent("tokens.json"))
            try Data("fixture".utf8).write(to: modelDir.appendingPathComponent("am.mvn"))
        }
    }
}
