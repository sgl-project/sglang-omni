import Foundation

nonisolated enum ModelWeightFileValidation {
    /// A partial multi-shard download is not installed just because one shard exists.
    static func hasCompleteIndex(in directory: URL) -> Bool {
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: index.path) else { return true }
        guard let data = try? Data(contentsOf: index),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = root["weight_map"] as? [String: String], !map.isEmpty else { return false }
        // Validate index paths, while allowing ordinary Hugging Face cache links
        // whose target blobs live outside the snapshot directory.
        let base = directory.standardizedFileURL.path + "/"
        return Set(map.values).allSatisfy { name in
            guard !name.hasPrefix("/") else { return false }
            let file = directory.appendingPathComponent(name).standardizedFileURL
            guard file.path.hasPrefix(base), file.pathExtension == "safetensors",
                  let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]) else { return false }
            return values.isRegularFile == true && (values.fileSize ?? 0) > 0
        }
    }
}
