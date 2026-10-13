// ModelSettingsSupport.swift
// Provides Model Settings Support for model settings.

import Foundation

enum ModelVisibilityStore {
    static let hiddenModelPrefix = "model:"

    static func decode(_ rawValue: String) -> Set<String> {
        guard let data = rawValue.data(using: .utf8),
              let values = try? JSONDecoder().decode([String].self, from: data)
        else {
            return []
        }
        return Set(values)
    }

    static func encode(_ values: Set<String>) -> String {
        guard let data = try? JSONEncoder().encode(values.sorted()) else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    static func modelKey(_ id: String) -> String {
        "\(hiddenModelPrefix)\(id)"
    }

    static func isModelHidden(_ id: String, in hiddenIDs: Set<String>) -> Bool {
        hiddenIDs.contains(modelKey(id))
    }
}

enum LocalASRConfigurationTarget: Equatable, Identifiable {
    case mlx(repo: String)

    var id: String {
        switch self {
        case .mlx(let repo):
            return "mlx:\(repo)"
        }
    }
}

enum LocalModelRemovalTarget: Equatable, Identifiable {
    case mlx(repo: String)
    case customLLM(repo: String)
    case ggufTranslation(modelID: GGUFTranslationModelID)

    var id: String {
        switch self {
        case .mlx(let repo):
            return "mlx:\(MLXModelManager.canonicalModelRepo(repo))"
        case .customLLM(let repo):
            return "custom-llm:\(CustomLLMModelManager.canonicalModelRepo(repo))"
        case .ggufTranslation(let modelID):
            return "gguf-translation:\(modelID.rawValue)"
        }
    }
}
