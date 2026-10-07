import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT

struct MLXLoadedModelBox: @unchecked Sendable {
    let loaded: LoadedASRModel

    nonisolated init(loaded: LoadedASRModel) {
        self.loaded = loaded
    }

    nonisolated init(model: any STTGenerationModel) {
        self.loaded = .mlx(model)
    }
}

nonisolated enum MLXSTTModelLoader {
    static func load(repo: String, directory: URL) async throws -> MLXLoadedModelBox {
        let model: any STTGenerationModel
        // The manager resolves migration aliases before choosing a directory. Never
        // infer an architecture from a repository name or load retired weights.
        guard MLXModelCatalog.availableModels.contains(where: { $0.id == repo }) else {
            throw NSError(
                domain: "MLXModelManager",
                code: 1001,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported local ASR model: \(repo)"]
            )
        }
        switch MLXModelCatalog.capability(for: repo).family {
        case .whisper:
            model = try await WhisperModel.fromDirectory(directory)
        case .senseVoice:
            model = try SenseVoiceModel.fromDirectory(directory)
        case .qwen3ASR:
            model = try await Qwen3ASRMemoryEfficientLoader.load(from: directory)
        case .mossTranscribeDiarize:
            model = try await MossTranscribeDiarizeModel.fromModelDirectory(directory)
        case .cohereTranscribe:
            model = try CohereTranscribeModel.fromDirectory(directory)
        case .parakeet:
            model = try ParakeetModel.fromDirectory(directory)
        case .nemotronASR:
            model = try NemotronASRModel.fromDirectory(directory)
        case .generic:
            throw NSError(
                domain: "MLXModelManager",
                code: 1001,
                userInfo: [NSLocalizedDescriptionKey: "Unsupported local ASR architecture."]
            )
        }

        return MLXLoadedModelBox(model: model)
    }
}
