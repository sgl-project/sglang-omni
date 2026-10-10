// Qwen3ASRMemoryEfficientLoader.swift
// Loads already-quantized Qwen3 ASR checkpoints without quantizing throwaway random weights.

import Foundation
import MLX
import MLXAudioSTT
import MLXLMCommon
import MLXNN
import Tokenizers

nonisolated enum Qwen3ASRMemoryEfficientLoader {
    static func load(from modelDirectory: URL) async throws -> Qwen3ASRModel {
        let configData = try Data(
            contentsOf: modelDirectory.appendingPathComponent("config.json")
        )
        let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: configData)
        let model = Qwen3ASRModel(config)

        try generateTokenizerJSONIfMissing(in: modelDirectory)
        model.tokenizer = try await Tokenizers.AutoTokenizer.from(modelFolder: modelDirectory)

        var weights: [String: MLXArray] = [:]
        let safetensorFiles = try FileManager.default
            .contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitizedWeights = Qwen3ASRModel.sanitize(
            weights: weights,
            skipLmHead: config.textConfig.tieWordEmbeddings
        )
        if let perLayerQuantization = config.perLayerQuantization {
            PrequantizedModelLoading.replaceLayers(
                in: model,
                weights: sanitizedWeights
            ) { path in
                guard !path.hasPrefix("audio_tower") else { return nil }
                return perLayerQuantization.quantization(layer: path)?.asTuple
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedWeights),
            verify: .all
        )
        try withError {
            eval(model)
        }
        return model
    }

    /// Qwen checkpoints sometimes provide vocab/merges but omit tokenizer.json.
    /// Keep the upstream compatibility behavior while avoiding its model loader.
    private static func generateTokenizerJSONIfMissing(in modelDirectory: URL) throws {
        let tokenizerJSONURL = modelDirectory.appendingPathComponent("tokenizer.json")
        guard !FileManager.default.fileExists(atPath: tokenizerJSONURL.path) else { return }

        let vocabURL = modelDirectory.appendingPathComponent("vocab.json")
        let mergesURL = modelDirectory.appendingPathComponent("merges.txt")
        let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path),
              FileManager.default.fileExists(atPath: mergesURL.path)
        else { return }

        let vocabData = try Data(contentsOf: vocabURL)
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        let mergesJSON = mergesText.components(separatedBy: "\n")
            .filter { !$0.hasPrefix("#") && !$0.isEmpty }
            .map {
                let escaped = $0
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "\"", with: "\\\"")
                return "\"\(escaped)\""
            }
            .joined(separator: ",")

        var addedTokensJSON = "[]"
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            let configData = try Data(contentsOf: tokenizerConfigURL)
            if let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
               let decoder = config["added_tokens_decoder"] as? [String: Any]
            {
                var tokens: [(Int, [String: Any])] = []
                for (idString, value) in decoder {
                    guard let id = Int(idString), let token = value as? [String: Any] else { continue }
                    tokens.append((id, [
                        "id": id,
                        "content": token["content"] ?? "",
                        "single_word": token["single_word"] ?? false,
                        "lstrip": token["lstrip"] ?? false,
                        "rstrip": token["rstrip"] ?? false,
                        "normalized": token["normalized"] ?? false,
                        "special": token["special"] ?? false,
                    ]))
                }
                tokens.sort { $0.0 < $1.0 }
                let tokenData = try JSONSerialization.data(withJSONObject: tokens.map(\.1))
                addedTokensJSON = String(data: tokenData, encoding: .utf8) ?? "[]"
            }
        }

        let preTokenizerPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        let escapedPattern = preTokenizerPattern
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        let vocabString = String(data: vocabData, encoding: .utf8) ?? "{}"
        let tokenizerJSON = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": \(addedTokensJSON),
          "normalizer": {"type": "NFC"},
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {
                "type": "Split",
                "pattern": {"Regex": "\(escapedPattern)"},
                "behavior": "Isolated",
                "invert": false
              },
              {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false
              }
            ]
          },
          "post_processor": null,
          "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
          },
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": \(vocabString),
            "merges": [\(mergesJSON)]
          }
        }
        """
        try tokenizerJSON.write(to: tokenizerJSONURL, atomically: true, encoding: .utf8)
    }
}
