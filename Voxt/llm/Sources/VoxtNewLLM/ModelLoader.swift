// ModelLoader.swift
// Voxt's MemoryEfficientModelContainerLoader (Voxt/Core/Models), text models
// only: it mirrors the pinned MLXLM factory flow while loading prequantized
// layers without throwaway graphs (PrequantizedModelLoading.swift).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

private struct TextInputProcessor: UserInputProcessor {
    let tokenizer: any MLXLMCommon.Tokenizer
    let messageGenerator: any MessageGenerator

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext
            )
            return LMInput(tokens: MLXArray(promptTokens))
        } catch MLXLMCommon.TokenizerError.missingChatTemplate {
            let prompt = messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            return LMInput(tokens: MLXArray(tokenizer.encode(text: prompt)))
        }
    }
}

enum ModelLoader {
    /// Loads the MLX text model in directory.
    @concurrent
    static func load(from directory: URL, using tokenizerLoader: any TokenizerLoader) async throws -> ModelContainer {
        let configuration = ResolvedModelConfiguration(directory: directory)
        let configData = try configurationData(for: configuration)
        let baseConfig = try decodeBaseConfiguration(configData, configuration: configuration)

        let model: any LanguageModel
        do {
            model = try await LLMModelFactory.shared.typeRegistry.createModel(
                configuration: configData,
                modelType: baseConfig.modelType
            )
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError("config.json", configuration.name, error)
        }

        async let tokenizerTask = tokenizerLoader.load(from: configuration.tokenizerDirectory)
        try loadWeights(from: configuration.modelDirectory, into: model, perLayerQuantization: baseConfig.perLayerQuantization)
        let tokenizer = try await tokenizerTask
        let modelConfiguration = resolvedModelConfiguration(configuration, baseConfig: baseConfig, model: model)
        let messageGenerator: any MessageGenerator = if let model = model as? any LLMModel {
            model.messageGenerator(tokenizer: tokenizer)
        } else {
            DefaultMessageGenerator()
        }
        let processor = TextInputProcessor(tokenizer: tokenizer, messageGenerator: messageGenerator)
        return ModelContainer(
            context: ModelContext(
                configuration: modelConfiguration,
                model: model,
                processor: processor,
                tokenizer: tokenizer
            )
        )
    }

    private static func configurationData(for configuration: ResolvedModelConfiguration) throws -> Data {
        let url = configuration.modelDirectory.appendingPathComponent("config.json")
        do {
            return try Data(contentsOf: url)
        } catch {
            throw ModelFactoryError.configurationFileError(url.lastPathComponent, configuration.name, error)
        }
    }

    private static func decodeBaseConfiguration(_ data: Data, configuration: ResolvedModelConfiguration) throws -> BaseConfiguration {
        do {
            return try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError("config.json", configuration.name, error)
        }
    }

    private static func resolvedModelConfiguration(
        _ configuration: ResolvedModelConfiguration,
        baseConfig: BaseConfiguration,
        model: any LanguageModel
    ) -> ModelConfiguration {
        var eosTokenIDs = baseConfig.effectiveEOSTokenIds
        var stopStrings = configuration.stopStrings
        let generationConfigURL = configuration.modelDirectory.appendingPathComponent("generation_config.json")
        if let generationData = try? Data(contentsOf: generationConfigURL),
           let generationConfig = try? JSONDecoder.json5().decode(GenerationConfigFile.self, from: generationData)
        {
            if let generationEOS = generationConfig.eosTokenIds?.values {
                eosTokenIDs = Set(generationEOS)
            }
            stopStrings.formUnion(generationConfig.stopStrings)
        }
        // The current LM API declares conventions on each concrete model.
        // Keep explicit configuration overrides ahead of architecture defaults.
        let toolCallFormat = configuration.toolCallFormat ?? model.toolCallFormat
        return ModelConfiguration(
            directory: configuration.modelDirectory,
            defaultPrompt: configuration.defaultPrompt,
            extraEOSTokens: configuration.extraEOSTokens,
            stopStrings: stopStrings,
            eosTokenIds: eosTokenIDs,
            toolCallFormat: toolCallFormat,
            reasoningConfig: configuration.reasoningConfig ?? model.reasoningConfig
        )
    }

    private static func loadWeights(
        from modelDirectory: URL,
        into model: any LanguageModel,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        var weights: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]
        for url in try safetensorURLs(in: modelDirectory) {
            let (fileWeights, fileMetadata) = try loadArraysAndMetadata(url: url)
            weights.merge(fileWeights) { _, new in new }
            if metadata.isEmpty {
                metadata = fileMetadata
            }
        }

        weights = model.sanitize(weights: weights, metadata: metadata)
        if let perLayerQuantization {
            let summary = PrequantizedModelLoading.replaceLayers(in: model, weights: weights) { path in
                perLayerQuantization.quantization(layer: path)?.asTuple
            }
            logLine("MLX prequantized layers installed: direct=\(summary.direct), fallback=\(summary.fallback)")
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        // New LM models materialize derived inference-only state after weights
        // are installed. Forward passes must not perform this mutation lazily.
        try model.prepare()
        try withError {
            eval(model)
        }
    }

    private static func safetensorURLs(in modelDirectory: URL) throws -> [URL] {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let data = try Data(contentsOf: indexURL)
            let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
            // A file the index names may be left out on purpose (an OptiQ
            // model's vision tower, optiq/): the text model's weights are
            // checked whole by update(verify: .all) below.
            return Set(index.weightMap.values).sorted().compactMap { name in
                let url = modelDirectory.appendingPathComponent(name)
                guard FileManager.default.fileExists(atPath: url.path) else {
                    logLine("\(name), in the safetensors index, is not there: skipped")
                    return nil
                }
                return url
            }
        }
        let contents = try FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)
        return contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
