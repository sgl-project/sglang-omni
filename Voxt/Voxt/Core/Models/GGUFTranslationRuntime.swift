import Foundation
import Darwin
import LlamaSwift

nonisolated private final class GGUFLlamaBackendLifetime: @unchecked Sendable {
    static let shared = GGUFLlamaBackendLifetime()

    private static let logCallback: ggml_log_callback = { level, text, _ in
        guard level == GGML_LOG_LEVEL_ERROR else { return }
        guard let text else { return }
        fputs(text, stderr)
    }

    private let lock = NSLock()
    private var leaseCount = 0

    func acquire() {
        lock.lock()
        defer { lock.unlock() }
        if leaseCount == 0 {
            llama_log_set(Self.logCallback, nil)
            llama_backend_init()
        }
        leaseCount += 1
    }

    func release() {
        lock.lock()
        defer { lock.unlock() }
        guard leaseCount > 0 else { return }
        leaseCount -= 1
        if leaseCount == 0 {
            llama_backend_free()
        }
    }
}

actor GGUFTranslationRuntime {

    private var loadedModelPath: String?
    private var loadedModel: OpaquePointer?
    private var hasBackendLease = false

    deinit {
        if let loadedModel {
            llama_model_free(loadedModel)
        }
        if hasBackendLease {
            GGUFLlamaBackendLifetime.shared.release()
        }
    }

    func shutdownForApplicationTermination() {
        if let loadedModel {
            llama_model_free(loadedModel)
            self.loadedModel = nil
            loadedModelPath = nil
        }
        if hasBackendLease {
            GGUFLlamaBackendLifetime.shared.release()
            hasBackendLease = false
        }
    }

    func generate(
        instructions: String,
        prompt: String,
        modelURL: URL,
        maxTokens: Int,
        onPartialText: (@Sendable (String) -> Void)?
    ) throws -> String {
        try Task.checkCancellation()
        let startedAt = CFAbsoluteTimeGetCurrent()
        acquireBackendIfNeeded()
        let model = try loadModelIfNeeded(at: modelURL)
        let vocab = llama_model_get_vocab(model)
        let formattedPrompt = try formattedPrompt(
            instructions: instructions,
            prompt: prompt,
            model: model
        )
        let promptTokens = try tokenize(prompt: formattedPrompt, vocab: vocab)
        let modelContextLimit = resolvedModelContextLimit(model)
        let requestedContextSize = max(1024, promptTokens.count + maxTokens + 32)
        let contextSize = min(modelContextLimit, requestedContextSize)
        let batchSize = min(contextSize, max(32, promptTokens.count))

        guard promptTokens.count < contextSize else {
            VoxtLog.llmWarning(
                "GGUF prompt exceeds available context. model=\(modelURL.lastPathComponent), promptTokens=\(promptTokens.count), maxTokens=\(maxTokens), trainedCtx=\(modelContextLimit), requestedCtx=\(requestedContextSize)"
            )
            throw NSError(
                domain: "Voxt.GGUFTranslation",
                code: -7,
                userInfo: [
                    NSLocalizedDescriptionKey: "Translation prompt is too long for the selected GGUF model context window."
                ]
            )
        }

        VoxtLog.llmDebug(
            "GGUF runtime start. model=\(modelURL.lastPathComponent), promptChars=\(formattedPrompt.count), promptTokens=\(promptTokens.count), maxTokens=\(maxTokens), nCtx=\(contextSize), nBatch=\(batchSize), trainedCtx=\(modelContextLimit)"
        )

        var contextParams = llama_context_default_params()
        contextParams.n_ctx = UInt32(contextSize)
        contextParams.n_batch = UInt32(batchSize)
        contextParams.n_threads = Int32(max(1, ProcessInfo.processInfo.activeProcessorCount - 2))
        contextParams.n_threads_batch = Int32(max(1, ProcessInfo.processInfo.activeProcessorCount - 1))

        guard let context = llama_init_from_model(model, contextParams) else {
            throw NSError(
                domain: "Voxt.GGUFTranslation",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create llama.cpp context."]
            )
        }
        defer { llama_free(context) }

        var batch = llama_batch_init(Int32(batchSize), 0, 1)
        defer { llama_batch_free(batch) }

        batch.n_tokens = Int32(promptTokens.count)
        for (index, token) in promptTokens.enumerated() {
            batch.token[index] = token
            batch.pos[index] = Int32(index)
            batch.n_seq_id[index] = 1
            if let seqIDs = batch.seq_id,
               let seqID = seqIDs[index] {
                seqID[0] = 0
            }
            batch.logits[index] = 0
        }
        if batch.n_tokens > 0 {
            batch.logits[Int(batch.n_tokens) - 1] = 1
        }

        let prefillStartedAt = CFAbsoluteTimeGetCurrent()
        guard llama_decode(context, batch) == 0 else {
            throw NSError(
                domain: "Voxt.GGUFTranslation",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Prompt evaluation failed."]
            )
        }
        try Task.checkCancellation()
        let prefillElapsedMs = millisecondsSince(prefillStartedAt)

        let sampler = try createSampler(vocab: vocab)
        defer { llama_sampler_free(sampler) }

        var generated = ""
        var outputAccumulator = GGUFUTF8OutputAccumulator()
        var tokenPosition = Int32(promptTokens.count)
        let eosToken = llama_vocab_eos(vocab)
        let eotToken = llama_vocab_eot(vocab)
        let contextLimit = Int(llama_n_ctx(context))
        var generatedTokenCount = 0
        var firstTokenLatencyMs: Int?
        var stopReason = "tokenLimit"

        let generationLimit = max(0, min(maxTokens, contextSize - promptTokens.count - 1))
        if generationLimit == 0 {
            stopReason = "noGenerationBudget"
        }
        for _ in 0..<generationLimit {
            try Task.checkCancellation()
            if Int(tokenPosition) >= contextLimit - 1 {
                stopReason = "contextLimit"
                break
            }

            let nextToken = llama_sampler_sample(sampler, context, -1)
            if nextToken == eosToken || nextToken == eotToken {
                stopReason = nextToken == eosToken ? "eos" : "eot"
                break
            }
            llama_sampler_accept(sampler, nextToken)
            generatedTokenCount += 1
            if firstTokenLatencyMs == nil {
                firstTokenLatencyMs = millisecondsSince(startedAt)
            }

            let pieceBytes = tokenPieceBytes(for: nextToken, vocab: vocab)
            if !pieceBytes.isEmpty,
               let decoded = outputAccumulator.append(pieceBytes) {
                generated = decoded
                onPartialText?(generated)
            }

            batch.n_tokens = 1
            batch.token[0] = nextToken
            batch.pos[0] = tokenPosition
            batch.n_seq_id[0] = 1
            if let seqIDs = batch.seq_id,
               let seqID = seqIDs[0] {
                seqID[0] = 0
            }
            batch.logits[0] = 1
            tokenPosition += 1

            guard llama_decode(context, batch) == 0 else {
                throw NSError(
                    domain: "Voxt.GGUFTranslation",
                    code: -4,
                    userInfo: [NSLocalizedDescriptionKey: "Token generation failed."]
                )
            }
        }

        let finalizedOutput = outputAccumulator.finalizedText()
        if generated != finalizedOutput {
            generated = finalizedOutput
            if !generated.isEmpty {
                onPartialText?(generated)
            }
        }
        if outputAccumulator.finalizedWithReplacementCharacters {
            VoxtLog.llmDebug(
                "GGUF runtime output contained invalid UTF-8 bytes after token accumulation. model=\(modelURL.lastPathComponent), outputTokens=\(generatedTokenCount)"
            )
        }

        let totalElapsedMs = millisecondsSince(startedAt)
        VoxtLog.llmDebug(
            "GGUF runtime finished. model=\(modelURL.lastPathComponent), outputChars=\(generated.count), outputTokens=\(generatedTokenCount), prefillMs=\(prefillElapsedMs), firstTokenMs=\(firstTokenLatencyMs.map(String.init) ?? "n/a"), totalMs=\(totalElapsedMs), stop=\(stopReason)"
        )
        return generated
    }

    private func acquireBackendIfNeeded() {
        guard !hasBackendLease else { return }
        GGUFLlamaBackendLifetime.shared.acquire()
        hasBackendLease = true
    }

    private func millisecondsSince(_ start: CFAbsoluteTime) -> Int {
        max(Int(((CFAbsoluteTimeGetCurrent() - start) * 1000).rounded()), 0)
    }

    private func resolvedModelContextLimit(_ model: OpaquePointer) -> Int {
        let trainedLimit = Int(llama_model_n_ctx_train(model))
        if trainedLimit > 0 {
            return max(1024, trainedLimit)
        }
        return 4096
    }

    private func formattedPrompt(
        instructions: String,
        prompt: String,
        model: OpaquePointer
    ) throws -> String {
        let fallbackSections = [instructions, prompt]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let fallbackPrompt = fallbackSections.joined(separator: "\n\n")
        guard !fallbackPrompt.isEmpty else { return "" }

        guard let templatePointer = llama_model_chat_template(model, nil) else {
            return fallbackPrompt
        }

        let systemText = instructions.trimmingCharacters(in: .whitespacesAndNewlines)
        let userText = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        var messages = [llama_chat_message]()
        if !systemText.isEmpty {
            messages.append(
                llama_chat_message(
                    role: strdup("system"),
                    content: strdup(systemText)
                )
            )
        }
        if !userText.isEmpty {
            messages.append(
                llama_chat_message(
                    role: strdup("user"),
                    content: strdup(userText)
                )
            )
        }

        defer {
            for message in messages {
                if let role = message.role {
                    free(UnsafeMutableRawPointer(mutating: role))
                }
                if let content = message.content {
                    free(UnsafeMutableRawPointer(mutating: content))
                }
            }
        }

        guard !messages.isEmpty else { return fallbackPrompt }

        var bufferSize = max(2048, fallbackPrompt.utf8.count * 4)
        while bufferSize <= 1_048_576 {
            var buffer = [CChar](repeating: 0, count: bufferSize)
            let renderedLength = llama_chat_apply_template(
                templatePointer,
                messages,
                messages.count,
                true,
                &buffer,
                Int32(buffer.count)
            )
            guard renderedLength > 0 else {
                return fallbackPrompt
            }
            if renderedLength < buffer.count {
                return String(cString: buffer)
            }
            bufferSize = Int(renderedLength) + 1
        }

        return fallbackPrompt
    }

    private func loadModelIfNeeded(at modelURL: URL) throws -> OpaquePointer {
        if let loadedModel,
           loadedModelPath == modelURL.path {
            return loadedModel
        }

        if let loadedModel {
            llama_model_free(loadedModel)
            self.loadedModel = nil
            self.loadedModelPath = nil
        }

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = -1

        guard let model = llama_model_load_from_file(modelURL.path, modelParams) else {
            throw NSError(
                domain: "Voxt.GGUFTranslation",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to load GGUF model from \(modelURL.lastPathComponent)."]
            )
        }

        loadedModel = model
        loadedModelPath = modelURL.path
        return model
    }

    private func tokenize(prompt: String, vocab: OpaquePointer?) throws -> [llama_token] {
        let utf8Count = prompt.utf8.count
        let maxTokenCount = max(utf8Count + 8, 128)
        var tokens = [llama_token](repeating: 0, count: maxTokenCount)
        let tokenCount = llama_tokenize(
            vocab,
            prompt,
            Int32(utf8Count),
            &tokens,
            Int32(maxTokenCount),
            true,
            true
        )
        guard tokenCount > 0 else {
            throw NSError(
                domain: "Voxt.GGUFTranslation",
                code: -5,
                userInfo: [NSLocalizedDescriptionKey: "Prompt tokenization failed."]
            )
        }
        return Array(tokens.prefix(Int(tokenCount)))
    }

    private func createSampler(vocab: OpaquePointer?) throws -> UnsafeMutablePointer<llama_sampler> {
        let params = llama_sampler_chain_default_params()
        guard let sampler = llama_sampler_chain_init(params) else {
            throw NSError(
                domain: "Voxt.GGUFTranslation",
                code: -6,
                userInfo: [NSLocalizedDescriptionKey: "Failed to initialize llama.cpp sampler."]
            )
        }
        let vocabularySize = Int32(llama_vocab_n_tokens(vocab))
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(vocabularySize, 64, 1.05, 0, 0))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(10))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.2))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32(Date().timeIntervalSince1970)))
        return sampler
    }

    private func tokenPieceBytes(for token: llama_token, vocab: OpaquePointer?) -> [UInt8] {
        var buffer = [CChar](repeating: 0, count: 64)
        while true {
            let length = llama_token_to_piece(
                vocab,
                token,
                &buffer,
                Int32(buffer.count),
                0,
                false
            )

            guard length != 0 else { return [] }

            if length > 0, length <= Int32(buffer.count) {
                let bytes = buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }
                return bytes
            }

            let requiredLength = Int(abs(length))
            let nextCapacity = max(buffer.count * 2, requiredLength + 1)
            buffer = [CChar](repeating: 0, count: nextCapacity)
        }
    }
}

nonisolated struct GGUFUTF8OutputAccumulator {
    private var bytes: [UInt8] = []
    private(set) var finalizedWithReplacementCharacters = false

    mutating func append(_ newBytes: [UInt8]) -> String? {
        guard !newBytes.isEmpty else {
            return String(data: Data(bytes), encoding: .utf8)
        }
        bytes.append(contentsOf: newBytes)
        return String(data: Data(bytes), encoding: .utf8)
    }

    mutating func finalizedText() -> String {
        if let decoded = String(data: Data(bytes), encoding: .utf8) {
            return decoded
        }
        finalizedWithReplacementCharacters = true
        return String(decoding: bytes, as: UTF8.self)
    }
}
