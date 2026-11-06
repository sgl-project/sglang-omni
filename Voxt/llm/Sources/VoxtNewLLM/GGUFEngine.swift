import Darwin
import Foundation
import LlamaSwift
import VoxtNewLLMKit

/// A GGUF model on llama.cpp (llama.swift): Voxt's GGUFTranslationRuntime
/// made general. The prompt is the model's own chat template applied to
/// the messages (llama_chat_apply_template); a context sized for each
/// chat; the sampler chain from the request's sampling.
final class GGUFEngine: @unchecked Sendable {
    private var model: OpaquePointer?
    private var backendReady = false

    private static let logCallback: ggml_log_callback = { level, text, _ in
        guard level == GGML_LOG_LEVEL_ERROR, let text else { return }
        fputs(text, stderr)
    }

    var isLoaded: Bool { model != nil }

    deinit { unload() }

    func load(_ spec: ModelSpec) throws {
        guard FileManager.default.fileExists(atPath: spec.path) else {
            throw RPCError(Code.loadFailed, "No GGUF file at \(spec.path)")
        }
        if !backendReady {
            llama_log_set(Self.logCallback, nil)
            llama_backend_init()
            backendReady = true
        }
        var params = llama_model_default_params()
        params.n_gpu_layers = -1
        guard let m = llama_model_load_from_file(spec.path, params) else {
            throw RPCError(Code.loadFailed, "llama.cpp could not load \(URL(fileURLWithPath: spec.path).lastPathComponent)")
        }
        model = m
    }

    func unload() {
        if let model {
            llama_model_free(model)
            self.model = nil
        }
    }

    func chat(_ p: ChatParams, onDelta: @escaping @Sendable (String) -> Void) throws -> ChatOutcome {
        guard let model else { throw RPCError(Code.internal, "No GGUF model is loaded") }
        // Each chat has a context of its own and Hy-MT2's prompts have no
        // shared prefix to speak of: there is nothing to warm.
        guard p.maxTokens > 0 else {
            return ChatOutcome(text: "", finish: Finish.length, usage: Usage(promptTokens: 0, completionTokens: 0),
                               promptMs: 0, firstTokenMs: 0, tokensPerSecond: 0)
        }
        let started = monotonicMs()
        let vocab = llama_model_get_vocab(model)
        let prompt = Self.formattedPrompt(p.messages, model: model)
        let tokens = try Self.tokenize(prompt, vocab: vocab)
        let trained = Int(llama_model_n_ctx_train(model))
        let limit = trained > 0 ? max(1024, trained) : 4096
        let contextSize = min(limit, max(1024, tokens.count + p.maxTokens + 32))
        guard tokens.count < contextSize else {
            throw RPCError(Code.contextLength, "The prompt (\(tokens.count) tokens) does not fit the model's context (\(limit))")
        }
        let batchSize = min(contextSize, max(32, tokens.count))
        var cp = llama_context_default_params()
        cp.n_ctx = UInt32(contextSize)
        cp.n_batch = UInt32(batchSize)
        cp.n_threads = Int32(max(1, ProcessInfo.processInfo.activeProcessorCount - 2))
        cp.n_threads_batch = Int32(max(1, ProcessInfo.processInfo.activeProcessorCount - 1))
        guard let context = llama_init_from_model(model, cp) else {
            throw RPCError(Code.internal, "llama.cpp could not make a context")
        }
        defer { llama_free(context) }
        var batch = llama_batch_init(Int32(batchSize), 0, 1)
        defer { llama_batch_free(batch) }

        for (i, t) in tokens.enumerated() {
            Self.set(&batch, i, token: t, pos: Int32(i), logits: i == tokens.count - 1)
        }
        batch.n_tokens = Int32(tokens.count)
        guard llama_decode(context, batch) == 0 else { throw RPCError(Code.internal, "llama.cpp could not evaluate the prompt") }
        let promptMs = monotonicMs() - started
        try Task.checkCancellation()

        let sampler = Self.sampler(p.sampling, vocab: vocab)
        defer { llama_sampler_free(sampler) }
        var collector = TextCollector()
        var pieces = UTF8Pieces()
        var firstMs: Double?
        var generated = 0
        var finish = Finish.length
        var position = Int32(tokens.count)
        let budget = max(0, min(p.maxTokens, contextSize - tokens.count - 1))
        let genStart = monotonicMs()
        loop: for _ in 0..<budget {
            try Task.checkCancellation()
            let next = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, next) {
                finish = Finish.stop
                break
            }
            llama_sampler_accept(sampler, next)
            generated += 1
            let text = pieces.add(Self.piece(next, vocab: vocab))
            if !text.isEmpty {
                if firstMs == nil { firstMs = monotonicMs() - started }
                guard collector.add(text) else {
                    finish = Finish.stop
                    break loop
                }
                onDelta(text)
            }
            batch.n_tokens = 0
            Self.set(&batch, 0, token: next, pos: position, logits: true)
            batch.n_tokens = 1
            position += 1
            guard llama_decode(context, batch) == 0 else { throw RPCError(Code.internal, "llama.cpp could not generate") }
        }
        let rest = pieces.flush()
        if !rest.isEmpty, !collector.looped, collector.add(rest) {
            onDelta(rest)
        }
        let genSeconds = (monotonicMs() - genStart) / 1000
        return ChatOutcome(
            text: collector.text, finish: finish,
            usage: Usage(promptTokens: tokens.count, completionTokens: generated),
            promptMs: promptMs, firstTokenMs: firstMs ?? monotonicMs() - started,
            tokensPerSecond: genSeconds > 0 ? Double(max(0, generated - 1)) / genSeconds : 0
        )
    }

    private static func set(_ batch: inout llama_batch, _ i: Int, token: llama_token, pos: Int32, logits: Bool) {
        batch.token[i] = token
        batch.pos[i] = pos
        batch.n_seq_id[i] = 1
        if let ids = batch.seq_id, let id = ids[i] {
            id[0] = 0
        }
        batch.logits[i] = logits ? 1 : 0
    }

    /// The sampler chain (Voxt's order: penalties, top-k, top-p, min-p,
    /// temperature, then the draw; greedy at temperature 0).
    private static func sampler(_ s: Sampling, vocab: OpaquePointer?) -> UnsafeMutablePointer<llama_sampler> {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        if let penalty = s.repetitionPenalty, penalty != 1 {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), Int32(s.repetitionContextSize ?? 64), Float(penalty), 0, 0))
        }
        if s.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
            return chain
        }
        if let k = s.topK, k > 0 { llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(k))) }
        if let p = s.topP, p > 0, p < 1 { llama_sampler_chain_add(chain, llama_sampler_init_top_p(Float(p), 1)) }
        if let m = s.minP, m > 0 { llama_sampler_chain_add(chain, llama_sampler_init_min_p(Float(m), 1)) }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(Float(s.temperature)))
        let seed = s.seed.map { UInt32(truncatingIfNeeded: $0) } ?? UInt32(truncatingIfNeeded: UInt64(Date().timeIntervalSince1970 * 1000))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
        return chain
    }

    /// The messages through the model's chat template; without one, the
    /// contents one after another.
    private static func formattedPrompt(_ messages: [ChatMessage], model: OpaquePointer) -> String {
        let fallback = messages.map(\.content).joined(separator: "\n\n")
        guard let template = llama_model_chat_template(model, nil) else { return fallback }
        let roles = messages.map { strdup($0.role) }
        let contents = messages.map { strdup($0.content) }
        defer {
            roles.forEach { free($0) }
            contents.forEach { free($0) }
        }
        let chat = zip(roles, contents).map { llama_chat_message(role: UnsafePointer($0), content: UnsafePointer($1)) }
        var size = max(2048, fallback.utf8.count * 4)
        while size <= 4 << 20 {
            var buffer = [CChar](repeating: 0, count: size)
            let n = llama_chat_apply_template(template, chat, chat.count, true, &buffer, Int32(buffer.count))
            guard n > 0 else { return fallback }
            if Int(n) < buffer.count { return String(cString: buffer) }
            size = Int(n) + 1
        }
        return fallback
    }

    private static func tokenize(_ prompt: String, vocab: OpaquePointer?) throws -> [llama_token] {
        let count = prompt.utf8.count
        var tokens = [llama_token](repeating: 0, count: max(count + 8, 128))
        let n = llama_tokenize(vocab, prompt, Int32(count), &tokens, Int32(tokens.count), true, true)
        guard n > 0 else { throw RPCError(Code.internal, "llama.cpp could not tokenize the prompt") }
        return Array(tokens.prefix(Int(n)))
    }

    private static func piece(_ token: llama_token, vocab: OpaquePointer?) -> [UInt8] {
        var buffer = [CChar](repeating: 0, count: 64)
        while true {
            let n = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false)
            if n == 0 { return [] }
            if n > 0, Int(n) <= buffer.count {
                return buffer.prefix(Int(n)).map { UInt8(bitPattern: $0) }
            }
            buffer = [CChar](repeating: 0, count: max(buffer.count * 2, Int(abs(n)) + 1))
        }
    }
}
