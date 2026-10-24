# Models

Model metadata, storage, download, debug, and custom model support shared by transcription and settings.

## Responsibilities

- Tracks local model locations, remote model configuration, and model-specific metadata.
- Supports custom LLM model downloads, validation, and installation-state reporting.
- Provides debug helpers and storage directory rules used by model management UI.

`SharedModelLoadCoordinator<Value>` coalesces current waiters with typed results. Invalidated/cancelled loads remain outstanding until their tasks exit; subsequent `cancelAll()` calls still return their completion barriers. `hasPendingLoad` describes shareable entries, while `hasOutstandingLoad` also covers retiring work and is the idle-reclamation guard.

ASR and Custom LLM managers share one complete shutdown task per instance, so repeated shutdown callers wait for the same download/load/use cleanup. Swift task completion is not a proof of native Metal worker quiescence.

## Runtime and download components

- `LocalModelManagerCatalog`: existing nested state/snapshot values and stateless catalog forwarding; manager-owned Published state and task dictionaries stay private.
- `CustomLLMRequestRuntime`: generation policy, text-only chat inputs, output parsing and diagnostics formatting. Mutable container lifetime and diagnostic publication remain in the manager. Image inputs and screenshot debug payloads have been removed.
- `CustomLLMModelCatalog.requiresVLMFactory`: selects the loading architecture, not an exposed image capability. MLXVLM remains necessary for existing Qwen3-VL, Ministral-3 and Gemma-4 checkpoints even when requests contain only text.
- `LocalTokenizerLoader`: adapter between Tokenizers and MLXLMCommon tokenizer contracts.
- `GGUFTranslationRuntime`: llama backend/model/context/sampler ownership and UTF-8 accumulation, separate from download/install UI state.
- `ResumableDownloadTypes`, `ResumableDownloadAttemptDelegate`, `ResumableModelDownloadSupport`: shared resumable transport, delegate synchronization, sidecar and retry handling for ASR/LLM/GGUF.
- `ModelDownloadSourceSelection.attemptCandidates`: shared retry-source ordering; resumed downloads stay on the saved source.
- `ModelDownloadProgress`: shared display estimates, clamped before integer conversion. ASR validation size comes from the downloaded repo, not whichever model the UI selects; Custom LLM captures its destination before suspending for metadata.

Use compiled requests for enhancement/translation/rewrite. `enhance(userPrompt:repo:)` remains used for titles/debug and model integration coverage; dictionary history scanning keeps its dedicated array-output contract. The retired raw/system-prompt and translate/rewrite overloads are no longer entry points.
