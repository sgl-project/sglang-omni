# LLM

Remote and local text-enhancement support for prompts, requests, streaming responses, and output cleanup.

## Responsibilities

- Builds enhancement execution plans and resolves provider/model strategy for rewrite flows.
- Sends remote LLM requests, parses streaming output, handles Codex model metadata, and sanitizes visible text.
- Keeps payload construction, endpoint handling, and response parsing reusable outside UI code.

## Remote runtime layout

| File | Responsibility |
| --- | --- |
| `RemoteLLMRuntimeClient.swift` | Compiled task routing, warmup, enhancement and dictionary-scan entry points |
| `RemoteLLMResponsesExecution.swift` | Responses request execution, streaming and response IDs |
| `RemoteLLMCompletionExecution.swift` | Chat execution, streaming fallback and endpoint retries |
| `RemoteLLMCompletionRequest.swift` | Chat request construction |
| `RemoteLLMLocalProviderPayloads.swift` | Ollama and oMLX payload configuration |
| `RemoteLLMGenerationSettings.swift` | Unified generation settings to provider fields |
| `RemoteLLMRuntimePolicy.swift` | Endpoint security, budgets, partial-delivery throttling and logging |
| `RemoteLLMEndpoints.swift`, `RemoteLLMMessages.swift`, `RemoteLLMStreamingParser.swift` | Shared endpoints, messages/Responses requests and response parsing |

These are extensions of the existing client, not new service layers. Helpers used across files are internal; helpers local to an implementation remain private. Normal task execution goes through `executeCompiledRequest`; `enhance(userPrompt:)` remains used by title/summary and debug flows.

`RewriteAnswerModels.swift` contains answer/conversation values and normalization shared with overlays and history. Delivery snapshots remain in `Core/SessionFinalizeContext.swift`.
