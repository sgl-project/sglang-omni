# voxtnew-llm

The Swift process that runs VoxtNew's language models, apart from the helper so that a model's memory (gigabytes) and a crash stay out of it:

- **one model at a time**, MLX (mlx-swift-lm, with Voxt's memory-efficient loader for prequantized and OptiQ models) or GGUF (llama.cpp through llama.swift, Voxt's runtime made general);
- **chats streamed** as `chat.delta` notifications, **cancelled** within a token by `job.cancel`, with Voxt's repetition guard;
- **memory reported** with every load and chat (the footprint, its peak, the resident size and MLX's counters).

It speaks [llm protocol 1](../docs/llm-protocol.md) over stdin and stdout; `internal/llmrt` supervises it (started on first use, restarted after a crash, stopped when idle and on quit, stderr to `~/Library/Logs/VoxtNew/llm.log`) and chooses every parameter: the model, its sampling, its token budget, its template variables ([llm-models.md](../docs/llm-models.md)). The process keeps no settings.

## Layout

```
llm/
  Package.swift              Swift 6.2, macOS 15; mlx-swift 0.31.6, mlx-swift-lm @c6446cf, swift-transformers 1.3.4,
                             llama.swift 2.10549.0 (Voxt's pins); Package.resolved is committed
  Sources/VoxtNewLLMKit/     no dependencies, tested with a fake engine:
    Wire.swift, WireTypes.swift   the protocol (mirror of internal/llmrt/wire)
    Server.swift                  routing, hello first, one queue for load/unload/chat, jobs and job.cancel, stdin
    Engine.swift                  the Engine protocol; TextCollector + RepetitionGuard (Voxt's); UTF8Pieces
  Sources/VoxtNewLLM/        the executable:
    main.swift                    stdout taken over for the protocol (stray prints go to stderr)
    Runtime.swift                 the Engine: one model, MLX or GGUF; memory (task_vm_info)
    MLXEngine.swift               load, chat (MLXLMCommon's token loop, a streaming detokenizer), unload
    ModelLoader.swift             Voxt's MemoryEfficientModelContainerLoader, text models only
    PrequantizedModelLoading.swift, LocalTokenizerLoader.swift   Voxt's, as is
    GGUFEngine.swift              llama.cpp: the model's chat template, a context per chat, the sampler chain
  Tests/VoxtNewLLMKitTests/  FixtureTests (llm/Fixtures), ServerTests (a fake engine: streaming, cancel, errors)
  Fixtures/                  golden transcripts, shared with Go
  Support/                   Info.plist (linked into the binary), llm.entitlements
```

## Build and test

On a Mac with Xcode (from Linux through `scripts/mac.sh`):

```sh
scripts/build-llm.sh                  # release build (xcodebuild), signed, into resources/darwin-arm64/llm/
scripts/build-llm.sh --out DIR        # elsewhere: VOXTNEW_LLM=DIR/voxtnew-llm
scripts/build-llm.sh --test           # the Kit's tests (SwiftPM, the Kit alone)
```

SwiftPM on the command line cannot compile MLX's Metal shaders, so the binary is built by `xcodebuild` (into `llm/.build/xcode`; the package checkouts in `llm/.build/xcode-packages`; about 2 GB together, a first build of a few minutes). The script stages `mlx.metallib` (MLX looks for it beside the binary), `llama.framework` (the binary's rpath is `@executable_path`) and `swift-transformers_Hub.bundle` beside `voxtnew-llm`, refuses a build with coverage instrumentation (Xcode's package schemes gather coverage), and checks that the staged binary answers `llm.hello`. `go tool mygo build` ships the folder as `VoxtNew.app/Contents/Resources/llm/`. It is signed ad hoc without the hardened runtime (library validation would refuse an ad hoc `llama.framework`), or with `VOXTNEW_SIGN_IDENTITY` and the hardened runtime; `mygo build` re-signs nested code with a real identity, keeping `Support/llm.entitlements` (empty: Metal needs none).

`swift test` alone would build MLX with SwiftPM too: `--test` builds only the test product.

## Trying it by hand

stdin must stay open while a chat runs (the process exits, cancelling, when it closes):

```sh
M=~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/<rev>
(printf '%s\n' '{"id":"1","method":"llm.hello","params":{"protocol":1,"app":"cli"}}' \
  "{\"id\":\"2\",\"method\":\"chat\",\"params\":{\"model\":{\"id\":\"llm.balanced\",\"format\":\"mlx\",\"path\":\"$M\"},\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in French.\"}],\"maxTokens\":32,\"sampling\":{\"temperature\":0}}}"; sleep 20) \
  | resources/darwin-arm64/llm/voxtnew-llm
```

In the app, `VOXTNEW_DEV_LLM=llm.balanced=<snapshot>` lists a model from any folder as installed (`internal/llmrt/dev.go`).
