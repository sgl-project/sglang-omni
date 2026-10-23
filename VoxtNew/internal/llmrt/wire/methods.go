package wire

// Methods (docs/llm-protocol.md §4). chat is a job: job.cancel stops it,
// and its notifications name it in "job". The others are control
// requests.
//
// Field conventions, which the fixtures hold both sides to: a Go field
// without omitempty is always on the wire (Swift: a non-optional); a Go
// field with omitempty is absent when zero (Swift: an optional left nil).
const (
	MethodHello     = "llm.hello"    // request; must be first
	MethodShutdown  = "llm.shutdown" // request: cancel the chat, answer, exit 0
	MethodJobCancel = "job.cancel"   // request

	MethodLoad   = "model.load"   // request: load a model (warm-up), unloading another
	MethodUnload = "model.unload" // request: free the model
	MethodStatus = "model.status" // request: the loaded model and the memory
	MethodChat   = "chat"         // request, job: one completion, streamed

	// Notifications, process → app.
	MethodChatDelta = "chat.delta" // the next piece of a chat's answer
	MethodLog       = "log"        // a log line (stderr stays free-form text)
)

// Formats of a model.
const (
	FormatMLX  = "mlx"  // a directory: config.json, safetensors, tokenizer
	FormatGGUF = "gguf" // one .gguf file
)

// Finish reasons of a chat.
const (
	FinishStop   = "stop"
	FinishLength = "length"
)

// HelloParams opens a session; Go sends it first.
type HelloParams struct {
	Protocol int    `json:"protocol"`
	App      string `json:"app"`
	LogLevel string `json:"logLevel,omitempty"` // "debug" | "info" (default)
}

// HelloResult describes the process.
type HelloResult struct {
	Protocol int      `json:"protocol"`
	Version  string   `json:"version"`
	PID      int      `json:"pid"`
	Methods  []string `json:"methods"`
	Device   Device   `json:"device"`
}

// Device is the Mac the process runs on.
type Device struct {
	Name        string `json:"name"`        // "Mac16,13"
	OS          string `json:"os"`          // "macOS 26.7"
	MemoryBytes int64  `json:"memoryBytes"` // physical memory
}

// CancelParams names the job (its request id) to cancel.
type CancelParams struct {
	Job string `json:"job"`
}

// CancelResult says whether a running or queued job was found.
type CancelResult struct {
	Cancelled bool `json:"cancelled"`
}

// Empty is the params/result of methods that take or return nothing.
type Empty struct{}

// LogParams is a structured log line.
type LogParams struct {
	Level   string `json:"level"` // "debug" | "info" | "warning" | "error"
	Message string `json:"message"`
	Job     string `json:"job,omitempty"`
}

// Model is a model on disk.
type Model struct {
	ID     string `json:"id"`     // the catalogue's id ("llm.balanced")
	Format string `json:"format"` // FormatMLX | FormatGGUF
	Path   string `json:"path"`   // the directory (mlx) or the file (gguf)
}

// Memory is the process's memory use, in bytes.
type Memory struct {
	FootprintBytes     int64 `json:"footprintBytes"`     // physical footprint now (Metal buffers included)
	PeakFootprintBytes int64 `json:"peakFootprintBytes"` // its peak over the process's life
	// ResidentBytes is the resident memory now, file-backed pages
	// included: a GGUF model's weights, mapped from its file, count here
	// and not in the footprint.
	ResidentBytes  int64 `json:"residentBytes"`
	MLXActiveBytes int64 `json:"mlxActiveBytes,omitempty"`
	MLXCacheBytes  int64 `json:"mlxCacheBytes,omitempty"`
	MLXPeakBytes   int64 `json:"mlxPeakBytes,omitempty"`
}

// LoadParams loads a model.
type LoadParams struct {
	Model Model `json:"model"`
	// CacheLimitMB bounds MLX's buffer cache (0: MLX's default).
	CacheLimitMB int `json:"cacheLimitMB,omitempty"`
}

// LoadResult is a loaded model.
type LoadResult struct {
	Model  Model   `json:"model"`
	Cached bool    `json:"cached,omitempty"` // it was loaded already
	LoadMs float64 `json:"loadMs"`
	Memory Memory  `json:"memory"`
}

// UnloadResult says whether a model was loaded.
type UnloadResult struct {
	Unloaded bool   `json:"unloaded"`
	Memory   Memory `json:"memory"`
}

// StatusResult is the loaded model (nil for none) and the memory.
type StatusResult struct {
	Model  *Model `json:"model,omitempty"`
	Memory Memory `json:"memory"`
}

// ChatMessage is one turn of a chat.
type ChatMessage struct {
	Role    string `json:"role"` // "system" | "user" | "assistant"
	Content string `json:"content"`
}

// Sampling is how the next token is chosen. Temperature 0 is greedy;
// absent fields are neutral (top-p 1, no top-k, min-p or penalty).
type Sampling struct {
	Temperature           float64 `json:"temperature"`
	TopP                  float64 `json:"topP,omitempty"`
	TopK                  int     `json:"topK,omitempty"`
	MinP                  float64 `json:"minP,omitempty"`
	RepetitionPenalty     float64 `json:"repetitionPenalty,omitempty"`
	RepetitionContextSize int     `json:"repetitionContextSize,omitempty"`
	Seed                  *uint64 `json:"seed,omitempty"`
}

// ChatParams is one completion.
type ChatParams struct {
	Model    Model         `json:"model"`
	Messages []ChatMessage `json:"messages"`
	// MaxTokens 0 evaluates the prompt's shared prefix into the prompt
	// cache and answers nothing (a warm-up).
	MaxTokens int      `json:"maxTokens"`
	Sampling  Sampling `json:"sampling"`
	// PrefillStepSize is MLX's prompt chunk (0: the engine's).
	PrefillStepSize int `json:"prefillStepSize,omitempty"`
	// Template are the chat template's variables (MLX only), such as
	// {"enable_thinking": false}.
	Template map[string]any `json:"template,omitempty"`
}

// ChatResult is a finished completion.
type ChatResult struct {
	Text    string  `json:"text"`
	Finish  string  `json:"finish"` // FinishStop | FinishLength
	Usage   Usage   `json:"usage"`
	Timings Timings `json:"timings"`
	Memory  Memory  `json:"memory"`
}

// Usage counts a chat's tokens.
type Usage struct {
	PromptTokens     int `json:"promptTokens"`
	CompletionTokens int `json:"completionTokens"`
}

// Timings of a chat, in milliseconds (the rate in tokens per second).
type Timings struct {
	LoadMs          float64 `json:"loadMs,omitempty"` // the chat loaded its model first
	PromptMs        float64 `json:"promptMs"`
	FirstTokenMs    float64 `json:"firstTokenMs"`
	TotalMs         float64 `json:"totalMs"`
	TokensPerSecond float64 `json:"tokensPerSecond"`
	// Cache is what the chat did with the prompt cache (MLX models).
	Cache *CacheUse `json:"cache,omitempty"`
}

// CacheUse is a chat's use of the prompt cache: Hit when it started from
// a snapshot of a prompt prefix it had seen; PrefixTokens the shared
// prefix it reused (a hit) or evaluated and kept (a miss), 0 when neither.
type CacheUse struct {
	Hit          bool `json:"hit"`
	PrefixTokens int  `json:"prefixTokens"`
}

// DeltaParams is the next piece of a chat's answer.
type DeltaParams struct {
	Job  string `json:"job"`
	Text string `json:"text"`
}

// ParamsFor returns a new value of the params type of method, or nil for
// an unknown method. ResultFor does the same for results.
func ParamsFor(method string) any {
	switch method {
	case MethodHello:
		return new(HelloParams)
	case MethodShutdown, MethodUnload, MethodStatus:
		return new(Empty)
	case MethodJobCancel:
		return new(CancelParams)
	case MethodLoad:
		return new(LoadParams)
	case MethodChat:
		return new(ChatParams)
	case MethodChatDelta:
		return new(DeltaParams)
	case MethodLog:
		return new(LogParams)
	}
	return nil
}

// ResultFor: see ParamsFor.
func ResultFor(method string) any {
	switch method {
	case MethodHello:
		return new(HelloResult)
	case MethodShutdown:
		return new(Empty)
	case MethodJobCancel:
		return new(CancelResult)
	case MethodLoad:
		return new(LoadResult)
	case MethodUnload:
		return new(UnloadResult)
	case MethodStatus:
		return new(StatusResult)
	case MethodChat:
		return new(ChatResult)
	}
	return nil
}

// Requests lists the protocol's requests (llm.hello's methods), and
// Notifications its notifications.
var (
	Requests      = []string{MethodHello, MethodShutdown, MethodJobCancel, MethodLoad, MethodUnload, MethodStatus, MethodChat}
	Notifications = []string{MethodChatDelta, MethodLog}
)

// IsJob reports whether method is a long-running, cancellable job.
func IsJob(method string) bool { return method == MethodChat }
