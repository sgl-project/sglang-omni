package omni

import (
	"slices"
	"time"
)

// The curated models' stable ids, the catalogue's (internal/models:
// settings.Feature.ASR keeps them); the UI never shows them.
const (
	ASRAccurate = "asr.accurate" // Qwen3-ASR 1.7B 8-bit, the default
	ASRFast     = "asr.fast"     // Qwen3-ASR 0.6B 4-bit
	VADSilero   = "vad.silero"   // Silero VAD v6
)

// Server kinds (qwen3_asr_server --model-kind).
const (
	kindQwen3ASR  = "qwen3_asr"
	kindSileroVAD = "silero_vad"
)

// Spec is one curated model: what to run, and the parameters tuned for it.
type Spec struct {
	ID string
	// Repo and Revision are the Hugging Face checkpoint the parameters
	// were tuned on (sglang_omni_mlx/native/ci/models.json pins its
	// files' SHA-256).
	Repo, Revision string
	// Kind is the server's --model-kind.
	Kind string
	// Languages are the language codes the model knows (lang.Match
	// spellings); a hint outside them is not sent.
	Languages []string
	ASR       *ASRTuning // speech models
	VAD       *VADTuning // Silero
}

// ASRTuning is a speech model's request policy (Voxt's, for Qwen3-ASR).
type ASRTuning struct {
	// TokensPerSecond and ExtraTokens make a Final's budget:
	// ceil(seconds × TokensPerSecond) + ExtraTokens, within
	// [MinTokens, MaxTokens] (Voxt: 28/s covers dense mixed Chinese and
	// English; the end token still stops early).
	TokensPerSecond float64
	ExtraTokens     int
	MinTokens       int
	MaxTokens       int
	// ChunkSeconds is the audio of one request (at most the server's
	// limit, 1200 s); longer recordings are cut near their quietest
	// CutWindow within CutSearch of each chunk's end, and a chunk shorter
	// than MinChunkSeconds is padded with silence. The encoder's memory
	// grows with a request's audio (eval.md "Speech server memory").
	ChunkSeconds    float64
	MinChunkSeconds float64
	CutSearch       time.Duration
	CutWindow       time.Duration
	// ContextTerms and ContextChars bound the dictionary's terms in the
	// context bias (Voxt's 500 and 8000; the fast model 30: more made it
	// slower and less accurate, eval.md #24).
	ContextTerms int
	ContextChars int
	// AudioLayout is the prompt's audio layout ("voxt_swift": one more mel
	// frame and its own audio token count, as Voxt's Swift model built it).
	AudioLayout string
	// The live preview's cadence (server flags): the first decode
	// FirstDecode after the first audible audio, then every
	// DecodeInterval; a segment is closed after MaxSegment.
	FirstDecode    time.Duration
	DecodeInterval time.Duration
	MaxSegment     time.Duration
	// A live session longer than PiecesAfter is recognised in pieces while
	// it records (pieces.go), so that Stop leaves only the tail: a piece
	// is cut in a pause of at least PauseMin, PieceMin to PieceSeconds
	// into the audio not yet recognised (the longest pause wins), or, with
	// no pause by PieceMax, at its quietest CutWindow. 0: never.
	PiecesAfter  time.Duration
	PieceMin     time.Duration
	PieceSeconds time.Duration
	PieceMax     time.Duration
	PauseMin     time.Duration
	// CacheMB bounds MLX's cache of freed buffers in the server
	// (--cache-limit-mb): without it one long Final kept up to 6 GB beside
	// the weights (eval.md "Speech server memory"). 0: MLX's own limit,
	// as with an older server that lacks the flag.
	CacheMB int
}

// VADTuning is Silero's: Threshold is the speech probability that counts
// as speech; the offline detection drops speech shorter than MinSpeech,
// joins pauses shorter than MinSilence and pads each range by SpeechPad
// (Silero's defaults).
type VADTuning struct {
	Threshold  float64
	MinSpeech  time.Duration
	MinSilence time.Duration
	SpeechPad  time.Duration
	// Chunk is the samples of one probability (Silero at 16 kHz: 512).
	Chunk int
}

// qwenLanguages are Qwen3-ASR's 30 languages (Voxt's list).
var qwenLanguages = []string{
	"zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi",
	"ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "tl", "fa", "el", "hu",
	"mk", "ro",
}

// qwenTuning is Voxt's Qwen3-ASR policy, for both sizes.
func qwenTuning() *ASRTuning {
	return &ASRTuning{
		TokensPerSecond: 28, ExtraTokens: 64, MinTokens: 256, MaxTokens: 8192,
		ChunkSeconds: 60, MinChunkSeconds: 1, CutSearch: 5 * time.Second, CutWindow: 100 * time.Millisecond,
		ContextTerms: 500, ContextChars: 8000,
		AudioLayout: "voxt_swift",
		FirstDecode: 100 * time.Millisecond, DecodeInterval: time.Second, MaxSegment: 30 * time.Second,
		PiecesAfter: 60 * time.Second, PieceMin: 20 * time.Second, PieceSeconds: 30 * time.Second, PieceMax: 45 * time.Second,
		PauseMin: 300 * time.Millisecond,
		CacheMB:  512,
	}
}

// fastTuning is qwenTuning with the dictionary's 30 most recently changed
// terms at most.
func fastTuning() *ASRTuning {
	t := qwenTuning()
	t.ContextTerms = 30
	return t
}

// specs are the curated models, the recommended speech model first.
var specs = []Spec{
	{
		ID: ASRAccurate, Repo: "mlx-community/Qwen3-ASR-1.7B-8bit", Revision: "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
		Kind: kindQwen3ASR, Languages: qwenLanguages, ASR: qwenTuning(),
	},
	{
		ID: ASRFast, Repo: "mlx-community/Qwen3-ASR-0.6B-4bit", Revision: "313d850181767edf09f00a9c289becca70e58cd0",
		Kind: kindQwen3ASR, Languages: qwenLanguages, ASR: fastTuning(),
	},
	{
		ID: VADSilero, Repo: "mlx-community/silero-vad-v6", Revision: "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06",
		Kind: kindSileroVAD,
		VAD:  &VADTuning{Threshold: 0.5, MinSpeech: 250 * time.Millisecond, MinSilence: 100 * time.Millisecond, SpeechPad: 30 * time.Millisecond, Chunk: 512},
	},
}

// Models are the curated models: the speech models (recommended first),
// then Silero VAD. Copies.
func Models() []Spec {
	out := make([]Spec, len(specs))
	for i, s := range specs {
		out[i] = s.clone()
	}
	return out
}

// ASRModels are the curated speech models, the recommended one first.
func ASRModels() []Spec {
	var out []Spec
	for _, s := range specs {
		if s.ASR != nil {
			out = append(out, s.clone())
		}
	}
	return out
}

// Lookup finds a curated model by its id.
func Lookup(id string) (Spec, bool) {
	for _, s := range specs {
		if s.ID == id {
			return s.clone(), true
		}
	}
	return Spec{}, false
}

func (s Spec) clone() Spec {
	s.Languages = slices.Clone(s.Languages)
	if s.ASR != nil {
		a := *s.ASR
		s.ASR = &a
	}
	if s.VAD != nil {
		v := *s.VAD
		s.VAD = &v
	}
	return s
}

// launchArgs are the server's flags for the model, after the common ones;
// cacheLimit: the server knows --cache-limit-mb (Runtime.cacheLimitKnown).
func (s Spec) launchArgs(cacheLimit bool) []string {
	args := []string{"--model-kind", s.Kind}
	if a := s.ASR; a != nil {
		args = append(args,
			"--first-decode-ms", itoa(int(a.FirstDecode/time.Millisecond)),
			"--decode-interval-ms", itoa(int(a.DecodeInterval/time.Millisecond)),
			"--max-segment-seconds", ftoa(a.MaxSegment.Seconds()))
		if cacheLimit && a.CacheMB > 0 {
			args = append(args, "--cache-limit-mb", itoa(a.CacheMB))
		}
	}
	return args
}
