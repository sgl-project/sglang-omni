package omni

import (
	"maps"
	"slices"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
)

// The curated models' stable ids, the catalogue's (internal/models:
// settings.Feature.ASR keeps them); the UI never shows them.
const (
	ASRAccurate = "asr.accurate" // Qwen3-ASR 1.7B 8-bit, the default
	ASRFast     = "asr.fast"     // Qwen3-ASR 0.6B 4-bit
	VADSilero   = "vad.silero"   // Silero VAD v6
	// Voxt 1.x's other local speech models.
	ASRQwen6bit     = "asr.qwen3-6bit"    // Qwen3-ASR 1.7B 6-bit
	ASRWhisperTurbo = "asr.whisper-turbo" // Whisper Large v3 Turbo
	ASRWhisperLarge = "asr.whisper-large" // Whisper Large v3
	ASRWhisperSmall = "asr.whisper-small" // Whisper Small
	ASRParakeet     = "asr.parakeet"      // Parakeet TDT 0.6B v3
	ASRSenseVoice   = "asr.sensevoice"    // SenseVoice Small
	ASRCohere       = "asr.cohere"        // Cohere Transcribe 03-2026
	ASRNemotron     = "asr.nemotron"      // Nemotron 3.5 ASR streaming 0.6B 8-bit
	ASRMoss         = "asr.moss"          // MOSS-Transcribe-Diarize
)

// Server kinds (--model-kind): qwen3_asr_server serves the first two; the
// others have a binary each.
const (
	kindQwen3ASR   = "qwen3_asr"
	kindSileroVAD  = "silero_vad"
	kindWhisper    = "whisper"
	kindParakeet   = "parakeet"
	kindSenseVoice = "sensevoice"
	kindCohere     = "cohere_transcribe"
	kindNemotron   = "nemotron_asr"
	kindMoss       = "moss_transcribe_diarize"
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
	// Binary is the server's file name beside Config.Binary ("": that
	// one, qwen3_asr_server). Silero runs as silero_vad_server, a copy of
	// it, so that Activity Monitor tells the voice detector from the
	// speech model; the other speech models have servers of their own.
	Binary string
}

// servedByQwen says whether qwen3_asr_server serves kind, so that a
// runtime without Binary may run it there.
func servedByQwen(kind string) bool { return kind == kindQwen3ASR || kind == kindSileroVAD }

// ASRTuning is a speech model's request policy (Voxt's, for Qwen3-ASR).
type ASRTuning struct {
	// TokensPerSecond and ExtraTokens make a Final's budget:
	// ceil(seconds × TokensPerSecond) + ExtraTokens, within
	// [MinTokens, MaxTokens] (Voxt: 28/s covers dense mixed Chinese and
	// English; the end token still stops early).
	// TokensPerSecond 0: each request's budget is MaxTokens (0: the
	// server's own), not shared by the chunks.
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
	// slower and less accurate, eval.md #24). ContextTerms 0: the model
	// takes no context (neither the terms nor the speech context).
	ContextTerms int
	ContextChars int
	// AudioLayout is the prompt's audio layout ("voxt_swift": one more mel
	// frame and its own audio token count, as Voxt's Swift model built it).
	AudioLayout string
	// Realtime: the server's /v1/realtime shows the text while the user
	// speaks. Without it the live text is a Final request over the audio
	// not yet in a piece, every DecodeInterval of new audio (preview.go).
	Realtime bool
	// The live preview's cadence (Qwen3-ASR's server flags): the first
	// decode FirstDecode after the first audible audio, then every
	// DecodeInterval; a segment is closed after MaxSegment.
	FirstDecode    time.Duration
	DecodeInterval time.Duration
	MaxSegment     time.Duration
	// HintKeys are the server's spellings of Languages where they differ
	// (Nemotron's "zh-CN" for "zh").
	HintKeys map[string]string
	// Instruction is the prompt of every request (and live session) of a
	// model that takes no context: MOSS's task, plain text without its
	// timestamps and speaker labels (the native CI's request).
	Instruction string
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
	// (--cache-limit-mb): with no cap one long Final kept up to 6 GB beside
	// the weights (eval.md "Speech server memory"). 0: the flag is not
	// passed, so the server's default applies (2 GiB; no cap on an older
	// server, which also lacks the flag).
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
		Realtime:    true, FirstDecode: 100 * time.Millisecond, DecodeInterval: time.Second, MaxSegment: 30 * time.Second,
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

// batchTuning is the policy of a model without a context or a realtime
// API (Voxt 1.x's batch preview): requests of at most chunk seconds, each
// with at most maxTokens (0: the server's budget), and the pieces sooner,
// so that each preview recognises at most PieceMax of audio.
func batchTuning(chunk float64, maxTokens int) *ASRTuning {
	return &ASRTuning{
		MaxTokens:    maxTokens,
		ChunkSeconds: chunk, MinChunkSeconds: 1, CutSearch: 5 * time.Second, CutWindow: 100 * time.Millisecond,
		DecodeInterval: time.Second,
		PiecesAfter:    20 * time.Second, PieceMin: 8 * time.Second, PieceSeconds: 15 * time.Second, PieceMax: 25 * time.Second,
		PauseMin: 300 * time.Millisecond,
	}
}

// streamTuning is batchTuning for a model with a realtime API of its own
// (Nemotron, MOSS): the live text is the socket's, the pieces as Qwen's.
func streamTuning(chunk float64) *ASRTuning {
	t := batchTuning(chunk, 0)
	t.Realtime = true
	t.PiecesAfter, t.PieceMin, t.PieceSeconds, t.PieceMax = 60*time.Second, 20*time.Second, 30*time.Second, 45*time.Second
	return t
}

// Voxt 1.x's language lists of its other models.
var (
	europeanLanguages = []string{
		"bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv",
		"lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
	}
	senseVoiceLanguages = []string{"zh", "en", "yue", "ja", "ko"}
	cohereLanguages     = []string{"zh", "en", "ja", "ko", "vi", "ar", "el", "pl", "nl", "pt", "it", "es", "de", "fr"}
	nemotronLanguages   = []string{
		"en", "es", "fr", "it", "pt", "nl", "de", "tr", "ru", "ar", "hi", "ja", "ko", "vi",
		"uk", "pl", "sv", "cs", "no", "da", "bg", "fi", "hr", "sk", "zh", "hu", "ro", "et",
	}
)

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
		ID: ASRQwen6bit, Repo: "mlx-community/Qwen3-ASR-1.7B-6bit", Revision: "edd077a475c4da058e25b6e6ec1199115ca1be2b",
		Kind: kindQwen3ASR, Languages: qwenLanguages, ASR: qwenTuning(),
	},
	{
		ID: ASRWhisperTurbo, Repo: "mlx-community/whisper-large-v3-turbo", Revision: "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
		Kind: kindWhisper, Binary: "whisper_server", Languages: lang.WhisperCodes(), ASR: batchTuning(30, 1024),
	},
	{
		ID: ASRWhisperLarge, Repo: "mlx-community/whisper-large-v3-mlx", Revision: "49e6aa286ad60c14352c404340ded53710378a11",
		Kind: kindWhisper, Binary: "whisper_server", Languages: lang.WhisperCodes(), ASR: batchTuning(30, 1024),
	},
	{
		ID: ASRWhisperSmall, Repo: "mlx-community/whisper-small-mlx", Revision: "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
		Kind: kindWhisper, Binary: "whisper_server", Languages: lang.WhisperCodes(), ASR: batchTuning(30, 1024),
	},
	{
		ID: ASRParakeet, Repo: "mlx-community/parakeet-tdt-0.6b-v3", Revision: "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
		Kind: kindParakeet, Binary: "parakeet_server", Languages: europeanLanguages, ASR: batchTuning(60, 0),
	},
	{
		ID: ASRSenseVoice, Repo: "mlx-community/SenseVoiceSmall", Revision: "8ddd966bd96243cff196422f81f0c5d955814792",
		Kind: kindSenseVoice, Binary: "sensevoice_server", Languages: senseVoiceLanguages, ASR: batchTuning(30, 0),
	},
	{
		ID: ASRCohere, Repo: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16", Revision: "06bbe0d853a93cbbf7c1f40cffccd4c5481459ef",
		Kind: kindCohere, Binary: "cohere_transcribe_server", Languages: cohereLanguages, ASR: batchTuning(30, 1024),
	},
	{
		ID: ASRNemotron, Repo: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit", Revision: "7279359e4481b5e9e185a318bd618e429c6d86cd",
		Kind: kindNemotron, Binary: "nemotron_asr_server", Languages: nemotronLanguages, ASR: func() *ASRTuning {
			t := streamTuning(60)
			t.HintKeys = map[string]string{"zh": "zh-CN", "ja": "ja-JP", "vi": "vi-VN"}
			return t
		}(),
	},
	{
		ID: ASRMoss, Repo: "OpenMOSS-Team/MOSS-Transcribe-Diarize", Revision: "704aa4a9c304e8520be88901e0d1960158ef5b15",
		Kind: kindMoss, Binary: "moss_transcribe_diarize_server", Languages: []string{"zh", "en"}, ASR: func() *ASRTuning {
			t := streamTuning(60)
			t.Instruction = "Transcribe the audio as plain text without timestamps or speaker labels."
			return t
		}(),
	},
	{
		ID: VADSilero, Repo: "mlx-community/silero-vad-v6", Revision: "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06",
		Kind: kindSileroVAD, Binary: "silero_vad_server",
		VAD: &VADTuning{Threshold: 0.5, MinSpeech: 250 * time.Millisecond, MinSilence: 100 * time.Millisecond, SpeechPad: 30 * time.Millisecond, Chunk: 512},
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
		a.HintKeys = maps.Clone(a.HintKeys)
		s.ASR = &a
	}
	if s.VAD != nil {
		v := *s.VAD
		s.VAD = &v
	}
	return s
}

// launchArgs are the server's flags for the model, after the common ones:
// Qwen3-ASR's live cadence and cache limit (the other servers refuse flags
// they lack); cacheLimit: the server knows --cache-limit-mb
// (Runtime.cacheLimitKnown).
func (s Spec) launchArgs(cacheLimit bool) []string {
	args := []string{"--model-kind", s.Kind}
	if a := s.ASR; a != nil && s.Kind == kindQwen3ASR {
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
