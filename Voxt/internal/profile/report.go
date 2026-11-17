package profile

import "time"

// Optional makes an unavailable measurement explicit instead of encoding
// it as a plausible zero.
type Optional[T any] struct {
	Value             *T     `json:"value"`
	UnavailableReason string `json:"unavailable_reason,omitempty"`
}

func Available[T any](value T) Optional[T] { return Optional[T]{Value: &value} }

func Unavailable[T any](reason string) Optional[T] {
	return Optional[T]{UnavailableReason: reason}
}

type OptionalFloat64 = Optional[float64]
type OptionalInt64 = Optional[int64]
type OptionalString = Optional[string]

// Stage is one completed interval. ASR chunks and language-model requests
// have an Index in their own sequence, in execution order.
// DraftReport is one ASR stage's speculative decoding counters.
type DraftReport struct {
	VerifyRounds int     `json:"verify_rounds"`
	Proposed     int     `json:"proposed"`
	Accepted     int     `json:"accepted"`
	AcceptRate   float64 `json:"accept_rate"`
}

type Stage struct {
	Name          string           `json:"name"`
	Track         string           `json:"track,omitempty"`
	Index         int              `json:"index"`
	WallMS        float64          `json:"wall_ms"`
	AudioMS       *int64           `json:"audio_ms,omitempty"`
	RTF           *float64         `json:"rtf,omitempty"`
	QueueAcquire  *OptionalFloat64 `json:"queue_acquire,omitempty"`
	ResultLatency *OptionalFloat64 `json:"result_latency,omitempty"`
	ASRFirstToken *OptionalFloat64 `json:"asr_first_token,omitempty"`
	Draft         *DraftReport     `json:"draft,omitempty"`
	Text          string           `json:"text,omitempty"`
	TextSHA256    string           `json:"text_sha256,omitempty"`
	Attempts      int              `json:"attempts,omitempty"`
	Segments      []SegmentTiming  `json:"segments,omitempty"`
	Warning       string           `json:"warning,omitempty"`
	Error         string           `json:"error,omitempty"`
}

// SegmentTiming records text returned by MTD without pretending that the
// batch API exposes an independent inference time for each returned segment.
type SegmentTiming struct {
	StartMS           int64           `json:"start_ms"`
	EndMS             int64           `json:"end_ms"`
	Speaker           string          `json:"speaker,omitempty"`
	Text              string          `json:"text"`
	TextSHA256        string          `json:"text_sha256"`
	InferenceDuration OptionalFloat64 `json:"inference_duration"`
}

type TextArtifact struct {
	Text   string `json:"text"`
	SHA256 string `json:"sha256"`
}

type Score struct {
	ReferenceTokens int     `json:"reference_tokens"`
	Substitutions   int     `json:"substitutions"`
	Deletions       int     `json:"deletions"`
	Insertions      int     `json:"insertions"`
	MixedCERWER     float64 `json:"mixed_cer_wer"`
}

type Evaluation struct {
	ASR   Score `json:"asr"`
	Final Score `json:"final"`
}

type MemorySample struct {
	Pressure         OptionalString `json:"pressure"`
	AvailablePercent OptionalInt64  `json:"available_percent"`
	SwapUsedBytes    OptionalInt64  `json:"swap_used_bytes"`
	SwapTotalBytes   OptionalInt64  `json:"swap_total_bytes"`
	Swapins          OptionalInt64  `json:"swapins"`
	Swapouts         OptionalInt64  `json:"swapouts"`
}

type HostSample struct {
	Thermal     OptionalString `json:"thermal"`
	PowerSource OptionalString `json:"power_source"`
}

type ResourceReport struct {
	PeakRSSBytes          OptionalInt64   `json:"peak_rss_bytes"`
	PeakPhysicalFootprint OptionalInt64   `json:"peak_physical_footprint_bytes"`
	GPUMemoryBytes        OptionalInt64   `json:"gpu_memory_bytes"`
	CPUUserMS             OptionalFloat64 `json:"cpu_user_ms"`
	CPUSystemMS           OptionalFloat64 `json:"cpu_system_ms"`
	GPUTimeMS             OptionalFloat64 `json:"gpu_time_ms"`
	EnergyJoules          OptionalFloat64 `json:"energy_joules"`
	MemoryStart           MemorySample    `json:"memory_start"`
	MemoryEnd             MemorySample    `json:"memory_end"`
	SwapinsDelta          OptionalInt64   `json:"swapins_delta"`
	SwapoutsDelta         OptionalInt64   `json:"swapouts_delta"`
	HostStart             HostSample      `json:"host_start"`
	HostEnd               HostSample      `json:"host_end"`
}

type TracingReport struct {
	Signposts                  string `json:"signposts"`
	SignpostsUnavailableReason string `json:"signposts_unavailable_reason,omitempty"`
	CPUProfile                 string `json:"cpu_profile"`
	HeapProfile                string `json:"heap_profile"`
	NativeLog                  string `json:"native_log"`
}

// Report is the result.json written for every validly configured run.
type Report struct {
	SchemaVersion   int                  `json:"schema_version"`
	Name            string               `json:"name,omitempty"`
	Status          string               `json:"status"`
	StartedAt       time.Time            `json:"started_at"`
	EndedAt         time.Time            `json:"ended_at"`
	TotalWallMS     float64              `json:"total_wall_ms"`
	AudioDurationMS int64                `json:"audio_duration_ms"`
	TotalRTF        OptionalFloat64      `json:"total_rtf"`
	FirstFinalText  OptionalFloat64      `json:"first_final_text"`
	Config          Config               `json:"config"`
	Stages          []Stage              `json:"stages"`
	ASRTranscript   TextArtifact         `json:"asr_transcript"`
	Transcript      TextArtifact         `json:"transcript"`
	Summary         *TextArtifact        `json:"summary,omitempty"`
	Evaluation      Optional[Evaluation] `json:"evaluation"`
	Resources       ResourceReport       `json:"resources"`
	Tracing         TracingReport        `json:"tracing"`
	Warnings        []string             `json:"warnings"`
	Errors          []string             `json:"errors"`
	redactor        *Redactor
}
