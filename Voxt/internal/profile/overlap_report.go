package profile

import "time"

var overlapLimitations = []string{
	"simulated_llm=true: local deterministic structured responses with injected latency; no remote quality, remote latency, reasoning, token, or first-token measurement",
	"only ASR uses the configured real model; LLM totals measure simulated calls, not a production language model",
	"recording is PCM replay on the audio clock into the experiment scheduler, not a production streaming session or UI",
	"one_shot versus serial changes both ASR and cleanup partitioning; the named deltas are arithmetic comparisons, not isolated causal effects",
	"meeting mode uses cleanup, summary map, and hierarchical final reduce without diarization or a production prompt",
	"backlog counts planned segment jobs incomplete at stop; final summary reduce and requests not yet created are not separate backlog items",
	"methods run in a fixed order on one warm ASR client, so model cache and thermal state can carry over",
}

type OverlapTiming struct {
	ASRTotalMS       OptionalFloat64 `json:"asr_total_ms"`
	ASRAfterStopMS   OptionalFloat64 `json:"asr_after_stop_ms"`
	LLMTotalMS       OptionalFloat64 `json:"llm_total_ms"`
	LLMAfterStopMS   OptionalFloat64 `json:"llm_after_stop_ms"`
	ASRCalls         int             `json:"asr_calls"`
	LLMCalls         int             `json:"llm_calls"`
	LLMSegmentsTotal int             `json:"llm_segments_total"`
}

type OverlapBacklog struct {
	ASRSegments OptionalInt64 `json:"asr_segments"`
	LLMSegments OptionalInt64 `json:"llm_segments"`
}

type OverlapStage struct {
	Stage
	StartedMS             float64         `json:"started_ms"`
	AfterStopMS           OptionalFloat64 `json:"after_stop_ms"`
	InputSHA256           string          `json:"input_sha256,omitempty"`
	PreviousContextSHA256 string          `json:"previous_context_sha256,omitempty"`
}

type OverlapStructure struct {
	Cleanup       []SimulatedPart `json:"cleanup"`
	SummaryMap    []string        `json:"summary_map"`
	SummaryReduce []string        `json:"summary_reduce"`
	FinalSummary  string          `json:"final_summary,omitempty"`
}

type SimulatedPart struct {
	Index int    `json:"i"`
	Text  string `json:"text"`
}

type OverlapStructureArtifact struct {
	Document          *OverlapStructure `json:"document"`
	SHA256            *string           `json:"sha256"`
	UnavailableReason string            `json:"unavailable_reason,omitempty"`
}

type OverlapComparison struct {
	ASRTextEqual             bool          `json:"asr_text_equal"`
	TextEqual                bool          `json:"text_equal"`
	TextSHA256               string        `json:"text_sha256"`
	OneShotTextSHA256        string        `json:"one_shot_text_sha256"`
	CommonPrefixRunes        int           `json:"common_prefix_runes"`
	CommonSuffixRunes        int           `json:"common_suffix_runes"`
	FirstTextDifferenceRune  OptionalInt64 `json:"first_text_difference_rune"`
	StructureEqual           bool          `json:"structure_equal"`
	StructureSHA256          string        `json:"structure_sha256"`
	OneShotStructureSHA256   string        `json:"one_shot_structure_sha256"`
	CleanupPartCountDelta    int           `json:"cleanup_part_count_delta"`
	SummaryMapCountDelta     int           `json:"summary_map_count_delta"`
	SummaryReduceCountDelta  int           `json:"summary_reduce_count_delta"`
	SummaryEqual             *bool         `json:"summary_equal"`
	SummaryUnavailableReason string        `json:"summary_unavailable_reason,omitempty"`
}

type OverlapSchedulingEffects struct {
	ASRChunkingAfterStopSavedMS OptionalFloat64 `json:"asr_chunking_after_stop_saved_ms"`
	LLMChunkingAfterStopSavedMS OptionalFloat64 `json:"llm_chunking_after_stop_saved_ms"`
	ASROverlapAfterStopSavedMS  OptionalFloat64 `json:"asr_overlap_after_stop_saved_ms"`
	LLMOverlapAfterStopSavedMS  OptionalFloat64 `json:"llm_overlap_after_stop_saved_ms"`
	StopToFinalSavedMS          OptionalFloat64 `json:"stop_to_final_saved_ms"`
}

// OverlapReport remains compatible with profile_loop.summarize_reports while
// adding scheduling fields that are absent from the original meeting profile.
type OverlapReport struct {
	SchemaVersion               int                       `json:"schema_version"`
	Name                        string                    `json:"name,omitempty"`
	Method                      string                    `json:"method"`
	Status                      string                    `json:"status"`
	SimulatedLLM                bool                      `json:"simulated_llm"`
	Limitations                 []string                  `json:"limitations"`
	StartedAt                   time.Time                 `json:"started_at"`
	RecordingStartedAt          *time.Time                `json:"recording_started_at"`
	StoppedAt                   *time.Time                `json:"stopped_at"`
	EndedAt                     time.Time                 `json:"ended_at"`
	TotalWallMS                 float64                   `json:"total_wall_ms"`
	AudioDurationMS             int64                     `json:"audio_duration_ms"`
	TotalRTF                    OptionalFloat64           `json:"total_rtf"`
	StopToFinalMS               OptionalFloat64           `json:"stop_to_final_ms"`
	FirstFinalText              OptionalFloat64           `json:"first_final_text"`
	FirstLLMText                OptionalFloat64           `json:"first_llm_text"`
	Timing                      OverlapTiming             `json:"timing"`
	BacklogAtStop               OverlapBacklog            `json:"backlog_at_stop"`
	Config                      OverlapConfig             `json:"config"`
	Stages                      []OverlapStage            `json:"stages"`
	ASRTranscript               *TextArtifact             `json:"asr_transcript"`
	Transcript                  *TextArtifact             `json:"transcript"`
	Summary                     *TextArtifact             `json:"summary"`
	SummaryUnavailableReason    string                    `json:"summary_unavailable_reason,omitempty"`
	Structure                   OverlapStructureArtifact  `json:"structure"`
	ComparisonToOneShot         *OverlapComparison        `json:"comparison_to_one_shot"`
	ComparisonUnavailableReason string                    `json:"comparison_unavailable_reason,omitempty"`
	SchedulingEffects           *OverlapSchedulingEffects `json:"scheduling_effects"`
	Warnings                    []string                  `json:"warnings"`
	Errors                      []string                  `json:"errors"`
}
