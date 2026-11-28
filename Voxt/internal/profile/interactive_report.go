package profile

import "time"

var interactiveLimitations = []string{
	"measurement-only scheduling: production transcription, diarization, session priority, and insertion behavior are unchanged",
	"baseline submits the long and interactive client calls concurrently on one shared client; a native worker may serialize or queue them and parallel inference is not claimed",
	"interactive_wait_ms is arrival to completed interactive response; interactive_total_ms is the client call wall and includes any native queueing",
	"call counts are harness Diarize/Transcribe client calls; model-internal Final subrequests are not separately observable",
	"priority uses contiguous PCM chunks without VAD, overlap, or deduplication; partitioning can change recognition and speaker labels",
	"priority_vad pre-plans the whole track with Silero and meeting.PlanChunks, keeps hard overlap, and deduplicates adjacent text; this is not streaming production VAD",
	"chunked MTD segments retain their raw local speaker labels with absolute timestamps; overlap ownership and global speaker identities are not reconstructed",
	"long_text_equal and segment comparisons are exact equality checks, not CER, WER, DER, or quality scores",
	"priority and priority_vad long_total_ms include the interactive pause between their first and last long calls",
	"the interactive request arrives 100 ms after the long client call starts; a run that arrives after the long work or has no remaining priority chunk to resume is invalid",
	"baseline, priority, then priority_vad run in a fixed order on one warm ASR model, so cache and thermal state can carry over; VAD planning runs only before the last method",
	"audio decode and model load are outside the measured method intervals; no second ASR process is created, while priority_vad uses one configured Silero VAD server",
}

const interactiveSpeakerStitchingReason = "chunked MTD exposes only chunk-local S01/S02 labels and timestamps; voice embeddings and global clustering are unavailable, so cross-chunk speaker stitching is not performed"

type InteractiveChunk struct {
	Index             int     `json:"index"`
	CursorStartSample int     `json:"cursor_start_sample"`
	CursorEndSample   int     `json:"cursor_end_sample"`
	StartedMS         float64 `json:"started_ms"`
	FinishedMS        float64 `json:"finished_ms"`
	TextSHA256        *string `json:"text_sha256"`
	Error             string  `json:"error,omitempty"`
}

type InteractiveResume struct {
	AfterChunk           int     `json:"after_chunk"`
	NextChunk            int     `json:"next_chunk"`
	StoppedMS            float64 `json:"stopped_ms"`
	InteractiveStartedMS float64 `json:"interactive_started_ms"`
	ResumedMS            float64 `json:"resumed_ms"`
}

type InteractiveSegment struct {
	ChunkIndex int     `json:"chunk_index"`
	StartMS    float64 `json:"start_ms"`
	EndMS      float64 `json:"end_ms"`
	Speaker    string  `json:"speaker"`
	Text       string  `json:"text"`
	TextSHA256 string  `json:"text_sha256"`
}

type InteractiveComparison struct {
	LongTextEqual          *bool   `json:"long_text_equal"`
	BaselineLongTextHash   *string `json:"baseline_long_text_hash"`
	LongTextHash           *string `json:"long_text_hash"`
	BaselineSegmentCount   int     `json:"baseline_segment_count"`
	SegmentCount           int     `json:"segment_count"`
	SegmentTimestampsEqual *bool   `json:"segment_timestamps_equal"`
	SpeakerLabelsEqual     *bool   `json:"speaker_labels_equal"`
}

type InteractiveMethodReport struct {
	Method                       string                 `json:"method"`
	Status                       string                 `json:"status"`
	InteractiveWaitMS            *float64               `json:"interactive_wait_ms"`
	LongTotalMS                  *float64               `json:"long_total_ms"`
	InteractiveTotalMS           *float64               `json:"interactive_total_ms"`
	InteractiveArrivedDuringLong bool                   `json:"interactive_arrived_during_long"`
	LongText                     *string                `json:"long_text"`
	LongTextHash                 *string                `json:"long_text_hash"`
	LongSegments                 []InteractiveSegment   `json:"long_segments"`
	ComparisonToBaseline         *InteractiveComparison `json:"comparison_to_baseline,omitempty"`
	InteractiveText              *string                `json:"interactive_text"`
	InteractiveTextHash          *string                `json:"interactive_text_hash"`
	LongCalls                    int                    `json:"long_calls"`
	InteractiveCalls             int                    `json:"interactive_calls"`
	Chunks                       []InteractiveChunk     `json:"chunks"`
	Resume                       *InteractiveResume     `json:"resume"`
	Errors                       []string               `json:"errors"`
}

type InteractiveReport struct {
	SchemaVersion    int                      `json:"schema_version"`
	Name             string                   `json:"name,omitempty"`
	Status           string                   `json:"status"`
	StartedAt        time.Time                `json:"started_at"`
	EndedAt          time.Time                `json:"ended_at"`
	Config           InteractiveConfig        `json:"config"`
	Baseline         InteractiveMethodReport  `json:"baseline"`
	Priority         InteractiveMethodReport  `json:"priority"`
	PriorityVAD      InteractiveMethodReport  `json:"priority_vad"`
	LongTextEqual    *bool                    `json:"long_text_equal"`
	SpeakerStitching Optional[bool]           `json:"speaker_stitching"`
	Resources        Optional[ResourceReport] `json:"resources"`
	Limitations      []string                 `json:"limitations"`
	Errors           []string                 `json:"errors"`
}

func compareInteractiveToBaseline(config InteractiveConfig, baseline, method InteractiveMethodReport) *InteractiveComparison {
	comparison := &InteractiveComparison{
		BaselineLongTextHash: baseline.LongTextHash, LongTextHash: method.LongTextHash,
		BaselineSegmentCount: len(baseline.LongSegments), SegmentCount: len(method.LongSegments),
	}
	if baseline.Status == "success" && method.Status == "success" && baseline.LongText != nil && method.LongText != nil {
		equal := *baseline.LongText == *method.LongText
		comparison.LongTextEqual = &equal
	}
	if config.Operation == "diarize" && baseline.Status == "success" && method.Status == "success" {
		timestampsEqual := len(baseline.LongSegments) == len(method.LongSegments)
		speakerLabelsEqual := timestampsEqual
		for index := 0; index < len(baseline.LongSegments) && index < len(method.LongSegments); index++ {
			baselineSegment := baseline.LongSegments[index]
			methodSegment := method.LongSegments[index]
			if baselineSegment.StartMS != methodSegment.StartMS || baselineSegment.EndMS != methodSegment.EndMS {
				timestampsEqual = false
			}
			if baselineSegment.Speaker != methodSegment.Speaker {
				speakerLabelsEqual = false
			}
		}
		comparison.SegmentTimestampsEqual = &timestampsEqual
		comparison.SpeakerLabelsEqual = &speakerLabelsEqual
	}
	return comparison
}
