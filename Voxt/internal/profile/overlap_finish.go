package profile

import (
	"context"
	"encoding/json"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func (r *overlapRun) finish() OverlapReport {
	finished := r.options.now()
	r.mu.Lock()
	defer r.mu.Unlock()
	report := OverlapReport{
		SchemaVersion: 1, Name: r.config.Name, Method: r.method, Status: "invalid", SimulatedLLM: true,
		Limitations: append([]string(nil), overlapLimitations...), StartedAt: r.started, EndedAt: finished,
		TotalWallMS: milliseconds(finished.Sub(r.started)), AudioDurationMS: asr.Duration(r.pcm, asr.SampleRate).Milliseconds(),
		TotalRTF: Unavailable[float64]("audio duration is unknown"), StopToFinalMS: Unavailable[float64]("the run did not reach a valid final result"),
		FirstFinalText: Unavailable[float64]("no non-empty final ASR text was returned"), FirstLLMText: Unavailable[float64]("no non-empty simulated LLM text was returned"),
		Timing: OverlapTiming{
			ASRTotalMS: Available(milliseconds(r.asrTotal)), ASRAfterStopMS: Unavailable[float64]("recording did not stop"),
			LLMTotalMS: Available(milliseconds(r.llmTotal)), LLMAfterStopMS: Unavailable[float64]("recording did not stop"),
			ASRCalls: r.asrCalls, LLMCalls: r.llmCalls, LLMSegmentsTotal: r.segmentCount,
		},
		BacklogAtStop: r.backlog, Config: r.config, Stages: append([]OverlapStage(nil), r.stages...),
		SummaryUnavailableReason: "the run did not produce a final meeting summary",
		Structure:                OverlapStructureArtifact{UnavailableReason: "the run did not produce a complete ordered structure"},
		Warnings:                 []string{}, Errors: append([]string(nil), r.errors...),
	}
	if report.AudioDurationMS > 0 {
		report.TotalRTF = Available(report.TotalWallMS / float64(report.AudioDurationMS))
	}
	if !r.recordingStarted.IsZero() {
		started := r.recordingStarted
		report.RecordingStartedAt = &started
		if !r.firstASRText.IsZero() {
			report.FirstFinalText = Available(milliseconds(r.firstASRText.Sub(started)))
		}
		if !r.firstLLMText.IsZero() {
			report.FirstLLMText = Available(milliseconds(r.firstLLMText.Sub(started)))
		}
	}
	if !r.stopped.IsZero() {
		stopped := r.stopped
		report.StoppedAt = &stopped
		report.Timing.ASRAfterStopMS = Available(milliseconds(r.asrAfterStop))
		report.Timing.LLMAfterStopMS = Available(milliseconds(r.llmAfterStop))
	} else {
		report.BacklogAtStop = OverlapBacklog{
			ASRSegments: Unavailable[int64]("recording did not stop"), LLMSegments: Unavailable[int64]("recording did not stop"),
		}
	}
	if !r.asrFailed && r.asrCompleted == r.segmentCount {
		text := joinOverlapParts(r.rawParts)
		report.ASRTranscript = &TextArtifact{Text: text, SHA256: HashText(text)}
	}
	complete := len(r.errors) == 0 && !r.stopped.IsZero() && (r.config.Mode != "meeting" || r.finalSummary != "")
	if complete {
		text := joinOverlapParts(r.cleanedParts)
		report.Status = "success"
		report.StopToFinalMS = Available(milliseconds(finished.Sub(r.stopped)))
		report.Transcript = &TextArtifact{Text: text, SHA256: HashText(text)}
		cleanup := make([]SimulatedPart, len(r.cleanedParts))
		for index, part := range r.cleanedParts {
			cleanup[index] = SimulatedPart{Index: index, Text: part}
		}
		document := OverlapStructure{Cleanup: cleanup, SummaryMap: append([]string{}, r.summaryMap...), SummaryReduce: append([]string{}, r.summaryReduce...), FinalSummary: r.finalSummary}
		encoded, _ := json.Marshal(document)
		structureSHA256 := HashText(string(encoded))
		report.Structure = OverlapStructureArtifact{Document: &document, SHA256: &structureSHA256}
		if r.config.Mode == "meeting" {
			report.Summary = &TextArtifact{Text: r.finalSummary, SHA256: HashText(r.finalSummary)}
			report.SummaryUnavailableReason = ""
		} else {
			report.SummaryUnavailableReason = "dictation runs cleanup only"
		}
	}
	return report
}

func addOverlapComparisons(reports []OverlapReport) {
	if len(reports) != 3 {
		return
	}
	baseline := &reports[0]
	for index := range reports {
		report := &reports[index]
		if baseline.Status != "success" || report.Status != "success" {
			report.ComparisonUnavailableReason = "one_shot and this run must both be valid"
			continue
		}
		comparison := compareOverlapReports(baseline, report)
		report.ComparisonToOneShot = &comparison
	}
	if reports[1].Status == "success" && baseline.Status == "success" {
		reports[1].SchedulingEffects = &OverlapSchedulingEffects{
			ASRChunkingAfterStopSavedMS: differenceOf(baseline.Timing.ASRAfterStopMS, reports[1].Timing.ASRAfterStopMS),
			LLMChunkingAfterStopSavedMS: differenceOf(baseline.Timing.LLMAfterStopMS, reports[1].Timing.LLMAfterStopMS),
			ASROverlapAfterStopSavedMS:  Unavailable[float64]("serial does not overlap recording"),
			LLMOverlapAfterStopSavedMS:  Unavailable[float64]("serial does not overlap recording"),
			StopToFinalSavedMS:          differenceOf(baseline.StopToFinalMS, reports[1].StopToFinalMS),
		}
	}
	if reports[2].Status == "success" && baseline.Status == "success" {
		effects := &OverlapSchedulingEffects{StopToFinalSavedMS: differenceOf(baseline.StopToFinalMS, reports[2].StopToFinalMS)}
		if reports[1].Status == "success" {
			effects.ASRChunkingAfterStopSavedMS = differenceOf(baseline.Timing.ASRAfterStopMS, reports[1].Timing.ASRAfterStopMS)
			effects.LLMChunkingAfterStopSavedMS = differenceOf(baseline.Timing.LLMAfterStopMS, reports[1].Timing.LLMAfterStopMS)
			effects.ASROverlapAfterStopSavedMS = differenceOf(reports[1].Timing.ASRAfterStopMS, reports[2].Timing.ASRAfterStopMS)
			effects.LLMOverlapAfterStopSavedMS = differenceOf(reports[1].Timing.LLMAfterStopMS, reports[2].Timing.LLMAfterStopMS)
		} else {
			effects.ASRChunkingAfterStopSavedMS = Unavailable[float64]("serial is invalid")
			effects.LLMChunkingAfterStopSavedMS = Unavailable[float64]("serial is invalid")
			effects.ASROverlapAfterStopSavedMS = Unavailable[float64]("serial is invalid")
			effects.LLMOverlapAfterStopSavedMS = Unavailable[float64]("serial is invalid")
		}
		reports[2].SchedulingEffects = effects
	}
}

func differenceOf(left, right OptionalFloat64) OptionalFloat64 {
	if left.Value == nil || right.Value == nil {
		return Unavailable[float64]("a timing input is unavailable")
	}
	return Available(*left.Value - *right.Value)
}

func compareOverlapReports(baseline, report *OverlapReport) OverlapComparison {
	prefix, suffix, firstDifference := textDifference(baseline.Transcript.Text, report.Transcript.Text)
	comparison := OverlapComparison{
		ASRTextEqual: baseline.ASRTranscript.SHA256 == report.ASRTranscript.SHA256,
		TextEqual:    baseline.Transcript.SHA256 == report.Transcript.SHA256,
		TextSHA256:   report.Transcript.SHA256, OneShotTextSHA256: baseline.Transcript.SHA256,
		CommonPrefixRunes: prefix, CommonSuffixRunes: suffix, FirstTextDifferenceRune: firstDifference,
		StructureEqual:  *baseline.Structure.SHA256 == *report.Structure.SHA256,
		StructureSHA256: *report.Structure.SHA256, OneShotStructureSHA256: *baseline.Structure.SHA256,
		CleanupPartCountDelta:    len(report.Structure.Document.Cleanup) - len(baseline.Structure.Document.Cleanup),
		SummaryMapCountDelta:     len(report.Structure.Document.SummaryMap) - len(baseline.Structure.Document.SummaryMap),
		SummaryReduceCountDelta:  len(report.Structure.Document.SummaryReduce) - len(baseline.Structure.Document.SummaryReduce),
		SummaryUnavailableReason: "dictation has no summary",
	}
	if baseline.Summary != nil && report.Summary != nil {
		equal := baseline.Summary.SHA256 == report.Summary.SHA256
		comparison.SummaryEqual = &equal
		comparison.SummaryUnavailableReason = ""
	}
	return comparison
}

func textDifference(baseline, candidate string) (int, int, OptionalInt64) {
	left, right := []rune(baseline), []rune(candidate)
	prefix := 0
	for prefix < len(left) && prefix < len(right) && left[prefix] == right[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < len(left)-prefix && suffix < len(right)-prefix && left[len(left)-1-suffix] == right[len(right)-1-suffix] {
		suffix++
	}
	if prefix == len(left) && prefix == len(right) {
		return prefix, suffix, Unavailable[int64]("texts are identical")
	}
	return prefix, suffix, Available(int64(prefix))
}

func (r *overlapRun) runConcurrent(ctx context.Context) {
	asrJobs := make(chan int, r.segmentCount)
	llmJobs := make(chan int, r.segmentCount)
	llmDone := make(chan struct{})
	go func() {
		defer close(llmJobs)
		for index := range asrJobs {
			if r.transcribe(ctx, index, pcmForChunk(r.pcm, r.chunks[index])) {
				llmJobs <- index
			}
		}
	}()
	go func() {
		defer close(llmDone)
		for index := range llmJobs {
			r.processSegment(ctx, index)
		}
	}()
	for index, chunk := range r.chunks {
		deadline := r.recordingStarted.Add(time.Duration(chunk.End) * time.Second / asr.SampleRate)
		if err := r.options.waitUntil(ctx, deadline); err != nil {
			r.fail(err)
			break
		}
		asrJobs <- index
		if r.options.afterFeed != nil {
			if err := r.options.afterFeed(ctx, index); err != nil {
				r.fail(err)
				break
			}
		}
	}
	r.markStop()
	close(asrJobs)
	<-llmDone
}
