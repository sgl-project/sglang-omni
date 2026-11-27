package omni

import (
	"os"
	"strconv"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type pauseRefreshHint struct {
	SpeechStartSample int `json:"speech_start_sample"`
	SpeechEndSample   int `json:"speech_end_sample"`
}

type realtimeAppend struct {
	Type           string            `json:"type"`
	Audio          string            `json:"audio"`
	PreviewRefresh *pauseRefreshHint `json:"preview_refresh,omitempty"`
}

type pauseRefreshDetector struct {
	detector           *vad.Detector
	closedSegmentCount int
}

func pauseRefreshExperimentMS() (int, bool) {
	rawValue, isSet := os.LookupEnv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS")
	if !isSet || strings.TrimSpace(rawValue) == "" || strings.TrimSpace(rawValue) == "0" {
		return 0, false
	}
	pauseMS, err := strconv.Atoi(strings.TrimSpace(rawValue))
	if err != nil || (pauseMS != 200 && pauseMS != 300) {
		return 0, true
	}
	return pauseMS, false
}

func pauseRefreshEnabled(request asr.StreamRequest, spec Spec, pauseDurationMS int) bool {
	return spec.Kind == kindQwen3ASR && spec.ASR != nil && spec.ASR.Realtime &&
		!spec.ASR.SpeechOnly && !request.LiveResult &&
		(request.SampleRate == 0 || request.SampleRate == sampleRate) &&
		(pauseDurationMS == 200 || pauseDurationMS == 300)
}

func (d *pauseRefreshDetector) observe(pcm []byte) *pauseRefreshHint {
	detector := d.detector
	detector.Write(pcm)
	closedSegments := detector.ClosedSegmentsSince(d.closedSegmentCount)
	d.closedSegmentCount += len(closedSegments)
	if len(closedSegments) == 0 || detector.Speaking() {
		return nil
	}
	segment := closedSegments[len(closedSegments)-1]
	return &pauseRefreshHint{SpeechStartSample: segment.Start, SpeechEndSample: segment.End}
}
