package omni

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

func recordedRequestStages(f *fake) []requestStage {
	f.mu.Lock()
	lines := slices.Clone(f.out)
	f.mu.Unlock()
	var stages []requestStage
	for _, line := range lines {
		if !strings.HasPrefix(line, `info omni: {"event":"omni_request_stage"`) {
			continue
		}
		var stage requestStage
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "info omni: ")), &stage); err != nil {
			panic(err)
		}
		stages = append(stages, stage)
	}
	return stages
}

func TestOfflineRequestStages(t *testing.T) {
	f, a := mossWithVAD(t, omnitest.Opts{})
	pcm := append(append(tone(time.Second, 0.3), silence(6*time.Second)...), tone(time.Second, 0.3)...)
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm", SampleRate: sampleRate}); err != nil {
		t.Fatal(err)
	}
	stages := recordedRequestStages(f)
	want := map[string]bool{
		"vad/acquire": false, "vad/multipart": false, "vad/http_headers": false,
		"vad/read_parse": false, "vad/total": false, "compact/total": false,
		"final/multipart": false, "final/http_headers": false, "final/sse": false, "final/total": false,
	}
	var id uint64
	for _, stage := range stages {
		key := stage.Kind + "/" + stage.Stage
		if _, ok := want[key]; !ok {
			t.Fatalf("unexpected stage %+v", stage)
		}
		want[key] = true
		if stage.Event != "omni_request_stage" || stage.RequestID == 0 || stage.DurationMS < 0 {
			t.Fatalf("invalid stage %+v", stage)
		}
		if id == 0 {
			id = stage.RequestID
		}
		if stage.RequestID != id {
			t.Fatalf("one request has several ids: %+v", stages)
		}
		if (key == "vad/total" || key == "compact/total") && stage.InputBytes != len(pcm) {
			t.Fatalf("input bytes in %+v, want %d", stage, len(pcm))
		}
		if key == "final/multipart" && (stage.InputBytes <= 0 || stage.InputBytes > len(pcm)) {
			t.Fatalf("compacted input bytes in %+v, want 1..%d", stage, len(pcm))
		}
	}
	for key, found := range want {
		if !found {
			t.Errorf("missing %s in %+v", key, stages)
		}
	}
}

func TestLogRequestStageDisabledWithoutID(t *testing.T) {
	var lines []string
	rt := &Runtime{cfg: Config{Log: func(_, msg string) { lines = append(lines, msg) }}}
	logRequestStage(rt, 0, "final", "total", time.Now(), 1, 0)
	if len(lines) != 0 {
		t.Fatalf("unexpected lines %q", lines)
	}
}
