package omni

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

func TestPauseRefreshExperimentConfiguration(t *testing.T) {
	for _, test := range []struct {
		value   string
		wantMS  int
		invalid bool
	}{
		{"", 0, false},
		{"0", 0, false},
		{"200", 200, false},
		{"300", 300, false},
		{"100", 0, true},
		{"bad", 0, true},
	} {
		t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS", test.value)
		gotMS, invalid := pauseRefreshExperimentMS()
		if gotMS != test.wantMS || invalid != test.invalid {
			t.Errorf("value %q: got (%d, %v), want (%d, %v)", test.value, gotMS, invalid, test.wantMS, test.invalid)
		}
	}
}

func TestPauseRefreshEligibility(t *testing.T) {
	qwen := Spec{Kind: kindQwen3ASR, ASR: &ASRTuning{Realtime: true}}
	for _, test := range []struct {
		name    string
		request asr.StreamRequest
		spec    Spec
		pauseMS int
		want    bool
	}{
		{"200 ms authoritative Qwen Final", asr.StreamRequest{SampleRate: sampleRate}, qwen, 200, true},
		{"300 ms authoritative Qwen Final", asr.StreamRequest{SampleRate: sampleRate}, qwen, 300, true},
		{"disabled", asr.StreamRequest{SampleRate: sampleRate}, qwen, 0, false},
		{"LiveResult", asr.StreamRequest{SampleRate: sampleRate, LiveResult: true}, qwen, 200, false},
		{"non realtime Qwen", asr.StreamRequest{SampleRate: sampleRate}, Spec{Kind: kindQwen3ASR, ASR: &ASRTuning{}}, 200, false},
		{"other model", asr.StreamRequest{SampleRate: sampleRate}, Spec{Kind: kindNemotron, ASR: &ASRTuning{Realtime: true}}, 200, false},
		{"speech compaction", asr.StreamRequest{SampleRate: sampleRate}, Spec{Kind: kindQwen3ASR, ASR: &ASRTuning{Realtime: true, SpeechOnly: true}}, 200, false},
		{"wrong sample rate", asr.StreamRequest{SampleRate: 8000}, qwen, 200, false},
		{"unsupported arm", asr.StreamRequest{SampleRate: sampleRate}, qwen, 150, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := pauseRefreshEnabled(test.request, test.spec, test.pauseMS); got != test.want {
				t.Errorf("pause refresh enabled = %v, want %v", got, test.want)
			}
		})
	}
}

func TestPauseRefreshDetectorUsesSocketSamplesAcrossChunkBoundaries(t *testing.T) {
	pcm := tone(600*time.Millisecond, 0.3)
	pcm = append(pcm, make([]byte, 300*sampleRate/1000*2)...)
	detector := pauseRefreshDetector{detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: 200})}
	chunkSizes := []int{1, 317, 2049, 71, 5201}
	var hints []*pauseRefreshHint
	for offset, chunk := 0, 0; offset < len(pcm); chunk++ {
		end := min(offset+chunkSizes[chunk%len(chunkSizes)], len(pcm))
		if hint := detector.observe(pcm[offset:end]); hint != nil {
			hints = append(hints, hint)
		}
		offset = end
	}
	if len(hints) != 1 {
		t.Fatalf("got %d pause hints, want one", len(hints))
	}
	if hints[0].SpeechStartSample != 0 || hints[0].SpeechEndSample != 600*sampleRate/1000 {
		t.Errorf("speech range %+v, want [0, %d)", hints[0], 600*sampleRate/1000)
	}

	leadingSilence := make([]byte, 500*sampleRate/1000*2)
	speech := tone(600*time.Millisecond, 0.3)
	leadingDetector := pauseRefreshDetector{detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: 200})}
	if hint := leadingDetector.observe(leadingSilence); hint != nil {
		t.Fatalf("leading silence produced a hint: %+v", hint)
	}
	if hint := leadingDetector.observe(speech); hint != nil {
		t.Fatalf("open speech produced a hint before its pause: %+v", hint)
	}
	if hint := leadingDetector.observe(make([]byte, 250*sampleRate/1000*2)); hint == nil || hint.SpeechStartSample < 500*sampleRate/1000 {
		t.Errorf("pause after leading silence has hint %+v", hint)
	}
}

func TestPauseRefreshHintSharesAppendAndAudioAccounting(t *testing.T) {
	pcm := tone(600*time.Millisecond, 0.3)
	pcm = append(pcm, make([]byte, 300*sampleRate/1000*2)...)
	stream := &live{
		preview: true,
		out:     make(chan []byte, 16),
		pauseRefresh: &pauseRefreshDetector{
			detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: 200}),
		},
	}
	for offset := 0; offset < len(pcm); offset += 100 * sampleRate / 1000 * 2 {
		end := min(offset+100*sampleRate/1000*2, len(pcm))
		if err := stream.Send(pcm[offset:end]); err != nil {
			t.Fatal(err)
		}
	}
	var hinted int
	var appendedSamples int
	for len(stream.out) > 0 {
		message := <-stream.out
		var appendMessage realtimeAppend
		if err := json.Unmarshal(message, &appendMessage); err != nil {
			t.Fatal(err)
		}
		appendedAudio, err := base64.StdEncoding.DecodeString(appendMessage.Audio)
		if err != nil {
			t.Fatal(err)
		}
		appendedSamples += len(appendedAudio) / 2
		if appendMessage.PreviewRefresh != nil {
			hinted++
			if appendMessage.PreviewRefresh.SpeechStartSample != 0 || appendMessage.PreviewRefresh.SpeechEndSample != 600*sampleRate/1000 {
				t.Errorf("wire hint %+v, want speech range [0, %d)", appendMessage.PreviewRefresh, 600*sampleRate/1000)
			}
		}
	}
	if hinted != 1 || appendedSamples != len(pcm)/2 || stream.queued != len(pcm) || len(stream.audio) != len(pcm) {
		t.Errorf("hints %d, appended samples %d, queued bytes %d, final bytes %d; want 1, %d, %d, %d", hinted, appendedSamples, stream.queued, len(stream.audio), len(pcm)/2, len(pcm), len(pcm))
	}
}

func TestPauseRefreshHintIsIgnoredByOlderRealtimeReaders(t *testing.T) {
	t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS", "200")
	fake := newFake(t, omnitest.Opts{Text: "final"}, nil)
	stream, err := fake.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	pcm := tone(600*time.Millisecond, 0.3)
	pcm = append(pcm, make([]byte, 300*sampleRate/1000*2)...)
	if err := stream.Send(pcm); err != nil {
		t.Fatal(err)
	}
	if err := stream.Finish(); err != nil {
		t.Fatal(err)
	}
	if events := drain(t, stream); len(events) == 0 || events[len(events)-1].Kind != asr.EventFinal {
		t.Fatalf("older reader did not fall back to Final: %+v", events)
	}
	fake.rt.Close()
	liveRecords := fake.records("live")
	if len(liveRecords) != 1 || liveRecords[0].Samples != len(pcm)/2 {
		t.Errorf("old reader received %+v, want all %d samples", liveRecords, len(pcm)/2)
	}
}
