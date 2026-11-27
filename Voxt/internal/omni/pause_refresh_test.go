package omni

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strconv"
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

func TestPauseRefreshDeferredExperimentConfiguration(t *testing.T) {
	for _, test := range []struct {
		value   string
		want    bool
		invalid bool
	}{
		{"", false, false},
		{"0", false, false},
		{"1", true, false},
		{"true", false, true},
	} {
		t.Setenv("VOXT_EXPERIMENT_ASR_DEFER_BUSY_PAUSE", test.value)
		got, invalid := pauseRefreshDeferredExperimentEnabled()
		if got != test.want || invalid != test.invalid {
			t.Errorf("value %q: got (%v, %v), want (%v, %v)", test.value, got, invalid, test.want, test.invalid)
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
	for _, pauseMS := range []int{200, 300} {
		t.Run(strconv.Itoa(pauseMS)+"ms", func(t *testing.T) {
			newDetector := func() *pauseRefreshDetector {
				return &pauseRefreshDetector{detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: pauseMS})}
			}
			pause := silence(time.Duration(pauseMS) * time.Millisecond)
			expectHint := func(got *pauseRefreshHint, startMilliseconds, endMilliseconds int) {
				t.Helper()
				if got == nil || got.SpeechStartSample != startMilliseconds*sampleRate/1000 || got.SpeechEndSample != endMilliseconds*sampleRate/1000 {
					t.Fatalf("hint %+v, want [%d, %d) ms", got, startMilliseconds, endMilliseconds)
				}
			}

			pcm := append(tone(600*time.Millisecond, 0.3), pause...)
			expectHint(newDetector().observe(pcm), 0, 600)
			detector := newDetector()
			// Direct detector writes can split bytes; the websocket test below uses PCM16-aligned chunks.
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
				t.Fatalf("arbitrarily chunked speech-pause produced %d hints", len(hints))
			}
			expectHint(hints[0], 0, 600)
			if hint := detector.observe(pause); hint != nil {
				t.Fatalf("silence re-emitted a consumed hint: %+v", hint)
			}

			leading := append(silence(500*time.Millisecond), tone(600*time.Millisecond, 0.3)...)
			leading = append(leading, pause...)
			expectHint(newDetector().observe(leading), 500, 1100)

			split := newDetector()
			first := append(tone(600*time.Millisecond, 0.3), pause...)
			expectHint(split.observe(first), 0, 600)
			if hint := split.observe(tone(120*time.Millisecond, 0.3)); hint != nil || !split.detector.Speaking() {
				t.Fatalf("resumed speech state: speaking=%v hint=%+v", split.detector.Speaking(), hint)
			}
			if hint := split.observe(tone(40*time.Millisecond, 0.3)); hint != nil {
				t.Fatalf("suppressed pause hint was re-emitted while speech continued: %+v", hint)
			}
			resumedStart := 600 + pauseMS
			expectHint(split.observe(pause), resumedStart, resumedStart+160)
			if hint := split.observe(pause); hint != nil {
				t.Fatalf("resumed pause was re-emitted: %+v", hint)
			}
		})
	}
}

func TestPauseRefreshConsumesOnlyLatestHintAcrossMultipleTransitions(t *testing.T) {
	for _, pauseMS := range []int{200, 300} {
		t.Run(strconv.Itoa(pauseMS)+"ms", func(t *testing.T) {
			pcm := append(tone(600*time.Millisecond, 0.3), silence(time.Duration(pauseMS)*time.Millisecond)...)
			pcm = append(pcm, tone(300*time.Millisecond, 0.3)...)
			pcm = append(pcm, silence(time.Duration(pauseMS)*time.Millisecond)...)
			pcm = append(pcm, tone(200*time.Millisecond, 0.3)...)
			pcm = append(pcm, silence(time.Duration(pauseMS)*time.Millisecond)...)

			detector := &pauseRefreshDetector{detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: pauseMS})}
			hint := detector.observe(pcm)
			lastSpeechStart := 600 + pauseMS + 300 + pauseMS
			want := &pauseRefreshHint{
				SpeechStartSample: lastSpeechStart * sampleRate / 1000,
				SpeechEndSample:   (lastSpeechStart + 200) * sampleRate / 1000,
			}
			if hint == nil || *hint != *want {
				t.Fatalf("latest hint %+v, want %+v", hint, want)
			}
			if hint := detector.observe(silence(time.Duration(pauseMS) * time.Millisecond)); hint != nil {
				t.Fatalf("consumed transitions were emitted again: %+v", hint)
			}
		})
	}
}

func TestPauseRefreshSpeechEpochTracksBufferedAndSplitResumption(t *testing.T) {
	for _, pauseMS := range []int{200, 300} {
		t.Run(strconv.Itoa(pauseMS)+"ms", func(t *testing.T) {
			newDetector := func() *pauseRefreshDetector {
				return &pauseRefreshDetector{detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: pauseMS})}
			}
			pause := silence(time.Duration(pauseMS) * time.Millisecond)
			buffered := newDetector()
			pcm := append(tone(600*time.Millisecond, 0.3), pause...)
			pcm = append(pcm, tone(300*time.Millisecond, 0.3)...)
			pcm = append(pcm, pause...)
			hint := buffered.observe(pcm)
			if hint == nil || buffered.detector.Speaking() || buffered.speechEpoch != 2 {
				t.Fatalf("buffered speech-pause-speech state: hint=%+v speaking=%v epoch=%d", hint, buffered.detector.Speaking(), buffered.speechEpoch)
			}

			split := newDetector()
			hint = split.observe(append(tone(600*time.Millisecond, 0.3), pause...))
			if hint == nil || split.speechEpoch != 1 {
				t.Fatalf("first pause state: hint=%+v epoch=%d", hint, split.speechEpoch)
			}
			if split.observe(tone(120*time.Millisecond, 0.3)) != nil || split.speechEpoch != 2 {
				t.Fatalf("split resumption did not advance the epoch: epoch=%d", split.speechEpoch)
			}
		})
	}
}

func TestPauseRefreshRejectsStaleHintAfterResumedSpeech(t *testing.T) {
	for _, pauseMS := range []int{200, 300} {
		t.Run(strconv.Itoa(pauseMS)+"ms", func(t *testing.T) {
			pcm := tone(600*time.Millisecond, 0.3)
			pcm = append(pcm, silence(time.Duration(pauseMS)*time.Millisecond)...)
			pcm = append(pcm, tone(120*time.Millisecond, 0.3)...)
			detector := &pauseRefreshDetector{detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: pauseMS})}
			hint := detector.observe(pcm)
			if !detector.detector.Speaking() {
				t.Fatal("test audio did not reach the resumed-speaking detector state")
			}
			if hint != nil {
				t.Errorf("stale pause hint %+v emitted while speech had resumed", hint)
			}
			if hint := detector.observe(tone(40*time.Millisecond, 0.3)); hint != nil {
				t.Errorf("suppressed stale hint was re-emitted: %+v", hint)
			}
			resumedStart := (600 + pauseMS) * sampleRate / 1000
			hint = detector.observe(silence(time.Duration(pauseMS) * time.Millisecond))
			if hint == nil || hint.SpeechStartSample != resumedStart ||
				hint.SpeechEndSample != resumedStart+160*sampleRate/1000 {
				t.Errorf("new pause hint %+v, want resumed speech [%d, %d)", hint, resumedStart, resumedStart+160*sampleRate/1000)
			}
		})
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
		if appendMessage.SpeechEpoch != nil {
			t.Errorf("discard arm emitted deferred speech metadata: %+v", appendMessage)
		}
	}
	if hinted != 1 || appendedSamples != len(pcm)/2 || stream.queued != len(pcm) || len(stream.audio) != len(pcm) {
		t.Errorf("hints %d, appended samples %d, queued bytes %d, final bytes %d; want 1, %d, %d, %d", hinted, appendedSamples, stream.queued, len(stream.audio), len(pcm)/2, len(pcm), len(pcm))
	}
}

func TestPauseRefreshSpeechEpochSharesAppendOrdering(t *testing.T) {
	pcm := tone(600*time.Millisecond, 0.3)
	pcm = append(pcm, make([]byte, 300*sampleRate/1000*2)...)
	stream := &live{
		preview:        true,
		deferBusyPause: true,
		out:            make(chan []byte, 16),
		pauseRefresh: &pauseRefreshDetector{
			detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: 300}),
		},
	}
	for offset := 0; offset < len(pcm); offset += 100 * sampleRate / 1000 * 2 {
		end := min(offset+100*sampleRate/1000*2, len(pcm))
		if err := stream.Send(pcm[offset:end]); err != nil {
			t.Fatal(err)
		}
	}
	var appendedSamples int
	var epochs []int
	var hinted int
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
		if appendMessage.SpeechEpoch == nil {
			t.Fatal("deferred append omitted the speech epoch")
		}
		epochs = append(epochs, *appendMessage.SpeechEpoch)
		if appendMessage.PreviewRefresh != nil {
			hinted++
		}
	}
	if appendedSamples != len(pcm)/2 || hinted != 1 || len(epochs) < 2 || epochs[len(epochs)-1] != 1 {
		t.Errorf("appended samples %d, hints %d, epochs %v", appendedSamples, hinted, epochs)
	}
}

func TestPauseRefreshHintIsIgnoredByOlderRealtimeReaders(t *testing.T) {
	t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS", "200")
	t.Setenv("VOXT_EXPERIMENT_ASR_DEFER_BUSY_PAUSE", "1")
	t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_TRACE", "0")
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
	if len(liveRecords) != 1 || liveRecords[0].Samples != len(pcm)/2 ||
		liveRecords[0].Fields["preview_refresh_enabled"] != "true" {
		t.Errorf("old reader received %+v, want all %d samples", liveRecords, len(pcm)/2)
	}
	if liveRecords[0].Fields["preview_refresh_defer_busy_enabled"] != "true" ||
		liveRecords[0].Fields["speech_epoch_seen"] != "true" {
		t.Errorf("old reader did not receive the optional deferred metadata: %+v", liveRecords[0].Fields)
	}
}

func TestPauseRefreshAttributionCanProfileCadenceOnlySessions(t *testing.T) {
	t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS", "0")
	t.Setenv("VOXT_EXPERIMENT_ASR_DEFER_BUSY_PAUSE", "1")
	t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_TRACE", "1")
	fake := newFake(t, omnitest.Opts{}, nil)
	stream, err := fake.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Send(make([]byte, 100*sampleRate/1000*2)); err != nil {
		t.Fatal(err)
	}
	if err := stream.Finish(); err != nil {
		t.Fatal(err)
	}
	drain(t, stream)
	fake.rt.Close()
	liveRecords := fake.records("live")
	if len(liveRecords) != 1 || liveRecords[0].Fields["preview_refresh_attribution_enabled"] != "true" {
		t.Fatalf("cadence trace opt-in missing: %+v", liveRecords)
	}
	if _, exists := liveRecords[0].Fields["preview_refresh_enabled"]; exists {
		t.Errorf("trace-only arm enabled pause refresh: %+v", liveRecords[0].Fields)
	}
	if liveRecords[0].Fields["speech_epoch_seen"] != "" {
		t.Errorf("trace-only arm sent deferred metadata: %+v", liveRecords[0].Fields)
	}
}

func TestPauseRefreshSessionOptInIsOmittedWhenDisabled(t *testing.T) {
	t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS", "0")
	t.Setenv("VOXT_EXPERIMENT_ASR_DEFER_BUSY_PAUSE", "1")
	t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_TRACE", "0")
	fake := newFake(t, omnitest.Opts{}, nil)
	stream, err := fake.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Finish(); err != nil {
		t.Fatal(err)
	}
	drain(t, stream)
	fake.rt.Close()
	liveRecords := fake.records("live")
	if len(liveRecords) != 1 {
		t.Fatalf("got %d live records, want one", len(liveRecords))
	}
	if _, exists := liveRecords[0].Fields["preview_refresh_enabled"]; exists {
		t.Errorf("disabled session sent pause refresh opt-in: %+v", liveRecords[0].Fields)
	}
	if _, exists := liveRecords[0].Fields["preview_refresh_defer_busy_enabled"]; exists {
		t.Errorf("disabled session sent deferred opt-in: %+v", liveRecords[0].Fields)
	}
}
