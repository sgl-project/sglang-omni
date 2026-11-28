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

func TestPauseRefreshConfiguration(t *testing.T) {
	for _, test := range []struct {
		value   string
		wantMS  int
		invalid bool
	}{
		{"", 0, false}, {"0", 0, false}, {"200", 200, false},
		{"300", 300, false}, {"100", 0, true}, {"bad", 0, true},
	} {
		t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS", test.value)
		got, invalid := pauseRefreshExperimentMS()
		if got != test.wantMS || invalid != test.invalid {
			t.Errorf("pause value %q: got (%d, %v), want (%d, %v)",
				test.value, got, invalid, test.wantMS, test.invalid)
		}
	}
	for _, test := range []struct {
		value   string
		want    bool
		invalid bool
	}{
		{"", false, false}, {"0", false, false}, {"1", true, false},
		{"true", false, true},
	} {
		t.Setenv("VOXT_EXPERIMENT_ASR_DEFER_BUSY_PAUSE", test.value)
		got, invalid := pauseRefreshDeferredExperimentEnabled()
		if got != test.want || invalid != test.invalid {
			t.Errorf("deferred value %q: got (%v, %v), want (%v, %v)",
				test.value, got, invalid, test.want, test.invalid)
		}
	}
	for _, test := range []struct {
		value string
		want  bool
	}{{"", false}, {"0", false}, {"1", true}, {"true", false}} {
		t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_TRACE", test.value)
		if got := pauseRefreshAttributionEnabled(); got != test.want {
			t.Errorf("trace value %q: got %v, want %v", test.value, got, test.want)
		}
	}

	qwen := Spec{Kind: kindQwen3ASR, ASR: &ASRTuning{Realtime: true}}
	for _, test := range []struct {
		name    string
		request asr.StreamRequest
		spec    Spec
		pauseMS int
		want    bool
	}{
		{"200 ms Qwen Final", asr.StreamRequest{SampleRate: sampleRate}, qwen, 200, true},
		{"300 ms Qwen Final", asr.StreamRequest{SampleRate: sampleRate}, qwen, 300, true},
		{"disabled", asr.StreamRequest{SampleRate: sampleRate}, qwen, 0, false},
		{"LiveResult", asr.StreamRequest{SampleRate: sampleRate, LiveResult: true}, qwen, 200, false},
		{"non realtime Qwen", asr.StreamRequest{SampleRate: sampleRate}, Spec{Kind: kindQwen3ASR, ASR: &ASRTuning{}}, 200, false},
		{"other model", asr.StreamRequest{SampleRate: sampleRate}, Spec{Kind: kindNemotron, ASR: &ASRTuning{Realtime: true}}, 200, false},
		{"speech compaction", asr.StreamRequest{SampleRate: sampleRate}, Spec{Kind: kindQwen3ASR, ASR: &ASRTuning{Realtime: true, SpeechOnly: true}}, 200, false},
		{"wrong sample rate", asr.StreamRequest{SampleRate: 8000}, qwen, 200, false},
		{"unsupported threshold", asr.StreamRequest{SampleRate: sampleRate}, qwen, 150, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := pauseRefreshEnabled(test.request, test.spec, test.pauseMS); got != test.want {
				t.Errorf("pause refresh enabled = %v, want %v", got, test.want)
			}
		})
	}
}

func TestPauseRefreshDetector(t *testing.T) {
	for _, pauseMS := range []int{200, 300} {
		t.Run(strconv.Itoa(pauseMS)+"ms", func(t *testing.T) {
			newDetector := func() *pauseRefreshDetector {
				return &pauseRefreshDetector{
					detector: vad.NewDetector(sampleRate, vad.Params{HangoverMS: pauseMS}),
				}
			}
			pause := silence(time.Duration(pauseMS) * time.Millisecond)
			expect := func(hint *pauseRefreshHint, startMS, endMS int) {
				t.Helper()
				if hint == nil || hint.SpeechStartSample != startMS*sampleRate/1000 ||
					hint.SpeechEndSample != endMS*sampleRate/1000 {
					t.Fatalf("hint %+v, want [%d, %d) ms", hint, startMS, endMS)
				}
			}

			pcm := append(silence(500*time.Millisecond), tone(600*time.Millisecond, 0.3)...)
			pcm = append(pcm, pause...)
			expect(newDetector().observe(pcm, false), 500, 1100)

			pcm = append(tone(600*time.Millisecond, 0.3), pause...)
			detector := newDetector()
			chunkSizes := []int{1, 317, 2049, 71, 5201}
			var hints []*pauseRefreshHint
			for offset, chunk := 0, 0; offset < len(pcm); chunk++ {
				end := min(offset+chunkSizes[chunk%len(chunkSizes)], len(pcm))
				if hint := detector.observe(pcm[offset:end], false); hint != nil {
					hints = append(hints, hint)
				}
				offset = end
			}
			if len(hints) != 1 {
				t.Fatalf("split speech-pause produced %d hints", len(hints))
			}
			expect(hints[0], 0, 600)
			if hint := detector.observe(pause, false); hint != nil {
				t.Fatalf("silence re-emitted a consumed hint: %+v", hint)
			}

			resumed := newDetector()
			if hint := resumed.observe(append(append(
				tone(600*time.Millisecond, 0.3), pause...),
				tone(120*time.Millisecond, 0.3)...), false); hint != nil ||
				!resumed.detector.Speaking() {
				t.Fatalf("speech resumption retained a stale hint: %+v", hint)
			}
			if hint := resumed.observe(tone(40*time.Millisecond, 0.3), false); hint != nil {
				t.Fatalf("continued speech re-emitted a hint: %+v", hint)
			}
			startMS := 600 + pauseMS
			expect(resumed.observe(pause, false), startMS, startMS+160)

			transitions := append(tone(600*time.Millisecond, 0.3), pause...)
			transitions = append(transitions, tone(300*time.Millisecond, 0.3)...)
			transitions = append(transitions, pause...)
			transitions = append(transitions, tone(200*time.Millisecond, 0.3)...)
			transitions = append(transitions, pause...)
			latest := newDetector()
			lastStartMS := 600 + pauseMS + 300 + pauseMS
			expect(latest.observe(transitions, true), lastStartMS, lastStartMS+200)
			if latest.speechEpoch != 3 {
				t.Fatalf("buffered transitions advanced epoch to %d, want 3", latest.speechEpoch)
			}
			if hint := latest.observe(pause, true); hint != nil {
				t.Fatalf("latest consumed transition was re-emitted: %+v", hint)
			}

			split := newDetector()
			expect(split.observe(pcm, true), 0, 600)
			if split.speechEpoch != 1 ||
				split.observe(tone(120*time.Millisecond, 0.3), true) != nil ||
				split.speechEpoch != 2 {
				t.Fatalf("split resumption epoch = %d, want 2", split.speechEpoch)
			}
		})
	}
}

func TestPauseRefreshAppend(t *testing.T) {
	pcm := append(tone(600*time.Millisecond, 0.3), make([]byte, 300*sampleRate/1000*2)...)
	for _, deferred := range []bool{false, true} {
		stream := &live{
			preview:        true,
			deferBusyPause: deferred,
			out:            make(chan []byte, 16),
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

		var hinted, appendedSamples int
		var lastEpoch int
		for len(stream.out) > 0 {
			message := <-stream.out
			var appendMessage realtimeAppend
			if err := json.Unmarshal(message, &appendMessage); err != nil {
				t.Fatal(err)
			}
			audio, err := base64.StdEncoding.DecodeString(appendMessage.Audio)
			if err != nil {
				t.Fatal(err)
			}
			appendedSamples += len(audio) / 2
			if hint := appendMessage.PreviewRefresh; hint != nil {
				hinted++
				if hint.SpeechStartSample != 0 ||
					hint.SpeechEndSample != 600*sampleRate/1000 ||
					appendedSamples < hint.SpeechEndSample {
					t.Errorf("hint %+v arrived after %d samples", hint, appendedSamples)
				}
			}
			if deferred {
				if appendMessage.SpeechEpoch == nil {
					t.Fatal("deferred append omitted speech epoch")
				}
				lastEpoch = *appendMessage.SpeechEpoch
			} else if appendMessage.SpeechEpoch != nil {
				t.Errorf("discard arm emitted speech epoch %d", *appendMessage.SpeechEpoch)
			}
		}
		if hinted != 1 || appendedSamples != len(pcm)/2 ||
			stream.queued != len(pcm) || len(stream.audio) != len(pcm) {
			t.Errorf("deferred=%v: hints=%d samples=%d queued=%d final=%d",
				deferred, hinted, appendedSamples, stream.queued, len(stream.audio))
		}
		if deferred && lastEpoch != 1 {
			t.Errorf("deferred speech epoch = %d, want 1", lastEpoch)
		}
	}
}

func TestPauseRefreshCompatibility(t *testing.T) {
	t.Run("older realtime reader and authoritative Final", func(t *testing.T) {
		t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_MS", "200")
		t.Setenv("VOXT_EXPERIMENT_ASR_DEFER_BUSY_PAUSE", "1")
		t.Setenv("VOXT_EXPERIMENT_ASR_PAUSE_REFRESH_TRACE", "0")
		fake := newFake(t, omnitest.Opts{Text: "final"}, nil)
		stream, err := fake.asr(t, ASRFast).Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
		if err != nil {
			t.Fatal(err)
		}
		pcm := append(tone(600*time.Millisecond, 0.3), silence(300*time.Millisecond)...)
		if err := stream.Send(pcm); err != nil {
			t.Fatal(err)
		}
		if err := stream.Finish(); err != nil {
			t.Fatal(err)
		}
		events := drain(t, stream)
		if len(events) == 0 || events[len(events)-1].Kind != asr.EventFinal {
			t.Fatalf("older reader did not finish with Final: %+v", events)
		}
		fake.rt.Close()
		records := fake.records("live")
		if len(records) != 1 || records[0].Samples != len(pcm)/2 ||
			records[0].Fields["preview_refresh_enabled"] != "true" ||
			records[0].Fields["preview_refresh_defer_busy_enabled"] != "true" ||
			records[0].Fields["speech_epoch_seen"] != "true" {
			t.Errorf("optional scheduling metadata or audio was lost: %+v", records)
		}
	})

	t.Run("disabled session omits metadata", func(t *testing.T) {
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
		records := fake.records("live")
		if len(records) != 1 {
			t.Fatalf("got %d live records, want one", len(records))
		}
		for _, field := range []string{
			"preview_refresh_enabled", "preview_refresh_defer_busy_enabled", "speech_epoch_seen",
		} {
			if _, exists := records[0].Fields[field]; exists {
				t.Errorf("disabled session sent %s: %+v", field, records[0].Fields)
			}
		}
	})

	t.Run("trace alone does not enable pause metadata", func(t *testing.T) {
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
		records := fake.records("live")
		if len(records) != 1 ||
			records[0].Fields["preview_refresh_attribution_enabled"] != "true" {
			t.Fatalf("trace opt-in missing: %+v", records)
		}
		for _, field := range []string{
			"preview_refresh_enabled", "preview_refresh_defer_busy_enabled", "speech_epoch_seen",
		} {
			if _, exists := records[0].Fields[field]; exists {
				t.Errorf("trace-only arm sent %s: %+v", field, records[0].Fields)
			}
		}
	})
}
