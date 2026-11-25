package session

import (
	"context"
	"math"
	"sort"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type initializationDelayCase struct {
	name     string
	llm      time.Duration
	snippets time.Duration
	url      time.Duration
}

type initializationLatencySample struct {
	open, pcm, partial                              time.Duration
	asrFinal, llmRequest, cleanText, cleanup, paste time.Duration
}

// BenchmarkASRInitializationIsolation compares the legacy combined barrier
// with the split production path using deterministic fake providers. Delays
// are artificial and are not evidence of real-model TTFT. Each iteration
// runs both paths back-to-back, alternating which path goes first.
func BenchmarkASRInitializationIsolation(b *testing.B) {
	cohorts := []initializationDelayCase{
		{name: "llm_0ms"},
		{name: "llm_100ms", llm: 100 * time.Millisecond},
		{name: "llm_300ms", llm: 300 * time.Millisecond},
		{name: "llm_1000ms", llm: time.Second},
		{name: "snippets_100ms", snippets: 100 * time.Millisecond},
		{name: "browser_url_100ms", url: 100 * time.Millisecond},
	}
	for _, delays := range cohorts {
		b.Run(delays.name, func(b *testing.B) {
			baseline := make([]initializationLatencySample, 0, b.N)
			candidate := make([]initializationLatencySample, 0, b.N)
			for i := 0; i < b.N; i++ {
				if i%2 == 0 {
					baseline = append(baseline, runInitializationLatencySample(b, delays, false))
					candidate = append(candidate, runInitializationLatencySample(b, delays, true))
				} else {
					candidate = append(candidate, runInitializationLatencySample(b, delays, true))
					baseline = append(baseline, runInitializationLatencySample(b, delays, false))
				}
			}
			reportLatencyPercentiles(b, "baseline", baseline)
			reportLatencyPercentiles(b, "candidate", candidate)
		})
	}
}

func runInitializationLatencySample(b *testing.B, delays initializationDelayCase, split bool) initializationLatencySample {
	b.Helper()
	provider := &fakeStreamer{final: "benchmark transcript"}
	model := &fakeLLM{text: "Cleaned transcript."}
	h := newHarness(b, provider, model, func(_ *Config, settings *settings.Settings) {
		settings.Features.Transcribe.Enhance = delays.url == 0
		if delays.url > 0 {
			settings.Features.Transcribe.Enhance = false
		}
	})
	h.audio.notifyAt = 5
	h.audio.notified = make(chan struct{})
	signals := observeInitializationTimings(h)

	target := slack
	if delays.url > 0 {
		target = chrome
		h.m.cfg.Scene = &SceneConfig{
			Rules: func() (appctx.Rules, bool) {
				return appctx.Rules{Groups: []appctx.Group{{
					ID: "url", Enabled: true, URLs: []string{"example.com"}, Features: appctx.DefaultFeatures,
					TranscribeCleanup: appctx.CleanupOn,
				}}}, true
			},
			IsBrowser: func(string) bool { return true }, URLTimeout: 5 * time.Second,
			URL: func(ctx context.Context, _ proto.App) (string, error) {
				if err := waitInjectedInitializationDelay(ctx, delays.url); err != nil {
					return "", err
				}
				return "https://example.com/chat", nil
			},
		}
	}
	h.m.cfg.Snippets = func(ctx context.Context) ([]snippet.Snippet, error) {
		if err := waitInjectedInitializationDelay(ctx, delays.snippets); err != nil {
			return nil, err
		}
		return nil, nil
	}

	services := func(ctx context.Context, effective settings.Settings) (Services, error) {
		if err := waitInjectedInitializationDelay(ctx, delays.llm); err != nil {
			return Services{}, err
		}
		service := Services{ASR: provider, ASRInfo: Model{ID: "fake-asr"}}
		if effective.Features.Transcribe.Enhance {
			service.LLM, service.LLMInfo = model, Model{ID: "fake-llm"}
		}
		return service, nil
	}
	if split {
		setIndependentResolvers(h, provider, func(ctx context.Context, _ Kind, effective settings.Settings) LLMReadiness {
			service, err := services(ctx, effective)
			if err != nil {
				return LLMReadiness{Outcome: LLMUnavailable, Err: err}
			}
			if service.LLM == nil {
				return LLMReadiness{Outcome: LLMNotRequested}
			}
			return LLMReadiness{LLM: service.LLM, LLMInfo: service.LLMInfo, Outcome: LLMAvailable}
		})
	} else {
		h.m.cfg.Resolve = func(ctx context.Context, _ Kind, effective settings.Settings, speech bool) (Services, error) {
			service, err := services(ctx, effective)
			if err != nil {
				return Services{}, err
			}
			if !speech {
				service.ASR, service.ASRInfo = nil, Model{}
			}
			return service, nil
		}
	}

	id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: target})
	if err != nil {
		b.Fatal(err)
	}
	started := <-signals.sessionStarted
	openDone := <-signals.asrOpenDone
	firstPCM := <-signals.firstPCM
	firstPartial := <-signals.firstPartial
	<-h.audio.notified // Minimum recording length before Stop.
	stopAt := time.Now()
	h.m.Stop(id)
	asrFinal := <-signals.asrFinal
	llmRequest := <-signals.llmRequest
	cleanText := <-signals.llmFirstDelta
	cleanupDone := <-signals.cleanupComplete
	pasteDone := <-signals.deliveryComplete
	select {
	case end := <-h.rec.ended:
		if end.State != Done {
			b.Fatalf("fake benchmark session ended in %s: %+v", end.State, end.Problem)
		}
	case <-time.After(5 * time.Second):
		b.Fatalf("fake benchmark session did not finish: %+v", h.m.Current())
	}
	h.m.Close()
	return initializationLatencySample{
		open:     openDone.Sub(started),
		pcm:      firstPCM.Sub(started),
		partial:  firstPartial.Sub(started),
		asrFinal: asrFinal.Sub(stopAt), llmRequest: llmRequest.Sub(stopAt), cleanText: cleanText.Sub(stopAt),
		cleanup: cleanupDone.Sub(stopAt), paste: pasteDone.Sub(stopAt),
	}
}

func waitInjectedInitializationDelay(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func reportLatencyPercentiles(b *testing.B, prefix string, samples []initializationLatencySample) {
	b.Helper()
	report := func(name string, value func(initializationLatencySample) time.Duration) {
		values := make([]float64, 0, len(samples))
		for _, sample := range samples {
			values = append(values, float64(value(sample))/float64(time.Millisecond))
		}
		sort.Float64s(values)
		percentile := func(p float64) float64 {
			index := max(0, min(len(values)-1, int(math.Ceil(float64(len(values))*p))-1))
			return values[index]
		}
		b.ReportMetric(percentile(.50), prefix+"_"+name+"_p50_ms")
		b.ReportMetric(percentile(.95), prefix+"_"+name+"_p95_ms")
	}
	b.ReportMetric(float64(len(samples)), prefix+"_samples")
	report("start_to_asr_open", func(s initializationLatencySample) time.Duration { return s.open })
	report("start_to_first_pcm", func(s initializationLatencySample) time.Duration { return s.pcm })
	report("start_to_first_partial", func(s initializationLatencySample) time.Duration { return s.partial })
	report("stop_to_asr_final", func(s initializationLatencySample) time.Duration { return s.asrFinal })
	report("stop_to_llm_request", func(s initializationLatencySample) time.Duration { return s.llmRequest })
	report("stop_to_first_cleaned_text", func(s initializationLatencySample) time.Duration { return s.cleanText })
	report("stop_to_cleanup_complete", func(s initializationLatencySample) time.Duration { return s.cleanup })
	report("stop_to_delivery", func(s initializationLatencySample) time.Duration { return s.paste })
}
