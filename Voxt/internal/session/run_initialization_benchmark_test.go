package session

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type initializationDelayCase struct {
	name           string
	llm            time.Duration
	snippets       time.Duration
	dictionary     time.Duration
	dictionaryHint bool
	url            time.Duration
	frames         int
	cleanAfterStop bool
	livePolish     bool
}

type initializationLatencySample struct {
	openStart, open, captured, sent, partial                time.Duration
	asrFinal, cleanText, cleanup, paste                     time.Duration
	audioDuration                                           time.Duration
	pcmBytes, frames, prefillCalls, chatsBefore, chatsAfter int
	insertions                                              int
	cleanupOutcome, deliveredText                           string
	cleanTextBeforeStop, finalEqual, livePolishReuse        bool
}

type initializationBenchmarkLLM struct {
	*fakeLLM
	stopAt      atomic.Int64
	prefills    atomic.Int32
	chatsBefore atomic.Int32
	chatsAfter  atomic.Int32
}

func (m *initializationBenchmarkLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	stopAt := m.stopAt.Load()
	if stopAt == 0 || time.Now().UnixNano() < stopAt {
		m.chatsBefore.Add(1)
	} else {
		m.chatsAfter.Add(1)
	}
	return m.fakeLLM.Chat(ctx, request)
}

func (m *initializationBenchmarkLLM) Prefill(context.Context, llm.Request) error {
	m.prefills.Add(1)
	return nil
}

func (m *initializationBenchmarkLLM) CanPrefill() bool { return false }

// BenchmarkASRInitializationIsolation compares a combined-resolver diagnostic
// with the split production path using paired, fixed recordings and fake providers.
// Injected delays are not evidence of real-model TTFT.
func BenchmarkASRInitializationIsolation(b *testing.B) {
	cohorts := []initializationDelayCase{
		{name: "clean_after_stop/llm_0ms", llm: 0, cleanAfterStop: true},
		{name: "clean_after_stop/llm_100ms", llm: 100 * time.Millisecond, cleanAfterStop: true},
		{name: "clean_after_stop/llm_300ms", llm: 300 * time.Millisecond, cleanAfterStop: true},
		{name: "clean_after_stop/llm_1000ms", llm: time.Second, cleanAfterStop: true},
		{name: "clean_after_stop/snippets_100ms", snippets: 100 * time.Millisecond, cleanAfterStop: true},
		{name: "clean_after_stop/dictionary_100ms_hints_off", dictionary: 100 * time.Millisecond, cleanAfterStop: true},
		{name: "clean_after_stop/dictionary_100ms_hints_on", dictionary: 100 * time.Millisecond, dictionaryHint: true, cleanAfterStop: true},
		{name: "clean_after_stop/browser_url_100ms", url: 100 * time.Millisecond, cleanAfterStop: true},
		{name: "clean_after_stop/short_before_llm_ready", llm: time.Second, frames: 4, cleanAfterStop: true},
		{name: "clean_after_stop/llm_ready_before_stop", llm: 100 * time.Millisecond, frames: 10, cleanAfterStop: true},
		{name: "live_polish/llm_ready_before_stop", llm: 100 * time.Millisecond, frames: 12, livePolish: true},
	}
	for _, delays := range cohorts {
		b.Run(delays.name, func(b *testing.B) {
			baseline := make([]initializationLatencySample, 0, b.N)
			candidate := make([]initializationLatencySample, 0, b.N)
			for i := 0; i < b.N; i++ {
				baselineFirst := i%4 == 0 || i%4 == 3
				if baselineFirst {
					baselineSample := runInitializationLatencySample(b, delays, false)
					candidateSample := runInitializationLatencySample(b, delays, true)
					assertPairedInitializationWork(b, baselineSample, candidateSample, i)
					baseline = append(baseline, baselineSample)
					candidate = append(candidate, candidateSample)
				} else {
					candidateSample := runInitializationLatencySample(b, delays, true)
					baselineSample := runInitializationLatencySample(b, delays, false)
					assertPairedInitializationWork(b, baselineSample, candidateSample, i)
					baseline = append(baseline, baselineSample)
					candidate = append(candidate, candidateSample)
				}
			}
			b.Log("comparison=COMBINED_VS_SPLIT_FAKE_PROVIDER_DIAGNOSTIC")
			reportLatencyPercentiles(b, "combined", baseline)
			reportLatencyPercentiles(b, "split", candidate)
			reportInitializationCounts(b, "combined", baseline)
			reportInitializationCounts(b, "split", candidate)
		})
	}
}

func runInitializationLatencySample(b *testing.B, delays initializationDelayCase, split bool) initializationLatencySample {
	b.Helper()
	frames := delays.frames
	if frames == 0 {
		frames = 10
	}
	final := "hello aura link and my email address."
	partials := []string{final}
	provider := &fakeStreamer{final: final, partials: partials}
	model := &initializationBenchmarkLLM{fakeLLM: &fakeLLM{text: "<output>Cleaned transcript VoxtNew and my email address.</output>"}}
	h := newHarness(b, provider, model, func(config *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = delays.url == 0
		currentSettings.Features.Transcribe.CleanAfterStop = delays.cleanAfterStop
		currentSettings.Dictionary.ASRHints = delays.dictionaryHint
		if delays.url > 0 {
			currentSettings.Features.Transcribe.Enhance = false
			config.Scene = &SceneConfig{
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
	})
	pcm := make([]byte, 0, frames*len(frame))
	for range frames {
		pcm = append(pcm, frame...)
	}
	h.audio.file = pcm
	h.audio.fileTick = 100 * time.Millisecond
	h.audio.drained = make(chan struct{})
	h.m.cfg.Glossary = func(ctx context.Context) ([]store.Term, error) {
		if err := waitInjectedInitializationDelay(ctx, delays.dictionary); err != nil {
			return nil, err
		}
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}
	h.m.cfg.Snippets = func(ctx context.Context) ([]snippet.Snippet, error) {
		if err := waitInjectedInitializationDelay(ctx, delays.snippets); err != nil {
			return nil, err
		}
		return []snippet.Snippet{{ID: "email", Trigger: "my email address", Text: "bench@example.com"}}, nil
	}
	signals := observeInitializationTimings(h)

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

	target := slack
	if delays.url > 0 {
		target = chrome
	}
	startAt := time.Now()
	id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: target})
	if err != nil {
		b.Fatal(err)
	}
	started := receiveBenchmarkTime(b, signals.sessionStarted, "session start")
	stopAtSignal := make(chan time.Time, 1)
	stopAfter := time.Duration(frames)*100*time.Millisecond + 150*time.Millisecond
	go func() {
		wait := time.Until(startAt.Add(stopAfter))
		if wait > 0 {
			timer := time.NewTimer(wait)
			defer timer.Stop()
			<-timer.C
		}
		stopAt := time.Now()
		model.stopAt.Store(stopAt.UnixNano())
		h.m.Stop(id)
		stopAtSignal <- stopAt
	}()
	openStart := receiveBenchmarkTime(b, signals.asrOpenStarted, "ASR Open start")
	openDone := receiveBenchmarkTime(b, signals.asrOpenDone, "ASR Open completion")
	captured := receiveBenchmarkTime(b, signals.firstPCMCaptured, "first captured PCM")
	sent := receiveBenchmarkTime(b, signals.firstPCM, "first PCM sent to ASR")
	partial := receiveBenchmarkTime(b, signals.firstPartial, "first nonempty ASR partial")
	stopAt := receiveBenchmarkTime(b, stopAtSignal, "fixed Stop schedule")
	select {
	case <-h.audio.drained:
	default:
		b.Fatalf("fixed recording did not finish before Stop: frames=%d, bytes=%d", h.audio.recorded(), len(pcm))
	}
	asrFinal := receiveBenchmarkTime(b, signals.asrFinal, "ASR Final")
	cleanupDone := receiveBenchmarkTime(b, signals.cleanupComplete, "cleanup completion")
	pasteDone := receiveBenchmarkTime(b, signals.deliveryComplete, "delivery completion")
	end := waitBenchmarkEnd(b, h)
	cleanTextAt, gotCleanText := optionalBenchmarkTime(signals.cleanupFirstText)
	cleanText := time.Duration(-1)
	if gotCleanText && !cleanTextAt.Before(stopAt) {
		cleanText = cleanTextAt.Sub(stopAt)
	}
	finalDelivered := ""
	insertions := 0
	if pasted := h.text.pastes(); len(pasted) > 0 {
		finalDelivered = pasted[0]
		insertions = len(pasted)
	}
	pcmBytes := len(h.audio.file)
	if pcmBytes != frames*len(frame) {
		b.Fatalf("fixed recording bytes=%d, expected=%d", pcmBytes, frames*len(frame))
	}
	if got := h.audio.recorded(); got != frames {
		b.Fatalf("fixed recording frames=%d, expected=%d", got, frames)
	}
	if len(finalDelivered) == 0 || finalDelivered != end.Result || insertions != 1 {
		b.Fatalf("final delivery text=%q, session result=%q, insertions=%d", finalDelivered, end.Result, insertions)
	}
	outcome := benchmarkCleanupOutcome(h)
	sample := initializationLatencySample{
		openStart: openStart.Sub(started), open: openDone.Sub(started), captured: captured.Sub(started),
		sent: sent.Sub(started), partial: partial.Sub(started), asrFinal: asrFinal.Sub(stopAt),
		cleanText: cleanText, cleanup: cleanupDone.Sub(stopAt), paste: pasteDone.Sub(stopAt),
		audioDuration: pcmDuration(pcmBytes), pcmBytes: pcmBytes, frames: frames,
		prefillCalls: int(model.prefills.Load()), chatsBefore: int(model.chatsBefore.Load()),
		chatsAfter: int(model.chatsAfter.Load()), insertions: insertions, cleanupOutcome: outcome,
		deliveredText: finalDelivered, cleanTextBeforeStop: gotCleanText && cleanTextAt.Before(stopAt),
		finalEqual:      finalDelivered == end.Result,
		livePolishReuse: delays.livePolish && model.chatsBefore.Load() > 0 && model.chatsAfter.Load() == 0,
	}
	h.m.Close()
	return sample
}

func receiveBenchmarkTime(b *testing.B, signal <-chan time.Time, name string) time.Time {
	b.Helper()
	select {
	case at := <-signal:
		return at
	case <-time.After(10 * time.Second):
		b.Fatalf("missing timing signal: %s", name)
		return time.Time{}
	}
}

func optionalBenchmarkTime(signal <-chan time.Time) (time.Time, bool) {
	select {
	case at := <-signal:
		return at, true
	default:
		return time.Time{}, false
	}
}

func waitBenchmarkEnd(b *testing.B, h *harness) Snapshot {
	b.Helper()
	select {
	case end := <-h.rec.ended:
		if end.State != Done {
			b.Fatalf("fake benchmark session ended in %s: %+v", end.State, end.Problem)
		}
		return end
	case <-time.After(10 * time.Second):
		b.Fatalf("fake benchmark session did not finish: %+v", h.m.Current())
		return Snapshot{}
	}
}

func assertPairedInitializationWork(b *testing.B, baseline, candidate initializationLatencySample, index int) {
	b.Helper()
	if baseline.pcmBytes != candidate.pcmBytes || baseline.frames != candidate.frames || baseline.audioDuration != candidate.audioDuration {
		b.Fatalf("pair %d used unequal recordings: combined=%d bytes/%d frames/%s split=%d bytes/%d frames/%s",
			index, baseline.pcmBytes, baseline.frames, baseline.audioDuration, candidate.pcmBytes, candidate.frames, candidate.audioDuration)
	}
	if baseline.deliveredText != candidate.deliveredText || !baseline.finalEqual || !candidate.finalEqual {
		b.Fatalf("pair %d delivered different text: combined=%q split=%q", index, baseline.deliveredText, candidate.deliveredText)
	}
	if baseline.cleanupOutcome != candidate.cleanupOutcome {
		b.Fatalf("pair %d cleanup outcomes differ: combined=%s split=%s", index, baseline.cleanupOutcome, candidate.cleanupOutcome)
	}
	if baseline.insertions != candidate.insertions {
		b.Fatalf("pair %d insertion counts differ: combined=%d split=%d", index, baseline.insertions, candidate.insertions)
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

func benchmarkCleanupOutcome(h *harness) string {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, outcome := range []string{"cleaned", "llm_error", "soft_deadline", "manual_skip", "unavailable", "not_requested", "cancelled"} {
		for _, entry := range h.logs {
			if strings.Contains(entry, "cleanup outcome="+outcome) {
				return outcome
			}
		}
	}
	return "unknown"
}

func reportLatencyPercentiles(b *testing.B, prefix string, samples []initializationLatencySample) {
	b.Helper()
	report := func(name string, value func(initializationLatencySample) time.Duration) {
		values := make([]float64, 0, len(samples))
		for _, sample := range samples {
			duration := value(sample)
			if duration >= 0 {
				values = append(values, float64(duration)/float64(time.Millisecond))
			}
		}
		if len(values) == 0 {
			b.Logf("%s_%s_ms=N/A", prefix, name)
			return
		}
		sort.Float64s(values)
		percentile := func(p float64) float64 {
			index := max(0, min(len(values)-1, int(math.Ceil(float64(len(values))*p))-1))
			return values[index]
		}
		p50, p95 := percentile(.50), percentile(.95)
		if p95 < p50 {
			b.Fatalf("%s_%s p95 %.4fms is below p50 %.4fms from sorted values %v", prefix, name, p95, p50, values)
		}
		b.ReportMetric(p50, prefix+"_"+name+"_p50_ms")
		b.ReportMetric(p95, prefix+"_"+name+"_p95_ms")
	}
	b.ReportMetric(float64(len(samples)), prefix+"_samples")
	report("start_to_asr_open_start", func(s initializationLatencySample) time.Duration { return s.openStart })
	report("start_to_asr_open_complete", func(s initializationLatencySample) time.Duration { return s.open })
	report("start_to_first_pcm_captured", func(s initializationLatencySample) time.Duration { return s.captured })
	report("start_to_first_pcm_sent", func(s initializationLatencySample) time.Duration { return s.sent })
	report("start_to_first_asr_partial", func(s initializationLatencySample) time.Duration { return s.partial })
	report("stop_to_asr_final", func(s initializationLatencySample) time.Duration { return s.asrFinal })
	report("stop_to_first_visible_clean_text", func(s initializationLatencySample) time.Duration { return s.cleanText })
	report("stop_to_cleanup_complete", func(s initializationLatencySample) time.Duration { return s.cleanup })
	report("stop_to_delivery", func(s initializationLatencySample) time.Duration { return s.paste })
}

func reportInitializationCounts(b *testing.B, prefix string, samples []initializationLatencySample) {
	b.Helper()
	total := len(samples)
	mean := func(value func(initializationLatencySample) int) float64 {
		var sum int
		for _, sample := range samples {
			sum += value(sample)
		}
		return float64(sum) / float64(total)
	}
	b.ReportMetric(mean(func(s initializationLatencySample) int { return s.pcmBytes }), prefix+"_pcm_bytes_per_sample")
	b.ReportMetric(mean(func(s initializationLatencySample) int { return s.frames }), prefix+"_frames_per_sample")
	b.ReportMetric(mean(func(s initializationLatencySample) int { return s.prefillCalls }), prefix+"_prefill_calls_per_sample")
	b.ReportMetric(mean(func(s initializationLatencySample) int { return s.chatsBefore }), prefix+"_chat_requests_before_stop_per_sample")
	b.ReportMetric(mean(func(s initializationLatencySample) int { return s.chatsAfter }), prefix+"_chat_requests_after_stop_per_sample")
	b.ReportMetric(mean(func(s initializationLatencySample) int { return s.insertions }), prefix+"_insertions_per_sample")
	var visibleBeforeStop, equal, reused int
	var outcome string
	for _, sample := range samples {
		if sample.cleanTextBeforeStop {
			visibleBeforeStop++
		}
		if sample.finalEqual {
			equal++
		}
		if sample.livePolishReuse {
			reused++
		}
		if outcome == "" {
			outcome = sample.cleanupOutcome
		} else if outcome != sample.cleanupOutcome {
			outcome = "mixed"
		}
	}
	b.Logf("%s cleanup_outcome=%s final_delivered_equal=%d/%d clean_text_before_stop=%d/%d live_polish_reuse=%d/%d prefill_calls=%d chat_before_stop=%d chat_after_stop=%d insertions=%d/%d",
		prefix, outcome, equal, total, visibleBeforeStop, total, reused, total,
		int(mean(func(s initializationLatencySample) int { return s.prefillCalls })*float64(total)),
		int(mean(func(s initializationLatencySample) int { return s.chatsBefore })*float64(total)),
		int(mean(func(s initializationLatencySample) int { return s.chatsAfter })*float64(total)),
		int(mean(func(s initializationLatencySample) int { return s.insertions })*float64(total)), total)
}
