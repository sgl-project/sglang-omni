package meeting

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

type profileEvent struct {
	Stage           string  `json:"stage"`
	Track           string  `json:"track"`
	StartedAfterT0  float64 `json:"started_after_t0_s"`
	EndedAfterT0    float64 `json:"ended_after_t0_s"`
	TextSHA256      string  `json:"text_sha256"`
	ContentSHA256   string  `json:"content_sha256,omitempty"`
	PayloadPreview  string  `json:"payload_preview,omitempty"`
	PayloadRunes    int     `json:"payload_runes,omitempty"`
	ResponsePreview string  `json:"response_preview,omitempty"`
	ResponseRunes   int     `json:"response_runes,omitempty"`
}

type profileTimedASR struct {
	speech *omni.ASR
	events *[]profileEvent
	t0     time.Time
	track  Source
}

func (a *profileTimedASR) Caps() asr.Caps { return a.speech.Caps() }

func (a *profileTimedASR) Transcribe(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	started := time.Now()
	res, err := a.speech.Transcribe(ctx, req)
	*a.events = append(*a.events, profileEvent{Stage: "asr", Track: string(a.track), StartedAfterT0: started.Sub(a.t0).Seconds(), EndedAfterT0: time.Since(a.t0).Seconds(), TextSHA256: profileHash(res.Text)})
	return res, err
}

func (a *profileTimedASR) Diarize(ctx context.Context, req asr.FileRequest) (asr.Result, error) {
	started := time.Now()
	res, err := a.speech.Diarize(ctx, req)
	text := res.Text
	if text == "" {
		var lines []string
		for _, seg := range res.Segments {
			lines = append(lines, seg.Text)
		}
		text = strings.Join(lines, "\n")
	}
	*a.events = append(*a.events, profileEvent{Stage: "diarize", Track: string(a.track), StartedAfterT0: started.Sub(a.t0).Seconds(), EndedAfterT0: time.Since(a.t0).Seconds(), TextSHA256: profileHash(text)})
	return res, err
}

type profileChat struct {
	client *llm.Client
	events *[]profileEvent
	t0     time.Time
}

func (c *profileChat) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	if effort := os.Getenv("VOXT_SYSTEM_PROFILE_THINKING"); effort != "" {
		req.Thinking = &llm.Thinking{Mode: llm.ThinkEffort, Effort: effort}
	}
	started := time.Now()
	res, err := c.client.Chat(ctx, req)
	text := ""
	if len(req.Messages) > 0 {
		text = req.Messages[len(req.Messages)-1].Content
	}
	event := profileEvent{Stage: "correction", StartedAfterT0: started.Sub(c.t0).Seconds(), EndedAfterT0: time.Since(c.t0).Seconds(), TextSHA256: profileHash(text),
		ContentSHA256: contentHash(text), PayloadPreview: previewRunes(text, 100), PayloadRunes: len([]rune(text))}
	if err != nil {
		event.ResponsePreview = "error: " + err.Error()
	} else {
		event.ResponsePreview = previewRunes(res.Text, 100)
		event.ResponseRunes = len([]rune(res.Text))
	}
	*c.events = append(*c.events, event)
	return res, err
}

func previewRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// contentHash identifies a correction batch by its lines, not by the
// request's line numbering: an early batch reindexes its lines but has
// the same content as the final batch it lands.
func contentHash(payload string) string {
	var lines []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(payload), &lines); err != nil {
		return ""
	}
	var b strings.Builder
	for _, l := range lines {
		fmt.Fprintf(&b, "%d:%s|", len([]rune(l.Text)), l.Text)
	}
	return profileHash(b.String())
}

type profileResult struct {
	Attempt          string            `json:"attempt"`
	Arm              string            `json:"arm"`
	MainSHA          string            `json:"main_sha"`
	Thinking         string            `json:"thinking,omitempty"`
	T0ToDone         float64           `json:"t0_to_done_s"`
	ASRTotal         float64           `json:"asr_total_s"`
	TailAfterLastASR float64           `json:"tail_after_last_asr_s"`
	RawSHA256        map[string]string `json:"raw_sha256_by_track"`
	CorrectedSHA256  string            `json:"corrected_sha256"`
	RawText          string            `json:"raw_text"`
	CorrectedText    string            `json:"corrected_text"`
	Events           []profileEvent    `json:"events"`
}

func profileHash(text string) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(text)))
}

func TestSystemOverlapProfile(t *testing.T) {
	out := os.Getenv("VOXT_SYSTEM_PROFILE_OUT")
	if out == "" {
		t.Skip("real profiling is opt-in")
	}
	if err := runSystemOverlapProfile(t, out); err != nil {
		t.Fatal(err)
	}
}

func runSystemOverlapProfile(t *testing.T, out string) error {
	t.Helper()
	ctx := context.Background()
	serverBinary := os.Getenv("VOXT_SYSTEM_PROFILE_SERVER")
	modelDir := os.Getenv("VOXT_SYSTEM_PROFILE_MODEL")
	draftDir := os.Getenv("VOXT_SYSTEM_PROFILE_DRAFT")
	endpoint := os.Getenv("VOXT_SYSTEM_PROFILE_LLM_URL")
	key := os.Getenv("VOXT_SYSTEM_PROFILE_LLM_KEY")
	model := os.Getenv("VOXT_SYSTEM_PROFILE_LLM_MODEL")
	if serverBinary == "" || modelDir == "" || draftDir == "" || endpoint == "" || key == "" || model == "" {
		return fmt.Errorf("profile configuration is incomplete")
	}
	preset, err := (llm.ProviderConfig{ID: llm.Custom, BaseURL: endpoint, Model: model}).Resolve()
	if err != nil {
		return fmt.Errorf("profile language model setup failed")
	}
	backend := llm.NewOpenAI(preset, key, nil)
	client := &llm.Client{Backend: backend}
	runtime := omni.NewRuntime(omni.Config{Binary: serverBinary, StallTimeout: 600 * time.Second})
	defer runtime.Close()
	speech, err := omni.NewASR(runtime, omni.ASRMoss, modelDir)
	if err != nil {
		return fmt.Errorf("profile speech model setup failed")
	}
	speech = speech.WithDraft(draftDir)
	t0 := time.Now()
	var asrEvents, chatEvents []profileEvent
	timed := &profileTimedASR{speech: speech, t0: t0, events: &asrEvents}
	chat := &profileChat{client: client, t0: t0, events: &chatEvents}
	root := t.TempDir()
	repo, err := sqlitestore.Open(ctx, filepath.Join(root, "db"))
	if err != nil {
		return fmt.Errorf("profile store setup failed")
	}
	defer repo.Close()
	meetDir := filepath.Join(root, "meetings", "m1")
	if err := os.MkdirAll(meetDir, 0o755); err != nil {
		return fmt.Errorf("profile meeting setup failed")
	}
	for _, item := range []struct{ env, name string }{{"VOXT_SYSTEM_PROFILE_MIC", "mic.wav"}, {"VOXT_SYSTEM_PROFILE_SYSTEM", "system.wav"}} {
		pcm, err := os.ReadFile(os.Getenv(item.env))
		if err != nil {
			return fmt.Errorf("profile input is unavailable")
		}
		if err := os.WriteFile(filepath.Join(meetDir, item.name), pcm, 0o444); err != nil {
			return fmt.Errorf("profile input copy failed")
		}
	}
	mt, err := repo.Meetings().CreateMeeting(ctx, store.Meeting{ID: "m1", CreatedAt: time.Now(), Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Audio: "m1", DurationMS: 600000})
	if err != nil {
		return fmt.Errorf("profile meeting creation failed")
	}
	cfg := Config{
		Store: repo.Meetings(), Dir: filepath.Join(root, "meetings"),
		Settings: func() settings.Meeting { return settings.Meeting{LLM: "profile"} },
		Privacy:  func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		ASR: func(_ context.Context, src Source) (TrackASR, error) {
			timed.track = src
			caps := timed.Caps()
			caps.MaxDuration = 60 * time.Second
			return TrackASR{Caps: caps, File: timed}, nil
		},
		Diarizer: func(context.Context) (asr.Diarizer, error) { return timed, nil },
		VAD:      wholeSpeech{},
		LLM:      func(context.Context, string) (SummaryChat, error) { return chat, nil },
	}
	manager := NewManager(cfg)
	manager.cfg.correctOverlap = manager.startCorrectOverlap(mt.ID)
	t0 = time.Now()
	timed.t0, chat.t0 = t0, t0
	if err := FinalizePass(ctx, manager.cfg, mt, nil); err != nil {
		return fmt.Errorf("profile final pass failed")
	}
	lastASR := time.Now()
	_, rawSegments, err := repo.Meetings().Meeting(ctx, mt.ID)
	if err != nil {
		return fmt.Errorf("profile raw transcript read failed")
	}
	rawText := profileSegmentsText(rawSegments)
	if err := manager.CorrectTranscript(ctx, mt.ID); err != nil {
		return fmt.Errorf("profile correction failed")
	}
	done := time.Now()
	_, correctedSegments, err := repo.Meetings().Meeting(ctx, mt.ID)
	if err != nil {
		return fmt.Errorf("profile corrected transcript read failed")
	}
	correctedText := profileSegmentsText(correctedSegments)
	result := profileResult{
		Attempt: os.Getenv("VOXT_SYSTEM_PROFILE_ATTEMPT"), Arm: os.Getenv("VOXT_SYSTEM_PROFILE_ARM"), MainSHA: os.Getenv("VOXT_SYSTEM_PROFILE_MAIN"),
		Thinking: os.Getenv("VOXT_SYSTEM_PROFILE_THINKING"),
		T0ToDone: done.Sub(t0).Seconds(), ASRTotal: lastASR.Sub(t0).Seconds(), TailAfterLastASR: done.Sub(lastASR).Seconds(),
		RawSHA256: map[string]string{"all": profileHash(rawText)}, CorrectedSHA256: profileHash(correctedText),
		RawText: rawText, CorrectedText: correctedText,
		Events: append(append([]profileEvent(nil), asrEvents...), chatEvents...),
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("profile result encoding failed")
	}
	if err := os.WriteFile(out, append(data, '\n'), 0o600); err != nil {
		return fmt.Errorf("profile result write failed")
	}
	return nil
}

func profileSegmentsText(segs []store.MeetingSegment) string {
	var lines []string
	for _, seg := range segs {
		lines = append(lines, string(seg.Source)+"\t"+seg.Text)
	}
	return strings.Join(lines, "\n")
}
