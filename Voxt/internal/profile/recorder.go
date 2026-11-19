package profile

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/eval"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

const fileASRFirstTokenReason = "file ASR and MTD return a complete result, not token events"

// Recorder owns one run's report. Its methods are safe for native log and
// resource goroutines as well as the sequential meeting pipeline.
type Recorder struct {
	mu           sync.Mutex
	report       Report
	now          func() time.Time
	started      time.Time
	finalStarted time.Time
	stageIndexes map[string]int
	signposts    *Signposts
	firstASR     sync.Once
	onFirstASR   func()
}

func NewRecorder(config Config) *Recorder {
	now := time.Now()
	signposts := NewSignposts()
	redactor := NewRedactor(config.LanguageModel.Endpoint, config.LanguageModel.Model)
	config.LanguageModel.Endpoint = ""
	config.LanguageModel.Model = ""
	report := Report{
		SchemaVersion:  1,
		Name:           config.Name,
		Status:         "running",
		StartedAt:      now,
		Config:         config,
		redactor:       redactor,
		Stages:         []Stage{},
		FirstFinalText: Unavailable[float64]("the final pass has not returned text"),
		Evaluation:     Unavailable[Evaluation]("reference not provided or evaluation did not complete"),
		Warnings:       []string{},
		Errors:         []string{},
		Tracing: TracingReport{
			CPUProfile:  "cpu.pprof",
			HeapProfile: "heap.pprof",
			NativeLog:   "native.log",
		},
	}
	if reason := signposts.UnavailableReason(); reason != "" {
		report.Tracing.Signposts = "unavailable"
		report.Tracing.SignpostsUnavailableReason = reason
	} else {
		report.Tracing.Signposts = "os_signpost intervals"
	}
	return &Recorder{
		report: report, now: time.Now, started: now,
		stageIndexes: map[string]int{}, signposts: signposts,
	}
}

func (r *Recorder) RegisterSecrets(values ...string) {
	r.report.redactor.Register(values...)
}

// SetFirstASRHook registers a callback that runs exactly once, right after
// the first ASR chunk finishes (successfully or not). The meeting profile
// run uses it for the first_request load-gate snapshot.
func (r *Recorder) SetFirstASRHook(hook func()) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.onFirstASR = hook
}

func (r *Recorder) Clean(text string) string {
	return r.report.redactor.Clean(text)
}

func (r *Recorder) CleanError(err error) error {
	if err == nil {
		return nil
	}
	return errors.New(r.Clean(err.Error()))
}

// HashText identifies exactly the text stored beside the hash.
func HashText(text string) string {
	digest := sha256.Sum256([]byte(text))
	return hex.EncodeToString(digest[:])
}

func milliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

func optionalMilliseconds(duration time.Duration, reason string) OptionalFloat64 {
	if duration <= 0 {
		return Unavailable[float64](reason)
	}
	return Available(milliseconds(duration))
}

type stageTiming struct {
	stage       Stage
	started     time.Time
	endSignpost func()
}

func (r *Recorder) beginStage(name, track string, audio time.Duration) stageTiming {
	r.mu.Lock()
	index := r.stageIndexes[name]
	r.stageIndexes[name] = index + 1
	started := r.now()
	if name == "final_pass" {
		r.finalStarted = started
	}
	r.mu.Unlock()
	stage := Stage{Name: name, Track: track, Index: index}
	if audio > 0 {
		audioMS := audio.Milliseconds()
		stage.AudioMS = &audioMS
	}
	return stageTiming{stage: stage, started: started, endSignpost: r.signposts.Begin(name)}
}

func (r *Recorder) finishStage(timing stageTiming, err error) {
	timing.endSignpost()
	timing.stage.WallMS = milliseconds(r.now().Sub(timing.started))
	if timing.stage.AudioMS != nil && *timing.stage.AudioMS > 0 {
		rtf := timing.stage.WallMS / float64(*timing.stage.AudioMS)
		timing.stage.RTF = &rtf
	}
	if err != nil {
		timing.stage.Error = r.Clean(err.Error())
	}
	if timing.stage.Warning != "" {
		timing.stage.Warning = r.Clean(timing.stage.Warning)
	}
	r.mu.Lock()
	r.report.Stages = append(r.report.Stages, timing.stage)
	r.mu.Unlock()
}

// Measure records one named interval around operation.
func (r *Recorder) Measure(name string, audio time.Duration, operation func() error) (err error) {
	timing := r.beginStage(name, "", audio)
	defer func() { r.finishStage(timing, err) }()
	return operation()
}

// SetLastAudio fills decode's duration once decoding has revealed it.
func (r *Recorder) SetLastAudio(audio time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.report.Stages) == 0 || audio <= 0 {
		return
	}
	stage := &r.report.Stages[len(r.report.Stages)-1]
	audioMS := audio.Milliseconds()
	stage.AudioMS = &audioMS
	rtf := stage.WallMS / float64(audioMS)
	stage.RTF = &rtf
}

func requestAudioDuration(request asr.FileRequest) time.Duration {
	rate := request.SampleRate
	if rate <= 0 {
		rate = asr.SampleRate
	}
	return asr.Duration(request.Audio, rate)
}

func requestResultText(result asr.Result) string {
	if strings.TrimSpace(result.Text) != "" || len(result.Segments) == 0 {
		return result.Text
	}
	parts := make([]string, 0, len(result.Segments))
	for _, segment := range result.Segments {
		parts = append(parts, segment.Text)
	}
	return strings.Join(parts, "\n")
}

func (r *Recorder) finishASR(timing stageTiming, result asr.Result, err error) {
	text := r.Clean(requestResultText(result))
	timing.stage.Text = text
	timing.stage.TextSHA256 = HashText(text)
	acquire := optionalMilliseconds(result.Acquire, "the ASR provider does not report acquire time")
	latency := optionalMilliseconds(result.Latency, "the ASR provider does not report Result.Latency")
	timing.stage.QueueAcquire = &acquire
	timing.stage.ResultLatency = &latency
	firstToken := Unavailable[float64](fileASRFirstTokenReason)
	timing.stage.ASRFirstToken = &firstToken
	if result.Draft != nil {
		rate := 0.0
		if result.Draft.Proposed > 0 {
			rate = float64(result.Draft.Accepted) / float64(result.Draft.Proposed)
		}
		timing.stage.Draft = &DraftReport{
			VerifyRounds: result.Draft.VerifyRounds,
			Proposed:     result.Draft.Proposed,
			Accepted:     result.Draft.Accepted,
			AcceptRate:   rate,
		}
	}
	if err == nil && strings.TrimSpace(text) != "" {
		r.mu.Lock()
		if r.report.FirstFinalText.Value == nil && !r.finalStarted.IsZero() {
			r.report.FirstFinalText = Available(milliseconds(r.now().Sub(r.finalStarted)))
		}
		r.mu.Unlock()
	}
	r.finishStage(timing, err)
	r.firstASR.Do(func() {
		r.mu.Lock()
		hook := r.onFirstASR
		r.mu.Unlock()
		if hook != nil {
			hook()
		}
	})
}

type timedFile struct {
	recorder *Recorder
	track    string
	file     asr.FileTranscriber
}

func (f timedFile) Caps() asr.Caps { return f.file.Caps() }

func (f timedFile) Transcribe(ctx context.Context, request asr.FileRequest) (asr.Result, error) {
	timing := f.recorder.beginStage("asr_chunk", f.track, requestAudioDuration(request))
	result, err := f.file.Transcribe(ctx, request)
	f.recorder.finishASR(timing, result, err)
	return result, err
}

func (r *Recorder) WrapFile(track string, file asr.FileTranscriber) asr.FileTranscriber {
	return timedFile{recorder: r, track: track, file: file}
}

type timedDiarizer struct {
	recorder *Recorder
	track    string
	diarizer asr.Diarizer
}

func (d timedDiarizer) Caps() asr.Caps { return d.diarizer.Caps() }

func (d timedDiarizer) Diarize(ctx context.Context, request asr.FileRequest) (asr.Result, error) {
	timing := d.recorder.beginStage("diarization", d.track, requestAudioDuration(request))
	result, err := d.diarizer.Diarize(ctx, request)
	for _, segment := range result.Segments {
		text := d.recorder.Clean(segment.Text)
		timing.stage.Segments = append(timing.stage.Segments, SegmentTiming{
			StartMS: segment.Start.Milliseconds(), EndMS: segment.End.Milliseconds(),
			Speaker: segment.Speaker, Text: text, TextSHA256: HashText(text),
			InferenceDuration: Unavailable[float64]("MTD returns all segments from one inference"),
		})
	}
	d.recorder.finishASR(timing, result, err)
	return result, err
}

func (r *Recorder) WrapDiarizer(track string, diarizer asr.Diarizer) asr.Diarizer {
	return timedDiarizer{recorder: r, track: track, diarizer: diarizer}
}

type timedVAD struct {
	recorder *Recorder
	model    vad.Model
}

func (v timedVAD) Speech(ctx context.Context, pcm []byte, rate int) ([]vad.Segment, error) {
	timing := v.recorder.beginStage("vad", "", asr.Duration(pcm, rate))
	segments, err := v.model.Speech(ctx, pcm, rate)
	v.recorder.finishStage(timing, nil)
	return segments, err
}

func (v timedVAD) Open(ctx context.Context) (vad.ModelStream, error) {
	return v.model.Open(ctx)
}

func (r *Recorder) WrapVAD(model vad.Model) vad.Model {
	return timedVAD{recorder: r, model: model}
}

type timedChat struct {
	recorder *Recorder
	kind     string
	chat     interface {
		Chat(context.Context, llm.Request) (llm.Response, error)
	}
}

func (c timedChat) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	timing := c.recorder.beginStage(c.kind, "", 0)
	response, err := c.chat.Chat(ctx, request)
	c.recorder.RegisterSecrets(response.Model)
	if err == nil && c.kind == "correction" {
		err = validateCorrectionResponse(response.Text)
	}
	timing.stage.Text = c.recorder.Clean(response.Text)
	timing.stage.TextSHA256 = HashText(timing.stage.Text)
	timing.stage.Attempts = response.Attempts
	latency := optionalMilliseconds(response.Latency, "the language model does not report attempt latency")
	timing.stage.ResultLatency = &latency
	c.recorder.finishStage(timing, err)
	return response, err
}

// WrapChat records correction batches or hierarchical summary requests.
func (r *Recorder) WrapChat(kind string, chat interface {
	Chat(context.Context, llm.Request) (llm.Response, error)
}) interface {
	Chat(context.Context, llm.Request) (llm.Response, error)
} {
	return timedChat{recorder: r, kind: kind, chat: chat}
}

// SetStopToSummary records the acceptance wall from the final pass's
// start to the summary's return.
func (r *Recorder) SetStopToSummary(elapsed time.Duration) {
	r.report.StopToSummaryMS = Available(float64(elapsed) / float64(time.Millisecond))
}

func (r *Recorder) Warn(warning string) {
	r.mu.Lock()
	r.report.Warnings = append(r.report.Warnings, r.Clean(warning))
	r.mu.Unlock()
}

func (r *Recorder) SetTranscripts(asrText, finalText, summary string) {
	asrText = r.Clean(asrText)
	finalText = r.Clean(finalText)
	summary = r.Clean(summary)
	r.mu.Lock()
	r.report.ASRTranscript = TextArtifact{Text: asrText, SHA256: HashText(asrText)}
	r.report.Transcript = TextArtifact{Text: finalText, SHA256: HashText(finalText)}
	if summary != "" {
		r.report.Summary = &TextArtifact{Text: summary, SHA256: HashText(summary)}
	}
	r.mu.Unlock()
}

func scoreOf(reference, hypothesis string) Score {
	score := eval.Compare(reference, hypothesis)
	return Score{
		ReferenceTokens: score.N, Substitutions: score.Sub,
		Deletions: score.Del, Insertions: score.Ins, MixedCERWER: score.Rate(),
	}
}

func (r *Recorder) SetEvaluation(reference, asrText, finalText string) {
	r.mu.Lock()
	r.report.Evaluation = Available(Evaluation{
		ASR: scoreOf(reference, asrText), Final: scoreOf(reference, finalText),
	})
	r.mu.Unlock()
}

func (r *Recorder) SetAudioDuration(audio time.Duration) {
	r.mu.Lock()
	r.report.AudioDurationMS = audio.Milliseconds()
	r.mu.Unlock()
}

func (r *Recorder) SetResources(resources ResourceReport) {
	r.mu.Lock()
	r.report.Resources = resources
	r.mu.Unlock()
}

// Finish marks a run successful only when no stage error or warning was
// observed. A warning is never evidence of a valid benchmark run.
func (r *Recorder) Finish(runErr error) Report {
	r.mu.Lock()
	defer r.mu.Unlock()
	if runErr != nil {
		r.report.Errors = append(r.report.Errors, r.Clean(runErr.Error()))
	}
	r.report.EndedAt = r.now()
	r.report.TotalWallMS = milliseconds(r.report.EndedAt.Sub(r.started))
	if r.report.AudioDurationMS > 0 {
		r.report.TotalRTF = Available(r.report.TotalWallMS / float64(r.report.AudioDurationMS))
	} else {
		r.report.TotalRTF = Unavailable[float64]("audio duration is unknown")
	}
	if r.report.FirstFinalText.Value == nil {
		r.report.FirstFinalText.UnavailableReason = "no non-empty final ASR text was returned"
	}
	stageFailed := false
	for _, stage := range r.report.Stages {
		if stage.Error != "" {
			stageFailed = true
			break
		}
	}
	if len(r.report.Errors) > 0 || len(r.report.Warnings) > 0 || stageFailed {
		r.report.Status = "failed"
	} else {
		r.report.Status = "success"
	}
	return r.report
}

// WriteReport atomically writes a private JSON result without any API key.
func WriteReport(path string, report Report) (returnErr error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	var encoded bytes.Buffer
	encoder := json.NewEncoder(&encoded)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(report); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".result-*.json")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() {
		if returnErr != nil {
			_ = os.Remove(temporaryPath)
		}
	}()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return err
	}
	content := encoded.String()
	if report.redactor != nil {
		content = report.redactor.CleanJSON(content)
	}
	if _, err := temporary.WriteString(content); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	return nil
}
