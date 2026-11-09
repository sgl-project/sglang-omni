package meeting

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

const (
	vadWindowBatch  = 10 * time.Minute
	vadOverlapBatch = time.Second
)

// ErrFinalizeRetry means the final pass stopped mid-track; live segments stay.
var ErrFinalizeRetry = errors.New("meeting: final pass incomplete")

// FinalizePass reruns file-mode ASR on archived WAVs and replaces live segments (C1-3a).
func FinalizePass(ctx context.Context, cfg Config, mt store.Meeting, onProgress func(float64)) error {
	if cfg.Settings().NoFinalPass {
		return nil
	}
	mode := fromStoreMode(mt.Mode)
	sources := mode.Sources()
	if len(sources) == 0 {
		return nil
	}
	audioDir := filepath.Join(cfg.Dir, mt.Audio)
	var holdRelease func()
	if cfg.ModelHold != nil {
		holdRelease = cfg.ModelHold.Hold()
		defer holdRelease()
	}
	var first error
	trackN := len(sources)
	for i, src := range sources {
		base := float64(i) / float64(trackN)
		err := finalizeTrack(ctx, cfg, mt, audioDir, src, func(p float64) {
			if onProgress != nil {
				onProgress(base + p/float64(trackN))
			}
		})
		if err != nil && first == nil {
			first = err
		}
	}
	if first != nil {
		_ = markFinalizeFailed(ctx, cfg, mt)
		return first
	}
	if err := applyAudioRetention(ctx, cfg, mt, mode, true); err != nil {
		return err
	}
	if !cfg.Privacy().KeepAudio && !mode.KeepsAudio() {
		mt.Audio = ""
		mt.AudioUntil = time.Time{}
	}
	mt.Status = store.MeetingDone
	return cfg.Store.UpdateMeeting(ctx, mt)
}

func finalizeTrack(ctx context.Context, cfg Config, mt store.Meeting, dir string, src Source, onProgress func(float64)) error {
	path := filepath.Join(dir, string(src)+".wav")
	pcm, rate, err := readTrackPCM(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	speech, err := speechRegions(ctx, pcm, rate, cfg.VAD)
	if err != nil {
		return err
	}
	total := len(pcm) / 2
	if total == 0 {
		return nil
	}
	chunkP := DefaultChunkParams()
	chunkP.Rate = rate
	if cfg.ASR != nil {
		ta, err := cfg.ASR(ctx, src)
		if err != nil {
			return err
		}
		if ta.Caps.MaxDuration > 0 {
			chunkP.CapMaxDur = ta.Caps.MaxDuration
		}
		if ta.File == nil {
			return fmt.Errorf("meeting: no file ASR for %s", src)
		}
		chunks := PlanChunks(speech, total, chunkP)
		req := asr.FileRequest{
			SampleRate: rate, Lang: ta.Lang, Prompt: ta.Prompt, Hints: ta.Hints, Format: "pcm",
		}
		segs, err := transcribeChunks(ctx, src, ta.File, req, rate, pcm, chunks, onProgress)
		if err != nil {
			return err
		}
		rows := make([]store.MeetingSegment, 0, len(segs))
		for _, s := range segs {
			rows = append(rows, segmentToStore(s, src))
		}
		return cfg.Store.ReplaceSegments(ctx, mt.Seq, toSegmentSource(src), rows)
	}
	return fmt.Errorf("meeting: ASR not configured")
}

func transcribeChunks(ctx context.Context, src Source, ft asr.FileTranscriber, req asr.FileRequest, rate int, pcm []byte, chunks []Chunk, onProgress func(float64)) ([]Segment, error) {
	var out []Segment
	var prevText string
	for i, ch := range chunks {
		if onProgress != nil && len(chunks) > 0 {
			onProgress(float64(i) / float64(len(chunks)))
		}
		req.Audio = pcm[ch.Start*2 : ch.End*2]
		if req.SampleRate == 0 {
			req.SampleRate = rate
		}
		res, err := ft.Transcribe(ctx, req)
		if err != nil {
			return out, fmt.Errorf("chunk %d: %w", i, err)
		}
		text := res.Text
		if i > 0 {
			text = DedupeOverlapText(prevText, text)
		}
		prevText = text
		if text == "" {
			continue
		}
		start := time.Duration(ch.Start) * time.Second / time.Duration(rate)
		end := time.Duration(ch.End) * time.Second / time.Duration(rate)
		seg := Segment{
			ID: core.NewID(time.Now()), Source: src,
			Start: start, End: end, Text: text, Final: true,
		}
		out = append(out, seg)
	}
	if onProgress != nil {
		onProgress(1)
	}
	return out, nil
}

func segmentToStore(s Segment, src Source) store.MeetingSegment {
	s.Source = src
	sp := s.Speaker
	if src == SourceMic && sp == "" {
		sp = "me"
	}
	return store.MeetingSegment{
		Source: toSegmentSource(src), Speaker: sp,
		StartMS: s.Start.Milliseconds(), EndMS: s.End.Milliseconds(),
		Text: s.Text, Translation: s.Translation, Final: true,
	}
}

func speechRegions(ctx context.Context, pcm []byte, rate int, model vad.Model) ([]vad.Segment, error) {
	if rate <= 0 {
		rate = asr.SampleRate
	}
	if model != nil {
		segs, err := model.Speech(ctx, pcm, rate)
		if err == nil && len(segs) > 0 {
			return segs, nil
		}
	}
	return batchedSpeechRegions(ctx, pcm, rate, model), nil
}

func batchedSpeechRegions(ctx context.Context, pcm []byte, rate int, model vad.Model) []vad.Segment {
	win := int(vadWindowBatch.Seconds() * float64(rate))
	ovl := int(vadOverlapBatch.Seconds() * float64(rate))
	samples := len(pcm) / 2
	var merged vad.Analysis
	for start := 0; start < samples; {
		end := min(start+win, samples)
		chunk := pcm[start*2 : end*2]
		var part vad.Analysis
		if model != nil {
			if segs, err := model.Speech(ctx, chunk, rate); err == nil && len(segs) > 0 {
				energy := vad.Analyze(chunk, rate, vad.Defaults())
				part = vad.FromSegments(segs, end-start, energy)
			}
		}
		if len(part.Segments) == 0 {
			part = vad.Analyze(chunk, rate, vad.Defaults())
		}
		for i := range part.Segments {
			part.Segments[i].Start += start
			part.Segments[i].End += start
		}
		merged = vad.Merge(merged, part)
		if end >= samples {
			break
		}
		start = max(end-ovl, start+1)
	}
	return merged.Segments
}

func readTrackPCM(path string) ([]byte, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	pcm, rate, ch, err := asr.ReadWAV(data)
	if err != nil {
		return nil, 0, err
	}
	if ch != 1 {
		return nil, 0, fmt.Errorf("meeting: want mono WAV")
	}
	return pcm, rate, nil
}

func markFinalizeFailed(ctx context.Context, cfg Config, mt store.Meeting) error {
	mode := fromStoreMode(mt.Mode)
	keep := cfg.Privacy().KeepAudio || mode.KeepsAudio()
	now := time.Now()
	if cfg.Now != nil {
		now = cfg.Now()
	}
	if keep {
		mt.AudioUntil = now.Add(7 * 24 * time.Hour)
	}
	mt.Status = store.MeetingInterrupted
	return cfg.Store.UpdateMeeting(ctx, mt)
}

func applyAudioRetention(ctx context.Context, cfg Config, mt store.Meeting, mode Mode, success bool) error {
	keep := cfg.Privacy().KeepAudio || mode.KeepsAudio()
	if !success || keep {
		return nil
	}
	dir := filepath.Join(cfg.Dir, mt.Audio)
	_ = os.RemoveAll(dir)
	mt.Audio = ""
	mt.AudioUntil = time.Time{}
	return cfg.Store.UpdateMeeting(ctx, mt)
}
