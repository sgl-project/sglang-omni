package meeting

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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

// ErrNoDiarizer means an import found no MOSS-Transcribe-Diarize: an
// imported file is one mixed track, so plain transcription would lose the
// speakers it was imported for (no silent fallback).
var ErrNoDiarizer = errors.New("meeting: MOSS-Transcribe-Diarize is not installed")

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
		defer func() {
			if holdRelease != nil {
				holdRelease()
			}
		}()
	}
	var first error
	trackN := len(sources)
	done := map[Source][]Segment{}
	for i, src := range sources {
		if src == SourceSystem && mode.Diarize() && holdRelease != nil {
			// MTD cannot start until the selected speech model's hold lets its server stop.
			holdRelease()
			holdRelease = nil
		}
		base := float64(i) / float64(trackN)
		segs, err := finalizeTrack(ctx, cfg, audioDir, src, mode, func(p float64) {
			if onProgress != nil {
				onProgress(base + p/float64(trackN))
			}
		})
		if err != nil {
			if first == nil {
				first = err
			}
			continue
		}
		if segs != nil {
			done[src] = segs
		}
	}
	// Speech from the speakers reaches the microphone too (#3982).
	mic, micOK := done[SourceMic]
	if sys, ok := done[SourceSystem]; ok && micOK {
		done[SourceMic] = dropEchoes(mic, sys)
	}
	for _, src := range sources {
		segs, ok := done[src]
		if !ok {
			continue
		}
		rows := make([]store.MeetingSegment, 0, len(segs))
		for _, s := range segs {
			rows = append(rows, segmentToStore(s, src))
		}
		if err := cfg.Store.ReplaceSegments(ctx, mt.Seq, toSegmentSource(src), rows); err != nil && first == nil {
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

// finalizeTrack transcribes one track's WAV again; nil segments mean the
// track has nothing to replace (no file, or no audio).
func finalizeTrack(ctx context.Context, cfg Config, dir string, src Source, mode Mode, onProgress func(float64)) ([]Segment, error) {
	diarize := mode.Diarize()
	path := filepath.Join(dir, string(src)+".wav")
	pcm, rate, err := readTrackPCM(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	total := len(pcm) / 2
	if total == 0 {
		return nil, nil
	}
	chunkP := DefaultChunkParams()
	chunkP.Rate = rate
	if cfg.ASR != nil {
		ta, aerr := cfg.ASR(ctx, src)
		req := asr.FileRequest{SampleRate: rate, Format: "pcm"}
		if aerr == nil {
			if ta.Caps.MaxDuration > 0 {
				chunkP.CapMaxDur = ta.Caps.MaxDuration
			}
			req.Lang, req.Prompt, req.Hints = ta.Lang, ta.Prompt, ta.Hints
		}
		if diarize && src == SourceSystem {
			var d asr.Diarizer
			if cfg.Diarizer != nil {
				var err error
				d, err = cfg.Diarizer(ctx)
				if err != nil {
					return nil, err
				}
			}
			if d != nil {
				// MTD alone does this track: the selected model's missing
				// file ASR (aerr) must not stop an import.
				segs, err := diarizeTrack(ctx, src, d, req, rate, pcm, onProgress)
				if err != nil {
					return nil, err
				}
				if segs == nil {
					segs = []Segment{}
				}
				return segs, nil
			}
			if mode == ModeImport {
				return nil, ErrNoDiarizer
			}
		}
		if aerr != nil {
			return nil, aerr
		}
		if ta.File == nil {
			return nil, fmt.Errorf("meeting: no file ASR for %s", src)
		}
		speech, err := speechRegions(ctx, pcm, rate, cfg.VAD)
		if err != nil {
			return nil, err
		}
		chunks := PlanChunks(speech, total, chunkP)
		segs, err := transcribeChunks(ctx, src, ta.File, req, rate, pcm, chunks, onProgress)
		if err != nil {
			return nil, err
		}
		if segs == nil {
			segs = []Segment{} // the track has audio but no words: still replace the live text
		}
		return segs, nil
	}
	return nil, fmt.Errorf("meeting: ASR not configured")
}

func diarizeTrack(ctx context.Context, src Source, d asr.Diarizer, req asr.FileRequest, rate int, pcm []byte, onProgress func(float64)) ([]Segment, error) {
	req.Audio = pcm
	res, err := d.Diarize(ctx, req)
	if err != nil {
		return nil, err
	}
	var out []Segment
	for _, part := range res.Segments {
		text := strings.TrimSpace(part.Text)
		if text == "" {
			continue
		}
		start := max(part.Start, time.Duration(0))
		out = append(out, Segment{
			ID: core.NewID(time.Now()), Source: src, Speaker: mtdSpeaker(part.Speaker),
			Start: start, End: max(part.End, start), Text: text, Final: true,
		})
	}
	if len(out) == 0 && strings.TrimSpace(res.Text) != "" {
		out = append(out, Segment{ID: core.NewID(time.Now()), Source: src, End: asr.Duration(pcm, rate), Text: strings.TrimSpace(res.Text), Final: true})
	}
	if onProgress != nil {
		onProgress(1)
	}
	return out, nil
}

// mtdSpeaker is MTD's S01 (or S1) as the store's s1; an imported file can
// have more voices than a call's other side, so any S<number> from 1 maps.
func mtdSpeaker(speaker string) string {
	n, ok := strings.CutPrefix(speaker, "S")
	if !ok {
		return ""
	}
	n = strings.TrimLeft(n, "0")
	if n == "" {
		return ""
	}
	if _, err := strconv.Atoi(n); err != nil {
		return ""
	}
	return "s" + n
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
	// The failure may be the context's own cancel: the interrupted status
	// must still land, or the meeting stays "finishing" forever.
	ctx = context.WithoutCancel(ctx)
	mode := fromStoreMode(mt.Mode)
	now := time.Now()
	if cfg.Now != nil {
		now = cfg.Now()
	}
	if !cfg.Privacy().KeepAudio && !mode.KeepsAudio() {
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
