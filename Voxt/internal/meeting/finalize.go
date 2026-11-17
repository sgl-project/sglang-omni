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
	var speech []vad.Segment
	if cfg.DisableVAD {
		speech = []vad.Segment{{Start: 0, End: total}}
	} else {
		speech, err = speechRegions(ctx, pcm, rate, cfg.VAD, cfg.VADFallback)
		if err != nil {
			return nil, err
		}
	}
	chunkP := DefaultChunkParams()
	if cfg.FinalizeChunkParams != nil {
		chunkP = *cfg.FinalizeChunkParams
	}
	chunkP.Rate = rate
	if cfg.ASR != nil {
		ta, err := cfg.ASR(ctx, src)
		if err != nil {
			return nil, err
		}
		if ta.Caps.MaxDuration > 0 {
			chunkP.CapMaxDur = ta.Caps.MaxDuration
		}
		if ta.File == nil {
			return nil, fmt.Errorf("meeting: no file ASR for %s", src)
		}
		chunks := PlanChunks(speech, total, chunkP)
		req := asr.FileRequest{
			SampleRate: rate, Lang: ta.Lang, Prompt: ta.Prompt, Hints: ta.Hints, Format: "pcm",
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
				segs, err := diarizeTrack(ctx, src, d, req, rate, pcm, speech, onProgress)
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

// diarizeChunkMax is how much of a track one MTD request covers: small
// enough that a dictation waits at most one request (the runtime lets it
// in between), and far from the native decoder's own 20-minute split.
const diarizeChunkMax = 2 * time.Minute

// mossPrompt is the instruction MOSS gets with no prompt of its own
// (moss_transcriber.cpp's kDefaultPrompt, repeated here: a prompt of ours
// replaces it, and the format must stay).
const mossPrompt = "Transcribe the audio into text. Start each segment with the start timestamp and speaker label ([S01], [S02], [S03], ...), write the corresponding spoken content, and end each segment with the ending timestamp to clearly mark the segment range."

// diarizeTrack transcribes the track with MTD in chunks, each request
// after the first carrying the last labeled lines, so a voice keeps its
// number across the boundaries (the native decoder starts its labels
// over per chunk).
func diarizeTrack(ctx context.Context, src Source, d asr.Diarizer, req asr.FileRequest, rate int, pcm []byte, speech []vad.Segment, onProgress func(float64)) ([]Segment, error) {
	p := DefaultChunkParams()
	p.Rate = rate
	p.MaxDur = diarizeChunkMax
	// Profiling experiment (round 26): pack speech regions up to the cap
	// instead of starting a new request at every pause over PauseMin.
	p.PauseMin = diarizeChunkMax
	chunks := PlanChunks(speech, len(pcm)/2, p)
	if len(chunks) == 0 {
		chunks = []Chunk{{Start: 0, End: len(pcm) / 2}}
	}
	var out []Segment
	var carry string
	var covered time.Duration // the earlier chunks returned audio up to here
	var prevText string       // the last emitted line, for the straddling one
	var tail []labeledLine    // the trailing emitted lines, for speaker anchors
	needTrim := false
	for i, ch := range chunks {
		if onProgress != nil {
			onProgress(float64(i) / float64(len(chunks)))
		}
		sub := req
		sub.Audio = pcm[ch.Start*2 : ch.End*2]
		sub.Prompt = mossPrompt
		if req.Prompt != "" {
			sub.Prompt += "\n" + req.Prompt
		}
		if carry != "" {
			sub.Prompt += "\nThis recording's earlier part ended with these labeled segments; the same voices keep their labels:\n" + carry
		}
		res, err := d.Diarize(ctx, sub)
		if err != nil {
			return nil, err
		}
		offset := time.Duration(ch.Start) * time.Second / time.Duration(rate)
		// A hard cut starts this chunk inside audio the previous chunk
		// returned; the model numbers what it hears there over, and the
		// repeated lines anchor its numbers to the earlier ones.
		anchors := anchorSpeakers(res.Segments, offset, covered, tail)
		marked := 0
		carry = ""
		for _, part := range res.Segments {
			text := strings.TrimSpace(part.Text)
			if text == "" {
				continue
			}
			marked++
			start := offset + max(part.Start, time.Duration(0))
			end := max(offset+part.End, start)
			if end <= covered {
				continue // already returned with the previous chunk
			}
			if needTrim {
				text = DedupeOverlapText(prevText, text)
				if text == "" {
					continue
				}
			}
			speaker := mtdSpeaker(part.Speaker)
			if anchored, ok := anchors[part.Speaker]; ok {
				speaker = anchored
			}
			out = append(out, Segment{
				ID: core.NewID(time.Now()), Source: src, Speaker: speaker,
				Start: start, End: end, Text: text, Final: true,
			})
			prevText = text
			needTrim = false
			if speaker != "" {
				tail = append(tail, labeledLine{words: splitWords(text), speaker: speaker})
				if len(tail) > 8 {
					tail = tail[len(tail)-8:]
				}
				carry += "[" + mtdLabel(speaker) + "] " + text + "\n"
			}
		}
		if marked == 0 && strings.TrimSpace(res.Text) != "" {
			out = append(out, Segment{ID: core.NewID(time.Now()), Source: src, Start: offset, End: offset + asr.Duration(sub.Audio, rate), Text: strings.TrimSpace(res.Text), Final: true})
		}
		if runes := []rune(carry); len(runes) > 300 { // the last lines, at most a screen
			carry = string(runes[len(runes)-300:])
		}
		covered = time.Duration(ch.End) * time.Second / time.Duration(rate)
		needTrim = true
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

// mtdLabel is a global speaker in the prompt's own format (s2 -> S02), so
// the next chunk continues the numbers it sees.
func mtdLabel(speaker string) string {
	n, ok := strings.CutPrefix(speaker, "s")
	if !ok {
		return speaker
	}
	v, err := strconv.Atoi(n)
	if err != nil || v <= 0 {
		return speaker
	}
	return fmt.Sprintf("S%02d", v)
}

// labeledLine is an emitted line's words and its global speaker.
type labeledLine struct {
	words   []string
	speaker string
}

// anchorSpeakers maps a chunk's labels to the earlier chunks' speakers.
// The model numbers the voices it hears from scratch, so a line it
// repeated from the previous chunk's audio (ending at or before covered)
// names the same voice under whatever number this chunk gave it.
func anchorSpeakers(parts []asr.SpeakerSegment, offset, covered time.Duration, tail []labeledLine) map[string]string {
	if covered <= 0 || len(tail) == 0 {
		return nil
	}
	anchors := map[string]string{}
	for _, part := range parts {
		if _, ok := anchors[part.Speaker]; ok {
			continue
		}
		if offset+part.End > covered {
			continue
		}
		words := splitWords(part.Text)
		if len(words) < 2 {
			continue // a one-word line anchors nothing: too many voices say it
		}
		for k := len(tail) - 1; k >= 0; k-- {
			if tail[k].speaker != "" && sameWords(tail[k].words, words) {
				anchors[part.Speaker] = tail[k].speaker
				break
			}
		}
	}
	return anchors
}

// sameWords says two lines match: equal, or the shorter a prefix of the
// longer (a cut can clip the repeated line).
func sameWords(a, b []string) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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

func speechRegions(ctx context.Context, pcm []byte, rate int, model vad.Model, onEnergyFallback func()) ([]vad.Segment, error) {
	if rate <= 0 {
		rate = asr.SampleRate
	}
	if model != nil {
		segs, err := model.Speech(ctx, pcm, rate)
		if err == nil && len(segs) > 0 {
			return segs, nil
		}
	}
	return batchedSpeechRegions(ctx, pcm, rate, model, onEnergyFallback), nil
}

func batchedSpeechRegions(ctx context.Context, pcm []byte, rate int, model vad.Model, onEnergyFallback func()) []vad.Segment {
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
			if model != nil && onEnergyFallback != nil {
				onEnergyFallback()
			}
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
