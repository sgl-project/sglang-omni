package main

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/speaker"
	"github.com/sgl-project/sglang-omni/Voxt/internal/speakerrt"
)

// Run on a Mac before releasing live labels, with VOXTNEW_EVAL=1 and
// VOXTNEW_SPEAKER_RUNTIME (docs/eval.md). Reads only licensed synthetic fixtures.
func TestEvalMeetingLiveSpeakers(t *testing.T) {
	e := evalSetup(t)
	binary := os.Getenv("VOXTNEW_SPEAKER_RUNTIME")
	if binary == "" {
		t.Skip("set VOXTNEW_SPEAKER_RUNTIME to the built speaker executable")
	}
	spec, _ := models.Lookup(models.SpeakerDiarization)
	dir := models.Installed(e.hub, spec)
	if dir == "" {
		t.Skip("install the pinned Nemotron 3 Diarization model before the Mac speaker eval")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	before := time.Now()
	stream, err := (speakerrt.Model{Binary: binary, Dir: dir}).Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	load := time.Since(before)
	ids := []string{"zh-filler-1", "en-1", "zh-filler-2", "zh-filler-4", "en-3", "mix-3", "mix-1", "en-5", "dict-2"}
	type turn struct {
		voice      string
		start, end time.Duration
	}
	var audio []byte
	var turns []turn
	for _, id := range ids {
		clip, _ := e.set.Clip(id)
		pcm := clip.PCM[:min(len(clip.PCM), asr.BytesPerSecond*4)]
		start := time.Duration(len(audio)) * time.Second / asr.BytesPerSecond
		audio = append(audio, pcm...)
		turns = append(turns, turn{clip.Voice, start, time.Duration(len(audio)) * time.Second / asr.BytesPerSecond})
		audio = append(audio, make([]byte, asr.BytesPerSecond)...)
	}
	audio = append(audio, make([]byte, asr.BytesPerSecond*2)...)
	var frames []speaker.Frame
	started := time.Now()
	for i := 0; i < len(audio); i += 6400 {
		got, err := stream.Push(ctx, audio[i:min(i+6400, len(audio))])
		if err != nil {
			t.Fatal(err)
		}
		frames = append(frames, got...)
	}
	elapsed := time.Since(started)
	voices := map[string]int{}
	report := fmt.Sprintf("Model: %s@%s, CPU only. Load %v, audio %.1f s, processing %v. UI, capture, ASR and OCR excluded.\n\n", spec.Repo, spec.Revision, load.Round(time.Millisecond), float64(len(audio))/asr.BytesPerSecond, elapsed.Round(time.Millisecond))
	for _, turn := range turns {
		counts := map[int]int{}
		total := 0
		for _, frame := range frames {
			if frame.Start < turn.start || frame.End > turn.end {
				continue
			}
			total++
			best, second := 0, float32(0)
			for i := 1; i < 8; i++ {
				if frame.Scores[i] > frame.Scores[best] {
					second = frame.Scores[best]
					best = i
				} else {
					second = max(second, frame.Scores[i])
				}
			}
			if frame.Scores[best] >= .7 && second < .35 {
				counts[best]++
			}
		}
		winner, n := -1, 0
		for id, count := range counts {
			if count > n {
				winner, n = id, count
			}
		}
		if winner < 0 {
			t.Errorf("%s: no confident speaker", turn.voice)
			continue
		}
		if old, ok := voices[turn.voice]; ok && old != winner {
			t.Errorf("%s changed slot %d -> %d", turn.voice, old, winner)
		}
		for voice, id := range voices {
			if voice != turn.voice && id == winner {
				t.Errorf("%s and %s merged", voice, turn.voice)
			}
		}
		voices[turn.voice] = winner
		report += fmt.Sprintf("- %s: slot %d, %d/%d frames confidently assigned.\n", turn.voice, winner, n, total)
	}
	t.Log(report)
	e.report(t, "meeting-live-speakers.md", report+"\nThis three-voice smoke check does not establish quality for similar voices, overlap or short turns. The four-voice probe merged two female voices. Run the real-meeting and complete-App checks in manual-test.md.\n")
}
