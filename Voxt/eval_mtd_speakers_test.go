package main

// MTD speaker/diarization checks, split from eval_mtd_test.go to keep
// every test file under 800 lines (internal/archtest TestFileSizes).

import (
	"context"
	"fmt"
	"maps"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/eval"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
)

func TestEvalMTDSpeakers(t *testing.T) {
	e := evalSetup(t)
	dir := os.Getenv("VOXTNEW_MTD_DIR")
	if dir == "" {
		t.Skip("set VOXTNEW_MTD_DIR to the MTD snapshot")
	}
	exe, _ := os.Executable()
	server, err := omni.Locate(exe)
	if err != nil {
		t.Fatal(err)
	}
	rt := omni.NewRuntime(omni.Config{Binary: server, Log: func(level, msg string) {
		if level != "info" && level != "debug" {
			t.Logf("omni %s: %s", level, msg)
		}
	}})
	defer rt.Close()
	ctx := context.Background()
	a, err := omni.NewASR(rt, models.ASRMoss, dir)
	if err != nil {
		t.Fatal(err)
	}
	a = a.WithInstruction("")
	sets := [][]string{
		{"zh-filler-1", "en-1"},
		{"zh-filler-1", "zh-filler-2", "en-1"},
		{"zh-filler-1", "zh-filler-2", "zh-filler-3", "en-1"},
	}
	var md strings.Builder
	fmt.Fprintf(&md, "# MTD speaker labels (%d rounds)\n\n%s\n\n", mtdRounds, mtdMachineNote())
	tb := table{head: []string{"clip", "layout (voice: seconds)", "segments", "labels right", "error rate", "Final (median)", "load average"}}
	for _, ids := range sets {
		pcm, ranges, ref := joinSpeakers(t, e.set, ids)
		name := fmt.Sprintf("%d speakers (%s)", len(ids), strings.Join(ids, " + "))
		var layout []string
		for _, r := range ranges {
			layout = append(layout, fmt.Sprintf("%s: %.1f–%.1f", r.voice, r.start, r.end))
		}
		var segs, right int
		var score eval.Score
		var times []time.Duration
		var loads []string
		for round := 0; round < mtdRounds; round++ {
			loads = append(loads, loadAvg())
			res, err := a.Transcribe(ctx, asr.FileRequest{Audio: pcm, Format: "pcm"})
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			times = append(times, res.Latency)
			got := parseMTDSegments(res.Text)
			s, r := speakerAccuracy(got, ranges)
			segs, right = segs+s, right+r
			score = score.Add(eval.Compare(ref, joinMTDText(res.Text)))
			t.Logf("%s round %d: %d segments, %d right, text %q", name, round+1, s, r, mdText(res.Text))
		}
		tb.add(name, strings.Join(layout, "; "), segs, fmt.Sprintf("%d of %d over %d rounds", right, segs, mtdRounds), pct(score.Rate()), ms(median(times)), loadRange(loads))
	}
	md.WriteString(tb.String())
	md.WriteString("\nA label is right when the segment's speaker (the best one-to-one assignment of labels to voices, so a voice split into two labels cannot score both right) matches the voice speaking at the segment's middle. The error rate is the joined text's against the clips' words.\n")
	e.report(t, "mtd-speakers.md", md.String())
}

// mtdRange is one speaker's stretch of a joined clip, in seconds.
type mtdRange struct {
	voice      string
	start, end float64
}

// mtdSegment is one parsed [start][Sxx]text[end] segment of MTD's answer.
type mtdSegment struct {
	start, end float64
	label      string
	text       string
}

// joinSpeakers joins the clips' PCM with 1.5 s of silence between them,
// and returns the layout (which voice is where) and the joined words.
func joinSpeakers(t *testing.T, set *eval.Set, ids []string) (pcm []byte, ranges []mtdRange, ref string) {
	t.Helper()
	const gap = 3 * asr.SampleRate / 2 // 1.5 s of silence
	var texts []string
	at := 0
	for _, id := range ids {
		c, ok := set.Clip(id)
		if !ok {
			t.Fatalf("no clip %q", id)
		}
		ranges = append(ranges, mtdRange{voice: c.Voice, start: float64(at) / asr.SampleRate, end: float64(at+len(c.PCM)/2) / asr.SampleRate})
		pcm = append(pcm, c.PCM...)
		at += len(c.PCM) / 2
		if id != ids[len(ids)-1] {
			pcm = append(pcm, make([]byte, gap*2)...)
			at += gap
		}
		texts = append(texts, c.Text)
	}
	return pcm, ranges, strings.Join(texts, " ")
}

// mtdSegmentPattern is MTD's [start][speaker]text[end] segment (the
// server's kSegmentPattern).
var mtdSegmentPattern = regexp.MustCompile(`\[(\d+(?:[.,]\d+)?)\]\[(S\d+)\]([\s\S]*?)\[(\d+(?:[.,]\d+)?)\]`)

// parseMTDSegments parses the segments of MTD's diarized answer.
func parseMTDSegments(text string) []mtdSegment {
	var out []mtdSegment
	for _, m := range mtdSegmentPattern.FindAllStringSubmatch(text, -1) {
		start, _ := strconv.ParseFloat(strings.ReplaceAll(m[1], ",", "."), 64)
		end, _ := strconv.ParseFloat(strings.ReplaceAll(m[4], ",", "."), 64)
		out = append(out, mtdSegment{start: start, end: end, label: m[2], text: m[3]})
	}
	return out
}

// joinMTDText is the answer's text without the segment tags.
func joinMTDText(text string) string {
	return strings.TrimSpace(mtdSegmentPattern.ReplaceAllString(text, "$3"))
}

// speakerAccuracy scores the labels: each segment's truth is the voice at
// its middle, and the score is the best one-to-one assignment of labels
// to voices, so a voice split into two labels cannot score both right.
// It returns the segment count and how many the assignment gets right.
func speakerAccuracy(segs []mtdSegment, ranges []mtdRange) (total, right int) {
	truth := func(s mtdSegment) string {
		mid := (s.start + s.end) / 2
		for _, r := range ranges {
			if mid >= r.start && mid < r.end {
				return r.voice
			}
		}
		return ""
	}
	votes := map[string]map[string]int{}
	for _, s := range segs {
		if votes[s.label] == nil {
			votes[s.label] = map[string]int{}
		}
		votes[s.label][truth(s)]++
	}
	labels := slices.Sorted(maps.Keys(votes))
	best := 0
	var assign func(i int, used map[string]bool, score int)
	assign = func(i int, used map[string]bool, score int) {
		if i == len(labels) {
			best = max(best, score)
			return
		}
		assign(i+1, used, score) // unassigned: its segments are wrong
		for voice, c := range votes[labels[i]] {
			if voice == "" || used[voice] {
				continue
			}
			used[voice] = true
			assign(i+1, used, score+c)
			delete(used, voice)
		}
	}
	if len(labels) <= 12 {
		assign(0, map[string]bool{}, 0)
	} else { // beyond a plausible speaker count, a greedy pass
		used := map[string]bool{}
		for _, label := range labels {
			bestVoice, n := "", -1
			for voice, c := range votes[label] {
				if voice != "" && !used[voice] && c > n {
					bestVoice, n = voice, c
				}
			}
			if bestVoice != "" {
				used[bestVoice] = true
				best += n
			}
		}
	}
	return len(segs), best
}

// TestEvalMTDLanguages tries MTD on one clip each of Japanese, Korean,
// Cantonese and Spanish, synthesised with macOS say (the HF corpus with
// these languages is not on this Mac; the clips are made at run time and
// never kept). Qwen3-ASR 0.6B, which knows all four, is the reference.

func TestSpeakerAccuracy(t *testing.T) {
	ranges := []mtdRange{{voice: "A", start: 0, end: 10}, {voice: "B", start: 10, end: 20}}
	seg := func(start, end float64, label string) mtdSegment {
		return mtdSegment{start: start, end: end, label: label}
	}
	// A split into S01 and S02, B as S03: 3 of 3 by majority, 2 of 3 one-to-one.
	if total, right := speakerAccuracy([]mtdSegment{seg(0, 5, "S01"), seg(5, 10, "S02"), seg(10, 20, "S03")}, ranges); total != 3 || right != 2 {
		t.Errorf("over-split: %d of %d, want 2 of 3", right, total)
	}
	// A and B both S01, C as S02: one of them wrong either way.
	ranges3 := append(ranges, mtdRange{voice: "C", start: 20, end: 30})
	if total, right := speakerAccuracy([]mtdSegment{seg(0, 5, "S01"), seg(10, 15, "S01"), seg(20, 25, "S02")}, ranges3); total != 3 || right != 2 {
		t.Errorf("under-merged: %d of %d, want 2 of 3", right, total)
	}
	// All right: A as S01, B as S02.
	if total, right := speakerAccuracy([]mtdSegment{seg(0, 5, "S01"), seg(10, 20, "S02")}, ranges); total != 2 || right != 2 {
		t.Errorf("clean: %d of %d, want 2 of 2", right, total)
	}
	// A gap segment (its middle in no voice) counts in the total, never right.
	if total, right := speakerAccuracy([]mtdSegment{seg(0, 5, "S01"), seg(10, 20, "S01"), seg(25, 30, "S02")}, ranges); total != 3 || right != 1 {
		t.Errorf("gap: %d of %d, want 1 of 3", right, total)
	}
}

// mtdEnglishCeiling is the most error the English clips may have as a
// group before the eval fails. The last good runtime measured 14.0% on
// them, the one with #4211's fix 18.6%;
// #4188 measured 69.8% and #4189 123.3% while the eval, which had no
// limit, still passed.
const mtdEnglishCeiling = 0.25

// mtdFloor is why a config fails: a transcript in the wrong language, or
// English clips whose group error rate passes mtdEnglishCeiling (#4188
// returned Vietnamese for an English clip and the eval still passed).
