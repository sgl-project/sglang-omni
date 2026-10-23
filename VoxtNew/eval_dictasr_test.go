package main

// TestEvalDictASR: the dictionary as the speech models' context (behavior.md
// 3.6, 3.6a). Every short clip, as recorded (0.3 s of silence after the
// speech) and with 3 s of room noise more (a dictation stopped late), and
// two recordings without speech (1.5 s of silence, of noise), each with no
// dictionary and with four (dictSmall, the eval set's nine, dictMedium,
// dictLarge). Each with a dictionary twice: the context as the model
// writes it ("raw": the terms sent as a free prompt, which nothing checks)
// and as VoxtNew delivers it (the terms as hints: the echo guard).
// Reports the error rate on the speech, the clips' terms written, the
// answers with a term nobody said (a term neither the clip's words nor
// the transcript without a dictionary have), and the time; then a long
// live session (long-zh and 3 s of noise, in pieces) with the user's
// three terms, raw and guarded (eval_test.go; dict-asr.md).

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni"
)

func TestEvalDictASR(t *testing.T) {
	e := evalSetup(t)
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
	silero, err := omni.NewVAD(rt, e.dir(t, models.VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	var clips []eval.Clip
	for _, c := range e.clips() {
		if c.Set != "long" {
			clips = append(clips, c)
		}
	}
	quiet := []eval.Clip{{ID: "silence", Lang: "zh", PCM: padAudio(nil, 1.5, 0, 1)}, {ID: "noise", Lang: "zh", PCM: padAudio(nil, 1.5, -55, 2)}}
	dicts := []struct {
		name  string
		terms []string
	}{{"small (3)", dictSmall}, {"eval (9)", e.set.Dictionary}, {"medium (30)", dictMedium}, {fmt.Sprintf("large (%d)", len(dictLarge())), dictLarge()}}
	pads := []struct {
		name string
		sec  float64
	}{{"as recorded", 0}, {"+3 s noise", 3}}
	ids := []string{models.ASRAccurate, models.ASRFast}
	if env := strings.TrimSpace(os.Getenv("VOXTNEW_EVAL_ASR")); env != "" {
		ids = strings.Split(env, ",")
	}
	var md strings.Builder
	md.WriteString("# The dictionary as the speech models' context\n\n")
	for _, id := range ids {
		a, err := omni.NewASR(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		a = a.WithVAD(silero)
		if _, err := a.Transcribe(ctx, asr.FileRequest{Audio: clips[0].PCM, Format: "pcm", Lang: "zh"}); err != nil {
			t.Fatal(err) // warm: the server loads
		}
		tb := table{head: []string{"dictionary", "audio", "way", "error", "terms written (without it)", "speech: answers with a term put in", "no speech: answers with a term", "median time"}}
		leaks := table{head: []string{"dictionary", "audio", "way", "clip", "transcript", "put in"}}
		for _, p := range pads {
			plain := map[string]string{}
			var sum eval.Score
			var times []time.Duration
			written, want := 0, 0
			for _, c := range clips {
				res, err := a.Transcribe(ctx, asr.FileRequest{Audio: padAudio(c.PCM, p.sec, -55, int64(len(c.PCM))), Format: "pcm", Lang: "zh"})
				if err != nil {
					t.Fatal(err)
				}
				plain[c.ID] = res.Text
				sum = sum.Add(eval.Compare(c.Text, res.Text))
				times = append(times, res.Latency)
				found := eval.TermsFound(c.Terms, res.Text)
				written, want = written+len(found), want+len(c.Terms)
			}
			tb.add("none", p.name, "", pct(sum.Rate()), fmt.Sprintf("%d of %d", written, want), "", "", ms(median(times)))
			for _, d := range dicts {
				for _, way := range []string{"raw", "guarded"} {
					request := func(pcm []byte) asr.FileRequest {
						r := asr.FileRequest{Audio: pcm, Format: "pcm", Lang: "zh", Hints: d.terms}
						if way == "raw" {
							r.Hints, r.Prompt = nil, strings.Join(d.terms, "\n")
						}
						return r
					}
					var sum eval.Score
					var times []time.Duration
					written, without, want, put, echoed := 0, 0, 0, 0, 0
					for _, c := range clips {
						res, err := a.Transcribe(ctx, request(padAudio(c.PCM, p.sec, -55, int64(len(c.PCM)))))
						if err != nil {
							t.Fatal(err)
						}
						sum = sum.Add(eval.Compare(c.Text, res.Text))
						times = append(times, res.Latency)
						var inDict []string
						for _, term := range c.Terms {
							if hasTerm(strings.Join(d.terms, "\n"), term) {
								inDict = append(inDict, term)
							}
						}
						found := eval.TermsFound(inDict, res.Text)
						written, want = written+len(found), want+len(inDict)
						without += len(eval.TermsFound(inDict, plain[c.ID]))
						if extra := termsPutIn(d.terms, res.Text, c.Text, plain[c.ID]); len(extra) > 0 {
							put++
							leaks.add(d.name, p.name, way, c.ID, mdText(res.Text), strings.Join(extra, ", "))
						}
					}
					for _, q := range quiet {
						res, err := a.Transcribe(ctx, request(padAudio(q.PCM, p.sec, -55, 3)))
						if err != nil {
							t.Fatal(err)
						}
						if extra := termsPutIn(d.terms, res.Text); len(extra) > 0 {
							echoed++
							leaks.add(d.name, p.name, way, q.ID, mdText(res.Text), fmt.Sprintf("%d terms", len(extra)))
						}
					}
					tb.add(d.name, p.name, way, pct(sum.Rate()), fmt.Sprintf("%d of %d (%d)", written, want, without),
						fmt.Sprintf("%d of %d", put, len(clips)), fmt.Sprintf("%d of %d", echoed, len(quiet)), ms(median(times)))
				}
			}
		}
		long := e.set.Join(e.set.Long[0])
		long.PCM = padAudio(long.PCM, 3, -55, 9)
		lives := table{head: []string{"way", "error", "pieces' text ends", "terms put in"}}
		for _, way := range []string{"raw", "guarded"} {
			req := asr.StreamRequest{SampleRate: asr.SampleRate, Lang: "zh", Hints: dictSmall}
			if way == "raw" {
				req.Hints, req.Prompt = nil, strings.Join(dictSmall, "\n")
			}
			var r longASRRun
			if err := streamLongReq(ctx, a, long, req, &r); err != nil {
				t.Fatal(err)
			}
			end := []rune(r.text)
			lives.add(way, pct(eval.Compare(long.Text, r.text).Rate()), mdText(string(end[max(0, len(end)-30):])), strings.Join(termsPutIn(dictSmall, r.text, long.Text), ", "))
		}
		fmt.Fprintf(&md, "## %s\n\n%s\nLong live session (long-zh and 3 s of noise, pieces), the small dictionary:\n\n%s\nAnswers with a term nobody said:\n\n%s\n", id, tb.String(), lives.String(), leaks.String())
		rt.Unload(id)
	}
	e.report(t, "dict-asr.md", md.String())
}
