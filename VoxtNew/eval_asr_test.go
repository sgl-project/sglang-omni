package main

// TestEvalASR: the speech models on the eval set (eval_test.go). For each
// model, every clip's Final under three language hints (the main language
// zh, as Voxt; none, the model detects; en, a user whose main language is
// English), the dictionary clips without the context bias too, and live
// sessions (the clip sent in real time) for the first partial and the
// Final after Finish. Reports error rates by kind, the dictionary's terms
// found, and the times (asr.md).

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

// evalHints are the language hints compared: the policy names and the
// hint each sends.
var evalHints = []struct{ name, lang string }{{"main zh", "zh"}, {"none", ""}, {"main en", "en"}}

// asrFloors are quick mode's ceilings of the error rate (the main-language
// hint, every clip run), per model: a regression, not a target.
var asrFloors = map[string]float64{models.ASRAccurate: 0.12, models.ASRFast: 0.2}

type asrResult struct {
	clip    eval.Clip
	text    map[string]string // by hint
	score   map[string]eval.Score
	latency map[string]time.Duration
	noBias  string
}

type liveResult struct {
	clip         eval.Clip
	first, final time.Duration
	partials     int
	text         string
}

func TestEvalASR(t *testing.T) {
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
	clips := e.clips()
	var md strings.Builder
	fmt.Fprintf(&md, "# Speech recognition (%d clips%s)\n\n", len(clips), map[bool]string{true: ", quick", false: ""}[e.quick])
	for _, id := range []string{models.ASRAccurate, models.ASRFast} {
		a, err := omni.NewASR(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		if _, err := a.Transcribe(ctx, asr.FileRequest{Audio: clips[0].PCM, Format: "pcm", Lang: "zh"}); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		cold := time.Since(t0)
		var results []asrResult
		for _, c := range clips {
			r := asrResult{clip: c, text: map[string]string{}, score: map[string]eval.Score{}, latency: map[string]time.Duration{}}
			for _, h := range evalHints {
				res, err := a.Transcribe(ctx, asr.FileRequest{Audio: c.PCM, Format: "pcm", Lang: h.lang, Hints: e.set.Dictionary})
				if err != nil {
					t.Fatalf("%s %s: %v", id, c.ID, err)
				}
				r.text[h.name], r.score[h.name], r.latency[h.name] = res.Text, eval.Compare(c.Text, res.Text), res.Latency
			}
			if c.Set == "dict" {
				res, err := a.Transcribe(ctx, asr.FileRequest{Audio: c.PCM, Format: "pcm", Lang: "zh"})
				if err != nil {
					t.Fatal(err)
				}
				r.noBias = res.Text
			}
			results = append(results, r)
		}
		var live []liveResult
		for i, c := range clips {
			if c.Set == "long" || e.quick && i > 0 {
				continue
			}
			l, err := streamClip(ctx, a, c, "zh", e.set.Dictionary)
			if err != nil {
				t.Fatalf("%s live %s: %v", id, c.ID, err)
			}
			live = append(live, l)
		}
		md.WriteString(asrReport(id, cold, results, live))
		if e.quick {
			var sum eval.Score
			for _, r := range results {
				sum = sum.Add(r.score["main zh"])
			}
			if sum.Rate() > asrFloors[id] {
				t.Errorf("%s: error rate %s over the quick floor %s", id, pct(sum.Rate()), pct(asrFloors[id]))
			}
		}
		rt.Unload(id)
	}
	e.report(t, "asr.md", md.String())
}

// streamClip sends c in real time (100 ms chunks) to a live session and
// times its first partial (after the speech starts: the clips have 0.3 s
// of silence first) and its Final (after Finish).
func streamClip(ctx context.Context, a *omni.ASR, c eval.Clip, lang string, hints []string) (liveResult, error) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	st, err := a.Open(ctx, asr.StreamRequest{SampleRate: asr.SampleRate, Lang: lang, Hints: hints})
	if err != nil {
		return liveResult{}, err
	}
	defer st.Close()
	out := liveResult{clip: c}
	var firstAt, finalAt time.Time
	done := make(chan error, 1)
	go func() {
		for ev := range st.Events() {
			switch ev.Kind {
			case asr.EventPartial:
				if out.partials == 0 {
					firstAt = time.Now()
				}
				out.partials++
			case asr.EventFinal:
				finalAt, out.text = time.Now(), ev.Text
				done <- nil
				return
			case asr.EventError:
				done <- ev.Err
				return
			}
		}
		done <- fmt.Errorf("the stream ended without a final")
	}()
	const chunk = asr.SampleRate / 10 * 2
	start := time.Now()
	for off := 0; off < len(c.PCM); off += chunk {
		if err := st.Send(c.PCM[off:min(off+chunk, len(c.PCM))]); err != nil {
			return out, err
		}
		time.Sleep(time.Until(start.Add(asr.Duration(c.PCM[:min(off+chunk, len(c.PCM))], asr.SampleRate))))
	}
	finished := time.Now()
	if err := st.Finish(); err != nil {
		return out, err
	}
	if err := <-done; err != nil {
		return out, err
	}
	if !firstAt.IsZero() {
		out.first = firstAt.Sub(start) - 300*time.Millisecond
	}
	out.final = finalAt.Sub(finished)
	return out, nil
}

func asrReport(id string, cold time.Duration, results []asrResult, live []liveResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "## %s\n\nError rate (characters of Chinese, words of English; fillers and punctuation aside) by the language hint, and the oracle (the clip's own language):\n\n", id)
	sets := []string{}
	by := map[string][]asrResult{}
	for _, r := range results {
		if len(by[r.clip.Set]) == 0 {
			sets = append(sets, r.clip.Set)
		}
		by[r.clip.Set] = append(by[r.clip.Set], r)
	}
	tb := table{head: []string{"clips", "n", "tokens", "main zh", "none", "main en", "oracle"}}
	for _, s := range append(sets, "all") {
		rs := by[s]
		if s == "all" {
			rs = results
		}
		sum := map[string]eval.Score{}
		for _, r := range rs {
			for _, h := range evalHints {
				sum[h.name] = sum[h.name].Add(r.score[h.name])
			}
			oracle := "main zh"
			if r.clip.Lang == "en" {
				oracle = "main en"
			}
			sum["oracle"] = sum["oracle"].Add(r.score[oracle])
		}
		tb.add(s, len(rs), sum["main zh"].N, pct(sum["main zh"].Rate()), pct(sum["none"].Rate()), pct(sum["main en"].Rate()), pct(sum["oracle"].Rate()))
	}
	b.WriteString(tb.String())

	terms, biased, plain := 0, 0, 0
	dt := table{head: []string{"clip", "terms", "with the dictionary", "without"}}
	for _, r := range results {
		if r.clip.Set != "dict" { // dict2's terms are not in the set's dictionary (TestEvalDictASR)
			continue
		}
		with, without := eval.TermsFound(r.clip.Terms, r.text["main zh"]), eval.TermsFound(r.clip.Terms, r.noBias)
		terms, biased, plain = terms+len(r.clip.Terms), biased+len(with), plain+len(without)
		dt.add(r.clip.ID, strings.Join(r.clip.Terms, ", "), mdText(r.text["main zh"]), mdText(r.noBias))
	}
	fmt.Fprintf(&b, "\nDictionary terms written exactly: %d of %d with the context bias, %d without.\n\n%s", biased, terms, plain, dt.String())

	var finals, long []time.Duration
	var audio time.Duration
	for _, r := range results {
		if r.clip.Set == "long" {
			long = append(long, r.latency["main zh"])
			continue
		}
		finals = append(finals, r.latency["main zh"])
		audio += r.clip.Duration()
	}
	var firsts, afters []time.Duration
	for _, l := range live {
		firsts, afters = append(firsts, l.first), append(afters, l.final)
	}
	fmt.Fprintf(&b, "\nTimes: the first Final (cold: the server starts and loads) %s; warm Finals of %d clips (%.1f s of audio on average) median %s, max %s",
		ms(cold), len(finals), audio.Seconds()/float64(max(len(finals), 1)), ms(median(finals)), ms(maxOf(finals)))
	if len(long) > 0 {
		fmt.Fprintf(&b, "; long clips (60 s+) %s", ms(maxOf(long)))
	}
	if len(live) > 0 {
		fmt.Fprintf(&b, ". Live (%d clips in real time): first partial after the speech starts median %s, max %s; the Final after Finish median %s, max %s",
			len(live), ms(median(firsts)), ms(maxOf(firsts)), ms(median(afters)), ms(maxOf(afters)))
	}
	b.WriteString(".\n\nBy clip, the error rate and the transcript (long ones: their first 60 characters) under each hint:\n\n")
	ct := table{head: []string{"clip", "main zh", "none", "main en"}}
	for _, r := range results {
		cell := func(h string) string {
			text := []rune(r.text[h])
			if r.clip.Set == "long" && len(text) > 60 {
				text = append(text[:60], '…')
			}
			return pct(r.score[h].Rate()) + " " + mdText(string(text))
		}
		ct.add(r.clip.ID, cell("main zh"), cell("none"), cell("main en"))
	}
	b.WriteString(ct.String() + "\n")
	return b.String()
}
