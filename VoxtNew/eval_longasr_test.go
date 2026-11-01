package main

// TestEvalLongASR: long recordings on the speech model (behavior.md
// 3.7a): the long clips (95 s, 104 s) and a synthetic one of about
// 4.8 minutes joined from the set's clips, each recognised three ways:
// one Final over the whole audio (Transcribe, as after Stop before the
// pieces), a live session sent in real time with that one Final after
// Finish, and a live session recognised in pieces while it records.
// Reports the error rate, the wait from Finish to the text, the live
// preview's cadence (the gaps between partials: the pieces share the
// server's one worker with it), the server's footprint and the Mac's
// swap. The full eval only (eval_test.go); VOXTNEW_EVAL_ASR=asr.fast,…
// picks the models (asr.accurate by default), VOXTNEW_EVAL_PIECES the
// piece sizes compared ("20/30/45,10/15/25": the shortest, the target and
// the longest piece in seconds; default the model's own),
// VOXTNEW_EVAL_WAYS=whole,pieces the ways run (all by default); the report is
// long-asr.md, the texts long-asr-texts.md.

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/eval"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni"
)

// longASRRun is one way of recognising a long clip.
type longASRRun struct {
	clip, way string
	audio     time.Duration
	score     eval.Score
	wait      time.Duration // the Final (whole), or Finish to the text (live)
	pieces    int
	gapMax    time.Duration // the longest wait between two partials
	gapP90    time.Duration
	partials  int
	footprint int64  // the speech server's largest footprint, bytes
	swap      string // the swap after the run
	text      string
}

func TestEvalLongASR(t *testing.T) {
	e := evalSetup(t)
	if e.quick {
		t.Skip("long recordings: the full eval only")
	}
	var clips []eval.Clip
	for _, c := range e.set.All() {
		if c.Set == "long" {
			clips = append(clips, c)
		}
	}
	clips = append(clips, synthLong(e.set, 285*time.Second))
	if only := os.Getenv("VOXTNEW_EVAL_ONLY"); only != "" {
		clips = slices.DeleteFunc(clips, func(c eval.Clip) bool { return !slices.Contains(strings.Split(only, ","), c.ID) })
	}
	ids := []string{models.ASRAccurate}
	if env := strings.TrimSpace(os.Getenv("VOXTNEW_EVAL_ASR")); env != "" {
		ids = strings.Split(env, ",")
	}
	exe, _ := os.Executable()
	server, err := omni.Locate(exe)
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	pieces := 0
	rt := omni.NewRuntime(omni.Config{Binary: server, Log: func(level, msg string) {
		if strings.Contains(msg, "piece ") && strings.Contains(msg, " of the recording,") {
			mu.Lock()
			pieces++
			mu.Unlock()
		}
		if level != "debug" {
			t.Logf("omni %s: %s", level, msg)
		}
	}})
	defer rt.Close()
	ctx := context.Background()
	silero, err := omni.NewVAD(rt, e.dir(t, models.VADSilero))
	if err != nil {
		t.Fatal(err)
	}
	var md, texts strings.Builder
	md.WriteString("# Long recordings on the speech model\n\n")
	texts.WriteString("# Long recordings: the texts\n\n")
	for _, id := range ids {
		a, err := omni.NewASR(rt, id, e.dir(t, id))
		if err != nil {
			t.Fatal(err)
		}
		a = a.WithVAD(silero)
		if _, err := a.Transcribe(ctx, asr.FileRequest{Audio: clips[0].PCM[:asr.SampleRate*2*3], Format: "pcm", Lang: "zh"}); err != nil {
			t.Fatalf("%s: %v", id, err) // warm: the server loads
		}
		var runs []longASRRun
		ways := []string{"whole", "live, one Final"}
		sizes := map[string]omni.Pieces{}
		for _, spec := range strings.Split(cmpOr(os.Getenv("VOXTNEW_EVAL_PIECES"), "model"), ",") {
			way := "live, pieces"
			if spec != "model" {
				var p omni.Pieces
				for i, f := range strings.Split(spec, "/") {
					v, _ := strconv.ParseFloat(f, 64)
					*[]*time.Duration{&p.Min, &p.Target, &p.Max}[min(i, 2)] = time.Duration(v * float64(time.Second))
				}
				way, sizes[way+" "+spec] = way+" "+spec, p
			}
			ways = append(ways, way)
		}
		if only := os.Getenv("VOXTNEW_EVAL_WAYS"); only != "" { // "whole,pieces": the ways whose names start so
			ways = slices.DeleteFunc(ways, func(w string) bool {
				return !slices.ContainsFunc(strings.Split(only, ","), func(p string) bool { return strings.HasPrefix(w, p) || strings.HasPrefix(w, "live, "+p) })
			})
		}
		for _, c := range clips {
			for _, way := range ways {
				mu.Lock()
				pieces = 0
				mu.Unlock()
				r := longASRRun{clip: c.ID, way: way, audio: c.Duration()}
				stopWatch := watchFootprint(rt, id, &r.footprint)
				switch way {
				case "whole":
					t0 := time.Now()
					res, err := a.Transcribe(ctx, asr.FileRequest{Audio: c.PCM, Format: "pcm", Lang: "zh", Hints: e.set.Dictionary})
					if err != nil {
						t.Fatalf("%s %s: %v", id, c.ID, err)
					}
					r.wait, r.text = time.Since(t0), res.Text
				case "live, one Final":
					err = streamLong(ctx, a.WithPieces(omni.Pieces{After: -1}), c, e.set.Dictionary, &r)
				default:
					err = streamLong(ctx, a.WithPieces(sizes[way]), c, e.set.Dictionary, &r)
				}
				stopWatch()
				if err != nil {
					t.Fatalf("%s %s %s: %v", id, c.ID, way, err)
				}
				mu.Lock()
				r.pieces = pieces
				mu.Unlock()
				r.score = eval.Compare(c.Text, r.text)
				if m, err := macx.ReadSystemMemory(); err == nil {
					r.swap = fmt.Sprintf("%.1f / %.1f GB", float64(m.SwapUsedBytes)/(1<<30), float64(m.SwapTotalBytes)/(1<<30))
				}
				t.Logf("%s %s %s: %s, wait %v, %d pieces, gaps max %v", id, c.ID, way, pct(r.score.Rate()), r.wait, r.pieces, r.gapMax)
				runs = append(runs, r)
			}
		}
		md.WriteString(longASRReport(id, runs))
		for _, r := range runs {
			fmt.Fprintf(&texts, "## %s %s, %s\n\n%s\n\n", id, r.clip, r.way, r.text)
		}
		rt.Unload(id)
	}
	e.report(t, "long-asr.md", md.String())
	e.report(t, "long-asr-texts.md", texts.String())
}

// synthLong joins the set's clips, then the long ones' clips again,
// until the audio is at least d (0.8 s between clips, as the long ones).
func synthLong(set *eval.Set, d time.Duration) eval.Clip {
	l := eval.Long{ID: "long-synth", Set: "long", Lang: "zh", Gap: 0.8}
	var order []string
	for _, c := range set.Clips {
		order = append(order, c.ID)
	}
	for _, lg := range set.Long {
		order = append(order, lg.Of...)
	}
	var total time.Duration
	for _, id := range order {
		c, _ := set.Clip(id)
		l.Of = append(l.Of, id)
		total += c.Duration() + 800*time.Millisecond
		if total >= d {
			break
		}
	}
	return set.Join(l)
}

// watchFootprint samples model id's server's footprint every half
// second into peak, until the returned stop.
func watchFootprint(rt *omni.Runtime, id string, peak *int64) func() {
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		tick := time.NewTicker(500 * time.Millisecond)
		defer tick.Stop()
		for {
			for _, p := range rt.Processes() {
				if p.ID == id {
					if m, err := macx.Memory(p.PID); err == nil && m.Footprint > *peak {
						*peak = m.Footprint
					}
				}
			}
			select {
			case <-done:
				return
			case <-tick.C:
			}
		}
	})
	return func() { close(done); wg.Wait() }
}

// streamLong sends c in real time (100 ms chunks) to a live session and
// records the partials' cadence and the wait from Finish to the text.
func streamLong(ctx context.Context, a *omni.ASR, c eval.Clip, hints []string, r *longASRRun) error {
	return streamLongReq(ctx, a, c, asr.StreamRequest{SampleRate: asr.SampleRate, Lang: "zh", Hints: hints}, r)
}

// streamLongReq is streamLong with the session's request.
func streamLongReq(ctx context.Context, a *omni.ASR, c eval.Clip, req asr.StreamRequest, r *longASRRun) error {
	ctx, cancel := context.WithTimeout(ctx, c.Duration()+5*time.Minute)
	defer cancel()
	st, err := a.Open(ctx, req)
	if err != nil {
		return err
	}
	defer st.Close()
	var partials []time.Time
	var finalAt time.Time
	done := make(chan error, 1)
	go func() {
		for ev := range st.Events() {
			switch ev.Kind {
			case asr.EventPartial:
				partials = append(partials, time.Now())
			case asr.EventFinal:
				finalAt, r.text = time.Now(), ev.Text
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
			return err
		}
		time.Sleep(time.Until(start.Add(asr.Duration(c.PCM[:min(off+chunk, len(c.PCM))], asr.SampleRate))))
	}
	finished := time.Now()
	if err := st.Finish(); err != nil {
		return err
	}
	if err := <-done; err != nil {
		return err
	}
	r.wait = finalAt.Sub(finished)
	var gaps []time.Duration
	for i := 1; i < len(partials); i++ {
		if partials[i].Before(finished) {
			gaps = append(gaps, partials[i].Sub(partials[i-1]))
		}
	}
	r.partials = len(partials)
	if len(gaps) > 0 {
		slices.Sort(gaps)
		r.gapMax, r.gapP90 = gaps[len(gaps)-1], gaps[len(gaps)*9/10]
	}
	return nil
}

func longASRReport(id string, runs []longASRRun) string {
	var b strings.Builder
	fmt.Fprintf(&b, "## %s\n\n", id)
	tb := table{head: []string{"clip", "audio", "way", "error", "edits (sub/del/ins)", "wait for the text", "real-time factor", "pieces", "partials", "gap p90 / max", "server footprint", "swap after"}}
	for _, r := range runs {
		gaps := "–"
		if r.partials > 0 {
			gaps = fmt.Sprintf("%s / %s", ms(r.gapP90), ms(r.gapMax))
		}
		rtf := "–"
		if r.way == "whole" {
			rtf = fmt.Sprintf("%.3f", r.wait.Seconds()/r.audio.Seconds())
		}
		tb.add(r.clip, fmt.Sprintf("%.0f s", r.audio.Seconds()), r.way, pct(r.score.Rate()), fmt.Sprintf("%d/%d/%d", r.score.Sub, r.score.Del, r.score.Ins), ms(r.wait), rtf, r.pieces, r.partials, gaps,
			fmt.Sprintf("%.2f GB", float64(r.footprint)/(1<<30)), r.swap)
	}
	b.WriteString(tb.String())
	return b.String()
}
