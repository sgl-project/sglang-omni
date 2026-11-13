//go:build darwin

package main

// The real speech models on a Mac (VOXTNEW_MAC_E2E=1, the runtime in
// VOXTNEW_OMNI_RUNTIME, the models' folders in VOXTNEW_OMNI_MODELS as
// "id=folder;…" with the catalogue's ids, vad.silero among them for
// Silero and a draft's (asr.fast.draft=…) for speculative decoding): for
// each speech model, a Final of the speech sample, cold (the
// server starts and loads) and warm, then a live Transcribe session with
// the real helper playing the sample from its file input (never the
// microphone or the speaker), and Silero's speech in the sample. It logs
// the times (MEASURE lines):
//
//	MAC_DIR=code/VoxtNew-omni scripts/mac.sh 'VOXTNEW_MAC_E2E=1 VOXTNEW_HELPER=… \
//	  VOXTNEW_OMNI_RUNTIME=… VOXTNEW_OMNI_MODELS="asr.fast=…;vad.silero=…" \
//	  go test -count=1 -run TestMacE2EOmni -v .'

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// macOmniLocal is one speech model as the session's local models.
type macOmniLocal struct {
	p *omni.ASR
	m local.Model
}

func (l macOmniLocal) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		return []local.Model{l.m}
	}
	return nil
}

func (l macOmniLocal) ASR(context.Context, string) (asr.Provider, local.Model, error) {
	return l.p, l.m, nil
}

func (l macOmniLocal) LLM(_ context.Context, id string) (local.Chat, local.Model, error) {
	return nil, local.Model{}, local.NotInstalledError(local.LLM, id)
}

func TestMacE2EOmni(t *testing.T) {
	bin := macE2E(t)
	folders := map[string]string{}
	var ids []string
	for _, item := range strings.Split(os.Getenv("VOXTNEW_OMNI_MODELS"), ";") {
		if id, dir, ok := strings.Cut(strings.TrimSpace(item), "="); ok {
			folders[id] = dir
			if m, _ := models.Lookup(id); m.Kind == models.ASR {
				ids = append(ids, id)
			}
		}
	}
	if len(ids) == 0 {
		t.Skip("set VOXTNEW_OMNI_MODELS (and VOXTNEW_OMNI_RUNTIME) to the models to try")
	}
	exe, _ := os.Executable()
	server, err := omni.Locate(exe)
	if err != nil {
		t.Fatal(err)
	}
	var logMu sync.Mutex
	var logged []string // the runtime's log: a draft's fallback would show here
	rt := omni.NewRuntime(omni.Config{Binary: server, Log: func(level, msg string) {
		t.Logf("%s %s", level, msg)
		logMu.Lock()
		logged = append(logged, msg)
		logMu.Unlock()
	}})
	defer rt.Close()
	var v *omni.VAD
	if dir := folders[models.VADSilero]; dir != "" {
		if v, err = omni.NewVAD(rt, dir); err != nil {
			t.Fatal(err)
		}
	}
	sample := speechSample()
	raw, err := os.ReadFile(sample)
	if err != nil {
		t.Fatal(err)
	}
	pcm, _, _, err := asr.ReadWAV(raw)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	hints := []string{"VoxtNew"}

	if v != nil {
		t0 := time.Now()
		segs, err := v.Speech(ctx, pcm, 16000)
		if err != nil || len(segs) == 0 {
			t.Fatalf("Silero: %v %v", segs, err)
		}
		t.Logf("MEASURE silero speech: %d segments, %v–%v, in %v (cold)", len(segs),
			time.Duration(segs[0].Start)*time.Second/16000, time.Duration(segs[len(segs)-1].End)*time.Second/16000, time.Since(t0).Round(time.Millisecond))
	}

	h := startMacHelper(t, bin, envCaptureFile+"="+sample)
	front := macCall[proto.App](t, h, proto.MethodAppFrontmost, nil)
	for _, id := range ids {
		base, err := omni.NewASR(rt, id, folders[id])
		if err != nil {
			t.Fatal(err)
		}
		m, _ := models.Lookup(id)
		if folders[m.Draft] != "" {
			t.Logf("%s with its draft %s", id, folders[m.Draft])
		}
		a := base.WithDraft(folders[m.Draft]).WithVAD(v)
		for _, round := range []string{"cold", "warm"} {
			t0 := time.Now()
			res, err := a.Transcribe(ctx, asr.FileRequest{Audio: pcm, Format: "pcm", Lang: "zh", Hints: hints})
			if err != nil {
				t.Fatalf("%s Final: %v", id, err)
			}
			t.Logf("MEASURE %s Final %s: %v for %v of audio: %q", id, round, time.Since(t0).Round(time.Millisecond), res.Duration, res.Text)
			if strings.TrimSpace(res.Text) == "" {
				t.Errorf("%s: an empty Final", id)
			}
		}
		macCheckDraft(t, rt, id, folders[m.Draft], &logMu, &logged)
		macOmniSession(t, h, front, macOmniLocal{p: a, m: local.Model{Kind: local.ASR, ID: id, Name: id}}, v != nil)
	}
}

// macCheckDraft: model id's server runs with draft on its command line
// (none when draft is ""), and the runtime never started it again without
// the draft: a server that refused the draft would decode plainly and
// still pass the Finals above.
func macCheckDraft(t *testing.T, rt *omni.Runtime, id, draft string, mu *sync.Mutex, logged *[]string) {
	t.Helper()
	pid := 0
	for _, p := range rt.Processes() {
		if p.ID == id {
			pid = p.PID
		}
	}
	if pid == 0 {
		t.Fatalf("%s: no server process", id)
	}
	out, err := exec.Command("ps", "-o", "command=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		t.Fatal(err)
	}
	cmd := string(out)
	if draft != "" && !strings.Contains(cmd, "--draft-model-path "+draft) {
		t.Errorf("%s runs without its draft: %s", id, cmd)
	}
	if draft == "" && strings.Contains(cmd, "--draft-model-path") {
		t.Errorf("%s runs with a draft: %s", id, cmd)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, m := range *logged {
		if strings.Contains(m, "starting without it") {
			t.Errorf("%s fell back to plain decoding: %s", id, m)
		}
	}
}

// macOmniSession runs a Transcribe session on model id, the sample played
// by the helper in real time, and logs its first partial and its Final.
func macOmniSession(t *testing.T, h *macHelper, front proto.App, svc macOmniLocal, silero bool) {
	id := svc.m.ID
	ms := newMacSession(t, h, front, svc, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		s.Features.Transcribe.ASR = id
	})
	t0 := time.Now()
	sid, err := ms.Start(session.StartRequest{Kind: store.KindTranscribe, Target: front})
	if err != nil {
		t.Fatal(err)
	}
	var heard, firstPartial time.Time
	partials, lastPartial := 0, ""
	stop := time.After(10*time.Second + 800*time.Millisecond) // the sample is 10.5 s
	var end session.Snapshot
	for end.ID == "" {
		select {
		case sn := <-ms.seen:
			if heard.IsZero() && !sn.HeardAt.IsZero() {
				heard = time.Now()
				stop = time.After(10*time.Second + 800*time.Millisecond)
			}
			if sn.State == session.Recording && sn.Partial != "" && sn.Partial != lastPartial {
				if firstPartial.IsZero() {
					firstPartial = time.Now()
				}
				lastPartial = sn.Partial
				partials++
			}
		case <-stop:
			stop = nil
			ms.Stop(sid)
		case end = <-ms.ended:
		case <-time.After(60 * time.Second):
			t.Fatal("no end")
		}
	}
	if end.State != session.Done || strings.TrimSpace(end.Raw) == "" {
		t.Fatalf("%s: end %s %+v %q", id, end.State, end.Problem, end.Raw)
	}
	first := "none"
	if !firstPartial.IsZero() {
		first = firstPartial.Sub(heard).Round(time.Millisecond).String()
	}
	tm := end.Timings
	t.Logf("MEASURE %s live: heard after %v, first partial %s after the first audio (%d partials), Final %v after Stop, audio %v, sent %v, total %v: %q",
		id, heard.Sub(t0).Round(time.Millisecond), first, partials, tm.ASR.Round(time.Millisecond), tm.Audio.Round(time.Millisecond),
		tm.Sent.Round(time.Millisecond), tm.Total.Round(time.Millisecond), end.Raw)
	if firstPartial.IsZero() {
		t.Errorf("%s: no partial while recording", id)
	}
	if silero && (ms.logged("voice activity model") || !ms.logged("held back")) {
		t.Errorf("%s: Silero did not gate the stream", id)
	}
}
