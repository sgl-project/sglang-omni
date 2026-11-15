package main

// The packaged app's smoke run (scripts/mac-regress.sh): started as
// Voxt.app/Contents/MacOS/Voxt --smoke <dir>, it runs instead of the
// app, with no window, no menu bar icon and no single-instance lock, on a
// data folder of its own (<dir>/data; the models are the model library's,
// $HUB): it starts the bundled helper, opens the first installed speech
// model (or the one VOXTNEW_SMOKE_MODELS=<asr id> names) through the app's
// own Library and runtime (the bundled Omni server), asks it for an answer
// (a second of silence), then quits as the app does. The script then
// checks that nothing it started is left running. Lines starting
// "smoke:" say what happened; the exit status is 0 when all of it worked.
//
// With VOXTNEW_SMOKE_LEGACY=1 it only imports the previous Voxt's data
// into that folder, as the import card does, and prints the terms and
// history entries there (.github/workflows/voxt-upgrade-test.yaml).

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func runSmoke(dir string) int {
	say := func(format string, args ...any) { fmt.Printf("smoke: "+format+"\n", args...) }
	resources, _ := mygo.App.Path(mygo.PathResources)
	lib, err := openLibrary(filepath.Join(dir, "data"))
	if err != nil {
		say("FAIL the library: %v", err)
		return 1
	}
	defer lib.close()
	// No shortcuts: the helper installs no event tap (the Mac may be in use).
	err = lib.settings.Update(context.Background(), func(s *settings.Settings) {
		s.Features.Transcribe.Hotkeys, s.Features.Translate.Hotkeys, s.Features.Rewrite.Hotkeys = nil, nil, nil
		s.Features.PasteLast, s.Features.Cancel = settings.Hotkey{}, settings.Hotkey{}
	})
	if err != nil {
		say("FAIL the settings: %v", err)
		return 1
	}
	a := newApp(lib, newHelperHost(resources, dir))
	queue := make(chan func(), 4096)
	a.postQueue = func(fn func()) { queue <- fn }
	pump := func(d time.Duration, cond func() bool) bool {
		deadline := time.After(d)
		for !cond() {
			select {
			case fn := <-queue:
				fn()
			case <-deadline:
				return false
			}
		}
		return true
	}
	quit := func(code int) int {
		a.quit()
		pump(300*time.Millisecond, func() bool { return false })
		say("quit")
		return code
	}
	if os.Getenv("VOXTNEW_SMOKE_LEGACY") == "1" {
		legacy := legacyLibrary()
		switch {
		case legacy == "":
			say("FAIL no previous Voxt on this Mac (no %s container)", legacyBundleID)
			return quit(1)
		case !legacyReadable(legacy):
			// The Developer ID build must read it without the open panel.
			say("FAIL the previous Voxt's database in %s is not readable without asking", legacy)
			return quit(1)
		default:
			say("legacy readable: %s", legacyDatabase(legacy))
		}
		a.importLegacyFrom(legacy)
		if !pump(time.Minute, func() bool { return !a.dv.importing }) || !a.ob.legacyDone {
			say("FAIL the import: %s", a.ob.legacy.text())
			return quit(1)
		}
		ctx := context.Background()
		terms, err := a.lib.store.Dictionary().List(ctx, "")
		if err != nil {
			say("FAIL the dictionary: %v", err)
			return quit(1)
		}
		entries, err := a.lib.store.History().List(ctx, store.HistoryQuery{})
		if err != nil {
			say("FAIL the history: %v", err)
			return quit(1)
		}
		say("legacy import: %d terms, %d history entries (%s)", len(terms), len(entries), a.ob.legacy.text())
		return quit(0)
	}
	a.connectHelper()
	a.startHelper()
	if !pump(30*time.Second, func() bool { return a.helperStatus().State == helper.Ready }) {
		say("FAIL the helper is %s: %v", a.helperStatus().State, a.helperStatus().Err)
		return quit(1)
	}
	say("helper ready")
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	type result struct {
		what string
		err  error
	}
	done := make(chan result, 1)
	go func() {
		asrs := a.local.Installed(local.ASR)
		if want := os.Getenv("VOXTNEW_SMOKE_MODELS"); want != "" {
			asrs = []local.Model{{ID: want}}
		}
		if len(asrs) == 0 {
			done <- result{err: fmt.Errorf("no speech model installed in %s", a.models.Hub())}
			return
		}
		t0 := time.Now()
		p, _, err := a.local.ASR(ctx, asrs[0].ID)
		ft, ok := p.(asr.FileTranscriber)
		if err != nil || !ok {
			done <- result{err: fmt.Errorf("speech model %s: %v", asrs[0].ID, err)}
			return
		}
		if _, err := ft.Transcribe(ctx, asr.FileRequest{Audio: make([]byte, 2*asr.SampleRate), Format: "pcm"}); err != nil {
			done <- result{err: fmt.Errorf("speech model %s: %w", asrs[0].ID, err)}
			return
		}
		done <- result{what: fmt.Sprintf("speech model %s answered in %v", asrs[0].ID, time.Since(t0).Round(time.Millisecond))}
	}()
	var r result
	deadline := time.After(4 * time.Minute)
wait:
	for {
		select {
		case fn := <-queue:
			fn()
		case r = <-done:
			break wait
		case <-deadline:
			r.err = fmt.Errorf("no answers in 4 minutes")
			break wait
		}
	}
	if r.err != nil || r.what == "" {
		say("FAIL %v", r.err)
		return quit(1)
	}
	say("%s", r.what)
	return quit(0)
}
