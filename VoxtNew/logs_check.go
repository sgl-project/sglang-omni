//go:build logscheck

package main

// The logs' screenshots on a Mac (scripts/mac-logs-shots.sh): with
// -tags logscheck, `VoxtNew --logs-check <dir> [--lang zh-Hans]` runs
// instead of the app: its window on Logs, over a library in a temporary
// directory (no lock, no helper, no family, nothing of the installed
// VoxtNew), with 40 sample lines. Once the page is laid out it writes
// <dir>/geometry.json (the window's number, its content on screen, each
// line's middle in it), for the script's driver, which works the real
// pointer and keys and captures the window. When <dir>/append appears it
// appends 5 lines (as a log does while the user reads: the order's
// shots) and removes the file; it quits when <dir>/finished appears, or
// after two minutes. Release builds do not have it.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

func init() { logsCheck = runLogsCheck }

// logsGeometry is what the driver needs: where the lines are on screen.
type logsGeometry struct {
	Window  int          `json:"window"`  // the window number, for screencapture -l
	Content [4]float64   `json:"content"` // x, y, width, height on screen (points, top-left)
	Lines   [][2]float64 `json:"lines"`   // each line's middle, in the content
	List    [4]float64   `json:"list"`    // the lines' list, in the content
}

func runLogsCheck(dir string) {
	tmp, err := os.MkdirTemp("", "voxtnew-logs-check")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer os.RemoveAll(tmp)
	mygo.App.SetName("VoxtNew Logs Check")
	lib, err := openLibrary(tmp)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	a := newApp(lib, newHelperHost("", tmp))
	a.updateSettings(func(s *settings.Settings) {
		s.General.Language = argAfter("--lang")
		s.Setup.Done = true // no setup guide over the page
	})
	samples := []struct {
		level core.Severity
		src   string
		msg   string
	}{
		{core.SeverityInfo, "app", "VoxtNew 0.1.0 started"},
		{core.SeverityInfo, "helper", "helper ready (protocol 1)"},
		{core.SeverityInfo, "app", "models: Qwen3-ASR 0.6B installed (claimed by VoxtNew, AudioSmith)"},
		{core.SeverityDebug, "helper:capture", "input: MacBook Air Microphone, 48 kHz"},
		{core.SeverityInfo, "app", "session t-%d: transcribe, 2.4 s of audio"},
		{core.SeverityWarning, "helper:capture", "AirPods Pro gave no audio within 1.2 s: recording from the default input"},
		{core.SeverityInfo, "app", "session t-%d: inserted 38 characters into Notes"},
		{core.SeverityError, "app", "llm: request timed out after 30 s (Gemma 3 4B), retrying once"},
	}
	at := time.Now().Add(-10 * time.Minute)
	for i := range 40 {
		s := samples[i%len(samples)]
		msg := s.msg
		if i%len(samples) == 4 || i%len(samples) == 6 {
			msg = fmt.Sprintf(msg, 100+i/len(samples))
		}
		a.logs.Append(core.LogEntry{Time: at.Add(time.Duration(i) * 7 * time.Second), Level: s.level, Source: s.src, Message: msg})
	}
	mygo.App.WhenReady(func() {
		a.openWindow()
		cb := a.win.ContentBounds()
		a.win.SetContentBounds(mygo.Rectangle{X: cb.X, Y: cb.Y, Width: 960, Height: 640})
		a.clearNotices() // the services' card: no models here
		a.navigate("/logs")
		go waitLogsGeometry(a, dir)
	})
	time.AfterFunc(2*time.Minute, func() { mygo.RunOnMain(mygo.App.Quit) })
	go func() {
		n := 200
		for {
			time.Sleep(200 * time.Millisecond)
			if _, err := os.Stat(filepath.Join(dir, "finished")); err == nil {
				mygo.RunOnMain(mygo.App.Quit)
				return
			}
			if os.Remove(filepath.Join(dir, "append")) == nil {
				for range 5 {
					n++
					a.logs.Append(core.LogEntry{Level: core.SeverityInfo, Source: "app", Message: fmt.Sprintf("session t-%d: inserted %d characters into Notes", n, 20+n)})
				}
			}
		}
	}()
	if err := mygo.App.Run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// waitLogsGeometry writes geometry.json once the lines are laid out; on a
// goroutine.
func waitLogsGeometry(a *app, dir string) {
	time.Sleep(time.Second) // the new size laid out
	for range 100 {
		time.Sleep(100 * time.Millisecond)
		var g logsGeometry
		mygo.RunOnMain(func() {
			s := &a.logSel
			if a.win == nil || len(s.spans) < 20 {
				return
			}
			cb := a.win.ContentBounds()
			g.Window = macx.WindowNumber(a.win.NativeHandle())
			g.Content = [4]float64{float64(cb.X), float64(cb.Y), float64(cb.Width), float64(cb.Height)}
			g.List = [4]float64{float64(s.origin.X), float64(s.origin.Y), float64(s.origin.W), float64(s.origin.H)}
			x := float64(s.origin.X + s.origin.W/2)
			for _, sp := range s.spans {
				g.Lines = append(g.Lines, [2]float64{x, float64(s.origin.Y + (sp.top+sp.bottom)/2)})
			}
		})
		if len(g.Lines) == 0 {
			continue
		}
		b, _ := json.Marshal(g)
		_ = os.WriteFile(filepath.Join(dir, "geometry.json"), b, 0o644)
		return
	}
}
