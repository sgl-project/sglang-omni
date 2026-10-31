package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// ovStep is what the overlay is after a step: its mode, the window's
// frame, whether it is on screen, takes the pointer, and the live text
// the face shows.
type ovStep struct {
	mode    overlayMode
	frame   mygo.Rectangle
	visible bool
	pass    bool
	text    string
}

func (a *app) ovNow() ovStep {
	text := ""
	if a.ov.mode == ovRecording || a.ov.mode == ovWorking {
		text = liveText(a.sess)
	}
	return ovStep{a.ov.mode, a.ov.frame, a.ov.visible, a.ov.passThrough, text}
}

// pillFrame is the window's frame for the pill at its anchor.
func pillFrame(o settings.Overlay) mygo.Rectangle {
	return layout.Frame(ovTestArea, o, layout.PillW, layout.PillH)
}

// anchored checks that frame f sits at o's anchor: centred, its anchor
// edge where the pill's is.
func anchored(t *testing.T, what string, o settings.Overlay, f mygo.Rectangle) {
	t.Helper()
	p := pillFrame(o)
	cx, pcx := f.X*2+f.Width, p.X*2+p.Width
	if d := cx - pcx; d < -1 || d > 1 {
		t.Errorf("%s: centre %d, the pill's %d (%+v)", what, cx/2, pcx/2, f)
	}
	if o.Position == "top" {
		if f.Y != p.Y {
			t.Errorf("%s: top %d, the pill's %d", what, f.Y, p.Y)
		}
	} else if f.Y+f.Height != p.Y+p.Height {
		t.Errorf("%s: bottom %d, the pill's %d", what, f.Y+f.Height, p.Y+p.Height)
	}
}

// TestOverlaySessionsStartSmall: the overlay's state machine, step by
// step, the window's frame at each: a session starts as the pill
// (150×48) with no text, grows with live text about its anchor (the
// bottom edge and the centre stay; the top edge at the top), and the next
// session starts as the pill again, whatever the last showed (10-15: the
// second session's pill showed in the first's grown, empty window).
// Escape, a notice, and two sessions in quick succession too.
func TestOverlaySessionsStartSmall(t *testing.T) {
	for _, pos := range []string{"bottom", "top"} {
		t.Run(pos, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			ft := new(fakeTimers).install(a)
			// Saved, so that a later save (Setup.Tried) keeps it.
			a.updateSettings(func(s *settings.Settings) { s.Overlay.Position = pos })
			o := a.prefs.Overlay
			pill := pillFrame(o)
			check := func(what string, want ovStep) {
				t.Helper()
				got := a.ovNow()
				if got != want {
					t.Errorf("%s:\n got %+v\nwant %+v", what, got, want)
				}
				if got.visible {
					anchored(t, what, o, got.frame)
				}
			}
			now := time.Now()
			// A: the shortcut shows the pill.
			snap(a, session.Snapshot{ID: "A", State: session.Recording, StartedAt: now})
			check("A starts", ovStep{ovRecording, pill, true, true, ""})
			if pill.Width != layout.PillW+2*layout.Pad || pill.Height != layout.PillH+2*layout.Pad {
				t.Fatalf("the pill's frame %+v", pill)
			}
			// A streaming service's partial: one line under the row.
			snap(a, session.Snapshot{ID: "A", State: session.Recording, StartedAt: now, HeardAt: now, Partial: "hello"})
			grown := layout.Frame(ovTestArea, o, layout.RowW+2*layout.LivePadX, layout.LiveH) // "hello" is narrower than the row
			check("A hears", ovStep{ovRecording, grown, true, true, "hello"})
			long := "So the plan for the release is to finish the onboarding first, then the history page"
			snap(a, session.Snapshot{ID: "A", State: session.Recording, StartedAt: now, HeardAt: now, Partial: long})
			widest := layout.Frame(ovTestArea, o, layout.LiveMaxW, layout.LiveH)
			check("A hears more", ovStep{ovRecording, widest, true, true, long})
			// Recognising, then cleaning up: the text stays, the face does
			// not shrink though the model's draft is shorter.
			snap(a, session.Snapshot{ID: "A", State: session.Recognizing, Partial: long})
			check("A recognises", ovStep{ovWorking, widest, true, true, long})
			snap(a, session.Snapshot{ID: "A", State: session.Composing, Partial: long, Draft: "So"})
			check("A cleans up", ovStep{ovWorking, widest, true, true, "So"})
			// Inserted: it fades and leaves, keeping its frame meanwhile.
			snap(a, session.Snapshot{ID: "A", State: session.Done, Result: "So.", Inserted: true, Partial: long})
			ft.advance(ovGone)
			if s := a.ovNow(); s.mode != ovHidden || s.visible || a.ov.shown {
				t.Fatalf("A ended: %+v", s)
			}
			// B: the pill again, no text.
			snap(a, session.Snapshot{ID: "B", State: session.Recording, StartedAt: now})
			check("B starts", ovStep{ovRecording, pill, true, true, ""})
			if a.ov.w != layout.PillW || a.ov.h != layout.PillH {
				t.Errorf("B's face %d×%d", a.ov.w, a.ov.h)
			}
			snap(a, session.Snapshot{ID: "B", State: session.Recording, StartedAt: now, HeardAt: now, Partial: "hi"})
			check("B hears", ovStep{ovRecording, grown, true, true, "hi"})
			// Escape: gone, and C starts small.
			snap(a, session.Snapshot{ID: "B", State: session.Cancelled, Partial: "hi"})
			ft.advance(ovGone)
			if a.ov.visible {
				t.Fatal("B cancelled: still on screen")
			}
			// C fails: a notice in the pill's place, which takes a click.
			snap(a, session.Snapshot{ID: "C", State: session.Recording, StartedAt: now})
			check("C starts", ovStep{ovRecording, pill, true, true, ""})
			snap(a, session.Snapshot{ID: "C", State: session.Failed, Problem: &session.Problem{Code: session.ProblemASR, Err: &asr.Error{Kind: asr.KindAuth}}})
			nf := layout.Frame(ovTestArea, o, layout.NoticeWidth(guessWidth(noticeText(a.notice.text)), true), layout.PillH)
			check("C's notice", ovStep{ovNotice, nf, true, false, ""})
			// D starts while the notice shows: the pill, the notice gone.
			snap(a, session.Snapshot{ID: "D", State: session.Recording, StartedAt: now})
			check("D starts over the notice", ovStep{ovRecording, pill, true, true, ""})
			if a.notice != nil {
				t.Error("C's notice stays")
			}
			// Quick succession: D grows, ends, and E starts mid-fade.
			snap(a, session.Snapshot{ID: "D", State: session.Recording, StartedAt: now, HeardAt: now, Partial: long})
			snap(a, session.Snapshot{ID: "D", State: session.Done, Result: "x", Inserted: true, Partial: long})
			ft.advance(20 * time.Millisecond)
			snap(a, session.Snapshot{ID: "E", State: session.Recording, StartedAt: now})
			check("E starts mid-fade", ovStep{ovRecording, pill, true, true, ""})
			ft.advance(time.Second)
			check("E after D's fade", ovStep{ovRecording, pill, true, true, ""})
			// E's first word, at once: grown about the same anchor.
			snap(a, session.Snapshot{ID: "E", State: session.Recording, StartedAt: now, HeardAt: now, Partial: "ok"})
			check("E hears", ovStep{ovRecording, grown, true, true, "ok"})
		})
	}
}

// TestOverlayAnchorAndRestyle: whatever the face (the pill, the time,
// a group, live text, a notice), the window keeps its anchor; a change
// of the settings while it shows moves it to the new anchor.
func TestOverlayAnchorAndRestyle(t *testing.T) {
	a, _, _ := newTestApp(t)
	now := time.Now()
	o := a.prefs.Overlay
	faces := []session.Snapshot{
		{ID: "s1", State: session.Recording, StartedAt: now},
		{ID: "s1", State: session.Recording, StartedAt: now, Scene: sceneMatch("Team chat", appctx.SourceApp, ""), Target: proto.App{Name: "Slack"}},
		{ID: "s1", State: session.Recording, StartedAt: now, Partial: "明天三点开会"},
		{ID: "s1", State: session.Recording, StartedAt: now, Partial: strings.Repeat("长句子", 40)},
		{ID: "s1", State: session.Recognizing},
	}
	for i, s := range faces {
		snap(a, s)
		anchored(t, fmt.Sprint("face ", i), o, a.ov.frame)
	}
	// The setting changes while it shows: the new anchor, the same face.
	a.prefs.Overlay.Position, a.prefs.Overlay.Margin = "top", 40
	a.overlayRestyle()
	anchored(t, "restyled", a.prefs.Overlay, a.ov.frame)
	if a.ov.frame.Y != ovTestArea.Y+40-layout.Pad {
		t.Errorf("top frame %+v", a.ov.frame)
	}
}

// TestOverlayTwoSessionsEndToEnd: the shortcut twice, through the real
// session manager and the fake helper: the first session's text (the
// file service's, while the model cleans it up) grows the overlay; the
// second starts as the pill, with no text, at the same anchor.
func TestOverlayTwoSessionsEndToEnd(t *testing.T) {
	chat := &fakeChat{text: "Hello, VoxtNew."}
	set, keys, _ := fileSetup(t, "hello aura link, the second line of a longer text", chat)
	chat.delay = 300 * time.Millisecond // cleaning up shows a while
	ev := script(t,
		hotkeyLine(0, "transcribe/0", proto.PhaseTrigger),
		hotkeyLine(400, "transcribe/0", proto.PhaseTrigger),
		hotkeyLine(1500, "transcribe/0", proto.PhaseTrigger),
		hotkeyLine(400, "transcribe/0", proto.PhaseTrigger),
	)
	a := newSessionApp(t, ev, set, keys)
	ft := new(fakeTimers).install(a.app)
	o := a.prefs.Overlay
	pill := pillFrame(o)
	var first string
	grew := false
	a.pump("the first recording", func() bool { return a.sess.State == session.Recording })
	first = a.sess.ID
	if s := a.ovNow(); s.frame != pill || s.text != "" || !s.visible {
		t.Fatalf("first starts: %+v", s)
	}
	a.pump("the first's end", func() bool {
		if a.sess.ID == first && a.ov.h == layout.LiveH {
			grew = true
			anchored(t, "first grown", o, a.ov.frame)
		}
		return a.sess.ID == first && a.sess.State.Ended()
	})
	if !grew {
		t.Error("the first never showed its text")
	}
	ft.advance(ovGone)
	if a.ov.visible {
		t.Fatal("the first stays on screen")
	}
	a.pump("the second recording", func() bool { return a.sess.ID != first && a.sess.State == session.Recording })
	if s := a.ovNow(); s.frame != pill || s.text != "" || !s.visible || a.ov.w != layout.PillW || a.ov.h != layout.PillH {
		t.Fatalf("second starts: %+v, face %d×%d", s, a.ov.w, a.ov.h)
	}
	a.pump("the second's end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
}

// TestOverlayCheckSteps: the Mac overlay check's states (overlay_check.go)
// are what they say, at one anchor; its sheet renders; its flag parses.
func TestOverlayCheckSteps(t *testing.T) {
	a, _, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	want := map[string]overlayMode{"pill": ovRecording, "live-short": ovRecording, "live-long": ovRecording, "working": ovWorking,
		"inserted": ovHidden, "pill-again": ovRecording, "notice": ovNotice, "notice-memory": ovNotice, "gone": ovHidden, "card-gone": ovHidden}
	for _, st := range cardCheckSteps("en") {
		if st.name != "card-gone" {
			want[st.name] = ovAnswer
		}
		want["draft"] = ovWorking
	}
	var first mygo.Rectangle
	for _, lang := range []string{"en", "zh-Hans"} {
		for _, st := range ovCheckSteps(lang) {
			st.apply(a)
			a.overlaySync()
			ft.advance(ovGone)
			if a.ov.mode != want[st.name] || a.ov.visible != (want[st.name] != ovHidden) {
				t.Errorf("%s/%s: %s, visible %v", lang, st.name, a.ov.mode, a.ov.visible)
			}
			if a.ov.visible {
				if first.Width == 0 {
					first = a.ov.frame
				}
				anchored(t, lang+"/"+st.name, a.prefs.Overlay, a.ov.frame)
			}
			switch st.name {
			case "pill", "pill-again":
				if a.ov.frame != pillFrame(a.prefs.Overlay) {
					t.Errorf("%s/%s: %+v", lang, st.name, a.ov.frame)
				}
			case "live-long":
				if a.ov.w != layout.LiveMaxW || a.ov.h != layout.LiveH {
					t.Errorf("%s/%s: %d×%d", lang, st.name, a.ov.w, a.ov.h)
				}
			case "card-short", "card-rewrite":
				if a.ov.w != layout.AnswerW || a.ov.h >= layout.CardChrome+3*layout.CardTextSize*2 {
					t.Errorf("%s/%s: %d×%d", lang, st.name, a.ov.w, a.ov.h)
				}
			case "card-long":
				if a.ov.h != layout.CardChrome+layout.CardBodyMax {
					t.Errorf("%s/%s: %d×%d", lang, st.name, a.ov.w, a.ov.h)
				}
			}
		}
	}
	old := os.Args
	t.Cleanup(func() { os.Args = old })
	os.Args = []string{"VoxtNew", "--overlay-check", "/tmp/x"}
	if overlayCheckDir() != "/tmp/x" {
		t.Error("the flag")
	}
	os.Args = []string{"VoxtNew"}
	if overlayCheckDir() != "" {
		t.Error("no flag")
	}
	// The sheet, from two captures.
	dir := t.TempDir()
	region := mygo.Rectangle{X: 100, Y: 700, Width: 516, Height: 150}
	var col []ovCheckShot
	for i := range 2 {
		img := image.NewRGBA(image.Rect(0, 0, 2*region.Width, 2*region.Height))
		draw.Draw(img, img.Bounds(), image.NewUniform(color.RGBA{40, 40, 50, 255}), image.Point{}, draw.Src)
		f := filepath.Join(dir, fmt.Sprint(i, ".png"))
		var b bytes.Buffer
		png.Encode(&b, img)
		os.WriteFile(f, b.Bytes(), 0o644)
		col = append(col, ovCheckShot{name: fmt.Sprint("step ", i), file: f, region: region, anchor: first})
	}
	out := filepath.Join(dir, "sheet.png")
	if err := ovCheckSheet(out, [][]ovCheckShot{col, col}); err != nil {
		t.Fatal(err)
	}
	if fi, err := os.Stat(out); err != nil || fi.Size() < 1000 {
		t.Errorf("sheet %v %v", fi, err)
	}
}
