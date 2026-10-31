package main

import (
	"bytes"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// ovFlowStep is what the overlay shows at one point of a flow.
type ovFlowStep struct {
	state  session.State
	mode   overlayMode
	w, h   int
	bottom int    // the face's bottom edge (the anchor), screen DIPs
	centre int    // twice its centre
	text   string // what its text shows: "", "partial", "draft", "raw", "card:<…>"
}

func (s ovFlowStep) String() string {
	return fmt.Sprintf("%s/%s %d×%d %s", s.state, s.mode, s.w, s.h, s.text)
}

// ovFlow records each distinct step of the overlay while a session runs;
// with shots, a picture of each (the view measures its texts).
type ovFlow struct {
	a     *sessionApp
	steps []ovFlowStep
	shots [][]byte
}

func (f *ovFlow) sample() {
	a := f.a
	st := ovFlowStep{state: a.sess.State, mode: a.ov.mode, w: a.ov.w, h: a.ov.h}
	if a.ov.shown {
		st.bottom = a.ov.frame.Y + a.ov.frame.Height - layout.Pad
		st.centre = 2*a.ov.frame.X + a.ov.frame.Width
	} else {
		st.mode, st.w, st.h = ovHidden, 0, 0
	}
	switch a.ov.mode {
	case ovRecording, ovWorking:
		line := liveLineText(a.sess)
		switch {
		case line == "":
		case strings.TrimSpace(a.sess.Draft) != "" && line == strings.Join(strings.Fields(a.sess.Draft), " "):
			st.text = "draft"
		case a.sess.State == session.Recording:
			st.text = "partial"
		default:
			st.text = "raw"
		}
	case ovAnswer:
		v := a.cardNow()
		st.text = "card"
		if v.conversation {
			st.text = "card:conversation"
		}
	}
	if n := len(f.steps); n > 0 {
		p := f.steps[n-1]
		if p.mode == st.mode && p.w == st.w && p.h == st.h && p.text == st.text && p.state == st.state {
			return
		}
	}
	f.steps = append(f.steps, st)
	if os.Getenv("VOXTNEW_FLOW_SHOTS") != "" {
		f.shots = append(f.shots, f.picture())
	}
}

// picture draws the overlay's window as it shows now, at 2×, over a dark
// desktop (nothing when hidden).
func (f *ovFlow) picture() []byte {
	a := f.a
	w, h := 120, 60
	if a.ov.shown {
		w, h = a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad
	}
	tt := hudTester(func(c *ui.Context) {
		ui.Box(c).Absolute().Fill().Background(ui.RGB(30, 31, 36))
		if a.ov.shown {
			ui.Box(c).Absolute().Fill().Children(func() { a.overlayView(c) })
		}
	}, w, h)
	tt.SetScale(2)
	frames(tt)
	var b bytes.Buffer
	png.Encode(&b, tt.Image())
	return b.Bytes()
}

// run pumps until cond, sampling the overlay at each step.
func (f *ovFlow) run(what string, cond func() bool) {
	f.a.pump(what, func() bool { f.sample(); return cond() })
}

func (f *ovFlow) String() string {
	var b strings.Builder
	for _, s := range f.steps {
		fmt.Fprintf(&b, "\n  %v", s)
	}
	return b.String()
}

// TestOverlayFlowsTrace logs each flow's steps (VOXTNEW_FLOWS=1).
func TestOverlayFlowsTrace(t *testing.T) {
	for _, fl := range overlayFlows() {
		t.Run(fl.name, func(t *testing.T) {
			f := fl.run(t)
			t.Logf("%s:%v", fl.name, f)
			if dir := os.Getenv("VOXTNEW_FLOW_SHOTS"); dir != "" {
				os.MkdirAll(dir, 0o755)
				for i, b := range f.shots {
					os.WriteFile(filepath.Join(dir, fmt.Sprintf("%s-%02d-%s.png", fl.name, i, strings.ReplaceAll(f.steps[i].String(), "/", "-"))), b, 0o644)
				}
			}
		})
	}
}

type ovFlowCase struct {
	name string
	run  func(t *testing.T) *ovFlow
}

func overlayFlows() []ovFlowCase {
	draft := func(text string, delay time.Duration) *fakeChat {
		return &fakeChat{text: text, delay: delay}
	}
	long := "We are moving the release to Thursday, to finish the onboarding and the history page first, and then the translation card."
	two := func(t *testing.T, opts string, kind, said string, chat *fakeChat, set func(*settings.Settings)) *ovFlow {
		s, keys, _ := fileSetup(t, said, chat)
		ev := script(t, hotkeyLine(0, kind+"/0", proto.PhaseTrigger), hotkeyLine(500, kind+"/0", proto.PhaseTrigger))
		a := newSessionApp(t, opts+ev, func(st *settings.Settings) {
			s(st)
			flowSettings(st)
			if set != nil {
				set(st)
			}
		}, keys)
		f := &ovFlow{a: a}
		f.run("the end", func() bool { return a.sess.State.Ended() })
		f.run("settle", func() bool { return true })
		return f
	}
	one := func(t *testing.T, opts string, kind string, chat *fakeChat, set func(*settings.Settings)) *ovFlow {
		s, keys, _ := fileSetup(t, "unused", chat)
		ev := script(t, hotkeyLine(0, kind+"/0", proto.PhaseTrigger))
		a := newSessionApp(t, opts+ev, func(st *settings.Settings) {
			s(st)
			flowSettings(st)
			if set != nil {
				set(st)
			}
		}, keys)
		f := &ovFlow{a: a}
		f.run("the end", func() bool { return a.sess.State.Ended() })
		return f
	}
	return []ovFlowCase{
		{"transcribe", func(t *testing.T) *ovFlow {
			return two(t, "", "transcribe", "so the release moves to thursday and we finish the onboarding first", draft("So the release moves to Thursday, and we finish the onboarding first.", 100*time.Millisecond), nil)
		}},
		{"transcribe-card", func(t *testing.T) *ovFlow {
			return two(t, "readonly,", "transcribe", "so the release moves to thursday and we finish the onboarding first", draft("So the release moves to Thursday, and we finish the onboarding first.", 100*time.Millisecond), nil)
		}},
		{"translate-speech", func(t *testing.T) *ovFlow {
			return two(t, "", "translate", "发布推迟到周四，先把引导页和历史页做完，再做翻译卡片", draft(long, 150*time.Millisecond), nil)
		}},
		{"translate-speech-card", func(t *testing.T) *ovFlow {
			return two(t, "readonly,", "translate", "发布推迟到周四，先把引导页和历史页做完，再做翻译卡片", draft(long, 150*time.Millisecond), nil)
		}},
		{"translate-selection", func(t *testing.T) *ovFlow {
			return one(t, helpertest.Opt("selection", "发布推迟到周四，先把引导页和历史页做完，再做翻译卡片")+",", "translate", draft(long, 150*time.Millisecond), nil)
		}},
		{"translate-selection-readonly", func(t *testing.T) *ovFlow {
			return one(t, "readonly,"+helpertest.Opt("selection", "发布推迟到周四，先把引导页和历史页做完，再做翻译卡片")+",", "translate", draft(long, 150*time.Millisecond), nil)
		}},
		{"rewrite", func(t *testing.T) *ovFlow {
			return two(t, "", "rewrite", "write that the release moves to thursday", draft(`{"title":"Release","content":"`+long+`"}`, 150*time.Millisecond), nil)
		}},
		{"rewrite-selection", func(t *testing.T) *ovFlow {
			return two(t, helpertest.Opt("selection", "release thursday onboarding first")+",", "rewrite", "make it a full sentence", draft(long, 150*time.Millisecond), nil)
		}},
		{"follow-up", func(t *testing.T) *ovFlow {
			chat := &fakeChat{texts: []string{`{"title":"Release","content":"` + long + `"}`, "Thursday: onboarding, history, then the card."}, delay: 150 * time.Millisecond}
			s, keys, tr := fileSetup(t, "write that the release moves to thursday", chat)
			ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
			a := newSessionApp(t, ev, func(st *settings.Settings) { s(st); flowSettings(st) }, keys)
			a.pump("the answer", func() bool { return a.sess.State.Ended() })
			f := &ovFlow{a: a}
			f.sample()
			tr.setText("shorter")
			first := a.sess.ID
			a.followUpAnswer("") // Continue
			f.run("the follow-up's recording", func() bool {
				return a.sess.ID != first && a.sess.State == session.Recording && a.sess.HeardAt.After(a.sess.StartedAt)
			})
			time.Sleep(400 * time.Millisecond)
			a.stopSession() // Done
			f.run("the follow-up's end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
			f.run("settle", func() bool { return true })
			return f
		}},
	}
}

// flowSettings: the pictures in Chinese.
func flowSettings(st *settings.Settings) {
	if os.Getenv("VOXTNEW_FLOW_SHOTS") != "" {
		st.General.Language = "zh-Hans"
	}
}

// TestOverlayFlows: every flow from its shortcut to its result, each step
// of the overlay recorded (10-15, third round): the anchor never moves;
// before the result, every flow shows the same pill (the mini pill, or
// one line of text under its row: as wide as the text up to 300); the
// card comes only with the result, at its one width, and never goes back
// to the pill; a follow-up stays in its card.
func TestOverlayFlows(t *testing.T) {
	want := map[string]string{
		"transcribe":                   "recording working",
		"transcribe-card":              "recording working answer",
		"translate-speech":             "working recording working",
		"translate-speech-card":        "working recording working answer",
		"translate-selection":          "working",
		"translate-selection-readonly": "working answer",
		"rewrite":                      "recording working answer",
		"rewrite-selection":            "recording working answer",
		"follow-up":                    "answer",
	}
	for _, fl := range overlayFlows() {
		t.Run(fl.name, func(t *testing.T) {
			f := fl.run(t)
			var modes []string
			var anchor *ovFlowStep
			card := false
			for i, s := range f.steps {
				if s.mode == ovHidden {
					continue
				}
				if n := len(modes); n == 0 || modes[n-1] != s.mode.String() {
					modes = append(modes, s.mode.String())
				}
				if anchor == nil {
					anchor = &f.steps[i]
				} else if s.bottom != anchor.bottom || abs(s.centre-anchor.centre) > 1 { // an odd width: half a point
					t.Errorf("step %v: the anchor moved: bottom %d (was %d), centre %d (was %d)", s, s.bottom, anchor.bottom, s.centre/2, anchor.centre/2)
				}
				switch s.mode {
				case ovAnswer:
					card = true
					if s.w != layout.AnswerW {
						t.Errorf("step %v: the card is %d wide, not %d", s, s.w, layout.AnswerW)
					}
					if fl.name != "follow-up" && s.state.Live() {
						t.Errorf("step %v: a card before the result", s)
					}
				case ovRecording, ovWorking:
					if card {
						t.Errorf("step %v: back to the pill after the card", s)
					}
					switch {
					case s.text == "" && (s.w != layout.PillW || s.h != layout.PillH):
						t.Errorf("step %v: the mini pill is %d×%d", s, s.w, s.h)
					case s.text != "" && (s.h != layout.LiveH || s.w < layout.PillW || s.w > layout.LiveMaxW):
						t.Errorf("step %v: the line of text is %d×%d (the pill's rule: %d to %d wide, %d high)", s, s.w, s.h, layout.PillW, layout.LiveMaxW, layout.LiveH)
					}
				}
			}
			if got := strings.Join(modes, " "); got != want[fl.name] {
				t.Errorf("modes %q, want %q:%v", got, want[fl.name], f)
			}
		})
	}
}
