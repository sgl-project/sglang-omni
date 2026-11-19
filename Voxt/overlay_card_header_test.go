package main

import (
	"math"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestLanguageMenuPlace: the language menu opens over the card when the
// overlay is at the bottom of the screen, under the capsule when it is at
// the top, its right edge under the capsule's; it goes when the card
// goes, or another app comes to the front.
func TestLanguageMenuPlace(t *testing.T) {
	for _, pos := range []string{"bottom", "top"} {
		t.Run(pos, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			shortScreen(t, a)
			// Saved, so that a later save (Setup.Tried) keeps it.
			a.updateSettings(func(s *settings.Settings) { s.Overlay.Position = pos })
			cardSteps()[0].apply(a) // a translation in the card
			a.overlaySync()
			renderOverlay(t, a)
			card := a.ov.frame
			tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
			frames(tt)
			capsule, ok := tt.Find(i18n.T("Continue")) // the capsules end at its right
			if !ok {
				t.Fatalf("no Continue in %q", firstTexts(tt))
			}
			a.toggleLangMenu()
			m := a.ov.lang.frame
			if !a.ov.lang.shown || a.ov.frame != card {
				t.Fatalf("shown %v, card %+v (was %+v)", a.ov.lang.shown, a.ov.frame, card)
			}
			faceY := card.Y + layout.Pad
			capRight := card.X + int(math.Round(float64(capsule.X+capsule.W))) // the menu ends under the capsules' right edge
			if m.X+m.Width-layout.Pad != capRight || m.Width != ovLangW+2*layout.Pad || m.Height != langMenuHeight()+2*layout.Pad {
				t.Errorf("menu %+v, the capsules' right edge %d", m, capRight)
			}
			switch {
			case pos == "bottom" && m.Y+m.Height-layout.Pad != faceY-ovLangGap:
				t.Errorf("bottom: the menu ends at %d, the card's top is %d", m.Y+m.Height-layout.Pad, faceY)
			case pos == "top" && m.Y+layout.Pad != faceY+layout.CardPadTop+layout.CardHeadH+ovLangGap:
				t.Errorf("top: the menu starts at %d", m.Y+layout.Pad)
			}
			if len(compose.TargetLanguages)*ovLangRowH+2*ovLangPad > ovLangMaxH && langMenuHeight() != ovLangMaxH {
				t.Errorf("height %d", langMenuHeight())
			}
			// Another app in front: it goes; the card stays.
			a.helperNotified(appActivated(proto.App{PID: 9, BundleID: "com.apple.Notes", Name: "Notes"}))
			if a.ov.lang.shown || a.answer.ui.langOpen || !a.answer.open {
				t.Errorf("another app: menu %v", a.ov.lang.shown)
			}
			// The card going takes it.
			a.toggleLangMenu()
			a.closeAnswer()
			if a.ov.lang.shown || a.answer.ui.langOpen {
				t.Error("the menu stays without the card")
			}
		})
	}
}

// TestCardHeaderFits: at 480 the header holds its actions whatever the
// title: the language capsule at its widest (128 of name) and four
// buttons; Continue and four; Done and four over a conversation. A long
// title ends in "…" before them; nothing overlaps or leaves the card.
func TestCardHeaderFits(t *testing.T) {
	long := "A very long answer title that the model wrote, longer than any header could hold in one line"
	for _, c := range []struct {
		name    string
		apply   func(a *app)
		actions []string // left to right
	}{
		{"translate", func(a *app) {
			snap(a, session.Snapshot{ID: "h1", Kind: store.KindTranslate, State: session.Recording})
			snap(a, session.Snapshot{ID: "h1", Kind: store.KindTranslate, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
				Selection: "x", TargetLanguage: "zh-Hant", Result: "你好。"}) // "Traditional Chinese": the widest name
		}, []string{"Translate into", "Continue", "Insert", "Copy", "Make the card larger", "Close"}},
		{"rewrite", func(a *app) {
			snap(a, session.Snapshot{ID: "h2", Kind: store.KindRewrite, State: session.Recording})
			snap(a, session.Snapshot{ID: "h2", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
				Raw: "write", Title: long, Result: "Text."})
		}, []string{"Continue", "Insert", "Copy", "Make the card larger", "Close"}},
		{"conversation", func(a *app) {
			snap(a, session.Snapshot{ID: "h3", Kind: store.KindRewrite, State: session.Recording})
			snap(a, session.Snapshot{ID: "h3", Kind: store.KindRewrite, State: session.Done, Card: true, Delivery: deliver.ReasonCard,
				Raw: "write", Title: long, Result: "Text."})
			a.followUp, a.answer.conversation = "h3", true
			snap(a, session.Snapshot{ID: "h4", Kind: store.KindRewrite, State: session.Recording, StartedAt: time.Now(), HeardAt: time.Now(), Levels: wave()})
		}, []string{"Done", "Insert", "Copy", "Make the card larger", "Close"}},
	} {
		for _, lang := range []string{"en", "zh-Hans"} {
			t.Run(c.name+"/"+lang, func(t *testing.T) {
				a, _, _ := newTestApp(t)
				if lang != "en" {
					a.updateSettings(func(s *settings.Settings) { s.General.Language = lang })
				}
				c.apply(a)
				renderOverlay(t, a)
				tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
				frames(tt)
				right := float32(layout.Pad + layout.AnswerW - layout.CardPadX)
				prev := float32(-1)
				for _, label := range c.actions {
					r, ok := tt.Find(i18n.T(label))
					if !ok {
						t.Fatalf("no %q in %q", i18n.T(label), firstTexts(tt))
					}
					if r.X < prev || r.X+r.W > right+0.5 || r.H > layout.CardBtn+0.5 {
						t.Errorf("%q at %+v (after %v, the card's padding at %v)", label, r, prev, right)
					}
					prev = r.X + r.W
				}
				first, _ := tt.Find(i18n.T(c.actions[0]))
				// The name at its widest (112) and the capsule's padding and
				// chevron; Chinese names are short.
				if want := map[string]float32{"en": 140, "zh-Hans": 70}[lang]; c.name == "translate" && first.W < want {
					t.Errorf("the capsule is %v wide: the language's name is cut", first.W)
				}
				if c.name == "translate" {
					title, ok := tt.Find(i18n.T("Translation"))
					// The capsule names the card: no title to cut to "…" beside
					// Continue (#3897). (The Chinese icon's label is the title.)
					if ok && title.W > layout.CardIcon+0.5 {
						t.Errorf("a title %+v beside the capsule", title)
					}
				}
				if c.name == "rewrite" {
					title, ok := tt.Find(long)
					if !ok {
						t.Fatalf("no title in %q", firstTexts(tt))
					}
					if title.X+title.W > first.X-layout.CardHeadGap+0.5 {
						t.Errorf("the title %+v runs into %q at %v", title, c.actions[0], first.X)
					}
				}
			})
		}
	}
}
