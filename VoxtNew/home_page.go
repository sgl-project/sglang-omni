package main

// Home (/home): a slogan with Transcribe's shortcut in a key
// cap; what is still missing before VoxtNew can work (a speech model,
// a language model, the Microphone, Accessibility, the shortcut), each
// line the way to fix it, and nothing once all is ready; until VoxtNew
// has given a result once, a Try area that runs a whole transcription on
// the page (Start, or the shortcut pressed while VoxtNew is in front);
// today's count of results and characters; the last five results, to
// copy or open in History.
//
// Today's numbers and the recent results are read from the history in the
// background: when Home shows, and again when a session keeps an entry
// (historyChanged).

import (
	"context"
	"strings"
	"time"
	"unicode"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// homeRecent is how many recent results Home lists.
const homeRecent = 5

// homeState is Home's view state.
type homeState struct {
	hm struct {
		// read: the history was read once; loading a read is on its way
		// (seq numbers them: only the last lands); stale an entry was kept
		// since.
		read, loading, stale bool
		seq                  int
		today                dayStats
		recent               []store.Entry
		err                  string
		// block says why the Try area's Start did not start.
		block *sessionNotice
		// tryOn: the Try area showed in this visit to Home (homeShowsTry).
		tryOn bool
	}
}

// dayStats are a day's finished results: how many, and their characters
// (without spaces), by feature too.
type dayStats struct {
	count, chars int
	byKind       map[store.Kind]int
}

// readHome reads the recent results and today's numbers (since midnight
// at now) from the history: entries come newest first, so it reads pages
// until one is older than today.
func readHome(ctx context.Context, h store.History, now time.Time) (dayStats, []store.Entry, error) {
	y, m, d := now.Date()
	midnight := time.Date(y, m, d, 0, 0, 0, 0, now.Location())
	st := dayStats{byKind: map[store.Kind]int{}}
	var recent []store.Entry
	const page = 200
	for off := 0; ; off += page {
		es, err := h.List(ctx, store.HistoryQuery{Limit: page, Offset: off})
		if err != nil {
			return st, recent, err
		}
		older := false
		for _, e := range es {
			done := e.Status == store.StatusDone && strings.TrimSpace(e.Result) != ""
			if done && len(recent) < homeRecent {
				recent = append(recent, e)
			}
			if e.CreatedAt.Before(midnight) {
				older = true
				continue
			}
			if done {
				st.count++
				st.chars += countChars(e.Result)
				st.byKind[e.Kind]++
			}
		}
		if len(es) < page || (older && len(recent) >= homeRecent) {
			return st, recent, nil
		}
	}
}

// loadHome reads Home's numbers and recent results in the background.
func (a *app) loadHome() {
	hm := &a.hm
	hm.seq++
	hm.loading, hm.stale = true, false
	seq := hm.seq
	h := a.lib.store.History()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		st, recent, err := readHome(ctx, h, time.Now())
		a.post(func() {
			if seq != a.hm.seq {
				return
			}
			hm := &a.hm
			hm.loading, hm.read = false, true
			if err != nil {
				hm.err = err.Error()
				return
			}
			hm.today, hm.recent, hm.err = st, recent, ""
			if len(recent) > 0 {
				a.markTried() // a history from before the flag
			}
		})
	})
}

// homeChanged runs when the history changed: Home reads it again now
// when it shows, else when it shows next.
func (a *app) homeChanged() {
	if a.currentPage() == "home" && !a.settingsOpen {
		a.loadHome()
		return
	}
	a.hm.stale = true
}

func (a *app) homePage(c *ui.Context) {
	t := c.Theme()
	hm := &a.hm
	if !hm.loading && (!hm.read || hm.stale) {
		a.loadHome()
	}
	if !a.perms.checked {
		a.recheckPermissions()
	}
	if a.sess.Try && a.sess.State == session.Recording {
		c.After(250 * time.Millisecond) // the Try area's clock
	}
	pageHeader(c, i18n.T("Home"), func() {
		if ghostIconButton(c.Key("home-guide"), icons.Compass, i18n.T("Setup guide")).Label(i18n.T("Open the setup guide")).Clicked() {
			a.openOnboarding()
		}
	})
	ui.Scroll(c.Key("home-scroll")).Grow(1).MinHeight(0).Children(func() {
		ui.Column(c).Padding(0, px(t, sp6), px(t, sp6)).Gap(px(t, sp3)).MaxWidth(px(t, 880)).Children(func() {
			a.homeSlogan(c)
			a.homeReadiness(c)
			mine := a.sess.Try && a.sess.ID != "" && !a.obOwns(a.sess.ID)
			if a.homeShowsTry(mine) {
				sectionTitle(c, i18n.T("Try it"))
				a.tryCard(c, tryOpts{
					prefix: "home", feature: hotkey.Transcribe, mine: mine,
					intro: a.tryIntro(hotkey.Transcribe), block: &hm.block,
					start: func() *sessionNotice { _, n := a.startTry(featureKind(hotkey.Transcribe)); return n },
				})
			}
			a.homeToday(c)
			a.homeRecentList(c)
		})
	})
	a.historyDeleteDialog(c) // the recent results' Delete (history_menu.go)
}

// ---- The slogan ----

// sloganMark stands for the keys in a translated hint, which is cut
// around it to put a key cap there.
const sloganMark = "\x00"

// sloganHint is the slogan's second half, how to start Transcribe with
// its first shortcut (h, nil for none), cut around the keys: "Press ",
// "fn", " to start". keys is "" without a shortcut: then before is the
// whole hint.
func sloganHint(h *settings.Hotkey) (before, keys, after string) {
	if h == nil {
		return i18n.T("Add a shortcut to start"), "", ""
	}
	var hint string
	switch h.Trigger {
	case settings.TriggerDouble:
		hint = i18n.T("Double-press %s to start", sloganMark)
	case settings.TriggerLong:
		hint = i18n.T("Hold %s and speak", sloganMark)
	default:
		hint = i18n.T("Press %s to start", sloganMark)
	}
	before, after, _ = strings.Cut(hint, sloganMark)
	return before, uitext.HotkeyText(h.Keys), after
}

// homeSlogan is the line under Home's title: what VoxtNew does, then how
// to start it (Transcribe's shortcut in a key cap, or a link to add one).
func (a *app) homeSlogan(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	var first *settings.Hotkey
	if hs := hotkey.HotkeysOf(a.prefs.Features, hotkey.Transcribe); len(hs) > 0 {
		first = &hs[0]
	}
	before, keys, after := sloganHint(first)
	text := func(s string, col ui.Color) {
		if s = strings.TrimSpace(s); s != "" {
			ui.Text(c, s).FontSize(px(t, fsBody)).TextColor(col).SingleLine()
		}
	}
	ui.Row(c.Key("home-slogan")).Wrap().GapX(px(t, 6)).GapY(px(t, sp1)).AlignItems(ui.Center).Children(func() {
		text(i18n.T("Speak anywhere, it types at your cursor"), k.Text)
		text("·", k.Text3)
		if keys == "" {
			l := ui.ButtonBase(c.Key("home-slogan-add")).Label(before)
			l.Children(func() {
				lt := ui.Text(c, before).FontSize(px(t, fsBody)).TextColor(k.AccentInk).SingleLine()
				if l.Hovered() {
					lt.Underline()
				}
			})
			if l.Clicked() {
				a.navigate("/transcribe")
			}
			return
		}
		// The hint's words and its key cap, in one piece that wraps whole.
		ui.Row(c).Gap(px(t, 6)).AlignItems(ui.Center).Shrink(0).Children(func() {
			text(before, k.Text2)
			keycap(c, keys)
			text(after, k.Text2)
		})
	})
}

// homeToday is today's numbers.
func (a *app) homeToday(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	hm := &a.hm
	sectionTitle(c, i18n.T("Today"))
	if !a.prefs.Privacy.KeepHistory {
		caption(c, i18n.T("History is off, so nothing is counted."))
		return
	}
	st := hm.today
	tile := func(key, value, label string) {
		fillCard(c.Key(key)).Gap(px(t, 2)).Grow(1).Basis(px(t, 140)).MinWidth(px(t, 120)).Padding(px(t, sp3), px(t, sp4)).Children(func() {
			ui.Text(c, value).FontSize(px(t, fsTitle)).FontWeight(600).SingleLine()
			ui.Text(c, label).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
		})
	}
	ui.Row(c.Key("home-today")).Wrap().Gap(px(t, sp3)).Children(func() {
		tile("home-count", itoa(st.count), i18n.N(st.count, "result", "results"))
		tile("home-chars", groupDigits(st.chars), i18n.N(st.chars, "character", "characters"))
		// By feature: each one's icon and count.
		fillCard(c.Key("home-kinds")).Gap(px(t, 2)).Grow(1).Basis(px(t, 140)).MinWidth(px(t, 120)).Padding(px(t, sp3), px(t, sp4)).Children(func() {
			ui.Row(c).Gap(px(t, sp3)).AlignItems(ui.Center).Height(px(t, fsTitle*1.3)).Children(func() {
				for _, kd := range store.Kinds {
					ui.Row(c.Key("kind-" + string(kd))).Gap(px(t, 4)).AlignItems(ui.Center).Label(uitext.KindTitle(kd) + " " + itoa(st.byKind[kd])).Tooltip(uitext.KindTitle(kd)).Children(func() {
						ui.Icon(c, kindIcon(kd).Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2)
						ui.Text(c, itoa(st.byKind[kd])).FontSize(px(t, fsSection)).FontWeight(600).SingleLine()
					})
				}
			})
			ui.Text(c, i18n.T("By feature")).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
		})
	})
}

// homeRecentList is the last results: each copies, or opens in History;
// its context menu copies, opens or deletes it.
func (a *app) homeRecentList(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	hm := &a.hm
	// No margin above, unlike sectionTitle's: Open History's face makes
	// the row taller than its title, which keeps Today's distance.
	ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
		sectionTitleText(c, i18n.T("Recent results")).Grow(1)
		if ghostButton(c.Key("home-history"), i18n.T("Open History")).Clicked() {
			a.navigate("/history")
		}
	})
	if hm.err != "" {
		notice(c, "home-err", noticeError, i18n.T("Could not read the history."), hm.err, nil, nil)
		return
	}
	if len(hm.recent) == 0 {
		if hm.read {
			caption(c, i18n.T("Results show here once you use a shortcut."))
		}
		return
	}
	c.After(time.Minute) // the times are relative
	card(c.Key("home-recent")).Padding(px(t, sp1)).Gap(0).Children(func() {
		for _, e := range hm.recent {
			title, _ := uitext.EntryTitle(e)
			row := ui.ButtonBase(c.Key("home-recent-"+e.ID)).Padding(px(t, sp2), px(t, sp3)).Radius(px(t, radiusControl)).
				Gap(px(t, sp3)).AlignItems(ui.Center).Label(title).Tooltip(i18n.T("Open in History"))
			if row.Hovered() {
				row.Background(k.Hover)
			}
			row.Children(func() {
				ui.Icon(c, kindIcon(e.Kind).Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2).Shrink(0).Label(uitext.KindTitle(e.Kind))
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
					ui.Text(c, title).SingleLine().Ellipsis("…").MinWidth(0)
					ui.Text(c, uitext.EntryMeta(e)).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0)
				})
				if iconButton(c.Key("copy"), icons.Copy, i18n.T("Copy")).Label(i18n.T("Copy: %s", title)).Clicked() {
					c.Services().WriteClipboard(e.Result)
					a.toast(i18n.T("Copied"))
				}
			})
			if row.Clicked() {
				a.openHistoryEntry(e)
			}
			a.entryMenu(c, row, e, false, func() { a.openHistoryEntry(e) }) // history_menu.go
		}
	})
}

// openHistoryEntry shows an entry in History: its feature's tab, its
// details open.
func (a *app) openHistoryEntry(e store.Entry) {
	h := &a.hist
	for i, kd := range store.Kinds {
		if kd == e.Kind {
			if h.tab != i {
				h.tab, h.listFor = i, ""
			}
		}
	}
	h.query = ""
	h.sel = e.ID
	a.navigate("/history")
}

// countChars counts the characters of s that are not spaces.
func countChars(s string) int {
	n := 0
	for _, r := range s {
		if !unicode.IsSpace(r) {
			n++
		}
	}
	return n
}

// groupDigits writes n with thousands separators: 12,345.
func groupDigits(n int) string {
	s := itoa(n)
	if n < 0 || len(s) <= 3 {
		return s
	}
	var b strings.Builder
	for i, r := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	return b.String()
}
