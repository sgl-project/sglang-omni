package main

// The Meetings page (/meetings): past meetings newest first (dev builds
// only until C1-10), laid out like History: a search, the list, and the
// details of one in a side panel beside it, or over it when the window is
// narrow. The meeting settings open in a dialog from the header.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const meetingsPageSize = 20

// meetingPanelW is the details panel's width (DIPs at 100 %): a
// transcript needs more room than a History entry.
const meetingPanelW = 420

type meetingsView struct {
	list    []store.Meeting
	read    bool
	loading bool
	err     string
	seq     int
	query   string
	find    finder // Return steps through the matches (find_marks.go)
	// lines are, while searching, the transcript line holding the search
	// of each meeting listed (by ID), for its row.
	lines   map[string]string
	page    int
	hasMore bool
	listUI  ui.ListState
	detail  *meetingDetail
	setOpen bool   // the meeting settings dialog
	delID   string // the meeting the Delete dialog deletes, "" for none
}

type meetingsState struct {
	mv       meetingsView
	mtPlayer meetingPlayer
}

func (n *meetingsView) wants() string {
	return strings.TrimSpace(n.query) + "\x00" + string(rune('0'+n.page))
}

func (a *app) loadMeetings() {
	n := &a.mv
	n.seq++
	n.loading = true
	seq := n.seq
	want := n.wants()
	q := store.MeetingQuery{
		Search: strings.TrimSpace(n.query),
		Limit:  meetingsPageSize + 1,
		Offset: n.page * meetingsPageSize,
	}
	repo := a.lib.store.Meetings()
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		list, err := repo.ListMeetings(ctx, q)
		var lines map[string]string
		if q.Search != "" && err == nil {
			lines = map[string]string{}
			for _, mt := range list {
				if _, segs, err := repo.Meeting(ctx, mt.ID); err == nil {
					lines[mt.ID] = meetingLine(segs, q.Search)
				}
			}
		}
		a.post(func() {
			n := &a.mv
			if seq != n.seq || want != n.wants() {
				return
			}
			n.loading, n.read = false, true
			if err != nil {
				n.err = err.Error()
				return
			}
			n.hasMore = len(list) > meetingsPageSize
			if n.hasMore {
				list = list[:meetingsPageSize]
			}
			n.list, n.lines, n.err = list, lines, ""
		})
	})
}

// meetingLine is the first line of segs' texts or translations holding
// q, "" for none (the store's search also finds words in other forms).
func meetingLine(segs []store.MeetingSegment, q string) string {
	for _, sg := range segs {
		for _, s := range []string{sg.Text, sg.Translation} {
			for l := range strings.SplitSeq(s, "\n") {
				if l = strings.TrimSpace(l); len(findFolded(l, q)) > 0 {
					return l
				}
			}
		}
	}
	return ""
}

// meetingRowTexts are the texts a meeting's row shows, where the search
// marks its matches: its title, and the transcript line holding the
// search, else the first line of its summary (meetingSnippet).
func (n *meetingsView) meetingRowTexts(mt store.Meeting) []string {
	line := n.lines[mt.ID]
	if line == "" {
		line = meetingSnippet(mt)
	}
	return []string{meetingDisplayTitle(mt), line}
}

// meetingTitle is a meeting's title, else when it started (for file
// names).
func meetingTitle(mt store.Meeting) string {
	if mt.Title != "" {
		return mt.Title
	}
	return mt.CreatedAt.Format("2006-01-02 15:04")
}

// meetingDisplayTitle is a meeting's title as the list and its details
// show it.
func meetingDisplayTitle(mt store.Meeting) string {
	if mt.Title != "" {
		return mt.Title
	}
	return i18n.T("Untitled meeting")
}

// meetingStatusLabel is the row's mark for a meeting that is not done, ""
// for a done one.
func meetingStatusLabel(s store.MeetingStatus) string {
	switch s {
	case store.MeetingRecording, store.MeetingAsking:
		return i18n.T("In progress")
	case store.MeetingInterrupted:
		return i18n.T("Interrupted")
	case store.MeetingFinishing:
		return i18n.T("Finishing")
	case store.MeetingFailed:
		return i18n.T("Failed")
	}
	return ""
}

// meetingMeta is the row's second line: when (unless the title says
// it), how long, what it captured.
func meetingMeta(mt store.Meeting) string {
	parts := []string{mt.CreatedAt.Format("2006-01-02 15:04")}
	if mt.DurationMS > 0 {
		parts = append(parts, meetingDuration(mt.DurationMS))
	}
	switch mt.Mode {
	case store.MeetingModeSubtitles:
		parts = append(parts, i18n.T("System audio only"))
	case store.MeetingModeRecording:
		parts = append(parts, i18n.T("Microphone only"))
	default:
		parts = append(parts, i18n.T("Microphone and system audio"))
	}
	return strings.Join(parts, " · ")
}

// meetingDuration is a meeting's length as a list shows it: minutes,
// with hours past an hour.
func meetingDuration(ms int64) string {
	m := int((time.Duration(ms)*time.Millisecond + 30*time.Second) / time.Minute)
	switch {
	case m < 1:
		return i18n.T("Under a minute")
	case m < 60:
		return i18n.T("%d min", m)
	default:
		return i18n.T("%d h %d min", m/60, m%60)
	}
}

// meetingGroup is the heading a meeting is listed under, by its day.
func meetingGroup(at, now time.Time) string {
	y, m, d := now.Date()
	today := time.Date(y, m, d, 0, 0, 0, 0, now.Location())
	switch {
	case !at.Before(today):
		return i18n.T("Today")
	case !at.Before(today.AddDate(0, 0, -1)):
		return i18n.T("Yesterday")
	case !at.Before(today.AddDate(0, 0, -7)):
		return i18n.T("Previous 7 days")
	default:
		return at.Format("2006-01")
	}
}

// meetingSnippet is a row's second line: its summary's opening, else
// what it captured.
func meetingSnippet(mt store.Meeting) string {
	if s, ok := meeting.ParseStoredSummary(mt.Summary); ok {
		if line, _, _ := strings.Cut(strings.TrimSpace(s.Body), "\n"); line != "" {
			return strings.TrimLeft(line, "-*# ")
		}
	}
	switch mt.Mode {
	case store.MeetingModeSubtitles:
		return i18n.T("System audio only")
	case store.MeetingModeRecording:
		return i18n.T("Microphone only")
	default:
		return i18n.T("Microphone and system audio")
	}
}

// meetingItem is a row of the list: a day's heading, or a meeting.
type meetingItem struct {
	head string
	mt   *store.Meeting
}

func meetingItems(list []store.Meeting, now time.Time) []meetingItem {
	var out []meetingItem
	last := ""
	for i := range list {
		if g := meetingGroup(list[i].CreatedAt, now); g != last {
			out = append(out, meetingItem{head: g})
			last = g
		}
		out = append(out, meetingItem{mt: &list[i]})
	}
	return out
}

func (a *app) meetingsPage(c *ui.Context) {
	if !a.meetingEnabled() {
		a.notFoundPage(c, "/meetings")
		return
	}
	t := c.Theme()
	n := &a.mv
	if !n.read && !n.loading {
		a.loadMeetings()
	}
	active := a.meetingActive()
	items := meetingItems(n.list, time.Now())
	q := strings.TrimSpace(n.query)
	hits := findHits(q, len(items), func(i int) []string {
		if mt := items[i].mt; mt != nil {
			return n.meetingRowTexts(*mt)
		}
		return nil // a day's heading
	})
	if n.find.advance(hits[len(items)]) {
		row, _ := rowOf(hits, n.find.cur)
		n.listUI.ScrollIntoView(row)
	}
	cur, curAt := rowOf(hits, n.find.cur)
	widePageHeader(c, i18n.T("Meetings"), func() {
		if len(n.list) > 0 || n.query != "" || n.page > 0 {
			if n.find.field(c, "meetings-search", &n.query, i18n.T("Search"), px(t, 240)).
				Label(i18n.T("Search meetings")).Changed() {
				n.page = 0
				a.loadMeetings()
			}
			if hits[len(items)] > 0 { // none: the list says nothing matches, or the store found other word forms
				n.find.count(c, q, hits[len(items)])
			}
		}
		if iconButton(c.Key("mt-settings"), icons.Tuning2, i18n.T("Meeting settings")).Shrink(0).Clicked() {
			n.setOpen = true
		}
		if active {
			if primaryButton(c.Key("mt-end"), i18n.T("End meeting")).Clicked() {
				a.meetingFinish()
			}
		} else if primaryButton(c.Key("mt-start"), i18n.T("Start meeting")).Clicked() {
			a.startMeeting()
		}
	})
	pageBody(c).Children(func() {
		pad := func(fn func()) {
			ui.Column(c).Padding(0, px(t, sp6), px(t, sp3)).Shrink(0).Children(fn)
		}
		if n.err != "" {
			pad(func() {
				notice(c, "mt-err", noticeError, i18n.T("Could not read the meetings."), n.err, func() {
					if ghostButton(c.Key("mt-retry"), i18n.T("Try again")).Clicked() {
						a.loadMeetings()
					}
				}, nil)
			})
		}
		list, d := n.list, n.detail
		w := px(t, meetingPanelW)
		inline := a.pageWidth(c)-w >= px(t, historyListMinW)
		ui.Row(c).Grow(1).MinHeight(0).AlignItems(ui.Stretch).Children(func() {
			ui.Column(c).Grow(1).MinWidth(0).Children(func() {
				switch {
				case !n.read:
					// The first read is on its way: no empty state yet.
				case len(list) == 0:
					a.meetingsEmpty(c)
				default:
					ui.List(c.Key("meetings-list"), &n.listUI, len(items)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp2)).
						ItemKey(func(i int) any {
							if it := items[i]; it.mt != nil {
								return it.mt.ID
							}
							return "head/" + items[i].head
						}).Rows(func(r ui.ListRow) {
						if it := items[r.Index]; it.mt != nil {
							at := -1
							if r.Index == cur {
								at = curAt
							}
							a.meetingRow(c, *it.mt, q, at)
						} else {
							ui.Text(c, it.head).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(tok(t).Text3).
								Padding(px(t, sp4), px(t, sp6), px(t, sp1))
						}
					})
					if n.page > 0 || n.hasMore {
						a.meetingsPager(c)
					}
				}
			})
			if d != nil {
				closePanel := func() {
					if n.delID == "" { // a click in the Delete dialog is outside the drawer
						a.stopMeetingPlayback()
						n.detail = nil
					}
				}
				sidePanel(c, "mt-detail", w, inline, i18n.T("Meeting details"), closePanel, func() {
					a.meetingDetailSection(c, closePanel)
				})
			}
		})
	})
	a.meetingSettingsDialog(c)
	a.meetingDeleteDialog(c)
}

// meetingsEmpty is the list's empty state: no meeting yet, or none
// matching the search.
func (a *app) meetingsEmpty(c *ui.Context) {
	n := &a.mv
	if q := strings.TrimSpace(n.query); q != "" {
		emptyState(c, icons.Magnifer, i18n.T("Nothing matches “%s”", q),
			i18n.T("Try another search, or clear the field to see every meeting."), func() {
				if ghostButton(c.Key("meetings-clear"), i18n.T("Clear search")).Clicked() {
					n.query, n.page = "", 0
					a.loadMeetings()
				}
			})
		return
	}
	emptyState(c, meetingIcon, i18n.T("No meetings yet."),
		i18n.T("Start a meeting to record your microphone and, when supported, audio from other apps."), func() {
			if ghostButton(c.Key("mt-empty-settings"), i18n.T("Meeting settings")).Clicked() {
				n.setOpen = true
			}
		})
}

// meetingRow is a meeting in the list: its title and status, and when,
// how long and what it captured. A click shows its details, or hides them.
// The matches of the search q are marked, at the current one (-1 for none).
func (a *app) meetingRow(c *ui.Context, mt store.Meeting, q string, at int) {
	t := c.Theme()
	k := tok(t)
	n := &a.mv
	texts := n.meetingRowTexts(mt)
	title := texts[0] // the day heads the group, the time ends the row
	sel := n.detail != nil && n.detail.id == mt.ID
	row := ui.ButtonBase(c.Key("mt-row-"+mt.ID)).Padding(px(t, sp3), px(t, sp3)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
		Label(title).Gap(px(t, sp2)).AlignItems(ui.Start)
	switch {
	case sel:
		row.Background(k.Pressed)
	case row.Hovered():
		row.Background(k.Hover)
	}
	if row.Clicked() {
		if sel {
			a.stopMeetingPlayback()
			n.detail = nil
		} else {
			a.openMeetingDetail(mt.ID)
		}
	}
	row.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, sp1)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				markedText(c, title, q, curIn(at, 0)).FontSize(px(t, fsBody+1)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				if s := meetingStatusLabel(mt.Status); s != "" {
					tagChip(c, s)
				}
				ui.Text(c, mt.CreatedAt.Format("15:04")).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Shrink(0)
			})
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				markedText(c, texts[1], q, curIn(at, len(findFolded(title, q)))).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				if mt.DurationMS > 0 {
					ui.Text(c, meetingDuration(mt.DurationMS)).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Shrink(0)
				}
			})
		})
	})
}

func (a *app) meetingsPager(c *ui.Context) {
	t := c.Theme()
	n := &a.mv
	ui.Row(c).Padding(0, px(t, sp6), px(t, sp3)).Gap(px(t, sp2)).AlignItems(ui.Center).Justify(ui.End).Shrink(0).Children(func() {
		if ghostButton(c.Key("mt-page-prev"), i18n.T("Previous page")).Disabled(n.page == 0).Clicked() {
			n.page--
			a.loadMeetings()
		}
		caption(c, i18n.T("Page %d", n.page+1))
		if ghostButton(c.Key("mt-page-next"), i18n.T("Next page")).Disabled(!n.hasMore).Clicked() {
			n.page++
			a.loadMeetings()
		}
	})
}

// meetingDetailHeader is the detail panel's title row (title, then More
// with Reprocess and Show in Finder, Delete and Close), its facts, and
// Continue for a meeting whose final pass did not finish.
func (a *app) meetingDetailHeader(c *ui.Context, mt store.Meeting, close func()) {
	t := c.Theme()
	ui.Column(c).Gap(px(t, sp1)).Children(func() { a.meetingDetailTitle(c, mt, close) })
}

func (a *app) meetingDetailTitle(c *ui.Context, mt store.Meeting, close func()) {
	t := c.Theme()
	ui.Row(c).Gap(px(t, sp1)).AlignItems(ui.Center).Children(func() {
		sectionHeading(c, meetingDisplayTitle(mt)).Grow(1).MinWidth(0).SingleLine().Ellipsis("…")
		if mt.Status == store.MeetingDone || mt.Audio != "" {
			moreMenu(c.Key("mt-more"), "", func(m *ui.Menu) {
				if mt.Status == store.MeetingDone && m.Item(i18n.T("Reprocess transcript")).Chosen() {
					a.meetingReprocess(mt.ID)
				}
				if mt.Audio != "" && m.Item(i18n.T("Show in Finder")).Chosen() {
					revealInFinder(filepath.Join(a.dataDir(), "meetings", mt.Audio))
				}
			})
		}
		active := a.meetingActive() && mt.Status == store.MeetingRecording
		if iconButton(c.Key("mt-del"), icons.TrashBinMinimalistic, i18n.T("Delete meeting…")).Disabled(active).Clicked() {
			a.mv.delID = mt.ID
		}
		if iconButton(c.Key("panel-close"), icons.Close, i18n.T("Close")).Clicked() {
			close()
		}
	})
	caption(c, meetingMeta(mt))
	if mt.Status == store.MeetingInterrupted || mt.Status == store.MeetingFinishing {
		ui.Row(c).Margin(px(t, sp2), 0, 0, 0).Children(func() {
			if ui.Button(c.Key("mt-continue"), i18n.T("Continue processing")).Clicked() {
				a.meetingReprocess(mt.ID)
			}
		})
	}
}

func (a *app) meetingDeleteDialog(c *ui.Context) {
	n := &a.mv
	open := n.delID != ""
	if !open {
		return
	}
	dialogFrame(c, "mt-del-dlg", &open, i18n.T("Delete this meeting?"), "", 420, func() {
		ui.Text(c, i18n.T("Its transcript, summary and audio are deleted. This cannot be undone."))
	}, func() {
		ui.Spacer(c)
		if ui.Button(c.Key("mt-del-cancel"), i18n.T("Cancel")).AutoFocus().Clicked() {
			open = false
		}
		if recordButton(c.Key("mt-del-ok"), i18n.T("Delete meeting")).Clicked() {
			open = false
			a.deleteMeeting(n.delID)
		}
	})
	if !open {
		n.delID = ""
	}
}

// deleteMeeting deletes a meeting and its audio in the background.
func (a *app) deleteMeeting(id string) {
	n := &a.mv
	audio := ""
	for _, mt := range n.list {
		if mt.ID == id {
			audio = mt.Audio
		}
	}
	if n.detail != nil && n.detail.id == id {
		a.stopMeetingPlayback()
		n.detail = nil
	}
	a.async(func() {
		_ = a.lib.store.Meetings().DeleteMeeting(context.Background(), id)
		if audio != "" {
			_ = os.RemoveAll(filepath.Join(a.dataDir(), "meetings", audio))
		}
		a.post(a.loadMeetings)
	})
}

func (a *app) meetingSettingsDialog(c *ui.Context) {
	n := &a.mv
	if !n.setOpen {
		return
	}
	dialogFrame(c, "mt-set", &n.setOpen, i18n.T("Meeting settings"), "", 560, func() {
		a.meetingSettings(c)
	}, func() {
		ui.Spacer(c)
		if primaryButton(c.Key("mt-set-done"), i18n.T("Done")).Clicked() {
			n.setOpen = false
		}
	})
}
