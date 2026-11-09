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
			n.list, n.err = list, ""
		})
	})
}

// meetingTitle is a meeting's title, else when it started.
func meetingTitle(mt store.Meeting) string {
	if mt.Title != "" {
		return mt.Title
	}
	return mt.CreatedAt.Format("2006-01-02 15:04")
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
	var parts []string
	if mt.Title != "" {
		parts = append(parts, mt.CreatedAt.Format("2006-01-02 15:04"))
	}
	if mt.DurationMS > 0 {
		parts = append(parts, (time.Duration(mt.DurationMS) * time.Millisecond).Round(time.Second).String())
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
	widePageHeader(c, i18n.T("Meetings"), func() {
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
		if len(n.list) > 0 || n.query != "" || n.page > 0 {
			ui.Row(c).Padding(px(t, sp1), px(t, sp6), px(t, sp3)).AlignItems(ui.Center).Shrink(0).Children(func() {
				if searchField(c, "meetings-search", &n.query, i18n.T("Search"), px(t, 360)).
					Label(i18n.T("Search meetings")).Changed() {
					n.page = 0
					a.loadMeetings()
				}
			})
		}
		if active {
			pad(func() { notice(c, "mt-active", noticeInfo, i18n.T("Meeting in progress"), "", nil, nil) })
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
					ui.List(c.Key("meetings-list"), &n.listUI, len(list)).Grow(1).MinHeight(0).Gap(px(t, 2)).Padding(0, 0, px(t, sp2)).
						ItemKey(func(i int) any { return list[i].ID }).Rows(func(r ui.ListRow) {
						a.meetingRow(c, list[r.Index])
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
				sidePanel(c, "mt-detail", w, inline, i18n.T("Meeting detail"), closePanel, func() {
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
				if ghostButton(c.Key("meetings-clear"), i18n.T("Clear the search")).Clicked() {
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
func (a *app) meetingRow(c *ui.Context, mt store.Meeting) {
	t := c.Theme()
	k := tok(t)
	n := &a.mv
	title := meetingTitle(mt)
	sel := n.detail != nil && n.detail.id == mt.ID
	row := ui.ButtonBase(c.Key("mt-row-"+mt.ID)).Padding(px(t, sp2), px(t, sp2)).Margin(0, px(t, sp4)).Radius(px(t, radiusControl)).
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
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 3)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ui.Text(c, title).SingleLine().Ellipsis("…").MinWidth(0).Grow(1)
				if s := meetingStatusLabel(mt.Status); s != "" {
					tagChip(c, s)
				}
			})
			ui.Text(c, meetingMeta(mt)).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0)
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

// meetingActions are the details panel's actions on the meeting: finish
// or redo its final pass, show its audio, delete it.
func (a *app) meetingActions(c *ui.Context, mt store.Meeting) {
	t := c.Theme()
	ui.Row(c).Gap(px(t, sp2)).Wrap().Children(func() {
		switch mt.Status {
		case store.MeetingInterrupted, store.MeetingFinishing:
			if primaryButton(c.Key("mt-continue"), i18n.T("Continue final pass")).Clicked() {
				a.meetingReprocess(mt.ID)
			}
		case store.MeetingDone:
			if ghostButton(c.Key("mt-reprocess"), i18n.T("Reprocess transcript")).Clicked() {
				a.meetingReprocess(mt.ID)
			}
		}
		if mt.Audio != "" {
			if ghostButton(c.Key("mt-show"), i18n.T("Show in Finder")).Clicked() {
				revealInFinder(filepath.Join(a.dataDir(), "meetings", mt.Audio))
			}
		}
		active := a.meetingActive() && mt.Status == store.MeetingRecording
		if ghostButton(c.Key("mt-del"), i18n.T("Delete")).Disabled(active).Clicked() {
			a.mv.delID = mt.ID
		}
	})
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
